"""Default long-term memory extractor."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

from context_vault.config import VaultConfig
from context_vault.interfaces import MemoryExtractor
from context_vault.models import ChatMessage, LongTermMemory


class RuleBasedMemoryExtractor(MemoryExtractor):
    """Extracts common stable facts without a vendor dependency."""

    def __init__(self, config: VaultConfig | None = None) -> None:
        self.config = config or VaultConfig()

    async def extract(
        self,
        conversation: list[ChatMessage],
        existing_memory: LongTermMemory,
    ) -> LongTermMemory:
        """Return updated long-term memory."""

        facts = dict(existing_memory.facts)
        for message in conversation:
            if message.role != "user":
                continue
            updates = self._extract_from_text(message.content)
            for key, value in updates.items():
                if not self._field_allowed(key):
                    continue
                facts[key] = _merge_fact(facts.get(key), value)
        return existing_memory.model_copy(
            update={
                "facts": facts,
                "version": existing_memory.version + 1,
                "updated_at": datetime.now(UTC),
            }
        )

    def _field_allowed(self, field: str) -> bool:
        include = set(self.config.long_term_include_fields)
        exclude = set(self.config.long_term_exclude_fields)
        return (not include or field in include) and field not in exclude

    def _extract_from_text(self, text: str) -> dict[str, Any]:
        clean = " ".join(text.strip().split())
        updates: dict[str, Any] = {}

        patterns = {
            "name": [
                r"\bmy name is (?P<value>[^.?!,;]{2,60})",
                r"\bi'?m (?P<value>[^.?!,;]{2,60})",
            ],
            "location": [
                r"\bi live in (?P<value>[^.?!,;]{2,80})",
                r"\bi moved to (?P<value>[^.?!,;]{2,80})",
                r"\bi am from (?P<value>[^.?!,;]{2,80})",
            ],
            "preferences": [
                r"\bi prefer (?P<value>[^.?!;]{2,120})",
                r"\bmy preferred (?P<value>[^.?!;]{2,120})",
            ],
            "interests": [
                r"\bi like (?P<value>[^.?!;]{2,120})",
                r"\bi love (?P<value>[^.?!;]{2,120})",
                r"\bi am interested in (?P<value>[^.?!;]{2,120})",
            ],
            "profession": [
                r"\bi work as (?P<value>[^.?!;]{2,120})",
                r"\bmy profession is (?P<value>[^.?!;]{2,120})",
            ],
            "goals": [
                r"\bmy goal is (?P<value>[^.?!;]{2,120})",
                r"\bi want to (?P<value>[^.?!;]{2,120})",
            ],
        }

        for field, field_patterns in patterns.items():
            for pattern in field_patterns:
                match = re.search(pattern, clean, flags=re.IGNORECASE)
                if match:
                    updates[field] = _clean_fact_value(field, match.group("value"))
                    break
        return updates


def _merge_fact(existing: Any, new_value: Any) -> Any:
    if existing is None:
        return new_value
    if existing == new_value:
        return existing
    if isinstance(existing, list):
        if new_value not in existing:
            return [*existing, new_value]
        return existing
    return [existing, new_value]


def _clean_fact_value(field: str, value: str) -> str:
    cleaned = value.strip(" .")
    if field == "name":
        cleaned = re.split(r"\s+and\s+i\s+", cleaned, maxsplit=1, flags=re.IGNORECASE)[0]
    return cleaned.strip(" .")
