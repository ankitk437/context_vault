"""LLM-backed long-term memory extraction."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from context_vault.config import VaultConfig
from context_vault.interfaces import LLMProvider, MemoryExtractor
from context_vault.models import ChatMessage, LongTermMemory


class LLMMemoryExtractor(MemoryExtractor):
    """Extracts stable long-term memory facts using a dedicated LLM provider."""

    def __init__(self, llm_provider: LLMProvider, config: VaultConfig | None = None) -> None:
        self.llm_provider = llm_provider
        self.config = config or VaultConfig()

    async def extract(
        self,
        conversation: list[ChatMessage],
        existing_memory: LongTermMemory,
    ) -> LongTermMemory:
        """Return updated long-term memory."""

        prompt = self._build_prompt(conversation, existing_memory)
        response = await self.llm_provider.generate(
            [
                ChatMessage(
                    role="system",
                    content=(
                        "You extract stable user memory for an LLM application. "
                        "Return only valid JSON. Do not include markdown."
                    ),
                ),
                ChatMessage(role="user", content=prompt),
            ],
            temperature=0,
        )
        extracted = _parse_json_object(response.content)
        facts = dict(existing_memory.facts)
        for key, value in extracted.items():
            if self._field_allowed(key) and value not in (None, "", [], {}):
                facts[key] = value
        return existing_memory.model_copy(
            update={
                "facts": facts,
                "version": existing_memory.version + 1,
                "updated_at": datetime.now(UTC),
            }
        )

    def _build_prompt(self, conversation: list[ChatMessage], existing_memory: LongTermMemory) -> str:
        include = ", ".join(self.config.long_term_include_fields) or "any stable user facts"
        exclude = ", ".join(self.config.long_term_exclude_fields) or "none"
        transcript = "\n".join(f"{message.role}: {message.content}" for message in conversation)
        configured_prompt = self.config.extraction_prompt or (
            "Extract only durable user facts that are useful in future conversations."
        )
        return (
            f"{configured_prompt}\n\n"
            f"Allowed fields: {include}\n"
            f"Excluded fields: {exclude}\n"
            f"Existing memory JSON:\n{json.dumps(existing_memory.facts, ensure_ascii=True)}\n\n"
            f"Conversation:\n{transcript}\n\n"
            "Return a JSON object of updated facts only. Example: "
            '{"name": "Ankit", "preferences": ["Python"]}'
        )

    def _field_allowed(self, field: str) -> bool:
        include = set(self.config.long_term_include_fields)
        exclude = set(self.config.long_term_exclude_fields)
        return (not include or field in include) and field not in exclude


def _parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        try:
            parsed = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            return {}
    return parsed if isinstance(parsed, dict) else {}
