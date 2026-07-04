"""Rule-based message importance scoring."""

from __future__ import annotations

import re

from context_vault.interfaces import ImportanceScorer
from context_vault.models import ChatMessage


class RuleBasedImportanceScorer(ImportanceScorer):
    """Scores messages with lightweight heuristics."""

    _high_signal_patterns = [
        re.compile(pattern, re.IGNORECASE)
        for pattern in [
            r"\bmy name is\b",
            r"\bi live in\b",
            r"\bi moved to\b",
            r"\bi prefer\b",
            r"\bmy preferred\b",
            r"\bmy goal is\b",
            r"\bi work as\b",
            r"\bi am allergic\b",
            r"\bremember that\b",
            r"\bfavorite\b|\bfavourite\b",
        ]
    ]
    _low_signal = {"ok", "okay", "thanks", "thank you", "nice", "cool", "yes", "no"}

    async def score(self, message: ChatMessage) -> float:
        """Return a score from 0 to 1."""

        text = message.content.strip()
        lowered = text.lower()
        if not text:
            return 0.0
        if lowered in self._low_signal:
            return 0.05
        if any(pattern.search(text) for pattern in self._high_signal_patterns):
            return 0.95
        if message.role == "user" and len(text) > 120:
            return 0.55
        if message.role == "assistant":
            return 0.25
        return 0.35
