"""Lightweight token counting utilities."""

from __future__ import annotations

from math import ceil

from context_vault.interfaces import TokenCounter
from context_vault.models import ChatMessage


class RoughTokenCounter(TokenCounter):
    """Fast approximate token counter.

    This keeps the core package provider-agnostic. Users can inject a tokenizer backed by
    tiktoken, sentencepiece, or a model-specific tokenizer when precision is required.
    """

    def count_text(self, text: str) -> int:
        """Estimate token count using a conservative character heuristic."""

        if not text:
            return 0
        return max(1, ceil(len(text) / 4))

    def count_messages(self, messages: list[ChatMessage]) -> int:
        """Estimate token count for chat messages, including small role overhead."""

        return sum(self.count_text(message.content) + 4 for message in messages)
