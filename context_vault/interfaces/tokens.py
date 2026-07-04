"""Token counting interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from context_vault.models import ChatMessage


class TokenCounter(ABC):
    """Estimates token usage."""

    @abstractmethod
    def count_text(self, text: str) -> int:
        """Return estimated token count for text."""

    @abstractmethod
    def count_messages(self, messages: list[ChatMessage]) -> int:
        """Return estimated token count for messages."""
