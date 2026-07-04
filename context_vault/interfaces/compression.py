"""Conversation compression interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from context_vault.models import ChatMessage, MemorySummary


class MemoryCompressor(ABC):
    """Compresses older conversation turns into summaries."""

    @abstractmethod
    async def compress(
        self,
        messages: list[ChatMessage],
        existing_summary: str | None = None,
    ) -> MemorySummary:
        """Compress messages into a summary."""
