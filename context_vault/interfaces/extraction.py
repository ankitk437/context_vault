"""Long-term memory extraction interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from context_vault.models import ChatMessage, LongTermMemory


class MemoryExtractor(ABC):
    """Extracts stable facts from conversation history."""

    @abstractmethod
    async def extract(
        self,
        conversation: list[ChatMessage],
        existing_memory: LongTermMemory,
    ) -> LongTermMemory:
        """Return updated long-term memory."""
