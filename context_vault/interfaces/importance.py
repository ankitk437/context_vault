"""Message importance scoring interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from context_vault.models import ChatMessage


class ImportanceScorer(ABC):
    """Scores how important a message is for future memory."""

    @abstractmethod
    async def score(self, message: ChatMessage) -> float:
        """Return a score from 0 to 1."""
