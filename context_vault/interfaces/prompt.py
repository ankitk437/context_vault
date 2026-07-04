"""Prompt builder interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from context_vault.models import ChatMessage, ContextBundle


class PromptBuilder(ABC):
    """Builds provider-agnostic prompt messages from selected context."""

    @abstractmethod
    async def build(self, bundle: ContextBundle) -> list[ChatMessage]:
        """Build chat messages in provider-agnostic format."""
