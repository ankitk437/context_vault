"""LLM provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from context_vault.models import ChatMessage, LLMResponse


class LLMProvider(ABC):
    """Provider-agnostic async interface for chat completion models."""

    @abstractmethod
    async def generate(self, messages: list[ChatMessage], **kwargs: Any) -> LLMResponse:
        """Generate a response from provider-specific chat messages."""
