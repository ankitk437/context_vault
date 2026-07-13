"""LLM provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from context_vault.models import ChatMessage, LLMResponse, LLMStreamEvent


class LLMProvider(ABC):
    """Provider-agnostic async interface for chat completion models."""

    @abstractmethod
    async def generate(self, messages: list[ChatMessage], **kwargs: Any) -> LLMResponse:
        """Generate a response from provider-specific chat messages."""

    async def stream_generate(
        self,
        messages: list[ChatMessage],
        **kwargs: Any,
    ) -> AsyncIterator[LLMStreamEvent]:
        """Stream a response, falling back to one full-response delta."""

        response = await self.generate(messages, **kwargs)
        if response.content:
            yield LLMStreamEvent(
                type="response.output_text.delta",
                delta=response.content,
                model=response.model,
            )
        yield LLMStreamEvent(type="response.completed", response=response, model=response.model)
