"""Test-friendly mock LLM provider."""

from __future__ import annotations

from collections import deque
from typing import Any

from context_vault.interfaces import LLMProvider
from context_vault.models import ChatMessage, LLMResponse


class MockLLMProvider(LLMProvider):
    """LLM provider with queued deterministic responses."""

    def __init__(self, responses: list[str] | None = None, model: str = "mock") -> None:
        self.model = model
        self.responses: deque[str] = deque(responses or ["ok"])
        self.calls: list[list[ChatMessage]] = []

    async def generate(self, messages: list[ChatMessage], **kwargs: Any) -> LLMResponse:
        """Return the next queued response, repeating the final default when exhausted."""

        self.calls.append([message.model_copy(deep=True) for message in messages])
        content = self.responses.popleft() if self.responses else "ok"
        return LLMResponse(content=content, model=self.model, metadata={"provider": "mock"})
