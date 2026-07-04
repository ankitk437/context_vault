"""Simple local LLM provider used for examples and smoke tests."""

from __future__ import annotations

from typing import Any

from context_vault.interfaces import LLMProvider
from context_vault.models import ChatMessage, LLMResponse


class EchoLLMProvider(LLMProvider):
    """LLM provider that echoes the latest user message."""

    def __init__(self, model: str = "echo") -> None:
        self.model = model

    async def generate(self, messages: list[ChatMessage], **kwargs: Any) -> LLMResponse:
        """Return a deterministic echo response."""

        latest_user = next((message for message in reversed(messages) if message.role == "user"), None)
        content = latest_user.content if latest_user else ""
        return LLMResponse(content=f"Echo: {content}", model=self.model, metadata={"provider": "echo"})
