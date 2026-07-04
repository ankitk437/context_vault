"""Use different LLM providers/models for different ContextVault actions."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from context_vault import ContextVault, VaultConfig
from context_vault.interfaces import LLMProvider
from context_vault.models import ChatMessage, LLMResponse


class TracingLLMProvider(LLMProvider):
    """Small provider that records when each action-specific model is used."""

    def __init__(self, model: str, response: str) -> None:
        self.model = model
        self.response = response
        self.calls = 0

    async def generate(self, messages: list[ChatMessage], **kwargs: Any) -> LLMResponse:
        self.calls += 1
        return LLMResponse(content=self.response, model=self.model)


async def main() -> None:
    chat_model = TracingLLMProvider("chat-model", "Hello from the chat model.")
    memory_model = TracingLLMProvider(
        "memory-extraction-model",
        '{"name": "Ankit", "preferences": "Python"}',
    )
    compression_model = TracingLLMProvider(
        "compression-model",
        "The user prefers Python and is testing per-action models.",
    )
    importance_model = TracingLLMProvider("importance-scoring-model", "0.9")

    vault = ContextVault(
        llm_provider=chat_model,
        memory_llm_provider=memory_model,
        compression_llm_provider=compression_model,
        importance_llm_provider=importance_model,
        config=VaultConfig(memory_update_frequency=1),
    )

    response = await vault.chat(
        session_id="per-action-demo",
        user_id="user-1",
        message="My name is Ankit and I prefer Python.",
    )

    await vault.memory_compressor.compress(
        [
            ChatMessage(role="user", content="I prefer Python."),
            ChatMessage(role="assistant", content="I will remember that."),
        ]
    )

    memory = await vault.storage.long_term.get_memory("user-1")

    print(f"chat response: {response.content}")
    print(f"stored memory: {memory.facts}")
    print(f"chat model calls: {chat_model.calls}")
    print(f"memory model calls: {memory_model.calls}")
    print(f"compression model calls: {compression_model.calls}")
    print(f"importance model calls: {importance_model.calls}")


if __name__ == "__main__":
    asyncio.run(main())
