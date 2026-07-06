"""End-to-end local workflow check for ContextVault.

This script exercises the major workflow pieces without external services:

- session-based short-term memory
- long-term memory extraction
- vector retrieval
- prompt construction
- event hooks
- per-action LLM providers
- automatic compression / summarization

Run:
    python examples/full_workflow_check.py
"""

from __future__ import annotations

import asyncio
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from context_vault import ContextVault, VaultConfig
from context_vault.interfaces import LLMProvider
from context_vault.models import ChatMessage, Document, LLMResponse
from context_vault.storage import InMemoryStorage
from context_vault.vectorstores import InMemoryVectorStore


class WorkflowLLMProvider(LLMProvider):
    """Deterministic provider that records calls for workflow verification."""

    def __init__(self, model: str, mode: str) -> None:
        self.model = model
        self.mode = mode
        self.calls: list[list[ChatMessage]] = []

    async def generate(self, messages: list[ChatMessage], **kwargs: Any) -> LLMResponse:
        self.calls.append([message.model_copy(deep=True) for message in messages])
        if self.mode == "memory":
            return LLMResponse(
                content=(
                    '{"name": "Ankit", "location": "Bangalore", '
                    '"preferences": "concise Python examples", '
                    '"interests": "LLM memory systems", '
                    '"goals": "publish ContextVault as an open-source package"}'
                ),
                model=self.model,
            )
        if self.mode == "compression":
            return LLMResponse(
                content=(
                    "The user is Ankit from Bangalore, prefers concise Python examples, "
                    "and is building ContextVault for LLM memory workflows."
                ),
                model=self.model,
            )
        if self.mode == "importance":
            return LLMResponse(content="0.9", model=self.model)
        return LLMResponse(
            content=f"chat-response-{len(self.calls)} from {self.model}",
            model=self.model,
        )


async def main() -> None:
    events_seen: list[str] = []

    async def record_event(event_name: str, payload: dict[str, Any]) -> None:
        events_seen.append(event_name)

    chat_model = WorkflowLLMProvider("workflow-chat-model", "chat")
    memory_model = WorkflowLLMProvider("workflow-memory-model", "memory")
    compression_model = WorkflowLLMProvider("workflow-compression-model", "compression")
    importance_model = WorkflowLLMProvider("workflow-importance-model", "importance")

    storage = InMemoryStorage()
    vector_store = InMemoryVectorStore()
    await vector_store.add_documents(
        [
            Document(
                content=(
                    "ContextVault is a Python library for LLM context management, "
                    "short-term memory, long-term memory, and summarization."
                ),
                metadata={"source": "contextvault-overview"},
            ),
            Document(
                content="Use vector retrieval to inject relevant documents into the LLM prompt.",
                metadata={"source": "retrieval-note"},
            ),
        ]
    )

    config = VaultConfig(
        max_context_tokens=220,
        reserved_output_tokens=40,
        memory_update_frequency=1,
        vector_search=True,
        vector_top_k=2,
        compression_threshold=0.25,
        long_term_include_fields=[
            "name",
            "location",
            "preferences",
            "interests",
            "goals",
        ],
    )

    vault = ContextVault(
        llm_provider=chat_model,
        memory_llm_provider=memory_model,
        compression_llm_provider=compression_model,
        importance_llm_provider=importance_model,
        storage=storage,
        vector_store=vector_store,
        config=config,
    )
    for event_name in [
        "before_context_build",
        "after_context_build",
        "before_llm_call",
        "after_llm_call",
        "before_memory_update",
        "after_memory_update",
        "before_summary",
        "after_summary",
    ]:
        vault.events.register(event_name, record_event)

    user_messages = [
        "My name is Ankit and I live in Bangalore.",
        "I prefer concise Python examples.",
        "I am interested in LLM memory systems.",
        "My goal is to publish ContextVault as an open-source package.",
        "How should vector retrieval work with conversation memory?",
    ]

    for message in user_messages:
        response = await vault.chat(
            session_id="full-workflow-session",
            user_id="user-1",
            message=message,
        )
        print(f"assistant: {response.content}")

    memory = await storage.long_term.get_memory("user-1")
    messages = await storage.short_term.get_messages("full-workflow-session")
    summary = await storage.short_term.get_summary("full-workflow-session")
    event_counts = Counter(events_seen)
    first_prompt = chat_model.calls[0]

    print("\n--- workflow result ---")
    print(f"short-term messages stored: {len(messages)}")
    print(f"long-term memory facts: {memory.facts}")
    print(f"summary created: {summary is not None}")
    if summary is not None:
        print(f"summary text: {summary.content}")
    print(f"chat model calls: {len(chat_model.calls)}")
    print(f"memory model calls: {len(memory_model.calls)}")
    print(f"compression model calls: {len(compression_model.calls)}")
    print(f"importance model calls: {len(importance_model.calls)}")
    print(f"events: {dict(sorted(event_counts.items()))}")
    print(
        "vector context injected: "
        f"{any('Retrieved context:' in message.content for message in first_prompt)}"
    )


if __name__ == "__main__":
    asyncio.run(main())
