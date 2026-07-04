# ContextVault

**The intelligent context management framework for LLM applications.**

ContextVault is a lightweight, provider-agnostic Python library for orchestrating context around LLM calls. It manages session history, long-term memory, summaries, token budgets, vector retrieval, prompt construction, and model calls without forcing a specific database, vector store, embedding model, or LLM provider.

It is intentionally not another LangChain. The core package is small, async-first, interface-driven, and easy to extend.

## Install

```bash
pip install llm-context-vault
```

For local development:

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import asyncio

from context_vault import ContextVault, VaultConfig
from context_vault.providers import EchoLLMProvider


async def main() -> None:
    vault = ContextVault(
        llm_provider=EchoLLMProvider(),
        config=VaultConfig(memory_update_frequency=1),
    )

    response = await vault.chat(
        session_id="abc123",
        user_id="user-1",
        message="My name is Ankit and I prefer Python.",
    )

    print(response.content)


asyncio.run(main())
```

If you provide only an LLM provider, ContextVault uses in-memory short-term and long-term storage by default.

## Architecture

```text
Application
  |
  v
ContextVault
  |
  |-- Session Manager
  |-- Context Planner
  |-- Token Budget Manager
  |-- Context Builder
  |-- Prompt Builder
  |-- Short-Term Memory
  |-- Long-Term Memory
  |-- Memory Extractor
  |-- Memory Compressor
  |-- Importance Scorer
  |-- Vector Retriever
  |-- LLM Provider
  |-- Storage Provider
  |-- Event Hooks
```

Every major component is replaceable through an abstract interface.

## Configuration

```python
from context_vault import VaultConfig

config = VaultConfig(
    max_context_tokens=128000,
    reserved_output_tokens=4000,
    memory_update_frequency=10,
    recent_message_limit="adaptive",
    summary_strategy="recursive",
    importance_strategy="rule_based",
    compression_threshold=0.85,
    auto_update_long_term=True,
    vector_search=False,
    vector_top_k="adaptive",
    prompt_order=[
        "system",
        "long_term_memory",
        "conversation_summary",
        "recent_messages",
        "retrieved_documents",
        "current_user_message",
    ],
)
```

## Custom LLM Provider

```python
from typing import Any

from context_vault.interfaces import LLMProvider
from context_vault.models import ChatMessage, LLMResponse


class MyLLMProvider(LLMProvider):
    async def generate(self, messages: list[ChatMessage], **kwargs: Any) -> LLMResponse:
        text = await call_my_model(messages, **kwargs)
        return LLMResponse(content=text, model="my-model")
```

## Custom Storage

```python
from context_vault.interfaces import ShortTermMemoryStore
from context_vault.models import ChatMessage, MemorySummary


class PostgresShortTermMemory(ShortTermMemoryStore):
    async def append_message(self, session_id: str, message: ChatMessage) -> None:
        ...

    async def get_messages(self, session_id: str, limit: int | None = None) -> list[ChatMessage]:
        ...

    async def count_messages(self, session_id: str) -> int:
        ...

    async def save_summary(self, session_id: str, summary: MemorySummary) -> None:
        ...

    async def get_summary(self, session_id: str) -> MemorySummary | None:
        ...
```

Wrap custom short-term and long-term stores in a `StorageProvider`, or use `InMemoryStorage` while developing.

## Custom Vector Store

```python
from typing import Any

from context_vault.interfaces import VectorStore
from context_vault.models import Document, SearchResult


class MyVectorStore(VectorStore):
    async def add_documents(self, documents: list[Document]) -> None:
        ...

    async def search(
        self, query: str, limit: int, filters: dict[str, Any] | None = None
    ) -> list[SearchResult]:
        ...

    async def delete_documents(self, ids: list[str]) -> None:
        ...
```

Vector retrieval is optional and disabled by default.

## Custom Memory Extractor

```python
from context_vault.interfaces import MemoryExtractor
from context_vault.models import ChatMessage, LongTermMemory


class MyExtractor(MemoryExtractor):
    async def extract(
        self,
        conversation: list[ChatMessage],
        existing_memory: LongTermMemory,
    ) -> LongTermMemory:
        ...
```

Use this when you want LLM-based extraction, stricter privacy rules, domain-specific memory fields, or custom merge logic.

## Custom Importance Scorer

```python
from context_vault.interfaces import ImportanceScorer
from context_vault.models import ChatMessage


class MyScorer(ImportanceScorer):
    async def score(self, message: ChatMessage) -> float:
        return 0.5
```

Importance scores guide compression and memory decisions.

## Token Budgeting

ContextVault allocates the input context window across configurable sections:

- system prompt
- current user message
- short-term memory
- long-term memory
- vector search results
- metadata
- reserved output tokens

The default planner includes as much useful context as possible without exceeding the configured budget.

## Development

```bash
pip install -e ".[dev]"
pytest
```

The first implementation includes in-memory providers, a local echo provider, a mock provider for tests, token budgeting, context planning, prompt building, memory extraction, compression, vector retrieval, and lifecycle hooks.

## Live OpenAI Smoke Test

Do not save API keys in the repository. Export the key only in your shell:

```bash
export OPENAI_API_KEY="your-key"
export OPENAI_MODEL="gpt-4o-mini"
python examples/openai_smoke_test.py
unset OPENAI_API_KEY
```

The script reads the key from `OPENAI_API_KEY` and does not write it to disk.

## Live OpenAI Memory Sequence Test

To verify that long-term memory builds over multiple interactions:

```bash
export OPENAI_API_KEY="your-key"
export OPENAI_CHAT_MODEL="gpt-4o-mini"
export OPENAI_MEMORY_MODEL="gpt-4o-mini"
python examples/openai_memory_sequence_test.py
unset OPENAI_API_KEY
```

The script sends several sequential messages and prints long-term memory after
each turn. It uses in-memory storage, so the memory exists only for that run.

## Memory Persistence Check

The default `InMemoryStorage` persists only while the storage object exists. Run:

```bash
python examples/memory_persistence_check.py
```

The example shows that memory is available in the same storage object, even across
multiple `ContextVault` instances, but disappears when you create fresh in-memory
storage. Use a database-backed `StorageProvider` for durable persistence.

## Different Models Per Action

You can use separate LLM providers for the main chat response, memory extraction,
compression, and importance scoring:

```python
from context_vault import ContextVault, VaultConfig
from context_vault.providers import OpenAIProvider

vault = ContextVault(
    llm_provider=OpenAIProvider(model="gpt-4o"),
    memory_llm_provider=OpenAIProvider(model="gpt-4o-mini"),
    compression_llm_provider=OpenAIProvider(model="gpt-4o-mini"),
    importance_llm_provider=OpenAIProvider(model="gpt-4o-mini"),
    config=VaultConfig(memory_update_frequency=5),
)
```

If you do not provide these action-specific providers, ContextVault keeps using
the lightweight rule-based defaults for memory extraction, compression, and
importance scoring. Run the local demo:

```bash
python examples/per_action_llm_models.py
```
