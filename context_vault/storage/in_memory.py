"""Thread-safe in-memory storage providers."""

from __future__ import annotations

import asyncio

from context_vault.interfaces import LongTermMemoryStore, ShortTermMemoryStore, StorageProvider
from context_vault.models import ChatMessage, LongTermMemory, MemorySummary


class InMemoryShortTermMemoryStore(ShortTermMemoryStore):
    """Stores session messages in process memory."""

    def __init__(self) -> None:
        self._messages: dict[str, list[ChatMessage]] = {}
        self._summaries: dict[str, MemorySummary] = {}
        self._lock = asyncio.Lock()

    async def append_message(self, session_id: str, message: ChatMessage) -> None:
        """Append one message to a session."""

        async with self._lock:
            self._messages.setdefault(session_id, []).append(message.model_copy(deep=True))

    async def get_messages(self, session_id: str, limit: int | None = None) -> list[ChatMessage]:
        """Return messages for a session, oldest first."""

        async with self._lock:
            messages = self._messages.get(session_id, [])
            selected = messages[-limit:] if limit is not None else messages
            return [message.model_copy(deep=True) for message in selected]

    async def count_messages(self, session_id: str) -> int:
        """Return number of messages stored for a session."""

        async with self._lock:
            return len(self._messages.get(session_id, []))

    async def save_summary(self, session_id: str, summary: MemorySummary) -> None:
        """Save the latest compressed summary for a session."""

        async with self._lock:
            self._summaries[session_id] = summary.model_copy(deep=True)

    async def get_summary(self, session_id: str) -> MemorySummary | None:
        """Return the latest compressed summary for a session."""

        async with self._lock:
            summary = self._summaries.get(session_id)
            return summary.model_copy(deep=True) if summary else None


class InMemoryLongTermMemoryStore(LongTermMemoryStore):
    """Stores long-term user memory in process memory."""

    def __init__(self) -> None:
        self._memory: dict[str, LongTermMemory] = {}
        self._lock = asyncio.Lock()

    async def get_memory(self, user_id: str) -> LongTermMemory:
        """Return long-term memory for a user, creating an empty memory if needed."""

        async with self._lock:
            if user_id not in self._memory:
                self._memory[user_id] = LongTermMemory(user_id=user_id)
            return self._memory[user_id].model_copy(deep=True)

    async def update_memory(self, user_id: str, memory: LongTermMemory) -> None:
        """Persist long-term memory for a user."""

        async with self._lock:
            data = memory.model_copy(deep=True)
            data.user_id = user_id
            self._memory[user_id] = data


class InMemoryStorage(StorageProvider):
    """Default storage provider combining in-memory short and long-term stores."""

    def __init__(
        self,
        short_term: InMemoryShortTermMemoryStore | None = None,
        long_term: InMemoryLongTermMemoryStore | None = None,
    ) -> None:
        self._short_term = short_term or InMemoryShortTermMemoryStore()
        self._long_term = long_term or InMemoryLongTermMemoryStore()

    @property
    def short_term(self) -> InMemoryShortTermMemoryStore:
        """Return the short-term memory store."""

        return self._short_term

    @property
    def long_term(self) -> InMemoryLongTermMemoryStore:
        """Return the long-term memory store."""

        return self._long_term
