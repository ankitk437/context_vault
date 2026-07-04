"""Storage and session interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod

from context_vault.models import ChatMessage, LongTermMemory, MemorySummary


class ShortTermMemoryStore(ABC):
    """Stores conversation messages and summaries per session."""

    @abstractmethod
    async def append_message(self, session_id: str, message: ChatMessage) -> None:
        """Append one message to a session."""

    @abstractmethod
    async def get_messages(self, session_id: str, limit: int | None = None) -> list[ChatMessage]:
        """Return messages for a session, oldest first."""

    @abstractmethod
    async def count_messages(self, session_id: str) -> int:
        """Return number of messages stored for a session."""

    @abstractmethod
    async def save_summary(self, session_id: str, summary: MemorySummary) -> None:
        """Save the latest compressed summary for a session."""

    @abstractmethod
    async def get_summary(self, session_id: str) -> MemorySummary | None:
        """Return the latest compressed summary for a session."""


class LongTermMemoryStore(ABC):
    """Stores stable facts per user."""

    @abstractmethod
    async def get_memory(self, user_id: str) -> LongTermMemory:
        """Return long-term memory for a user, creating an empty memory if needed."""

    @abstractmethod
    async def update_memory(self, user_id: str, memory: LongTermMemory) -> None:
        """Persist long-term memory for a user."""


class StorageProvider(ABC):
    """Groups short-term and long-term memory stores."""

    @property
    @abstractmethod
    def short_term(self) -> ShortTermMemoryStore:
        """Return the short-term memory store."""

    @property
    @abstractmethod
    def long_term(self) -> LongTermMemoryStore:
        """Return the long-term memory store."""

    async def close(self) -> None:
        """Close any resources held by the storage provider."""


class SessionManager(ABC):
    """Coordinates session-scoped conversation operations."""

    @abstractmethod
    async def append_message(self, session_id: str, message: ChatMessage) -> None:
        """Append a message to a session."""

    @abstractmethod
    async def get_recent_messages(
        self, session_id: str, limit: int | None = None
    ) -> list[ChatMessage]:
        """Return recent messages for the session."""

    @abstractmethod
    async def count_messages(self, session_id: str) -> int:
        """Return session message count."""
