"""Default session management."""

from __future__ import annotations

from context_vault.interfaces import SessionManager, ShortTermMemoryStore
from context_vault.models import ChatMessage


class DefaultSessionManager(SessionManager):
    """Session manager backed by a short-term memory store."""

    def __init__(self, store: ShortTermMemoryStore) -> None:
        self._store = store

    async def append_message(self, session_id: str, message: ChatMessage) -> None:
        """Append a message to a session."""

        await self._store.append_message(session_id, message)

    async def get_recent_messages(
        self, session_id: str, limit: int | None = None
    ) -> list[ChatMessage]:
        """Return recent messages for the session."""

        return await self._store.get_messages(session_id, limit=limit)

    async def count_messages(self, session_id: str) -> int:
        """Return session message count."""

        return await self._store.count_messages(session_id)
