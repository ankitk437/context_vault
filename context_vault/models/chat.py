"""Chat message models."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

ChatRole = Literal["system", "user", "assistant", "tool"]


class ChatMessage(BaseModel):
    """A provider-agnostic chat message."""

    role: ChatRole
    content: str
    id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)
    importance: float | None = Field(default=None, ge=0, le=1)
    token_count: int | None = Field(default=None, ge=0)

    def is_empty(self) -> bool:
        """Return whether the message has only whitespace content."""

        return not self.content.strip()
