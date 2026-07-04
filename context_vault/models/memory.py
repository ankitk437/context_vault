"""Memory models."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class LongTermMemory(BaseModel):
    """Stable facts remembered about a user."""

    user_id: str
    facts: dict[str, Any] = Field(default_factory=dict)
    version: int = Field(default=1, ge=1)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

    def filtered(self, include: list[str], exclude: list[str]) -> "LongTermMemory":
        """Return a copy containing only configured fact fields."""

        include_set = set(include)
        exclude_set = set(exclude)
        facts = {
            key: value
            for key, value in self.facts.items()
            if (not include_set or key in include_set) and key not in exclude_set
        }
        return self.model_copy(update={"facts": facts})

    def render(self, include: list[str] | None = None, exclude: list[str] | None = None) -> str:
        """Render memory as compact prompt text."""

        memory = self.filtered(include or [], exclude or [])
        if not memory.facts:
            return ""
        lines = ["Known stable user facts:"]
        for key in sorted(memory.facts):
            value = memory.facts[key]
            if isinstance(value, list):
                rendered_value = ", ".join(str(item) for item in value)
            else:
                rendered_value = str(value)
            lines.append(f"- {key}: {rendered_value}")
        return "\n".join(lines)


class MemorySummary(BaseModel):
    """A compressed summary of older conversation context."""

    content: str
    id: str = Field(default_factory=lambda: str(uuid4()))
    source_message_ids: list[str] = Field(default_factory=list)
    level: int = Field(default=1, ge=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    token_count: int | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
