"""LLM response models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class LLMResponse(BaseModel):
    """Provider-agnostic response returned from an LLM provider."""

    content: str
    model: str | None = None
    raw: Any = None
    usage: dict[str, int] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
