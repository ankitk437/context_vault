"""Document and vector-search result models."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class Document(BaseModel):
    """A document that can be stored in a vector store or injected into context."""

    content: str
    id: str = Field(default_factory=lambda: str(uuid4()))
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    """A ranked vector-search result."""

    document: Document
    score: float = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
