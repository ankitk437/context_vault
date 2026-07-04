"""Vector store interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from context_vault.models import Document, SearchResult


class VectorStore(ABC):
    """Provider-agnostic vector store interface."""

    @abstractmethod
    async def add_documents(self, documents: list[Document]) -> None:
        """Add documents to the vector store."""

    @abstractmethod
    async def search(
        self, query: str, limit: int, filters: dict[str, Any] | None = None
    ) -> list[SearchResult]:
        """Search for documents relevant to a query."""

    @abstractmethod
    async def delete_documents(self, ids: list[str]) -> None:
        """Delete documents by id."""

    async def update(self, documents: list[Document]) -> None:
        """Update documents by replacing existing documents with matching ids."""

        await self.delete_documents([document.id for document in documents])
        await self.add_documents(documents)
