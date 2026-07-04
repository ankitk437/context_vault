"""In-memory vector store."""

from __future__ import annotations

import asyncio
import math
import re
from typing import Any

from context_vault.interfaces import EmbeddingProvider, VectorStore
from context_vault.models import Document, SearchResult


class InMemoryVectorStore(VectorStore):
    """Small in-memory vector store with optional embeddings.

    When no embedding provider is supplied, search falls back to lexical token overlap.
    """

    def __init__(self, embedding_provider: EmbeddingProvider | None = None) -> None:
        self.embedding_provider = embedding_provider
        self._documents: dict[str, Document] = {}
        self._vectors: dict[str, list[float]] = {}
        self._lock = asyncio.Lock()

    async def add_documents(self, documents: list[Document]) -> None:
        """Add documents to the vector store."""

        vectors: list[list[float]] | None = None
        if self.embedding_provider is not None:
            vectors = await self.embedding_provider.embed([document.content for document in documents])
        async with self._lock:
            for index, document in enumerate(documents):
                self._documents[document.id] = document.model_copy(deep=True)
                if vectors is not None:
                    self._vectors[document.id] = vectors[index]

    async def search(
        self, query: str, limit: int, filters: dict[str, Any] | None = None
    ) -> list[SearchResult]:
        """Search for documents relevant to a query."""

        if limit <= 0:
            return []
        async with self._lock:
            documents = [
                document.model_copy(deep=True)
                for document in self._documents.values()
                if self._matches_filters(document, filters)
            ]
            vectors = dict(self._vectors)

        if self.embedding_provider is not None:
            query_vector = (await self.embedding_provider.embed([query]))[0]
            scored = [
                SearchResult(
                    document=document,
                    score=max(0.0, _cosine_similarity(query_vector, vectors.get(document.id, []))),
                )
                for document in documents
            ]
        else:
            query_tokens = _tokens(query)
            scored = [
                SearchResult(document=document, score=_lexical_score(query_tokens, document.content))
                for document in documents
            ]
        scored.sort(key=lambda result: result.score, reverse=True)
        return [result for result in scored[:limit] if result.score > 0]

    async def delete_documents(self, ids: list[str]) -> None:
        """Delete documents by id."""

        async with self._lock:
            for document_id in ids:
                self._documents.pop(document_id, None)
                self._vectors.pop(document_id, None)

    def _matches_filters(self, document: Document, filters: dict[str, Any] | None) -> bool:
        if not filters:
            return True
        return all(document.metadata.get(key) == value for key, value in filters.items())


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9_]+", text.lower()))


def _lexical_score(query_tokens: set[str], text: str) -> float:
    if not query_tokens:
        return 0.0
    document_tokens = _tokens(text)
    if not document_tokens:
        return 0.0
    overlap = len(query_tokens & document_tokens)
    return overlap / math.sqrt(len(query_tokens) * len(document_tokens))


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)
