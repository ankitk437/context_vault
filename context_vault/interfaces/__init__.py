"""Abstract extension points for ContextVault."""

from context_vault.interfaces.compression import MemoryCompressor
from context_vault.interfaces.embeddings import EmbeddingProvider
from context_vault.interfaces.events import EventHook
from context_vault.interfaces.extraction import MemoryExtractor
from context_vault.interfaces.guardrails import Guardrail
from context_vault.interfaces.importance import ImportanceScorer
from context_vault.interfaces.llm import LLMProvider
from context_vault.interfaces.memory import (
    LongTermMemoryStore,
    SessionManager,
    ShortTermMemoryStore,
    StorageProvider,
)
from context_vault.interfaces.prompt import PromptBuilder
from context_vault.interfaces.tokens import TokenCounter
from context_vault.interfaces.vector import VectorStore

__all__ = [
    "EmbeddingProvider",
    "EventHook",
    "Guardrail",
    "ImportanceScorer",
    "LLMProvider",
    "LongTermMemoryStore",
    "MemoryCompressor",
    "MemoryExtractor",
    "PromptBuilder",
    "SessionManager",
    "ShortTermMemoryStore",
    "StorageProvider",
    "TokenCounter",
    "VectorStore",
]
