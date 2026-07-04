"""ContextVault facade."""

from __future__ import annotations

from typing import Any

from context_vault.compression import DefaultMemoryCompressor, LLMMemoryCompressor
from context_vault.config import VaultConfig
from context_vault.core.session import DefaultSessionManager
from context_vault.events import EventManager
from context_vault.importance import LLMImportanceScorer, RuleBasedImportanceScorer
from context_vault.interfaces import (
    EmbeddingProvider,
    ImportanceScorer,
    LLMProvider,
    MemoryCompressor,
    MemoryExtractor,
    PromptBuilder,
    SessionManager,
    StorageProvider,
    TokenCounter,
    VectorStore,
)
from context_vault.memory import LLMMemoryExtractor, RuleBasedMemoryExtractor
from context_vault.models import LLMResponse
from context_vault.pipeline import ChatPipeline
from context_vault.planner import ContextPlanner, TokenBudgetManager
from context_vault.prompt import DefaultPromptBuilder
from context_vault.storage import InMemoryStorage
from context_vault.utils import RoughTokenCounter
from context_vault.vectorstores import InMemoryVectorStore


class ContextVault:
    """High-level API for context-aware LLM interactions."""

    def __init__(
        self,
        *,
        llm_provider: LLMProvider,
        memory_llm_provider: LLMProvider | None = None,
        compression_llm_provider: LLMProvider | None = None,
        importance_llm_provider: LLMProvider | None = None,
        storage: StorageProvider | None = None,
        vector_store: VectorStore | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        config: VaultConfig | None = None,
        session_manager: SessionManager | None = None,
        token_counter: TokenCounter | None = None,
        context_planner: ContextPlanner | None = None,
        token_budget_manager: TokenBudgetManager | None = None,
        prompt_builder: PromptBuilder | None = None,
        memory_extractor: MemoryExtractor | None = None,
        memory_compressor: MemoryCompressor | None = None,
        importance_scorer: ImportanceScorer | None = None,
        events: EventManager | None = None,
    ) -> None:
        self.config = config or VaultConfig()
        self.storage = storage or InMemoryStorage()
        self.token_counter = token_counter or RoughTokenCounter()
        self.llm_provider = llm_provider
        self.memory_llm_provider = memory_llm_provider
        self.compression_llm_provider = compression_llm_provider
        self.importance_llm_provider = importance_llm_provider
        self.importance_scorer = importance_scorer or (
            LLMImportanceScorer(importance_llm_provider)
            if importance_llm_provider is not None
            else RuleBasedImportanceScorer()
        )
        self.session_manager = session_manager or DefaultSessionManager(self.storage.short_term)
        self.vector_store = vector_store
        if self.vector_store is None and self.config.vector_search:
            self.vector_store = InMemoryVectorStore(embedding_provider=embedding_provider)
        self.token_budget_manager = token_budget_manager or TokenBudgetManager(
            self.config, self.token_counter
        )
        self.context_planner = context_planner or ContextPlanner(self.config, self.token_counter)
        self.prompt_builder = prompt_builder or DefaultPromptBuilder(self.config)
        self.memory_extractor = memory_extractor or (
            LLMMemoryExtractor(memory_llm_provider, self.config)
            if memory_llm_provider is not None
            else RuleBasedMemoryExtractor(self.config)
        )
        self.memory_compressor = memory_compressor or (
            LLMMemoryCompressor(
                compression_llm_provider,
                token_counter=self.token_counter,
            )
            if compression_llm_provider is not None
            else DefaultMemoryCompressor(
                token_counter=self.token_counter,
                importance_scorer=self.importance_scorer,
            )
        )
        self.events = events or EventManager()
        self.pipeline = ChatPipeline(
            llm_provider=self.llm_provider,
            storage=self.storage,
            session_manager=self.session_manager,
            vector_store=self.vector_store,
            config=self.config,
            token_counter=self.token_counter,
            token_budget_manager=self.token_budget_manager,
            context_planner=self.context_planner,
            prompt_builder=self.prompt_builder,
            memory_extractor=self.memory_extractor,
            memory_compressor=self.memory_compressor,
            importance_scorer=self.importance_scorer,
            events=self.events,
        )

    async def chat(
        self,
        *,
        session_id: str,
        user_id: str,
        message: str,
        metadata: dict[str, Any] | None = None,
        **llm_kwargs: Any,
    ) -> LLMResponse:
        """Send one user message through the context orchestration pipeline."""

        return await self.pipeline.chat(
            session_id=session_id,
            user_id=user_id,
            message=message,
            metadata=metadata,
            **llm_kwargs,
        )

    async def close(self) -> None:
        """Close resources held by configured providers."""

        await self.storage.close()
