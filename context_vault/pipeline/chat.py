"""Main chat orchestration pipeline."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from context_vault.config import VaultConfig
from context_vault.events import EventManager
from context_vault.interfaces import (
    Guardrail,
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
from context_vault.models import (
    ChatMessage,
    LLMResponse,
    LLMStreamEvent,
    LongTermMemory,
    MemorySummary,
    SearchResult,
    TokenBudget,
)
from context_vault.planner import ContextPlanner, TokenBudgetManager

logger = logging.getLogger(__name__)


class ChatPipeline:
    """Coordinates the full ContextVault chat flow."""

    def __init__(
        self,
        *,
        llm_provider: LLMProvider,
        storage: StorageProvider,
        session_manager: SessionManager,
        vector_store: VectorStore | None,
        config: VaultConfig,
        token_counter: TokenCounter,
        token_budget_manager: TokenBudgetManager,
        context_planner: ContextPlanner,
        prompt_builder: PromptBuilder,
        memory_extractor: MemoryExtractor,
        memory_compressor: MemoryCompressor,
        importance_scorer: ImportanceScorer,
        guardrails: list[Guardrail],
        events: EventManager,
    ) -> None:
        self.llm_provider = llm_provider
        self.storage = storage
        self.session_manager = session_manager
        self.vector_store = vector_store
        self.config = config
        self.token_counter = token_counter
        self.token_budget_manager = token_budget_manager
        self.context_planner = context_planner
        self.prompt_builder = prompt_builder
        self.memory_extractor = memory_extractor
        self.memory_compressor = memory_compressor
        self.importance_scorer = importance_scorer
        self.guardrails = guardrails
        self.events = events

    async def chat(
        self,
        *,
        session_id: str,
        user_id: str,
        message: str,
        metadata: dict[str, Any] | None = None,
        **llm_kwargs: Any,
    ) -> LLMResponse:
        """Run one ContextVault chat interaction."""

        guardrail_metadata = metadata or {}
        user_message = ChatMessage(role="user", content=message, metadata=guardrail_metadata)
        blocked_response = await self._apply_input_guardrails(
            session_id=session_id,
            user_id=user_id,
            user_message=user_message,
            metadata=guardrail_metadata,
        )
        if blocked_response is not None:
            await self.session_manager.append_message(session_id, user_message)
            await self.session_manager.append_message(
                session_id,
                ChatMessage(
                    role="assistant",
                    content=blocked_response.content,
                    metadata={"model": blocked_response.model, **blocked_response.metadata},
                ),
            )
            return blocked_response

        user_message.importance = await self.importance_scorer.score(user_message)

        await self.events.emit(
            "before_context_build",
            {"session_id": session_id, "user_id": user_id, "message": user_message},
        )

        token_budget = self.token_budget_manager.allocate(self.config.system_prompt, user_message)
        recent_messages = await self.session_manager.get_recent_messages(session_id)
        long_term_memory = await self.storage.long_term.get_memory(user_id)
        retrieved_documents = await self._retrieve_documents(user_message, token_budget)
        summary = await self._maybe_compress(session_id, recent_messages, token_budget)

        context_bundle = await self.context_planner.plan(
            system_prompt=self.config.system_prompt,
            current_user_message=user_message,
            recent_messages=recent_messages,
            long_term_memory=long_term_memory,
            conversation_summary=summary,
            retrieved_documents=retrieved_documents,
            token_budget=token_budget,
        )
        prompt_messages = await self.prompt_builder.build(context_bundle)
        prompt_messages = await self._apply_prompt_guardrails(
            session_id=session_id,
            user_id=user_id,
            prompt_messages=prompt_messages,
            metadata=guardrail_metadata,
        )

        await self.events.emit(
            "after_context_build",
            {
                "session_id": session_id,
                "user_id": user_id,
                "context": context_bundle,
                "prompt_messages": prompt_messages,
            },
        )

        await self.events.emit(
            "before_llm_call",
            {"session_id": session_id, "user_id": user_id, "messages": prompt_messages},
        )
        response = await self.llm_provider.generate(prompt_messages, **llm_kwargs)
        await self.events.emit(
            "after_llm_call",
            {"session_id": session_id, "user_id": user_id, "response": response},
        )
        response = await self._apply_output_guardrails(
            session_id=session_id,
            user_id=user_id,
            response=response,
            metadata=guardrail_metadata,
        )

        assistant_message = ChatMessage(
            role="assistant",
            content=response.content,
            metadata={"model": response.model, **response.metadata},
        )
        assistant_message.importance = await self.importance_scorer.score(assistant_message)
        await self.session_manager.append_message(session_id, user_message)
        await self.session_manager.append_message(session_id, assistant_message)

        await self._maybe_update_long_term_memory(
            session_id=session_id,
            user_id=user_id,
            existing_memory=long_term_memory,
            new_messages=[user_message, assistant_message],
        )

        return response

    async def chat_stream(
        self,
        *,
        session_id: str,
        user_id: str,
        message: str,
        metadata: dict[str, Any] | None = None,
        **llm_kwargs: Any,
    ) -> AsyncIterator[LLMStreamEvent]:
        """Run one ContextVault chat interaction and stream LLM text deltas."""

        guardrail_metadata = metadata or {}
        user_message = ChatMessage(role="user", content=message, metadata=guardrail_metadata)
        blocked_response = await self._apply_input_guardrails(
            session_id=session_id,
            user_id=user_id,
            user_message=user_message,
            metadata=guardrail_metadata,
        )
        if blocked_response is not None:
            await self._store_completed_turn(
                session_id=session_id,
                user_message=user_message,
                response=blocked_response,
                existing_memory=None,
                user_id=user_id,
            )
            if blocked_response.content:
                yield LLMStreamEvent(
                    type="response.output_text.delta",
                    delta=blocked_response.content,
                    model=blocked_response.model,
                    metadata=blocked_response.metadata,
                )
            yield LLMStreamEvent(
                type="response.completed",
                response=blocked_response,
                model=blocked_response.model,
            )
            return

        user_message.importance = await self.importance_scorer.score(user_message)

        await self.events.emit(
            "before_context_build",
            {"session_id": session_id, "user_id": user_id, "message": user_message},
        )

        token_budget = self.token_budget_manager.allocate(self.config.system_prompt, user_message)
        recent_messages = await self.session_manager.get_recent_messages(session_id)
        long_term_memory = await self.storage.long_term.get_memory(user_id)
        retrieved_documents = await self._retrieve_documents(user_message, token_budget)
        summary = await self._maybe_compress(session_id, recent_messages, token_budget)

        context_bundle = await self.context_planner.plan(
            system_prompt=self.config.system_prompt,
            current_user_message=user_message,
            recent_messages=recent_messages,
            long_term_memory=long_term_memory,
            conversation_summary=summary,
            retrieved_documents=retrieved_documents,
            token_budget=token_budget,
        )
        prompt_messages = await self.prompt_builder.build(context_bundle)
        prompt_messages = await self._apply_prompt_guardrails(
            session_id=session_id,
            user_id=user_id,
            prompt_messages=prompt_messages,
            metadata=guardrail_metadata,
        )

        await self.events.emit(
            "after_context_build",
            {
                "session_id": session_id,
                "user_id": user_id,
                "context": context_bundle,
                "prompt_messages": prompt_messages,
            },
        )

        await self.events.emit(
            "before_llm_call",
            {"session_id": session_id, "user_id": user_id, "messages": prompt_messages},
        )
        response_parts: list[str] = []
        streamed_response: LLMResponse | None = None
        async for event in self.llm_provider.stream_generate(prompt_messages, **llm_kwargs):
            if event.type == "response.output_text.delta":
                response_parts.append(event.delta)
                yield event
            elif event.type == "response.completed":
                streamed_response = event.response
            else:
                yield event

        response = streamed_response or LLMResponse(
            content="".join(response_parts),
            model=getattr(self.llm_provider, "model", None),
        )
        if not response.content and response_parts:
            response = response.model_copy(update={"content": "".join(response_parts)})

        await self.events.emit(
            "after_llm_call",
            {"session_id": session_id, "user_id": user_id, "response": response},
        )
        guarded_response = await self._apply_output_guardrails(
            session_id=session_id,
            user_id=user_id,
            response=response,
            metadata=guardrail_metadata,
        )
        if guarded_response.content != response.content:
            yield LLMStreamEvent(
                type="response.output_text.rewrite",
                delta=guarded_response.content,
                response=guarded_response,
                model=guarded_response.model,
                metadata=guarded_response.metadata,
            )

        await self._store_completed_turn(
            session_id=session_id,
            user_id=user_id,
            user_message=user_message,
            response=guarded_response,
            existing_memory=long_term_memory,
        )
        yield LLMStreamEvent(
            type="response.completed",
            response=guarded_response,
            model=guarded_response.model,
        )

    async def _store_completed_turn(
        self,
        *,
        session_id: str,
        user_id: str,
        user_message: ChatMessage,
        response: LLMResponse,
        existing_memory: LongTermMemory | None,
    ) -> None:
        assistant_message = ChatMessage(
            role="assistant",
            content=response.content,
            metadata={"model": response.model, **response.metadata},
        )
        assistant_message.importance = await self.importance_scorer.score(assistant_message)
        await self.session_manager.append_message(session_id, user_message)
        await self.session_manager.append_message(session_id, assistant_message)
        if existing_memory is None:
            return
        await self._maybe_update_long_term_memory(
            session_id=session_id,
            user_id=user_id,
            existing_memory=existing_memory,
            new_messages=[user_message, assistant_message],
        )

    async def _apply_input_guardrails(
        self,
        *,
        session_id: str,
        user_id: str,
        user_message: ChatMessage,
        metadata: dict[str, Any],
    ) -> LLMResponse | None:
        for guardrail in self.guardrails:
            await self.events.emit(
                "before_input_guardrail",
                {
                    "session_id": session_id,
                    "user_id": user_id,
                    "guardrail": guardrail,
                    "message": user_message,
                },
            )
            result = await guardrail.check_input(
                session_id=session_id,
                user_id=user_id,
                message=user_message,
                metadata=metadata,
            )
            await self.events.emit(
                "after_input_guardrail",
                {
                    "session_id": session_id,
                    "user_id": user_id,
                    "guardrail": guardrail,
                    "message": user_message,
                    "result": result,
                },
            )
            if result.action == "rewrite" and result.content is not None:
                user_message.content = result.content
                user_message.metadata.setdefault("guardrails", []).append(
                    self._guardrail_metadata(guardrail, result)
                )
            elif result.action == "block":
                return self._guardrail_response(guardrail, result, blocked=True)
        return None

    async def _apply_prompt_guardrails(
        self,
        *,
        session_id: str,
        user_id: str,
        prompt_messages: list[ChatMessage],
        metadata: dict[str, Any],
    ) -> list[ChatMessage]:
        messages = prompt_messages
        for guardrail in self.guardrails:
            await self.events.emit(
                "before_prompt_guardrail",
                {
                    "session_id": session_id,
                    "user_id": user_id,
                    "guardrail": guardrail,
                    "messages": messages,
                },
            )
            messages = await guardrail.transform_prompt(
                session_id=session_id,
                user_id=user_id,
                prompt_messages=messages,
                metadata=metadata,
            )
            await self.events.emit(
                "after_prompt_guardrail",
                {
                    "session_id": session_id,
                    "user_id": user_id,
                    "guardrail": guardrail,
                    "messages": messages,
                },
            )
        return messages

    async def _apply_output_guardrails(
        self,
        *,
        session_id: str,
        user_id: str,
        response: LLMResponse,
        metadata: dict[str, Any],
    ) -> LLMResponse:
        current = response
        for guardrail in self.guardrails:
            await self.events.emit(
                "before_output_guardrail",
                {
                    "session_id": session_id,
                    "user_id": user_id,
                    "guardrail": guardrail,
                    "response": current,
                },
            )
            result = await guardrail.check_output(
                session_id=session_id,
                user_id=user_id,
                response=current,
                metadata=metadata,
            )
            await self.events.emit(
                "after_output_guardrail",
                {
                    "session_id": session_id,
                    "user_id": user_id,
                    "guardrail": guardrail,
                    "response": current,
                    "result": result,
                },
            )
            if result.action == "rewrite" and result.content is not None:
                current = self._rewrite_response(current, guardrail, result)
            elif result.action == "block":
                current = self._guardrail_response(guardrail, result, blocked=True)
                break
        return current

    def _rewrite_response(
        self,
        response: LLMResponse,
        guardrail: Guardrail,
        result: Any,
    ) -> LLMResponse:
        metadata = {
            **response.metadata,
            "guardrail_rewritten": True,
            "guardrail": self._guardrail_metadata(guardrail, result),
        }
        return response.model_copy(update={"content": result.content, "metadata": metadata})

    def _guardrail_response(
        self,
        guardrail: Guardrail,
        result: Any,
        *,
        blocked: bool,
    ) -> LLMResponse:
        content = result.content or self.config.guardrail_block_response
        return LLMResponse(
            content=content,
            model="guardrail",
            metadata={
                "guardrail_blocked": blocked,
                "guardrail": self._guardrail_metadata(guardrail, result),
            },
        )

    def _guardrail_metadata(self, guardrail: Guardrail, result: Any) -> dict[str, Any]:
        return {
            "name": guardrail.__class__.__name__,
            "action": result.action,
            "reason": result.reason,
            **result.metadata,
        }

    async def _retrieve_documents(
        self, user_message: ChatMessage, token_budget: TokenBudget
    ) -> list[SearchResult]:
        if not self.config.vector_search or self.vector_store is None:
            return []
        if isinstance(self.config.vector_top_k, int):
            limit = self.config.vector_top_k
        else:
            vector_budget = token_budget.allocations.get("vector_search", 0)
            limit = max(1, min(20, vector_budget // 250))
        return await self.vector_store.search(user_message.content, limit=limit)

    async def _maybe_compress(
        self,
        session_id: str,
        messages: list[ChatMessage],
        token_budget: TokenBudget,
    ) -> MemorySummary | None:
        if self.config.summary_strategy == "disabled" or not messages:
            return await self.storage.short_term.get_summary(session_id)

        short_budget = token_budget.allocations.get("short_term_memory", 0)
        threshold = int(short_budget * self.config.compression_threshold)
        if threshold <= 0 or self.token_counter.count_messages(messages) <= threshold:
            return await self.storage.short_term.get_summary(session_id)

        keep_recent = 6
        if len(messages) <= keep_recent:
            return await self.storage.short_term.get_summary(session_id)

        older_messages = messages[:-keep_recent]
        existing_summary = await self.storage.short_term.get_summary(session_id)
        older_ids = [message.id for message in older_messages]
        if existing_summary is not None and existing_summary.source_message_ids == older_ids:
            return existing_summary

        await self.events.emit(
            "before_summary",
            {"session_id": session_id, "messages": older_messages, "summary": existing_summary},
        )
        summary = await self.memory_compressor.compress(
            older_messages,
            existing_summary=existing_summary.content if existing_summary else None,
        )
        if existing_summary is not None:
            summary.level = existing_summary.level + 1
        await self.storage.short_term.save_summary(session_id, summary)
        await self.events.emit(
            "after_summary",
            {"session_id": session_id, "messages": older_messages, "summary": summary},
        )
        return summary

    async def _maybe_update_long_term_memory(
        self,
        *,
        session_id: str,
        user_id: str,
        existing_memory: LongTermMemory,
        new_messages: list[ChatMessage],
    ) -> None:
        if not self.config.auto_update_long_term:
            return
        message_count = await self.session_manager.count_messages(session_id)
        interaction_count = message_count // 2
        if interaction_count == 0:
            return
        if interaction_count % self.config.memory_update_frequency != 0:
            return

        conversation = await self.session_manager.get_recent_messages(session_id, limit=20)
        if not conversation:
            conversation = new_messages
        conversation = self._filter_messages_for_long_term_memory(conversation)
        if not conversation:
            return
        await self.events.emit(
            "before_memory_update",
            {"session_id": session_id, "user_id": user_id, "memory": existing_memory},
        )
        updated = await self.memory_extractor.extract(conversation, existing_memory)
        await self.storage.long_term.update_memory(user_id, updated)
        await self.events.emit(
            "after_memory_update",
            {"session_id": session_id, "user_id": user_id, "memory": updated},
        )
        logger.debug(
            "Updated long-term memory",
            extra={"session_id": session_id, "user_id": user_id},
        )

    def _filter_messages_for_long_term_memory(
        self,
        messages: list[ChatMessage],
    ) -> list[ChatMessage]:
        threshold = self.config.long_term_importance_threshold
        if threshold is None:
            return messages
        return [
            message
            for message in messages
            if message.importance is not None and message.importance >= threshold
        ]
