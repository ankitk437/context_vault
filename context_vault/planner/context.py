"""Adaptive context planning."""

from __future__ import annotations

from context_vault.config import VaultConfig
from context_vault.interfaces import TokenCounter
from context_vault.models import (
    ChatMessage,
    ContextBundle,
    Document,
    LongTermMemory,
    MemorySummary,
    SearchResult,
    TokenBudget,
)
from context_vault.utils import trim_text_to_token_budget


class ContextPlanner:
    """Selects context sections that fit inside allocated token budgets."""

    def __init__(self, config: VaultConfig, token_counter: TokenCounter) -> None:
        self.config = config
        self.token_counter = token_counter

    async def plan(
        self,
        *,
        system_prompt: str,
        current_user_message: ChatMessage,
        recent_messages: list[ChatMessage],
        long_term_memory: LongTermMemory | None,
        conversation_summary: MemorySummary | None,
        retrieved_documents: list[SearchResult],
        token_budget: TokenBudget,
    ) -> ContextBundle:
        """Build a context bundle that respects the token budget."""

        short_budget = token_budget.allocations.get("short_term_memory", 0)
        long_budget = token_budget.allocations.get("long_term_memory", 0)
        vector_budget = token_budget.allocations.get("vector_search", 0)

        selected_summary = self._fit_summary(conversation_summary, short_budget // 3)
        summary_tokens = (
            self.token_counter.count_text(selected_summary.content) if selected_summary else 0
        )
        selected_messages = self._fit_recent_messages(
            recent_messages,
            max(0, short_budget - summary_tokens),
        )
        selected_memory = self._fit_long_term_memory(long_term_memory, long_budget)
        selected_documents = self._fit_documents(retrieved_documents, vector_budget)

        return ContextBundle(
            system_prompt=system_prompt,
            current_user_message=current_user_message,
            token_budget=token_budget,
            long_term_memory=selected_memory,
            conversation_summary=selected_summary,
            recent_messages=selected_messages,
            retrieved_documents=selected_documents,
        )

    def _fit_recent_messages(
        self, messages: list[ChatMessage], token_budget: int
    ) -> list[ChatMessage]:
        if token_budget <= 0:
            return []
        source_messages = messages
        if isinstance(self.config.recent_message_limit, int):
            source_messages = source_messages[-self.config.recent_message_limit :]

        selected_reversed: list[ChatMessage] = []
        used = 0
        for message in reversed(source_messages):
            message_tokens = self.token_counter.count_messages([message])
            if used + message_tokens > token_budget:
                break
            selected_reversed.append(message)
            used += message_tokens
        return list(reversed(selected_reversed))

    def _fit_summary(
        self, summary: MemorySummary | None, token_budget: int
    ) -> MemorySummary | None:
        if summary is None or not summary.content or token_budget <= 0:
            return None
        content = trim_text_to_token_budget(summary.content, token_budget, self.token_counter)
        return summary.model_copy(
            update={"content": content, "token_count": self.token_counter.count_text(content)}
        )

    def _fit_long_term_memory(
        self, memory: LongTermMemory | None, token_budget: int
    ) -> LongTermMemory | None:
        if memory is None or not memory.facts or token_budget <= 0:
            return None
        filtered = memory.filtered(
            self.config.long_term_include_fields, self.config.long_term_exclude_fields
        )
        selected_facts: dict[str, object] = {}
        for key in sorted(filtered.facts):
            candidate_facts = {**selected_facts, key: filtered.facts[key]}
            candidate = filtered.model_copy(update={"facts": candidate_facts})
            if self.token_counter.count_text(candidate.render()) <= token_budget:
                selected_facts = candidate_facts
            elif not selected_facts:
                trimmed_value = trim_text_to_token_budget(
                    str(filtered.facts[key]), max(1, token_budget - 12), self.token_counter
                )
                selected_facts[key] = trimmed_value
                break
        if not selected_facts:
            return None
        return filtered.model_copy(update={"facts": selected_facts})

    def _fit_documents(
        self, documents: list[SearchResult], token_budget: int
    ) -> list[SearchResult]:
        if token_budget <= 0:
            return []
        selected: list[SearchResult] = []
        used = 0
        for result in documents:
            document_tokens = self.token_counter.count_text(result.document.content)
            if used + document_tokens <= token_budget:
                selected.append(result)
                used += document_tokens
                continue
            remaining = token_budget - used
            if remaining <= 0:
                break
            trimmed = trim_text_to_token_budget(
                result.document.content, remaining, self.token_counter
            )
            if trimmed:
                selected.append(
                    result.model_copy(
                        update={
                            "document": Document(
                                id=result.document.id,
                                content=trimmed,
                                metadata=result.document.metadata,
                            )
                        }
                    )
                )
            break
        return selected
