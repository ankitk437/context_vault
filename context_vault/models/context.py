"""Context planning models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from context_vault.models.chat import ChatMessage
from context_vault.models.documents import SearchResult
from context_vault.models.memory import LongTermMemory, MemorySummary


class TokenBudget(BaseModel):
    """Allocated token budget for one LLM request."""

    max_context_tokens: int = Field(gt=0)
    reserved_output_tokens: int = Field(ge=0)
    available_input_tokens: int = Field(ge=0)
    fixed_tokens: dict[str, int] = Field(default_factory=dict)
    allocations: dict[str, int] = Field(default_factory=dict)

    @property
    def flexible_input_tokens(self) -> int:
        """Return tokens left after fixed sections such as system and current user input."""

        return max(0, self.available_input_tokens - sum(self.fixed_tokens.values()))


class ContextBundle(BaseModel):
    """All context sections selected for a single LLM request."""

    system_prompt: str
    current_user_message: ChatMessage
    token_budget: TokenBudget
    long_term_memory: LongTermMemory | None = None
    conversation_summary: MemorySummary | None = None
    recent_messages: list[ChatMessage] = Field(default_factory=list)
    retrieved_documents: list[SearchResult] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
