"""Runtime configuration for ContextVault."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


RecentMessageLimit = int | Literal["adaptive"]
VectorTopK = int | Literal["adaptive"]


class VaultConfig(BaseModel):
    """Configuration for the ContextVault orchestration pipeline."""

    max_context_tokens: int = Field(default=128_000, gt=0)
    reserved_output_tokens: int = Field(default=4_000, ge=0)
    memory_update_frequency: int = Field(default=10, gt=0)
    recent_message_limit: RecentMessageLimit = "adaptive"
    summary_strategy: Literal["recursive", "replace", "disabled"] = "recursive"
    importance_strategy: Literal["rule_based", "llm", "embedding"] = "rule_based"
    compression_threshold: float = Field(default=0.85, gt=0, le=1)
    auto_update_long_term: bool = True
    vector_search: bool = False
    vector_top_k: VectorTopK = "adaptive"
    system_prompt: str = "You are a helpful assistant."
    extraction_prompt: str | None = None
    prompt_order: list[str] = Field(
        default_factory=lambda: [
            "system",
            "long_term_memory",
            "conversation_summary",
            "recent_messages",
            "retrieved_documents",
            "current_user_message",
        ]
    )
    long_term_include_fields: list[str] = Field(
        default_factory=lambda: ["name", "location", "preferences", "goals", "interests"]
    )
    long_term_exclude_fields: list[str] = Field(default_factory=list)
    token_budget_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "short_term_memory": 0.40,
            "long_term_memory": 0.20,
            "vector_search": 0.30,
            "metadata": 0.10,
        }
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("reserved_output_tokens")
    @classmethod
    def _reserved_must_fit_context(cls, value: int) -> int:
        return value

    @field_validator("prompt_order")
    @classmethod
    def _prompt_order_not_empty(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("prompt_order must include at least one section")
        return value

    @field_validator("token_budget_weights")
    @classmethod
    def _weights_must_be_non_negative(cls, value: dict[str, float]) -> dict[str, float]:
        if not value:
            raise ValueError("token_budget_weights must not be empty")
        negative = [key for key, weight in value.items() if weight < 0]
        if negative:
            raise ValueError(f"token_budget_weights cannot be negative: {negative}")
        if sum(value.values()) <= 0:
            raise ValueError("token_budget_weights must sum to a positive value")
        return value

    @model_validator(mode="after")
    def _validate_context_capacity(self) -> "VaultConfig":
        if self.reserved_output_tokens >= self.max_context_tokens:
            raise ValueError("reserved_output_tokens must be less than max_context_tokens")
        overlap = set(self.long_term_include_fields) & set(self.long_term_exclude_fields)
        if overlap:
            raise ValueError(f"fields cannot be both included and excluded: {sorted(overlap)}")
        return self
