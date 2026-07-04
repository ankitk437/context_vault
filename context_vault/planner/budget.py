"""Token budget allocation."""

from __future__ import annotations

from context_vault.config import VaultConfig
from context_vault.interfaces import TokenCounter
from context_vault.models import ChatMessage, TokenBudget


class TokenBudgetManager:
    """Allocates a context window across configurable sections."""

    def __init__(self, config: VaultConfig, token_counter: TokenCounter) -> None:
        self.config = config
        self.token_counter = token_counter

    def allocate(self, system_prompt: str, current_user_message: ChatMessage) -> TokenBudget:
        """Allocate token budgets for one request."""

        available_input_tokens = max(
            0, self.config.max_context_tokens - self.config.reserved_output_tokens
        )
        fixed_tokens = {
            "system": self.token_counter.count_text(system_prompt),
            "current_user_message": self.token_counter.count_messages([current_user_message]),
        }
        flexible_tokens = max(0, available_input_tokens - sum(fixed_tokens.values()))
        total_weight = sum(self.config.token_budget_weights.values())
        allocations = {
            name: int(flexible_tokens * (weight / total_weight))
            for name, weight in self.config.token_budget_weights.items()
        }
        allocated = sum(allocations.values())
        if allocations and allocated < flexible_tokens:
            first_key = next(iter(allocations))
            allocations[first_key] += flexible_tokens - allocated
        return TokenBudget(
            max_context_tokens=self.config.max_context_tokens,
            reserved_output_tokens=self.config.reserved_output_tokens,
            available_input_tokens=available_input_tokens,
            fixed_tokens=fixed_tokens,
            allocations=allocations,
        )
