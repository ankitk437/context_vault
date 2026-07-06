"""Keyword-based guardrails."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from context_vault.interfaces import Guardrail
from context_vault.models import ChatMessage, GuardrailResult, LLMResponse


class KeywordBlockGuardrail(Guardrail):
    """Block inputs or outputs containing configured terms."""

    def __init__(
        self,
        *,
        blocked_input_terms: Iterable[str] | None = None,
        blocked_output_terms: Iterable[str] | None = None,
        response: str = "I cannot help with that request.",
        case_sensitive: bool = False,
    ) -> None:
        self.blocked_input_terms = list(blocked_input_terms or [])
        self.blocked_output_terms = list(blocked_output_terms or [])
        self.response = response
        self.case_sensitive = case_sensitive

    async def check_input(
        self,
        *,
        session_id: str,
        user_id: str,
        message: ChatMessage,
        metadata: dict[str, Any],
    ) -> GuardrailResult:
        term = self._find_term(message.content, self.blocked_input_terms)
        if term is None:
            return GuardrailResult.allow()
        return GuardrailResult.block(
            self.response,
            reason=f"input contained blocked term: {term}",
            guardrail="keyword_block",
            term=term,
        )

    async def check_output(
        self,
        *,
        session_id: str,
        user_id: str,
        response: LLMResponse,
        metadata: dict[str, Any],
    ) -> GuardrailResult:
        term = self._find_term(response.content, self.blocked_output_terms)
        if term is None:
            return GuardrailResult.allow()
        return GuardrailResult.rewrite(
            self.response,
            reason=f"output contained blocked term: {term}",
            guardrail="keyword_block",
            term=term,
        )

    def _find_term(self, text: str, terms: list[str]) -> str | None:
        haystack = text if self.case_sensitive else text.lower()
        for term in terms:
            needle = term if self.case_sensitive else term.lower()
            if needle and needle in haystack:
                return term
        return None
