"""Guardrail extension interface."""

from __future__ import annotations

from typing import Any

from context_vault.models import ChatMessage, GuardrailResult, LLMResponse


class Guardrail:
    """Extensible input, prompt, and output policy hook."""

    async def check_input(
        self,
        *,
        session_id: str,
        user_id: str,
        message: ChatMessage,
        metadata: dict[str, Any],
    ) -> GuardrailResult:
        """Validate or rewrite the current user message."""

        return GuardrailResult.allow()

    async def transform_prompt(
        self,
        *,
        session_id: str,
        user_id: str,
        prompt_messages: list[ChatMessage],
        metadata: dict[str, Any],
    ) -> list[ChatMessage]:
        """Optionally add or rewrite prompt messages before the LLM call."""

        return prompt_messages

    async def check_output(
        self,
        *,
        session_id: str,
        user_id: str,
        response: LLMResponse,
        metadata: dict[str, Any],
    ) -> GuardrailResult:
        """Validate or rewrite the model response before it is stored or returned."""

        return GuardrailResult.allow()
