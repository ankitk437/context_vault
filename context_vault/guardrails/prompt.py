"""Prompt-instruction guardrails."""

from __future__ import annotations

from typing import Any

from context_vault.interfaces import Guardrail
from context_vault.models import ChatMessage


class PromptInstructionGuardrail(Guardrail):
    """Inject policy instructions into the prompt sent to the LLM."""

    def __init__(self, instructions: str, *, position: str = "after_system") -> None:
        self.instructions = instructions.strip()
        self.position = position

    async def transform_prompt(
        self,
        *,
        session_id: str,
        user_id: str,
        prompt_messages: list[ChatMessage],
        metadata: dict[str, Any],
    ) -> list[ChatMessage]:
        if not self.instructions:
            return prompt_messages

        guardrail_message = ChatMessage(
            role="system",
            content=f"Guardrail policy:\n{self.instructions}",
            metadata={"guardrail": self.__class__.__name__},
        )
        messages = [message.model_copy(deep=True) for message in prompt_messages]
        if self.position == "first":
            return [guardrail_message, *messages]
        if self.position == "last":
            return [*messages, guardrail_message]

        insert_at = 1 if messages and messages[0].role == "system" else 0
        return [*messages[:insert_at], guardrail_message, *messages[insert_at:]]
