"""Student and teacher interaction guardrails."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from context_vault.guardrails.keyword import KeywordBlockGuardrail
from context_vault.guardrails.prompt import PromptInstructionGuardrail
from context_vault.models import ChatMessage, GuardrailResult, LLMResponse


class StudentTeacherGuardrail(PromptInstructionGuardrail):
    """Guardrail preset for education-focused student/teacher conversations."""

    def __init__(
        self,
        *,
        student_age: int | None = None,
        role: str = "teacher",
        subject: str | None = None,
        blocked_input_terms: Iterable[str] | None = None,
        blocked_output_terms: Iterable[str] | None = None,
        response: str = (
            "I cannot help with that, but I can help explain the concept in a safe, "
            "age-appropriate way."
        ),
        extra_instructions: str | None = None,
    ) -> None:
        self.student_age = student_age
        self.role = role
        self.subject = subject
        self.keyword_guardrail = KeywordBlockGuardrail(
            blocked_input_terms=blocked_input_terms,
            blocked_output_terms=blocked_output_terms,
            response=response,
        )
        super().__init__(self._build_instructions(extra_instructions))

    async def check_input(
        self,
        *,
        session_id: str,
        user_id: str,
        message: ChatMessage,
        metadata: dict[str, Any],
    ) -> GuardrailResult:
        return await self.keyword_guardrail.check_input(
            session_id=session_id,
            user_id=user_id,
            message=message,
            metadata=metadata,
        )

    async def check_output(
        self,
        *,
        session_id: str,
        user_id: str,
        response: LLMResponse,
        metadata: dict[str, Any],
    ) -> GuardrailResult:
        return await self.keyword_guardrail.check_output(
            session_id=session_id,
            user_id=user_id,
            response=response,
            metadata=metadata,
        )

    def _build_instructions(self, extra_instructions: str | None) -> str:
        audience = "the student"
        if self.student_age is not None:
            audience = f"a {self.student_age}-year-old student"

        subject_line = f" The subject is {self.subject}." if self.subject else ""
        lines = [
            f"You are acting as a {self.role} for {audience}.{subject_line}",
            "Keep responses educational, respectful, and age-appropriate.",
            "Prefer guided explanations, hints, examples, and questions over direct shortcuts.",
            "Do not shame the learner or use manipulative, adult, violent, or unsafe framing.",
            "If the request is outside the learning scope, redirect to a safe educational answer.",
        ]
        if extra_instructions:
            lines.append(extra_instructions.strip())
        return "\n".join(lines)
