"""Student-teacher guardrail example.

This example is fully local and deterministic. It shows how to keep a learning
assistant inside an age-appropriate student/teacher interaction pattern.

Run:
    python examples/student_teacher_guardrails.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from context_vault import ContextVault
from context_vault.guardrails import StudentTeacherGuardrail
from context_vault.providers import MockLLMProvider


async def main() -> None:
    provider = MockLLMProvider(
        [
            "Here is a guided explanation with a small example.",
            "This answer includes a disallowed phrase.",
        ],
        model="local-demo-model",
    )
    guardrail = StudentTeacherGuardrail(
        student_age=13,
        subject="science",
        blocked_input_terms=["adult-only"],
        blocked_output_terms=["disallowed phrase"],
        extra_instructions="Ask one short follow-up question when useful.",
    )
    vault = ContextVault(llm_provider=provider, guardrails=[guardrail])

    normal = await vault.chat(
        session_id="classroom-1",
        user_id="student-1",
        message="Can you explain photosynthesis?",
    )
    blocked_input = await vault.chat(
        session_id="classroom-1",
        user_id="student-1",
        message="adult-only topic please",
    )
    rewritten_output = await vault.chat(
        session_id="classroom-1",
        user_id="student-1",
        message="Give another science example.",
    )

    print(f"normal response: {normal.content}")
    print(f"blocked input response: {blocked_input.content}")
    print(f"rewritten output response: {rewritten_output.content}")
    print("\nPrompt sent to LLM for first turn:")
    for message in provider.calls[0]:
        print(f"- {message.role}: {message.content}")


if __name__ == "__main__":
    asyncio.run(main())
