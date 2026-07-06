from __future__ import annotations

import unittest

from context_vault import ContextVault
from context_vault.guardrails import KeywordBlockGuardrail, StudentTeacherGuardrail
from context_vault.providers import MockLLMProvider
from context_vault.storage import InMemoryStorage


class GuardrailTests(unittest.IsolatedAsyncioTestCase):
    async def test_input_guardrail_blocks_before_llm_call(self) -> None:
        provider = MockLLMProvider(["should not be called"])
        storage = InMemoryStorage()
        vault = ContextVault(
            llm_provider=provider,
            storage=storage,
            guardrails=[
                KeywordBlockGuardrail(
                    blocked_input_terms=["unsafe"],
                    response="Please ask a safe learning question.",
                )
            ],
        )

        response = await vault.chat(
            session_id="s1",
            user_id="u1",
            message="unsafe request",
        )

        messages = await storage.short_term.get_messages("s1")
        self.assertEqual(response.content, "Please ask a safe learning question.")
        self.assertEqual(response.model, "guardrail")
        self.assertEqual(len(provider.calls), 0)
        self.assertEqual([message.role for message in messages], ["user", "assistant"])

    async def test_student_teacher_guardrail_injects_prompt_policy(self) -> None:
        provider = MockLLMProvider(["answer"])
        vault = ContextVault(
            llm_provider=provider,
            guardrails=[StudentTeacherGuardrail(student_age=12, subject="math")],
        )

        await vault.chat(
            session_id="s1",
            user_id="u1",
            message="explain fractions",
        )

        prompt = provider.calls[0]
        self.assertTrue(any("Guardrail policy:" in message.content for message in prompt))
        self.assertTrue(any("12-year-old student" in message.content for message in prompt))
        self.assertTrue(any("math" in message.content for message in prompt))

    async def test_output_guardrail_rewrites_model_response(self) -> None:
        provider = MockLLMProvider(["this contains forbidden text"])
        vault = ContextVault(
            llm_provider=provider,
            guardrails=[
                KeywordBlockGuardrail(
                    blocked_output_terms=["forbidden"],
                    response="Here is a safer educational response.",
                )
            ],
        )

        response = await vault.chat(
            session_id="s1",
            user_id="u1",
            message="hello",
        )

        self.assertEqual(response.content, "Here is a safer educational response.")
        self.assertTrue(response.metadata["guardrail_rewritten"])


if __name__ == "__main__":
    unittest.main()
