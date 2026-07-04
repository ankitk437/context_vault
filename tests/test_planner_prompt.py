from __future__ import annotations

import unittest

from context_vault.config import VaultConfig
from context_vault.models import ChatMessage, LongTermMemory
from context_vault.planner import ContextPlanner, TokenBudgetManager
from context_vault.prompt import DefaultPromptBuilder
from context_vault.utils import RoughTokenCounter


class PlannerPromptTests(unittest.IsolatedAsyncioTestCase):
    async def test_token_budget_allocates_sections(self) -> None:
        config = VaultConfig(max_context_tokens=1_000, reserved_output_tokens=100)
        counter = RoughTokenCounter()
        manager = TokenBudgetManager(config, counter)

        budget = manager.allocate("system", ChatMessage(role="user", content="hello"))

        self.assertEqual(budget.available_input_tokens, 900)
        self.assertIn("short_term_memory", budget.allocations)
        self.assertGreater(budget.allocations["short_term_memory"], 0)

    async def test_context_planner_respects_recent_message_limit(self) -> None:
        config = VaultConfig(recent_message_limit=2)
        counter = RoughTokenCounter()
        manager = TokenBudgetManager(config, counter)
        planner = ContextPlanner(config, counter)
        current = ChatMessage(role="user", content="now")
        budget = manager.allocate(config.system_prompt, current)
        history = [ChatMessage(role="user", content=str(index)) for index in range(5)]

        bundle = await planner.plan(
            system_prompt=config.system_prompt,
            current_user_message=current,
            recent_messages=history,
            long_term_memory=None,
            conversation_summary=None,
            retrieved_documents=[],
            token_budget=budget,
        )

        self.assertEqual([message.content for message in bundle.recent_messages], ["3", "4"])

    async def test_prompt_builder_includes_memory_before_current_message(self) -> None:
        config = VaultConfig()
        counter = RoughTokenCounter()
        manager = TokenBudgetManager(config, counter)
        planner = ContextPlanner(config, counter)
        builder = DefaultPromptBuilder(config)
        current = ChatMessage(role="user", content="Who am I?")
        budget = manager.allocate(config.system_prompt, current)
        memory = LongTermMemory(user_id="u1", facts={"name": "Ankit"})

        bundle = await planner.plan(
            system_prompt=config.system_prompt,
            current_user_message=current,
            recent_messages=[],
            long_term_memory=memory,
            conversation_summary=None,
            retrieved_documents=[],
            token_budget=budget,
        )
        messages = await builder.build(bundle)

        self.assertEqual(messages[-1].content, "Who am I?")
        self.assertTrue(any("name: Ankit" in message.content for message in messages))


if __name__ == "__main__":
    unittest.main()
