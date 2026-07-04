from __future__ import annotations

import unittest

from context_vault import ContextVault, VaultConfig
from context_vault.compression import LLMMemoryCompressor
from context_vault.importance import LLMImportanceScorer
from context_vault.memory import LLMMemoryExtractor
from context_vault.providers import MockLLMProvider


class PerActionLLMTests(unittest.IsolatedAsyncioTestCase):
    async def test_constructor_uses_per_action_llm_components(self) -> None:
        chat = MockLLMProvider(["chat"], model="chat")
        memory = MockLLMProvider(['{"name": "Ankit"}'], model="memory")
        compression = MockLLMProvider(["summary"], model="compression")
        importance = MockLLMProvider(["0.9", "0.9"], model="importance")

        vault = ContextVault(
            llm_provider=chat,
            memory_llm_provider=memory,
            compression_llm_provider=compression,
            importance_llm_provider=importance,
            config=VaultConfig(memory_update_frequency=1),
        )

        self.assertIsInstance(vault.memory_extractor, LLMMemoryExtractor)
        self.assertIsInstance(vault.memory_compressor, LLMMemoryCompressor)
        self.assertIsInstance(vault.importance_scorer, LLMImportanceScorer)

    async def test_memory_action_can_use_different_model_from_chat(self) -> None:
        chat = MockLLMProvider(["chat answer"], model="chat-model")
        memory = MockLLMProvider(['{"name": "Ankit"}'], model="memory-model")

        vault = ContextVault(
            llm_provider=chat,
            memory_llm_provider=memory,
            config=VaultConfig(memory_update_frequency=1),
        )

        response = await vault.chat(
            session_id="s1",
            user_id="u1",
            message="My name is Ankit.",
        )
        stored = await vault.storage.long_term.get_memory("u1")

        self.assertEqual(response.model, "chat-model")
        self.assertEqual(stored.facts["name"], "Ankit")
        self.assertEqual(len(chat.calls), 1)
        self.assertEqual(len(memory.calls), 1)


if __name__ == "__main__":
    unittest.main()
