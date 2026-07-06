from __future__ import annotations

import unittest

from context_vault import ContextVault, VaultConfig
from context_vault.interfaces import ImportanceScorer
from context_vault.models import ChatMessage, Document
from context_vault.providers import EchoLLMProvider, MockLLMProvider
from context_vault.storage import InMemoryStorage
from context_vault.vectorstores import InMemoryVectorStore


class FixedImportanceScorer(ImportanceScorer):
    def __init__(self, score: float) -> None:
        self.score_value = score

    async def score(self, message: ChatMessage) -> float:
        return self.score_value


class ChatPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_chat_stores_user_and_assistant_messages(self) -> None:
        storage = InMemoryStorage()
        vault = ContextVault(llm_provider=EchoLLMProvider(), storage=storage)

        response = await vault.chat(
            session_id="s1",
            user_id="u1",
            message="hello",
        )

        messages = await storage.short_term.get_messages("s1")
        self.assertEqual(response.content, "Echo: hello")
        self.assertEqual([message.role for message in messages], ["user", "assistant"])

    async def test_memory_update_frequency_extracts_long_term_memory(self) -> None:
        storage = InMemoryStorage()
        vault = ContextVault(
            llm_provider=EchoLLMProvider(),
            storage=storage,
            config=VaultConfig(memory_update_frequency=1),
        )

        await vault.chat(
            session_id="s1",
            user_id="u1",
            message="My name is Ankit. I live in Bangalore.",
        )

        memory = await storage.long_term.get_memory("u1")
        self.assertEqual(memory.facts["name"], "Ankit")
        self.assertEqual(memory.facts["location"], "Bangalore")

    async def test_long_term_memory_can_use_importance_threshold(self) -> None:
        storage = InMemoryStorage()
        vault = ContextVault(
            llm_provider=EchoLLMProvider(),
            storage=storage,
            importance_scorer=FixedImportanceScorer(0.95),
            config=VaultConfig(memory_update_frequency=1, long_term_importance_threshold=0.8),
        )

        await vault.chat(
            session_id="s1",
            user_id="u1",
            message="My name is Ankit.",
        )

        memory = await storage.long_term.get_memory("u1")
        self.assertEqual(memory.facts["name"], "Ankit")

    async def test_low_importance_messages_are_not_promoted_to_long_term_memory(self) -> None:
        storage = InMemoryStorage()
        vault = ContextVault(
            llm_provider=EchoLLMProvider(),
            storage=storage,
            importance_scorer=FixedImportanceScorer(0.2),
            config=VaultConfig(memory_update_frequency=1, long_term_importance_threshold=0.8),
        )

        await vault.chat(
            session_id="s1",
            user_id="u1",
            message="My name is Ankit.",
        )

        memory = await storage.long_term.get_memory("u1")
        self.assertEqual(memory.facts, {})

    async def test_can_seed_long_term_memory_from_known_user_metadata(self) -> None:
        provider = MockLLMProvider(["done"])
        vault = ContextVault(
            llm_provider=provider,
            config=VaultConfig(
                long_term_include_fields=["name", "grade_level", "preferences"],
                long_term_exclude_fields=["private_note"],
            ),
        )

        seeded = await vault.remember_user(
            user_id="u1",
            facts={
                "name": "Ankit",
                "grade_level": "8",
                "preferences": "short examples",
                "private_note": "do not store",
            },
            metadata={"source": "user_profile"},
            respect_include_fields=True,
        )

        await vault.chat(session_id="s1", user_id="u1", message="Explain gravity.")

        prompt_text = "\n".join(message.content for message in provider.calls[0])
        self.assertEqual(seeded.facts["name"], "Ankit")
        self.assertEqual(seeded.metadata["source"], "user_profile")
        self.assertNotIn("private_note", seeded.facts)
        self.assertIn("- name: Ankit", prompt_text)
        self.assertIn("- grade_level: 8", prompt_text)
        self.assertIn("- preferences: short examples", prompt_text)

    async def test_vector_search_is_disabled_by_default(self) -> None:
        vector_store = InMemoryVectorStore()
        await vector_store.add_documents([Document(content="Python architecture guide")])
        provider = MockLLMProvider(["done"])
        vault = ContextVault(llm_provider=provider, vector_store=vector_store)

        await vault.chat(session_id="s1", user_id="u1", message="Python")

        prompt = provider.calls[0]
        self.assertFalse(any("Retrieved context:" in message.content for message in prompt))

    async def test_vector_search_can_inject_documents(self) -> None:
        vector_store = InMemoryVectorStore()
        await vector_store.add_documents([Document(content="Python architecture guide")])
        provider = MockLLMProvider(["done"])
        vault = ContextVault(
            llm_provider=provider,
            vector_store=vector_store,
            config=VaultConfig(vector_search=True, vector_top_k=3),
        )

        await vault.chat(session_id="s1", user_id="u1", message="Python")

        prompt = provider.calls[0]
        self.assertTrue(any("Retrieved context:" in message.content for message in prompt))


if __name__ == "__main__":
    unittest.main()
