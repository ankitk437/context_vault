from __future__ import annotations

import unittest

from context_vault.models import ChatMessage, LongTermMemory, MemorySummary
from context_vault.storage import InMemoryLongTermMemoryStore, InMemoryShortTermMemoryStore


class InMemoryStorageTests(unittest.IsolatedAsyncioTestCase):
    async def test_short_term_memory_stores_messages_by_session(self) -> None:
        store = InMemoryShortTermMemoryStore()
        await store.append_message("s1", ChatMessage(role="user", content="hello"))
        await store.append_message("s2", ChatMessage(role="user", content="other"))

        messages = await store.get_messages("s1")

        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].content, "hello")

    async def test_short_term_memory_returns_recent_limit(self) -> None:
        store = InMemoryShortTermMemoryStore()
        for index in range(5):
            await store.append_message("s1", ChatMessage(role="user", content=str(index)))

        messages = await store.get_messages("s1", limit=2)

        self.assertEqual([message.content for message in messages], ["3", "4"])

    async def test_summary_roundtrip(self) -> None:
        store = InMemoryShortTermMemoryStore()
        summary = MemorySummary(content="older context")

        await store.save_summary("s1", summary)

        loaded = await store.get_summary("s1")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.content, "older context")

    async def test_long_term_memory_roundtrip(self) -> None:
        store = InMemoryLongTermMemoryStore()
        memory = LongTermMemory(user_id="u1", facts={"name": "Ankit"})

        await store.update_memory("u1", memory)
        loaded = await store.get_memory("u1")

        self.assertEqual(loaded.facts["name"], "Ankit")


if __name__ == "__main__":
    unittest.main()
