"""Check ContextVault memory persistence behavior.

The default in-memory storage persists for the lifetime of the storage object only.
It is useful for tests and demos, but it is not durable across process restarts.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from context_vault import ContextVault, VaultConfig
from context_vault.providers import EchoLLMProvider
from context_vault.storage import InMemoryStorage


async def main() -> None:
    config = VaultConfig(memory_update_frequency=1)
    shared_storage = InMemoryStorage()

    first_vault = ContextVault(
        llm_provider=EchoLLMProvider(),
        storage=shared_storage,
        config=config,
    )

    await first_vault.chat(
        session_id="persistence-demo",
        user_id="user-1",
        message="My name is Ankit. I live in Bangalore. I prefer Python.",
    )

    same_storage_memory = await shared_storage.long_term.get_memory("user-1")
    same_storage_messages = await shared_storage.short_term.get_messages("persistence-demo")

    print("1. Same vault/storage object:")
    print(f"   messages stored: {len(same_storage_messages)}")
    print(f"   long-term facts: {same_storage_memory.facts}")

    second_vault = ContextVault(
        llm_provider=EchoLLMProvider(),
        storage=shared_storage,
        config=config,
    )
    shared_again_memory = await second_vault.storage.long_term.get_memory("user-1")
    shared_again_messages = await second_vault.storage.short_term.get_messages("persistence-demo")

    print("2. New vault with the same storage object:")
    print(f"   messages stored: {len(shared_again_messages)}")
    print(f"   long-term facts: {shared_again_memory.facts}")

    fresh_vault = ContextVault(
        llm_provider=EchoLLMProvider(),
        storage=InMemoryStorage(),
        config=config,
    )
    fresh_memory = await fresh_vault.storage.long_term.get_memory("user-1")
    fresh_messages = await fresh_vault.storage.short_term.get_messages("persistence-demo")

    print("3. New vault with fresh in-memory storage:")
    print(f"   messages stored: {len(fresh_messages)}")
    print(f"   long-term facts: {fresh_memory.facts}")
    print()
    print("Result: default InMemoryStorage is not durable. Use a custom database-backed")
    print("StorageProvider when you need memory to survive process restarts.")


if __name__ == "__main__":
    asyncio.run(main())
