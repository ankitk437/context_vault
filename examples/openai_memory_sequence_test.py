"""Live test: verify long-term memory builds across sequential interactions.

Set OPENAI_API_KEY in your shell before running this script. Do not commit API keys.

This test uses OpenAI for:
- the main chat response
- long-term memory extraction

It prints memory after every turn so you can see facts accumulate.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from context_vault import ContextVault, VaultConfig
from context_vault.providers import OpenAIProvider
from context_vault.storage import InMemoryStorage


async def main() -> int:
    if importlib.util.find_spec("openai") is None:
        print("The `openai` package is not installed in this Python environment.")
        print("Run: python3 -m pip install -r requirements.txt")
        print("Or:  python3 -m pip install openai")
        return 2

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY is not set. Export it in your shell, then rerun this script.")
        return 2

    chat_model = os.getenv("OPENAI_CHAT_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    memory_model = os.getenv("OPENAI_MEMORY_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    storage = InMemoryStorage()
    vault = ContextVault(
        llm_provider=OpenAIProvider(model=chat_model, api_key=api_key),
        memory_llm_provider=OpenAIProvider(model=memory_model, api_key=api_key),
        storage=storage,
        config=VaultConfig(
            memory_update_frequency=1,
            max_context_tokens=8_000,
            long_term_include_fields=[
                "name",
                "location",
                "preferences",
                "goals",
                "interests",
                "profession",
            ],
            extraction_prompt=(
                "Extract only stable user facts. Keep values short. "
                "Return JSON using only allowed fields."
            ),
        ),
    )

    session_id = "openai-memory-sequence"
    user_id = "sequence-test-user"
    messages = [
        "My name is Ankit.",
        "I live in Bangalore and I work as a Python backend engineer.",
        "I prefer concise technical answers and I am interested in LLM memory systems.",
        "My current goal is to build ContextVault as an open-source Python package.",
    ]

    print(f"chat model: {chat_model}")
    print(f"memory model: {memory_model}")
    print()

    for index, message in enumerate(messages, start=1):
        response = await vault.chat(
            session_id=session_id,
            user_id=user_id,
            message=message,
            temperature=0,
        )
        memory = await storage.long_term.get_memory(user_id)
        print(f"turn {index} user: {message}")
        print(f"turn {index} assistant: {response.content[:160].strip()}")
        print(f"turn {index} long-term memory: {memory.facts}")
        print()

    final_memory = await storage.long_term.get_memory(user_id)
    expected_fields = {"name", "location", "preferences", "goals", "interests", "profession"}
    found_fields = expected_fields & set(final_memory.facts)
    print(f"final remembered fields: {sorted(found_fields)}")
    print(f"final memory version: {final_memory.version}")
    return 0 if {"name", "location", "preferences", "goals"} <= found_fields else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
