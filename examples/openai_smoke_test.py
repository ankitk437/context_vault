"""Live OpenAI smoke test.

Set OPENAI_API_KEY in your shell before running this script. Do not commit API keys.
Optionally set OPENAI_MODEL; the default is intentionally easy to override.
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

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    vault = ContextVault(
        llm_provider=OpenAIProvider(model=model, api_key=api_key),
        config=VaultConfig(memory_update_frequency=1, max_context_tokens=8_000),
    )

    response = await vault.chat(
        session_id="openai-smoke-test",
        user_id="local-test-user",
        message="Reply with exactly: ContextVault OpenAI smoke test passed",
        temperature=0,
    )
    print(response.content)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
