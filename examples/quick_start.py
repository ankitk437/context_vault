"""Minimal ContextVault quick start."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from context_vault import ContextVault, VaultConfig
from context_vault.providers import EchoLLMProvider


async def main() -> None:
    vault = ContextVault(
        llm_provider=EchoLLMProvider(),
        config=VaultConfig(memory_update_frequency=1),
    )

    response = await vault.chat(
        session_id="demo-session",
        user_id="user-1",
        message="My name is Ankit and I prefer Python for backend work.",
    )
    print(response.content)


if __name__ == "__main__":
    asyncio.run(main())
