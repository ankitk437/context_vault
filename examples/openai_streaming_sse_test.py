"""Live OpenAI streaming example with SSE-style events.

Set OPENAI_API_KEY in your shell before running this script. Do not commit API keys.

Run:
    python examples/openai_streaming_sse_test.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from context_vault import ContextVault, VaultConfig
from context_vault.providers import OpenAIProvider


def format_sse(event_type: str, payload: dict[str, object]) -> str:
    """Format one Server-Sent Event frame."""

    data = json.dumps(payload, ensure_ascii=True)
    return f"event: {event_type}\ndata: {data}\n\n"


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

    model = os.getenv("OPENAI_STREAM_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    vault = ContextVault(
        llm_provider=OpenAIProvider(model=model, api_key=api_key),
        config=VaultConfig(memory_update_frequency=1),
    )

    async for event in vault.chat_stream(
        session_id="openai-streaming-sse",
        user_id="streaming-test-user",
        message="Explain quicksort in three short bullets.",
        temperature=0,
    ):
        payload: dict[str, object] = {"type": event.type}
        if event.delta:
            payload["delta"] = event.delta
        if event.response is not None:
            payload["response"] = {
                "content": event.response.content,
                "model": event.response.model,
                "metadata": event.response.metadata,
            }
        print(format_sse(event.type, payload), end="", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
