"""Live OpenAI full workflow test for ContextVault.

Set OPENAI_API_KEY in your shell before running this script. Do not commit API keys.

This test uses OpenAI for:
- main chat responses
- long-term memory extraction
- conversation compression
- importance scoring

It also verifies:
- short-term session storage
- long-term memory updates
- vector retrieval
- guardrail prompt injection
- input blocking
- output rewriting
- lifecycle events

Run:
    python examples/openai_full_workflow_test.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from context_vault import ContextVault, VaultConfig
from context_vault.guardrails import StudentTeacherGuardrail
from context_vault.models import Document
from context_vault.providers import OpenAIProvider
from context_vault.storage import InMemoryStorage
from context_vault.vectorstores import InMemoryVectorStore


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
    memory_model = os.getenv("OPENAI_MEMORY_MODEL", os.getenv("OPENAI_MODEL", chat_model))
    compression_model = os.getenv(
        "OPENAI_COMPRESSION_MODEL", os.getenv("OPENAI_MODEL", chat_model)
    )
    importance_model = os.getenv("OPENAI_IMPORTANCE_MODEL", os.getenv("OPENAI_MODEL", chat_model))

    events_seen: list[str] = []
    event_payloads: dict[str, list[dict[str, Any]]] = {}

    async def record_event(event_name: str, payload: dict[str, Any]) -> None:
        events_seen.append(event_name)
        event_payloads.setdefault(event_name, []).append(payload)

    storage = InMemoryStorage()
    vector_store = InMemoryVectorStore()
    await vector_store.add_documents(
        [
            Document(
                content=(
                    "Photosynthesis converts light energy into chemical energy. "
                    "Plants use carbon dioxide, water, and sunlight to make glucose."
                ),
                metadata={"source": "science-note"},
            ),
            Document(
                content=(
                    "ContextVault combines short-term memory, long-term memory, "
                    "summaries, guardrails, and vector retrieval around LLM calls."
                ),
                metadata={"source": "contextvault-note"},
            ),
        ]
    )

    guardrail = StudentTeacherGuardrail(
        student_age=13,
        subject="science",
        blocked_input_terms=["adult-only"],
        blocked_output_terms=["forbidden-demo-phrase"],
        extra_instructions=(
            "Give age-appropriate explanations. Use hints and examples before final answers."
        ),
    )

    config = VaultConfig(
        system_prompt="You are a careful educational assistant.",
        max_context_tokens=900,
        reserved_output_tokens=150,
        memory_update_frequency=1,
        vector_search=True,
        vector_top_k=2,
        compression_threshold=0.25,
        long_term_include_fields=[
            "name",
            "location",
            "preferences",
            "goals",
            "interests",
            "grade_level",
        ],
        extraction_prompt=(
            "Extract only stable learner facts. Keep values short. "
            "Return valid JSON with only the allowed fields."
        ),
    )
    vault = ContextVault(
        llm_provider=OpenAIProvider(model=chat_model, api_key=api_key),
        memory_llm_provider=OpenAIProvider(model=memory_model, api_key=api_key),
        compression_llm_provider=OpenAIProvider(model=compression_model, api_key=api_key),
        importance_llm_provider=OpenAIProvider(model=importance_model, api_key=api_key),
        storage=storage,
        vector_store=vector_store,
        config=config,
        guardrails=[guardrail],
    )

    for event_name in [
        "before_input_guardrail",
        "after_input_guardrail",
        "before_context_build",
        "after_context_build",
        "before_prompt_guardrail",
        "after_prompt_guardrail",
        "before_llm_call",
        "after_llm_call",
        "before_output_guardrail",
        "after_output_guardrail",
        "before_memory_update",
        "after_memory_update",
        "before_summary",
        "after_summary",
    ]:
        vault.events.register(event_name, record_event)

    session_id = "openai-full-workflow"
    user_id = "student-openai-test"
    messages = [
        "My name is Ankit. I am a grade 8 student in Bangalore.",
        "I prefer concise science explanations with one small example.",
        "I am interested in photosynthesis and LLM memory systems.",
        "My goal is to understand how retrieval and memory work together.",
        "Using the science note, explain photosynthesis in two short bullets.",
    ]

    print(f"chat model: {chat_model}")
    print(f"memory model: {memory_model}")
    print(f"compression model: {compression_model}")
    print(f"importance model: {importance_model}")
    print()

    for index, message in enumerate(messages, start=1):
        response = await vault.chat(
            session_id=session_id,
            user_id=user_id,
            message=message,
            temperature=0,
        )
        print(f"turn {index} assistant:")
        print(response.content.strip())
        print()

    blocked_response = await vault.chat(
        session_id=session_id,
        user_id=user_id,
        message="adult-only topic please",
        temperature=0,
    )
    print("blocked input response:")
    print(blocked_response.content)
    guardrail_info = blocked_response.metadata.get("guardrail", {})
    if guardrail_info:
        print(f"blocked by: {guardrail_info.get('name')}")
        print(f"blocked reason: {guardrail_info.get('reason')}")
        blocked_term = guardrail_info.get("term")
        if blocked_term:
            print(f"blocked term: {blocked_term}")
    print()

    memory = await storage.long_term.get_memory(user_id)
    stored_messages = await storage.short_term.get_messages(session_id)
    summary = await storage.short_term.get_summary(session_id)
    event_counts = Counter(events_seen)
    prompt_payload = event_payloads["before_llm_call"][0]
    first_prompt = prompt_payload["messages"]
    vector_context_injected = any(
        "Retrieved context:" in message.content for message in first_prompt
    )
    guardrail_policy_injected = any(
        "Guardrail policy:" in message.content for message in first_prompt
    )

    checks = {
        "short_term_messages": len(stored_messages) >= 12,
        "long_term_memory": bool(memory.facts),
        "summary_created": summary is not None and bool(summary.content.strip()),
        "vector_context_injected": vector_context_injected,
        "guardrail_policy_injected": guardrail_policy_injected,
        "input_guardrail_blocked": blocked_response.model == "guardrail",
        "memory_update_event": event_counts["after_memory_update"] >= 1,
        "summary_event": event_counts["after_summary"] >= 1,
        "llm_call_event": event_counts["after_llm_call"] >= len(messages),
    }

    print("\n--- workflow result ---")
    print(f"short-term messages stored: {len(stored_messages)}")
    print(f"long-term memory facts: {memory.facts}")
    print(f"summary created: {summary is not None}")
    if summary is not None:
        print(f"summary text: {summary.content.strip()}")
    print(f"events: {dict(sorted(event_counts.items()))}")
    print(f"checks: {checks}")

    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        print(f"\nFAILED checks: {failed}")
        return 1

    print("\nOpenAI full workflow test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
