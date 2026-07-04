"""LLM-backed memory compression."""

from __future__ import annotations

from context_vault.interfaces import LLMProvider, MemoryCompressor, TokenCounter
from context_vault.models import ChatMessage, MemorySummary
from context_vault.utils import RoughTokenCounter, trim_text_to_token_budget


class LLMMemoryCompressor(MemoryCompressor):
    """Compresses older conversation turns using a dedicated LLM provider."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        token_counter: TokenCounter | None = None,
        max_summary_tokens: int = 1_000,
    ) -> None:
        self.llm_provider = llm_provider
        self.token_counter = token_counter or RoughTokenCounter()
        self.max_summary_tokens = max_summary_tokens

    async def compress(
        self,
        messages: list[ChatMessage],
        existing_summary: str | None = None,
    ) -> MemorySummary:
        """Compress messages into a summary."""

        if not messages and not existing_summary:
            return MemorySummary(content="", source_message_ids=[])

        transcript = "\n".join(f"{message.role}: {message.content}" for message in messages)
        prompt_parts = [
            "Create a compact conversation summary for future LLM context.",
            "Preserve durable user facts, decisions, preferences, unresolved tasks, and constraints.",
            "Remove filler, greetings, and low-value chatter.",
        ]
        if existing_summary:
            prompt_parts.append(f"Previous summary:\n{existing_summary}")
        prompt_parts.append(f"Older conversation:\n{transcript}")
        prompt_parts.append("Return only the summary text.")
        response = await self.llm_provider.generate(
            [
                ChatMessage(
                    role="system",
                    content="You summarize conversation history for compact LLM context.",
                ),
                ChatMessage(role="user", content="\n\n".join(prompt_parts)),
            ],
            temperature=0,
        )
        content = trim_text_to_token_budget(
            response.content.strip(), self.max_summary_tokens, self.token_counter
        )
        return MemorySummary(
            content=content,
            source_message_ids=[message.id for message in messages],
            level=2 if existing_summary else 1,
            token_count=self.token_counter.count_text(content),
        )
