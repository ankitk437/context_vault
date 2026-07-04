"""Default conversation compressor."""

from __future__ import annotations

from context_vault.interfaces import ImportanceScorer, MemoryCompressor, TokenCounter
from context_vault.models import ChatMessage, MemorySummary
from context_vault.utils import RoughTokenCounter, trim_text_to_token_budget


class DefaultMemoryCompressor(MemoryCompressor):
    """Compresses older conversation into a compact recursive summary."""

    def __init__(
        self,
        token_counter: TokenCounter | None = None,
        importance_scorer: ImportanceScorer | None = None,
        max_summary_tokens: int = 1_000,
    ) -> None:
        self.token_counter = token_counter or RoughTokenCounter()
        self.importance_scorer = importance_scorer
        self.max_summary_tokens = max_summary_tokens

    async def compress(
        self,
        messages: list[ChatMessage],
        existing_summary: str | None = None,
    ) -> MemorySummary:
        """Compress messages into a summary."""

        if not messages and not existing_summary:
            return MemorySummary(content="", source_message_ids=[])

        scored: list[tuple[float, ChatMessage]] = []
        for message in messages:
            score = (
                await self.importance_scorer.score(message)
                if self.importance_scorer is not None
                else _fallback_importance(message)
            )
            scored.append((score, message))
        high_signal = [message for score, message in scored if score >= 0.5]
        if not high_signal:
            high_signal = [message for _, message in scored[-8:]]

        lines: list[str] = []
        level = 1
        if existing_summary:
            lines.append("Previous summary:")
            lines.append(existing_summary)
            level = 2
        if high_signal:
            lines.append("Important older conversation:")
            for message in high_signal:
                lines.append(f"- {message.role}: {message.content}")

        content = trim_text_to_token_budget(
            "\n".join(lines).strip(), self.max_summary_tokens, self.token_counter
        )
        return MemorySummary(
            content=content,
            source_message_ids=[message.id for message in messages],
            level=level,
            token_count=self.token_counter.count_text(content),
        )


def _fallback_importance(message: ChatMessage) -> float:
    if message.importance is not None:
        return message.importance
    if message.role == "user":
        return 0.45
    return 0.25
