"""LLM-backed importance scoring."""

from __future__ import annotations

import re

from context_vault.interfaces import ImportanceScorer, LLMProvider
from context_vault.models import ChatMessage


class LLMImportanceScorer(ImportanceScorer):
    """Scores message importance using a dedicated LLM provider."""

    def __init__(self, llm_provider: LLMProvider) -> None:
        self.llm_provider = llm_provider

    async def score(self, message: ChatMessage) -> float:
        """Return a score from 0 to 1."""

        response = await self.llm_provider.generate(
            [
                ChatMessage(
                    role="system",
                    content=(
                        "Score how important a chat message is for future memory. "
                        "Return only a number from 0 to 1."
                    ),
                ),
                ChatMessage(
                    role="user",
                    content=f"Role: {message.role}\nMessage:\n{message.content}",
                ),
            ],
            temperature=0,
        )
        return _parse_score(response.content)


def _parse_score(text: str) -> float:
    match = re.search(r"0(?:\.\d+)?|1(?:\.0+)?", text)
    if match is None:
        return 0.0
    return max(0.0, min(1.0, float(match.group(0))))
