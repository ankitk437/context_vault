"""Guardrail result models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

GuardrailAction = Literal["allow", "rewrite", "block"]


class GuardrailResult(BaseModel):
    """Decision returned by a guardrail check."""

    action: GuardrailAction = "allow"
    content: str | None = None
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def allow(cls, **metadata: Any) -> GuardrailResult:
        """Allow the current input or output."""

        return cls(action="allow", metadata=metadata)

    @classmethod
    def rewrite(
        cls,
        content: str,
        *,
        reason: str | None = None,
        **metadata: Any,
    ) -> GuardrailResult:
        """Replace the current input or output content."""

        return cls(action="rewrite", content=content, reason=reason, metadata=metadata)

    @classmethod
    def block(
        cls,
        content: str | None = None,
        *,
        reason: str | None = None,
        **metadata: Any,
    ) -> GuardrailResult:
        """Stop the workflow and return a safe response."""

        return cls(action="block", content=content, reason=reason, metadata=metadata)
