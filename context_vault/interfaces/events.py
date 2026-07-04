"""Event hook interface."""

from __future__ import annotations

from typing import Any, Protocol


class EventHook(Protocol):
    """Callable lifecycle hook."""

    async def __call__(self, event_name: str, payload: dict[str, Any]) -> None:
        """Handle a lifecycle event."""
