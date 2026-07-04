"""Lifecycle event manager."""

from __future__ import annotations

import inspect
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Any

from context_vault.interfaces import EventHook

logger = logging.getLogger(__name__)

SyncOrAsyncHook = EventHook | Callable[[str, dict[str, Any]], None | Awaitable[None]]


class EventManager:
    """Registers and emits lifecycle hooks."""

    def __init__(self, hooks: dict[str, list[SyncOrAsyncHook]] | None = None) -> None:
        self._hooks: dict[str, list[SyncOrAsyncHook]] = defaultdict(list)
        for event_name, event_hooks in (hooks or {}).items():
            for hook in event_hooks:
                self.register(event_name, hook)

    def register(self, event_name: str, hook: SyncOrAsyncHook) -> None:
        """Register a hook for one event name."""

        self._hooks[event_name].append(hook)

    async def emit(self, event_name: str, payload: dict[str, Any]) -> None:
        """Emit an event to registered hooks."""

        for hook in self._hooks.get(event_name, []):
            try:
                result = hook(event_name, payload)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                logger.exception("ContextVault event hook failed", extra={"event": event_name})
