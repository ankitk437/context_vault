"""Optional OpenAI provider adapter."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from context_vault.exceptions import ProviderError
from context_vault.interfaces import LLMProvider
from context_vault.models import ChatMessage, LLMResponse, LLMStreamEvent


class OpenAIProvider(LLMProvider):
    """OpenAI chat-completions adapter.

    The `openai` dependency is optional. Install with `llm-context-vault[openai]`.
    """

    def __init__(self, model: str, api_key: str | None = None, **client_kwargs: Any) -> None:
        self.model = model
        self.api_key = api_key
        self.client_kwargs = client_kwargs
        self._client: Any | None = None

    def _client_instance(self) -> Any:
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:
                raise ProviderError(
                    "OpenAIProvider requires the optional `openai` package. "
                    "Install it with `python3 -m pip install openai`, "
                    "`python3 -m pip install -r requirements.txt`, or "
                    "`python3 -m pip install llm-context-vault[openai]`."
                ) from exc
            kwargs = dict(self.client_kwargs)
            if self.api_key is not None:
                kwargs["api_key"] = self.api_key
            self._client = AsyncOpenAI(**kwargs)
        return self._client

    async def generate(self, messages: list[ChatMessage], **kwargs: Any) -> LLMResponse:
        """Generate a response using OpenAI chat completions."""

        client = self._client_instance()
        response = await client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": message.role, "content": message.content}
                for message in messages
                if message.role in {"system", "user", "assistant"}
            ],
            **kwargs,
        )
        content = response.choices[0].message.content or ""
        usage = {}
        if response.usage is not None:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        return LLMResponse(content=content, model=self.model, raw=response, usage=usage)

    async def stream_generate(
        self,
        messages: list[ChatMessage],
        **kwargs: Any,
    ) -> AsyncIterator[LLMStreamEvent]:
        """Stream a response using OpenAI chat-completions streaming."""

        client = self._client_instance()
        kwargs.pop("stream", None)
        stream = await client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": message.role, "content": message.content}
                for message in messages
                if message.role in {"system", "user", "assistant"}
            ],
            stream=True,
            **kwargs,
        )
        parts: list[str] = []
        finish_reason: str | None = None
        async for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.finish_reason:
                finish_reason = choice.finish_reason
            delta = choice.delta.content or ""
            if not delta:
                continue
            parts.append(delta)
            yield LLMStreamEvent(
                type="response.output_text.delta",
                delta=delta,
                model=self.model,
            )

        response = LLMResponse(
            content="".join(parts),
            model=self.model,
            metadata={"finish_reason": finish_reason} if finish_reason else {},
        )
        yield LLMStreamEvent(type="response.completed", response=response, model=self.model)
