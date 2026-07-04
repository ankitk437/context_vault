"""Default prompt builder."""

from __future__ import annotations

from context_vault.config import VaultConfig
from context_vault.interfaces import PromptBuilder
from context_vault.models import ChatMessage, ContextBundle


class DefaultPromptBuilder(PromptBuilder):
    """Constructs provider-agnostic chat messages from a context bundle."""

    def __init__(self, config: VaultConfig | None = None) -> None:
        self.config = config or VaultConfig()

    async def build(self, bundle: ContextBundle) -> list[ChatMessage]:
        """Build chat messages in the configured prompt order."""

        messages: list[ChatMessage] = []
        for section in self.config.prompt_order:
            if section == "system":
                self._append_system(messages, bundle.system_prompt)
            elif section == "long_term_memory":
                self._append_long_term_memory(messages, bundle)
            elif section == "conversation_summary":
                self._append_summary(messages, bundle)
            elif section == "recent_messages":
                messages.extend(bundle.recent_messages)
            elif section == "retrieved_documents":
                self._append_documents(messages, bundle)
            elif section == "current_user_message":
                messages.append(bundle.current_user_message)
        return messages

    def _append_system(self, messages: list[ChatMessage], system_prompt: str) -> None:
        if system_prompt.strip():
            messages.append(ChatMessage(role="system", content=system_prompt.strip()))

    def _append_long_term_memory(
        self, messages: list[ChatMessage], bundle: ContextBundle
    ) -> None:
        if bundle.long_term_memory is None:
            return
        rendered = bundle.long_term_memory.render(
            self.config.long_term_include_fields, self.config.long_term_exclude_fields
        )
        if rendered:
            messages.append(ChatMessage(role="system", content=rendered))

    def _append_summary(self, messages: list[ChatMessage], bundle: ContextBundle) -> None:
        if bundle.conversation_summary is not None and bundle.conversation_summary.content:
            messages.append(
                ChatMessage(
                    role="system",
                    content=f"Compressed conversation summary:\n{bundle.conversation_summary.content}",
                )
            )

    def _append_documents(self, messages: list[ChatMessage], bundle: ContextBundle) -> None:
        if not bundle.retrieved_documents:
            return
        lines = ["Retrieved context:"]
        for index, result in enumerate(bundle.retrieved_documents, start=1):
            source = result.document.metadata.get("source", result.document.id)
            lines.append(f"[{index}] {source} (score={result.score:.3f})")
            lines.append(result.document.content)
        messages.append(ChatMessage(role="system", content="\n".join(lines)))
