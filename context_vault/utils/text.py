"""Text manipulation helpers."""

from __future__ import annotations

from context_vault.interfaces import TokenCounter


def trim_text_to_token_budget(text: str, token_budget: int, token_counter: TokenCounter) -> str:
    """Trim text to an approximate token budget."""

    if token_budget <= 0:
        return ""
    if token_counter.count_text(text) <= token_budget:
        return text

    # The default counter estimates one token per four characters. Use binary search so
    # custom token counters also work without assuming their internals.
    low = 0
    high = len(text)
    best = ""
    while low <= high:
        mid = (low + high) // 2
        candidate = text[:mid].rstrip()
        if token_counter.count_text(candidate) <= token_budget:
            best = candidate
            low = mid + 1
        else:
            high = mid - 1
    if best and len(best) < len(text):
        return f"{best.rstrip()}..."
    return best
