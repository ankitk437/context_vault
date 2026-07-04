"""Custom exceptions for ContextVault."""


class ContextVaultError(Exception):
    """Base exception for ContextVault errors."""


class ConfigurationError(ContextVaultError):
    """Raised when ContextVault is configured incorrectly."""


class ProviderError(ContextVaultError):
    """Raised when a provider fails."""


class TokenBudgetExceededError(ContextVaultError):
    """Raised when context cannot fit within the configured token budget."""
