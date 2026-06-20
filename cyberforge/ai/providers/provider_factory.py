# cyberforge/ai/providers/provider_factory.py
"""
Factory pattern implementation for AI providers.

Dynamically instantiates the correct provider based on configuration.
Providers are stateless and receive API keys via ProviderContext at runtime.
"""

from __future__ import annotations

import logging
from threading import RLock
from typing import final

from cyberforge.ai.providers.base_provider import BaseProvider
from cyberforge.ai.providers.gemini_provider import GeminiProvider
from cyberforge.core.enums import ModelProvider
from cyberforge.core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

# Public API contract
__all__ = ["ProviderFactory"]


@final
class ProviderFactory:
    """
    Registry and factory for AI providers.

    Responsible for:
        - Provider registration (thread-safe)
        - Provider instantiation
        - Decoupling orchestrators from provider implementations

    The factory is stateless and can be used as a singleton.

    Example:
        >>> provider = ProviderFactory.create(
        ...     provider=ModelProvider.GEMINI,
        ...     model_name="gemini-2.0-flash"
        ... )
        >>> # Then use provider with ProviderContext at request time.
    """

    # Registry mapping ModelProvider to provider class.
    _registry: dict[ModelProvider, type[BaseProvider]] = {
        ModelProvider.GEMINI: GeminiProvider,
        # ModelProvider.OPENAI: OpenAIProvider,  # Future
        # ModelProvider.CLAUDE: ClaudeProvider,  # Future
        # ModelProvider.OLLAMA: OllamaProvider,  # Future
    }

    # Core providers that cannot be unregistered or overridden.
    _core_providers: frozenset[ModelProvider] = frozenset({
        ModelProvider.GEMINI,
    })

    # Reentrant lock for thread-safe registry modifications and reads.
    _lock: RLock = RLock()

    @classmethod
    def register(
        cls,
        provider: ModelProvider,
        provider_class: type[BaseProvider],
        override: bool = False,
    ) -> None:
        """
        Register a provider implementation dynamically.

        This allows extending the factory without modifying the core code.
        Useful for plugins or custom providers.

        Args:
            provider: The ModelProvider enum member.
            provider_class: The concrete BaseProvider subclass.
            override: If True, allow re-registration of an existing provider.
                If False (default), raise ConfigurationError if already registered.

        Raises:
            TypeError: If provider_class is not a subclass of BaseProvider.
            ConfigurationError: If the provider is already registered and
                override is False, or if attempting to override a core provider.
        """
        if not issubclass(provider_class, BaseProvider):
            raise TypeError(
                f"provider_class must be a subclass of BaseProvider, got {provider_class}"
            )

        with cls._lock:
            # Prevent accidental re-registration without override.
            if not override and provider in cls._registry:
                raise ConfigurationError(
                    f"Provider '{provider.value}' is already registered. "
                    "Use override=True to replace it."
                )

            # Prevent overriding core providers, even with override=True.
            if (
                override
                and provider in cls._core_providers
                and provider in cls._registry
            ):
                raise ConfigurationError(
                    f"Core provider '{provider.value}' cannot be overridden."
                )

            cls._registry[provider] = provider_class
            logger.info(
                "Registered provider '%s' -> %s",
                provider.value,
                provider_class.__name__,
            )

    @classmethod
    def unregister(cls, provider: ModelProvider) -> bool:
        """
        Unregister a provider implementation.

        Core providers (defined in `_core_providers`) cannot be unregistered.

        Args:
            provider: The ModelProvider enum member to unregister.

        Returns:
            True if the provider was unregistered, False if it wasn't registered.

        Raises:
            ConfigurationError: If attempting to unregister a core provider.
        """
        with cls._lock:
            if provider in cls._core_providers:
                raise ConfigurationError(
                    f"Core provider '{provider.value}' cannot be unregistered."
                )

            if provider not in cls._registry:
                logger.warning(
                    "Provider '%s' is not registered, cannot unregister.",
                    provider.value,
                )
                return False

            del cls._registry[provider]
            logger.info("Unregistered provider '%s'", provider.value)
            return True

    @classmethod
    def is_registered(cls, provider: ModelProvider) -> bool:
        """
        Check if a provider is registered.

        Args:
            provider: The ModelProvider enum member.

        Returns:
            True if the provider is registered, False otherwise.
        """
        with cls._lock:
            return provider in cls._registry

    @classmethod
    def create(
        cls,
        provider: ModelProvider,
        model_name: str,
    ) -> BaseProvider:
        """
        Create an instance of the requested provider.

        Args:
            provider: Provider identifier (e.g., ModelProvider.GEMINI).
            model_name: Model version to instantiate (e.g., "gemini-2.0-flash").

        Returns:
            A configured provider instance (stateless, without API key).

        Raises:
            ConfigurationError: If the provider is not registered.
        """
        with cls._lock:
            provider_class = cls._registry.get(provider)

            if provider_class is None:
                available = ", ".join(p.value for p in cls._registry)
                raise ConfigurationError(
                    f"Provider '{provider.value}' is not registered. "
                    f"Available providers: {available}"
                )

        logger.info(
            "Creating provider '%s' with model '%s'",
            provider.value,
            model_name,
        )

        # Note: API key is NOT passed here; it will be provided
        # via ProviderContext at request time.
        return provider_class(model_name=model_name)

    @classmethod
    def list_registered(cls) -> list[ModelProvider]:
        """
        Return a list of all registered provider types.

        Returns:
            List of registered ModelProvider enum members.
        """
        with cls._lock:
            return list(cls._registry.keys())
