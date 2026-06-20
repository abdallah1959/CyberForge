# cyberforge/extractors/extractor_factory.py
"""
Extractor Factory.

Implements the Factory Design Pattern to dynamically route URLs to their
corresponding concrete Extractor implementations based on the domain name.
Ensures the Orchestrator remains fully decoupled from specific platforms.

This factory is thread-safe, type-safe, and supports dynamic registration
and unregistration of extractors, following the same pattern as ProviderFactory.
"""

import logging
from threading import RLock
from typing import final
from urllib.parse import urlparse

from cyberforge.core.exceptions import ConfigurationError, ExtractionError
from cyberforge.extractors.base_extractor import BaseExtractor
from cyberforge.extractors.ctftime.ctftime_extractor import CTFTimeExtractor

logger = logging.getLogger(__name__)

# Public API contract
__all__ = ["ExtractorFactory"]


@final
class ExtractorFactory:
    """
    Registry and factory for dynamically instantiating data extractors.

    Thread-safe and supports:
        - Registration of new extractors.
        - Unregistration of existing extractors.
        - URL-based routing to the appropriate extractor.
        - Domain-based routing.
        - Listing registered extractors.

    Usage:
        >>> extractor = ExtractorFactory.create_from_url("https://ctftime.org/writeup/123")
        >>> extractor = ExtractorFactory.create_from_domain("ctftime.org")
        >>> ExtractorFactory.register("hackerone.com", HackerOneExtractor)
        >>> if ExtractorFactory.is_registered("hackerone.com"): ...
        >>> ExtractorFactory.unregister("hackerone.com")
    """

    # Registry mapping domain to extractor class.
    _registry: dict[str, type[BaseExtractor]] = {
        "ctftime.org": CTFTimeExtractor,
        # Future platforms:
        # "hackerone.com": HackerOneExtractor,
        # "exploit-db.com": ExploitDBExtractor,
        # "bugcrowd.com": BugcrowdExtractor,
    }

    # Core extractors that cannot be unregistered or overridden.
    _core_extractors: frozenset[str] = frozenset({
        "ctftime.org",
    })

    # Reentrant lock for thread-safe registry modifications and reads.
    _lock: RLock = RLock()

    @classmethod
    def register(
        cls,
        domain: str,
        extractor_class: type[BaseExtractor],
        override: bool = False,
    ) -> None:
        """
        Register an extractor implementation for a given domain.

        Args:
            domain: The domain (e.g., "hackerone.com").
            extractor_class: The concrete BaseExtractor subclass.
            override: If True, allow re-registration of an existing extractor.

        Raises:
            TypeError: If extractor_class is not a subclass of BaseExtractor.
            ConfigurationError: If the extractor is already registered and
                override is False, or if attempting to override a core extractor.
        """
        if not issubclass(extractor_class, BaseExtractor):
            raise TypeError(
                f"extractor_class must be a subclass of BaseExtractor, got {extractor_class}"
            )

        domain = domain.lower()
        with cls._lock:
            if not override and domain in cls._registry:
                raise ConfigurationError(
                    f"Extractor for domain '{domain}' is already registered. "
                    "Use override=True to replace it."
                )

            if (
                override
                and domain in cls._core_extractors
                and domain in cls._registry
            ):
                raise ConfigurationError(
                    f"Core extractor for domain '{domain}' cannot be overridden."
                )

            cls._registry[domain] = extractor_class
            logger.info(
                "Registered extractor for domain '%s' -> %s",
                domain,
                extractor_class.__name__,
            )

    @classmethod
    def unregister(cls, domain: str) -> bool:
        """
        Unregister an extractor for a given domain.

        Core extractors (defined in `_core_extractors`) cannot be unregistered.

        Args:
            domain: The domain to unregister.

        Returns:
            True if the extractor was unregistered, False if it wasn't registered.

        Raises:
            ConfigurationError: If attempting to unregister a core extractor.
        """
        domain = domain.lower()
        with cls._lock:
            if domain in cls._core_extractors:
                raise ConfigurationError(
                    f"Core extractor for domain '{domain}' cannot be unregistered."
                )

            if domain not in cls._registry:
                logger.warning(
                    "Extractor for domain '%s' is not registered, cannot unregister.",
                    domain,
                )
                return False

            del cls._registry[domain]
            logger.info("Unregistered extractor for domain '%s'", domain)
            return True

    @classmethod
    def is_registered(cls, domain: str) -> bool:
        """
        Check if an extractor is registered for a given domain.

        Args:
            domain: The domain to check.

        Returns:
            True if the extractor is registered, False otherwise.
        """
        domain = domain.lower()
        with cls._lock:
            return domain in cls._registry

    @classmethod
    def list_registered(cls) -> list[str]:
        """
        Return a list of all registered domains.

        Returns:
            List of registered domain strings.
        """
        with cls._lock:
            return list(cls._registry.keys())

    @classmethod
    def create_from_domain(cls, domain: str) -> BaseExtractor:
        """
        Create an extractor instance for a given domain.

        Args:
            domain: The domain (e.g., "ctftime.org").

        Returns:
            A configured extractor instance.

        Raises:
            ConfigurationError: If no extractor is registered for the domain.
        """
        domain = domain.lower()
        with cls._lock:
            extractor_class = cls._registry.get(domain)

            if extractor_class is None:
                available = ", ".join(cls._registry.keys())
                raise ConfigurationError(
                    f"Extractor for domain '{domain}' is not registered. "
                    f"Available domains: {available}"
                )

        logger.info("Creating extractor for domain '%s'", domain)
        return extractor_class()

    @classmethod
    def create_from_url(cls, url: str) -> BaseExtractor:
        """
        Parse a URL and return an initialized instance of the correct Extractor.

        Args:
            url: The full target URL.

        Returns:
            An instantiated object that implements BaseExtractor.

        Raises:
            ExtractionError: If the URL format is invalid or the domain is not supported.
        """
        if not url or not isinstance(url, str):
            raise ExtractionError("Provided URL is empty or invalid.")

        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()

        if not domain:
            raise ExtractionError(f"Invalid URL format: '{url}' (no domain found)")

        # Strip 'www.' if present.
        if domain.startswith("www."):
            domain = domain[4:]

        # Try exact match first.
        with cls._lock:
            extractor_class = cls._registry.get(domain)
            if extractor_class is not None:
                logger.debug(
                    "Routing URL '%s' to extractor for domain '%s'",
                    url,
                    domain,
                )
                return extractor_class()

            # Try suffix matching (e.g., "sub.ctftime.org" -> "ctftime.org").
            for registered_domain, registered_class in cls._registry.items():
                if domain.endswith(f".{registered_domain}"):
                    logger.debug(
                        "Routing URL '%s' to extractor for domain '%s' (suffix match)",
                        url,
                        registered_domain,
                    )
                    return registered_class()

        # Fallback if no extractor supports this domain.
        raise ExtractionError(
            f"Unsupported platform domain '{domain}'. Please register an extractor."
        )
