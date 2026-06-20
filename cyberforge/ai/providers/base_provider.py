# cyberforge/ai/providers/base_provider.py
"""
Abstract Base Class (Interface) for AI model providers.

Establishes a strict, enterprise‑grade contract for all Large Language Model (LLM)
integrations. This interface is designed to be:

- Provider‑agnostic (no provider‑specific logic)
- Async‑native (all I/O operations are asynchronous)
- Type‑safe (using Pydantic‑compatible types)
- Securely decoupled from API key management (keys are injected at runtime)
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeAlias

# Forward references for docstrings only; avoids linter warnings while
# keeping the contract explicit.
if TYPE_CHECKING:
    from cyberforge.core.exceptions import ProviderError, RateLimitError

# Type alias for the schema definition, improving readability and
# enabling future evolution (e.g., adding validation, versioning).
Schema: TypeAlias = Mapping[str, Any]

# Public API contract
__all__ = ["BaseProvider", "Schema"]


class BaseProvider(ABC):
    """
    The universal contract for AI text generation and parsing services.

    This abstract base class defines the interface that all concrete provider
    implementations (Gemini, OpenAI, Claude, Ollama, etc.) must adhere to.

    Key design decisions:
        - API keys are not stored in the provider; they are obtained from
          the ``KeyRotationManager`` at runtime, enabling seamless rotation.
        - All methods are asynchronous to support high‑throughput pipelines.
        - The provider exposes its identity via the ``provider_name`` property
          for metrics and logging purposes.
        - Schemas are passed as structured ``Mapping`` objects, not raw JSON strings,
          to ensure type safety and avoid redundant parsing.
        - Health checks raise explicit exceptions rather than returning booleans,
          providing clear failure reasons.

    The returned dictionary from ``generate_parsed_response`` must conform to
    the structure defined by the provided ``schema``. The caller (typically the
    ``AIParserService``) is responsible for validating this dictionary against
    the corresponding Pydantic model (e.g., ``ParsedWriteup``) before storage.

    Example:
        >>> class GeminiProvider(BaseProvider):
        ...     @property
        ...     def provider_name(self) -> ModelProvider:
        ...         return ModelProvider.GEMINI
        ...
        ...     async def generate_parsed_response(
        ...         self,
        ...         system_prompt: str,
        ...         user_prompt: str,
        ...         schema: Schema,
        ...         **kwargs: Any,
        ...     ) -> dict[str, Any]:
        ...         # Implementation here
        ...         ...
        ...
        ...     async def health_check(self) -> None:
        ...         # Implementation here
        ...         ...

    Future Evolution (v2/v3):
        As CyberForge scales, the contract may evolve to support:
            - Batch Processing: Accepting multiple prompts/schemas in one call.
            - Streaming: Yielding partial responses for real‑time feedback.
            - Tool Calling: Integrating external tools (e.g., search, code execution).
            - Multi‑Agent Workflows: Coordinating multiple providers/models.

        A natural progression would be to introduce a structured response container
        (e.g., ``ProviderResponse``) that encapsulates:
            - data (dict[str, Any]): The extracted content.
            - model_name (str): The actual model used.
            - provider (ModelProvider): The provider that served the request.
            - prompt_tokens (int): Token count for the input.
            - completion_tokens (int): Token count for the output.
            - duration_seconds (float): Execution time.

        This evolution can be introduced without breaking existing providers by
        allowing the return type to be either ``dict[str, Any]`` or the new
        ``ProviderResponse``, or by providing a wrapper method.

        In future versions (v3+), we may also consider replacing ``ABC`` with
        ``typing.Protocol`` to support more flexible dependency injection and
        structural subtyping, if the project moves toward a more decoupled,
        interface‑based architecture. However, for now, ``ABC`` is the correct
        choice because it enforces explicit implementation of all abstract methods.
    """

    # -------------------------------------------------------------------------
    # Provider Identity
    # -------------------------------------------------------------------------

    @property
    @abstractmethod
    def provider_name(self) -> ModelProvider:
        """
        Returns the unique identifier for this provider.

        This property is used for:
            - Telemetry and metrics (e.g., ``PipelineMetrics.provider_used``)
            - Logging and debugging
            - Provider‑specific routing in the factory
        """
        ...

    # -------------------------------------------------------------------------
    # Core Generation Contract
    # -------------------------------------------------------------------------

    @abstractmethod
    async def generate_parsed_response(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Schema,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Transmits the extraction prompt to the AI model and retrieves the parsed response.

        This method is the primary entry point for the AI pipeline. It takes
        the system instructions, the raw text to analyse, and a structured
        schema, and returns the extracted data as a validated dictionary.

        Args:
            system_prompt: Core behavioural instructions for the AI (e.g., "You
                are a strictly technical cybersecurity data extractor...").
            user_prompt: The raw, unstructured text content to be analysed.
            schema: A structured mapping representing the expected output schema.
                This should be the Pydantic model's ``model_json_schema()`` or
                a compatible dict. The returned dictionary must conform to this
                structure.
            **kwargs: Extensible parameters for provider‑specific tuning.
                Common examples include:
                    - temperature (float, 0.0–1.0)
                    - max_tokens (int)
                    - top_p (float)
                    - seed (int)

        Returns:
            A dictionary containing the extracted data, conforming to the
            structure defined by ``schema``.

        Raises:
            ProviderError: For any provider‑side failure (network, auth, malformed
                response, etc.).
            RateLimitError: If the provider rejects the request due to quota or
                rate‑limit constraints. This triggers the retry/rotation logic
                in the caller layer.

        Note:
            - The caller is responsible for passing a valid API key via the
              ``KeyRotationManager`` before invoking this method. The provider
              itself does not store or manage keys.
            - The returned dictionary will be validated downstream by the
              appropriate Pydantic model (e.g., ``ParsedWriteup``) before
              persistence or further processing.
            - For future evolution (v2+), see the class-level "Future Evolution"
              section which discusses structured response containers like
              ``ProviderResponse``.
        """
        ...

    # -------------------------------------------------------------------------
    # Health & Connectivity Contract
    # -------------------------------------------------------------------------

    @abstractmethod
    async def health_check(self) -> None:
        """
        Validates the provider's authentication credentials and network reachability.

        This method is used for:
            - Startup validation (ensuring the provider is operational before
              accepting traffic)
            - Health probes in production environments
            - Diagnostic debugging when providers fail

        Raises:
            ProviderError: If the provider is unreachable, credentials are invalid,
                or any other health‑related failure occurs.

        Note:
            This method should be lightweight and quick to execute. It should
            not perform expensive operations (e.g., full text generation).

        Future Evolution (v2+):
            For richer observability, consider returning a ``HealthStatus``
            object (see class-level "Future Evolution" section for details).
        """
        ...
