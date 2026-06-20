# cyberforge/ai/providers/gemini_provider.py
"""
Google Gemini AI implementation of the BaseProvider contract.

Uses the modern `google-genai` SDK (v1.0+) for native async support,
structured output, and stateless client instantiation.
"""

import asyncio
import copy
import json
import logging
import time
from types import MappingProxyType
from typing import Any, cast

from google import genai
from google.genai import types

from cyberforge.ai.models import (
    GenerationOptions,
    HealthCheckMode,
    ProviderContext,
    ProviderResponse,
)
from cyberforge.ai.providers.base_provider import BaseProvider, Schema
from cyberforge.core.constants import (
    DEFAULT_HEALTHCHECK_TIMEOUT_SECONDS,
    DEFAULT_PROVIDER_TIMEOUT_SECONDS,
)
from cyberforge.core.enums import ModelProvider
from cyberforge.core.exceptions import (
    AuthenticationError,
    ProviderConnectionError,
    ProviderError,
    RateLimitError,
)

logger = logging.getLogger(__name__)

# Public API contract
__all__ = ["GeminiProvider"]

# Expanded keywords for rate-limit detection across different SDK versions.
# Google may use various phrasings: "429", "quota", "resource exhausted", etc.
RATE_LIMIT_KEYWORDS = (
    "429",
    "quota",
    "resource exhausted",
    "resource_exhausted",
    "rate limit",
    "rate_limit",
)


class GeminiProvider(BaseProvider):
    """
    Concrete implementation for Google's Gemini LLM models using the modern SDK.

    This provider adheres to the BaseProvider contract:
        - API keys are not stored; they are passed via ``ProviderContext``.
        - All I/O is asynchronous.
        - Schemas are passed as structured mappings and enforced natively
          via the SDK's ``response_schema`` parameter.
        - Each request creates a new client instance, avoiding global state.
        - Responses include usage metadata (tokens), duration, and raw text.
        - Provider‑specific metadata (finish_reason, model_version, candidate_count)
          is captured in the ``metadata`` field of ``ProviderResponse``.
        - Timeout protection is applied at the coroutine level using
          ``asyncio.wait_for`` in addition to SDK timeout settings.

    Design Decisions:
        - A new client is created for every request to ensure complete
          statelessness and thread‑safety.
        - Prompts are structured using ``types.Content`` with appropriate roles
          to align with Gemini 2.x best practices.
        - Timeout values default to constants from ``core.constants`` but can be
          overridden via ``ProviderContext.timeout``.

    Future Enhancements:
        - The ``metadata`` field may be replaced with a strongly‑typed
          ``ProviderMetadata`` dataclass for better observability integration.
        - A separate ``SchemaValidator`` layer will be introduced to validate
          the incoming schema before it reaches the provider, preventing
          costly API calls with invalid schemas.
    """

    def __init__(self, model_name: str) -> None:
        """
        Initialises the Gemini provider with a model name.

        Args:
            model_name: The Gemini model version to use (e.g., 'gemini-2.0-flash').
        """
        self._model_name = model_name

    @property
    def provider_name(self) -> ModelProvider:
        """Returns the provider identifier."""
        return ModelProvider.GEMINI

    @staticmethod
    def _extract_status_code(exc: Exception) -> int | None:
        """
        Extracts HTTP status code from a Google API exception.

        The SDK may change the attribute name or location of the status code
        between versions. This helper centralises the extraction logic.

        Args:
            exc: The exception to inspect.

        Returns:
            The status code as an integer, or None if not available.
        """
        for attr in ("status_code", "code", "status", "http_status"):
            if hasattr(exc, attr):
                value = getattr(exc, attr)
                if isinstance(value, int):
                    return value
                # Some SDK versions may return the status code as a string.
                if isinstance(value, str) and value.isdigit():
                    return int(value)
        return None

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        """
        Determines if an exception is a rate-limit (quota) error.

        Checks both the HTTP status code and the exception message string
        using a set of keywords to handle variations between SDK versions.

        Args:
            exc: The exception to inspect.

        Returns:
            True if the exception indicates a rate-limit error, False otherwise.
        """
        status_code = GeminiProvider._extract_status_code(exc)
        if status_code == 429:
            return True

        exc_str = str(exc).lower()
        return any(keyword in exc_str for keyword in RATE_LIMIT_KEYWORDS)

    @staticmethod
    def _clean_json_schema(schema_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Recursively removes keys from a JSON schema that the Gemini API 
        does not support (e.g., 'examples', 'title', 'default', 'uniqueItems').
        """
        cleaned = copy.deepcopy(schema_dict)

        def _clean(d: Any) -> None:
            if isinstance(d, dict):
                # Keys strictly forbidden by Gemini's structured output parser
                for key in ["examples", "title", "default", "uniqueItems"]:
                    d.pop(key, None)
                for value in d.values():
                    _clean(value)
            elif isinstance(d, list):
                for item in d:
                    _clean(item)

        _clean(cleaned)
        return cleaned

    def _create_client(self, context: ProviderContext) -> genai.Client:
        """
        Creates a new stateless Gemini client for the given context.

        Args:
            context: Authentication context containing the API key, base URL, and timeout.

        Returns:
            A configured `genai.Client` instance.
        """
        http_options = None
        if context.base_url or context.timeout is not None:
            http_options = types.HttpOptions(
                base_url=context.base_url,
                timeout=context.timeout,
            )
        return genai.Client(
            api_key=context.api_key,
            http_options=http_options,
        )

    async def generate_parsed_response(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Schema,
        context: ProviderContext,
        options: GenerationOptions,
    ) -> ProviderResponse:
        """
        Generates a structured JSON response using the Gemini API.

        Args:
            system_prompt: Core behavioural instructions.
            user_prompt: The raw text to analyse.
            schema: The expected output schema. Must be compatible with
                Pydantic's ``model_json_schema()`` format.
            context: Authentication and connection context.
            options: Generation parameters (temperature, max_tokens, etc.).

        Returns:
            A ``ProviderResponse`` containing the parsed data and metrics.
        """
        client = self._create_client(context)

        # Build prompt using structured Content objects (Gemini 2.x+ style).
        combined_prompt = f"{system_prompt}\n\nTarget Text:\n{user_prompt}"
        contents = [
            types.Content(
                role="user",
                parts=[types.Part(text=combined_prompt)],
            )
        ]

        start_time = time.monotonic()
        raw_response: str | None = None
        metadata: dict[str, Any] = {}

        try:
            logger.debug("Sending request to Gemini model: %s", self._model_name)

            # Clean the schema to prevent 400 Bad Request errors from Gemini
            cleaned_schema = self._clean_json_schema(cast(dict[str, Any], schema))

            config_kwargs = {
                "temperature": options.temperature,
                "response_mime_type": "application/json",
                "response_schema": cleaned_schema,
            }
            if options.max_tokens is not None:
                config_kwargs["max_output_tokens"] = options.max_tokens
            if options.top_p is not None:
                config_kwargs["top_p"] = options.top_p
            if options.seed is not None:
                config_kwargs["seed"] = options.seed

            # Apply timeout protection at the coroutine level.
            timeout = context.timeout or DEFAULT_PROVIDER_TIMEOUT_SECONDS
            try:
                response = await asyncio.wait_for(
                    client.aio.models.generate_content(
                        model=self._model_name,
                        contents=contents,
                        config=types.GenerateContentConfig(**config_kwargs),
                    ),
                    timeout=timeout,
                )
            except asyncio.TimeoutError as exc:
                logger.error("Gemini request timed out after %.2f seconds", timeout)
                raise ProviderConnectionError(
                    f"Gemini request timed out after {timeout}s"
                ) from exc

            duration = time.monotonic() - start_time

            # Capture raw response safely, ensuring it is a string or None.
            raw_text = getattr(response, "text", None)
            raw_response = str(raw_text) if raw_text is not None else None

            # Use raw_response for all subsequent checks to avoid multiple
            # accesses to the SDK response object.
            if not raw_response:
                raise ProviderError("Gemini returned an empty response.")

            # Extract provider metadata.
            if response.candidates:
                candidate = response.candidates[0]
                finish_reason = candidate.finish_reason
                if finish_reason:
                    metadata["finish_reason"] = finish_reason.name
                metadata["candidate_count"] = len(response.candidates)

            # Extract model version if available.
            if hasattr(response, "model_version") and response.model_version:
                metadata["model_version"] = response.model_version

            # Parse JSON response.
            try:
                parsed_data = json.loads(raw_response)
            except json.JSONDecodeError as exc:
                raise ProviderError(
                    f"Gemini response was not valid JSON: {raw_response[:200]}..."
                ) from exc

            # Extract token usage.
            prompt_tokens = completion_tokens = total_tokens = 0
            if response.usage_metadata:
                usage = response.usage_metadata
                prompt_tokens = usage.prompt_token_count or 0
                completion_tokens = usage.candidates_token_count or 0
                total_tokens = usage.total_token_count or 0
                logger.info(
                    "Gemini usage: prompt_tokens=%d, completion_tokens=%d, total_tokens=%d",
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                )

            return ProviderResponse(
                data=parsed_data,
                provider=self.provider_name,
                model_name=self._model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                duration_seconds=duration,
                raw_response=raw_response,
                metadata=MappingProxyType(metadata) if metadata else None,
            )

        except genai.errors.ClientError as exc:
            status_code = self._extract_status_code(exc)

            if status_code == 429 or self._is_rate_limit_error(exc):
                logger.warning("Gemini API quota exhausted.")
                raise RateLimitError(f"Gemini Rate Limit Exceeded: {exc}") from exc

            if status_code in (401, 403):
                logger.error("Gemini authentication/authorisation error: %s", exc)
                raise AuthenticationError(f"Gemini Auth Error: {exc}") from exc

            if status_code and status_code >= 500:
                logger.error("Gemini server error: %s", exc)
                raise ProviderConnectionError(f"Gemini Server Error: {exc}") from exc

            logger.error("Gemini client error: %s", exc)
            raise ProviderConnectionError(f"Gemini API Error: {exc}") from exc

        except (RateLimitError, AuthenticationError, ProviderConnectionError, ProviderError):
            raise

        except (ConnectionError, TimeoutError, OSError) as exc:
            logger.error("Gemini network error: %s", exc)
            raise ProviderConnectionError(f"Gemini Network Error: {exc}") from exc

        except Exception as exc:
            logger.error("Unexpected error during Gemini generation: %s", exc)
            raise ProviderError(f"Unexpected Gemini Error: {exc}") from exc

    async def health_check(
        self,
        context: ProviderContext,
        mode: HealthCheckMode = HealthCheckMode.CONNECTIVITY,
    ) -> None:
        """
        Validates the Gemini API key and network reachability.

        Args:
            context: Authentication context containing the API key and optional base URL.
            mode: One of ``HealthCheckMode.CONNECTIVITY`` (lightweight) or
                ``HealthCheckMode.VALIDATION`` (checks model accessibility).
        """
        if not isinstance(mode, HealthCheckMode):
            raise ValueError(
                f"Invalid health check mode: {mode}. "
                f"Allowed: {', '.join(m.value for m in HealthCheckMode)}"
            )

        client = self._create_client(context)
        timeout = context.timeout or DEFAULT_HEALTHCHECK_TIMEOUT_SECONDS

        try:
            # Apply timeout protection for health checks.
            if mode == HealthCheckMode.VALIDATION:
                await asyncio.wait_for(
                    client.aio.models.get(model=f"models/{self._model_name}"),
                    timeout=timeout,
                )
            else:
                await asyncio.wait_for(
                    client.aio.models.list(page_size=1),
                    timeout=timeout,
                )
        except asyncio.TimeoutError as exc:
            logger.error("Gemini health check timed out after %.2f seconds", timeout)
            raise ProviderConnectionError(
                f"Gemini health check timed out after {timeout}s"
            ) from exc
        except genai.errors.ClientError as exc:
            status_code = self._extract_status_code(exc)
            if status_code in (401, 403):
                logger.error("Gemini health check auth error: %s", exc)
                raise AuthenticationError(f"Gemini health check failed: {exc}") from exc
            logger.error("Gemini health check failed (mode=%s): %s", mode, exc)
            raise ProviderError(f"Gemini health check failed: {exc}") from exc
        except (RateLimitError, AuthenticationError, ProviderConnectionError, ProviderError):
            raise
        except (ConnectionError, TimeoutError, OSError) as exc:
            logger.error("Gemini health check network error: %s", exc)
            raise ProviderConnectionError(
                f"Gemini health check network error: {exc}"
            ) from exc
        except Exception as exc:
            logger.error("Gemini health check unexpected error: %s", exc)
            raise ProviderError(
                f"Gemini health check unexpected error: {exc}"
            ) from exc