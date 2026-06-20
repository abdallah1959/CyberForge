# cyberforge/ai/ai_parser_service.py
"""
Core orchestration service for AI parsing.

Integrates the retry engine, key rotation, provider factory, prompt builder,
and strict Pydantic validation into a single async pipeline.

Flow:
    1. Build system prompt via PromptBuilder.
    2. Get provider via ProviderFactory.
    3. Execute with retry logic + key rotation.
    4. Validate response via Pydantic.
    5. Record telemetry.
    6. Return ParsedWriteup.

DEPENDENCIES:
    Required files (must exist):
        - cyberforge/ai/models.py          (ProviderContext, GenerationOptions, ProviderResponse)
        - cyberforge/ai/providers/provider_factory.py  (ProviderFactory)
        - cyberforge/ai/prompt_builder.py  (PromptBuilder, PromptBuildOptions)
        - cyberforge/core/exceptions.py    (All custom exceptions)
        - cyberforge/schemas/parsed_writeup.py

    Note on PromptBuildOptions:
        The `reasoning_mode` field must be defined as `bool | None` in
        cyberforge/ai/prompt_builder.py to support auto-detection.

    Note on Orchestrator Compatibility:
        This service is async. The orchestrator must use:
        `parsed_writeup = await parser_service.parse_text(...)`
"""

import asyncio
import logging
import random
from typing import Any

from pydantic import ValidationError

from cyberforge.ai.key_rotation_manager import KeyRotationManager
from cyberforge.ai.models import GenerationOptions, ProviderContext, ProviderResponse
from cyberforge.ai.prompt_builder import PromptBuilder, PromptBuildOptions
from cyberforge.ai.providers.provider_factory import ProviderFactory
from cyberforge.config import Settings, settings as global_settings
from cyberforge.core.enums import ModelProvider
from cyberforge.core.exceptions import (
    AuthenticationError,
    ConfigurationError,
    ModelSwitchRequestSignal,
    ParsingFailureError,
    ProviderConnectionError,
    ProviderError,
    RateLimitError,
    RetryableError,
)
from cyberforge.schemas.parsed_writeup import ParsedWriteup

logger = logging.getLogger(__name__)

# Public API
__all__ = ["AIParserService"]


class AIParserService:
    """
    Orchestrator for transforming raw text into structured JSON datasets using AI.

    Responsibilities:
        - Prompt building (via PromptBuilder)
        - Provider selection (via ProviderFactory)
        - Key rotation (via KeyRotationManager)
        - Retry execution (with backoff)
        - Model switching (via ModelSwitchRequestSignal)
        - Response validation (via Pydantic)
        - Telemetry recording

    Future Improvements:
        - Extract retry logic into a separate RetryEngine class (V2).
        - Consider making AuthenticationError non-retryable after a few attempts.
        - Add support for configurable timeouts and generation parameters.
    """

    def __init__(
        self,
        provider: ModelProvider | None = None,
        model_name: str | None = None,
        settings: Settings | None = None,
    ) -> None:
        """
        Initialises the AI Parser Service.

        Args:
            provider: Optional provider override. If None, uses settings.AI_PROVIDER.
            model_name: Optional model override. If None, uses settings.DEFAULT_MODEL.
            settings: Optional settings override. If None, uses global settings.
        """
        self._settings = settings or global_settings

        self.provider_type = provider or self._settings.AI_PROVIDER
        self.model_name = model_name or self._settings.DEFAULT_MODEL

        # Initialise the provider (stateless, no API key yet).
        self.provider = ProviderFactory.create(
            provider=self.provider_type,
            model_name=self.model_name,
        )

        # Initialise key rotation manager from settings.
        self.key_manager = self._create_key_manager()

        logger.info(
            "AIParserService initialised: provider=%s, model=%s",
            self.provider_type.value,
            self.model_name,
        )

    def _create_key_manager(self) -> KeyRotationManager:
        """Create a KeyRotationManager from the current settings."""
        keys: tuple[str, ...] = ()

        if self.provider_type == ModelProvider.GEMINI:
            keys = self._settings.GEMINI_API_KEYS
        elif self.provider_type == ModelProvider.OPENAI:
            keys = self._settings.OPENAI_API_KEYS
        elif self.provider_type == ModelProvider.CLAUDE:
            keys = self._settings.CLAUDE_API_KEYS
        elif self.provider_type == ModelProvider.OLLAMA:
            # Ollama doesn't need keys, but we need a dummy for the manager.
            keys = ("no-key-required-for-ollama",)

        if not keys:
            raise ConfigurationError(
                f"No API keys configured for provider: {self.provider_type.value}. "
                f"Please set the appropriate API_KEYS in your environment."
            )

        return KeyRotationManager(keys)

    async def parse_text(self, raw_text: str) -> ParsedWriteup:
        """
        Parses raw cybersecurity text into a validated structured schema.

        Args:
            raw_text: The unstructured text from a vulnerability report.

        Returns:
            A strictly validated ParsedWriteup object.

        Raises:
            ValueError: If raw_text is empty.
            ParsingFailureError: If the AI output fails schema validation.
            ProviderError: If the provider fails unrecoverably.
            RateLimitError: If all keys are exhausted.
            ModelSwitchRequestSignal: If a model switch is requested (propagated).
        """
        if not raw_text or not raw_text.strip():
            raise ValueError("Provided raw text is empty.")

        # Truncate text if it exceeds maximum context length.
        safe_text = raw_text[: self._settings.MAX_TEXT_LENGTH]

        # Build system prompt.
        # NOTE: PromptBuildOptions.reasoning_mode=None is supported if the field
        # is defined as bool | None. Ensure prompt_builder.py has this type.
        prompt_options = PromptBuildOptions(
            strict_json=True,
            extraction_mode=True,
            reasoning_mode=None,  # Auto-detect based on model tier.
        )
        system_prompt = PromptBuilder.build_system_prompt(
            model_name=self.model_name,
            provider=self.provider_type,
            options=prompt_options,
        )

        # Build the schema (as a dict, not JSON string).
        schema = ParsedWriteup.model_json_schema()

        # Generation options.
        gen_options = GenerationOptions(
            temperature=0.1,
            max_tokens=4096,
        )

        # Execute with retry and key rotation.
        response = await self._execute_with_retry(
            system_prompt=system_prompt,
            user_prompt=safe_text,
            schema=schema,
            gen_options=gen_options,
            max_retries=3,
        )

        # Validate and return.
        return self._validate_response(response)

    async def _execute_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
        gen_options: GenerationOptions,
        max_retries: int = 3,
    ) -> ProviderResponse:
        """
        Execute a provider call with retry logic and key rotation.

        Implements:
            1. Exponential backoff with jitter.
            2. Key rotation on RateLimitError.
            3. Same-key retry on ProviderConnectionError.
            4. Model switching via ModelSwitchRequestSignal.
        """
        attempt = 0
        last_error: Exception | None = None

        while attempt <= max_retries:
            try:
                # Build context with the current key.
                context = ProviderContext(
                    api_key=self.key_manager.get_current_key(),
                    timeout=30.0,  # Could be from settings.
                )

                # Invoke the provider.
                logger.debug(
                    "Provider call attempt %d/%d: provider=%s, model=%s",
                    attempt + 1,
                    max_retries + 1,
                    self.provider_type.value,
                    self.model_name,
                )

                response = await self.provider.generate_parsed_response(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    schema=schema,
                    context=context,
                    options=gen_options,
                )

                # Success - record telemetry and return.
                self._record_telemetry(response)
                return response

            except RateLimitError as exc:
                # Rate limit: rotate the key and retry.
                logger.warning(
                    "Rate limit hit for key (attempt %d/%d): %s",
                    attempt + 1,
                    max_retries + 1,
                    exc,
                )
                try:
                    self.key_manager.mark_current_exhausted_and_rotate()
                    logger.info("Rotated to next available key.")
                except Exception as rotation_error:
                    logger.critical("Key rotation failed: %s", rotation_error)
                    raise ConfigurationError(
                        "All API keys exhausted. No fallback available."
                    ) from rotation_error

                last_error = exc
                attempt += 1

            except ProviderConnectionError as exc:
                # Network/connection error: retry with the same key.
                logger.warning(
                    "Provider connection error (attempt %d/%d): %s",
                    attempt + 1,
                    max_retries + 1,
                    exc,
                )
                last_error = exc
                attempt += 1
                if attempt <= max_retries:
                    delay = min(2.0 * (2 ** attempt), 30.0)
                    jittered = delay * random.uniform(0.5, 1.0)
                    await asyncio.sleep(jittered)

            except ModelSwitchRequestSignal as exc:
                # Model switch requested - update provider, keys, and retry.
                logger.info(
                    "Model switch requested: %s -> %s (provider: %s)",
                    self.model_name,
                    exc.target_model,
                    exc.target_provider.value,
                )

                # Update provider type and model name.
                self.provider_type = exc.target_provider
                self.model_name = exc.target_model

                # Recreate the provider with the new model.
                self.provider = ProviderFactory.create(
                    provider=self.provider_type,
                    model_name=self.model_name,
                )

                # Recreate the key manager for the new provider.
                self.key_manager = self._create_key_manager()

                # Retry with the new provider and keys.
                attempt += 1

            except AuthenticationError as exc:
                # Auth failure: rotate key immediately.
                # Note: In future, consider making this non-retryable after a few attempts.
                logger.error("Authentication error: %s", exc)
                self.key_manager.mark_current_exhausted_and_rotate()
                last_error = exc
                attempt += 1

            except ProviderError as exc:
                # Other provider errors: retry with backoff.
                logger.error("Provider error: %s", exc)
                last_error = exc
                attempt += 1
                if attempt <= max_retries:
                    delay = min(2.0 * (2 ** attempt), 30.0)
                    jittered = delay * random.uniform(0.5, 1.0)
                    await asyncio.sleep(jittered)

            except Exception as exc:
                # Unexpected errors: retry but log more aggressively.
                logger.exception("Unexpected error: %s", exc)
                last_error = exc
                attempt += 1
                if attempt <= max_retries:
                    delay = min(2.0 * (2 ** attempt), 30.0)
                    jittered = delay * random.uniform(0.5, 1.0)
                    await asyncio.sleep(jittered)

        # All retries exhausted.
        raise RetryableError(
            message=(
                f"Provider call failed after {max_retries + 1} attempts. "
                f"Last error: {last_error}"
            ),
            error_code="CF-RETRY-EXHAUSTED",
            context={"provider": self.provider_type.value, "model": self.model_name},
        ) from last_error

    def _validate_response(self, response: ProviderResponse) -> ParsedWriteup:
        """
        Validate the provider response against the Pydantic schema.

        Uses the already-parsed `response.data` dict to avoid redundant JSON parsing.
        """
        try:
            logger.debug("Validating provider response against ParsedWriteup schema.")
            validated = ParsedWriteup.model_validate(response.data)
            logger.info(
                "Successfully parsed writeup: %s (tokens=%d, duration=%.2fs)",
                validated.title,
                response.total_tokens,
                response.duration_seconds,
            )
            return validated

        except ValidationError as exc:
            logger.error(
                "Pydantic validation failed: %s",
                exc,
            )
            # Log the raw response for debugging (truncated).
            raw = response.raw_response or ""
            logger.debug("Raw response (truncated): %s...", raw[:500])
            raise ParsingFailureError(
                f"AI response failed schema validation: {exc}",
                context={"provider": response.provider.value, "model": response.model_name},
            ) from exc

    def _record_telemetry(self, response: ProviderResponse) -> None:
        """
        Record telemetry for the successful provider call.

        This could be extended to send metrics to Prometheus, Datadog, etc.
        """
        logger.info(
            "Provider telemetry: provider=%s, model=%s, "
            "prompt_tokens=%d, completion_tokens=%d, total_tokens=%d, duration=%.2fs",
            response.provider.value,
            response.model_name,
            response.prompt_tokens,
            response.completion_tokens,
            response.total_tokens,
            response.duration_seconds,
        )
