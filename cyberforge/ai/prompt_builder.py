# cyberforge/ai/prompt_builder.py
"""
Dynamic prompt engineering module.

Builds tailored system prompts based on model capabilities.
Uses a capability registry to determine optimal prompting strategies,
avoiding brittle model-name checks.

Capabilities include:
    - Tier (SMALL, MEDIUM, LARGE)
    - JSON schema support
    - Streaming support
    - Tool calling support
    - Vision support
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum, auto

from cyberforge.core.constants import DEFAULT_AI_SYSTEM_PROMPT
from cyberforge.core.enums import ModelProvider

logger = logging.getLogger(__name__)

# Public API
__all__ = [
    "ModelTier",
    "ModelCapabilities",
    "ModelCapabilityRegistry",
    "PromptBuildOptions",
    "PromptBuilder",
    "build_system_prompt",
]


# =============================================================================
# Model Tier Classification
# =============================================================================


class ModelTier(Enum):
    """
    Intelligence/capability tiers for AI models.

    SMALL: Fast, cheap, but less accurate (e.g., Gemini Flash, GPT-4o-mini)
    MEDIUM: Balanced (e.g., Gemini Pro, GPT-4o)
    LARGE: Most capable, slower, expensive (e.g., Gemini Ultra, Claude Opus)
    """

    SMALL = auto()
    MEDIUM = auto()
    LARGE = auto()


# =============================================================================
# Model Capabilities (Rich metadata)
# =============================================================================


@dataclass(frozen=True)
class ModelCapabilities:
    """
    Complete capability metadata for an AI model.
    """

    tier: ModelTier
    supports_json_schema: bool = True
    supports_streaming: bool = True
    supports_tools: bool = False
    supports_vision: bool = False


# =============================================================================
# Model Capability Registry
# =============================================================================


class ModelCapabilityRegistry:
    """
    Registry mapping model names to their capabilities.

    Supports exact matches and regex‑based matching (e.g., r"^gemini-.*-flash.*")
    to handle new model versions gracefully.
    """

    # Default capabilities for unknown models.
    DEFAULT_CAPABILITIES: ModelCapabilities = ModelCapabilities(
        tier=ModelTier.MEDIUM,
        supports_json_schema=True,
        supports_streaming=True,
        supports_tools=False,
        supports_vision=False,
    )

    # Provider-level fallback capabilities.
    _provider_fallbacks: dict[ModelProvider, ModelCapabilities] = {
        ModelProvider.GEMINI: ModelCapabilities(
            tier=ModelTier.MEDIUM,
            supports_json_schema=True,
            supports_streaming=True,
            supports_tools=True,
            supports_vision=True,
        ),
        ModelProvider.OPENAI: ModelCapabilities(
            tier=ModelTier.MEDIUM,
            supports_json_schema=True,
            supports_streaming=True,
            supports_tools=True,
            supports_vision=True,
        ),
        ModelProvider.CLAUDE: ModelCapabilities(
            tier=ModelTier.MEDIUM,
            supports_json_schema=True,
            supports_streaming=True,
            supports_tools=True,
            supports_vision=True,
        ),
        ModelProvider.OLLAMA: ModelCapabilities(
            tier=ModelTier.SMALL,
            supports_json_schema=True,
            supports_streaming=True,
            supports_tools=False,
            supports_vision=False,
        ),
    }

    # Exact model -> capabilities mapping (case‑sensitive key).
    _exact_mappings: dict[str, ModelCapabilities] = {
        # Gemini family
        "gemini-2.0-flash": ModelCapabilities(
            tier=ModelTier.SMALL,
            supports_json_schema=True,
            supports_streaming=True,
            supports_tools=True,
            supports_vision=True,
        ),
        "gemini-2.0-flash-lite": ModelCapabilities(
            tier=ModelTier.SMALL,
            supports_json_schema=True,
            supports_streaming=True,
            supports_tools=False,
            supports_vision=False,
        ),
        "gemini-1.5-flash": ModelCapabilities(
            tier=ModelTier.SMALL,
            supports_json_schema=True,
            supports_streaming=True,
            supports_tools=True,
            supports_vision=True,
        ),
        "gemini-1.5-pro": ModelCapabilities(
            tier=ModelTier.LARGE,
            supports_json_schema=True,
            supports_streaming=True,
            supports_tools=True,
            supports_vision=True,
        ),
        "gemini-2.5-pro": ModelCapabilities(
            tier=ModelTier.LARGE,
            supports_json_schema=True,
            supports_streaming=True,
            supports_tools=True,
            supports_vision=True,
        ),
        "gemini-2.5-flash": ModelCapabilities(
            tier=ModelTier.SMALL,
            supports_json_schema=True,
            supports_streaming=True,
            supports_tools=True,
            supports_vision=True,
        ),
        # OpenAI family
        "gpt-4o-mini": ModelCapabilities(
            tier=ModelTier.SMALL,
            supports_json_schema=True,
            supports_streaming=True,
            supports_tools=True,
            supports_vision=True,
        ),
        "gpt-4o": ModelCapabilities(
            tier=ModelTier.MEDIUM,
            supports_json_schema=True,
            supports_streaming=True,
            supports_tools=True,
            supports_vision=True,
        ),
        "gpt-5": ModelCapabilities(
            tier=ModelTier.LARGE,
            supports_json_schema=True,
            supports_streaming=True,
            supports_tools=True,
            supports_vision=True,
        ),
        "o1-mini": ModelCapabilities(
            tier=ModelTier.SMALL,
            supports_json_schema=False,
            supports_streaming=False,
            supports_tools=False,
            supports_vision=False,
        ),
        "o1-preview": ModelCapabilities(
            tier=ModelTier.LARGE,
            supports_json_schema=False,
            supports_streaming=False,
            supports_tools=False,
            supports_vision=False,
        ),
        # Claude family
        "claude-3-5-sonnet": ModelCapabilities(
            tier=ModelTier.MEDIUM,
            supports_json_schema=True,
            supports_streaming=True,
            supports_tools=True,
            supports_vision=True,
        ),
        "claude-3-opus": ModelCapabilities(
            tier=ModelTier.LARGE,
            supports_json_schema=True,
            supports_streaming=True,
            supports_tools=True,
            supports_vision=True,
        ),
        "claude-3-7-sonnet": ModelCapabilities(
            tier=ModelTier.MEDIUM,
            supports_json_schema=True,
            supports_streaming=True,
            supports_tools=True,
            supports_vision=True,
        ),
        # Ollama local models
        "llama3": ModelCapabilities(
            tier=ModelTier.MEDIUM,
            supports_json_schema=False,
            supports_streaming=True,
            supports_tools=False,
            supports_vision=False,
        ),
        "llama3.2": ModelCapabilities(
            tier=ModelTier.SMALL,
            supports_json_schema=False,
            supports_streaming=True,
            supports_tools=False,
            supports_vision=False,
        ),
        "llama3.3": ModelCapabilities(
            tier=ModelTier.MEDIUM,
            supports_json_schema=False,
            supports_streaming=True,
            supports_tools=False,
            supports_vision=False,
        ),
        "mistral": ModelCapabilities(
            tier=ModelTier.MEDIUM,
            supports_json_schema=False,
            supports_streaming=True,
            supports_tools=False,
            supports_vision=False,
        ),
        "phi3": ModelCapabilities(
            tier=ModelTier.SMALL,
            supports_json_schema=False,
            supports_streaming=True,
            supports_tools=False,
            supports_vision=False,
        ),
        "qwen2.5": ModelCapabilities(
            tier=ModelTier.MEDIUM,
            supports_json_schema=False,
            supports_streaming=True,
            supports_tools=False,
            supports_vision=False,
        ),
    }

    # Lowercase index for faster case‑insensitive exact matches.
    _lowercase_exact_mappings: dict[str, ModelCapabilities] = {
        k.lower(): v for k, v in _exact_mappings.items()
    }

    # Regex patterns for prefix/partial matching.
    # Each entry is a tuple (compiled_regex, capabilities).
    _regex_mappings: list[tuple[re.Pattern, ModelCapabilities]] = [
        # Gemini Flash models (any version)
        (re.compile(r"^gemini-.*-flash.*$", re.IGNORECASE),
         ModelCapabilities(
             tier=ModelTier.SMALL,
             supports_json_schema=True,
             supports_streaming=True,
             supports_tools=True,
             supports_vision=True,
         )),
        # Gemini Pro models
        (re.compile(r"^gemini-.*-pro.*$", re.IGNORECASE),
         ModelCapabilities(
             tier=ModelTier.LARGE,
             supports_json_schema=True,
             supports_streaming=True,
             supports_tools=True,
             supports_vision=True,
         )),
        # GPT models (fallback)
        (re.compile(r"^gpt-.*$", re.IGNORECASE),
         ModelCapabilities(
             tier=ModelTier.MEDIUM,
             supports_json_schema=True,
             supports_streaming=True,
             supports_tools=True,
             supports_vision=True,
         )),
        # Claude models (fallback)
        (re.compile(r"^claude-.*$", re.IGNORECASE),
         ModelCapabilities(
             tier=ModelTier.MEDIUM,
             supports_json_schema=True,
             supports_streaming=True,
             supports_tools=True,
             supports_vision=True,
         )),
        # Ollama models (fallback)
        (re.compile(r"^(llama|mistral|phi|qwen).*$", re.IGNORECASE),
         ModelCapabilities(
             tier=ModelTier.MEDIUM,
             supports_json_schema=False,
             supports_streaming=True,
             supports_tools=False,
             supports_vision=False,
         )),
    ]

    # Future Enhancement:
    # Consider adding @lru_cache(maxsize=256) to get_capabilities()
    # if the system handles millions of requests and model names are repeated.
    # This optimisation is not needed for the current scale.

    @classmethod
    def get_capabilities(
        cls,
        model_name: str,
        provider: ModelProvider | None = None,
    ) -> ModelCapabilities:
        """
        Return the capabilities for a given model name.

        Resolution order:
            1. Exact match in _exact_mappings.
            2. Case‑insensitive exact match (using _lowercase_exact_mappings).
            3. Regex match in _regex_mappings.
            4. Provider fallback (if provider given).
            5. Default capabilities.
        """
        # 1. Exact match
        if model_name in cls._exact_mappings:
            return cls._exact_mappings[model_name]

        # 2. Case‑insensitive exact match (fast)
        lower_name = model_name.lower()
        if lower_name in cls._lowercase_exact_mappings:
            return cls._lowercase_exact_mappings[lower_name]

        # 3. Regex match
        for pattern, caps in cls._regex_mappings:
            if pattern.match(model_name):
                return caps

        # 4. Provider fallback
        if provider is not None and provider in cls._provider_fallbacks:
            return cls._provider_fallbacks[provider]

        # 5. Default
        logger.debug(
            "Unknown model '%s', defaulting to default capabilities.",
            model_name,
        )
        return cls.DEFAULT_CAPABILITIES

    @classmethod
    def register_model(
        cls,
        model_name: str,
        capabilities: ModelCapabilities,
    ) -> None:
        """
        Register a new model with exact matching.

        Args:
            model_name: The exact model name (case‑sensitive).
            capabilities: The capabilities to assign.
        """
        cls._exact_mappings[model_name] = capabilities
        cls._lowercase_exact_mappings[model_name.lower()] = capabilities
        logger.info("Registered exact model '%s'", model_name)

    @classmethod
    def register_pattern(
        cls,
        regex: str,
        capabilities: ModelCapabilities,
        flags: int = re.IGNORECASE,
    ) -> None:
        """
        Register a model pattern using a regular expression.

        This is useful for matching future model versions that follow a known
        naming convention (e.g., r"^gemini-.*-flash.*").

        Args:
            regex: A string containing a regular expression.
            capabilities: The capabilities to assign to matching models.
            flags: Regex flags (default: re.IGNORECASE).
        """
        pattern = re.compile(regex, flags)
        cls._regex_mappings.insert(0, (pattern, capabilities))
        logger.info("Registered pattern '%s'", regex)

    @classmethod
    def is_small_model(cls, model_name: str, provider: ModelProvider | None = None) -> bool:
        """Convenience method to check if a model is in the SMALL tier."""
        return cls.get_capabilities(model_name, provider).tier == ModelTier.SMALL

    @classmethod
    def is_large_model(cls, model_name: str, provider: ModelProvider | None = None) -> bool:
        """Convenience method to check if a model is in the LARGE tier."""
        return cls.get_capabilities(model_name, provider).tier == ModelTier.LARGE


# =============================================================================
# Prompt Build Options
# =============================================================================


@dataclass(frozen=True)
class PromptBuildOptions:
    """
    Configuration options for building system prompts.

    Fields:
        strict_json: Force JSON response without markdown.
        extraction_mode: Enable specialised extraction instructions.
        reasoning_mode: Explicit override for reasoning instructions.
                         If None, automatically enable for SMALL tier models.
        few_shot_examples: Optional tuple of (input, output) examples.
    """

    strict_json: bool = True
    extraction_mode: bool = True
    reasoning_mode: bool | None = None
    few_shot_examples: tuple[tuple[str, str], ...] | None = None


# =============================================================================
# Prompt Builder
# =============================================================================


class PromptBuilder:
    """
    Stateless utility for constructing AI prompts.

    Uses ModelCapabilityRegistry to determine the appropriate prompting
    strategy based on model capabilities.

    Future Evolution:
        If the number of prompt types grows (extraction, classification,
        summarisation, etc.), consider splitting into separate prompt
        builders or templates. For V1, this single builder is sufficient.
    """

    @staticmethod
    def build_system_prompt(
        model_name: str,
        provider: ModelProvider | None = None,
        options: PromptBuildOptions | None = None,
    ) -> str:
        """
        Build a tailored system prompt.

        Args:
            model_name: The target AI model name.
            provider: The provider (used for fallback capability detection).
            options: Optional configuration for prompt customisation.

        Returns:
            A system prompt string optimised for the model's capabilities.
        """
        if options is None:
            options = PromptBuildOptions()

        caps = ModelCapabilityRegistry.get_capabilities(model_name, provider)

        parts = [DEFAULT_AI_SYSTEM_PROMPT]

        # Add extraction mode instructions if enabled.
        if options.extraction_mode:
            parts.append(
                "EXTRACTION MODE: You are an expert cybersecurity data extractor. "
                "Your task is to extract structured vulnerability data from the provided text. "
                "Focus on identifying payloads, evidence, and vulnerability metadata."
            )

        # Determine if reasoning should be enabled.
        if options.reasoning_mode is None:
            enable_reasoning = caps.tier == ModelTier.SMALL
        else:
            enable_reasoning = options.reasoning_mode

        if enable_reasoning:
            parts.append(
                "EXTRACTION PROCEDURE:\n"
                "1. Identify all evidence artifacts (logs, requests, code snippets).\n"
                "2. Identify all attack payloads (injection strings, exploit code).\n"
                "3. Map the findings to the required JSON schema fields.\n"
                "4. Output only the JSON object, with no extra text or markdown."
            )

        # Add strict JSON instruction if enabled.
        if options.strict_json and caps.supports_json_schema:
            parts.append(
                "OUTPUT FORMAT: Return ONLY a valid JSON object. "
                "Do not include any text, markdown, or explanations outside the JSON."
            )

        # Add few-shot examples if provided.
        if options.few_shot_examples:
            examples = "\n\n".join(
                f"Example:\nInput: {inp}\nOutput: {out}"
                for inp, out in options.few_shot_examples
            )
            parts.append(f"REFERENCE EXAMPLES:\n{examples}")

        return "\n\n".join(parts)


# =============================================================================
# Convenience Function
# =============================================================================


def build_system_prompt(
    model_name: str,
    provider: ModelProvider | None = None,
    options: PromptBuildOptions | None = None,
) -> str:
    """
    Convenience function wrapper around PromptBuilder.

    Args:
        model_name: The target AI model name.
        provider: The provider (used for fallback tier detection).
        options: Optional configuration for prompt customisation.

    Returns:
        A system prompt string optimised for the model's capabilities.
    """
    return PromptBuilder.build_system_prompt(model_name, provider, options)
