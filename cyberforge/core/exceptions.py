# cyberforge/core/exceptions.py
"""
CyberForge exception hierarchy, signals, and context.

This file is the main entry point for all exceptions, signals, retry policies,
and context utilities. It is structured as a single file for ease of use,
but internally organised into logical sections.

REQUIRES:
    Python 3.11+ (for StrEnum and TypedDict features).

IMPORTANT:
    - All OperationalSignal subclasses inherit from BaseException.
      They CANNOT be caught by ``except Exception``.
    - Avoid bare ``except:`` or ``except Exception:`` in orchestrator code.
    - Use specific error codes rather than CF_UNKNOWN.

FUTURE ROADMAP:
    Consider splitting into separate files (error_codes.py, retry.py, signals.py,
    context.py) as the codebase grows beyond this single-file organisation.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum, StrEnum
from hashlib import sha256
from typing import Any, NotRequired, TypedDict

from cyberforge.core.enums import ModelProvider

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

__all__ = [
    # Severity & Codes
    "ErrorSeverity",
    "ErrorCode",
    # Retry
    "JitterStrategy",
    "RetryPolicy",
    "RetryPolicyMetadata",
    # Context
    "ErrorContext",
    "ErrorExtra",
    "ExtraPrimitive",
    "ExtraValue",
    # Signals
    "SignalPriority",
    "CircuitBreakerState",
    "OperationalSignal",
    "CriticalControlSignal",
    "ControlFlowSignal",
    "ModelSwitchRequestSignal",
    "CircuitBreakerSignal",
    "ProviderDegradedSignal",
    # Base Exceptions
    "CyberForgeBaseException",
    "OperationalError",
    "ConfigurationError",
    "ProviderError",
    "RetryableError",
    "AuthenticationError",
    "ProviderConnectionError",
    "RateLimitError",
    "QuotaExhaustedError",
    "ParsingFailureError",
    "DataValidationError",
    "ExtractionError",
    "StorageError",
    "StorageConnectionError",      # New
    "DataPersistenceError",        # New
    # Helpers
    "is_control_signal",
    "is_critical_signal",
]


# =============================================================================
# 1. Error Severity & Error Codes
# =============================================================================


class ErrorSeverity(Enum):
    """Severity levels for exceptions, used for monitoring and alerting."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    FATAL = "FATAL"


class ErrorCode(StrEnum):
    """
    Standardised error codes for CyberForge exceptions.

    Prefer specific codes over CF_UNKNOWN. If you use CF_UNKNOWN frequently,
    consider adding a new code to this enum.
    """

    # Provider-layer errors
    CF_AUTH = "CF-AUTH"
    CF_RATE_LIMIT = "CF-RATE-LIMIT"
    CF_QUOTA = "CF-QUOTA"
    CF_NETWORK = "CF-NETWORK"

    # Configuration errors
    CF_CONFIG = "CF-CONFIG"

    # Data processing errors
    CF_PARSE = "CF-PARSE"
    CF_VALIDATE = "CF-VALIDATE"
    CF_EXTRACT = "CF-EXTRACT"
    CF_STORAGE = "CF-STORAGE"
    CF_STORAGE_CONNECTION = "CF-STORAGE-CONNECTION"
    CF_DATA_PERSISTENCE = "CF-DATA-PERSISTENCE"

    # Control signals (informational)
    CF_SWITCH = "CF-SWITCH"
    CF_CIRCUIT_OPEN = "CF-CIRCUIT-OPEN"
    CF_CIRCUIT_HALF_OPEN = "CF-CIRCUIT-HALF-OPEN"
    CF_DEGRADED = "CF-DEGRADED"

    # Fallback
    CF_UNKNOWN = "CF-UNKNOWN"


# =============================================================================
# 2. Retry Policy
# =============================================================================


class JitterStrategy(Enum):
    FULL = "FULL"
    EQUAL = "EQUAL"
    NONE = "NONE"


@dataclass(frozen=True)
class RetryPolicy:
    """
    Advanced retry policy for retryable errors.

    The retry engine MUST read ``exc.retry_policy`` directly from the exception
    instance, not from ``exc.context["retry_policy"]`` (which is for logging).
    """

    max_retries: int = 3
    base_delay_seconds: float = 2.0
    max_delay_seconds: float = 60.0
    backoff_multiplier: float = 2.0
    jitter_strategy: JitterStrategy = JitterStrategy.FULL
    retry_budget: int | None = None
    per_provider_budget: int | None = None


class RetryPolicyMetadata(TypedDict):
    """TypedDict for storing retry policy in ErrorExtra."""
    max_retries: int
    base_delay_seconds: float
    max_delay_seconds: float
    backoff_multiplier: float
    jitter_strategy: str
    retry_budget: int | None
    per_provider_budget: int | None


# =============================================================================
# 3. Error Context (with type-safe extra)
# =============================================================================

# Primitive types allowed in extra fields, including None for optional values.
ExtraPrimitive = str | int | float | bool | None
ExtraValue = ExtraPrimitive | list[ExtraPrimitive] | Mapping[str, ExtraPrimitive]


class ErrorExtra(dict[str, ExtraValue]):
    """
    A dictionary-like container for extra context fields with type safety.

    Allows storing:
        - Scalar values (str, int, float, bool, None)
        - Lists of scalar values
        - Nested dictionaries of scalar values

    This provides flexibility while maintaining a degree of type safety suitable
    for logging and serialization.
    """

    def __init__(self, data: Mapping[str, ExtraValue] | None = None) -> None:
        super().__init__()
        if data:
            for key, value in data.items():
                self[key] = value

    def __setitem__(self, key: str, value: ExtraValue) -> None:
        self._validate_value(value)
        super().__setitem__(key, value)

    def update(self, *args, **kwargs) -> None:  # type: ignore
        for k, v in dict(*args, **kwargs).items():
            self[k] = v

    def setdefault(self, key: str, default: ExtraValue) -> ExtraValue:  # type: ignore
        if key not in self:
            self[key] = default
        return self[key]

    @staticmethod
    def _validate_value(value: ExtraValue) -> None:
        """Recursively validate that the value is of an allowed type."""
        # Allow None
        if value is None:
            return
        # Allow scalars
        if isinstance(value, (str, int, float, bool)):
            return
        # Allow lists of scalars
        if isinstance(value, list):
            for item in value:
                ErrorExtra._validate_value(item)
            return
        # Allow dictionaries of scalars
        if isinstance(value, dict):
            for v in value.values():
                ErrorExtra._validate_value(v)
            return
        raise TypeError(
            f"Extra value must be None, str, int, float, bool, list of those, "
            f"or dict of those, got {type(value).__name__}"
        )


class ErrorContext(TypedDict, total=False):
    """
    Standardised context fields for exception logging and tracing.

    All fields are optional. Fields are added as needed.
    """

    # Provider context
    provider: NotRequired[ModelProvider]
    model_name: NotRequired[str]

    # Request tracing
    request_id: NotRequired[str]
    correlation_id: NotRequired[str]
    retry_attempt: NotRequired[int]

    # API key management
    api_key_id: NotRequired[str]

    # Signal-specific fields
    target_provider: NotRequired[ModelProvider]
    target_model: NotRequired[str]

    circuit_state: NotRequired[str]
    reason: NotRequired[str]
    failure_threshold: NotRequired[int]
    success_threshold: NotRequired[int]
    cooldown_seconds: NotRequired[float]
    severity_score: NotRequired[float]

    # Extra flexible data
    extra: NotRequired[ErrorExtra]


# =============================================================================
# 4. Metadata & Fingerprint Helpers
# =============================================================================


def _validate_provider_fields(context: ErrorContext) -> None:
    """
    Runtime validation for provider fields in ErrorContext.

    Ensures that 'provider' and 'target_provider' are ModelProvider instances.
    Raises TypeError if validation fails.
    """
    for field in ("provider", "target_provider"):
        value = context.get(field)  # type: ignore[attr-defined]
        if value is not None and not isinstance(value, ModelProvider):
            raise TypeError(
                f"ErrorContext['{field}'] must be a ModelProvider instance, "
                f"got {type(value).__name__}"
            )


def _build_context(
    base_context: ErrorContext | None,
    updates: dict[str, Any],
) -> ErrorContext:
    """
    Safely merge updates into a new context dictionary.

    Validates that provider and target_provider are of the correct type.
    """
    ctx = dict(base_context or {})
    for field in ("provider", "target_provider"):
        if field in updates:
            value = updates[field]
            if value is not None and not isinstance(value, ModelProvider):
                raise TypeError(
                    f"ErrorContext['{field}'] must be a ModelProvider instance, "
                    f"got {type(value).__name__}"
                )
    ctx.update(updates)
    _validate_provider_fields(ctx)  # type: ignore[arg-type]
    return ctx


def _generate_fingerprint(
    exception_type: str,
    error_code: str | None,
    context: ErrorContext | None,
) -> str:
    """
    Generates a deterministic fingerprint for an exception or signal.

    Uses exception type, error code, provider, and model name to produce
    a 96-bit (24 hex) hash for error grouping and deduplication.
    """
    ctx = context or {}
    provider = ctx.get("provider")
    provider_str = provider.value if isinstance(provider, ModelProvider) else ""

    components = [
        exception_type,
        error_code or "",
        provider_str,
        ctx.get("model_name", ""),
    ]
    raw = "|".join(components)
    return sha256(raw.encode("utf-8")).hexdigest()[:24]


def _init_metadata(
    self: Any,
    message: str,
    error_code: ErrorCode | str | None = None,
    context: ErrorContext | None = None,
) -> None:
    """Common initialisation logic for exception metadata."""
    self.error_code = error_code.value if isinstance(error_code, ErrorCode) else error_code
    self.context = context or {}
    self.correlation_id: str | None = self.context.get("correlation_id")
    self.timestamp = datetime.now(timezone.utc)
    self.fingerprint = _generate_fingerprint(
        exception_type=self.__class__.__name__,
        error_code=self.error_code,
        context=self.context,
    )
    self._message = message


# =============================================================================
# 5. Signals (Operational Signals)
# =============================================================================


class SignalPriority(Enum):
    LOW = 1
    HIGH = 2
    CRITICAL = 3


class CircuitBreakerState(Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class OperationalSignal(BaseException):
    """
    Base class for operational signals (control-flow interrupts).

    Inherits from BaseException – CANNOT be caught by except Exception.
    """

    def __init__(
        self,
        message: str,
        priority: SignalPriority = SignalPriority.LOW,
        error_code: ErrorCode | str | None = None,
        context: ErrorContext | None = None,
    ) -> None:
        self.priority = priority
        _init_metadata(self, message, error_code, context)
        super().__init__(message)

    def __str__(self) -> str:
        return self._message

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"error_code='{self.error_code}', "
            f"fingerprint='{self.fingerprint}')"
        )


class CriticalControlSignal(OperationalSignal):
    """Critical signals that must never be swallowed by generic handlers."""

    def __init__(
        self,
        message: str,
        priority: SignalPriority = SignalPriority.CRITICAL,
        error_code: ErrorCode | str | None = None,
        context: ErrorContext | None = None,
    ) -> None:
        super().__init__(message, priority, error_code, context)


class ControlFlowSignal(CriticalControlSignal):
    """Base class for intentional control-flow interrupts."""

    def __init__(
        self,
        message: str,
        error_code: ErrorCode | str | None = None,
        context: ErrorContext | None = None,
    ) -> None:
        super().__init__(message, SignalPriority.CRITICAL, error_code, context)


class ModelSwitchRequestSignal(ControlFlowSignal):
    """Signal requesting a dynamic provider/model switch."""

    def __init__(
        self,
        target_provider: ModelProvider,
        target_model: str,
        context: ErrorContext | None = None,
    ) -> None:
        self.target_provider = target_provider
        self.target_model = target_model
        ctx = _build_context(
            context,
            {
                "target_provider": target_provider,
                "target_model": target_model,
            },
        )
        message = (
            f"Runtime interrupt: Dynamic switch requested to provider "
            f"'{target_provider.value}' using model '{target_model}'."
        )
        super().__init__(message, ErrorCode.CF_SWITCH, ctx)


class CircuitBreakerSignal(CriticalControlSignal):
    """Signal indicating that a circuit breaker has opened or is half-open."""

    def __init__(
        self,
        state: CircuitBreakerState,
        provider: ModelProvider,
        reason: str | None = None,
        failure_threshold: int = 5,
        success_threshold: int = 3,
        cooldown_seconds: float = 60.0,
        context: ErrorContext | None = None,
    ) -> None:
        if state not in (CircuitBreakerState.OPEN, CircuitBreakerState.HALF_OPEN):
            raise ValueError(
                f"CircuitBreakerSignal can only signal OPEN or HALF_OPEN states. "
                f"Received: {state.value}"
            )
        self.state = state
        self.provider = provider
        self.reason = reason
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.cooldown_seconds = cooldown_seconds

        ctx = _build_context(
            context,
            {
                "circuit_state": state.value,
                "provider": provider,
                "reason": reason,
                "failure_threshold": failure_threshold,
                "success_threshold": success_threshold,
                "cooldown_seconds": cooldown_seconds,
            },
        )
        message = (
            f"Circuit breaker '{state.value}' for provider '{provider.value}'"
            f"{f': {reason}' if reason else ''}"
        )
        error_code = (
            ErrorCode.CF_CIRCUIT_OPEN
            if state == CircuitBreakerState.OPEN
            else ErrorCode.CF_CIRCUIT_HALF_OPEN
        )
        super().__init__(message, SignalPriority.CRITICAL, error_code, ctx)


class ProviderDegradedSignal(CriticalControlSignal):
    """Signal indicating a provider is degraded but not failing."""

    def __init__(
        self,
        provider: ModelProvider,
        reason: str | None = None,
        severity_score: float | None = None,
        cooldown_seconds: float | None = None,
        context: ErrorContext | None = None,
    ) -> None:
        if severity_score is not None and not (0.0 <= severity_score <= 1.0):
            raise ValueError(f"severity_score must be between 0.0 and 1.0, got {severity_score}")
        self.provider = provider
        self.reason = reason
        self.severity_score = severity_score
        self.cooldown_seconds = cooldown_seconds

        ctx = _build_context(
            context,
            {
                "provider": provider,
                "reason": reason,
                "severity_score": severity_score,
                "cooldown_seconds": cooldown_seconds,
            },
        )
        message = (
            f"Provider '{provider.value}' is degraded"
            f"{f': {reason}' if reason else ''}"
            f"{f' (score={severity_score:.2f})' if severity_score is not None else ''}"
        )
        super().__init__(message, SignalPriority.CRITICAL, ErrorCode.CF_DEGRADED, ctx)


# =============================================================================
# 6. Base Exceptions
# =============================================================================


class CyberForgeBaseException(Exception):
    """Base exception class for all custom CyberForge errors."""

    def __init__(
        self,
        message: str,
        severity: ErrorSeverity | None = None,
        error_code: ErrorCode | str | None = None,
        context: ErrorContext | None = None,
    ) -> None:
        self.severity = severity
        _init_metadata(self, message, error_code, context)
        super().__init__(message)

    def __str__(self) -> str:
        return self._message

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"error_code='{self.error_code}', "
            f"fingerprint='{self.fingerprint}')"
        )


class OperationalError(CyberForgeBaseException):
    """Base class for operational errors (actual failures)."""


class ConfigurationError(OperationalError):
    """Configuration error (FATAL)."""

    def __init__(self, message: str, context: ErrorContext | None = None) -> None:
        super().__init__(message, ErrorSeverity.FATAL, ErrorCode.CF_CONFIG, context)


class ProviderError(OperationalError):
    """Base class for provider-side errors."""

    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.HIGH,
        error_code: ErrorCode | str | None = ErrorCode.CF_UNKNOWN,
        context: ErrorContext | None = None,
    ) -> None:
        if context:
            _validate_provider_fields(context)
        self._provider = context.get("provider") if context else None
        self._model_name = context.get("model_name") if context else None
        super().__init__(message, severity, error_code, context)

    @property
    def provider(self) -> ModelProvider | None:
        return self._provider

    @property
    def model_name(self) -> str | None:
        return self._model_name


class RetryableError(ProviderError):
    """Base class for provider errors that are safe to retry."""

    def __init__(
        self,
        message: str,
        retry_policy: RetryPolicy | None = None,
        error_code: ErrorCode | str | None = None,
        context: ErrorContext | None = None,
    ) -> None:
        self.retry_policy = retry_policy
        ctx = _build_context(context, {})
        if retry_policy is not None:
            extra = ctx.get("extra", ErrorExtra())
            if not isinstance(extra, ErrorExtra):
                extra = ErrorExtra(extra)
            extra["retry_policy"] = {
                "max_retries": retry_policy.max_retries,
                "base_delay_seconds": retry_policy.base_delay_seconds,
                "max_delay_seconds": retry_policy.max_delay_seconds,
                "backoff_multiplier": retry_policy.backoff_multiplier,
                "jitter_strategy": retry_policy.jitter_strategy.value,
                "retry_budget": retry_policy.retry_budget,
                "per_provider_budget": retry_policy.per_provider_budget,
            }
            ctx["extra"] = extra
        super().__init__(message, ErrorSeverity.MEDIUM, error_code, ctx)

    @property
    def retryable(self) -> bool:
        return True

    @property
    def fatal(self) -> bool:
        return False


# -----------------------------------------------------------------------------
# Specific provider errors
# -----------------------------------------------------------------------------

class AuthenticationError(ProviderError):
    def __init__(self, message: str, context: ErrorContext | None = None) -> None:
        super().__init__(message, ErrorSeverity.HIGH, ErrorCode.CF_AUTH, context)


class ProviderConnectionError(RetryableError):
    def __init__(
        self,
        message: str,
        retry_policy: RetryPolicy | None = None,
        context: ErrorContext | None = None,
    ) -> None:
        super().__init__(message, retry_policy, ErrorCode.CF_NETWORK, context)


class RateLimitError(RetryableError):
    def __init__(
        self,
        message: str,
        retry_policy: RetryPolicy | None = None,
        context: ErrorContext | None = None,
    ) -> None:
        super().__init__(message, retry_policy, ErrorCode.CF_RATE_LIMIT, context)


class QuotaExhaustedError(ProviderError):
    def __init__(self, message: str, context: ErrorContext | None = None) -> None:
        super().__init__(message, ErrorSeverity.HIGH, ErrorCode.CF_QUOTA, context)

    @property
    def retryable(self) -> bool:
        return False

    @property
    def fatal(self) -> bool:
        return True


# -----------------------------------------------------------------------------
# Data processing errors
# -----------------------------------------------------------------------------

class ParsingFailureError(OperationalError):
    def __init__(self, message: str, context: ErrorContext | None = None) -> None:
        super().__init__(message, ErrorSeverity.MEDIUM, ErrorCode.CF_PARSE, context)


class DataValidationError(OperationalError):
    def __init__(self, message: str, context: ErrorContext | None = None) -> None:
        super().__init__(message, ErrorSeverity.HIGH, ErrorCode.CF_VALIDATE, context)


class ExtractionError(OperationalError):
    def __init__(self, message: str, context: ErrorContext | None = None) -> None:
        super().__init__(message, ErrorSeverity.HIGH, ErrorCode.CF_EXTRACT, context)


# -----------------------------------------------------------------------------
# Storage exceptions
# -----------------------------------------------------------------------------

class StorageError(OperationalError):
    """
    Base class for storage-related failures.
    """

    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.HIGH,
        error_code: ErrorCode | str | None = ErrorCode.CF_STORAGE,
        context: ErrorContext | None = None,
    ) -> None:
        super().__init__(message, severity, error_code, context)


class StorageConnectionError(StorageError):
    """
    Raised when the storage backend is unreachable.
    """

    def __init__(self, message: str, context: ErrorContext | None = None) -> None:
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            error_code=ErrorCode.CF_STORAGE_CONNECTION,
            context=context,
        )


class DataPersistenceError(StorageError):
    """
    Raised when a write/update/delete operation fails.
    """

    def __init__(self, message: str, context: ErrorContext | None = None) -> None:
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            error_code=ErrorCode.CF_DATA_PERSISTENCE,
            context=context,
        )


# =============================================================================
# 7. Helper Functions
# =============================================================================


def is_control_signal(exc: BaseException) -> bool:
    """Return True if the exception is a control-flow signal."""
    return isinstance(exc, OperationalSignal)


def is_critical_signal(exc: BaseException) -> bool:
    """Return True if the exception is a critical control-flow signal."""
    return isinstance(exc, CriticalControlSignal)
