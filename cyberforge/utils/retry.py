# cyberforge/utils/retry.py
"""
Enterprise-grade retry engine.

Implements exponential backoff with full jitter to prevent the Thundering Herd problem.
This layer is completely stateless and decoupled from business logic or CLI interactions.
"""

import logging
import random
import time
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

# TypeVar for generic return type hinting
T = TypeVar("T")


def execute_with_retry(
    func: Callable[..., T],
    retryable_exceptions: tuple[type[Exception], ...],
    max_retries: int = 3,
    base_delay_seconds: float = 2.0,
    max_delay_seconds: float = 60.0,
    *args: Any,
    **kwargs: Any,
) -> T:
    """
    Executes a callable with an exponential backoff and jitter retry strategy.

    Args:
        func: The target callable to execute.
        retryable_exceptions: A tuple of exception classes that should trigger a retry.
        max_retries: The maximum number of retry attempts before raising the exception.
        base_delay_seconds: The initial delay before the first retry.
        max_delay_seconds: The maximum allowed delay between retries to prevent infinite stalls.
        *args: Positional arguments to pass to the target function.
        **kwargs: Keyword arguments to pass to the target function.

    Returns:
        The return value of the target function.

    Raises:
        ValueError: If max_retries, base_delay_seconds, or max_delay_seconds are invalid.
        Exception: Re-raises the target exception if max_retries is exceeded,
                   or immediately raises any exception not explicitly listed in retryable_exceptions.
    """
    if max_retries < 0:
        raise ValueError("max_retries must be a non-negative integer.")
    if base_delay_seconds <= 0 or max_delay_seconds <= 0:
        raise ValueError("Delay timings must be strictly positive floats.")
    if base_delay_seconds > max_delay_seconds:
        raise ValueError(
            "base_delay_seconds cannot exceed max_delay_seconds."
        )

    attempt = 0

    while True:
        try:
            return func(*args, **kwargs)

        except retryable_exceptions as e:
            if attempt >= max_retries:
                logger.error(
                    "Target function '%s' failed after %d retries. "
                    "Final exception: %s - %s",
                    func.__name__,
                    max_retries,
                    type(e).__name__,
                    str(e),
                )
                raise  # Preserve the original traceback

            # AWS Full Jitter Algorithm: sleep = random_between(0, min(cap, base * 2 ** attempt))
            exponential_delay = min(
                max_delay_seconds,
                base_delay_seconds * (2 ** attempt)
            )
            jittered_delay = random.uniform(0.0, exponential_delay)

            logger.warning(
                "Execution failed due to %s: %s. "
                "Retrying in %.2f seconds... (Attempt %d/%d)",
                type(e).__name__,
                str(e),
                jittered_delay,
                attempt + 1,
                max_retries,
            )

            time.sleep(jittered_delay)
            attempt += 1
