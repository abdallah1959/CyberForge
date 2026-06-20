# cyberforge/ai/key_rotation_manager.py
"""
API Key rotation and state management system.

Completely decoupled from the retry engine and user interface logic.
Manages a pool of API keys with exhaustion tracking and round‑robin rotation.
"""

import logging
from threading import Lock
from typing import Set

from cyberforge.core.exceptions import ConfigurationError, QuotaExhaustedError

# Public API contract
__all__ = ["KeyRotationManager"]

logger = logging.getLogger(__name__)


class KeyRotationManager:
    """
    Manages state and rotation for a pool of API keys.

    Tracks exhaustion levels to seamlessly switch keys during 429 Rate Limit events.
    This class is thread‑safe: all state‑mutating methods are protected by a lock.

    The rotation algorithm uses a round‑robin strategy that skips exhausted keys,
    and raises QuotaExhaustedError when all keys are exhausted.

    Example:
        >>> manager = KeyRotationManager(["key1", "key2", "key3"])
        >>> key = manager.get_current_key()
        >>> # If key is rate‑limited:
        >>> manager.mark_current_exhausted_and_rotate()
        >>> next_key = manager.get_current_key()
    """

    def __init__(self, api_keys: tuple[str, ...]) -> None:
        """
        Initialises the key rotation manager with a pool of API keys.

        Args:
            api_keys: A tuple of API key strings. Must contain at least one key.

        Raises:
            ConfigurationError: If the key pool is empty.
        """
        if not api_keys:
            raise ConfigurationError(
                "KeyRotationManager requires at least one API key to initialise. "
                "Please configure the appropriate API_KEYS in your environment."
            )
        self._keys: tuple[str, ...] = api_keys
        self._current_index: int = 0
        self._exhausted_indices: Set[int] = set()
        self._lock: Lock = Lock()

    def get_current_key(self) -> str:
        """
        Retrieves the currently active API key.

        Returns:
            The current API key as a string.

        Raises:
            QuotaExhaustedError: If all keys are marked as exhausted.
        """
        with self._lock:
            if len(self._exhausted_indices) >= len(self._keys):
                logger.critical(
                    "All %d API keys have been marked as exhausted.",
                    len(self._keys),
                )
                raise QuotaExhaustedError(
                    "No active API keys available. Pipeline stalled."
                )
            return self._keys[self._current_index]

    def mark_current_exhausted_and_rotate(self) -> None:
        """
        Flags the current key as exhausted and rotates to the next available key.

        This method is idempotent: if the key is already exhausted, it still
        rotates to the next valid key.

        Raises:
            QuotaExhaustedError: If no fallback keys are available after rotation.
        """
        with self._lock:
            logger.warning(
                "Flagging API key at index %d as exhausted.",
                self._current_index,
            )
            self._exhausted_indices.add(self._current_index)

            if len(self._exhausted_indices) >= len(self._keys):
                raise QuotaExhaustedError(
                    "Key rotation failed: All fallback keys are also exhausted."
                )

            # Round‑robin search for the next valid key.
            initial_index = self._current_index
            while True:
                self._current_index = (self._current_index + 1) % len(self._keys)
                if self._current_index not in self._exhausted_indices:
                    logger.info(
                        "Successfully rotated to API key at index %d.",
                        self._current_index,
                    )
                    break
                if self._current_index == initial_index:
                    # This should never happen because we already checked
                    # that not all keys are exhausted, but we keep it as a safety net.
                    raise QuotaExhaustedError(
                        "Rotation logic error: Infinite loop prevented."
                    )

    def reset_exhaustion_state(self) -> None:
        """
        Clears all exhaustion flags and resets the current key to the first one.

        This is typically called after a global cooldown period or when
        manual intervention is required.
        """
        with self._lock:
            logger.info(
                "Resetting exhaustion states for all %d API keys.",
                len(self._keys),
            )
            self._exhausted_indices.clear()
            self._current_index = 0

    @property
    def available_key_count(self) -> int:
        """
        Returns the number of currently available (non‑exhausted) keys.

        This is useful for monitoring and health checks.
        """
        with self._lock:
            return len(self._keys) - len(self._exhausted_indices)

    @property
    def total_key_count(self) -> int:
        """Returns the total number of API keys in the pool."""
        return len(self._keys)
