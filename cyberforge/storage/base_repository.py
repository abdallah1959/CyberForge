# cyberforge/storage/base_repository.py
"""
Abstract Base Repository for the CyberForge Storage Layer.

Defines the strict enterprise contract for data persistence mechanisms.
Ensures decoupling between the application logic and the database/export technologies.

This contract is designed to be:
    - Async-native (all I/O operations are asynchronous)
    - Bulk-ready (supports batch operations for high throughput)
    - Type-safe (uses Pydantic models and modern Python typing)
    - Observable (includes health checks, counts, and existence checks)
    - Symmetric (full CRUD operations for both ExtractedItem and ParsedWriteup)

Future Backends:
    - SQLiteRepository
    - PostgreSQLRepository
    - JSONExporter
    - ParquetExporter

Note on Exceptions:
    The docstrings reference StorageConnectionError and DataPersistenceError.
    These must be defined in cyberforge/core/exceptions.py before implementing
    concrete repositories. They should inherit from CyberForgeBaseException.
"""

from abc import ABC, abstractmethod
from typing import TypeAlias

from cyberforge.schemas.extracted_item import ExtractedItem
from cyberforge.schemas.parsed_writeup import ParsedWriteup

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

__all__ = [
    "BaseRepository",
    "RepositoryId",
]

# Type alias for repository identifiers (UUIDs, ULIDs, or database primary keys).
RepositoryId: TypeAlias = str


class BaseRepository(ABC):
    """
    Universal interface for all storage backends (e.g., SQLite, PostgreSQL, JSON Exporters).

    This contract guarantees seamless integration with Pydantic V2 models across
    the entire CyberForge pipeline.

    All methods are asynchronous to support non-blocking I/O in the async-first
    architecture of CyberForge. Implementations should use async database drivers
    (e.g., aiosqlite, asyncpg) to avoid blocking the event loop.

    The contract is fully symmetrical: every operation available for ParsedWriteup
    is also available for ExtractedItem, ensuring that both raw and processed data
    can be managed identically.
    """

    # -------------------------------------------------------------------------
    # Repository Metadata
    # -------------------------------------------------------------------------

    @property
    @abstractmethod
    def repository_name(self) -> str:
        """
        Return the name of the repository (e.g., "sqlite", "postgres", "json").

        This is useful for logging, monitoring, and multi-repository routing.
        """
        ...

    @abstractmethod
    async def initialize(self) -> None:
        """
        Asynchronously bootstraps the storage backend, creating schemas and indexes.

        This method is called during application startup to ensure the backend
        is ready for operations. It should be idempotent (safe to call multiple times).

        Raises:
            StorageConnectionError: If the backend configuration fails or
                the database is unreachable.
        """
        ...

    # -------------------------------------------------------------------------
    # Write Operations for ExtractedItem (Single & Bulk)
    # -------------------------------------------------------------------------

    @abstractmethod
    async def save_extracted_item(self, item: ExtractedItem) -> RepositoryId:
        """
        Persists a raw, unstructured extracted item to the storage backend.

        Args:
            item: The raw ExtractedItem Pydantic model.

        Returns:
            The unique identifier of the saved record.

        Raises:
            StorageConnectionError: If the backend is unreachable.
            DataPersistenceError: If the write operation fails.
        """
        ...

    @abstractmethod
    async def save_many_extracted_items(self, items: list[ExtractedItem]) -> list[RepositoryId]:
        """
        Persists multiple raw extracted items in a single batch operation.

        Critical for high-throughput ingestion pipelines.

        Args:
            items: A list of ExtractedItem Pydantic models.

        Returns:
            A list of unique identifiers for each saved record, in the same order.

        Raises:
            StorageConnectionError: If the backend is unreachable.
            DataPersistenceError: If the batch write operation fails.
        """
        ...

    # -------------------------------------------------------------------------
    # Write Operations for ParsedWriteup (Single & Bulk)
    # -------------------------------------------------------------------------

    @abstractmethod
    async def save_parsed_writeup(
        self,
        writeup: ParsedWriteup,
        source_id: RepositoryId | None = None,
    ) -> RepositoryId:
        """
        Persists a validated, AI-parsed cybersecurity writeup to the storage backend.

        Args:
            writeup: The strictly validated ParsedWriteup Pydantic model.
            source_id: An optional reference ID linking back to the raw ExtractedItem.

        Returns:
            The unique identifier of the saved record.

        Raises:
            StorageConnectionError: If the backend is unreachable.
            DataPersistenceError: If the write operation fails.
        """
        ...

    @abstractmethod
    async def save_many_parsed_writeups(
        self,
        writeups: list[ParsedWriteup],
    ) -> list[RepositoryId]:
        """
        Persists multiple parsed writeups in a single batch operation.

        Critical for high-throughput pipelines processing thousands of writeups.

        Args:
            writeups: A list of validated ParsedWriteup Pydantic models.

        Returns:
            A list of unique identifiers for each saved record, in the same order.

        Raises:
            StorageConnectionError: If the backend is unreachable.
            DataPersistenceError: If the batch write operation fails.
        """
        ...

    # -------------------------------------------------------------------------
    # Read Operations for ExtractedItem
    # -------------------------------------------------------------------------

    @abstractmethod
    async def get_extracted_item(self, item_id: RepositoryId) -> ExtractedItem | None:
        """
        Retrieves a raw extracted item from storage by its unique identifier.

        Args:
            item_id: The unique identifier of the target record.

        Returns:
            The ExtractedItem object if found, otherwise None.

        Raises:
            StorageConnectionError: If the backend is unreachable.
        """
        ...

    # -------------------------------------------------------------------------
    # Read Operations for ParsedWriteup
    # -------------------------------------------------------------------------

    @abstractmethod
    async def get_parsed_writeup(self, writeup_id: RepositoryId) -> ParsedWriteup | None:
        """
        Retrieves a parsed writeup from storage by its unique identifier.

        Args:
            writeup_id: The unique identifier of the target record.

        Returns:
            The ParsedWriteup object if found, otherwise None.

        Raises:
            StorageConnectionError: If the backend is unreachable.
        """
        ...

    @abstractmethod
    async def get_all_parsed_writeups(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ParsedWriteup]:
        """
        Retrieves a paginated list of parsed writeups for bulk processing or dataset generation.

        Args:
            limit: The maximum number of records to return.
            offset: The number of records to skip (for pagination).

        Returns:
            A list of instantiated ParsedWriteup Pydantic models.

        Raises:
            StorageConnectionError: If the backend is unreachable.
        """
        ...

    # -------------------------------------------------------------------------
    # Existence & Counting
    # -------------------------------------------------------------------------

    @abstractmethod
    async def exists_extracted_item(self, item_id: RepositoryId) -> bool:
        """
        Check if an extracted item exists in the storage backend.

        Args:
            item_id: The unique identifier of the target record.

        Returns:
            True if the record exists, False otherwise.

        Raises:
            StorageConnectionError: If the backend is unreachable.
        """
        ...

    @abstractmethod
    async def exists_parsed_writeup(self, writeup_id: RepositoryId) -> bool:
        """
        Check if a parsed writeup exists in the storage backend.

        Args:
            writeup_id: The unique identifier of the target record.

        Returns:
            True if the record exists, False otherwise.

        Raises:
            StorageConnectionError: If the backend is unreachable.
        """
        ...

    @abstractmethod
    async def count_extracted_items(self) -> int:
        """
        Return the total number of extracted items in the storage backend.

        Returns:
            The total record count.

        Raises:
            StorageConnectionError: If the backend is unreachable.
        """
        ...

    @abstractmethod
    async def count_parsed_writeups(self) -> int:
        """
        Return the total number of parsed writeups in the storage backend.

        Returns:
            The total record count.

        Raises:
            StorageConnectionError: If the backend is unreachable.
        """
        ...

    # -------------------------------------------------------------------------
    # Delete Operations (Optional, but recommended)
    # -------------------------------------------------------------------------

    @abstractmethod
    async def delete_extracted_item(self, item_id: RepositoryId) -> bool:
        """
        Delete an extracted item from storage by its unique identifier.

        Args:
            item_id: The unique identifier of the target record.

        Returns:
            True if the record was deleted, False if it did not exist.

        Raises:
            StorageConnectionError: If the backend is unreachable.
            DataPersistenceError: If the delete operation fails.
        """
        ...

    @abstractmethod
    async def delete_parsed_writeup(self, writeup_id: RepositoryId) -> bool:
        """
        Delete a parsed writeup from storage by its unique identifier.

        Args:
            writeup_id: The unique identifier of the target record.

        Returns:
            True if the record was deleted, False if it did not exist.

        Raises:
            StorageConnectionError: If the backend is unreachable.
            DataPersistenceError: If the delete operation fails.
        """
        ...

    # -------------------------------------------------------------------------
    # Health & Connectivity
    # -------------------------------------------------------------------------

    @abstractmethod
    async def health_check(self) -> None:
        """
        Verify the connection and operational status of the storage backend.

        This method should be lightweight and quick to execute. It is used for:
            - Startup validation
            - Health probes in production
            - Diagnostic debugging

        Raises:
            StorageConnectionError: If the backend is unreachable or unhealthy.
        """
        ...
