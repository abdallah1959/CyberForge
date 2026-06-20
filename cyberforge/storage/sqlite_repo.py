# cyberforge/storage/sqlite_repo.py
"""
SQLite implementation of the BaseRepository contract.

Provides robust, asynchronous, and type-safe data persistence using aiosqlite.
Automatically handles serialization/deserialization of Pydantic V2 models,
including complex nested JSON structures for payloads and evidences.

Supports:
    - Full CRUD operations for both ExtractedItem and ParsedWriteup
    - Bulk operations (save_many) with optional source_id mapping
    - Foreign key constraints with PRAGMA foreign_keys = ON
    - Transaction boundaries with explicit BEGIN/ROLLBACK
    - Connection reuse via context manager
    - ULID-based primary keys for time-sortable IDs
    - WAL journal mode for better concurrency
    - Performance indexes for common query patterns
"""

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import aiosqlite
from ulid import ULID

from cyberforge.core.exceptions import DataPersistenceError, StorageConnectionError
from cyberforge.schemas.extracted_item import ExtractedItem
from cyberforge.schemas.parsed_writeup import ParsedWriteup
from cyberforge.storage.base_repository import BaseRepository, RepositoryId

logger = logging.getLogger(__name__)


class SQLiteRepository(BaseRepository):
    """
    Asynchronous SQLite repository for local, high-performance data storage.

    Uses ULID for primary keys (time-sortable, collision-resistant).
    All operations are async and thread-safe via aiosqlite.

    Example:
        >>> async with SQLiteRepository("data.db") as repo:
        ...     await repo.health_check()
        ...     writeup_id = await repo.save_parsed_writeup(writeup)
    """

    def __init__(self, db_path: str | Path) -> None:
        """
        Initialises the repository.

        Args:
            db_path: Path to the SQLite database file.
        """
        self._db_path = str(db_path)
        self._initialized = False
        self._connection: aiosqlite.Connection | None = None

    @property
    def repository_name(self) -> str:
        return "sqlite"

    # -------------------------------------------------------------------------
    # Connection & Lifecycle Management
    # -------------------------------------------------------------------------

    async def __aenter__(self) -> "SQLiteRepository":
        """Async context manager entry."""
        await self._ensure_initialized()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    async def close(self) -> None:
        """Close the database connection if open."""
        if self._connection is not None:
            await self._connection.close()
            self._connection = None
            self._initialized = False
            logger.debug("SQLite connection closed.")

    async def initialize(self) -> None:
        """
        Public API method to bootstrap the repository backend.
        Ensures all tables, constraints, and performance indexes are generated.
        """
        await self._initialize_schema()

    @asynccontextmanager
    async def _get_connection(self) -> AsyncIterator[aiosqlite.Connection]:
        """
        Get a database connection, ensuring PRAGMA settings are applied.

        If a persistent connection exists, reuse it. Otherwise, create a new one.
        """
        if self._connection is None:
            try:
                self._connection = await aiosqlite.connect(self._db_path)
                # Enable foreign key constraints.
                await self._connection.execute("PRAGMA foreign_keys = ON")
                # Enable WAL mode for better concurrency.
                await self._connection.execute("PRAGMA journal_mode = WAL")
            except aiosqlite.Error as exc:
                raise StorageConnectionError(
                    f"Failed to connect to SQLite database at {self._db_path}: {exc}"
                ) from exc

        # Ensure the connection is still alive.
        try:
            await self._connection.execute("SELECT 1")
        except aiosqlite.Error:
            # Connection is dead, create a new one.
            if self._connection is not None:
                await self._connection.close()
            self._connection = await aiosqlite.connect(self._db_path)
            await self._connection.execute("PRAGMA foreign_keys = ON")
            await self._connection.execute("PRAGMA journal_mode = WAL")

        yield self._connection

    async def _ensure_initialized(self) -> None:
        """Ensure the database schema is created before any operation."""
        if not self._initialized:
            await self._initialize_schema()

    async def _initialize_schema(self) -> None:
        """Create the database tables and indexes if they don't exist."""
        create_extracted_table = """
        CREATE TABLE IF NOT EXISTS extracted_items (
            id TEXT PRIMARY KEY,
            source_name TEXT NOT NULL,
            url TEXT NOT NULL,
            raw_content TEXT NOT NULL,
            author TEXT,
            published_date TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """

        create_parsed_table = """
        CREATE TABLE IF NOT EXISTS parsed_writeups (
            id TEXT PRIMARY KEY,
            source_id TEXT,
            title TEXT NOT NULL,
            target_system TEXT NOT NULL,
            severity TEXT NOT NULL,
            cve_id TEXT,
            summary TEXT NOT NULL,
            payloads_json TEXT NOT NULL,
            evidences_json TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(source_id) REFERENCES extracted_items(id) ON DELETE SET NULL
        );
        """

        # Performance indexes for common queries.
        create_indexes = """
        CREATE INDEX IF NOT EXISTS idx_parsed_created_at ON parsed_writeups(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_parsed_severity ON parsed_writeups(severity);
        CREATE INDEX IF NOT EXISTS idx_parsed_cve ON parsed_writeups(cve_id);
        CREATE INDEX IF NOT EXISTS idx_extracted_source ON extracted_items(source_name);
        """

        async with self._get_connection() as conn:
            try:
                await conn.execute(create_extracted_table)
                await conn.execute(create_parsed_table)
                # Use executescript for multiple statements.
                await conn.executescript(create_indexes)
                await conn.commit()
                self._initialized = True
                logger.info(f"SQLiteRepository initialized at {self._db_path}")
            except aiosqlite.Error as exc:
                raise StorageConnectionError(
                    f"Failed to initialize SQLite schemas: {exc}"
                ) from exc

    @staticmethod
    def _generate_id() -> RepositoryId:
        """Generate a time-sortable ULID as the primary key."""
        # Using python-ulid (https://github.com/mdomke/python-ulid)
        return str(ULID())

    # -------------------------------------------------------------------------
    # Core CRUD Operations (as required by BaseRepository)
    # -------------------------------------------------------------------------

    async def exists(self, writeup_id: RepositoryId) -> bool:
        """Check if a ParsedWriteup exists by ID."""
        return await self.exists_parsed_writeup(writeup_id)

    async def count(self) -> int:
        """Return the total number of ParsedWriteup records."""
        return await self.count_parsed_writeups()

    # -------------------------------------------------------------------------
    # Write Operations for ExtractedItem
    # -------------------------------------------------------------------------

    async def save_extracted_item(self, item: ExtractedItem) -> RepositoryId:
        await self._ensure_initialized()

        record_id = self._generate_id()
        query = """
            INSERT INTO extracted_items
            (id, source_name, url, raw_content, author, published_date)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        params = (
            record_id,
            item.source_name,
            str(item.url),
            item.raw_content,
            item.author,
            item.published_date,
        )

        async with self._get_connection() as conn:
            try:
                await conn.execute(query, params)
                await conn.commit()
                return record_id
            except aiosqlite.Error as exc:
                raise DataPersistenceError(
                    f"Failed to save ExtractedItem: {exc}"
                ) from exc

    async def save_many_extracted_items(self, items: list[ExtractedItem]) -> list[RepositoryId]:
        await self._ensure_initialized()

        records = []
        ids = []
        for item in items:
            rid = self._generate_id()
            ids.append(rid)
            records.append(
                (
                    rid,
                    item.source_name,
                    str(item.url),
                    item.raw_content,
                    item.author,
                    item.published_date,
                )
            )

        query = """
            INSERT INTO extracted_items
            (id, source_name, url, raw_content, author, published_date)
            VALUES (?, ?, ?, ?, ?, ?)
        """

        async with self._get_connection() as conn:
            try:
                await conn.execute("BEGIN TRANSACTION")
                await conn.executemany(query, records)
                await conn.commit()
                return ids
            except aiosqlite.Error as exc:
                await conn.rollback()
                raise DataPersistenceError(
                    f"Failed to batch save ExtractedItems: {exc}"
                ) from exc

    # -------------------------------------------------------------------------
    # Write Operations for ParsedWriteup
    # -------------------------------------------------------------------------

    async def save_parsed_writeup(
        self,
        writeup: ParsedWriteup,
        source_id: RepositoryId | None = None,
    ) -> RepositoryId:
        await self._ensure_initialized()

        record_id = self._generate_id()
        query = """
            INSERT INTO parsed_writeups
            (id, source_id, title, target_system, severity, cve_id, summary,
             payloads_json, evidences_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        payloads_json = json.dumps(
            [p.model_dump(mode="json") for p in writeup.payloads]
        )
        evidences_json = json.dumps(
            [e.model_dump(mode="json") for e in writeup.evidences]
        )
        params = (
            record_id,
            source_id,
            writeup.title,
            writeup.target_system,
            writeup.severity.value,
            writeup.cve_id,
            writeup.summary,
            payloads_json,
            evidences_json,
        )

        async with self._get_connection() as conn:
            try:
                await conn.execute(query, params)
                await conn.commit()
                return record_id
            except aiosqlite.Error as exc:
                raise DataPersistenceError(
                    f"Failed to save ParsedWriteup: {exc}"
                ) from exc

    async def save_many_parsed_writeups(
        self,
        writeups: list[ParsedWriteup],
        source_ids: list[RepositoryId] | None = None,
    ) -> list[RepositoryId]:
        """
        Persists multiple parsed writeups in a single batch operation.

        Args:
            writeups: List of ParsedWriteup models.
            source_ids: Optional list of source IDs in the same order as writeups.
                        If None, source_id is set to NULL for all records.

        Returns:
            List of generated IDs.

        Raises:
            ValueError: If source_ids length does not match writeups.
            DataPersistenceError: If the batch operation fails.
        """
        await self._ensure_initialized()

        if source_ids is not None and len(source_ids) != len(writeups):
            raise ValueError("source_ids length must match writeups length")

        records = []
        ids = []
        for idx, writeup in enumerate(writeups):
            rid = self._generate_id()
            ids.append(rid)
            payloads_json = json.dumps(
                [p.model_dump(mode="json") for p in writeup.payloads]
            )
            evidences_json = json.dumps(
                [e.model_dump(mode="json") for e in writeup.evidences]
            )
            source_id = source_ids[idx] if source_ids else None
            records.append(
                (
                    rid,
                    source_id,
                    writeup.title,
                    writeup.target_system,
                    writeup.severity.value,
                    writeup.cve_id,
                    writeup.summary,
                    payloads_json,
                    evidences_json,
                )
            )

        query = """
            INSERT INTO parsed_writeups
            (id, source_id, title, target_system, severity, cve_id, summary,
             payloads_json, evidences_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        async with self._get_connection() as conn:
            try:
                await conn.execute("BEGIN TRANSACTION")
                await conn.executemany(query, records)
                await conn.commit()
                return ids
            except aiosqlite.Error as exc:
                await conn.rollback()
                raise DataPersistenceError(
                    f"Failed to batch save ParsedWriteups: {exc}"
                ) from exc

    # -------------------------------------------------------------------------
    # Read Operations for ExtractedItem
    # -------------------------------------------------------------------------

    async def get_extracted_item(self, item_id: RepositoryId) -> ExtractedItem | None:
        await self._ensure_initialized()

        query = """
            SELECT source_name, url, raw_content, author, published_date
            FROM extracted_items
            WHERE id = ?
        """
        async with self._get_connection() as conn:
            try:
                conn.row_factory = aiosqlite.Row
                cursor = await conn.execute(query, (item_id,))
                row = await cursor.fetchone()
                if row is None:
                    return None
                return ExtractedItem.model_validate(dict(row))
            except aiosqlite.Error as exc:
                raise StorageConnectionError(
                    f"Failed to read ExtractedItem: {exc}"
                ) from exc
            finally:
                conn.row_factory = None

    # -------------------------------------------------------------------------
    # Read Operations for ParsedWriteup
    # -------------------------------------------------------------------------

    async def get_parsed_writeup(self, writeup_id: RepositoryId) -> ParsedWriteup | None:
        await self._ensure_initialized()

        query = """
            SELECT title, target_system, severity, cve_id, summary,
                   payloads_json, evidences_json
            FROM parsed_writeups
            WHERE id = ?
        """
        async with self._get_connection() as conn:
            try:
                conn.row_factory = aiosqlite.Row
                cursor = await conn.execute(query, (writeup_id,))
                row = await cursor.fetchone()
                if row is None:
                    return None

                data = dict(row)
                data["payloads"] = json.loads(data.pop("payloads_json"))
                data["evidences"] = json.loads(data.pop("evidences_json"))
                return ParsedWriteup.model_validate(data)
            except (aiosqlite.Error, json.JSONDecodeError) as exc:
                raise StorageConnectionError(
                    f"Failed to read ParsedWriteup: {exc}"
                ) from exc
            finally:
                conn.row_factory = None

    async def get_all_parsed_writeups(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ParsedWriteup]:
        await self._ensure_initialized()

        query = """
            SELECT title, target_system, severity, cve_id, summary,
                   payloads_json, evidences_json
            FROM parsed_writeups
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
        """
        results = []
        async with self._get_connection() as conn:
            try:
                conn.row_factory = aiosqlite.Row
                cursor = await conn.execute(query, (limit, offset))
                async for row in cursor:
                    data = dict(row)
                    data["payloads"] = json.loads(data.pop("payloads_json"))
                    data["evidences"] = json.loads(data.pop("evidences_json"))
                    results.append(ParsedWriteup.model_validate(data))
                return results
            except (aiosqlite.Error, json.JSONDecodeError) as exc:
                raise StorageConnectionError(
                    f"Failed to fetch parsed writeups: {exc}"
                ) from exc
            finally:
                conn.row_factory = None

    # -------------------------------------------------------------------------
    # Existence & Counting (detailed versions)
    # -------------------------------------------------------------------------

    async def exists_extracted_item(self, item_id: RepositoryId) -> bool:
        await self._ensure_initialized()
        async with self._get_connection() as conn:
            try:
                cursor = await conn.execute(
                    "SELECT 1 FROM extracted_items WHERE id = ?", (item_id,)
                )
                return await cursor.fetchone() is not None
            except aiosqlite.Error as exc:
                raise StorageConnectionError(
                    f"Existence check failed: {exc}"
                ) from exc

    async def exists_parsed_writeup(self, writeup_id: RepositoryId) -> bool:
        await self._ensure_initialized()
        async with self._get_connection() as conn:
            try:
                cursor = await conn.execute(
                    "SELECT 1 FROM parsed_writeups WHERE id = ?", (writeup_id,)
                )
                return await cursor.fetchone() is not None
            except aiosqlite.Error as exc:
                raise StorageConnectionError(
                    f"Existence check failed: {exc}"
                ) from exc

    async def count_extracted_items(self) -> int:
        await self._ensure_initialized()
        async with self._get_connection() as conn:
            try:
                cursor = await conn.execute("SELECT COUNT(*) FROM extracted_items")
                row = await cursor.fetchone()
                return row[0] if row else 0
            except aiosqlite.Error as exc:
                raise StorageConnectionError(
                    f"Count operation failed: {exc}"
                ) from exc

    async def count_parsed_writeups(self) -> int:
        await self._ensure_initialized()
        async with self._get_connection() as conn:
            try:
                cursor = await conn.execute("SELECT COUNT(*) FROM parsed_writeups")
                row = await cursor.fetchone()
                return row[0] if row else 0
            except aiosqlite.Error as exc:
                raise StorageConnectionError(
                    f"Count operation failed: {exc}"
                ) from exc

    # -------------------------------------------------------------------------
    # Delete Operations
    # -------------------------------------------------------------------------

    async def delete_extracted_item(self, item_id: RepositoryId) -> bool:
        await self._ensure_initialized()
        async with self._get_connection() as conn:
            try:
                cursor = await conn.execute(
                    "DELETE FROM extracted_items WHERE id = ?", (item_id,)
                )
                await conn.commit()
                return cursor.rowcount > 0
            except aiosqlite.Error as exc:
                await conn.rollback()
                raise DataPersistenceError(
                    f"Delete operation failed: {exc}"
                ) from exc

    async def delete_parsed_writeup(self, writeup_id: RepositoryId) -> bool:
        await self._ensure_initialized()
        async with self._get_connection() as conn:
            try:
                cursor = await conn.execute(
                    "DELETE FROM parsed_writeups WHERE id = ?", (writeup_id,)
                )
                await conn.commit()
                return cursor.rowcount > 0
            except aiosqlite.Error as exc:
                await conn.rollback()
                raise DataPersistenceError(
                    f"Delete operation failed: {exc}"
                ) from exc

    # -------------------------------------------------------------------------
    # Health Check
    # -------------------------------------------------------------------------

    async def health_check(self) -> None:
        await self._ensure_initialized()
        async with self._get_connection() as conn:
            try:
                await conn.execute("SELECT 1")
            except aiosqlite.Error as exc:
                logger.error(f"SQLite health check failed: {exc}")
                raise StorageConnectionError(
                    f"Database health check failed: {exc}"
                ) from exc
