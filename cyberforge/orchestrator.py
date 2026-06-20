# cyberforge/orchestrator.py
"""
CyberForge Core Orchestrator.

Acts as the "Maestro" of the pipeline. It glues together the Ingestion (Extractors),
Processing (AI Parser), and Persistence (Storage) layers.
Designed to be fully asynchronous, fault-tolerant, and ready for high-throughput batch processing.

The orchestrator is responsible for:
    - Selecting the appropriate extractor via ExtractorFactory.
    - Managing the lifecycle of repository connections (reused across batch).
    - Coordinating the extraction → parsing → storage pipeline.
    - Handling errors gracefully and propagating control-flow signals.
    - Recording telemetry and audit events (future enhancement).

Future Optimisation:
    Currently, a new extractor session is created for each URL (via `async with extractor`).
    For high-throughput scenarios, consider implementing an ExtractorPool or ExtractorManager
    that reuses extractors per source across multiple URLs.
"""

import asyncio
import logging
from typing import Any, Optional

from cyberforge.ai.ai_parser_service import AIParserService
from cyberforge.config import Settings, settings as global_settings
from cyberforge.core.exceptions import (
    ControlFlowSignal,
    DataPersistenceError,
    ExtractionError,
    ParsingFailureError,
    ProviderError,
    StorageConnectionError,
)
from cyberforge.extractors.extractor_factory import ExtractorFactory
from cyberforge.schemas.parsed_writeup import ParsedWriteup
from cyberforge.storage.base_repository import BaseRepository, RepositoryId
from cyberforge.storage.sqlite_repo import SQLiteRepository

logger = logging.getLogger(__name__)


class CyberForgeOrchestrator:
    """
    Manages the end-to-end data engineering pipeline.

    Usage:
        1. Instantiate with a settings object (or use default).
        2. Call `initialize()` to set up database schema and health checks.
        3. Call `process_single_url()` or `process_batch()`.

    The orchestrator uses `ExtractorFactory` to obtain the correct extractor
    for each URL based on its domain or source type.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        db_path: str = "cyberforge_data.sqlite",
    ) -> None:
        """
        Initialises the orchestrator.

        Args:
            settings: Application settings (defaults to global settings).
            db_path: Path to the SQLite database file.
        """
        self._settings = settings or global_settings
        self._db_path = db_path

        # Core services.
        self._parser_service = AIParserService(settings=self._settings)

        # Repository (to be initialised in `initialize`).
        self._repository: Optional[BaseRepository] = None

        # Telemetry and audit (placeholders for future implementation).
        self._metrics_tracker: Optional[Any] = None
        self._audit_trail: Optional[Any] = None

    # -------------------------------------------------------------------------
    # Initialisation & Lifecycle
    # -------------------------------------------------------------------------

    async def initialize(self) -> None:
        """
        Bootstraps the orchestrator.

        - Creates the repository instance and ensures the database schema exists.
        - Registers extractors with the factory (if needed).
        - Performs health checks.

        This method must be called before processing any URLs.
        """
        logger.info("Initialising CyberForge Orchestrator...")

        # 1. Repository setup.
        self._repository = SQLiteRepository(self._db_path)
        await self._repository.initialize()
        await self._repository.health_check()
        logger.info("Storage layer is healthy.")

        logger.info("CyberForge Orchestrator initialised successfully.")

    async def close(self) -> None:
        """Closes the repository and any other resources."""
        if self._repository is not None:
            await self._repository.close()
            self._repository = None

    # -------------------------------------------------------------------------
    # Single URL Processing (with optional repository reuse)
    # -------------------------------------------------------------------------

    async def _process_single_url_with_repo(
        self,
        url: str,
        repo: BaseRepository,
    ) -> tuple[str, Optional[RepositoryId] | Exception]:
        """
        Internal method to process a single URL using an already-open repository.

        Args:
            url: The target URL.
            repo: An already-initialised repository instance.

        Returns:
            A tuple of (url, result) where result is either a RepositoryId
            or an Exception.
        """
        logger.info(f"--- Starting Pipeline for: {url} ---")

        try:
            extractor = ExtractorFactory.create_from_url(url)
        except ExtractionError as e:
            logger.error(f"Failed to get extractor for URL '{url}': {e}")
            return (url, e)

        # Open the extractor context for this URL.
        # Note: For future optimisation, consider reusing extractors per source.
        async with extractor as ext:
            # Phase 1: Extraction
            try:
                logger.debug("Executing Phase 1: Extraction")
                extracted_item = await ext.extract_single(url)
                raw_id = await repo.save_extracted_item(extracted_item)
                logger.info(f"Raw data extracted and saved with ID: {raw_id}")
            except ExtractionError as e:
                logger.error(f"Pipeline halted at Phase 1 (Extraction) for {url}: {e}")
                return (url, e)

            # Phase 2: AI Parsing
            try:
                logger.debug("Executing Phase 2: AI Parsing")
                parsed_writeup = await self._parser_service.parse_text(extracted_item.raw_content)
                logger.info(f"AI Parsing successful. Title: {parsed_writeup.title}")
            except ParsingFailureError as e:
                logger.error(f"Pipeline halted at Phase 2 (Parsing) for {url}: {e}")
                return (url, e)
            except ControlFlowSignal:
                # Propagate signals to the orchestrator.
                raise
            except ProviderError as e:
                logger.error(f"Provider error during AI parsing for {url}: {e}")
                return (url, e)
            except Exception as e:
                # Unexpected errors (should not happen; log and return).
                logger.error(f"Unexpected error during AI parsing for {url}: {e}")
                return (url, e)

            # Phase 3: Storage
            try:
                logger.debug("Executing Phase 3: Storage Integration")
                final_id = await repo.save_parsed_writeup(writeup=parsed_writeup, source_id=raw_id)
                logger.info(f"Pipeline complete! Final Writeup saved with ID: {final_id}")
                return (url, final_id)
            except (StorageConnectionError, DataPersistenceError) as e:
                logger.error(f"Storage error during Phase 3 for {url}: {e}")
                return (url, e)
            except Exception as e:
                logger.error(f"Unexpected error during Phase 3 for {url}: {e}")
                return (url, e)

    async def process_single_url(self, url: str) -> Optional[RepositoryId]:
        """
        Processes a single URL through the full pipeline.

        Args:
            url: The target URL to process.

        Returns:
            The repository ID of the saved `ParsedWriteup`, or `None` on failure.

        Raises:
            ControlFlowSignal: If a signal is raised (e.g., ModelSwitchRequestSignal)
                and not handled; the caller may catch and act accordingly.
        """
        if self._repository is None:
            raise RuntimeError("Orchestrator not initialised. Call `initialize()` first.")

        async with self._repository as repo:
            _, result = await self._process_single_url_with_repo(url, repo)
            if isinstance(result, Exception):
                return None
            return result

    # -------------------------------------------------------------------------
    # Batch Processing (with shared repository context)
    # -------------------------------------------------------------------------

    async def process_batch(self, urls: list[str]) -> dict[str, Any]:
        """
        Processes a batch of URLs concurrently.

        The repository is opened once and reused for all URLs in the batch,
        significantly improving performance.

        Args:
            urls: List of target URLs.

        Returns:
            A summary dictionary containing:
                - total: total number of URLs.
                - succeeded: number of successfully processed URLs.
                - failed: number of failed URLs.
                - results: list of (url, id_or_exception).
        """
        if self._repository is None:
            raise RuntimeError("Orchestrator not initialised. Call `initialize()` first.")

        logger.info(f"Starting batch orchestration for {len(urls)} URLs...")

        async with self._repository as repo:
            tasks = [
                self._process_single_url_with_repo(url, repo)
                for url in urls
            ]
            results = await asyncio.gather(*tasks)

        summary = {
            "total": len(urls),
            "succeeded": 0,
            "failed": 0,
            "results": [],
        }

        for url, result in results:
            if isinstance(result, Exception):
                logger.error(f"Error processing {url}: {result}")
                summary["failed"] += 1
                summary["results"].append((url, result))
            elif result is not None:
                summary["succeeded"] += 1
                summary["results"].append((url, result))
            else:
                summary["failed"] += 1
                summary["results"].append((url, None))

        logger.info(
            f"Batch completed. Successfully processed: {summary['succeeded']}/{summary['total']}"
        )
        return summary
