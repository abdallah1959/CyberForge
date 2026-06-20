# cyberforge.py
"""
CyberForge Entry Point (CLI)

This is the main entry point for running the CyberForge pipeline.
It loads configuration, initialises the orchestrator, and processes
a batch of URLs from CTFTime (or other sources).

Usage:
    python -m cyberforge

The pipeline will:
    1. Load settings from environment variables.
    2. Initialise the SQLite database.
    3. Process a list of test URLs.
    4. Print a summary of the results.
    5. Gracefully shut down.
"""

import asyncio
import logging
import sys

from cyberforge.config import settings
from cyberforge.orchestrator import CyberForgeOrchestrator

# Configure logging to see every step in the terminal.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)


async def main() -> None:
    """Main async entry point for the CyberForge pipeline."""
    logger.info("Starting CyberForge Pipeline...")

    # 1. Initialise the orchestrator with settings and database path.
    # NOTE: The orchestrator now accepts `settings` (not `config`).
    orchestrator = CyberForgeOrchestrator(
        settings=settings,
        db_path="cyberforge_data.sqlite"
    )

    # 2. Initialise the database schema and health checks.
    await orchestrator.initialize()

    # 3. List of test URLs from CTFTime.
    test_urls = [
        "https://ctftime.org/writeup/38531",
        "https://ctftime.org/writeup/38466"
    ]

    # 4. Run the pipeline.
    try:
        # NOTE: process_batch() now returns a summary dict.
        summary = await orchestrator.process_batch(test_urls)
        logger.info(f"Pipeline Execution Summary: {summary}")
    except Exception as e:
        logger.error(f"Pipeline crashed: {e}")
    finally:
        # 5. Gracefully close the orchestrator (closes database connection, etc.)
        await orchestrator.close()


if __name__ == "__main__":
    # Run the async event loop.
    asyncio.run(main())
