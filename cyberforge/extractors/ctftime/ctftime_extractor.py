# cyberforge/extractors/ctftime/ctftime_extractor.py
"""
CTFTime Extractor Implementation.

Responsible for asynchronously scraping vulnerability writeups from ctftime.org.
Inherits the HTML noise-reduction logic from legacy CyberHub for token efficiency,
but upgrades the networking layer to non-blocking aiohttp and completely decouples
AI parsing and storage logic.

Future Enhancement:
    To avoid code duplication across multiple HTML extractors (HackerOne, ExploitDB),
    this class should inherit from a common BaseHttpExtractor that provides:
        - Shared ClientSession lifecycle.
        - Standardised retry logic with exponential backoff.
        - Rate limiting and header management.
        - Unified health check semantics.
    This will be introduced when a second HTTP-based extractor is added.
"""

import asyncio
import logging
import random
from datetime import datetime
from typing import AsyncIterator
from urllib.parse import urlparse

import aiohttp
from bs4 import BeautifulSoup

from cyberforge.core.exceptions import ExtractionError
from cyberforge.extractors.base_extractor import (
    BaseExtractor,
    ExtractorContext,
    ExtractionOptions,
    SourceType,
)
from cyberforge.schemas.extracted_item import ExtractedItem

logger = logging.getLogger(__name__)


class CTFTimeExtractor(BaseExtractor):
    """
    Concrete async extractor for CTFTime writeups.

    Features:
        - Automated HTML noise reduction for LLM-ready text.
        - Safe concurrency limits (max 5 simultaneous requests).
        - Persistent ClientSession for connection reuse.
        - Retry logic with exponential backoff.
        - Rate limiting with optional per-second limit.
        - Proper published date extraction when available.

    Example:
        >>> extractor = CTFTimeExtractor()
        >>> async with extractor:
        ...     item = await extractor.extract_single(
        ...         "https://ctftime.org/writeup/12345"
        ...     )
        ...     async for item in extractor.extract_batch([...]):
        ...         print(item)
    """

    BASE_URL = "https://ctftime.org"
    DEFAULT_USER_AGENT = "CyberForge-DataEngine/2.0 (Enterprise Scraper)"
    DEFAULT_MAX_LENGTH = 120_000

    def __init__(self) -> None:
        """Initialises the CTFTime extractor."""
        self._session: aiohttp.ClientSession | None = None

    @property
    def source_name(self) -> str:
        return "CTFTime"

    @property
    def source_type(self) -> SourceType:
        return SourceType.HTML

    # -------------------------------------------------------------------------
    # Lifecycle Management
    # -------------------------------------------------------------------------

    async def __aenter__(self) -> "CTFTimeExtractor":
        """Async context manager entry - creates the shared session."""
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit - closes the shared session."""
        await self.close()

    async def close(self) -> None:
        """Close the shared ClientSession if open."""
        if self._session is not None:
            await self._session.close()
            self._session = None
            logger.debug("CTFTimeExtractor session closed.")

    async def _ensure_session(self) -> None:
        """Ensure a shared ClientSession exists."""
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=30.0)
            self._session = aiohttp.ClientSession(timeout=timeout)
            logger.debug("CTFTimeExtractor session created.")

    def _build_headers(self, context: ExtractorContext | None) -> dict[str, str]:
        """Constructs headers to mimic legitimate traffic."""
        user_agent = (
            context.user_agent if context and context.user_agent else self.DEFAULT_USER_AGENT
        )
        return {
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
        }

    # -------------------------------------------------------------------------
    # Core Extraction Methods
    # -------------------------------------------------------------------------

    async def extract_single(
        self,
        url: str,
        context: ExtractorContext | None = None,
        options: ExtractionOptions | None = None,
    ) -> ExtractedItem:
        """
        Extracts a single CTFTime writeup from the given URL.

        Args:
            url: The CTFTime writeup URL.
            context: Optional runtime configuration.
            options: Optional extraction options.

        Returns:
            An ExtractedItem containing the cleaned text.

        Raises:
            ExtractionError: If extraction fails.
        """
        # Validate URL.
        parsed = urlparse(url)
        if not parsed.netloc.endswith("ctftime.org"):
            raise ExtractionError(f"URL domain mismatch. Expected ctftime.org, got: {url}")

        logger.info(f"Extracting writeup from: {url}")

        await self._ensure_session()
        # Satisfy type checker: session is guaranteed non-None after _ensure_session.
        assert self._session is not None
        html = await self._fetch_html_with_retry(url, context)
        clean_text = self._clean_and_extract_text(html)

        if not clean_text:
            raise ExtractionError(f"Extraction yielded empty text for URL: {url}")

        # Try to extract the actual publication date.
        published_date = self._extract_published_date(html)

        return ExtractedItem(
            source_name=self.source_name,
            url=url,
            raw_content=clean_text,
            author=None,  # CTFTime doesn't prominently display authors.
            published_date=published_date,
        )

    async def extract_batch(
        self,
        urls: list[str],
        context: ExtractorContext | None = None,
        options: ExtractionOptions | None = None,
    ) -> AsyncIterator[ExtractedItem]:
        """
        Asynchronously yields extracted items from a list of URLs.

        Uses a shared session and semaphore for concurrency control.

        Args:
            urls: A list of CTFTime writeup URLs.
            context: Optional runtime configuration.
            options: Optional extraction options.

        Yields:
            ExtractedItem models as they are successfully retrieved.

        Raises:
            ExtractionError: If a critical failure occurs.
        """
        logger.info(f"Initiating batch extraction for {len(urls)} URLs from {self.source_name}.")

        await self._ensure_session()
        assert self._session is not None

        # Max concurrent requests.
        semaphore = asyncio.Semaphore(5)

        # Rate limiting: token bucket approximation.
        rate_limit = context.rate_limit_per_second if context else None
        rate_limiter = None
        if rate_limit:
            # Simple rate limiter: allows up to rate_limit requests per second.
            rate_limiter = asyncio.Semaphore(1)
            delay = 1.0 / rate_limit

        async def _bounded_extract(url: str) -> ExtractedItem | None:
            async with semaphore:
                if rate_limiter:
                    async with rate_limiter:
                        await asyncio.sleep(delay)
                try:
                    return await self.extract_single(url, context, options)
                except ExtractionError as e:
                    logger.error(f"Failed to extract {url}: {e}")
                    return None

        tasks = [asyncio.create_task(_bounded_extract(url)) for url in urls]

        for task in asyncio.as_completed(tasks):
            result = await task
            if result is not None:
                yield result

    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------

    async def _fetch_html_with_retry(
        self,
        url: str,
        context: ExtractorContext | None = None,
    ) -> str:
        """
        Fetch HTML with retry logic and exponential backoff.

        Uses max_retries from context if provided.
        """
        max_retries = context.max_retries if context else 3
        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                return await self._fetch_html(url, context)
            except ExtractionError as e:
                last_error = e
                if attempt < max_retries:
                    # Exponential backoff with jitter.
                    delay = (2 ** attempt) * random.uniform(0.5, 1.5)
                    logger.warning(
                        f"Retry {attempt + 1}/{max_retries} for {url} "
                        f"after {delay:.2f}s: {e}"
                    )
                    await asyncio.sleep(delay)

        raise ExtractionError(
            f"Failed to fetch {url} after {max_retries + 1} attempts. "
            f"Last error: {last_error}"
        ) from last_error

    async def _fetch_html(self, url: str, context: ExtractorContext | None) -> str:
        """Asynchronously fetches HTML content using the shared session."""
        timeout_seconds = context.timeout if context else 30.0
        headers = self._build_headers(context)

        await self._ensure_session()
        assert self._session is not None
        session = self._session

        try:
            # Use per-request timeout instead of mutating the session.
            timeout = aiohttp.ClientTimeout(total=timeout_seconds)
            async with session.get(url, headers=headers, timeout=timeout) as response:
                if response.status == 429:
                    raise ExtractionError(f"Rate limited (HTTP 429) by CTFTime at {url}")
                if response.status in (403, 404, 500, 502, 503, 504):
                    raise ExtractionError(f"HTTP {response.status} Error fetching URL: {url}")

                response.raise_for_status()
                return await response.text()

        except asyncio.TimeoutError as exc:
            raise ExtractionError(f"Timeout ({timeout_seconds}s) fetching {url}") from exc
        except aiohttp.ClientError as exc:
            raise ExtractionError(f"Network error fetching {url}: {exc}") from exc

    def _clean_and_extract_text(self, html_content: str) -> str:
        """
        Legacy Noise-Reduction: Removes scripts, navs, and footers before
        extracting text to ensure high signal-to-noise ratio for LLMs.
        """
        soup = BeautifulSoup(html_content, "html.parser")

        # Strip noisy elements.
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()

        # Extract purely textual content.
        text = soup.get_text(separator="\n", strip=True)

        if len(text) > self.DEFAULT_MAX_LENGTH:
            logger.warning(
                f"Extracted text exceeded {self.DEFAULT_MAX_LENGTH} chars. Truncating."
            )
            text = text[: self.DEFAULT_MAX_LENGTH]

        return text

    def _extract_published_date(self, html_content: str) -> str | None:
        """
        Attempts to extract the actual publication date from the HTML.

        CTFTime often has dates in specific meta tags or elements.
        Returns ISO 8601 formatted date string if possible, otherwise None.
        """
        soup = BeautifulSoup(html_content, "html.parser")
        date_str = None

        # Try meta tags first.
        for meta in soup.find_all("meta"):
            if meta.get("property") in ("article:published_time", "og:published_time"):
                date_str = meta.get("content")
                break

            if meta.get("name") in ("published_date", "publish_date"):
                date_str = meta.get("content")
                break

        # Try common date patterns in the page.
        if not date_str:
            date_patterns = [
                "time",  # <time> tag
                ".date",
                ".published",
                ".post-date",
            ]

            for pattern in date_patterns:
                element = soup.select_one(pattern)
                if element:
                    text = element.get_text(strip=True)
                    if text:
                        date_str = text
                        break

        # Normalise to ISO 8601 if possible.
        if date_str:
            try:
                # Attempt to parse common date formats.
                for fmt in (
                    "%Y-%m-%dT%H:%M:%S%z",
                    "%Y-%m-%d %H:%M:%S",
                    "%Y-%m-%d",
                    "%b %d, %Y",
                    "%d %b %Y",
                ):
                    try:
                        dt = datetime.strptime(date_str, fmt)
                        return dt.isoformat()
                    except ValueError:
                        continue

                # If it's already ISO 8601, return as is.
                if "T" in date_str or " " in date_str:
                    return date_str
            except ValueError:
                # If parsing fails, return the raw string.
                return date_str

        return None

    # -------------------------------------------------------------------------
    # Health Check
    # -------------------------------------------------------------------------

    async def health_check(self, context: ExtractorContext | None = None) -> None:
        """
        Verifies connectivity to CTFTime and checks for IP blocking.

        Args:
            context: Optional runtime configuration.

        Raises:
            ExtractionError: If the health check fails.
        """
        logger.debug(f"Running health check for {self.source_name}")

        # Build a fast context for health checks.
        if context:
            fast_context = ExtractorContext(
                timeout=5.0,
                max_retries=1,
                user_agent=context.user_agent,
                rate_limit_per_second=context.rate_limit_per_second,
            )
        else:
            fast_context = ExtractorContext(
                timeout=5.0,
                max_retries=1,
            )

        try:
            await self._fetch_html(self.BASE_URL, fast_context)
        except ExtractionError:
            raise
        except asyncio.TimeoutError as e:
            raise ExtractionError(f"{self.source_name} health check timed out.") from e
        except aiohttp.ClientError as e:
            raise ExtractionError(f"{self.source_name} health check failed: {e}") from e
