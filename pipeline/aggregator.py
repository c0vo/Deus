"""
News Aggregator Pipeline

Orchestrates all news sources: fetches from each, deduplicates
against the SQLite database, and stores new articles.
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional

from config.logging_config import get_logger
from config.settings import settings
from data.database import Database
from data.models import NewsArticle
from data.filters import FINANCIAL_KEYWORDS, TICKER_PATTERN, EXCLUDED_WORDS
from data.sources.alpha_vantage_source import AlphaVantageSource
from data.sources.base import NewsSource
from data.sources.finnhub_source import FinnhubSource
from data.sources.hackernews_source import HackerNewsSource
from data.sources.nitter_source import NitterSource
from data.sources.reddit_source import RedditSource
from data.sources.rss_source import create_default_rss_sources

log = get_logger(__name__)


class NewsAggregator:
    """
    Fan-in aggregator across all configured news sources.

    Fetches from every source, deduplicates by URL against the database,
    and inserts only new articles. Returns the list of newly inserted articles.
    """

    # Cap on sources fetched at once. Without it every source starts
    # simultaneously; the RSS list alone is configurable and can grow past the
    # default executor's worker count, which is shared with the LLM call sites.
    MAX_CONCURRENT_FETCHES = 8

    def __init__(self, db: Database, sources: Optional[list[NewsSource]] = None):
        self.db = db
        self.sources = sources or self._create_default_sources()
        # Per-source URL cache to avoid re-fetching recently seen URLs at high frequency
        self._recent_urls: dict[str, dict[str, float]] = {}
        self._url_cache_ttl = 4 * 3600  # 4 hours in seconds
        self._fetch_semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_FETCHES)

    def _create_default_sources(self) -> list[NewsSource]:
        """Instantiate all configured news sources."""
        sources: list[NewsSource] = []

        # RSS feeds (always available, no API keys)
        sources.extend(create_default_rss_sources())

        # Finnhub (optional, requires API key)
        sources.append(FinnhubSource())

        # Alpha Vantage (optional, requires API key)
        sources.append(AlphaVantageSource())

        # Reddit (no API key, uses public JSON)
        sources.append(RedditSource())

        # Nitter / X-Twitter (no API key, scrapes public instances)
        sources.append(NitterSource())

        # Hacker News (no API key, public Firebase API)
        sources.append(HackerNewsSource())

        return sources

    async def fetch_all(self) -> list[NewsArticle]:
        """
        Fetch from all sources, deduplicate, and store new articles.

        Returns the list of newly inserted articles (not duplicates).
        """
        log.info(
            "aggregator.starting",
            source_count=len(self.sources),
            source_names=[s.name for s in self.sources],
        )

        async def fetch_and_store(source: NewsSource) -> list[NewsArticle]:
            try:
                articles = await self._safe_fetch(source)
                
                # Deduplicate without inserting
                new_articles_to_enrich = []
                source_name = source.name
                source_cache = self._recent_urls.get(source_name, {})
                now = time.time()
                # Clean stale cache entries for this source
                source_cache = {k: v for k, v in source_cache.items() if now - v < self._url_cache_ttl}
                seen_urls = set()
                for article in articles:
                    if article.url in seen_urls:
                        continue
                    seen_urls.add(article.url)
                    # Check in-memory cache first (faster than DB query)
                    if article.url in source_cache:
                        continue
                    if self.db.url_exists(article.url):
                        # Add to cache so we skip it on future cycles
                        source_cache[article.url] = now
                        continue
                    new_articles_to_enrich.append(article)
                self._recent_urls[source_name] = source_cache
                
                # Enrich unique new articles
                if hasattr(source, "enrich"):
                    await source.enrich(new_articles_to_enrich)
                    
                # Store them
                new_articles = []
                for article in new_articles_to_enrich:
                    # Pre-filter: mark non-financial articles as noise immediately
                    # to skip expensive LLM classification
                    if not _has_financial_content(article):
                        article.event_type = "noise"
                    if self.db.insert_article(article):
                        new_articles.append(article)
                
                log.info(
                    "aggregator.source_complete",
                    source=source.name,
                    fetched=len(articles),
                    inserted=len(new_articles)
                )
                return new_articles
            except Exception as e:
                log.error("aggregator.source_failed", source=source.name, error=str(e))
                return []

        tasks = [asyncio.create_task(fetch_and_store(source)) for source in self.sources]
        
        all_new_articles: list[NewsArticle] = []
        for coro in asyncio.as_completed(tasks):
            try:
                new_arts = await coro
                all_new_articles.extend(new_arts)
            except Exception as e:
                log.error("aggregator.task_failed", error=str(e))

        log.info(
            "aggregator.complete",
            total_inserted=len(all_new_articles)
        )

        return all_new_articles

    async def _safe_fetch(self, source: NewsSource) -> list[NewsArticle]:
        """Fetch from a single source with error handling."""
        async with self._fetch_semaphore:
            try:
                return await source.fetch()
            except Exception as e:
                log.error(
                    "aggregator.source_failed",
                    source=source.name,
                    error=str(e),
                )
                return []



    def get_source_health(self) -> dict[str, str]:
        """
        Check which sources are configured and available.

        Returns a dict of source_name -> status string.
        """
        health = {}
        for source in self.sources:
            if isinstance(source, FinnhubSource):
                health[source.name] = (
                    "configured" if settings.has_key("finnhub_api_key") else "no API key"
                )
            elif isinstance(source, AlphaVantageSource):
                health[source.name] = (
                    "configured"
                    if settings.has_key("alpha_vantage_api_key")
                    else "no API key"
                )
            elif isinstance(source, NitterSource):
                health[source.name] = (
                    f"{len(settings.nitter_instance_list)} instances configured"
                )
            elif isinstance(source, RedditSource):
                health[source.name] = (
                    f"{len(settings.reddit_subreddit_list)} subreddits"
                )
            else:
                health[source.name] = "ready"
        return health


def _has_financial_content(article: NewsArticle) -> bool:
    """
    Quick pre-filter: check if an article has financial keywords or ticker patterns.
    Returns True if likely financial (should be LLM-classified), False if noise.
    """
    text = f"{article.headline} {article.summary or ''}".upper()

    # Check for $TICKER pattern
    if "$" in article.headline or "$" in (article.summary or ""):
        return True

    # Check for ticker pattern with excluded words filter
    for match in TICKER_PATTERN.findall(text):
        if match not in EXCLUDED_WORDS:
            return True

    # Check for financial keywords
    text_lower = text.lower()
    for kw in FINANCIAL_KEYWORDS:
        if kw in text_lower:
            return True

    return False
