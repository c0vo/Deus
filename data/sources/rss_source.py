"""
RSS Feed News Source Adapter

Fetches articles from configurable RSS feeds using feedparser.
One class handles multiple feed instances (Reuters, CNBC, WSJ, etc.).
"""

from __future__ import annotations

import asyncio
import calendar
import hashlib
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Optional

import feedparser

from config.logging_config import get_logger
from data.models import NewsArticle
from data.sources.base import NewsSource

log = get_logger(__name__)


class RSSSource(NewsSource):
    """
    Configurable RSS feed adapter.

    Create one instance per feed:
        reuters = RSSSource("reuters", "https://feeds.reuters.com/reuters/businessNews")
        cnbc = RSSSource("cnbc", "https://search.cnbc.com/rs/search/combinedcms/view.xml")
    """

    def __init__(self, feed_name: str, feed_url: str, max_items: int = 30):
        self._name = feed_name
        self._feed_url = feed_url
        self._max_items = max_items

    @property
    def name(self) -> str:
        return self._name

    @property
    def source_type(self) -> str:
        return "rss"

    async def fetch(self) -> list[NewsArticle]:
        """Fetch and parse the RSS feed."""
        try:
            # feedparser is synchronous — run in executor
            loop = asyncio.get_event_loop()
            feed = await loop.run_in_executor(None, feedparser.parse, self._feed_url)

            if feed.bozo and not feed.entries:
                log.warning(
                    "rss.parse_error",
                    source=self._name,
                    error=str(feed.bozo_exception),
                )
                return []

            articles = []
            for entry in feed.entries[: self._max_items]:
                article = self._parse_entry(entry)
                if article:
                    articles.append(article)

            log.info(
                "rss.fetched",
                source=self._name,
                count=len(articles),
            )
            return articles

        except Exception as e:
            log.error("rss.fetch_failed", source=self._name, error=str(e))
            return []

    def _parse_entry(self, entry: dict) -> Optional[NewsArticle]:
        """Parse a single RSS feed entry into a NewsArticle."""
        try:
            # Extract URL
            url = entry.get("link", "")
            if not url:
                return None

            # Generate a stable ID from the URL
            url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
            article_id = f"rss_{self._name}_{url_hash}"

            # Extract headline
            headline = entry.get("title", "").strip()
            if not headline:
                return None

            # Extract summary — try 'summary', then 'description'
            summary = entry.get("summary", entry.get("description", "")).strip()
            # Strip HTML tags from summary (basic cleanup)
            if "<" in summary:
                import re
                summary = re.sub(r"<[^>]+>", "", summary).strip()

            # Parse publication date
            published_at = self._parse_date(entry)

            return NewsArticle(
                id=article_id,
                headline=headline,
                summary=summary[:2000],  # Cap summary length
                source_name=self._name,
                source_type=self.source_type,
                url=url,
                published_at=published_at,
                raw_data={
                    "feed_title": entry.get("title", ""),
                    "feed_link": entry.get("link", ""),
                },
            )
        except Exception as e:
            log.warning(
                "rss.parse_entry_failed",
                source=self._name,
                error=str(e),
            )
            return None

    def _parse_date(self, entry: dict) -> datetime:
        """Parse the publication date from an RSS entry."""
        # Try 'published_parsed' (struct_time from feedparser)
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            try:
                return datetime.fromtimestamp(calendar.timegm(entry.published_parsed), tz=timezone.utc)
            except (ValueError, OverflowError, OSError):
                pass

        # Try 'published' as a raw string
        raw_date = entry.get("published", entry.get("updated", ""))
        if raw_date:
            try:
                parsed = parsedate_to_datetime(raw_date)
                return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                pass

        # Fallback: current time
        return datetime.now(timezone.utc)


# ── Pre-configured RSS feed instances ────────────────────────────────────

def create_default_rss_sources() -> list[RSSSource]:
    """Create the default set of RSS feed sources."""
    return [
        RSSSource(
            "reuters",
            "https://feeds.reuters.com/reuters/businessNews",
        ),
        RSSSource(
            "cnbc",
            "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
        ),
        RSSSource(
            "wsj_markets",
            "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
        ),
        RSSSource(
            "yahoo_finance",
            "https://finance.yahoo.com/news/rssindex",
        ),
        RSSSource(
            "nyt_business",
            "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",
        ),
        RSSSource(
            "google_news_business",
            "https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx6TVdZU0FtVnVHZ0pWVXlnQVAB",
        ),
    ]
