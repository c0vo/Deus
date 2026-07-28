"""
Abstract base class for all news sources.

Every source adapter must implement `fetch()` which returns a list of
NewsArticle instances. Sources handle their own errors gracefully —
a failed source returns an empty list, never crashes the pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from data.models import NewsArticle


class NewsSource(ABC):
    """Base class for news source adapters."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this source (e.g., 'reuters', 'finnhub')."""
        ...

    @property
    @abstractmethod
    def source_type(self) -> str:
        """Category: 'rss', 'api', 'social', or 'scrape'."""
        ...

    @abstractmethod
    async def fetch(self) -> list[NewsArticle]:
        """
        Fetch news articles from this source.

        Returns a list of NewsArticle instances. Must never raise —
        return an empty list on any failure.
        """
        ...

    async def enrich(self, articles: list[NewsArticle]) -> None:
        """
        Optional post-deduplication enrichment step.
        Called on articles that were newly inserted into the database.
        Modify the articles in-place.
        """
        pass
