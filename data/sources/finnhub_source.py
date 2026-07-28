"""
Finnhub News Source Adapter

Fetches general market news from the Finnhub API.
Requires FINNHUB_API_KEY in .env — gracefully skips if missing.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Optional

import httpx

from config.logging_config import get_logger
from config.settings import settings
from data.models import NewsArticle
from data.sources.base import NewsSource
from data.filters import FINANCIAL_KEYWORDS, TICKER_PATTERN, EXCLUDED_WORDS

log = get_logger(__name__)

FINNHUB_NEWS_URL = "https://finnhub.io/api/v1/news"


class FinnhubSource(NewsSource):
    """Fetches general market news from Finnhub."""

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = settings.finnhub_api_key if api_key is None else api_key

    @property
    def name(self) -> str:
        return "finnhub"

    @property
    def source_type(self) -> str:
        return "api"

    async def fetch(self) -> list[NewsArticle]:
        """Fetch general news from Finnhub API."""
        if not self._api_key:
            log.info("finnhub.skipped", reason="No API key configured")
            return []

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    FINNHUB_NEWS_URL,
                    params={
                        "category": "general",
                        "token": self._api_key,
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            articles = []
            for item in data:
                article = self._parse_item(item)
                if article and self._is_relevant(article):
                    articles.append(article)
                    if len(articles) >= 20:  # Cap at 20 high quality articles
                        break

            log.info("finnhub.fetched", count=len(articles))
            return articles

        except httpx.HTTPStatusError as e:
            log.error(
                "finnhub.http_error",
                status=e.response.status_code,
                detail=str(e),
            )
            return []
        except Exception as e:
            log.error("finnhub.fetch_failed", error=str(e))
            return []

    def _is_relevant(self, article: NewsArticle) -> bool:
        """Filter out non-financial or low-value articles."""
        combined_text = f"{article.headline} {article.summary}"
        
        # 1. Has ticker mentions?
        words = set(TICKER_PATTERN.findall(combined_text))
        if words - EXCLUDED_WORDS:
            return True
            
        # 2. Has general financial keywords?
        combined_text_lower = combined_text.lower()
        if any(keyword in combined_text_lower for keyword in FINANCIAL_KEYWORDS):
            return True
            
        return False

    def _parse_item(self, item: dict) -> Optional[NewsArticle]:
        """Parse a single Finnhub news item."""
        try:
            url = item.get("url", "")
            headline = item.get("headline", "").strip()

            if not url or not headline:
                return None

            # Finnhub provides Unix timestamp
            timestamp = item.get("datetime", 0)
            published_at = datetime.fromtimestamp(timestamp, tz=timezone.utc)

            # Finnhub item IDs
            finnhub_id = item.get("id", "")
            url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()[:12]
            article_id = f"finnhub_{finnhub_id}" if finnhub_id else f"finnhub_{url_hash}"

            return NewsArticle(
                id=article_id,
                headline=headline,
                summary=item.get("summary", "")[:2000],
                source_name=self.name,
                source_type=self.source_type,
                url=url,
                published_at=published_at,
                raw_data={
                    "category": item.get("category", ""),
                    "source": item.get("source", ""),
                    "related": item.get("related", ""),
                    "image": item.get("image", ""),
                },
            )
        except Exception as e:
            log.warning("finnhub.parse_failed", error=str(e))
            return None
