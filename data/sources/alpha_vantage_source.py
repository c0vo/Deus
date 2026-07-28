"""
Alpha Vantage News Source Adapter

Fetches news with pre-computed sentiment from the Alpha Vantage API.
Requires ALPHA_VANTAGE_API_KEY in .env — gracefully skips if missing.
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

log = get_logger(__name__)

AV_NEWS_URL = "https://www.alphavantage.co/query"


class AlphaVantageSource(NewsSource):
    """Fetches news with sentiment from Alpha Vantage."""

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = settings.alpha_vantage_api_key if api_key is None else api_key

    @property
    def name(self) -> str:
        return "alpha_vantage"

    @property
    def source_type(self) -> str:
        return "api"

    async def fetch(self) -> list[NewsArticle]:
        """Fetch news from Alpha Vantage NEWS_SENTIMENT endpoint."""
        if not self._api_key:
            log.info("alpha_vantage.skipped", reason="No API key configured")
            return []

        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.get(
                    AV_NEWS_URL,
                    params={
                        "function": "NEWS_SENTIMENT",
                        "apikey": self._api_key,
                        "sort": "LATEST",
                        "limit": 50,
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            # Check for API errors (AV returns 200 with error messages)
            if "Error Message" in data or "Note" in data:
                msg = data.get("Error Message", data.get("Note", "Unknown error"))
                log.warning("alpha_vantage.api_error", message=msg)
                return []

            feed = data.get("feed", [])
            articles = []
            for item in feed:
                article = self._parse_item(item)
                # Only keep articles that are explicitly tied to at least one stock ticker
                if article and article.affected_tickers:
                    articles.append(article)
                    if len(articles) >= 20:  # Cap at top 20 high-quality articles
                        break

            log.info("alpha_vantage.fetched", count=len(articles))
            return articles

        except httpx.HTTPStatusError as e:
            log.error(
                "alpha_vantage.http_error",
                status=e.response.status_code,
                detail=str(e),
            )
            return []
        except Exception as e:
            log.error("alpha_vantage.fetch_failed", error=str(e))
            return []

    def _parse_item(self, item: dict) -> Optional[NewsArticle]:
        """Parse a single Alpha Vantage news item."""
        try:
            url = item.get("url", "")
            headline = item.get("title", "").strip()

            if not url or not headline:
                return None

            # Generate stable ID
            url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
            article_id = f"av_{url_hash}"

            # Parse AV timestamp format: "20250615T143000"
            published_at = self._parse_av_date(
                item.get("time_published", "")
            )

            # Extract pre-computed sentiment
            sentiment_score = self._extract_sentiment(item)

            # Extract ticker symbols from ticker_sentiment
            tickers = []
            for ts in item.get("ticker_sentiment", []):
                ticker = ts.get("ticker", "")
                if ticker and not ticker.startswith("CRYPTO:"):
                    tickers.append(ticker)

            return NewsArticle(
                id=article_id,
                headline=headline,
                summary=item.get("summary", "")[:2000],
                source_name=self.name,
                source_type=self.source_type,
                url=url,
                published_at=published_at,
                # AV gives us pre-computed sentiment — store it
                sentiment_score=sentiment_score,
                affected_tickers=tickers[:10],  # Cap at 10 tickers
                raw_data={
                    "source": item.get("source", ""),
                    "category_within_source": item.get("category_within_source", ""),
                    "overall_sentiment_label": item.get("overall_sentiment_label", ""),
                    "topics": [t.get("topic", "") for t in item.get("topics", [])],
                },
            )
        except Exception as e:
            log.warning("alpha_vantage.parse_failed", error=str(e))
            return None

    def _parse_av_date(self, date_str: str) -> datetime:
        """Parse Alpha Vantage date format: '20250615T143000'."""
        if date_str:
            try:
                return datetime.strptime(date_str, "%Y%m%dT%H%M%S").replace(
                    tzinfo=timezone.utc
                )
            except ValueError:
                pass
        return datetime.now(timezone.utc)

    def _extract_sentiment(self, item: dict) -> Optional[float]:
        """Extract overall sentiment score from AV response."""
        try:
            score_str = item.get("overall_sentiment_score", "")
            if score_str:
                return float(score_str)
        except (ValueError, TypeError):
            pass
        return None
