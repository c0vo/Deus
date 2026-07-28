"""
Hacker News Source Adapter

Fetches top stories from the Hacker News Firebase API and filters
for finance/market/tech-related content. No API key needed.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional

import httpx

from config.logging_config import get_logger
from data.models import NewsArticle
from data.sources.base import NewsSource

log = get_logger(__name__)

HN_API_BASE = "https://hacker-news.firebaseio.com/v0"

# Keywords to filter HN stories for market relevance
FINANCE_KEYWORDS = {
    # Markets & finance
    "stock", "stocks", "market", "trading", "investor", "investment",
    "ipo", "earnings", "revenue", "profit", "valuation", "nasdaq",
    "s&p", "dow", "bull", "bear", "recession", "inflation", "deflation",
    "interest rate", "fed", "federal reserve", "treasury", "bond",
    "dividend", "hedge fund", "private equity", "venture capital",
    "fintech", "crypto", "bitcoin", "ethereum", "blockchain",
    # Major companies
    "apple", "google", "microsoft", "amazon", "nvidia", "meta",
    "tesla", "openai", "anthropic", "deepmind",
    # Tech/Economy
    "layoff", "layoffs", "acquisition", "merger", "antitrust",
    "regulation", "sec", "ftc", "tariff", "trade war", "sanctions",
    "gdp", "unemployment", "economy", "economic",
    # AI / Tech sector
    "ai", "artificial intelligence", "machine learning", "llm",
    "semiconductor", "chip", "gpu",
}


class HackerNewsSource(NewsSource):
    """
    Fetches top stories from Hacker News, filtered for market relevance.

    Uses the official Firebase API (free, no auth, structured JSON).
    """

    def __init__(self, max_stories: int = 30, top_n_ids: int = 60):
        self._max_stories = max_stories
        # Fetch more IDs than we need since we filter by keywords
        self._top_n_ids = top_n_ids

    @property
    def name(self) -> str:
        return "hackernews"

    @property
    def source_type(self) -> str:
        return "social"

    async def fetch(self) -> list[NewsArticle]:
        """Fetch top HN stories and filter for finance/market relevance."""
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                # Step 1: Get top story IDs
                resp = await client.get(f"{HN_API_BASE}/topstories.json")
                resp.raise_for_status()
                story_ids = resp.json()[: self._top_n_ids]

                # Step 2: Fetch each story (concurrently, bounded)
                tasks = [
                    self._fetch_story(client, sid) for sid in story_ids
                ]
                stories = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter for market-relevant stories
            articles = []
            for story in stories:
                if isinstance(story, NewsArticle):
                    if self._is_relevant(story):
                        articles.append(story)
                        if len(articles) >= self._max_stories:
                            break

            log.info(
                "hackernews.fetched",
                total_checked=len(story_ids),
                relevant=len(articles),
            )
            return articles

        except Exception as e:
            log.error("hackernews.fetch_failed", error=str(e))
            return []

    async def _fetch_story(
        self, client: httpx.AsyncClient, story_id: int
    ) -> Optional[NewsArticle]:
        """Fetch a single HN story by ID."""
        try:
            resp = await client.get(f"{HN_API_BASE}/item/{story_id}.json")
            resp.raise_for_status()
            item = resp.json()

            if not item or item.get("type") != "story":
                return None

            title = item.get("title", "").strip()
            url = item.get("url", "")

            if not title:
                return None

            # For self-posts (Ask HN, Show HN), use the HN link
            if not url:
                url = f"https://news.ycombinator.com/item?id={story_id}"

            published_at = datetime.fromtimestamp(
                item.get("time", 0), tz=timezone.utc
            )

            return NewsArticle(
                id=f"hn_{story_id}",
                headline=f"[HN] {title}",
                summary=item.get("text", "")[:2000] if item.get("text") else "",
                source_name=self.name,
                source_type=self.source_type,
                url=url,
                published_at=published_at,
                raw_data={
                    "hn_id": story_id,
                    "score": item.get("score", 0),
                    "num_comments": item.get("descendants", 0),
                    "author": item.get("by", ""),
                    "domain": self._extract_domain(url),
                },
            )
        except Exception:
            return None

    def _is_relevant(self, article: NewsArticle) -> bool:
        """Check if an article is relevant to finance/markets."""
        text = f"{article.headline} {article.summary}".lower()
        return any(kw in text for kw in FINANCE_KEYWORDS)

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL for display."""
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc
        except Exception:
            return ""
