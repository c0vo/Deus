"""
Reddit News Source Adapter

Fetches posts from Reddit using the public RSS endpoints to avoid 403 Blocked errors.
Uses www.reddit.com/r/{subreddit}/hot/.rss
"""

from __future__ import annotations

import hashlib
import calendar
from datetime import datetime, timezone
from typing import Optional
import re
from data.filters import FINANCIAL_KEYWORDS, REDDIT_KEYWORDS, TICKER_PATTERN

import feedparser
import httpx
from bs4 import BeautifulSoup

from config.logging_config import get_logger
from config.settings import settings
from data.models import NewsArticle
from data.sources.base import NewsSource

log = get_logger(__name__)

# Respectful User-Agent
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ProjectScrooge/2.0"

class RedditSource(NewsSource):
    """
    Fetches hot posts from configured subreddits via Reddit's public RSS API.
    """

    def __init__(
        self,
        subreddits: Optional[list[str]] = None,
        max_posts_per_sub: int = 25,
    ):
        self._subreddits = subreddits or settings.reddit_subreddit_list
        self._max_posts = max_posts_per_sub

    @property
    def name(self) -> str:
        return "reddit"

    @property
    def source_type(self) -> str:
        return "social"

    async def fetch(self) -> list[NewsArticle]:
        """Fetch hot posts from all configured subreddits."""
        if not self._subreddits:
            log.info("reddit.skipped", reason="No subreddits configured")
            return []

        all_articles: list[NewsArticle] = []

        import asyncio
        loop = asyncio.get_running_loop()

        for sub in self._subreddits:
            url = f"https://www.reddit.com/r/{sub}/hot/.rss"
            
            async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
                def fetch_rss():
                    # feedparser.parse can take custom headers via agent
                    return feedparser.parse(url, agent=USER_AGENT)
                    
                try:
                    feed = await loop.run_in_executor(None, fetch_rss)
                    if feed.bozo and hasattr(feed, 'status') and feed.status == 429:
                        log.warning("reddit.rate_limited", subreddit=sub)
                        continue
                        
                    articles = []
                    for entry in feed.entries[:self._max_posts]:
                        article = self._parse_post(entry, sub)
                        if article:
                            articles.append(article)
                            # Comment fetching moved to enrich()
                    all_articles.extend(articles)
                except Exception as e:
                    log.error("reddit.fetch_failed", subreddit=sub, error=str(e))

        log.info(
            "reddit.fetched",
            total=len(all_articles),
            subreddits=self._subreddits,
        )
        return all_articles

    def _parse_post(self, entry: dict, subreddit: str) -> Optional[NewsArticle]:
        """Parse a single Reddit RSS entry into a NewsArticle."""
        try:
            title = entry.get("title", "").strip()
            if not title:
                return None

            url = entry.get("link", "")
            if not url:
                return None

            # Generate stable ID
            reddit_id = entry.get("id", url)
            url_hash = hashlib.md5(reddit_id.encode("utf-8")).hexdigest()[:12]
            article_id = f"reddit_{subreddit}_{url_hash}"

            # parse date
            if "published_parsed" in entry and entry.published_parsed:
                dt = datetime.fromtimestamp(calendar.timegm(entry.published_parsed), tz=timezone.utc)
            else:
                dt = datetime.now(timezone.utc)

            summary = entry.get("summary", "")[:1500]

            return NewsArticle(
                id=article_id,
                headline=f"[r/{subreddit}] {title}",
                summary=summary,
                source_name=f"reddit_{subreddit}",
                source_type=self.source_type,
                url=url,
                published_at=dt,
                raw_data={"subreddit": subreddit}
            )
        except Exception as e:
            log.warning("reddit.parse_failed", subreddit=subreddit, error=str(e))
            return None

    async def _fetch_comments_html(self, client: httpx.AsyncClient, post_url: str, limit: int = 20) -> list[dict]:
        """Fetch top comments for a specific post by scraping old.reddit.com HTML."""
        old_url = post_url.replace("www.reddit.com", "old.reddit.com")
        
        try:
            resp = await client.get(old_url, headers={"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"})
            if resp.status_code != 200:
                log.warning("reddit.html_failed", url=old_url, status=resp.status_code)
                return []
                
            soup = BeautifulSoup(resp.text, 'html.parser')
            comments = []
            
            comment_divs = soup.select('div.comment')
            for div in comment_divs[:limit]:
                author_tag = div.select_one('a.author')
                author = author_tag.text if author_tag else "[deleted]"
                
                body_div = div.select_one('div.md')
                if not body_div:
                    continue
                body = body_div.get_text(separator=' ', strip=True)
                
                if author == "AutoModerator" or not body:
                    continue
                    
                comments.append({"author": author, "body": body})
                
            return comments
            
        except Exception as e:
            log.error("reddit.comments_failed", url=old_url, error=str(e))
            return []

    def _should_fetch_comments(self, article: NewsArticle) -> bool:
        combined_text = f"{article.headline} {article.summary}"
        
        # 1. Ticker symbols
        if TICKER_PATTERN.search(combined_text):
            return True
            
        # 2. Financial/Reddit keywords
        combined_text_lower = combined_text.lower()
        all_keywords = FINANCIAL_KEYWORDS | REDDIT_KEYWORDS
        for keyword in all_keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', combined_text_lower):
                return True
                
        return False

    async def enrich(self, articles: list[NewsArticle]) -> None:
        """Fetch comments only for new articles that pass the heuristic pre-filter."""
        if not articles:
            return
            
        import asyncio
        async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
            for article in articles:
                if self._should_fetch_comments(article):
                    comments = await self._fetch_comments_html(client, article.url)
                    article.raw_data["comments"] = comments
                    await asyncio.sleep(0.8)  # respect rate limits
                else:
                    article.raw_data["comments"] = []
                    log.debug("reddit.skipped_comments", url=article.url, reason="failed pre-filter")
