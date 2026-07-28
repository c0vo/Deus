"""
Nitter (X/Twitter) News Source Adapter

Scrapes tweets from public Nitter instances to follow specific accounts.
No API key needed. Handles failover across multiple Nitter mirror instances.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Optional

import httpx
from bs4 import BeautifulSoup

from config.logging_config import get_logger
from config.settings import settings
from data.models import NewsArticle
from data.sources.base import NewsSource

log = get_logger(__name__)


class NitterSource(NewsSource):
    """
    Scrapes tweets from Nitter instances for configured X/Twitter accounts.

    Nitter instances are tried in order — if one fails, the next is attempted.
    All instances may be down; the source gracefully returns an empty list.
    """

    def __init__(
        self,
        accounts: Optional[list[str]] = None,
        instances: Optional[list[str]] = None,
        max_tweets_per_account: int = 15,
    ):
        self._accounts = accounts or settings.nitter_account_list
        self._instances = instances or settings.nitter_instance_list
        self._max_tweets = max_tweets_per_account

    @property
    def name(self) -> str:
        return "nitter"

    @property
    def source_type(self) -> str:
        return "scrape"

    async def fetch(self) -> list[NewsArticle]:
        """Fetch tweets from all configured accounts via Nitter."""
        if not self._accounts:
            log.info("nitter.skipped", reason="No accounts configured")
            return []

        if not self._instances:
            log.info("nitter.skipped", reason="No Nitter instances configured")
            return []

        all_articles: list[NewsArticle] = []

        async with httpx.AsyncClient(
            timeout=12.0,
            follow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            },
        ) as client:
            for account in self._accounts:
                tweets = await self._fetch_account(client, account)
                all_articles.extend(tweets)

        log.info(
            "nitter.fetched",
            total=len(all_articles),
            accounts=self._accounts,
        )
        return all_articles

    async def _fetch_account(
        self, client: httpx.AsyncClient, account: str
    ) -> list[NewsArticle]:
        """Try each Nitter instance until one succeeds for this account."""
        for instance in self._instances:
            try:
                url = f"https://{instance}/{account}"
                resp = await client.get(url)

                if resp.status_code == 200:
                    tweets = self._parse_nitter_page(
                        resp.text, account, instance
                    )
                    if tweets:
                        return tweets[: self._max_tweets]

                log.debug(
                    "nitter.instance_failed",
                    instance=instance,
                    account=account,
                    status=resp.status_code,
                )
            except Exception as e:
                log.debug(
                    "nitter.instance_error",
                    instance=instance,
                    account=account,
                    error=str(e),
                )
                continue

        log.warning(
            "nitter.all_instances_failed",
            account=account,
            instances_tried=len(self._instances),
        )
        return []

    def _parse_nitter_page(
        self, html: str, account: str, instance: str
    ) -> list[NewsArticle]:
        """Parse tweets from a Nitter profile page."""
        try:
            soup = BeautifulSoup(html, "html.parser")
            articles = []

            # Nitter uses .timeline-item for each tweet
            tweet_divs = soup.select(".timeline-item")
            if not tweet_divs:
                # Some instances use different selectors
                tweet_divs = soup.select(".tweet-body")

            for tweet_div in tweet_divs:
                article = self._parse_tweet(tweet_div, account, instance)
                if article:
                    articles.append(article)

            return articles
        except Exception as e:
            log.warning(
                "nitter.parse_page_failed",
                account=account,
                error=str(e),
            )
            return []

    def _parse_tweet(
        self, tweet_div, account: str, instance: str
    ) -> Optional[NewsArticle]:
        """Parse a single tweet div into a NewsArticle."""
        try:
            # Extract tweet text
            content_div = tweet_div.select_one(".tweet-content, .media-body")
            if not content_div:
                return None

            text = content_div.get_text(strip=True)
            if not text or len(text) < 10:
                return None

            # Extract tweet link (for unique URL)
            link_el = tweet_div.select_one(".tweet-link, .tweet-date a")
            tweet_path = ""
            if link_el and link_el.get("href"):
                tweet_path = link_el["href"]

            # Build canonical URL (pointing to X/Twitter, not Nitter)
            if tweet_path:
                tweet_url = f"https://x.com{tweet_path}"
            else:
                # Fallback: generate from hash
                text_hash = hashlib.md5(text.encode()).hexdigest()[:12]
                tweet_url = f"https://x.com/{account}/status/{text_hash}"

            # Generate stable ID
            url_hash = hashlib.md5(tweet_url.encode()).hexdigest()[:12]
            article_id = f"nitter_{account}_{url_hash}"

            # Extract date
            date_el = tweet_div.select_one(".tweet-date a, time")
            published_at = self._parse_tweet_date(date_el)

            # Use first ~100 chars as headline, full text as summary
            headline = f"@{account}: {text[:120]}{'...' if len(text) > 120 else ''}"

            return NewsArticle(
                id=article_id,
                headline=headline,
                summary=text[:2000],
                source_name=f"nitter_{account}",
                source_type=self.source_type,
                url=tweet_url,
                published_at=published_at,
                raw_data={
                    "account": account,
                    "instance": instance,
                    "full_text": text,
                },
            )
        except Exception as e:
            log.warning(
                "nitter.parse_tweet_failed",
                account=account,
                error=str(e),
            )
            return None

    def _parse_tweet_date(self, date_el) -> datetime:
        """Parse date from a Nitter date element."""
        if date_el:
            # Try title attribute (often has ISO format)
            title = date_el.get("title", "")
            if title:
                for fmt in (
                    "%b %d, %Y · %I:%M %p %Z",
                    "%Y-%m-%dT%H:%M:%S",
                    "%d/%m/%Y, %H:%M:%S",
                ):
                    try:
                        return datetime.strptime(title, fmt).replace(
                            tzinfo=timezone.utc
                        )
                    except ValueError:
                        continue

            # Try datetime attribute on <time> elements
            dt_attr = date_el.get("datetime", "")
            if dt_attr:
                try:
                    return datetime.fromisoformat(dt_attr.replace("Z", "+00:00"))
                except ValueError:
                    pass

        return datetime.now(timezone.utc)
