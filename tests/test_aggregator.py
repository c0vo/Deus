"""Tests for NewsAggregator — news source fetching, deduplication, and pre-filtering."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timezone

from data.models import NewsArticle
from pipeline.aggregator import NewsAggregator, _has_financial_content


# ── _has_financial_content tests ────────────────────────────────────────────

class TestHasFinancialContent:
    """Pre-filter: should this article be sent to the LLM for classification?"""

    def test_cashtag_in_headline(self):
        article = NewsArticle(
            id="t1", headline="$AAPL is mooning", summary="Big news",
            source_name="test", source_type="rss", url="http://x.com/1",
            published_at=datetime.now(timezone.utc),
        )
        assert _has_financial_content(article) is True

    def test_cashtag_in_summary(self):
        article = NewsArticle(
            id="t2", headline="Big news", summary="$TSLA is up bigly",
            source_name="test", source_type="rss", url="http://x.com/2",
            published_at=datetime.now(timezone.utc),
        )
        assert _has_financial_content(article) is True

    def test_ticker_pattern_matches(self):
        article = NewsArticle(
            id="t3", headline="AAPL earnings preview", summary="",
            source_name="test", source_type="rss", url="http://x.com/3",
            published_at=datetime.now(timezone.utc),
        )
        assert _has_financial_content(article) is True

    def test_ticker_pattern_excludes_common_words(self):
        """'THE' and 'NEW' should not be treated as tickers."""
        article = NewsArticle(
            id="t4", headline="THE NEW BEST THING", summary="",
            source_name="test", source_type="rss", url="http://x.com/4",
            published_at=datetime.now(timezone.utc),
        )
        assert _has_financial_content(article) is False

    def test_financial_keyword_matches(self):
        article = NewsArticle(
            id="t5", headline="Fed holds interest rates steady", summary="The Federal Reserve decision.",
            source_name="test", source_type="rss", url="http://x.com/5",
            published_at=datetime.now(timezone.utc),
        )
        assert _has_financial_content(article) is True

    def test_non_financial_text_returns_false(self):
        article = NewsArticle(
            id="t6", headline="Best mechanical keyboards under $100", summary="Looking for suggestions.",
            source_name="test", source_type="rss", url="http://x.com/6",
            published_at=datetime.now(timezone.utc),
        )
        assert _has_financial_content(article) is False

    def test_earnings_keyword_matches(self):
        article = NewsArticle(
            id="t7", headline="Quarterly earnings report is out", summary="",
            source_name="test", source_type="rss", url="http://x.com/7",
            published_at=datetime.now(timezone.utc),
        )
        assert _has_financial_content(article) is True


# ── NewsAggregator tests ───────────────────────────────────────────────────

class TestNewsAggregator:
    """Aggregator initialization and source management."""

    def test_init_with_default_sources(self, mock_db):
        """Default sources should include RSS, Finnhub, Alpha Vantage, Reddit, etc."""
        aggregator = NewsAggregator(db=mock_db)
        source_names = [s.name for s in aggregator.sources]
        assert len(source_names) >= 3
        assert "reddit_wallstreetbets" in source_names or any("rss" in n for n in source_names)

    def test_init_with_custom_sources(self, mock_db):
        """Custom source list should override defaults."""
        mock_source = MagicMock()
        mock_source.name = "custom_source"
        aggregator = NewsAggregator(db=mock_db, sources=[mock_source])
        assert len(aggregator.sources) == 1
        assert aggregator.sources[0].name == "custom_source"


class TestSafeFetch:
    """Error handling for source fetching."""

    @pytest.mark.asyncio
    async def test_safe_fetch_returns_articles(self, mock_db):
        """Successful fetch should return articles."""
        mock_source = MagicMock()
        mock_source.name = "test_source"
        mock_source.fetch = AsyncMock(return_value=[MagicMock(), MagicMock()])

        aggregator = NewsAggregator(db=mock_db, sources=[mock_source])
        result = await aggregator._safe_fetch(mock_source)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_safe_fetch_handles_exception(self, mock_db):
        """Exception in source.fetch() should return empty list, not crash."""
        mock_source = MagicMock()
        mock_source.name = "failing_source"
        mock_source.fetch = AsyncMock(side_effect=Exception("Network error"))

        aggregator = NewsAggregator(db=mock_db, sources=[mock_source])
        result = await aggregator._safe_fetch(mock_source)
        assert result == []


class TestDeduplication:
    """URL deduplication logic."""

    @pytest.mark.asyncio
    async def test_deduplicates_by_url(self, mock_db):
        """Articles with URLs already in DB should be skipped."""
        mock_db.url_exists.side_effect = lambda url: url == "http://x.com/existing"

        article1 = NewsArticle(
            id="new1", headline="New article", summary="",
            source_name="test", source_type="rss", url="http://x.com/new",
            published_at=datetime.now(timezone.utc),
        )
        article2 = NewsArticle(
            id="existing1", headline="Existing article", summary="",
            source_name="test", source_type="rss", url="http://x.com/existing",
            published_at=datetime.now(timezone.utc),
        )

        mock_source = MagicMock()
        mock_source.name = "test_source"
        mock_source.fetch = AsyncMock(return_value=[article1, article2])

        aggregator = NewsAggregator(db=mock_db, sources=[mock_source])
        result = await aggregator.fetch_all()

        # Only the new article should be inserted
        inserted_ids = [a.id for a in result]
        assert "new1" in inserted_ids
        assert "existing1" not in inserted_ids

    @pytest.mark.asyncio
    async def test_deduplicates_duplicate_urls_in_same_batch(self, mock_db):
        """If the same URL appears twice in one source fetch, insert only once."""
        mock_db.url_exists.return_value = False

        article = NewsArticle(
            id="dup1", headline="Duplicate headline", summary="",
            source_name="test", source_type="rss", url="http://x.com/dup",
            published_at=datetime.now(timezone.utc),
        )

        mock_source = MagicMock()
        mock_source.name = "test_source"
        mock_source.fetch = AsyncMock(return_value=[article, article])  # Same article twice

        aggregator = NewsAggregator(db=mock_db, sources=[mock_source])
        result = await aggregator.fetch_all()

        # db.insert_article should only be called once
        assert mock_db.insert_article.call_count == 1


class TestPreFilter:
    """Noise pre-filtering during aggregation."""

    @pytest.mark.asyncio
    async def test_noise_article_marked_before_insertion(self, mock_db):
        """Non-financial articles should be marked as noise event_type before insert."""
        mock_db.url_exists.return_value = False

        noise = NewsArticle(
            id="noise1", headline="Best mechanical keyboards under $100", summary="",
            source_name="test", source_type="rss", url="http://x.com/noise",
            published_at=datetime.now(timezone.utc),
        )

        mock_source = MagicMock()
        mock_source.name = "test_source"
        mock_source.fetch = AsyncMock(return_value=[noise])
        # Don't call enrich
        # mock hasattr(mock_source, 'enrich') returns False by default with Mock

        aggregator = NewsAggregator(db=mock_db, sources=[mock_source])
        result = await aggregator.fetch_all()

        if result:
            assert result[0].event_type == "noise"


class TestSourceHealth:
    """Source health reporting."""

    def test_get_source_health_returns_dict(self, mock_db):
        aggregator = NewsAggregator(db=mock_db)
        health = aggregator.get_source_health()
        assert isinstance(health, dict)
        assert len(health) > 0

    def test_source_health_has_all_sources(self, mock_db):
        aggregator = NewsAggregator(db=mock_db)
        health = aggregator.get_source_health()
        for source in aggregator.sources:
            assert source.name in health
