"""Tests for NewsAggregator — news source fetching, deduplication, and pre-filtering."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timezone

from data.models import NewsArticle
from pipeline.aggregator import NewsAggregator, _has_financial_content


# ── _has_financial_content tests ────────────────────────────────────────────

def _article(headline, summary="", comments=None, source="test_source"):
    return NewsArticle(
        id="fixture",
        headline=headline,
        summary=summary,
        source_name=source,
        source_type="social" if comments else "rss",
        url="https://example.com/fixture",
        published_at=datetime.now(timezone.utc),
        raw_data={"comments": [{"body": c} for c in comments]} if comments else {},
    )


# (headline, summary, should_reach_the_classifier)
#
# The must-keep half guards against over-correction: this filter is the last
# thing standing between a real story and being dropped at ingestion, where
# nothing downstream will notice. Entries marked with a score are drawn from
# articles the ranker actually scored that highly in the production database.
PREFILTER_FIXTURES = [
    # ── must keep ───────────────────────────────────────────────────────
    ("U.S. says it targeted Iranian forces after attacks that killed two American troops",
     "The U.S. military struck Iranian coastal surveillance facilities.", True),      # scored 9.2
    ("Putin details Russia's fuel shortages after Ukrainian drone strikes",
     "Drone attacks have disrupted refining capacity.", True),                        # scored 8.2
    ("SpaceX landed in millions of 401(k)s through index funds", "", True),           # scored 7.5
    ("Fed holds interest rates steady", "The Federal Reserve decision.", True),
    ("CPI comes in hotter than expected", "Inflation accelerated last month.", True),
    ("AAPL earnings preview", "", True),
    ("Why I'm loading up on $GME", "", True),
    ("Nvidia beats on revenue, raises guidance", "", True),
    ("Trump announces new tariffs on imported steel", "", True),
    ("Shareholder approval clears the merger", "", True),

    # ── must drop ───────────────────────────────────────────────────────
    ("A steak dinner tour in Ohio", "", False),
    ("Taylor Swift announces new album", "", False),
    ("Celebrity wedding photos leaked online", "", False),
    ("Best mechanical keyboards under $100", "Looking for suggestions.", False),
    ("Crowded Airport Lounges Are Rolling Out Grab-and-Go Options", "", False),
    ("How to cook the perfect steak every time", "", False),
]


@pytest.mark.parametrize("headline,summary,expected", PREFILTER_FIXTURES)
def test_prefilter_fixtures(headline, summary, expected):
    assert _has_financial_content(_article(headline, summary)) is expected


class TestPrefilterRegressions:
    """The specific defects this filter has had, pinned so they cannot return."""

    def test_ordinary_words_are_not_read_as_tickers(self):
        """
        Uppercasing before matching [A-Z]{2,5} turned STEAK, TOUR and OHIO into
        tickers, so nearly every article passed as financial.
        """
        assert _has_financial_content(_article("A steak dinner tour in Ohio")) is False

    def test_a_bare_dollar_amount_is_not_a_cashtag(self):
        assert _has_financial_content(_article("Best keyboards under $100")) is False

    def test_a_real_cashtag_still_passes(self):
        assert _has_financial_content(_article("Thoughts on $TSLA today?")) is True

    def test_shouty_headline_is_not_a_list_of_tickers(self):
        assert _has_financial_content(_article("THE NEW BEST THING")) is False

    def test_shouty_headline_with_real_content_still_passes(self):
        assert _has_financial_content(_article("FED HOLDS RATES STEADY")) is True

    def test_reddit_ticker_only_in_comments_is_kept(self):
        """
        Enrichment runs before this filter, so comments are available. A post
        whose signal lives only in the replies previously survived by accident,
        on a false positive from the uppercase bug.
        """
        article = _article(
            "What are your thoughts on this one?",
            "Is it still a buy or is it drilling?",
            comments=["I have $SPCE calls at 10!"],
            source="reddit_wallstreetbets",
        )
        assert _has_financial_content(article) is True

    def test_reddit_noise_with_noise_comments_is_dropped(self):
        article = _article(
            "What is the best mechanical keyboard?",
            "Looking for suggestions.",
            comments=["I like MX Browns."],
            source="reddit_investing",
        )
        assert _has_financial_content(article) is False


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
