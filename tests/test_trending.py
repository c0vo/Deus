"""Tests for pipeline.trending — the shared trending cache used by /trending and /api/trending."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from tests.conftest import make_llm_response

import pipeline.trending as trending
from pipeline.trending import TRENDING_TTL, get_trending_with_summaries


@pytest.fixture(autouse=True)
def clear_cache():
    """The cache is module-level, so it must not leak between tests."""
    trending._trending_cache.clear()
    yield
    trending._trending_cache.clear()


@pytest.fixture
def db():
    db = MagicMock()
    db.get_top_trending_tickers.return_value = [
        {"ticker": "AAPL", "mention_count": 12, "avg_sentiment": 0.4},
        {"ticker": "TSLA", "mention_count": 8, "avg_sentiment": -0.3},
    ]
    db.get_recent_summaries_for_ticker.return_value = ["Some recent news."]
    db.get_recent_articles_for_ticker.return_value = [{"headline": "An article"}]
    db.log_llm_usage.return_value = None
    return db


@pytest.fixture(autouse=True)
def _model_configured(monkeypatch):
    monkeypatch.setattr("pipeline.trending.is_llm_configured", lambda: True)
    monkeypatch.setattr("pipeline.trending.settings.model_trending", "test/trending")


@pytest.fixture
def no_llm(monkeypatch):
    """No model configured — exercises the cache without a network call."""
    monkeypatch.setattr("pipeline.trending.is_llm_configured", lambda: False)
    yield


class TestCaching:

    async def test_first_call_is_not_cached(self, db, no_llm):
        _, was_cached = await get_trending_with_summaries(db)
        assert was_cached is False

    async def test_second_call_is_served_from_cache(self, db, no_llm):
        await get_trending_with_summaries(db)
        _, was_cached = await get_trending_with_summaries(db)
        assert was_cached is True

    async def test_cache_hit_does_not_hit_the_database(self, db, no_llm):
        await get_trending_with_summaries(db)
        db.get_top_trending_tickers.reset_mock()
        await get_trending_with_summaries(db)
        db.get_top_trending_tickers.assert_not_called()

    async def test_refresh_bypasses_the_cache(self, db, no_llm):
        await get_trending_with_summaries(db)
        _, was_cached = await get_trending_with_summaries(db, refresh=True)
        assert was_cached is False

    async def test_bot_and_api_share_one_cache_entry(self, db, no_llm):
        """Both callers default to hours=24, limit=15, so one warms the other."""
        await get_trending_with_summaries(db, hours=24, limit=15)
        _, was_cached = await get_trending_with_summaries(db, hours=24, limit=15)
        assert was_cached is True

    async def test_different_windows_are_cached_separately(self, db, no_llm):
        await get_trending_with_summaries(db, hours=24)
        _, was_cached = await get_trending_with_summaries(db, hours=48)
        assert was_cached is False

    async def test_expired_entry_is_recomputed(self, db, no_llm):
        await get_trending_with_summaries(db)
        data, ts = trending._trending_cache["24_15"]
        trending._trending_cache["24_15"] = (data, ts - TRENDING_TTL - 1)
        _, was_cached = await get_trending_with_summaries(db)
        assert was_cached is False

    async def test_expired_entries_are_evicted_on_write(self, db, no_llm):
        """/trending takes free-text hours, so stale keys must not accumulate."""
        await get_trending_with_summaries(db, hours=99)
        data, ts = trending._trending_cache["99_15"]
        trending._trending_cache["99_15"] = (data, ts - TRENDING_TTL - 1)
        await get_trending_with_summaries(db, hours=24)
        assert "99_15" not in trending._trending_cache

    async def test_empty_result_is_not_cached(self, db, no_llm):
        """An empty answer usually means a cold database, not a real result."""
        db.get_top_trending_tickers.return_value = []
        data, _ = await get_trending_with_summaries(db)
        assert data == []
        assert trending._trending_cache == {}


class TestPayload:

    async def test_rows_carry_the_api_contract_fields(self, db, no_llm):
        data, _ = await get_trending_with_summaries(db)
        assert set(data[0]) >= {"ticker", "mention_count", "avg_sentiment", "summary", "articles"}

    async def test_has_context_flags_missing_news(self, db, no_llm):
        db.get_recent_summaries_for_ticker.return_value = []
        data, _ = await get_trending_with_summaries(db)
        assert all(r["has_context"] is False for r in data)

    async def test_summary_falls_back_when_llm_unavailable(self, db, no_llm):
        data, _ = await get_trending_with_summaries(db)
        assert data[0]["summary"] == "No AI summary available."

    async def test_ticker_order_is_preserved(self, db, no_llm):
        data, _ = await get_trending_with_summaries(db)
        assert [r["ticker"] for r in data] == ["AAPL", "TSLA"]


class TestSummaries:

    async def test_successful_batch_call_populates_summaries(self, db):
        note = MagicMock(ticker="AAPL", summary="Trending on earnings.")
        response = make_llm_response(text="[]", parsed=[note])

        with patch("pipeline.trending.complete", AsyncMock(return_value=response)),              patch("pipeline.trending.notes_to_dict",
                   return_value={"AAPL": "Trending on earnings."}):
            data, _ = await get_trending_with_summaries(db)

        assert data[0]["summary"] == "Trending on earnings."

    async def test_one_batch_call_covers_every_ticker(self, db):
        mock_complete = AsyncMock(return_value=make_llm_response(text="[]", parsed=[]))

        with patch("pipeline.trending.complete", mock_complete),              patch("pipeline.trending.notes_to_dict", return_value={}):
            await get_trending_with_summaries(db)

        assert mock_complete.await_count == 1

    async def test_llm_failure_degrades_instead_of_raising(self, db):
        with patch("pipeline.trending.complete",
                   AsyncMock(side_effect=RuntimeError("api down"))):
            data, _ = await get_trending_with_summaries(db)

        assert data[0]["summary"] == "No AI summary available."

    async def test_usage_is_logged(self, db):
        with patch("pipeline.trending.complete",
                   AsyncMock(return_value=make_llm_response(text="[]", parsed=[]))),              patch("pipeline.trending.notes_to_dict", return_value={}):
            await get_trending_with_summaries(db)

        db.log_llm_usage.assert_called_once()
        kwargs = db.log_llm_usage.call_args.kwargs
        assert kwargs["operation"] == "trending_summary_batch"
        # Real reported cost, not an estimate from a price table.
        assert kwargs["cost_usd"] == pytest.approx(0.0001)
