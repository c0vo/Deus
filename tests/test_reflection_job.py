"""Tests for run_reflection_job — watchlist gating, batch bounds, and correction caching.

The reflection job used to scan every resolved multi-agent prediction with no
watchlist filter, so ad-hoc debates on untracked tickers (Debate Arena, /predict)
kept paying for reflections and full correction debates forever. These tests pin
the gating down.

No live API calls: the DeepSeek client and StockPredictor are both mocked.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.scheduler import (
    REFLECTION_BATCH_LIMIT,
    REFLECTION_LOOKBACK_DAYS,
    REFLECTION_REPREDICT_LIMIT,
    run_reflection_job,
)


# ── Fixtures ────────────────────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE predictions (
    id TEXT PRIMARY KEY,
    ticker TEXT NOT NULL,
    predicted_direction TEXT NOT NULL,
    confidence REAL NOT NULL,
    horizon_days INTEGER NOT NULL DEFAULT 1,
    model_type TEXT NOT NULL,
    feature_snapshot TEXT,
    llm_narrative TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    resolve_after DATE NOT NULL,
    actual_direction TEXT,
    actual_change_pct REAL,
    is_correct INTEGER,
    resolved_at DATETIME
);
CREATE TABLE reflection_log (
    id INTEGER PRIMARY KEY,
    ticker TEXT,
    prediction_id INTEGER,
    date TEXT,
    lesson_learned TEXT,
    was_successful BOOLEAN,
    scope TEXT DEFAULT 'ticker',
    sector TEXT,
    tags TEXT
);
"""


def _days_ago(n: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=n)).strftime("%Y-%m-%d")


class FakeDB:
    """Real in-memory SQLite for predictions/reflection_log, stubs for the rest.

    Uses a real connection so the recency cutoff and LIMIT in the job's query are
    genuinely exercised rather than mocked away.
    """

    def __init__(self, tracked=("AAPL",), cached_advisory=None):
        self._conn = sqlite3.connect(":memory:")
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._tracked = list(tracked)
        self._cached_advisory = cached_advisory
        self.reflections = []
        self.usage_log = []

    @contextmanager
    def connection(self):
        yield self._conn

    def get_tracked_tickers(self):
        return list(self._tracked)

    def get_cached_advisory(self, ticker, days=5):
        return self._cached_advisory

    def insert_reflection(self, ticker, prediction_id, date, lesson_learned,
                          was_successful, scope="ticker", sector=None, tags=None):
        self.reflections.append({"ticker": ticker, "prediction_id": prediction_id,
                                 "lesson": lesson_learned, "was_successful": was_successful})
        self._conn.execute(
            "INSERT INTO reflection_log (ticker, prediction_id, date, lesson_learned,"
            " was_successful, scope, sector, tags) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (ticker, prediction_id, date, lesson_learned, was_successful, scope, sector, tags),
        )

    def log_llm_usage(self, **kwargs):
        self.usage_log.append(kwargs)

    # ── test helpers ──
    def add_prediction(self, pred_id, ticker, is_correct=1, resolved_days_ago=1,
                       model_type="multi_agent", confidence=0.7):
        self._conn.execute(
            "INSERT INTO predictions (id, ticker, predicted_direction, confidence,"
            " model_type, llm_narrative, resolve_after, actual_direction,"
            " actual_change_pct, is_correct, resolved_at)"
            " VALUES (?, ?, 'UP', ?, ?, 'because reasons', ?, 'DOWN', -1.2, ?, ?)",
            (pred_id, ticker, confidence, model_type, _days_ago(resolved_days_ago),
             is_correct, _days_ago(resolved_days_ago)),
        )


def _mock_complete():
    """A stubbed `complete` returning a valid ReflectionLesson JSON."""
    from tests.conftest import make_llm_response

    return AsyncMock(return_value=make_llm_response(json.dumps({
        "lesson_learned": "Confidence outran the evidence.",
        "failure_mode": "overconfidence",
        "success_mode": "none",
        "actionable_fix": "Down-weight single-source catalysts.",
        "should_adjust_strategy": True,
    }), prompt_tokens=100, completion_tokens=20))


@contextmanager
def _patched(complete=None):
    """Patch run_reflection_job's LLM call and its lazily-imported predictor."""
    complete = complete if complete is not None else _mock_complete()
    predictor_cls = MagicMock()
    predictor_cls.return_value.predict_with_agents = AsyncMock(
        return_value={"final_advisory": "HOLD"}
    )
    with patch("orchestrator.scheduler.complete", complete),          patch("orchestrator.scheduler.is_llm_configured", return_value=True),          patch("orchestrator.scheduler.settings.model_reflection", "test/reflection"),          patch("pipeline.predictor.StockPredictor", predictor_cls):
        yield complete, predictor_cls.return_value.predict_with_agents


# ── Watchlist gating ────────────────────────────────────────────────────────

class TestWatchlistGating:
    """Untracked tickers must cost zero tokens."""

    @pytest.mark.asyncio
    async def test_untracked_ticker_spends_nothing(self):
        db = FakeDB(tracked=["AAPL"])
        db.add_prediction("p1", "TSLA", is_correct=0)

        with _patched() as (complete, predict_with_agents):
            await run_reflection_job(db)

        assert complete.await_count == 0
        assert predict_with_agents.await_count == 0
        assert db.reflections == []

    @pytest.mark.asyncio
    async def test_mixed_batch_only_reflects_tracked(self):
        db = FakeDB(tracked=["AAPL"])
        db.add_prediction("p1", "TSLA", is_correct=1)
        db.add_prediction("p2", "AAPL", is_correct=1)
        db.add_prediction("p3", "NVDA", is_correct=1)

        with _patched() as (complete, _):
            await run_reflection_job(db)

        assert complete.await_count == 1
        assert [r["ticker"] for r in db.reflections] == ["AAPL"]

    @pytest.mark.asyncio
    async def test_ticker_match_is_case_insensitive(self):
        """Watchlist is upper-cased on write; predictions.ticker is not normalized."""
        db = FakeDB(tracked=["AAPL"])
        db.add_prediction("p1", "aapl", is_correct=1)

        with _patched() as (complete, _):
            await run_reflection_job(db)

        assert complete.await_count == 1
        assert db.reflections[0]["ticker"] == "aapl"

    @pytest.mark.asyncio
    async def test_empty_watchlist_short_circuits_before_client(self):
        db = FakeDB(tracked=[])
        db.add_prediction("p1", "AAPL", is_correct=0)

        with _patched() as (complete, predict_with_agents):
            await run_reflection_job(db)

        assert complete.await_count == 0
        assert predict_with_agents.await_count == 0


# ── Correction debate gating ────────────────────────────────────────────────

class TestCorrectionGating:
    """The expensive re-debate only fires for tracked, wrong, uncached predictions."""

    @pytest.mark.asyncio
    async def test_correct_prediction_skips_debate(self):
        db = FakeDB(tracked=["AAPL"], cached_advisory=None)
        db.add_prediction("p1", "AAPL", is_correct=1)

        with _patched() as (complete, predict_with_agents):
            await run_reflection_job(db)

        assert complete.await_count == 1
        assert predict_with_agents.await_count == 0
        assert db.reflections[0]["was_successful"] is True

    @pytest.mark.asyncio
    async def test_wrong_prediction_without_cache_triggers_debate(self):
        db = FakeDB(tracked=["AAPL"], cached_advisory=None)
        db.add_prediction("p1", "AAPL", is_correct=0)

        with _patched() as (complete, predict_with_agents):
            await run_reflection_job(db)

        assert complete.await_count == 1
        predict_with_agents.assert_awaited_once_with("AAPL")

    @pytest.mark.asyncio
    async def test_wrong_prediction_with_fresh_cache_skips_debate(self):
        """A fresh advisory already exists — re-debating would pay for it twice."""
        db = FakeDB(tracked=["AAPL"], cached_advisory={"final_advisory": "HOLD"})
        db.add_prediction("p1", "AAPL", is_correct=0)

        with _patched() as (complete, predict_with_agents):
            await run_reflection_job(db)

        assert complete.await_count == 1
        assert predict_with_agents.await_count == 0
        # The lesson is still recorded — only the debate is skipped.
        assert len(db.reflections) == 1
        assert db.reflections[0]["was_successful"] is False

    @pytest.mark.asyncio
    async def test_cached_skip_does_not_send_correction_alert(self):
        db = FakeDB(tracked=["AAPL"], cached_advisory={"final_advisory": "HOLD"})
        db.add_prediction("p1", "AAPL", is_correct=0)
        alert_manager = MagicMock()
        alert_manager.bot.send_message = AsyncMock()

        with _patched():
            await run_reflection_job(db, alert_manager=alert_manager)

        assert alert_manager.bot.send_message.await_count == 0


# ── Query bounds ────────────────────────────────────────────────────────────

class TestQueryBounds:
    """Recency cutoff and batch cap keep the nightly scan bounded."""

    @pytest.mark.asyncio
    async def test_predictions_older_than_lookback_are_excluded(self):
        db = FakeDB(tracked=["AAPL"])
        db.add_prediction("old", "AAPL", is_correct=1,
                          resolved_days_ago=REFLECTION_LOOKBACK_DAYS + 5)
        db.add_prediction("new", "AAPL", is_correct=1, resolved_days_ago=1)

        with _patched() as (complete, _):
            await run_reflection_job(db)

        assert complete.await_count == 1
        assert [r["prediction_id"] for r in db.reflections] == ["new"]

    @pytest.mark.asyncio
    async def test_batch_is_capped(self):
        db = FakeDB(tracked=["AAPL"])
        for i in range(REFLECTION_BATCH_LIMIT + 10):
            db.add_prediction(f"p{i}", "AAPL", is_correct=1, resolved_days_ago=1)

        with _patched() as (complete, _):
            await run_reflection_job(db)

        assert complete.await_count == REFLECTION_BATCH_LIMIT

    @pytest.mark.asyncio
    async def test_non_multi_agent_predictions_ignored(self):
        db = FakeDB(tracked=["AAPL"])
        db.add_prediction("p1", "AAPL", is_correct=1, model_type="per_ticker")

        with _patched() as (complete, _):
            await run_reflection_job(db)

        assert complete.await_count == 0

    @pytest.mark.asyncio
    async def test_already_reflected_predictions_are_not_reprocessed(self):
        db = FakeDB(tracked=["AAPL"])
        db.add_prediction("p1", "AAPL", is_correct=1)

        with _patched() as (complete, _):
            await run_reflection_job(db)
            assert complete.await_count == 1
            await run_reflection_job(db)
            assert complete.await_count == 1

        assert len(db.reflections) == 1


# ── Correction cap ──────────────────────────────────────────────────────────

class TestCorrectionCap:
    """
    Reflection is one cheap call per prediction, but each correction is a full
    multi-agent debate. With REFLECTION_BATCH_LIMIT at 25 an uncapped run could
    spend ~150 calls on the priciest models in a single nightly job.
    """

    @pytest.mark.asyncio
    async def test_corrections_are_capped(self):
        db = FakeDB(tracked=["A", "B", "C", "D", "E", "F"])
        for i, ticker in enumerate(["A", "B", "C", "D", "E", "F"]):
            db.add_prediction(f"p{i}", ticker, is_correct=0)

        with _patched() as (_, predict):
            await run_reflection_job(db)

        assert predict.await_count == REFLECTION_REPREDICT_LIMIT

    @pytest.mark.asyncio
    async def test_every_miss_is_still_reflected_on(self):
        """The cap limits re-debates, not the cheap reflection calls."""
        db = FakeDB(tracked=["A", "B", "C", "D", "E", "F"])
        for i, ticker in enumerate(["A", "B", "C", "D", "E", "F"]):
            db.add_prediction(f"p{i}", ticker, is_correct=0)

        with _patched() as (complete, _):
            await run_reflection_job(db)

        assert complete.await_count == 6
        assert len(db.reflections) == 6

    @pytest.mark.asyncio
    async def test_highest_confidence_misses_are_re_debated_first(self):
        db = FakeDB(tracked=["LOW", "MID", "HIGH"])
        db.add_prediction("p1", "LOW", is_correct=0, confidence=0.10)
        db.add_prediction("p2", "MID", is_correct=0, confidence=0.50)
        db.add_prediction("p3", "HIGH", is_correct=0, confidence=0.95)

        with patch("orchestrator.scheduler.REFLECTION_REPREDICT_LIMIT", 1), _patched() as (_, predict):
            await run_reflection_job(db)

        assert predict.await_count == 1
        assert predict.await_args.args[0] == "HIGH"

    @pytest.mark.asyncio
    async def test_one_ticker_missing_two_horizons_is_re_debated_once(self):
        """Two wrong horizons on the same name are one useful re-debate."""
        db = FakeDB(tracked=["AAPL"])
        db.add_prediction("p1", "AAPL", is_correct=0, confidence=0.8)
        db.add_prediction("p2", "AAPL", is_correct=0, confidence=0.6)

        with _patched() as (_, predict):
            await run_reflection_job(db)

        assert predict.await_count == 1

    @pytest.mark.asyncio
    async def test_a_failing_correction_does_not_block_the_others(self):
        db = FakeDB(tracked=["A", "B"])
        db.add_prediction("p1", "A", is_correct=0, confidence=0.9)
        db.add_prediction("p2", "B", is_correct=0, confidence=0.8)

        with _patched() as (_, predict):
            predict.side_effect = [RuntimeError("debate failed"), {"final_advisory": "HOLD"}]
            await run_reflection_job(db)

        assert predict.await_count == 2

    @pytest.mark.asyncio
    async def test_correct_predictions_are_never_re_debated(self):
        db = FakeDB(tracked=["A", "B", "C"])
        for i, ticker in enumerate(["A", "B", "C"]):
            db.add_prediction(f"p{i}", ticker, is_correct=1)

        with _patched() as (_, predict):
            await run_reflection_job(db)

        assert predict.await_count == 0
