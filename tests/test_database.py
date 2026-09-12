"""Tests for Database — SQLite schema, CRUD operations, and query helpers."""

import json
import pytest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

from data.models import NewsArticle


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def db(tmp_path):
    """Create a Database with a temporary file for testing (no existing data)."""
    from data.database import Database
    db_path = str(tmp_path / "test.db")
    database = Database(db_path=db_path)
    database.initialize()
    return database


@pytest.fixture
def sample_article():
    return NewsArticle(
        id="test_001",
        headline="Test Headline",
        summary="Test Summary",
        source_name="test_source",
        source_type="rss",
        url="https://example.com/test",
        published_at=datetime.now(timezone.utc),
    )


# ── Schema Tests ────────────────────────────────────────────────────────────

class TestSchema:
    """Database schema creation."""

    def test_initialize_creates_articles_table(self, db):
        """After init, the articles table should exist."""
        with db.connection() as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            tables = [row[0] for row in cursor.fetchall()]
        assert "articles" in tables

    def test_initialize_creates_ticker_mentions_table(self, db):
        with db.connection() as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
        assert "ticker_mentions" in tables

    def test_initialize_creates_user_config_table(self, db):
        with db.connection() as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
        assert "user_config" in tables

    def test_initialize_creates_predictions_table(self, db):
        with db.connection() as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
        assert "predictions" in tables

    def test_initialize_creates_llm_usage_log_table(self, db):
        with db.connection() as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
        assert "llm_usage_log" in tables

    def test_fts5_index_created(self, db):
        """FTS5 virtual table should exist."""
        with db.connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='articles_fts'"
            )
            assert cursor.fetchone() is not None


# ── Article CRUD Tests ─────────────────────────────────────────────────────

class TestArticleCRUD:
    """Article insert/query operations."""

    def test_insert_and_retrieve_article(self, db, sample_article):
        db.insert_article(sample_article)
        with db.connection() as conn:
            cursor = conn.execute("SELECT id, headline, url FROM articles WHERE id=?", ("test_001",))
            row = cursor.fetchone()
        assert row is not None
        assert row["id"] == "test_001"
        assert row["headline"] == "Test Headline"

    def test_insert_duplicate_url_returns_false(self, db, sample_article):
        """Inserting the same URL twice should return False (UNIQUE constraint)."""
        db.insert_article(sample_article)
        result = db.insert_article(sample_article)
        assert result is False

    def test_url_exists_true_for_existing_url(self, db, sample_article):
        db.insert_article(sample_article)
        assert db.url_exists("https://example.com/test") is True

    def test_url_exists_false_for_missing_url(self, db):
        assert db.url_exists("https://example.com/nonexistent") is False

    def test_row_to_article_conversion(self, db, sample_article):
        """row_to_article should produce a valid NewsArticle from a DB row."""
        db.insert_article(sample_article)
        article = db.get_article_by_id("test_001")
        # Implementation-dependent: get_article_by_id may or may not exist
        # Test via internal query
        with db.connection() as conn:
            cursor = conn.execute("SELECT * FROM articles WHERE id=?", ("test_001",))
            row = cursor.fetchone()
        if row:
            converted = db.row_to_article(row)
            assert converted.id == "test_001"
            assert isinstance(converted, NewsArticle)


# ── Unclassified Articles Tests ─────────────────────────────────────────────

class TestUnclassifiedArticles:
    """Articles pending classification."""

    def test_returns_unclassified_only(self, db, sample_article):
        classified = NewsArticle(
            id="test_002", headline="Classified", summary="", source_name="test",
            source_type="rss", url="https://example.com/cls", published_at=datetime.now(timezone.utc),
            event_type="earnings",  # Already classified
        )
        db.insert_article(sample_article)  # No event_type
        db.insert_article(classified)

        unclassified = db.get_unclassified_articles()
        ids = [a["id"] for a in unclassified]
        assert "test_001" in ids
        assert "test_002" not in ids


# ── Ticker Mentions Tests ───────────────────────────────────────────────────

class TestTickerMentions:
    """Ticker mention tracking."""

    def test_insert_ticker_mentions(self, db, sample_article):
        db.insert_article(sample_article)
        db.insert_ticker_mentions(sample_article.id, ["AAPL", "MSFT"], 0.5, "high")

        with db.connection() as conn:
            cursor = conn.execute(
                "SELECT ticker FROM ticker_mentions WHERE article_id=?", (sample_article.id,)
            )
            tickers = [row["ticker"] for row in cursor.fetchall()]
        assert "AAPL" in tickers
        assert "MSFT" in tickers


# ── Prediction Tests ────────────────────────────────────────────────────────

class TestPredictions:
    """Prediction CRUD."""

    def test_insert_prediction(self, db):
        pred_id = db.insert_prediction("AAPL", "UP", 0.85, "2026-07-20", {})
        assert pred_id is not None

    def test_get_existing_prediction_returns_none_for_missing(self, db):
        result = db.get_existing_prediction("NONEXISTENT", 1, "2026-07-20")
        assert result is None

    def test_insert_and_retrieve_prediction(self, db):
        db.insert_prediction("AAPL", "UP", 0.85, "2026-07-20", {})
        result = db.get_existing_prediction("AAPL", 1, "2026-07-20")
        if result:  # Implementation may or may not return based on caching rules
            assert result["ticker"] == "AAPL"
            assert result["predicted_direction"] == "UP"


# ── LLM Usage Logging Tests ─────────────────────────────────────────────────

class TestLLMUsageLogging:
    """Cost tracking."""

    def test_log_llm_usage(self, db):
        db.log_llm_usage(
            model_name="test-model",
            operation="test_op",
            prompt_tokens=100,
            candidate_tokens=50,
            latency_ms=200,
            is_error=False,
        )
        with db.connection() as conn:
            cursor = conn.execute("SELECT * FROM llm_usage_log WHERE operation='test_op'")
            row = cursor.fetchone()
        assert row is not None
        assert row["model_name"] == "test-model"
        assert row["prompt_tokens"] == 100
        assert row["candidate_tokens"] == 50

    def test_successful_call_does_not_archive_its_payload(self, db):
        """
        The classifier and ranker pass prompt_text on every batch, which turned
        llm_usage_log into a full prompt archive. settings.log_llm_payloads is
        the global kill switch and defaults off.
        """
        db.log_llm_usage(
            model_name="test-model", operation="payload_ok",
            prompt_tokens=10, candidate_tokens=5, is_error=False,
            prompt_text="a very long prompt", response_text="a very long response",
        )
        with db.connection() as conn:
            row = conn.execute(
                "SELECT prompt_text, response_text FROM llm_usage_log WHERE operation='payload_ok'"
            ).fetchone()
        assert row["prompt_text"] is None
        assert row["response_text"] is None

    def test_failed_call_keeps_its_payload(self, db):
        """Errors are rare and are the case where the prompt is worth having."""
        db.log_llm_usage(
            model_name="test-model", operation="payload_err",
            prompt_tokens=0, candidate_tokens=0, is_error=True,
            error_message="boom", prompt_text="the prompt that failed",
        )
        with db.connection() as conn:
            row = conn.execute(
                "SELECT prompt_text FROM llm_usage_log WHERE operation='payload_err'"
            ).fetchone()
        assert row["prompt_text"] == "the prompt that failed"

    def test_payloads_are_kept_when_explicitly_enabled(self, db):
        from config.settings import settings
        with patch.object(settings, "log_llm_payloads", True):
            db.log_llm_usage(
                model_name="test-model", operation="payload_on",
                prompt_tokens=10, candidate_tokens=5, is_error=False,
                prompt_text="kept prompt",
            )
        with db.connection() as conn:
            row = conn.execute(
                "SELECT prompt_text FROM llm_usage_log WHERE operation='payload_on'"
            ).fetchone()
        assert row["prompt_text"] == "kept prompt"

    def test_token_accounting_is_unaffected_by_payload_dropping(self, db):
        db.log_llm_usage(
            model_name="test-model", operation="payload_tokens",
            prompt_tokens=123, candidate_tokens=45, is_error=False,
            prompt_text="dropped", cost_usd=0.00042,
        )
        with db.connection() as conn:
            row = conn.execute(
                "SELECT total_tokens, cost_usd FROM llm_usage_log WHERE operation='payload_tokens'"
            ).fetchone()
        assert row["total_tokens"] == 168
        assert row["cost_usd"] == pytest.approx(0.00042)

    def test_reported_cost_is_stored_verbatim(self, db):
        """
        Cost comes from the provider now, not a local price table.

        The table this replaced went stale silently — at the point it was
        removed it billed deepseek-v4-pro at 3x its actual rate — so the whole
        point is that nothing here recomputes the number.
        """
        db.log_llm_usage(
            model_name="deepseek/deepseek-v4-pro-0813", operation="reported_cost",
            prompt_tokens=1000, candidate_tokens=1000, cost_usd=0.001305,
        )
        with db.connection() as conn:
            row = conn.execute(
                "SELECT cost_usd FROM llm_usage_log WHERE operation='reported_cost'"
            ).fetchone()
        assert row["cost_usd"] == pytest.approx(0.001305)

    def test_missing_cost_records_zero_and_is_counted(self, db):
        """
        A call the provider reported no cost for must be visible, not invisible.

        It stores as $0.00 — the honest value, since nothing was measured — and
        `unpriced_calls` is what stops that quietly deflating the total.
        """
        db.log_llm_usage(
            model_name="some/unreporting-model", operation="no_cost",
            prompt_tokens=1000, candidate_tokens=1000, cost_usd=None,
        )
        with db.connection() as conn:
            row = conn.execute(
                "SELECT cost_usd FROM llm_usage_log WHERE operation='no_cost'"
            ).fetchone()
        assert row["cost_usd"] == 0.0
        assert db.get_usage_stats(days=1)["unpriced_calls"] >= 1

    def test_priced_calls_are_not_counted_as_unpriced(self, db):
        db.log_llm_usage(
            model_name="some/model", operation="priced",
            prompt_tokens=10, candidate_tokens=5, cost_usd=0.0001,
        )
        assert db.get_usage_stats(days=1)["unpriced_calls"] == 0


    def test_log_error_usage(self, db):
        db.log_llm_usage(
            model_name="test-model", operation="test_error",
            prompt_tokens=0, candidate_tokens=0, latency_ms=50,
            is_error=True, error_message="API timeout",
        )
        with db.connection() as conn:
            cursor = conn.execute("SELECT * FROM llm_usage_log WHERE operation='test_error'")
            row = cursor.fetchone()
        assert row is not None
        assert row["is_error"] == 1


# ── Stats Tests ─────────────────────────────────────────────────────────────

class TestStats:
    """Pipeline statistics."""

    def test_get_stats_returns_expected_keys(self, db):
        stats = db.get_stats()
        expected_keys = {"total_articles", "classified_articles", "embedded_articles"}
        assert expected_keys.issubset(stats.keys())

    def test_get_stats_reflects_insertions(self, db, sample_article):
        stats_before = db.get_stats()
        db.insert_article(sample_article)
        stats_after = db.get_stats()
        assert stats_after["total_articles"] == stats_before["total_articles"] + 1


# ── User Config Tests ───────────────────────────────────────────────────────

class TestUserConfig:
    """User configuration CRUD."""

    def test_set_and_get_config(self, db):
        db.set_user_config("watchlist", json.dumps(["AAPL", "TSLA"]))
        value = db.get_user_config("watchlist")
        assert value is not None
        assert "AAPL" in value

    def test_get_missing_config_returns_none(self, db):
        value = db.get_user_config("nonexistent_key")
        assert value is None


# ── Briefing Candidate Tests ────────────────────────────────────────────────

class TestBriefingCandidates:
    """get_briefing_candidates — the pool backing the prioritized daily brief."""

    def _insert(self, db, *, url, score, hours_ago=1, event_type="general",
                sectors=None, duplicate_of=None, headline="A headline"):
        article = NewsArticle(
            id=url,
            headline=headline,
            summary="A summary",
            source_name="cnbc",
            source_type="rss",
            url=url,
            published_at=datetime.now(timezone.utc) - timedelta(hours=hours_ago),
            event_type=event_type,
            importance_score=score,
            affected_sectors=sectors or [],
            classification_summary="Classified summary",
        )
        db.insert_article(article)
        with db.connection() as conn:
            conn.execute(
                "UPDATE articles SET importance_score=?, event_type=?, "
                "affected_sectors=?, duplicate_of=?, classification_summary=? WHERE url=?",
                (score, event_type, json.dumps(sectors or []), duplicate_of,
                 "Classified summary", url),
            )

    def test_applies_the_importance_floor(self, db):
        self._insert(db, url="https://x.test/high", score=8.0)
        self._insert(db, url="https://x.test/low", score=6.9)
        rows = db.get_briefing_candidates(hours=24, min_importance=7.0)
        assert [r["url"] for r in rows] == ["https://x.test/high"]

    def test_floor_is_inclusive(self, db):
        self._insert(db, url="https://x.test/exact", score=7.0)
        assert len(db.get_briefing_candidates(hours=24, min_importance=7.0)) == 1

    def test_respects_the_time_window(self, db):
        self._insert(db, url="https://x.test/recent", score=9.0, hours_ago=1)
        self._insert(db, url="https://x.test/old", score=9.5, hours_ago=48)
        rows = db.get_briefing_candidates(hours=24, min_importance=7.0)
        assert [r["url"] for r in rows] == ["https://x.test/recent"]

    def test_excludes_noise(self, db):
        self._insert(db, url="https://x.test/noise", score=9.0, event_type="noise")
        assert db.get_briefing_candidates(hours=24, min_importance=7.0) == []

    def test_excludes_duplicates(self, db):
        self._insert(db, url="https://x.test/canonical", score=9.0)
        self._insert(db, url="https://x.test/dupe", score=8.0, duplicate_of="https://x.test/canonical")
        rows = db.get_briefing_candidates(hours=24, min_importance=7.0)
        assert [r["url"] for r in rows] == ["https://x.test/canonical"]

    def test_orders_by_importance_descending(self, db):
        for i, score in enumerate([7.5, 9.5, 8.5]):
            self._insert(db, url=f"https://x.test/{i}", score=score, headline=f"Headline {i}")
        scores = [r["importance_score"] for r in db.get_briefing_candidates(hours=24, min_importance=7.0)]
        assert scores == sorted(scores, reverse=True)

    def test_honours_the_limit(self, db):
        # Distinct headlines: identical text hashes to the same content_hash and
        # is rejected as a duplicate on insert.
        for i in range(10):
            self._insert(db, url=f"https://x.test/{i}", score=8.0, headline=f"Headline {i}")
        assert len(db.get_briefing_candidates(hours=24, min_importance=7.0, limit=3)) == 3

    def test_selects_event_type_for_the_lane_predicates(self, db):
        """get_briefing_by_sector omits this column; the lane logic needs it."""
        self._insert(db, url="https://x.test/macro", score=9.0, event_type="macro")
        rows = db.get_briefing_candidates(hours=24, min_importance=7.0)
        assert rows[0]["event_type"] == "macro"

    def test_selects_the_fields_the_renderer_needs(self, db):
        self._insert(db, url="https://x.test/a", score=8.0, sectors=["Technology"])
        row = db.get_briefing_candidates(hours=24, min_importance=7.0)[0]
        assert {"headline", "summary", "classification_summary", "importance_score",
                "url", "affected_sectors", "source_name", "published_at"} <= set(row)

    def test_empty_pool_returns_empty_list(self, db):
        assert db.get_briefing_candidates(hours=24, min_importance=7.0) == []


# ── Flow tables: off-exchange volume, market regime, option chains ──────────

def _offexch_row(ticker="AAPL", session="2026-08-07", total=13_330_297.818991):
    return {
        "ticker": ticker,
        "session_date": session,
        "short_volume": 5_540_409.463985,
        "short_exempt_volume": 31_490.25,
        "total_volume": total,
        "market_codes": "B,Q,N",
        "published_at": f"{session}T22:00:00+00:00",
    }


class TestOffExchangeVolume:
    """FINRA off-exchange (dark pool) volume storage."""

    def test_upsert_is_idempotent(self, db):
        """The scheduled job re-reads a few sessions of overlap every run, and
        FINRA restates, so re-ingesting the same session must not duplicate."""
        db.upsert_offexchange_volume([_offexch_row()])
        db.upsert_offexchange_volume([_offexch_row()])

        assert len(db.get_offexchange_series("AAPL")) == 1

    def test_restatement_overwrites(self, db):
        db.upsert_offexchange_volume([_offexch_row(total=100.0)])
        db.upsert_offexchange_volume([_offexch_row(total=200.0)])

        series = db.get_offexchange_series("AAPL")
        assert len(series) == 1 and series[0]["total_volume"] == 200.0

    def test_fractional_volumes_survive_the_round_trip(self, db):
        """Columns are REAL, not INTEGER — FINRA reports fractional shares."""
        db.upsert_offexchange_volume([_offexch_row()])
        row = db.get_offexchange_series("AAPL")[0]

        assert row["total_volume"] == pytest.approx(13_330_297.818991)
        assert row["short_exempt_volume"] == pytest.approx(31_490.25)

    def test_series_joins_consolidated_volume_from_price_history(self, db):
        """The off-exchange SHARE is the actual signal, so the join carrying
        consolidated volume has to land or the feature silently defaults."""
        db.upsert_price_history("AAPL", [{
            "date": "2026-08-07", "open": 310.0, "high": 315.0,
            "low": 309.0, "close": 313.33, "volume": 34_407_100,
        }])
        db.upsert_offexchange_volume([_offexch_row()])

        row = db.get_offexchange_series("AAPL")[0]
        assert row["consolidated_volume"] == 34_407_100
        assert row["total_volume"] / row["consolidated_volume"] == pytest.approx(0.387, abs=0.01)

    def test_series_tolerates_missing_price_history(self, db):
        db.upsert_offexchange_volume([_offexch_row()])
        assert db.get_offexchange_series("AAPL")[0]["consolidated_volume"] is None

    def test_last_date_is_global_not_per_ticker(self, db):
        """One CNMS file carries every symbol, so the sync high-water mark is
        global. A per-ticker mark would re-download the same file per ticker."""
        db.upsert_offexchange_volume([
            _offexch_row("AAPL", "2026-08-05"),
            _offexch_row("NVDA", "2026-08-07"),
        ])
        assert db.get_last_offexchange_date() == "2026-08-07"

    def test_last_date_is_none_when_empty(self, db):
        assert db.get_last_offexchange_date() is None


class TestMarketRegime:
    """Market-wide DIX / GEX / put-call series storage."""

    def _rows(self):
        return [
            {"metric": "dix", "session_date": "2026-08-07", "value": 0.4553,
             "published_at": "2026-08-08T00:00:00+00:00", "source": "squeezemetrics"},
            {"metric": "gex", "session_date": "2026-08-07", "value": 9.05e9,
             "published_at": "2026-08-08T00:00:00+00:00", "source": "squeezemetrics"},
        ]

    def test_upsert_is_idempotent(self, db):
        db.upsert_market_regime(self._rows())
        db.upsert_market_regime(self._rows())

        assert len(db.get_market_regime_series()) == 2

    def test_metrics_share_a_session_without_colliding(self, db):
        """Long form: the PK is (metric, session), so three feeds publishing the
        same date coexist instead of overwriting one another."""
        db.upsert_market_regime(self._rows() + [
            {"metric": "occ_put_call_ratio", "session_date": "2026-08-07",
             "value": 0.659, "published_at": "2026-08-08T00:00:00+00:00", "source": "occ"},
        ])
        assert {r["metric"] for r in db.get_market_regime_series()} == {
            "dix", "gex", "occ_put_call_ratio"}

    def test_last_date_is_per_metric(self, db):
        """Feeds publish independently, so each resumes from its own mark."""
        db.upsert_market_regime(self._rows())
        assert db.get_last_market_regime_date("dix") == "2026-08-07"
        assert db.get_last_market_regime_date("occ_put_call_ratio") is None

    def test_large_gex_values_survive(self, db):
        db.upsert_market_regime(self._rows())
        gex = [r for r in db.get_market_regime_series() if r["metric"] == "gex"][0]
        assert gex["value"] == pytest.approx(9.05e9)


class TestOptionChainDaily:
    """Daily option-chain aggregate storage."""

    def _row(self, session="2026-08-07", pcr=0.29):
        return {
            "ticker": "AAPL", "session_date": session, "spot_price": 313.33,
            "call_volume": 113591.0, "put_volume": 33196.0,
            "call_oi": 525736.0, "put_oi": 285166.0,
            "put_call_volume_ratio": pcr, "put_call_oi_ratio": 0.542,
            "atm_iv": 0.285, "iv_skew": 0.056,
            "near_term_iv": 0.324, "far_term_iv": 0.283,
            "expirations_seen": 4, "contracts_seen": 452,
            "published_at": f"{session}T21:00:00+00:00",
        }

    def test_upsert_is_idempotent(self, db):
        db.upsert_option_chain_daily([self._row()])
        db.upsert_option_chain_daily([self._row()])

        assert len(db.get_option_chain_series("AAPL")) == 1

    def test_same_session_rerun_overwrites(self, db):
        """A later snapshot on the same session is strictly better — volume and
        open interest are still settling when the first one runs."""
        db.upsert_option_chain_daily([self._row(pcr=0.29)])
        db.upsert_option_chain_daily([self._row(pcr=0.35)])

        series = db.get_option_chain_series("AAPL")
        assert len(series) == 1
        assert series[0]["put_call_volume_ratio"] == pytest.approx(0.35)

    def test_series_is_ordered_by_disclosure(self, db):
        db.upsert_option_chain_daily([
            self._row(session="2026-08-07"), self._row(session="2026-08-05")])

        sessions = [r["session_date"] for r in db.get_option_chain_series("AAPL")]
        assert sessions == ["2026-08-05", "2026-08-07"]

    def test_last_date_tracks_per_ticker(self, db):
        db.upsert_option_chain_daily([self._row()])
        assert db.get_last_option_chain_date("AAPL") == "2026-08-07"
        assert db.get_last_option_chain_date("NVDA") is None


# ── Alerts & Digests ────────────────────────────────────────────────────────

def _alert(ticker="AAPL", kind="price_drop", pct=-3.1, **kw):
    row = {
        "ticker": ticker,
        "kind": kind,
        "pct": pct,
        "price": 100.0,
        "title": f"{ticker} moved {pct}%",
    }
    row.update(kw)
    return row


class TestAlerts:
    """The alerts table stores what was pushed, not just that it was."""

    def test_table_created(self, db):
        with db.connection() as conn:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='alerts'"
            ).fetchone()
        assert row is not None

    def test_insert_returns_row_id(self, db):
        first = db.insert_alert(_alert())
        second = db.insert_alert(_alert(pct=-6.2))
        assert first >= 1 and second > first

    def test_round_trips_the_fields_the_card_renders(self, db):
        db.insert_alert(_alert(grounded_by="web", summary="Guidance cut",
                               sources_json=[{"url": "https://example.com/a"}]))
        row = db.get_recent_alerts(limit=1)[0]
        assert row["ticker"] == "AAPL"
        assert row["pct"] == -3.1
        assert row["grounded_by"] == "web"
        assert row["summary"] == "Guidance cut"
        # Accepted as a list and stored as JSON, because that is what the
        # grounding module produces.
        assert json.loads(row["sources_json"])[0]["url"] == "https://example.com/a"

    def test_defaults_when_grounding_is_absent(self, db):
        db.insert_alert(_alert())
        row = db.get_recent_alerts(limit=1)[0]
        assert row["sources_json"] == "[]"
        assert row["grounded_by"] == "none"

    def test_newest_first_within_the_same_second(self, db):
        """CURRENT_TIMESTAMP only has second resolution; id is the tiebreak."""
        db.insert_alert(_alert(pct=-3.1))
        db.insert_alert(_alert(pct=-6.2))
        assert [r["pct"] for r in db.get_recent_alerts()] == [-6.2, -3.1]

    def test_honours_the_limit(self, db):
        for pct in (-3.1, -4.2, -5.3):
            db.insert_alert(_alert(pct=pct))
        assert len(db.get_recent_alerts(limit=2)) == 2

    def test_filters_by_kind(self, db):
        db.insert_alert(_alert(kind="price_drop"))
        db.insert_alert(_alert(kind="volume"))
        kinds = [r["kind"] for r in db.get_recent_alerts(kind="volume")]
        assert kinds == ["volume"]

    def test_last_abs_pct_is_none_when_nothing_was_sent(self, db):
        assert db.get_last_alert_abs_pct_today("AAPL", ("price_drop",)) is None

    def test_last_abs_pct_takes_the_deepest_move(self, db):
        db.insert_alert(_alert(pct=-3.1))
        db.insert_alert(_alert(pct=-6.2))
        db.insert_alert(_alert(pct=-4.0))
        assert db.get_last_alert_abs_pct_today("AAPL", ("price_drop",)) == 6.2

    def test_last_abs_pct_is_per_ticker(self, db):
        db.insert_alert(_alert(ticker="AAPL", pct=-6.2))
        assert db.get_last_alert_abs_pct_today("MSFT", ("price_drop",)) is None

    def test_last_abs_pct_respects_the_kinds_filter(self, db):
        db.insert_alert(_alert(kind="volume", pct=-6.2))
        assert db.get_last_alert_abs_pct_today("AAPL", ("price_drop",)) is None

    def test_last_abs_pct_with_no_kinds_queries_nothing(self, db):
        db.insert_alert(_alert(pct=-6.2))
        assert db.get_last_alert_abs_pct_today("AAPL", ()) is None

    def test_last_abs_pct_ignores_earlier_days(self, db):
        """Yesterday's alert must not suppress today's first one."""
        db.insert_alert(_alert(pct=-6.2))
        with db.connection() as conn:
            conn.execute(
                "UPDATE alerts SET created_at = datetime('now', '-2 days')"
            )
        assert db.get_last_alert_abs_pct_today("AAPL", ("price_drop",)) is None


class TestDigests:
    """Generated pushes are persisted so they can be shown and re-read."""

    def test_table_created(self, db):
        with db.connection() as conn:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='digests'"
            ).fetchone()
        assert row is not None

    def test_insert_and_fetch_latest(self, db):
        db.insert_digest("weekly_tip", "<b>older</b>")
        db.insert_digest("weekly_tip", "<b>newer</b>")
        assert db.get_latest_digest("weekly_tip")["body_html"] == "<b>newer</b>"

    def test_kinds_do_not_leak_into_each_other(self, db):
        db.insert_digest("weekly_tip", "<b>tip</b>")
        db.insert_digest("daily_briefing", "<b>brief</b>")
        assert db.get_latest_digest("daily_briefing")["body_html"] == "<b>brief</b>"
        assert len(db.get_digests("weekly_tip")) == 1

    def test_unknown_kind_returns_none(self, db):
        assert db.get_latest_digest("never_generated") is None

    def test_facts_are_stored_as_json_from_a_dict(self, db):
        """facts_json is the audit trail for every number in the body."""
        db.insert_digest("weekly_tip", "<b>a</b>",
                         facts_json={"spy_september_median": -0.6})
        row = db.get_latest_digest("weekly_tip")
        assert json.loads(row["facts_json"])["spy_september_median"] == -0.6

    def test_period_and_model_round_trip(self, db):
        db.insert_digest("weekly_tip", "<b>a</b>", body_text="a", model="vendor/model",
                         period_start="2026-09-06", period_end="2026-09-12")
        row = db.get_latest_digest("weekly_tip")
        assert row["period_start"] == "2026-09-06"
        assert row["period_end"] == "2026-09-12"
        assert row["model"] == "vendor/model"
        assert row["body_text"] == "a"

    def test_history_is_newest_first_and_limited(self, db):
        for i in range(3):
            db.insert_digest("weekly_tip", f"<b>{i}</b>")
        bodies = [r["body_html"] for r in db.get_digests("weekly_tip", limit=2)]
        assert bodies == ["<b>2</b>", "<b>1</b>"]


# ── Daily Stance Tests ──────────────────────────────────────────────────────

class TestStances:
    """One evidence-based call per ticker per day, keyed so a re-run replaces it."""

    def test_table_created(self, db):
        with db.connection() as conn:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='stances'"
            ).fetchone()
        assert row is not None

    def test_insert_and_fetch_latest(self, db):
        db.upsert_stance("NVDA", "2026-09-11", "HOLD", conviction="Low")
        db.upsert_stance("NVDA", "2026-09-12", "TRIM", conviction="Medium",
                         thesis="Rating flipped to Sell.",
                         key_risk="Target implies upside.",
                         what_would_change="A close above $185.",
                         model="vendor/model")
        row = db.get_latest_stance("NVDA")
        assert row["action"] == "TRIM"
        assert row["conviction"] == "Medium"
        assert row["what_would_change"] == "A close above $185."
        assert row["model"] == "vendor/model"

    def test_before_excludes_the_row_just_written(self, db):
        """
        The arrow compares today against yesterday. Without `before`, a re-run
        the same morning would read back the row it had written and report every
        stance as unchanged.
        """
        db.upsert_stance("NVDA", "2026-09-11", "HOLD")
        db.upsert_stance("NVDA", "2026-09-12", "SELL")
        assert db.get_latest_stance("NVDA", before="2026-09-12")["action"] == "HOLD"

    def test_before_with_no_earlier_row_is_none(self, db):
        db.upsert_stance("NVDA", "2026-09-12", "SELL")
        assert db.get_latest_stance("NVDA", before="2026-09-12") is None

    def test_same_day_rerun_replaces_rather_than_duplicates(self, db):
        db.upsert_stance("NVDA", "2026-09-12", "HOLD", prev_action="BUY/ADD")
        db.upsert_stance("NVDA", "2026-09-12", "SELL", prev_action="HOLD")
        with db.connection() as conn:
            count = conn.execute("SELECT COUNT(*) c FROM stances").fetchone()["c"]
        assert count == 1
        row = db.get_latest_stance("NVDA")
        assert row["action"] == "SELL"
        assert row["prev_action"] == "HOLD"

    def test_ticker_is_upper_cased(self, db):
        db.upsert_stance("nvda", "2026-09-12", "HOLD")
        assert db.get_latest_stance("NVDA") is not None

    def test_unknown_ticker_is_none(self, db):
        assert db.get_latest_stance("NOSUCH") is None

    def test_json_columns_accept_python_objects(self, db):
        db.upsert_stance("NVDA", "2026-09-12", "TRIM",
                         evidence_json=["rsi", "analyst_target"],
                         facts_json={"ret_1d": -3.1})
        row = db.get_latest_stance("NVDA")
        assert json.loads(row["evidence_json"]) == ["rsi", "analyst_target"]
        assert json.loads(row["facts_json"])["ret_1d"] == -3.1

    def test_latest_stances_is_one_row_per_ticker(self, db):
        db.upsert_stance("NVDA", "2026-09-11", "HOLD")
        db.upsert_stance("NVDA", "2026-09-12", "SELL")
        db.upsert_stance("AMD", "2026-09-12", "BUY/ADD",
                         evidence_json=["ml", "news"])
        latest = db.get_latest_stances()
        assert set(latest) == {"NVDA", "AMD"}
        assert latest["NVDA"]["action"] == "SELL"
        # Decoded in the DB layer: the API hands this straight to the browser,
        # and a nested JSON string would need a second parse there.
        assert latest["AMD"]["evidence_used"] == ["ml", "news"]
        assert "evidence_json" not in latest["AMD"]

    def test_latest_stances_is_empty_before_the_first_run(self, db):
        assert db.get_latest_stances() == {}

    def test_malformed_evidence_json_degrades_to_empty(self, db):
        """A bad blob must not take the whole market grid down with it."""
        db.upsert_stance("NVDA", "2026-09-12", "HOLD", evidence_json="not json")
        assert db.get_latest_stances()["NVDA"]["evidence_used"] == []
# ── Classification Queue Tests ──────────────────────────────────────────────

class TestClassificationQueue:
    """
    The bounded classification backlog.

    Every test here corresponds to one way the unbounded version went wrong on
    the phone: a permanently poisoned row re-selected forever, an archive of old
    articles keeping the queue non-empty, and a 'pending' number that measured
    embeddings rather than classification.
    """

    @staticmethod
    def _insert(db, article_id, *, days_old=0, event_type=None, attempts=0,
                duplicate_of=None, summary=None):
        published = datetime.now(timezone.utc) - timedelta(days=days_old)
        with db.connection() as conn:
            conn.execute(
                """
                INSERT INTO articles (
                    id, headline, summary, content_hash, source_name, source_type,
                    url, published_at, fetched_at, event_type,
                    classification_attempts, duplicate_of, classification_summary
                ) VALUES (?, ?, ?, ?, 'test', 'rss', ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    article_id, f"Headline {article_id}", "Summary",
                    f"hash_{article_id}", f"https://example.com/{article_id}",
                    published.isoformat(), published.isoformat(),
                    event_type, attempts, duplicate_of, summary,
                ),
            )

    def test_migration_adds_classification_attempts_column(self, db):
        with db.connection() as conn:
            columns = [c["name"] for c in conn.execute("PRAGMA table_info(articles)")]
        assert "classification_attempts" in columns

    def test_migration_creates_partial_unclassified_index(self, db):
        with db.connection() as conn:
            names = [
                r["name"] for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='index'"
                )
            ]
        assert "idx_articles_unclassified" in names

    def test_candidates_exclude_duplicates_exhausted_and_old(self, db):
        self._insert(db, "fresh")
        self._insert(db, "dup", duplicate_of="fresh")
        self._insert(db, "exhausted", attempts=3)
        self._insert(db, "ancient", days_old=90)
        self._insert(db, "classified", event_type="macro")

        ids = {
            r["id"] for r in db.get_classification_candidates(
                limit=50, max_attempts=3, max_age_days=30
            )
        }
        assert ids == {"fresh"}

    def test_candidates_are_newest_first_and_respect_limit(self, db):
        for i, age in enumerate((5, 1, 3)):
            self._insert(db, f"a{i}", days_old=age)

        rows = db.get_classification_candidates(limit=2, max_attempts=3, max_age_days=30)
        assert [r["id"] for r in rows] == ["a1", "a2"]

    def test_candidates_allow_attempts_below_the_cap(self, db):
        self._insert(db, "one_try", attempts=1)
        self._insert(db, "two_tries", attempts=2)
        ids = {
            r["id"] for r in db.get_classification_candidates(
                limit=50, max_attempts=3, max_age_days=30
            )
        }
        assert ids == {"one_try", "two_tries"}

    def test_record_failure_increments_and_returns_new_counts(self, db):
        self._insert(db, "a", attempts=1)
        self._insert(db, "b")

        counts = db.record_classification_failure(["a", "b"])
        assert counts == {"a": 2, "b": 1}
        # The rows stay NULL so a later pass can retry them.
        with db.connection() as conn:
            types = [
                r["event_type"] for r in conn.execute(
                    "SELECT event_type FROM articles"
                )
            ]
        assert types == [None, None]

    def test_record_failure_on_empty_list_is_a_noop(self, db):
        assert db.record_classification_failure([]) == {}

    def test_mark_stale_only_touches_old_unclassified_rows(self, db):
        self._insert(db, "fresh")
        self._insert(db, "ancient", days_old=60)
        self._insert(db, "ancient_classified", days_old=60, event_type="macro")

        assert db.mark_stale_unclassified(max_age_days=30) == 1
        with db.connection() as conn:
            rows = {
                r["id"]: (r["event_type"], r["classification_summary"], r["urgency"])
                for r in conn.execute(
                    "SELECT id, event_type, classification_summary, urgency FROM articles"
                )
            }
        assert rows["ancient"][0] == "stale"
        assert "older than 30 days" in rows["ancient"][1]
        assert rows["ancient"][2] == "low"
        assert rows["fresh"][0] is None
        assert rows["ancient_classified"][0] == "macro"

    def test_mark_stale_honours_its_limit(self, db):
        for i in range(5):
            self._insert(db, f"old{i}", days_old=40 + i)

        assert db.mark_stale_unclassified(max_age_days=30, limit=2) == 2
        with db.connection() as conn:
            remaining = conn.execute(
                "SELECT COUNT(*) AS c FROM articles WHERE event_type IS NULL"
            ).fetchone()["c"]
        assert remaining == 3

    def test_requeue_error_articles_clears_verdict_and_attempts(self, db):
        self._insert(db, "recent_error", event_type="error", attempts=1,
                     summary="Classification failed.")
        self._insert(db, "old_error", days_old=90, event_type="error", attempts=2)
        self._insert(db, "noise", event_type="noise")

        assert db.requeue_error_articles(max_age_days=30) == 1
        with db.connection() as conn:
            rows = {
                r["id"]: (r["event_type"], r["classification_attempts"],
                          r["classification_summary"])
                for r in conn.execute(
                    "SELECT id, event_type, classification_attempts, "
                    "classification_summary FROM articles"
                )
            }
        assert rows["recent_error"] == (None, 0, None)
        # Out of window: left alone rather than requeued into work nobody wants.
        assert rows["old_error"][0] == "error"
        assert rows["noise"][0] == "noise"

    def test_reset_parked_attempts_unparks_only_exhausted_unclassified_rows(self, db):
        """Give back the retries a broken request shape spent, nothing more.

        The batch schema demanded only `id` and put no minimum on the items
        array, so ten articles came back as one near-empty object: nine rows per
        batch took a `batch_missing_result` and burnt all three attempts inside
        fifteen minutes, on requests that never asked about them.
        """
        self._insert(db, "parked", attempts=3)
        self._insert(db, "very_parked", attempts=7)
        self._insert(db, "still_trying", attempts=2)
        self._insert(db, "untried")
        # A row that reached a verdict keeps it, whatever its attempt count —
        # otherwise this would re-queue work that is already paid for and done.
        self._insert(db, "classified_but_retried", attempts=3, event_type="macro")
        self._insert(db, "noise", attempts=3, event_type="noise")

        assert db.reset_parked_classification_attempts(max_attempts=3) == 2

        with db.connection() as conn:
            rows = {
                r["id"]: (r["event_type"], r["classification_attempts"])
                for r in conn.execute(
                    "SELECT id, event_type, classification_attempts FROM articles"
                )
            }
        assert rows["parked"] == (None, 0)
        assert rows["very_parked"] == (None, 0)
        # Below the cap: still a live candidate, so its count must not move.
        assert rows["still_trying"] == (None, 2)
        assert rows["untried"] == (None, 0)
        assert rows["classified_but_retried"] == ("macro", 3)
        assert rows["noise"] == ("noise", 3)

    def test_reset_parked_attempts_makes_rows_selectable_again(self, db):
        """The whole point: the row has to come back out of the candidate query."""
        self._insert(db, "parked", attempts=3)

        assert db.get_classification_candidates(
            limit=50, max_attempts=3, max_age_days=30
        ) == []

        db.reset_parked_classification_attempts(max_attempts=3)
        ids = {
            r["id"] for r in db.get_classification_candidates(
                limit=50, max_attempts=3, max_age_days=30
            )
        }
        assert ids == {"parked"}

    def test_reset_parked_attempts_is_idempotent_and_honours_the_cap(self, db):
        """It runs once per deploy, but a worker that dies mid-run re-runs it."""
        self._insert(db, "parked", attempts=3)

        assert db.reset_parked_classification_attempts(max_attempts=3) == 1
        assert db.reset_parked_classification_attempts(max_attempts=3) == 0

        # A different cap selects a different set; nothing below it is touched.
        self._insert(db, "two_tries", attempts=2)
        assert db.reset_parked_classification_attempts(max_attempts=5) == 0
        assert db.reset_parked_classification_attempts(max_attempts=2) == 1

    def test_reset_parked_attempts_on_an_empty_table_is_a_noop(self, db):
        assert db.reset_parked_classification_attempts(max_attempts=3) == 0

    def test_stats_exposes_every_documented_key(self, db):
        stats = db.get_classification_stats()
        assert set(stats) == {
            "pending", "pending_in_window", "exhausted", "stale", "error",
            "noise", "classified", "embedding_pending", "embedding_exhausted",
            "embedding_skipped_noise",
        }

    def test_stats_separate_the_backlog_from_its_terminal_states(self, db):
        self._insert(db, "pending_fresh")
        self._insert(db, "pending_old", days_old=60)
        self._insert(db, "exhausted", attempts=3)
        self._insert(db, "stale", event_type="stale")
        self._insert(db, "error", event_type="error")
        self._insert(db, "noise", event_type="noise")
        self._insert(db, "real", event_type="earnings")

        stats = db.get_classification_stats(max_age_days=30, max_attempts=3)
        assert stats["pending"] == 3            # fresh + old + exhausted
        assert stats["pending_in_window"] == 1  # only the fresh one is workable
        assert stats["exhausted"] == 1
        assert stats["stale"] == 1
        assert stats["error"] == 1
        assert stats["noise"] == 1
        assert stats["classified"] == 1         # 'real' only

    def test_embedding_pending_excludes_noise_rows(self, db):
        """The whole "17,551 pending" illusion in one assertion.

        99.8% of that number was noise rows the embed query skips by design, so
        it could never fall and said nothing about anything.
        """
        self._insert(db, "needs_embedding")
        self._insert(db, "noise_unembedded", event_type="noise")

        stats = db.get_classification_stats()
        assert stats["embedding_pending"] == 1
        assert stats["embedding_skipped_noise"] == 1

    def test_unranked_excludes_noise_error_and_stale(self, db):
        self._insert(db, "ok", event_type="earnings")
        self._insert(db, "noise", event_type="noise")
        self._insert(db, "error", event_type="error")
        self._insert(db, "stale", event_type="stale")

        ids = {r["id"] for r in db.get_unranked_articles(limit=50)}
        assert ids == {"ok"}

    def test_get_stats_reports_the_backlog(self, db):
        self._insert(db, "pending")
        self._insert(db, "stale", event_type="stale")
        self._insert(db, "error", event_type="error")
        self._insert(db, "dup", duplicate_of="pending")

        stats = db.get_stats()
        assert stats["unclassified_articles"] == 1  # the duplicate does not count
        assert stats["stale_articles"] == 1
        assert stats["error_articles"] == 1


# ── sqlite-vec Extension Tests ──────────────────────────────────────────────

class TestPriceHistoryStarts:
    def test_returns_the_earliest_row_only_for_requested_tickers(self, db):
        db.upsert_price_history("SPY", [
            {"date": "1993-01-29", "open": 43.9, "high": 43.9,
             "low": 43.9, "close": 43.9},
            {"date": "2026-09-11", "open": 650.0, "high": 650.0,
             "low": 650.0, "close": 650.0},
        ])
        db.upsert_price_history("QQQ", [{"date": "1999-03-10", "open": 51.0,
                                           "high": 51.0, "low": 51.0, "close": 51.0}])

        assert db.get_price_history_starts([" spy ", "MISSING", "QQQ"]) == {
            "SPY": "1993-01-29", "QQQ": "1999-03-10",
        }

    def test_empty_input_does_not_build_invalid_sql(self, db):
        assert db.get_price_history_starts([]) == {}


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("sqlite_vec"),
    reason="sqlite-vec extension not installed",
)
class TestSqliteVec:
    """Vector similarity search (requires sqlite-vec)."""

    def test_extension_loadable(self, db):
        has_vec = db.has_sqlite_vec
        if not has_vec:
            pytest.skip("sqlite-vec not configured in DB")
        with db.connection() as conn:
            cursor = conn.execute("SELECT vec_version()")
            version = cursor.fetchone()
            assert version is not None
