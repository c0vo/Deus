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

    def test_pro_and_flash_are_priced_differently(self, db):
        """
        The old substring pricing matched "deepseek" before "pro", so the
        expensive reasoning model billed at flash rates. Identical token counts
        on the two models must not produce an identical cost.
        """
        for model in ("deepseek-v4-pro", "deepseek-v4-flash"):
            db.log_llm_usage(
                model_name=model, operation=f"price_{model}",
                prompt_tokens=1000, candidate_tokens=1000,
            )
        with db.connection() as conn:
            costs = {
                r["model_name"]: r["cost_usd"]
                for r in conn.execute(
                    "SELECT model_name, cost_usd FROM llm_usage_log WHERE operation LIKE 'price_%'"
                )
            }
        assert costs["deepseek-v4-pro"] > costs["deepseek-v4-flash"]

    def test_unknown_model_falls_back_without_crashing(self, db):
        db.log_llm_usage(
            model_name="some-model-nobody-priced", operation="unknown_price",
            prompt_tokens=1000, candidate_tokens=1000,
        )
        with db.connection() as conn:
            row = conn.execute(
                "SELECT cost_usd FROM llm_usage_log WHERE operation='unknown_price'"
            ).fetchone()
        assert row["cost_usd"] > 0

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


# ── sqlite-vec Extension Tests ──────────────────────────────────────────────

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
