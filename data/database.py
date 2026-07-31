"""
Deus — SQLite Database Manager

Manages the SQLite database: connection lifecycle, schema creation,
and common query helpers. Uses FTS5 for full-text search.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Generator, Optional

import numpy as np

from config.logging_config import get_logger
from config.settings import settings
from data.models import NewsArticle

log = get_logger(__name__)

# ── LLM Pricing ──────────────────────────────────────────────────────────
#
# USD per 1M tokens, as (prompt, completion). Keys are matched EXACTLY against
# the model_name that was logged — deliberately not a substring ladder.
#
# The previous substring version silently mispriced the two most expensive
# operations in the system: `"deepseek" in name` was tested before `"pro"`, so
# deepseek-v4-pro billed at flash rates, and no Gemini model in settings.py
# contains "pro" or "8b", so every Gemini call fell through to a stale default.
#
# VERIFY THESE AGAINST YOUR PROVIDER'S CURRENT PRICE LIST. Model names in this
# project are env-configurable, so a model can be swapped in without a price;
# that case logs a warning and falls back to a deliberately pessimistic rate,
# on the principle that an unpriced model should over-report, never under-.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "deepseek-v4-flash":      (0.14, 0.28),
    "deepseek-v4-pro":        (1.25, 5.00),
    "gemini-2.5-flash-lite":  (0.075, 0.30),
    "gemini-3.1-flash-lite":  (0.075, 0.30),
    "gemini-3-flash-preview": (0.30, 1.20),
    "gemini-embedding-001":   (0.15, 0.0),
}

_PRICING_FALLBACK = (0.30, 1.20)


def _resolve_pricing(model_name: str) -> tuple[float, float]:
    """Returns (per-prompt-token, per-completion-token) cost in USD."""
    rates = MODEL_PRICING.get(model_name)
    if rates is None:
        rates = _PRICING_FALLBACK
        log.warning(
            "db.pricing.unknown_model",
            model_name=model_name,
            fallback_per_1m=rates,
            hint="Add this model to MODEL_PRICING in data/database.py",
        )
    return rates[0] / 1_000_000, rates[1] / 1_000_000


# ── Schema Definition ────────────────────────────────────────────────────

SCHEMA_SQL = """
-- Core news articles table
CREATE TABLE IF NOT EXISTS articles (
    id TEXT PRIMARY KEY,
    headline TEXT NOT NULL,
    summary TEXT DEFAULT '',
    content_hash TEXT UNIQUE,
    source_name TEXT NOT NULL,
    source_type TEXT NOT NULL,
    url TEXT UNIQUE NOT NULL,
    published_at DATETIME NOT NULL,
    fetched_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    -- LLM Classification
    event_type TEXT,
    sentiment_score REAL,
    urgency TEXT,
    suggested_direction TEXT,
    affected_sectors TEXT,           -- JSON array
    affected_tickers TEXT,           -- JSON array
    classification_summary TEXT,

    -- LLM Ranking
    importance_score REAL,

    -- Embedding vector
    embedding BLOB,

    -- Raw data
    raw_data TEXT DEFAULT '{}'
);

-- Full-text search index for keyword queries
CREATE VIRTUAL TABLE IF NOT EXISTS articles_fts USING fts5(
    headline, summary, content='articles', content_rowid='rowid'
);

-- FTS triggers to keep the index in sync
CREATE TRIGGER IF NOT EXISTS articles_ai AFTER INSERT ON articles BEGIN
    INSERT INTO articles_fts(rowid, headline, summary)
    VALUES (new.rowid, new.headline, new.summary);
END;

CREATE TRIGGER IF NOT EXISTS articles_ad AFTER DELETE ON articles BEGIN
    INSERT INTO articles_fts(articles_fts, rowid, headline, summary)
    VALUES ('delete', old.rowid, old.headline, old.summary);
END;

CREATE TRIGGER IF NOT EXISTS articles_au AFTER UPDATE ON articles BEGIN
    INSERT INTO articles_fts(articles_fts, rowid, headline, summary)
    VALUES ('delete', old.rowid, old.headline, old.summary);
    INSERT INTO articles_fts(rowid, headline, summary)
    VALUES (new.rowid, new.headline, new.summary);
END;

-- Performance index for time-range queries (sector heatmap, trending, etc.)
CREATE INDEX IF NOT EXISTS idx_articles_published_at ON articles(published_at);

-- Trending tickers aggregate
CREATE TABLE IF NOT EXISTS ticker_mentions (
    ticker TEXT NOT NULL,
    article_id TEXT NOT NULL REFERENCES articles(id),
    sentiment_score REAL,
    urgency TEXT,
    mentioned_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ticker, article_id)
);

-- User configuration (watchlist, alert thresholds)
CREATE TABLE IF NOT EXISTS user_config (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Alerts tracking to prevent spam
CREATE TABLE IF NOT EXISTS sent_alerts (
    article_id TEXT,
    alert_type TEXT,
    sent_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (article_id, alert_type)
);

CREATE TABLE IF NOT EXISTS price_alerts (
    ticker TEXT,
    alert_date DATE DEFAULT CURRENT_DATE,
    PRIMARY KEY (ticker, alert_date)
);

-- Token and Cost tracking
CREATE TABLE IF NOT EXISTS llm_usage_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    model_name TEXT,
    operation TEXT,
    prompt_tokens INTEGER,
    candidate_tokens INTEGER,
    total_tokens INTEGER,
    cost_usd REAL
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_articles_published ON articles(published_at DESC);
CREATE INDEX IF NOT EXISTS idx_articles_source ON articles(source_name);
CREATE INDEX IF NOT EXISTS idx_articles_urgency ON articles(urgency);
CREATE INDEX IF NOT EXISTS idx_articles_importance ON articles(importance_score DESC);
CREATE INDEX IF NOT EXISTS idx_ticker_mentions_ticker ON ticker_mentions(ticker);
CREATE INDEX IF NOT EXISTS idx_llm_usage_timestamp ON llm_usage_log(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_llm_usage_model_op ON llm_usage_log(model_name, operation);

-- ML Predictions tracking
CREATE TABLE IF NOT EXISTS predictions (
    id TEXT PRIMARY KEY,
    ticker TEXT NOT NULL,
    predicted_direction TEXT NOT NULL,       -- 'UP' or 'DOWN'
    confidence REAL NOT NULL,               -- 0.0 to 1.0
    horizon_days INTEGER NOT NULL DEFAULT 1,
    model_type TEXT NOT NULL,               -- 'per_ticker', 'sector', 'universal', 'llm_only'
    feature_snapshot TEXT,                   -- JSON blob of feature vector
    llm_narrative TEXT,                      -- Gemini-generated explanation
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    resolve_after DATE NOT NULL,            -- date when this prediction should be checked
    -- Resolution fields (filled by daily job)
    actual_direction TEXT,                   -- 'UP' or 'DOWN' or NULL
    actual_change_pct REAL,                 -- actual % change
    is_correct INTEGER,                     -- 1=correct, 0=incorrect, NULL=unresolved
    resolved_at DATETIME
);

-- Price history cache (OHLCV from Yahoo Finance)
CREATE TABLE IF NOT EXISTS price_history (
    ticker TEXT NOT NULL,
    date DATE NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL NOT NULL,
    volume INTEGER,
    fetched_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ticker, date)
);
-- Ticker Info Cache
CREATE TABLE IF NOT EXISTS ticker_info (
    ticker TEXT PRIMARY KEY,
    sector TEXT,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for new tables
CREATE INDEX IF NOT EXISTS idx_predictions_ticker ON predictions(ticker);
CREATE INDEX IF NOT EXISTS idx_predictions_resolve ON predictions(resolve_after);
CREATE INDEX IF NOT EXISTS idx_predictions_unresolved ON predictions(is_correct) WHERE is_correct IS NULL;
CREATE INDEX IF NOT EXISTS idx_price_history_ticker ON price_history(ticker, date);
-- Reflection Log
CREATE TABLE IF NOT EXISTS reflection_log (
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

-- Predictions Cache
CREATE TABLE IF NOT EXISTS predictions_cache (
    ticker TEXT,
    date TEXT,
    advisory_json TEXT,
    PRIMARY KEY(ticker, date)
);

-- Conversations
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Messages
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id, timestamp);

-- Pipeline cycle metrics for real-time dashboard telemetry
CREATE TABLE IF NOT EXISTS pipeline_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recorded_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    cycle_duration_seconds REAL,
    articles_fetched INTEGER DEFAULT 0,
    articles_inserted INTEGER DEFAULT 0,
    articles_classified INTEGER DEFAULT 0,
    articles_ranked INTEGER DEFAULT 0,
    articles_embedded INTEGER DEFAULT 0,
    alerts_generated INTEGER DEFAULT 0,
    errors_count INTEGER DEFAULT 0,
    llm_calls_count INTEGER DEFAULT 0,
    llm_cost_estimate REAL DEFAULT 0.0
);
CREATE INDEX IF NOT EXISTS idx_pipeline_metrics_time ON pipeline_metrics(recorded_at DESC);

-- Sector sentiment snapshots (every ~15 min)
CREATE TABLE IF NOT EXISTS sector_sentiment_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sector TEXT NOT NULL,
    snapshot_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    avg_sentiment REAL,
    article_count INTEGER DEFAULT 0,
    bullish_count INTEGER DEFAULT 0,
    bearish_count INTEGER DEFAULT 0,
    neutral_count INTEGER DEFAULT 0,
    avg_importance REAL DEFAULT 0.0,
    top_tickers_json TEXT,
    sentiment_momentum REAL DEFAULT 0.0
);
CREATE INDEX IF NOT EXISTS idx_sector_sentiment_lookup ON sector_sentiment_snapshots(sector, snapshot_time);

-- Sector rotation signals
CREATE TABLE IF NOT EXISTS sector_rotation_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    detected_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    from_sector TEXT DEFAULT 'Market_Neutral',
    to_sector TEXT NOT NULL,
    signal_strength REAL DEFAULT 0.0,
    reasoning TEXT,
    triggered_by TEXT DEFAULT 'sentiment_shift',
    is_active INTEGER DEFAULT 1,
    acknowledged INTEGER DEFAULT 0
);

-- Daily sector snapshot (for historical tracking)
CREATE TABLE IF NOT EXISTS sector_daily_snapshot (
    sector TEXT NOT NULL,
    date DATE NOT NULL,
    mention_count INTEGER DEFAULT 0,
    avg_sentiment REAL DEFAULT 0.0,
    avg_importance REAL DEFAULT 0.0,
    bullish_ratio REAL DEFAULT 0.0,
    top_tickers TEXT,
    PRIMARY KEY (sector, date)
);

-- Hot tickers auto-discovered (not on user watchlist)
CREATE TABLE IF NOT EXISTS hot_tickers (
    ticker TEXT PRIMARY KEY,
    mention_count INTEGER DEFAULT 0,
    avg_sentiment REAL DEFAULT 0.0,
    sectors_json TEXT DEFAULT '[]',
    rationale TEXT DEFAULT '',
    first_detected_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_detected_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- IPO tracking
CREATE TABLE IF NOT EXISTS ipo_tracker (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_name TEXT NOT NULL,
    ticker TEXT,
    ipo_date DATE,
    offering_price REAL,
    status TEXT DEFAULT 'rumored',
    sector TEXT,
    estimated_valuation TEXT,
    source_article_id TEXT REFERENCES articles(id),
    notes TEXT,
    detected_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    metadata_json TEXT DEFAULT '{}'
);

-- Upcoming ticker events (earnings, product launches, etc.)
CREATE TABLE IF NOT EXISTS ticker_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    event_type TEXT NOT NULL,
    event_date DATE NOT NULL,
    event_title TEXT,
    confidence TEXT DEFAULT 'confirmed',
    source TEXT DEFAULT 'llm_extracted',
    source_article_id TEXT REFERENCES articles(id),
    notes TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_ticker_events_date ON ticker_events(event_date);
CREATE INDEX IF NOT EXISTS idx_ticker_events_ticker ON ticker_events(ticker, event_date);

-- Trend forecasts (LLM-generated forward-looking analysis)
CREATE TABLE IF NOT EXISTS trend_forecasts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT,
    sector TEXT,
    forecast_type TEXT NOT NULL,
    scenario_label TEXT,
    time_horizon TEXT DEFAULT '1m',
    confidence REAL DEFAULT 0.0,
    narrative TEXT NOT NULL,
    key_drivers_json TEXT DEFAULT '[]',
    supporting_evidence TEXT,
    generated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    expires_at DATETIME,
    is_active INTEGER DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_trend_forecasts_active ON trend_forecasts(is_active, sector, ticker);

-- ── Smart money: insider trades, >5% stakes, institutional flow ──────────
--
-- Every table here is indexed on the date the information became PUBLIC, not
-- the date the underlying event happened. A Form 4 covers a trade made up to
-- two business days before it was filed; a 13F reports a quarter that ended up
-- to 45 days earlier. Feature queries filter on filed_at so the model is never
-- shown something that had not been disclosed yet.

-- SEC Form 4 — insider transactions (non-derivative only)
CREATE TABLE IF NOT EXISTS insider_transactions (
    id TEXT PRIMARY KEY,                 -- accession_no + row index
    ticker TEXT NOT NULL,
    issuer_cik TEXT,
    insider_name TEXT,
    insider_title TEXT,
    is_officer INTEGER DEFAULT 0,
    is_director INTEGER DEFAULT 0,
    is_ten_pct_owner INTEGER DEFAULT 0,
    transaction_date DATE NOT NULL,      -- when the trade happened
    filed_at DATETIME NOT NULL,          -- when it became public (as-of key)
    transaction_code TEXT,               -- P=buy, S=sale, A=grant, M=exercise, F=tax, G=gift
    is_discretionary INTEGER DEFAULT 0,  -- 1 only for P/S; the rest are comp mechanics
    shares REAL,
    price_per_share REAL,
    value_usd REAL,                      -- signed: negative when disposed
    shares_owned_after REAL,
    is_10b5_1 INTEGER,                   -- NULL = unknown (checkbox only exists post-Apr-2023)
    accession_no TEXT,
    raw_data TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_insider_ticker_filed
    ON insider_transactions(ticker, filed_at);

-- SEC Schedule 13D / 13G — crossing the 5% ownership threshold
CREATE TABLE IF NOT EXISTS institutional_stakes (
    id TEXT PRIMARY KEY,                 -- accession number
    ticker TEXT NOT NULL,
    filer_name TEXT,
    filer_cik TEXT,
    form_type TEXT,                      -- 'SC 13D', 'SC 13G', or an /A amendment
    is_activist INTEGER DEFAULT 0,       -- 13D signals intent to influence; 13G is passive
    is_amendment INTEGER DEFAULT 0,
    pct_of_class REAL,
    shares REAL,
    event_date DATE,
    filed_at DATETIME NOT NULL,          -- as-of key
    accession_no TEXT,
    raw_data TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_stakes_ticker_filed
    ON institutional_stakes(ticker, filed_at);

-- Korean daily net buy/sell by investor class. Korea discloses per-ticker
-- institutional flow daily, which the US has no free equivalent for.
--
-- flow_unit records whether the net columns are share counts or KRW: Naver
-- publishes volume, the KRX Open API publishes value. Features normalise
-- against total_value in the same unit, so the two must never be summed
-- together without conversion.
CREATE TABLE IF NOT EXISTS kr_investor_flows (
    ticker TEXT NOT NULL,                -- bare six-digit KRX code
    trade_date DATE NOT NULL,
    inst_net REAL,                       -- 기관합계
    foreign_net REAL,                    -- 외국인
    retail_net REAL,                     -- 개인 (NULL from Naver)
    pension_net REAL,                    -- 연기금 (NULL from Naver)
    financial_inv_net REAL,              -- 금융투자 (NULL from Naver)
    trust_net REAL,                      -- 투신 (NULL from Naver)
    total_value REAL,                    -- 거래량 or 거래대금, for normalisation
    flow_unit TEXT DEFAULT 'shares',     -- 'shares' | 'krw'
    source TEXT DEFAULT 'naver',
    PRIMARY KEY (ticker, trade_date)
);
CREATE INDEX IF NOT EXISTS idx_kr_flows_date ON kr_investor_flows(trade_date);
"""


class Database:
    """
    SQLite database manager for Deus.

    Usage:
        db = Database()
        db.initialize()  # Creates tables if needed

        with db.connection() as conn:
            conn.execute("SELECT ...")
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or settings.db_path
        # Ensure the storage directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._sqlite_vec_log_emitted = False

    def initialize(self) -> None:
        """Create all tables, indexes, and triggers if they don't exist."""
        with self.connection() as conn:
            conn.executescript(SCHEMA_SQL)
            
            # Migration: add new columns for in-depth LLM tracking
            #
            # The two articles.*_scanned_at columns record that an extraction
            # was *attempted*, which is not the same as it having produced a
            # row in ipo_tracker / ticker_events. Without them the scanners
            # select their candidates purely by "has no tracker row yet", so
            # every article the model declines to extract from comes back on
            # the next scan and is paid for again, for the whole 48-72h window.
            for col_sql in [
                "ALTER TABLE llm_usage_log ADD COLUMN latency_ms INTEGER",
                "ALTER TABLE llm_usage_log ADD COLUMN is_error INTEGER DEFAULT 0",
                "ALTER TABLE llm_usage_log ADD COLUMN error_message TEXT",
                "ALTER TABLE llm_usage_log ADD COLUMN prompt_text TEXT",
                "ALTER TABLE llm_usage_log ADD COLUMN response_text TEXT",
                "ALTER TABLE articles ADD COLUMN ipo_scanned_at DATETIME",
                "ALTER TABLE articles ADD COLUMN event_scanned_at DATETIME",
                # Bounds embedding retries. A permanently unembeddable row (empty
                # text, say) would otherwise sit in the `embedding IS NULL` queue
                # forever, occupying a slot in every batch.
                "ALTER TABLE articles ADD COLUMN embed_attempts INTEGER DEFAULT 0",
                # Which provider supplied a KR flow row, and in what unit. Naver
                # reports share counts, the KRX Open API reports KRW; summing the
                # two without conversion would silently corrupt the feature.
                "ALTER TABLE kr_investor_flows ADD COLUMN flow_unit TEXT DEFAULT 'shares'",
                "ALTER TABLE kr_investor_flows ADD COLUMN source TEXT DEFAULT 'naver'",
            ]:
                try:
                    conn.execute(col_sql)
                except sqlite3.OperationalError:
                    pass  # Column likely already exists
            
            # Migration: drop full_text column from articles and rebuild FTS index automatically
            cursor = conn.execute("PRAGMA table_info(articles)")
            columns = [col["name"] for col in cursor.fetchall()]
            if "full_text" in columns:
                try:
                    conn.execute("DROP TRIGGER IF EXISTS articles_ai")
                    conn.execute("DROP TRIGGER IF EXISTS articles_ad")
                    conn.execute("DROP TRIGGER IF EXISTS articles_au")
                    conn.execute("ALTER TABLE articles DROP COLUMN full_text")
                    conn.execute("DROP TABLE IF EXISTS articles_fts")
                    # Recreate triggers and the new FTS table schema without full_text
                    conn.executescript(SCHEMA_SQL)
                    # Rebuild the FTS index
                    conn.execute("INSERT INTO articles_fts(articles_fts) VALUES('rebuild')")
                    log.info("database.migration", msg="Dropped full_text column and rebuilt FTS index.")
                except Exception as e:
                    log.error("database.migration.failed", error=str(e))

            # Migration: add scope, sector, tags to reflection_log
            for col_sql in [
                "ALTER TABLE reflection_log ADD COLUMN scope TEXT DEFAULT 'ticker'",
                "ALTER TABLE reflection_log ADD COLUMN sector TEXT",
                "ALTER TABLE reflection_log ADD COLUMN tags TEXT",
            ]:
                try:
                    conn.execute(col_sql)
                except sqlite3.OperationalError:
                    pass  # Column likely already exists

            # Backfill existing rows that have no scope
            conn.execute(
                "UPDATE reflection_log SET scope = 'ticker' WHERE scope IS NULL OR scope = ''"
            )

            # Migration: semantic-duplicate tracking.
            # `duplicate_of` points at the article this one duplicates; NULL means
            # it is the canonical copy. `dedup_checked` marks that the comparison
            # has run, so the backfill job can converge instead of re-scanning
            # every non-duplicate article forever.
            for col_sql in [
                "ALTER TABLE articles ADD COLUMN duplicate_of TEXT",
                "ALTER TABLE articles ADD COLUMN dedup_checked INTEGER DEFAULT 0",
                # JSON array of ISO 3166-1 alpha-2 codes. NULL means "never
                # examined", [] means "examined, no geography" — the geo
                # backfill relies on that distinction to converge.
                "ALTER TABLE articles ADD COLUMN countries TEXT",
            ]:
                try:
                    conn.execute(col_sql)
                except sqlite3.OperationalError:
                    pass  # Column likely already exists

            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_articles_duplicate_of "
                "ON articles(duplicate_of)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_articles_dedup_checked "
                "ON articles(dedup_checked) WHERE embedding IS NOT NULL"
            )

            # Migration: purge ticker_events rows with an empty event_date.
            # The column is DATE NOT NULL but SQLite does not reject '', and ''
            # sorts below every real date — so these rows satisfied the
            # `event_date <= end` bound on every "upcoming events" query and
            # leaked into the calendar forever. Both write paths now skip
            # undated events, so this is a one-time cleanup.
            conn.execute(
                "DELETE FROM ticker_events "
                "WHERE event_date IS NULL OR TRIM(event_date) = ''"
            )

        log.info("database.initialized", path=self.db_path)

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Context manager for database connections.

        Enables WAL mode for concurrent reads and foreign keys.
        Auto-commits on success, rolls back on exception.
        """
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        
        self.has_sqlite_vec = False
        try:
            if settings.sqlite_vec_path:
                conn.enable_load_extension(True)
                conn.load_extension(settings.sqlite_vec_path)
                conn.enable_load_extension(False)
                self.has_sqlite_vec = True
            else:
                import sqlite_vec
                conn.enable_load_extension(True)
                sqlite_vec.load(conn)
                conn.enable_load_extension(False)
                self.has_sqlite_vec = True
            if not getattr(self, '_sqlite_vec_log_emitted', False):
                log.info("sqlite_vec.loaded_successfully")
                self._sqlite_vec_log_emitted = True
        except Exception as e:
            if not getattr(self, '_sqlite_vec_log_emitted', False):
                log.warning("sqlite_vec.load_failed_falling_back_to_numpy", error=str(e))
                self._sqlite_vec_log_emitted = True
            
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ── Article CRUD ─────────────────────────────────────────────────────

    def insert_article(self, article: NewsArticle) -> bool:
        """
        Insert a new article into the database.

        Returns True if inserted, False if duplicate (URL already exists).
        """
        with self.connection() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO articles (
                        id, headline, summary, content_hash, source_name, source_type,
                        url, published_at, fetched_at,
                        event_type, sentiment_score, urgency, suggested_direction,
                        affected_sectors, affected_tickers, classification_summary,
                        importance_score, raw_data
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        article.id,
                        article.headline,
                        article.summary,
                        article.content_hash,
                        article.source_name,
                        article.source_type,
                        article.url,
                        article.published_at.isoformat(),
                        article.fetched_at.isoformat(),
                        article.event_type,
                        article.sentiment_score,
                        article.urgency,
                        article.suggested_direction,
                        json.dumps(article.affected_sectors),
                        json.dumps(article.affected_tickers),
                        article.classification_summary,
                        article.importance_score,
                        json.dumps(article.raw_data),
                    ),
                )
                return True
            except sqlite3.IntegrityError:
                # Duplicate URL or ID
                return False

    def url_exists(self, url: str) -> bool:
        """Check if an article URL already exists in the database (for dedup)."""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM articles WHERE url = ? LIMIT 1", (url,)
            ).fetchone()
            return row is not None

    def row_to_article(self, row: sqlite3.Row | dict[str, Any]) -> NewsArticle:
        """Convert a stored article row back into the pipeline DTO."""
        data = dict(row)

        def parse_json_list(value: Any) -> list[str]:
            if isinstance(value, list):
                return [str(item) for item in value]
            if not value:
                return []
            try:
                parsed = json.loads(value)
            except (TypeError, json.JSONDecodeError):
                return []
            return [str(item) for item in parsed] if isinstance(parsed, list) else []

        def parse_json_object(value: Any) -> dict:
            if isinstance(value, dict):
                return value
            if not value:
                return {}
            try:
                parsed = json.loads(value)
            except (TypeError, json.JSONDecodeError):
                return {}
            return parsed if isinstance(parsed, dict) else {}

        def parse_datetime(value: Any) -> datetime:
            if isinstance(value, datetime):
                return value
            if value:
                try:
                    parsed = datetime.fromisoformat(str(value))
                    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
                except ValueError:
                    pass
            return datetime.now(timezone.utc)

        return NewsArticle(
            id=data["id"],
            headline=data["headline"],
            summary=data.get("summary") or "",
            content_hash=data.get("content_hash") or "",
            source_name=data["source_name"],
            source_type=data["source_type"],
            url=data["url"],
            published_at=parse_datetime(data.get("published_at")),
            fetched_at=parse_datetime(data.get("fetched_at")),
            event_type=data.get("event_type"),
            sentiment_score=data.get("sentiment_score"),
            urgency=data.get("urgency"),
            suggested_direction=data.get("suggested_direction"),
            affected_sectors=parse_json_list(data.get("affected_sectors")),
            affected_tickers=parse_json_list(data.get("affected_tickers")),
            classification_summary=data.get("classification_summary"),
            importance_score=data.get("importance_score"),
            raw_data=parse_json_object(data.get("raw_data")),
        )

    def get_recent_articles(
        self, limit: int = 50, source: Optional[str] = None
    ) -> list[dict]:
        """Fetch recent articles, optionally filtered by source."""
        with self.connection() as conn:
            if source:
                rows = conn.execute(
                    """
                    SELECT * FROM articles
                    WHERE source_name = ? AND duplicate_of IS NULL
                    ORDER BY published_at DESC LIMIT ?
                    """,
                    (source, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM articles
                    WHERE duplicate_of IS NULL
                    ORDER BY published_at DESC LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            return [dict(row) for row in rows]

    def get_unclassified_articles(self, limit: int = 50) -> list[dict]:
        """Fetch articles that haven't been classified by the LLM yet."""
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM articles
                WHERE event_type IS NULL
                ORDER BY published_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]
            
    def get_recent_summaries_for_ticker(self, ticker: str, hours: int = 24) -> list[str]:
        """Get top-ranked classification summaries for articles mentioning a ticker recently."""
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT a.classification_summary 
                FROM articles a
                JOIN ticker_mentions tm ON a.id = tm.article_id
                WHERE tm.ticker = ? AND a.published_at >= ? AND a.classification_summary IS NOT NULL
                  AND (a.event_type IS NULL OR a.event_type != 'noise')
                ORDER BY a.importance_score DESC NULLS LAST, a.published_at DESC
                LIMIT 5
                """,
                (ticker, cutoff),
            ).fetchall()
            return [row["classification_summary"] for row in rows]

    def get_recent_articles_for_ticker(self, ticker: str, hours: int = 24, limit: int = 5) -> list[dict]:
        """Get top-ranked full article details for articles mentioning a ticker recently."""
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT a.headline, a.source_name, a.published_at, a.sentiment_score,
                       a.classification_summary, a.url, a.importance_score
                FROM articles a
                JOIN ticker_mentions tm ON a.id = tm.article_id
                WHERE tm.ticker = ? AND a.published_at >= ?
                  AND (a.event_type IS NULL OR a.event_type != 'noise')
                ORDER BY a.importance_score DESC NULLS LAST, a.published_at DESC
                LIMIT ?
                """,
                (ticker, cutoff, limit),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_unranked_articles(self, limit: int = 50) -> list[dict]:
        """Fetch classified but unranked articles."""
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM articles
                WHERE event_type IS NOT NULL 
                  AND event_type != 'noise'
                  AND importance_score IS NULL
                ORDER BY published_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]

    def update_classification(
        self,
        article_id: str,
        event_type: str,
        sentiment_score: float,
        urgency: str,
        suggested_direction: str,
        affected_sectors: list[str],
        affected_tickers: list[str],
        classification_summary: str,
        countries: Optional[list[str]] = None,
    ) -> None:
        """Update an article's classification fields after LLM processing."""
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE articles SET
                    event_type = ?,
                    sentiment_score = ?,
                    urgency = ?,
                    suggested_direction = ?,
                    affected_sectors = ?,
                    affected_tickers = ?,
                    classification_summary = ?,
                    countries = COALESCE(?, countries)
                WHERE id = ?
                """,
                (
                    event_type,
                    sentiment_score,
                    urgency,
                    suggested_direction,
                    json.dumps(affected_sectors),
                    json.dumps(affected_tickers),
                    classification_summary,
                    json.dumps(countries) if countries is not None else None,
                    article_id,
                ),
            )

    def update_ranking(
        self, article_id: str, importance_score: float
    ) -> None:
        """Update an article's ranking fields after LLM batch ranking."""
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE articles SET importance_score = ?
                WHERE id = ?
                """,
                (importance_score, article_id),
            )

    def update_embedding(self, article_id: str, embedding: np.ndarray) -> None:
        """Store the embedding vector as a BLOB."""
        with self.connection() as conn:
            conn.execute(
                "UPDATE articles SET embedding = ? WHERE id = ?",
                (embedding.astype(np.float32).tobytes(), article_id),
            )

    # ── Ticker Mentions ──────────────────────────────────────────────────

    def insert_ticker_mentions(
        self,
        article_id: str,
        tickers: list[str],
        sentiment_score: Optional[float],
        urgency: Optional[str],
    ) -> None:
        """Record ticker mentions from a classified article."""
        with self.connection() as conn:
            for ticker in tickers:
                try:
                    conn.execute(
                        """
                        INSERT INTO ticker_mentions (ticker, article_id, sentiment_score, urgency)
                        VALUES (?, ?, ?, ?)
                        """,
                        (ticker.upper(), article_id, sentiment_score, urgency),
                    )
                except sqlite3.IntegrityError:
                    pass  # Already recorded

    def get_top_trending_tickers(self, hours: int = 24, limit: int = 15) -> list[dict]:
        """
        Get top trending tickers overall based on mention count.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT affected_tickers, sentiment_score
                FROM articles
                WHERE published_at >= ? AND affected_tickers IS NOT NULL
                """,
                (cutoff,)
            ).fetchall()

        ticker_counts = {}
        for row in rows:
            try:
                tickers = json.loads(row["affected_tickers"])
                sentiment = row["sentiment_score"] or 0.0
            except (json.JSONDecodeError, TypeError):
                continue
            
            for ticker in tickers:
                if ticker not in ticker_counts:
                    ticker_counts[ticker] = {"mention_count": 0, "sentiment_sum": 0.0}
                ticker_counts[ticker]["mention_count"] += 1
                ticker_counts[ticker]["sentiment_sum"] += sentiment

        sorted_tickers = sorted(
            [
                {
                    "ticker": t,
                    "mention_count": d["mention_count"],
                    "avg_sentiment": d["sentiment_sum"] / d["mention_count"]
                }
                for t, d in ticker_counts.items()
            ],
            key=lambda x: x["mention_count"],
            reverse=True
        )
        return sorted_tickers[:limit]

    def get_briefing_by_sector(self, hours: int = 24, limit: int = 10) -> dict[str, list[dict]]:
        """
        Get top ranked articles in the last N hours, grouped by sector.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT headline, summary, classification_summary, importance_score, url,
                       affected_sectors, affected_tickers, source_name, published_at,
                       sentiment_score, suggested_direction
                FROM articles
                WHERE importance_score IS NOT NULL AND published_at >= ?
                  AND event_type != 'noise' AND duplicate_of IS NULL
                ORDER BY importance_score DESC
                LIMIT ?
                """,
                (cutoff, limit)
            ).fetchall()
            
        result = {}
        for row in rows:
            sectors = ["General"]
            if row["affected_sectors"]:
                try:
                    parsed = json.loads(row["affected_sectors"])
                    if parsed:
                        sectors = parsed
                except (json.JSONDecodeError, TypeError):
                    pass
            
            # Put article in its primary sector
            primary_sector = sectors[0] if sectors else "General"
            if primary_sector not in result:
                result[primary_sector] = []
            result[primary_sector].append(dict(row))
            
        return result

    # ── Sector & Hot Ticker Methods ─────────────────────────────────────────

    def insert_rotation_signal(self, signal: dict) -> None:
        """Store a sector rotation detection signal."""
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO sector_rotation_signals
                    (from_sector, to_sector, signal_strength, reasoning,
                     triggered_by, is_active)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    signal.get("from_sector", "Market_Neutral"),
                    signal["to_sector"],
                    signal.get("signal_strength", 0.0),
                    signal.get("reasoning", ""),
                    signal.get("triggered_by", "sentiment_shift"),
                    signal.get("is_active", 1),
                )
            )

    def get_active_rotation_signals(self) -> list[dict]:
        """Get active sector rotation signals from the last 48 hours."""
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM sector_rotation_signals
                WHERE is_active = 1 AND detected_at >= datetime('now', '-48 hours')
                ORDER BY signal_strength DESC
                """
            ).fetchall()
            return [dict(row) for row in rows]

    def upsert_hot_ticker(self, data: dict) -> None:
        """Insert or update a hot ticker discovered by the analyzer."""
        now = datetime.now(timezone.utc).isoformat()
        with self.connection() as conn:
            existing = conn.execute(
                "SELECT mention_count, sectors_json FROM hot_tickers WHERE ticker = ?",
                (data["ticker"],)
            ).fetchone()
            if existing:
                rationale = data.get("rationale", "")
                conn.execute(
                    """
                    UPDATE hot_tickers SET mention_count = ?, avg_sentiment = ?,
                        sectors_json = ?, rationale = ?, last_detected_at = ?
                    WHERE ticker = ?
                    """,
                    (data["mention_count"], data["avg_sentiment"],
                     json.dumps(data.get("sectors", [])),
                     rationale, now, data["ticker"])
                )
            else:
                rationale = data.get("rationale", "")
                conn.execute(
                    """
                    INSERT INTO hot_tickers (ticker, mention_count, avg_sentiment,
                        sectors_json, rationale, first_detected_at, last_detected_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (data["ticker"], data["mention_count"], data["avg_sentiment"],
                     json.dumps(data.get("sectors", [])),
                     rationale, now, now)
                )

    def get_hot_tickers(self, limit: int = 20, exclude_watchlist: bool = True) -> list[dict]:
        """Get tickers with surging mentions, optionally excluding the user's watchlist."""
        with self.connection() as conn:
            if exclude_watchlist:
                # Get watchlist from user_config
                watchlist_str = conn.execute(
                    "SELECT value FROM user_config WHERE key = 'tracked_tickers'"
                ).fetchone()
                watchlist = json.loads(watchlist_str["value"]) if watchlist_str else []
                placeholders = ",".join("?" for _ in watchlist)
                query = f"""
                    SELECT * FROM hot_tickers
                    WHERE ticker NOT IN ({placeholders})
                    ORDER BY mention_count DESC LIMIT ?
                """ if watchlist else """
                    SELECT * FROM hot_tickers
                    ORDER BY mention_count DESC LIMIT ?
                """
                params = watchlist + [limit] if watchlist else [limit]
                rows = conn.execute(query, params).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM hot_tickers ORDER BY mention_count DESC LIMIT ?",
                    (limit,)
                ).fetchall()
            return [dict(row) for row in rows]

    # ── Full-Text Search ─────────────────────────────────────────────────

    # ── Vector Similarity Search ─────────────────────────────────────────

    def get_all_embeddings(self, exclude_noise: bool = False) -> list[tuple[str, np.ndarray]]:
        """
        Load all article embeddings for similarity search.

        Returns list of (article_id, embedding_vector) tuples.
        """
        with self.connection() as conn:
            query = "SELECT id, embedding FROM articles WHERE embedding IS NOT NULL"
            if exclude_noise:
                query += " AND (event_type IS NULL OR event_type != 'noise')"
            rows = conn.execute(query).fetchall()
            results = []
            for row in rows:
                vec = np.frombuffer(row["embedding"], dtype=np.float32)
                results.append((row["id"], vec))
            return results

    # ── Semantic Deduplication ───────────────────────────────────────────

    def find_duplicate(
        self,
        article_id: str,
        embedding: np.ndarray,
        published_at: str,
        window_days: int = 3,
        threshold: float = 0.70,
    ) -> Optional[tuple[str, float]]:
        """
        Find the nearest prior article to ``article_id`` within a time window.

        The window matters twice over: it keeps a two-year-old story from
        suppressing today's, and it bounds the scan. Without it every candidate
        query is a full scan of the embedding column.

        Returns ``(source_article_id, similarity)`` or None.
        """
        if embedding is None:
            return None

        if not getattr(self, "has_sqlite_vec", False):
            return self._find_duplicate_numpy(
                article_id, embedding, published_at, window_days, threshold
            )

        with self.connection() as conn:
            row = conn.execute(
                """
                SELECT id, vec_distance_cosine(embedding, ?) AS distance
                FROM articles
                WHERE id != ?
                  AND embedding IS NOT NULL
                  AND duplicate_of IS NULL
                  AND published_at BETWEEN datetime(?, ?) AND datetime(?, ?)
                ORDER BY distance
                LIMIT 1
                """,
                (
                    embedding.tobytes(),
                    article_id,
                    published_at, f"-{window_days} days",
                    published_at, f"+{window_days} days",
                ),
            ).fetchone()

        if not row or row["distance"] is None:
            return None
        similarity = 1.0 - row["distance"]
        return (row["id"], similarity) if similarity > threshold else None

    def _find_duplicate_numpy(
        self,
        article_id: str,
        embedding: np.ndarray,
        published_at: str,
        window_days: int,
        threshold: float,
    ) -> Optional[tuple[str, float]]:
        """Fallback for builds without sqlite-vec. Still windowed."""
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT id, embedding FROM articles
                WHERE id != ?
                  AND embedding IS NOT NULL
                  AND duplicate_of IS NULL
                  AND published_at BETWEEN datetime(?, ?) AND datetime(?, ?)
                """,
                (
                    article_id,
                    published_at, f"-{window_days} days",
                    published_at, f"+{window_days} days",
                ),
            ).fetchall()

        norm_a = np.linalg.norm(embedding)
        if norm_a == 0:
            return None

        best_id, best_sim = None, 0.0
        for row in rows:
            vec = np.frombuffer(row["embedding"], dtype=np.float32)
            denom = norm_a * np.linalg.norm(vec)
            if denom == 0:
                continue
            sim = float(np.dot(embedding, vec) / denom)
            if sim > best_sim:
                best_id, best_sim = row["id"], sim

        return (best_id, best_sim) if best_id and best_sim > threshold else None

    def mark_duplicate(self, article_id: str, source_article_id: str) -> None:
        """Flag an article as a duplicate of another. Reversible: clear the column."""
        with self.connection() as conn:
            conn.execute(
                "UPDATE articles SET duplicate_of = ?, dedup_checked = 1 WHERE id = ?",
                (source_article_id, article_id),
            )

    def mark_dedup_checked(self, article_id: str) -> None:
        """Record that the duplicate comparison ran and found nothing."""
        with self.connection() as conn:
            conn.execute(
                "UPDATE articles SET dedup_checked = 1 WHERE id = ?", (article_id,)
            )

    def record_embed_failure(self, article_id: str) -> None:
        """Counts a failed embedding attempt so the retry loop terminates."""
        with self.connection() as conn:
            conn.execute(
                "UPDATE articles SET embed_attempts = COALESCE(embed_attempts, 0) + 1 WHERE id = ?",
                (article_id,),
            )

    def mark_scan_attempted(self, article_id: str, scan: str) -> None:
        """
        Record that an extraction scan was attempted on this article.

        Deliberately separate from whether the scan produced anything. The IPO
        and event scanners only ever wrote a row on a *successful* extraction,
        so an article the model declined to extract from stayed in the
        candidate pool and was re-sent on every subsequent scan until it aged
        out of the window. Stamping the attempt caps the cost at one call per
        article.
        """
        column = {"ipo": "ipo_scanned_at", "event": "event_scanned_at"}.get(scan)
        if not column:
            raise ValueError(f"Unknown scan type: {scan!r}")

        with self.connection() as conn:
            conn.execute(
                f"UPDATE articles SET {column} = CURRENT_TIMESTAMP WHERE id = ?",
                (article_id,),
            )

    def get_dedup_backlog(self, limit: int = 200) -> list[dict]:
        """Embedded articles that have never been compared, oldest first."""
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT id, published_at, embedding FROM articles
                WHERE dedup_checked = 0 AND embedding IS NOT NULL
                ORDER BY published_at ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_dedup_stats(self) -> dict:
        """Counts behind the embedding-coverage figure on the dashboard."""
        with self.connection() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*) AS total,
                    SUM(CASE WHEN embedding IS NOT NULL THEN 1 ELSE 0 END) AS embedded,
                    SUM(CASE WHEN duplicate_of IS NOT NULL THEN 1 ELSE 0 END) AS duplicates,
                    SUM(CASE WHEN dedup_checked = 0 AND embedding IS NOT NULL
                             THEN 1 ELSE 0 END) AS unchecked
                FROM articles
                """
            ).fetchone()
            return {
                "total": row["total"] or 0,
                "embedded": row["embedded"] or 0,
                "duplicates": row["duplicates"] or 0,
                "unchecked": row["unchecked"] or 0,
            }

    # ── Geography ────────────────────────────────────────────────────────

    def get_news_geo(self, hours: int = 24, recent_limit: int = 40) -> dict:
        """
        News volume per country plus the latest tagged stories.

        The counts fill the globe; the recent list drives the event markers.
        Countries are stored as a JSON array per article, so aggregation is
        done in Python — the row count here is small (one window of news) and
        SQLite has no native JSON array expansion worth the complexity.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()

        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT id, headline, source_name, published_at, countries,
                       sentiment_score, importance_score, url, event_type
                FROM articles
                WHERE countries IS NOT NULL
                  AND countries != '[]'
                  AND published_at >= ?
                  AND duplicate_of IS NULL
                  AND (event_type IS NULL OR event_type != 'noise')
                ORDER BY published_at DESC
                """,
                (cutoff,),
            ).fetchall()

        counts: dict[str, dict] = {}
        recent: list[dict] = []

        for row in rows:
            try:
                codes = json.loads(row["countries"]) or []
            except (json.JSONDecodeError, TypeError):
                continue
            if not codes:
                continue

            sentiment = row["sentiment_score"]
            for code in codes:
                bucket = counts.setdefault(
                    code, {"country": code, "count": 0, "sentiment_sum": 0.0,
                           "scored": 0, "max_importance": 0.0}
                )
                bucket["count"] += 1
                if sentiment is not None:
                    bucket["sentiment_sum"] += sentiment
                    bucket["scored"] += 1
                if row["importance_score"]:
                    bucket["max_importance"] = max(
                        bucket["max_importance"], row["importance_score"]
                    )

            if len(recent) < recent_limit:
                recent.append({
                    "id": row["id"],
                    "headline": row["headline"],
                    "source_name": row["source_name"],
                    "published_at": row["published_at"],
                    "countries": codes,
                    "sentiment_score": sentiment,
                    "importance_score": row["importance_score"],
                    "event_type": row["event_type"],
                    "url": row["url"],
                })

        countries = [
            {
                "country": code,
                "count": b["count"],
                "avg_sentiment": (b["sentiment_sum"] / b["scored"]) if b["scored"] else 0.0,
                "max_importance": b["max_importance"],
            }
            for code, b in counts.items()
        ]
        countries.sort(key=lambda c: -c["count"])

        return {
            "countries": countries,
            "recent": recent,
            "window_hours": hours,
            "total_tagged": len(rows),
        }

    def get_geo_backlog_count(self) -> int:
        """Articles still awaiting a country tag."""
        with self.connection() as conn:
            return conn.execute(
                "SELECT COUNT(*) AS c FROM articles WHERE countries IS NULL"
            ).fetchone()["c"]

    # ── User Config ──────────────────────────────────────────────────────

    def get_config(self, key: str, default: str = "{}") -> str:
        """Get a user config value by key."""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT value FROM user_config WHERE key = ?", (key,)
            ).fetchone()
            return row["value"] if row else default

    def set_config(self, key: str, value: str) -> None:
        """Set a user config value (upsert)."""
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO user_config (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = ?, updated_at = ?
                """,
                (
                    key,
                    value,
                    datetime.now(timezone.utc).isoformat(),
                    value,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            
    # ── Multi-Agent Enhancements ─────────────────────────────────────────

    def get_recent_reflections(self, ticker: str, limit: int = 3) -> list[str]:
        """Fetch recent lesson_learned texts for a ticker (backward-compat)."""
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT lesson_learned FROM reflection_log
                WHERE ticker = ?
                ORDER BY date DESC
                LIMIT ?
                """,
                (ticker, limit)
            ).fetchall()
            return [row["lesson_learned"] for row in rows]

    def get_relevant_reflections(self, ticker: str, limit: int = 5) -> dict:
        """Get reflections relevant to a ticker: own + sector + market-wide.

        Returns a structured dict with three sections for better agent prompting:
        ``ticker_lessons``, ``sector_lessons``, ``market_lessons``.
        """
        sector = self.get_ticker_sector(ticker)
        with self.connection() as conn:
            # Ticker-specific (highest priority)
            ticker_rows = conn.execute(
                """SELECT lesson_learned, was_successful, date, scope, sector
                   FROM reflection_log
                   WHERE scope = 'ticker' AND ticker = ?
                   ORDER BY date DESC LIMIT ?""",
                (ticker, limit)
            ).fetchall()

            # Sector reflections
            sector_rows = []
            if sector and sector != "Unknown":
                sector_rows = conn.execute(
                    """SELECT lesson_learned, was_successful, date, scope, sector
                       FROM reflection_log
                       WHERE scope = 'sector' AND sector = ?
                       ORDER BY date DESC LIMIT ?""",
                    (sector, limit)
                ).fetchall()

            # Market-wide reflections (applies to all tickers)
            market_rows = conn.execute(
                """SELECT lesson_learned, was_successful, date, scope, sector
                   FROM reflection_log
                   WHERE scope = 'market'
                   ORDER BY date DESC LIMIT ?""",
                (limit,)
            ).fetchall()

        return {
            "ticker_lessons": [dict(r) for r in ticker_rows],
            "sector_lessons": [dict(r) for r in sector_rows],
            "market_lessons": [dict(r) for r in market_rows],
        }

    def insert_reflection(
        self, ticker: str, prediction_id: int, date: str,
        lesson_learned: str, was_successful: bool,
        scope: str = "ticker", sector: Optional[str] = None,
        tags: Optional[str] = None
    ) -> None:
        """Insert a reflection lesson."""
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO reflection_log
                    (ticker, prediction_id, date, lesson_learned, was_successful, scope, sector, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (ticker, prediction_id, date, lesson_learned, was_successful, scope, sector, tags)
            )

    def get_cached_advisory(self, ticker: str, days: int = 5) -> dict | None:
        """Fetch cached Multi-Agent advisory from the last N days."""
        cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
        with self.connection() as conn:
            row = conn.execute(
                """
                SELECT advisory_json, date FROM predictions_cache 
                WHERE ticker = ? AND date >= ? 
                ORDER BY date DESC LIMIT 1
                """,
                (ticker, cutoff_date)
            ).fetchone()
            if row and row["advisory_json"]:
                try:
                    data = json.loads(row["advisory_json"])
                    data["_cache_date"] = row["date"]
                    return data
                except Exception:
                    return None
            return None

    def set_cached_advisory(self, ticker: str, date: str, advisory_json: str) -> None:
        """Save Multi-Agent advisory to cache."""
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO predictions_cache (ticker, date, advisory_json)
                VALUES (?, ?, ?)
                ON CONFLICT(ticker, date) DO UPDATE SET advisory_json = ?
                """,
                (ticker, date, advisory_json, advisory_json)
            )

    def get_recent_debates(self, limit: int = 50) -> list[dict]:
        """Fetch most recent debate entries across all tickers, newest first."""
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT ticker, date FROM predictions_cache
                ORDER BY date DESC, ticker ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_tracked_tickers(self) -> list[str]:
        """Get the list of tracked tickers for market data."""
        val = self.get_config("tracked_tickers", '["QQQ", "VOO"]')
        try:
            return json.loads(val)
        except json.JSONDecodeError:
            return ["QQQ", "VOO"]
            
    def add_tracked_ticker(self, ticker: str) -> bool:
        """Adds a ticker to the watchlist. Returns True if added, False if already exists."""
        ticker = ticker.upper()
        tickers = self.get_tracked_tickers()
        if ticker in tickers:
            return False
        tickers.append(ticker)
        self.set_config("tracked_tickers", json.dumps(tickers))
        return True
        
    def remove_tracked_ticker(self, ticker: str) -> bool:
        """Removes a ticker from the watchlist. Returns True if removed, False if not found."""
        ticker = ticker.upper()
        tickers = self.get_tracked_tickers()
        if ticker not in tickers:
            return False
        tickers.remove(ticker)
        self.set_config("tracked_tickers", json.dumps(tickers))
        return True

    def get_ticker_sector(self, ticker: str) -> str:
        """Fetch ticker sector, caching it in the DB to avoid repeated yfinance calls."""
        ticker = ticker.upper()
        with self.connection() as conn:
            row = conn.execute("SELECT sector FROM ticker_info WHERE ticker = ?", (ticker,)).fetchone()
            if row and row["sector"]:
                return row["sector"]
        
        # If not cached, fetch via yfinance
        try:
            import yfinance as yf
            info = yf.Ticker(ticker).info
            sector = info.get("sector", "Unknown")
        except Exception as e:
            log.warning("database.get_sector_failed", ticker=ticker, error=str(e))
            sector = "Unknown"
            
        with self.connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO ticker_info (ticker, sector, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)",
                (ticker, sector)
            )
            
        return sector

    # ── Alert History ────────────────────────────────────────────────────

    def was_alert_sent(self, article_id: str, alert_type: str) -> bool:
        """Check if an alert was already sent for this article."""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM sent_alerts WHERE article_id = ? AND alert_type = ?",
                (article_id, alert_type)
            ).fetchone()
            return bool(row)

    def record_price_alert(self, ticker: str) -> None:
        """Record that a price alert was sent for a ticker today."""
        with self.connection() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO price_alerts (ticker, alert_date) VALUES (?, date('now', 'localtime'))",
                (ticker,)
            )

    def was_price_alert_sent_today(self, ticker: str) -> bool:
        """Check if a price alert was already sent for this ticker today."""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM price_alerts WHERE ticker = ? AND alert_date = date('now', 'localtime')",
                (ticker,)
            ).fetchone()
            return row is not None

    def record_alert(self, article_id: str, alert_type: str) -> None:
        """Record that an alert was sent."""
        with self.connection() as conn:
            try:
                conn.execute(
                    "INSERT INTO sent_alerts (article_id, alert_type) VALUES (?, ?)",
                    (article_id, alert_type),
                )
            except sqlite3.IntegrityError:
                pass  # Already recorded

    # ── Stats ────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Get database statistics for the /status command."""
        with self.connection() as conn:
            total = conn.execute("SELECT COUNT(*) as c FROM articles").fetchone()["c"]
            classified = conn.execute(
                "SELECT COUNT(*) as c FROM articles WHERE event_type IS NOT NULL AND event_type != 'noise'"
            ).fetchone()["c"]
            noise = conn.execute(
                "SELECT COUNT(*) as c FROM articles WHERE event_type = 'noise'"
            ).fetchone()["c"]
            embedded = conn.execute(
                "SELECT COUNT(*) as c FROM articles WHERE embedding IS NOT NULL"
            ).fetchone()["c"]
            duplicates = conn.execute(
                "SELECT COUNT(*) as c FROM articles WHERE duplicate_of IS NOT NULL"
            ).fetchone()["c"]
            sources = conn.execute(
                """
                SELECT source_name, COUNT(*) as c, MAX(fetched_at) as last_fetch
                FROM articles WHERE duplicate_of IS NULL GROUP BY source_name
                """
            ).fetchall()

            db_size_bytes = Path(self.db_path).stat().st_size if Path(self.db_path).exists() else 0

            return {
                "total_articles": total,
                "classified_articles": classified,
                "noise_articles": noise,
                "embedded_articles": embedded,
                "duplicate_articles": duplicates,
                "db_size_mb": round(db_size_bytes / (1024 * 1024), 2),
                "sources": [dict(s) for s in sources],
            }

    # ── Usage Tracking ───────────────────────────────────────────────────

    def log_llm_usage(
        self, 
        model_name: str, 
        operation: str, 
        prompt_tokens: int, 
        candidate_tokens: int,
        latency_ms: Optional[int] = None,
        is_error: bool = False,
        error_message: Optional[str] = None,
        prompt_text: Optional[str] = None,
        response_text: Optional[str] = None
    ) -> None:
        """Logs LLM token usage, latencies, errors, and text."""
        price_per_prompt, price_per_candidate = _resolve_pricing(model_name)

        cost_usd = (prompt_tokens * price_per_prompt) + (candidate_tokens * price_per_candidate)
        total_tokens = prompt_tokens + candidate_tokens

        try:
            with self.connection() as conn:
                conn.execute(
                    """
                    INSERT INTO llm_usage_log 
                    (model_name, operation, prompt_tokens, candidate_tokens, total_tokens, cost_usd, latency_ms, is_error, error_message, prompt_text, response_text)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (model_name, operation, prompt_tokens, candidate_tokens, total_tokens, cost_usd, latency_ms, 1 if is_error else 0, error_message, prompt_text, response_text)
                )
            log.debug("db.usage_logged", operation=operation, cost_usd=cost_usd, is_error=is_error)
        except sqlite3.Error as e:
            log.error("db.usage_log_failed", error=str(e))

    def get_usage_stats(self, days: Optional[int] = None) -> dict:
        """
        Usage totals plus a per-day/model/operation breakdown.

        `total_tokens` / `total_cost_usd` and `details` are computed over the
        SAME window, so the per-model table always sums to the headline. Pass
        `days=None` for all time. All-time figures are returned alongside under
        `all_time_*` so a windowed view can still show the lifetime number
        without conflating the two — previously the headline was all-time while
        `details` was hardcoded to 7 days, and the two could never reconcile.
        """
        try:
            with self.connection() as conn:
                if days is not None:
                    window_clause = "WHERE timestamp >= date('now', ?)"
                    params: tuple = (f"-{int(days)} days",)
                else:
                    window_clause = ""
                    params = ()

                row = conn.execute(
                    f"""
                    SELECT SUM(total_tokens) as t, SUM(cost_usd) as c,
                           SUM(prompt_tokens) as p, SUM(candidate_tokens) as k,
                           COUNT(*) as n
                    FROM llm_usage_log {window_clause}
                    """,
                    params,
                ).fetchone()

                all_time = conn.execute(
                    "SELECT SUM(total_tokens) as t, SUM(cost_usd) as c FROM llm_usage_log"
                ).fetchone()

                details_rows = conn.execute(
                    f"""
                    SELECT date(timestamp) as day, model_name, operation,
                           SUM(total_tokens) as tokens, SUM(cost_usd) as cost,
                           COUNT(*) as requests_count,
                           SUM(is_error) as error_count,
                           AVG(latency_ms) as avg_latency_ms
                    FROM llm_usage_log {window_clause}
                    GROUP BY day, model_name, operation
                    ORDER BY day DESC, cost DESC
                    """,
                    params,
                ).fetchall()

                return {
                    "window_days": days,
                    "total_tokens": row["t"] or 0,
                    "total_cost_usd": row["c"] or 0.0,
                    "total_prompt_tokens": row["p"] or 0,
                    "total_candidate_tokens": row["k"] or 0,
                    "total_requests": row["n"] or 0,
                    "all_time_tokens": all_time["t"] or 0,
                    "all_time_cost_usd": all_time["c"] or 0.0,
                    "details": [dict(r) for r in details_rows],
                }
        except sqlite3.Error as e:
            log.error("db.usage_stats_failed", error=str(e))
            return {
                "window_days": days,
                "total_tokens": 0, "total_cost_usd": 0.0,
                "total_prompt_tokens": 0, "total_candidate_tokens": 0,
                "total_requests": 0,
                "all_time_tokens": 0, "all_time_cost_usd": 0.0,
                "details": [],
            }

    def check_for_api_spikes(self) -> Optional[str]:
        """
        Checks for unusual API usage spikes.
        Condition 1: > 50 requests in the last hour.
        Condition 2: Today's usage is > 200% higher than the 7-day daily average.
        Returns an alert message string if a spike is detected, else None.
        """
        try:
            with self.connection() as conn:
                # Burst check: > 50 requests in last 1 hour
                burst_count = conn.execute(
                    "SELECT COUNT(*) as c FROM llm_usage_log WHERE timestamp >= datetime('now', '-1 hour')"
                ).fetchone()["c"]
                
                if burst_count > 50:
                    last_alert = self.get_config("last_api_burst_alert_time", "")
                    now_iso = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
                    # Only alert if we haven't alerted in the last hour
                    if not last_alert or (datetime.now(timezone.utc) - datetime.fromisoformat(last_alert)).total_seconds() > 3600:
                        self.set_config("last_api_burst_alert_time", now_iso)
                        return f"⚠️ <b>API Burst Alert</b>: High volume of requests detected ({burst_count} in the last hour)."

                # Percentage check: today's tokens vs 7-day average
                today_tokens = conn.execute(
                    "SELECT SUM(total_tokens) as t FROM llm_usage_log WHERE date(timestamp) = date('now', 'localtime')"
                ).fetchone()["t"] or 0
                
                if today_tokens > 0:
                    # Calculate average daily tokens for the 7 days prior to today
                    historical_avg = conn.execute(
                        """
                        SELECT AVG(daily_tokens) as avg_tokens FROM (
                            SELECT date(timestamp) as d, SUM(total_tokens) as daily_tokens 
                            FROM llm_usage_log 
                            WHERE date(timestamp) < date('now', 'localtime') 
                              AND date(timestamp) >= date('now', 'localtime', '-7 days')
                            GROUP BY d
                        )
                        """
                    ).fetchone()["avg_tokens"] or 0
                    
                    if historical_avg > 0 and today_tokens > historical_avg * 3: # > 200% higher means > 3x
                        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                        last_daily_alert = self.get_config("last_api_daily_alert_date", "")
                        if last_daily_alert != today_str:
                            self.set_config("last_api_daily_alert_date", today_str)
                            return f"📈 <b>API Usage Spike</b>: Today's token usage ({today_tokens:,}) is over 200% higher than the 7-day average ({int(historical_avg):,})."

        except sqlite3.Error as e:
            log.error("db.check_spikes_failed", error=str(e))
            
        return None

    # ── ML Predictions ───────────────────────────────────────────────────

    def insert_prediction(self, prediction_data: dict) -> str:
        """Insert a new prediction row, returns ID."""
        import uuid
        pred_id = prediction_data.get("id") or str(uuid.uuid4())
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO predictions (
                    id, ticker, predicted_direction, confidence, horizon_days,
                    model_type, feature_snapshot, llm_narrative, resolve_after
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pred_id,
                    prediction_data["ticker"],
                    prediction_data["predicted_direction"],
                    prediction_data["confidence"],
                    prediction_data.get("horizon_days", 1),
                    prediction_data["model_type"],
                    json.dumps(prediction_data.get("feature_snapshot", {})),
                    prediction_data.get("llm_narrative", ""),
                    prediction_data["resolve_after"]
                )
            )
        return pred_id

    def get_existing_prediction(self, ticker: str, horizon_days: int, date: str) -> dict | None:
        """Check for cached prediction on a specific date (date string format YYYY-MM-DD)."""
        with self.connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM predictions
                WHERE ticker = ? AND horizon_days = ? AND date(created_at) = date(?)
                ORDER BY created_at DESC LIMIT 1
                """,
                (ticker, horizon_days, date)
            ).fetchone()
            if not row:
                return None
            result = dict(row)
            if result.get("feature_snapshot"):
                result["feature_snapshot"] = json.loads(result["feature_snapshot"])
            return result

    def get_unresolved_predictions(self) -> list[dict]:
        """Predictions where is_correct IS NULL and resolve_after <= today."""
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM predictions
                WHERE is_correct IS NULL AND resolve_after <= date('now', 'localtime')
                """
            ).fetchall()
            results = []
            for row in rows:
                r = dict(row)
                if r.get("feature_snapshot"):
                    r["feature_snapshot"] = json.loads(r["feature_snapshot"])
                results.append(r)
            return results

    def resolve_prediction(self, prediction_id: str, actual_direction: str, actual_change_pct: float, is_correct: bool) -> None:
        """Fills resolution fields."""
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE predictions SET
                    actual_direction = ?,
                    actual_change_pct = ?,
                    is_correct = ?,
                    resolved_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (actual_direction, actual_change_pct, int(is_correct), prediction_id)
            )

    def get_prediction_accuracy(self, ticker: str = None) -> dict:
        """Aggregated accuracy stats."""
        with self.connection() as conn:
            query = "SELECT COUNT(*) as total, SUM(is_correct) as correct FROM predictions WHERE is_correct IS NOT NULL"
            params = ()
            if ticker:
                query += " AND ticker = ?"
                params = (ticker,)
            
            row = conn.execute(query, params).fetchone()
            total = row["total"] or 0
            correct = row["correct"] or 0
            incorrect = total - correct
            accuracy = (correct / total * 100) if total > 0 else 0.0
            
            return {
                "total": total,
                "correct": correct,
                "incorrect": incorrect,
                "accuracy_pct": accuracy
            }

    def get_recent_predictions(self, ticker: str = None, limit: int = 10) -> list[dict]:
        """Recent predictions with outcomes."""
        with self.connection() as conn:
            query = "SELECT * FROM predictions"
            params = ()
            if ticker:
                query += " WHERE ticker = ?"
                params = (ticker,)
            query += " ORDER BY created_at DESC LIMIT ?"
            params += (limit,)
            
            rows = conn.execute(query, params).fetchall()
            results = []
            for row in rows:
                r = dict(row)
                if r.get("feature_snapshot"):
                    r["feature_snapshot"] = json.loads(r["feature_snapshot"])
                results.append(r)
            return results

    # ── Price History ────────────────────────────────────────────────────

    def upsert_price_history(self, ticker: str, rows: list[dict]) -> None:
        """Bulk INSERT OR REPLACE for OHLCV data."""
        with self.connection() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO price_history (
                    ticker, date, open, high, low, close, volume
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (ticker, r["date"], r["open"], r["high"], r["low"], r["close"], r.get("volume", 0))
                    for r in rows
                ]
            )

    # ── Sentiment Features ───────────────────────────────────────────────

    def get_ticker_sentiment_features(
        self,
        ticker: str,
        lookback_days: int = 7,
        as_of: Optional[datetime] = None,
    ) -> dict:
        """Aggregate sentiment from ticker_mentions + articles with temporal granularity.

        Returns separate 1d/3d/7d sentiment averages so the model can distinguish
        between fresh vs stale sentiment signals. Also computes momentum (1d - 7d)
        and news velocity (recent article rate vs historical baseline).

        Args:
            as_of: Point in time to evaluate from. Defaults to now. Every window is
                bounded on BOTH sides by this — during model training the caller
                walks backwards through history, and an unbounded upper edge would
                feed the model news that had not been published yet.
        """
        with self.connection() as conn:
            now = as_of or datetime.now(timezone.utc)
            if now.tzinfo is None:
                now = now.replace(tzinfo=timezone.utc)
            upper = now.isoformat()
            cutoff_1d = (now - timedelta(days=1)).isoformat()
            cutoff_3d = (now - timedelta(days=3)).isoformat()
            cutoff_7d = (now - timedelta(days=7)).isoformat()
            cutoff_14d = (now - timedelta(days=14)).isoformat()
            cutoff_lookback = (now - timedelta(days=lookback_days)).isoformat()

            # ── Sentiment averages over 1d, 3d, 7d windows ──
            def _avg_sentiment(cutoff):
                row = conn.execute(
                    """
                    SELECT AVG(sentiment_score) as avg_s, COUNT(*) as cnt
                    FROM ticker_mentions
                    WHERE ticker = ? AND mentioned_at >= ? AND mentioned_at <= ?
                      AND sentiment_score IS NOT NULL
                    """,
                    (ticker, cutoff, upper)
                ).fetchone()
                return (row["avg_s"] or 0.0, row["cnt"] or 0)

            avg_1d, count_1d = _avg_sentiment(cutoff_1d)
            avg_3d, count_3d = _avg_sentiment(cutoff_3d)
            avg_7d, count_7d = _avg_sentiment(cutoff_7d)

            # Sentiment momentum: how much has sentiment shifted recently vs baseline
            sentiment_momentum = avg_1d - avg_7d

            # ── News velocity: articles per day in last 3d vs last 14d ──
            _, count_14d = _avg_sentiment(cutoff_14d)
            recent_rate = count_3d / 3.0 if count_3d else 0.0
            baseline_rate = count_14d / 14.0 if count_14d else 0.0
            news_velocity = (recent_rate / baseline_rate) if baseline_rate > 0 else 1.0

            # ── Article-level features (importance, direction, urgency) ──
            articles = conn.execute(
                """
                SELECT a.importance_score, a.suggested_direction, a.urgency
                FROM articles a
                JOIN ticker_mentions tm ON a.id = tm.article_id
                WHERE tm.ticker = ? AND a.published_at >= ? AND a.published_at <= ?
                """,
                (ticker, cutoff_lookback, upper)
            ).fetchall()
            
            importance_scores = [a["importance_score"] for a in articles if a["importance_score"] is not None]
            avg_importance = sum(importance_scores) / len(importance_scores) if importance_scores else 0.0
            
            bullish_count = sum(1 for a in articles if a["suggested_direction"] == "bullish")
            total_direction = sum(1 for a in articles if a["suggested_direction"] in ("bullish", "bearish"))
            bullish_ratio = bullish_count / total_direction if total_direction > 0 else 0.5
            
            urgency_map = {"low": 0, "medium": 1, "high": 2, "critical": 3}
            max_urgency = max([urgency_map.get(a["urgency"], 0) for a in articles] + [0])

            return {
                "sentiment_avg_1d": float(avg_1d),
                "sentiment_avg_3d": float(avg_3d),
                "sentiment_avg_7d": float(avg_7d),
                "sentiment_momentum": float(sentiment_momentum),
                "news_velocity": float(news_velocity),
                "avg_importance": float(avg_importance),
                "bullish_ratio": float(bullish_ratio),
                "max_urgency_24h": float(max_urgency),
            }

    # ── Smart money: insider / institutional / KR flows ──────────────────

    def upsert_insider_transactions(self, rows: list[dict]) -> int:
        """Insert Form 4 transaction rows, ignoring ones already stored.

        Returns the number of new rows. INSERT OR IGNORE on the accession-derived
        primary key makes re-scanning a window idempotent, which matters because
        the scheduled job always re-reads a few days of overlap.
        """
        if not rows:
            return 0
        inserted = 0
        with self.connection() as conn:
            for r in rows:
                cur = conn.execute(
                    """
                    INSERT OR IGNORE INTO insider_transactions (
                        id, ticker, issuer_cik, insider_name, insider_title,
                        is_officer, is_director, is_ten_pct_owner,
                        transaction_date, filed_at, transaction_code,
                        is_discretionary, shares, price_per_share, value_usd,
                        shares_owned_after, is_10b5_1, accession_no, raw_data
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        r["id"], r["ticker"], r.get("issuer_cik"),
                        r.get("insider_name"), r.get("insider_title"),
                        r.get("is_officer", 0), r.get("is_director", 0),
                        r.get("is_ten_pct_owner", 0),
                        r["transaction_date"], r["filed_at"],
                        r.get("transaction_code"), r.get("is_discretionary", 0),
                        r.get("shares"), r.get("price_per_share"), r.get("value_usd"),
                        r.get("shares_owned_after"), r.get("is_10b5_1"),
                        r.get("accession_no"), json.dumps(r.get("raw_data", {})),
                    ),
                )
                inserted += cur.rowcount or 0
        return inserted

    def upsert_institutional_stakes(self, rows: list[dict]) -> int:
        """Insert 13D/13G filing rows, ignoring duplicates."""
        if not rows:
            return 0
        inserted = 0
        with self.connection() as conn:
            for r in rows:
                cur = conn.execute(
                    """
                    INSERT OR IGNORE INTO institutional_stakes (
                        id, ticker, filer_name, filer_cik, form_type,
                        is_activist, is_amendment, pct_of_class, shares,
                        event_date, filed_at, accession_no, raw_data
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        r["id"], r["ticker"], r.get("filer_name"), r.get("filer_cik"),
                        r.get("form_type"), r.get("is_activist", 0),
                        r.get("is_amendment", 0), r.get("pct_of_class"), r.get("shares"),
                        r.get("event_date"), r["filed_at"], r.get("accession_no"),
                        json.dumps(r.get("raw_data", {})),
                    ),
                )
                inserted += cur.rowcount or 0
        return inserted

    def upsert_kr_flows(self, rows: list[dict]) -> int:
        """Insert or replace daily KRX investor-flow rows."""
        if not rows:
            return 0
        with self.connection() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO kr_investor_flows (
                    ticker, trade_date, inst_net, foreign_net, retail_net,
                    pension_net, financial_inv_net, trust_net, total_value,
                    flow_unit, source
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                [
                    (
                        r["ticker"], r["trade_date"], r.get("inst_net"),
                        r.get("foreign_net"), r.get("retail_net"),
                        r.get("pension_net"), r.get("financial_inv_net"),
                        r.get("trust_net"), r.get("total_value"),
                        r.get("flow_unit", "shares"), r.get("source", "naver"),
                    )
                    for r in rows
                ],
            )
        return len(rows)

    def get_insider_series(self, ticker: str) -> list[dict]:
        """Full insider history for one ticker, ordered by disclosure date.

        Returned whole rather than windowed because model training evaluates
        ~1200 as-of dates per ticker; the caller caches this once and slices it
        in memory instead of opening a connection per date.
        """
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT filed_at, transaction_date, transaction_code,
                       is_discretionary, insider_name, insider_title,
                       is_officer, is_director, is_ten_pct_owner,
                       shares, price_per_share, value_usd, is_10b5_1
                FROM insider_transactions
                WHERE ticker = ?
                ORDER BY filed_at ASC
                """,
                (ticker.upper(),),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_stakes_series(self, ticker: str) -> list[dict]:
        """Full 13D/13G filing history for one ticker, ordered by disclosure date."""
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT filed_at, form_type, is_activist, is_amendment,
                       filer_name, pct_of_class, shares, event_date
                FROM institutional_stakes
                WHERE ticker = ?
                ORDER BY filed_at ASC
                """,
                (ticker.upper(),),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_kr_flow_series(self, ticker: str) -> list[dict]:
        """Full daily KRX investor-flow history for one ticker."""
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT trade_date, inst_net, foreign_net, retail_net,
                       pension_net, financial_inv_net, trust_net, total_value,
                       flow_unit, source
                FROM kr_investor_flows
                WHERE ticker = ?
                ORDER BY trade_date ASC
                """,
                (ticker,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_last_insider_filed_at(self, ticker: str) -> Optional[str]:
        """Most recent disclosure date stored, so a sync can resume from there."""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT MAX(filed_at) AS m FROM insider_transactions WHERE ticker = ?",
                (ticker.upper(),),
            ).fetchone()
        return row["m"] if row and row["m"] else None

    def get_recent_insider_activity(self, days: int = 30, limit: int = 100,
                                    discretionary_only: bool = True) -> list[dict]:
        """Cross-ticker recent insider transactions, newest disclosure first."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        clause = "AND is_discretionary = 1" if discretionary_only else ""
        with self.connection() as conn:
            rows = conn.execute(
                f"""
                SELECT ticker, insider_name, insider_title, is_officer, is_director,
                       is_ten_pct_owner, transaction_date, filed_at, transaction_code,
                       shares, price_per_share, value_usd, is_10b5_1
                FROM insider_transactions
                WHERE filed_at >= ? {clause}
                ORDER BY filed_at DESC, ABS(COALESCE(value_usd, 0)) DESC
                LIMIT ?
                """,
                (cutoff, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_recent_stakes(self, days: int = 90, limit: int = 50) -> list[dict]:
        """Cross-ticker recent 13D/13G filings, newest first."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT ticker, filer_name, form_type, is_activist, is_amendment,
                       pct_of_class, shares, event_date, filed_at
                FROM institutional_stakes
                WHERE filed_at >= ?
                ORDER BY filed_at DESC
                LIMIT ?
                """,
                (cutoff, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Conversations & Messages ─────────────────────────────────────────

    def create_conversation(self, conversation_id: str, title: str = "New Conversation") -> None:
        """Create a new conversation."""
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO conversations (id, title)
                VALUES (?, ?)
                """,
                (conversation_id, title)
            )

    def get_conversation(self, conversation_id: str) -> Optional[dict]:
        """Get a conversation by ID."""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
            return dict(row) if row else None

    def insert_message(self, message_id: str, conversation_id: str, role: str, content: str) -> None:
        """Insert a message into a conversation."""
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO messages (id, conversation_id, role, content)
                VALUES (?, ?, ?, ?)
                """,
                (message_id, conversation_id, role, content)
            )
            conn.execute(
                """
                UPDATE conversations SET updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (conversation_id,)
            )

    def get_messages(self, conversation_id: str, limit: int = 50) -> list[dict]:
        """Get messages for a conversation, ordered by timestamp ascending."""
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM messages
                WHERE conversation_id = ?
                ORDER BY timestamp ASC
                LIMIT ?
                """,
                (conversation_id, limit)
            ).fetchall()
            return [dict(row) for row in rows]

    def insert_pipeline_metrics(self, duration_seconds: float, counts: dict) -> None:
        """Record a pipeline cycle's timing and throughput metrics."""
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO pipeline_metrics (
                    cycle_duration_seconds, articles_fetched, articles_inserted,
                    articles_classified, articles_ranked, articles_embedded,
                    alerts_generated, errors_count, llm_calls_count, llm_cost_estimate
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    duration_seconds,
                    counts.get("fetched", 0),
                    counts.get("inserted", 0),
                    counts.get("classified", 0),
                    counts.get("ranked", 0),
                    counts.get("embedded", 0),
                    counts.get("alerts", 0),
                    counts.get("errors", 0),
                    counts.get("llm_calls", 0),
                    counts.get("llm_cost", 0.0),
                ),
            )

    def get_recent_pipeline_metrics(self, limit: int = 20) -> list[dict]:
        """Get the most recent pipeline cycle metrics."""
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM pipeline_metrics ORDER BY recorded_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]
