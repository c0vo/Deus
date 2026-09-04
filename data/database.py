"""
Deus — SQLite Database Manager

Manages the SQLite database: connection lifecycle, schema creation,
and common query helpers. Uses FTS5 for full-text search.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Generator, Optional

import numpy as np

from config.logging_config import get_logger
from config.settings import settings
from data.models import NewsArticle

log = get_logger(__name__)

# ── LLM cost ─────────────────────────────────────────────────────────────
#
# There is no price table here any more, on purpose.
#
# There used to be one — USD per 1M tokens, keyed on the exact model name — and
# it went stale exactly as fast as providers repriced. At the point it was
# removed it billed deepseek-v4-pro at $1.25/$5.00 against an actual
# $0.435/$0.87, a 3x over-report on the second most expensive operation in the
# system, and no Gemini entry matched a model that was still in use.
#
# Every response now carries the cost the provider actually charged, so that is
# what gets stored. A call that reports nothing records 0.0 and is counted
# separately as an unpriced call rather than being quietly estimated — an
# under-report you can see beats an over-report you cannot.

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

-- FINRA off-exchange volume — the free dark-pool proxy.
--
-- CNMSshvol reports only trades printed to a FINRA TRF, i.e. everything that
-- did NOT execute on a lit exchange. Measured off-exchange share runs 33-47% of
-- consolidated volume for large caps, so total_volume here is NOT the day's
-- volume — price_history.volume is, and the ratio between them is the signal.
--
-- session_date is when the trading happened; published_at is when FINRA posted
-- the file (~18:00 ET the same session) and is the as-of key features filter on.
-- The gap is hours rather than Form 4's days, but the column is kept for the
-- same reason: every feature query filters on one consistent kind of date.
--
-- Volumes are REAL, not INTEGER. FINRA reports fractional share-adjusted
-- figures (e.g. 5540409.463985) and int() would throw on every row.
CREATE TABLE IF NOT EXISTS offexchange_volume (
    ticker TEXT NOT NULL,
    session_date DATE NOT NULL,
    short_volume REAL,
    short_exempt_volume REAL,
    total_volume REAL,                   -- off-exchange only
    market_codes TEXT,                   -- 'B,Q,N' — which TRFs reported
    published_at DATETIME NOT NULL,      -- as-of key
    source TEXT DEFAULT 'finra_cnms',
    PRIMARY KEY (ticker, session_date)
);
CREATE INDEX IF NOT EXISTS idx_offexch_ticker_pub
    ON offexchange_volume(ticker, published_at);
CREATE INDEX IF NOT EXISTS idx_offexch_session ON offexchange_volume(session_date);

-- Market-wide regime series, long form: one row per (metric, session).
--
-- Long rather than wide because the feeds publish independently — a wide table
-- forces one shared row per date, so a vendor that has not published yet writes
-- NULLs indistinguishable from a real zero. Long form also means a fourth
-- series is a new `metric` value rather than a schema change, which matters
-- because this project has no migration framework.
--
-- Unlike everything else in this section these rows are not per-ticker: they
-- describe the whole market, and the predictor loads them once under a sentinel
-- cache key rather than once per symbol.
CREATE TABLE IF NOT EXISTS market_regime_daily (
    metric TEXT NOT NULL,                -- 'dix' | 'gex' | 'occ_put_call_ratio'
    session_date DATE NOT NULL,
    value REAL NOT NULL,
    published_at DATETIME NOT NULL,      -- as-of key
    source TEXT DEFAULT '',              -- 'squeezemetrics' | 'occ'
    PRIMARY KEY (metric, session_date)
);
CREATE INDEX IF NOT EXISTS idx_regime_pub ON market_regime_daily(published_at);

-- Daily option-chain aggregates, one row per ticker per session.
--
-- Pre-aggregated deliberately. A per-contract table would add ~1,000 rows per
-- ticker per day (~3M rows/year for a dozen tickers) to buy back nothing the
-- aggregates below do not already carry.
--
-- This table is the one part of the system that cannot be backfilled: yfinance
-- exposes only the current chain, with no history and no Greeks, so the panel
-- accrues exactly one session per day and no amount of catch-up recovers a
-- session that was not captured. That is why the collector runs even though no
-- feature reads this yet — features arrive at FEATURE_SCHEMA_VERSION 4, once
-- roughly a year of sessions exists.
--
-- Every IV column is computed under a filter (out-of-the-money, non-zero bid,
-- near the money) because Yahoo's implied vols are unusable on deep-ITM and
-- illiquid strikes — a deep-ITM AAPL call quotes 133% IV with spot at 313.
-- See pipeline.options_flow for the exact filter.
CREATE TABLE IF NOT EXISTS option_chain_daily (
    ticker TEXT NOT NULL,
    session_date DATE NOT NULL,
    spot_price REAL,
    call_volume REAL,
    put_volume REAL,
    call_oi REAL,
    put_oi REAL,
    put_call_volume_ratio REAL,
    put_call_oi_ratio REAL,
    atm_iv REAL,                         -- front expiry, nearest strike
    iv_skew REAL,                        -- OTM put IV minus OTM call IV
    near_term_iv REAL,                   -- front expiry ATM
    far_term_iv REAL,                    -- longest captured expiry ATM
    expirations_seen INTEGER,
    contracts_seen INTEGER,
    published_at DATETIME NOT NULL,      -- as-of key: the snapshot instant
    source TEXT DEFAULT 'yfinance',
    PRIMARY KEY (ticker, session_date)
);
CREATE INDEX IF NOT EXISTS idx_optchain_ticker_pub
    ON option_chain_daily(ticker, published_at);

-- Analyst consensus and price targets, one row per ticker per session.
--
-- Sell-side opinion: how many analysts say buy, and where they think the price
-- is going. Fetched, never computed — these are other people's published views.
--
-- Shares option_chain_daily's defining constraint: the free sources expose only
-- the CURRENT consensus, so this accrues one session per day and a missed
-- session is gone permanently. That is also why no predictor feature reads it.
-- Training on a single repeated snapshot stamped across historical dates would
-- leak today's opinion into every past row; features wait until real
-- point-in-time history exists.
--
-- recommendation_mean follows Yahoo's scale, where LOW IS BULLISH: 1.0 is a
-- unanimous strong buy and 5.0 a unanimous strong sell. Anything ranking or
-- charting this column has to invert it, which is why the raw buckets are
-- stored alongside rather than only the mean.
CREATE TABLE IF NOT EXISTS analyst_consensus_daily (
    ticker TEXT NOT NULL,
    session_date DATE NOT NULL,
    strong_buy INTEGER,
    buy INTEGER,
    hold INTEGER,
    sell INTEGER,
    strong_sell INTEGER,
    analyst_count INTEGER,               -- Yahoo's numberOfAnalystOpinions
    recommendation_key TEXT,             -- 'strong_buy' | 'buy' | 'hold' | ...
    recommendation_mean REAL,            -- 1.0 bullish .. 5.0 bearish
    target_mean REAL,
    target_high REAL,
    target_low REAL,
    target_median REAL,
    spot_price REAL,                     -- price when captured, so the implied
                                         -- upside stays reproducible later
    published_at DATETIME NOT NULL,      -- as-of key: the snapshot instant
    source TEXT DEFAULT 'yfinance',
    PRIMARY KEY (ticker, session_date)
);
CREATE INDEX IF NOT EXISTS idx_analyst_ticker_pub
    ON analyst_consensus_daily(ticker, published_at);

-- Technical ratings, one row per ticker per timeframe per session.
--
-- The TradingView-methodology bull/bear gauge: 26 indicators each vote +1/0/-1,
-- averaged per group and bucketed into STRONG_SELL..STRONG_BUY. See
-- pipeline.technical_rating for the vote rules and their provenance.
--
-- timeframe is part of the primary key because 'short'/'medium'/'long' are the
-- same formula over daily/weekly/monthly bars — three independent ratings for
-- one ticker on one session, not three columns of one rating.
--
-- Unlike the two tables above this one IS backfillable: a rating is a pure
-- function of stored OHLCV, so deepening price_history retroactively populates
-- every past session.
--
-- summary_score is the mean of the two GROUP scores, not of all 26 votes, which
-- over-weights the moving averages relative to a flat average. That is
-- TradingView's definition; bars_available is kept so a rating computed on a
-- thin history is identifiable after the fact.
CREATE TABLE IF NOT EXISTS technical_rating_daily (
    ticker TEXT NOT NULL,
    session_date DATE NOT NULL,
    timeframe TEXT NOT NULL,             -- 'short' | 'medium' | 'long'
    summary_score REAL,                  -- -1.0 .. +1.0
    summary_label TEXT,
    ma_score REAL,                       -- 15 moving-average votes
    ma_label TEXT,
    osc_score REAL,                      -- 11 oscillator votes
    osc_label TEXT,
    buy_votes INTEGER,
    neutral_votes INTEGER,
    sell_votes INTEGER,
    bars_available INTEGER,
    published_at DATETIME NOT NULL,      -- as-of key
    PRIMARY KEY (ticker, session_date, timeframe)
);
CREATE INDEX IF NOT EXISTS idx_techrating_ticker_pub
    ON technical_rating_daily(ticker, published_at);

-- Cross-process SSE outbox.
--
-- The ingest pipeline runs in a separate process from the API (see worker.py),
-- so the in-memory event bus in api/sse_manager.py can no longer reach the
-- dashboard's SSE subscribers. Publishers INSERT here; the API process tails
-- the table and fans events out to connected clients.
--
-- This is a relay, not a log — rows are trimmed on a timer. It also means SSE
-- now survives an API restart, which the in-memory bus never did.
CREATE TABLE IF NOT EXISTS sse_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_sse_events_created ON sse_events(created_at);

-- Last known quote per ticker, refreshed by the worker's price_feed job.
--
-- /api/markets reads this instead of calling Yahoo inline. The ticker tape
-- polls that endpoint from every open tab, and doing one live HTTP request per
-- tracked ticker per poll made the response take seconds and drew Yahoo rate
-- limiting, which then tied up executor threads and slowed everything else.
CREATE TABLE IF NOT EXISTS latest_prices (
    ticker TEXT PRIMARY KEY,
    price REAL NOT NULL,
    previous_close REAL,
    daily_change_pct REAL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ── Thesis Engine ────────────────────────────────────────────────────
--
-- A "thesis" is a causal chain: an emerging theme, decomposed into the
-- bottlenecks it creates, with the companies positioned at each one.
--
-- The point is to reach the second- and third-order chokepoints before the
-- market connects them to the theme. The first-order beneficiary is already
-- priced in by the time it reaches the news, so every candidate carries two
-- independent scores: conviction (how load-bearing the company is to the
-- chain) and crowding (how priced-in it already is). edge_score multiplies
-- conviction by (1 - crowding), which is the whole "buy the rumour" premise
-- reduced to one sortable number.
CREATE TABLE IF NOT EXISTS theses (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    summary TEXT DEFAULT '',
    -- 'auto' = detected from article-embedding clusters, 'user' = seeded by hand.
    seed_kind TEXT DEFAULT 'auto',
    seed_query TEXT DEFAULT '',
    -- SHA1 of the seeding cluster's top member article ids. Regenerating the
    -- same theme supersedes the old row instead of duplicating it; centroids
    -- are not comparable across runs because the space is mean-centered.
    seed_fingerprint TEXT,
    -- Tickers the seed articles already name. Handed to the model as the
    -- explicitly priced-in set it must reason PAST, not toward.
    consensus_tickers_json TEXT DEFAULT '[]',
    model_name TEXT,
    -- Volume growth of the seeding cluster: recent window vs prior baseline.
    -- This is the selector that distinguishes an emerging theme from a loud one.
    -- NULL means "not computable", which is different from 0.0 ("flat").
    -- Ingest is bursty enough that a raw count ratio would measure worker
    -- uptime, so this is share-of-voice normalised and withheld below a floor.
    acceleration REAL,
    acceleration_basis TEXT DEFAULT 'sov',
    article_count_recent INTEGER DEFAULT 0,
    article_count_base INTEGER DEFAULT 0,
    evidence_json TEXT DEFAULT '[]',
    status TEXT DEFAULT 'active',
    generated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    refreshed_at DATETIME,
    expires_at DATETIME
);
CREATE INDEX IF NOT EXISTS idx_theses_status ON theses(status, generated_at DESC);

-- One node per causal hop. Stored flat with parent_id rather than nested,
-- because the LLM emits a flat list keyed by node_key — nested tree JSON is
-- where structured output reliably breaks.
CREATE TABLE IF NOT EXISTS thesis_nodes (
    id TEXT PRIMARY KEY,
    thesis_id TEXT NOT NULL REFERENCES theses(id),
    parent_id TEXT,
    node_key TEXT NOT NULL,
    -- 1 = first-order (already priced in), 2-3 = where the value is.
    order_depth INTEGER NOT NULL DEFAULT 1,
    claim TEXT NOT NULL,
    -- Why the parent causes this. Forces the model past bare assertion.
    mechanism TEXT DEFAULT '',
    -- capacity | input_material | energy | labor | regulatory | logistics | capital
    bottleneck_type TEXT DEFAULT '',
    -- What observation would kill this link. Gives the re-score job something
    -- concrete to check a thesis against later.
    falsifier TEXT DEFAULT '',
    -- now | 1-2q | 2-4q | 2y+ . A chokepoint that binds in two years is a
    -- different trade from one binding this quarter.
    lead_time TEXT DEFAULT '',
    confidence REAL DEFAULT 0.5,
    is_leaf INTEGER DEFAULT 0,
    searched INTEGER DEFAULT 0,
    -- Sources backing THIS hop, as a JSON list of
    -- {kind, title, url, source, published_at, article_id, similarity}.
    -- kind is 'internal' (our own corpus) or 'web'. Every node carries them,
    -- including hop 1: internal grounding costs no API call, so the reason
    -- searches stop at leaves does not apply here.
    evidence_json TEXT DEFAULT '[]',
    UNIQUE (thesis_id, node_key)
);
CREATE INDEX IF NOT EXISTS idx_thesis_nodes_thesis ON thesis_nodes(thesis_id, order_depth);
CREATE INDEX IF NOT EXISTS idx_thesis_nodes_parent ON thesis_nodes(parent_id);

-- Companies sitting at a bottleneck.
--
-- listing_status is deliberately not a filter: the universe of *tradeable*
-- candidates is US-listed, but a chain that omits the actual chokepoint owner
-- is a wrong chain even when you cannot buy it. Foreign and private operators
-- are retained as context with us_proxy pointing at the nearest tradeable name.
CREATE TABLE IF NOT EXISTS thesis_candidates (
    id TEXT PRIMARY KEY,
    thesis_id TEXT NOT NULL REFERENCES theses(id),
    node_id TEXT REFERENCES thesis_nodes(id),
    company_name TEXT NOT NULL,
    ticker TEXT,
    -- What the model guessed, kept even when resolution overrides it. LLM
    -- ticker guesses are wrong often enough that the audit trail is worth a column.
    ticker_guess TEXT,
    market TEXT,
    -- resolved | unresolved | unlisted | ambiguous
    resolution_status TEXT DEFAULT 'pending',
    alt_tickers_json TEXT DEFAULT '[]',
    -- us_listed | adr | foreign_unlisted | private | unresolved
    listing_status TEXT DEFAULT 'unresolved',
    parent_company TEXT,
    us_proxy TEXT,
    role_in_chain TEXT DEFAULT '',
    -- Estimated share of revenue exposed to this bottleneck. A pure-play moves
    -- on the thesis; a conglomerate dilutes it below the noise floor. This is
    -- the field that stops the engine from just naming megacaps.
    exposure REAL DEFAULT 0.0,
    exposure_rationale TEXT DEFAULT '',
    exposure_basis TEXT DEFAULT '',
    substitutability TEXT DEFAULT 'medium',
    evidence_json TEXT DEFAULT '[]',
    conviction REAL DEFAULT 0.0,
    crowding REAL,
    -- EARLY | BUILDING | CROWDED | POST_NEWS
    rumour_stage TEXT,
    edge_score REAL,
    -- Fraction of crowding signals backed by real rows. A freshly discovered
    -- third-hop supplier has no mention history and no price history, and
    -- "nobody is talking about it" is legitimately the EARLY signal — so the
    -- absence cannot be an error, but it also must not fake a high score.
    data_coverage REAL DEFAULT 0.0,
    promoted INTEGER DEFAULT 0,
    first_seen_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_scored_at DATETIME
);
CREATE INDEX IF NOT EXISTS idx_thesis_candidates_thesis ON thesis_candidates(thesis_id, edge_score DESC);
CREATE INDEX IF NOT EXISTS idx_thesis_candidates_ticker ON thesis_candidates(ticker);
CREATE INDEX IF NOT EXISTS idx_thesis_candidates_node ON thesis_candidates(node_id);

-- Daily re-score history. Generation is expensive and runs once a day;
-- re-scoring is free and runs over every live candidate, so this is where the
-- EARLY -> CROWDED transition becomes visible. It is also what makes the
-- engine accountable: "we flagged X as EARLY on date D — what happened?"
CREATE TABLE IF NOT EXISTS thesis_candidate_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id TEXT NOT NULL REFERENCES thesis_candidates(id),
    ticker TEXT,
    as_of_date DATE NOT NULL,
    price REAL,
    crowding REAL,
    rumour_stage TEXT,
    edge_score REAL,
    data_coverage REAL,
    -- Every component value and weight that produced the score above, so a
    -- past call can be re-derived rather than re-guessed.
    components_json TEXT DEFAULT '{}',
    ret_1m REAL,
    ret_3m REAL,
    dist_from_52w_high REAL,
    volume_ratio REAL,
    mentions_30d INTEGER DEFAULT 0,
    mention_accel REAL,
    UNIQUE(candidate_id, as_of_date)
);
CREATE INDEX IF NOT EXISTS idx_thesis_snapshots_cand ON thesis_candidate_snapshots(candidate_id, as_of_date);

-- Company name -> ticker cache, so repeat theses do not re-hit yfinance.
--
-- resolved_at is checked against a real TTL, unlike ticker_info.updated_at
-- which is written and never read — a transient yfinance failure there caches
-- "Unknown" permanently.
CREATE TABLE IF NOT EXISTS company_resolution (
    company_name_key TEXT PRIMARY KEY,
    company_name TEXT,
    ticker TEXT,
    listing_status TEXT,
    exchange TEXT,
    us_proxy TEXT,
    resolved_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
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

            # Migration: mark which producer put a row in hot_tickers.
            # sector_analyzer re-upserts every 15 minutes and rewrites
            # mention_count and rationale wholesale, so a thesis-promoted
            # ticker would lose its provenance within the quarter-hour.
            # promote_thesis_ticker() writes only these two columns.
            for col_sql in [
                "ALTER TABLE hot_tickers ADD COLUMN thesis_id TEXT",
                "ALTER TABLE hot_tickers ADD COLUMN source TEXT DEFAULT 'sector_analyzer'",
                # Per-hop sources. Nodes written before this column existed
                # keep '[]' and simply render without citations.
                "ALTER TABLE thesis_nodes ADD COLUMN evidence_json TEXT DEFAULT '[]'",
            ]:
                try:
                    conn.execute(col_sql)
                except sqlite3.OperationalError:
                    pass  # Column likely already exists

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
            conn.enable_load_extension(True)
            if settings.sqlite_vec_path:
                conn.load_extension(settings.sqlite_vec_path)
            else:
                import sqlite_vec
                sqlite_vec.load(conn)
            self.has_sqlite_vec = True
            if not getattr(self, '_sqlite_vec_log_emitted', False):
                log.info("sqlite_vec.loaded_successfully")
                self._sqlite_vec_log_emitted = True
        except Exception as e:
            if not getattr(self, '_sqlite_vec_log_emitted', False):
                log.warning("sqlite_vec.load_failed_falling_back_to_numpy", error=str(e))
                self._sqlite_vec_log_emitted = True
        finally:
            # Must run on the failure path too. A SQLITE_VEC_PATH that does not
            # resolve raises above, and without this the connection is handed
            # back with extension loading still enabled.
            try:
                conn.enable_load_extension(False)
            except Exception:
                pass

        conn.execute("PRAGMA journal_mode=WAL")
        # The API and the worker share this file. Without an explicit
        # busy_timeout a reader fails outright the moment the worker is
        # mid-write; with it, SQLite waits and retries instead.
        conn.execute("PRAGMA busy_timeout=5000")
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

    def get_briefing_candidates(
        self,
        hours: int = 24,
        min_importance: float = 7.0,
        limit: int = 40,
    ) -> list[dict]:
        """
        Get the candidate pool for the prioritized daily brief.

        Deliberately separate from get_briefing_by_sector, which still backs the
        web dashboard's sector board and its 40-item windows. This one applies
        an importance floor and returns a flat list; lane assignment happens in
        data.taxonomy, because matching sector stems and headline keywords in
        SQL would mean either a wall of LIKEs or registered custom functions.

        Selects event_type as well — the lane predicates need it and the sector
        query does not return it.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT headline, summary, classification_summary, importance_score, url,
                       affected_sectors, affected_tickers, source_name, published_at,
                       sentiment_score, suggested_direction, event_type
                FROM articles
                WHERE importance_score >= ? AND published_at >= ?
                  AND event_type != 'noise' AND duplicate_of IS NULL
                ORDER BY importance_score DESC
                LIMIT ?
                """,
                (min_importance, cutoff, limit)
            ).fetchall()

        return [dict(row) for row in rows]

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
        response_text: Optional[str] = None,
        cost_usd: Optional[float] = None
    ) -> None:
        """
        Logs LLM token usage, latencies, errors, cost, and text.

        `cost_usd` is what the provider reported for this call. None means it
        reported nothing — the row is stored at 0.0 and surfaces in
        `get_usage_stats()['unpriced_calls']`, so a lane that stops reporting
        cost shows up as a number instead of silently deflating the total.
        """
        # Global kill switch for payload archiving. config.usage.track_llm's
        # store_text is the per-call-site opt-in; this takes precedence over it.
        # The classifier and ranker pass both fields on every call, which at
        # pipeline volume turns this table into a full prompt archive. Error
        # payloads are kept: they are rare, and they are the only case where
        # having the exact prompt back is worth the space.
        if not settings.log_llm_payloads and not is_error:
            prompt_text = None
            response_text = None

        if cost_usd is None and not is_error:
            log.warning(
                "db.usage.cost_missing",
                model_name=model_name,
                operation=operation,
                impact="recorded as $0.00; see unpriced_calls on /api/usage",
            )
        cost_usd = float(cost_usd or 0.0)
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
                           COUNT(*) as n,
                           -- Successful calls the provider reported no cost for.
                           -- They are stored at 0.00, so without this the total
                           -- would understate spend with nothing to show for it.
                           SUM(CASE WHEN is_error = 0 AND COALESCE(cost_usd, 0) = 0
                                    AND total_tokens > 0 THEN 1 ELSE 0 END) as unpriced
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
                    "unpriced_calls": row["unpriced"] or 0,
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
                "total_requests": 0, "unpriced_calls": 0,
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

    def get_recent_predictions(self, ticker: str = None, limit: int = 10,
                               active_only: bool = False) -> list[dict]:
        """
        Recent predictions with outcomes.

        active_only keeps a prediction only while its horizon has not yet
        elapsed — a 5-day call made three weeks ago is spent, a 1-year call made
        three weeks ago is still live. The dashboard needs this: without it the
        market grid renders months-old directions as though they were current.
        History views deliberately leave it off, since old rows are the point.
        """
        with self.connection() as conn:
            query = "SELECT * FROM predictions"
            clauses = []
            params = ()
            if ticker:
                clauses.append("ticker = ?")
                params = (ticker,)
            if active_only:
                clauses.append(
                    "datetime(created_at, '+' || horizon_days || ' days') "
                    ">= datetime('now')"
                )
            if clauses:
                query += " WHERE " + " AND ".join(clauses)
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

    def get_price_history(self, ticker: str, limit: Optional[int] = None) -> list[dict]:
        """Stored daily OHLCV for one ticker, oldest first.

        Rows missing a close are dropped here rather than downstream: they are
        Yahoo padding for untraded sessions, and a NaN close silently poisons
        every rolling window that spans it.

        `limit` takes the most RECENT n bars while still returning them oldest
        first, which is what indicator warm-up needs — a plain LIMIT would take
        the oldest n and rate the ticker as of years ago.
        """
        sql = """
            SELECT date, open, high, low, close, volume
            FROM price_history
            WHERE ticker = ? AND close IS NOT NULL
            ORDER BY date DESC
        """
        params: list = [ticker.upper()]
        if limit:
            sql += " LIMIT ?"
            params.append(int(limit))
        with self.connection() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [dict(r) for r in reversed(rows)]

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

    def get_insider_window_rollup(self, days: int = 30) -> list[dict]:
        """Per-ticker open-market insider aggregates across the whole window.

        Aggregated in SQL rather than folded out of a fetched page: the
        transaction list the UI renders is capped, and deriving totals from that
        capped slice made every headline figure a function of the page size.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT ticker,
                       COALESCE(SUM(CASE WHEN transaction_code = 'P'
                                         THEN ABS(value_usd) END), 0) AS buy_value,
                       COALESCE(SUM(CASE WHEN transaction_code = 'S'
                                         THEN ABS(value_usd) END), 0) AS sell_value,
                       SUM(CASE WHEN transaction_code = 'P' THEN 1 ELSE 0 END) AS buy_count,
                       SUM(CASE WHEN transaction_code = 'S' THEN 1 ELSE 0 END) AS sell_count,
                       -- NULL and empty names drop out of COUNT(DISTINCT ...),
                       -- matching the truthiness check this replaced.
                       COUNT(DISTINCT CASE WHEN transaction_code = 'P'
                                           AND insider_name <> ''
                                           THEN insider_name END) AS distinct_buyers
                FROM insider_transactions
                WHERE filed_at >= ? AND is_discretionary = 1
                GROUP BY ticker
                """,
                (cutoff,),
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

    # ── Off-exchange (dark pool) volume & market regime ──────────────────

    def upsert_offexchange_volume(self, rows: list[dict]) -> int:
        """Insert or replace daily FINRA off-exchange volume rows.

        REPLACE rather than IGNORE because FINRA restates: a file can be
        reposted with corrected figures, and the scheduled job deliberately
        re-reads a few sessions of overlap to pick those up.
        """
        if not rows:
            return 0
        with self.connection() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO offexchange_volume (
                    ticker, session_date, short_volume, short_exempt_volume,
                    total_volume, market_codes, published_at, source
                ) VALUES (?,?,?,?,?,?,?,?)
                """,
                [
                    (
                        r["ticker"], r["session_date"], r.get("short_volume"),
                        r.get("short_exempt_volume"), r.get("total_volume"),
                        r.get("market_codes"), r["published_at"],
                        r.get("source", "finra_cnms"),
                    )
                    for r in rows
                ],
            )
        return len(rows)

    def get_offexchange_series(self, ticker: str) -> list[dict]:
        """Full off-exchange history for one ticker, ordered by disclosure date.

        Joined to price_history so each row carries the day's consolidated
        volume alongside the off-exchange figure. The ratio between the two is
        the actual dark-pool-share signal, and computing it here keeps the
        feature code a plain _cached_series consumer instead of making it reach
        across into the predictor's separate price cache.

        consolidated_volume is NULL when price_history has no row for that
        session; callers must skip rather than divide.
        """
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT o.session_date, o.published_at, o.short_volume,
                       o.short_exempt_volume, o.total_volume, o.market_codes,
                       p.volume AS consolidated_volume
                FROM offexchange_volume o
                LEFT JOIN price_history p
                       ON p.ticker = o.ticker AND p.date = o.session_date
                WHERE o.ticker = ?
                ORDER BY o.published_at ASC
                """,
                (ticker.upper(),),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_last_offexchange_date(self) -> Optional[str]:
        """Newest session stored, across every ticker.

        Deliberately takes no ticker argument, unlike get_last_insider_filed_at:
        one CNMS file carries every symbol, so a sync either has a session or it
        does not. Resuming per-ticker would re-download the same file once per
        ticker to discover it already had the rows.
        """
        with self.connection() as conn:
            row = conn.execute(
                "SELECT MAX(session_date) AS m FROM offexchange_volume"
            ).fetchone()
        return row["m"] if row and row["m"] else None

    def get_recent_offexchange(self, ticker: str, days: int = 20) -> list[dict]:
        """Recent off-exchange sessions for one ticker, newest first (UI/report use)."""
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT o.session_date, o.short_volume, o.short_exempt_volume,
                       o.total_volume, p.volume AS consolidated_volume
                FROM offexchange_volume o
                LEFT JOIN price_history p
                       ON p.ticker = o.ticker AND p.date = o.session_date
                WHERE o.ticker = ?
                ORDER BY o.session_date DESC
                LIMIT ?
                """,
                (ticker.upper(), days),
            ).fetchall()
        return [dict(r) for r in rows]

    def upsert_market_regime(self, rows: list[dict]) -> int:
        """Insert or replace market-wide regime rows, keyed (metric, session)."""
        if not rows:
            return 0
        with self.connection() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO market_regime_daily (
                    metric, session_date, value, published_at, source
                ) VALUES (?,?,?,?,?)
                """,
                [
                    (
                        r["metric"], r["session_date"], r["value"],
                        r["published_at"], r.get("source", ""),
                    )
                    for r in rows
                ],
            )
        return len(rows)

    def get_market_regime_series(self) -> list[dict]:
        """Every market-regime row, all metrics interleaved, by disclosure date.

        Returned unpivoted and unfiltered because the consumer slices it by
        as-of date before grouping: the predictor caches this once under a
        single sentinel key and reuses it for every ticker, since the series
        describes the market rather than any one symbol.
        """
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT metric, session_date, value, published_at, source
                FROM market_regime_daily
                ORDER BY published_at ASC
                """
            ).fetchall()
        return [dict(r) for r in rows]

    def get_last_market_regime_date(self, metric: str) -> Optional[str]:
        """Newest session stored for one metric, so a sync can resume from there."""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT MAX(session_date) AS m FROM market_regime_daily WHERE metric = ?",
                (metric,),
            ).fetchone()
        return row["m"] if row and row["m"] else None

    def get_recent_market_regime(self, metric: str, days: int = 60) -> list[dict]:
        """Recent sessions for one regime metric, newest first (UI/report use)."""
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT session_date, value
                FROM market_regime_daily
                WHERE metric = ?
                ORDER BY session_date DESC
                LIMIT ?
                """,
                (metric, days),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Option-chain snapshots ───────────────────────────────────────────

    def upsert_option_chain_daily(self, rows: list[dict]) -> int:
        """Insert or replace daily option-chain aggregate rows.

        REPLACE so a same-day re-run overwrites rather than being rejected: a
        later snapshot on the same session is strictly better than an earlier
        one, since volume and open interest are still settling.
        """
        if not rows:
            return 0
        with self.connection() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO option_chain_daily (
                    ticker, session_date, spot_price, call_volume, put_volume,
                    call_oi, put_oi, put_call_volume_ratio, put_call_oi_ratio,
                    atm_iv, iv_skew, near_term_iv, far_term_iv,
                    expirations_seen, contracts_seen, published_at, source
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                [
                    (
                        r["ticker"], r["session_date"], r.get("spot_price"),
                        r.get("call_volume"), r.get("put_volume"),
                        r.get("call_oi"), r.get("put_oi"),
                        r.get("put_call_volume_ratio"), r.get("put_call_oi_ratio"),
                        r.get("atm_iv"), r.get("iv_skew"),
                        r.get("near_term_iv"), r.get("far_term_iv"),
                        r.get("expirations_seen"), r.get("contracts_seen"),
                        r["published_at"], r.get("source", "yfinance"),
                    )
                    for r in rows
                ],
            )
        return len(rows)

    def get_option_chain_series(self, ticker: str) -> list[dict]:
        """Full option-chain snapshot history for one ticker, by disclosure date."""
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT session_date, published_at, spot_price,
                       call_volume, put_volume, call_oi, put_oi,
                       put_call_volume_ratio, put_call_oi_ratio,
                       atm_iv, iv_skew, near_term_iv, far_term_iv
                FROM option_chain_daily
                WHERE ticker = ?
                ORDER BY published_at ASC
                """,
                (ticker.upper(),),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_last_option_chain_date(self, ticker: str) -> Optional[str]:
        """Newest snapshot session stored for one ticker."""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT MAX(session_date) AS m FROM option_chain_daily WHERE ticker = ?",
                (ticker.upper(),),
            ).fetchone()
        return row["m"] if row and row["m"] else None

    # ── Analyst consensus & price targets ────────────────────────────────

    def upsert_analyst_consensus(self, rows: list[dict]) -> int:
        """Insert or replace daily analyst consensus rows.

        REPLACE so a same-day re-run overwrites rather than being rejected, and
        so the retry job can repair a partial morning capture.
        """
        if not rows:
            return 0
        with self.connection() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO analyst_consensus_daily (
                    ticker, session_date, strong_buy, buy, hold, sell, strong_sell,
                    analyst_count, recommendation_key, recommendation_mean,
                    target_mean, target_high, target_low, target_median,
                    spot_price, published_at, source
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                [
                    (
                        r["ticker"], r["session_date"],
                        r.get("strong_buy"), r.get("buy"), r.get("hold"),
                        r.get("sell"), r.get("strong_sell"),
                        r.get("analyst_count"), r.get("recommendation_key"),
                        r.get("recommendation_mean"),
                        r.get("target_mean"), r.get("target_high"),
                        r.get("target_low"), r.get("target_median"),
                        r.get("spot_price"), r["published_at"],
                        r.get("source", "yfinance"),
                    )
                    for r in rows
                ],
            )
        return len(rows)

    def get_analyst_consensus_series(self, ticker: str) -> list[dict]:
        """Full stored history for one ticker, oldest first.

        Ordered by published_at rather than session_date because that is the
        as-of key every consumer slices on.
        """
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT session_date, strong_buy, buy, hold, sell, strong_sell,
                       analyst_count, recommendation_key, recommendation_mean,
                       target_mean, target_high, target_low, target_median,
                       spot_price, published_at
                FROM analyst_consensus_daily
                WHERE ticker = ?
                ORDER BY published_at ASC
                """,
                (ticker.upper(),),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_latest_analyst_consensus(self, ticker: str) -> Optional[dict]:
        """Newest stored consensus row for one ticker, or None."""
        with self.connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM analyst_consensus_daily
                WHERE ticker = ?
                ORDER BY published_at DESC
                LIMIT 1
                """,
                (ticker.upper(),),
            ).fetchone()
        return dict(row) if row else None

    def get_last_analyst_consensus_date(self, ticker: str) -> Optional[str]:
        """Newest consensus session stored for one ticker."""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT MAX(session_date) AS m FROM analyst_consensus_daily "
                "WHERE ticker = ?",
                (ticker.upper(),),
            ).fetchone()
        return row["m"] if row and row["m"] else None

    # ── Technical ratings ────────────────────────────────────────────────

    def upsert_technical_ratings(self, rows: list[dict]) -> int:
        """Insert or replace technical rating rows, keyed by timeframe too."""
        if not rows:
            return 0
        with self.connection() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO technical_rating_daily (
                    ticker, session_date, timeframe,
                    summary_score, summary_label, ma_score, ma_label,
                    osc_score, osc_label, buy_votes, neutral_votes, sell_votes,
                    bars_available, published_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                [
                    (
                        r["ticker"], r["session_date"], r["timeframe"],
                        r.get("summary_score"), r.get("summary_label"),
                        r.get("ma_score"), r.get("ma_label"),
                        r.get("osc_score"), r.get("osc_label"),
                        r.get("buy_votes"), r.get("neutral_votes"),
                        r.get("sell_votes"), r.get("bars_available"),
                        r["published_at"],
                    )
                    for r in rows
                ],
            )
        return len(rows)

    def get_technical_rating_series(self, ticker: str,
                                    timeframe: Optional[str] = None) -> list[dict]:
        """Stored ratings for one ticker, oldest first, optionally one timeframe."""
        sql = """
            SELECT session_date, timeframe, summary_score, summary_label,
                   ma_score, ma_label, osc_score, osc_label,
                   buy_votes, neutral_votes, sell_votes, bars_available,
                   published_at
            FROM technical_rating_daily
            WHERE ticker = ?
        """
        params: list = [ticker.upper()]
        if timeframe:
            sql += " AND timeframe = ?"
            params.append(timeframe)
        sql += " ORDER BY published_at ASC"
        with self.connection() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [dict(r) for r in rows]

    def get_latest_technical_ratings(self, ticker: str) -> list[dict]:
        """Newest rating per timeframe for one ticker.

        One row per timeframe, which is what the panel and the debate report
        both want. The window function keeps this a single scan instead of one
        query per timeframe.
        """
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT timeframe, session_date, summary_score, summary_label,
                       ma_score, ma_label, osc_score, osc_label,
                       buy_votes, neutral_votes, sell_votes, bars_available,
                       published_at
                FROM (
                    SELECT *, ROW_NUMBER() OVER (
                        PARTITION BY timeframe ORDER BY published_at DESC
                    ) AS rn
                    FROM technical_rating_daily
                    WHERE ticker = ?
                )
                WHERE rn = 1
                """,
                (ticker.upper(),),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_last_technical_rating_date(self, ticker: str) -> Optional[str]:
        """Newest rating session stored for one ticker."""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT MAX(session_date) AS m FROM technical_rating_daily "
                "WHERE ticker = ?",
                (ticker.upper(),),
            ).fetchone()
        return row["m"] if row and row["m"] else None

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

    # ── SSE Outbox ───────────────────────────────────────────────────────
    # Bridges the worker process to the API process. See sse_events in
    # SCHEMA_SQL and the tailer in api/sse_manager.py.

    def insert_sse_event(self, topic: str, payload: str) -> None:
        """Publish one event for the API process to pick up and stream."""
        with self.connection() as conn:
            conn.execute(
                "INSERT INTO sse_events (topic, payload) VALUES (?, ?)",
                (topic, payload),
            )

    def get_sse_events_since(self, last_id: int, limit: int = 200) -> list[dict]:
        """Events newer than last_id, oldest first, for the API's tailer."""
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT id, topic, payload FROM sse_events "
                "WHERE id > ? ORDER BY id ASC LIMIT ?",
                (last_id, limit),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_max_sse_event_id(self) -> int:
        """Highest event id, so a starting tailer skips the existing backlog."""
        with self.connection() as conn:
            row = conn.execute("SELECT MAX(id) AS max_id FROM sse_events").fetchone()
            return (row["max_id"] if row else 0) or 0

    def trim_sse_events(self, keep_minutes: int = 10) -> int:
        """Drop already-delivered events. The outbox is a relay, not a log."""
        with self.connection() as conn:
            cur = conn.execute(
                "DELETE FROM sse_events WHERE created_at < datetime('now', ?)",
                (f"-{keep_minutes} minutes",),
            )
            return cur.rowcount

    # ── Latest Prices ────────────────────────────────────────────────────
    # Written by pipeline/price_feed.py in the worker, read by /api/markets.

    def upsert_latest_prices(self, quotes: list[dict]) -> int:
        """Store freshly fetched quotes. Each dict needs ticker/price/previous_close."""
        if not quotes:
            return 0
        with self.connection() as conn:
            conn.executemany(
                """
                INSERT INTO latest_prices (
                    ticker, price, previous_close, daily_change_pct, updated_at
                ) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(ticker) DO UPDATE SET
                    price = excluded.price,
                    previous_close = excluded.previous_close,
                    daily_change_pct = excluded.daily_change_pct,
                    updated_at = CURRENT_TIMESTAMP
                """,
                [
                    (
                        q["ticker"],
                        q["price"],
                        q.get("previous_close"),
                        q.get("daily_change_pct"),
                    )
                    for q in quotes
                ],
            )
        return len(quotes)

    def get_latest_prices(self, tickers: Optional[list[str]] = None) -> dict[str, dict]:
        """Last known quote per ticker, keyed by ticker."""
        with self.connection() as conn:
            if tickers:
                placeholders = ",".join("?" for _ in tickers)
                rows = conn.execute(
                    f"SELECT * FROM latest_prices WHERE ticker IN ({placeholders})",
                    tuple(tickers),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM latest_prices").fetchall()
            return {row["ticker"]: dict(row) for row in rows}

    def get_closes_from_history(self, tickers: list[str]) -> dict[str, dict]:
        """
        Fall back to stored OHLCV when latest_prices has no row yet.

        Keeps the dashboard showing real numbers on a cold start, before the
        worker's first price refresh lands, instead of a grid of zeros.
        """
        if not tickers:
            return {}
        out: dict[str, dict] = {}
        with self.connection() as conn:
            for ticker in tickers:
                rows = conn.execute(
                    "SELECT close FROM price_history WHERE ticker = ? "
                    "ORDER BY date DESC LIMIT 2",
                    (ticker,),
                ).fetchall()
                closes = [r["close"] for r in rows if r["close"] is not None]
                if not closes:
                    continue
                current = closes[0]
                previous = closes[1] if len(closes) > 1 else current
                out[ticker] = {
                    "ticker": ticker,
                    "price": current,
                    "previous_close": previous,
                    "daily_change_pct": (
                        ((current - previous) / previous * 100) if previous else 0.0
                    ),
                    "updated_at": None,
                }
        return out

    # ── Thesis Engine ────────────────────────────────────────────────────

    def get_embeddings_since(
        self,
        cutoff_iso: str,
        exclude_social: bool = True,
        limit: int = 3000,
    ) -> list[dict]:
        """Embedded, non-noise, canonical articles published since *cutoff_iso*.

        get_all_embeddings() has no date filter and loads every row in the
        table; the theme seeder only ever wants a window, and the table only
        grows. Social sources are excluded by default because Reddit posts
        cluster by register — WSB slang looks alike regardless of subject —
        which produces large clusters that are not topics.
        """
        sql = """
            SELECT id, headline, published_at, importance_score, source_type,
                   source_name, affected_tickers, affected_sectors, embedding
            FROM articles
            WHERE embedding IS NOT NULL
              AND published_at >= ?
              AND duplicate_of IS NULL
              AND (event_type IS NULL OR event_type != 'noise')
        """
        if exclude_social:
            sql += " AND source_type != 'social'"
        sql += " ORDER BY published_at DESC LIMIT ?"
        with self.connection() as conn:
            rows = conn.execute(sql, (cutoff_iso, limit)).fetchall()
        return [dict(r) for r in rows]

    def count_articles_in_window(
        self,
        start_iso: str,
        end_iso: str,
        exclude_social: bool = True,
    ) -> int:
        """Total canonical article count in a window.

        The denominator for share-of-voice. Ingest here is bursty — weekly
        volume over the last two months runs 855, 339, 1, 522, 32, 0, 644 —
        so a bare count ratio between two windows measures whether the worker
        was up, not whether the world was talking.
        """
        sql = """
            SELECT COUNT(*) AS n FROM articles
            WHERE published_at >= ? AND published_at < ?
              AND duplicate_of IS NULL
              AND (event_type IS NULL OR event_type != 'noise')
        """
        if exclude_social:
            sql += " AND source_type != 'social'"
        with self.connection() as conn:
            return int(conn.execute(sql, (start_iso, end_iso)).fetchone()["n"])

    def get_ticker_mention_counts(
        self,
        tickers: list[str],
        recent_days: int = 7,
        base_days: int = 30,
        as_of: Optional[str] = None,
    ) -> dict[str, dict]:
        """Mention counts per ticker in a recent and a baseline window.

        Returns {ticker: {recent, base, total_recent, total_base}}. The totals
        are corpus-wide so the caller can share-of-voice normalise rather than
        compare raw counts across windows of differing ingest health.
        """
        if not tickers:
            return {}
        end = as_of or datetime.now(timezone.utc).isoformat()
        end_dt = datetime.fromisoformat(end)
        recent_start = (end_dt - timedelta(days=recent_days)).isoformat()
        base_start = (end_dt - timedelta(days=base_days)).isoformat()
        placeholders = ",".join("?" for _ in tickers)
        out = {t: {"recent": 0, "base": 0} for t in tickers}
        with self.connection() as conn:
            for key, start in (("recent", recent_start), ("base", base_start)):
                rows = conn.execute(
                    f"""
                    SELECT ticker, COUNT(*) AS n FROM ticker_mentions
                    WHERE ticker IN ({placeholders})
                      AND mentioned_at >= ? AND mentioned_at < ?
                    GROUP BY ticker
                    """,
                    [*tickers, start, end],
                ).fetchall()
                for r in rows:
                    out[r["ticker"]][key] = int(r["n"])
            totals = {}
            for key, start in (("total_recent", recent_start), ("total_base", base_start)):
                totals[key] = int(conn.execute(
                    "SELECT COUNT(*) AS n FROM ticker_mentions "
                    "WHERE mentioned_at >= ? AND mentioned_at < ?",
                    (start, end),
                ).fetchone()["n"])
        for t in out:
            out[t].update(totals)
        return out

    # ── Thesis persistence ───────────────────────────────────────────────

    def deactivate_theses(self, fingerprint: str) -> int:
        """Archive prior theses seeded from the same cluster.

        Deliberately one call made once, before inserting a whole thesis.
        TrendForecaster._store_forecast deactivates inside its per-row loop, so
        storing scenario 2 archives scenario 1 and only the last row of a batch
        survives; do not copy that shape.
        """
        if not fingerprint:
            return 0
        with self.connection() as conn:
            cur = conn.execute(
                "UPDATE theses SET status = 'archived' "
                "WHERE seed_fingerprint = ? AND status = 'active'",
                (fingerprint,),
            )
            return cur.rowcount

    def insert_thesis(self, thesis: dict) -> str:
        """Insert a thesis row, returning its id."""
        thesis_id = thesis.get("id") or str(uuid.uuid4())
        expires_at = thesis.get("expires_at")
        if expires_at is None:
            days = getattr(settings, "thesis_expiry_days", 45)
            expires_at = (datetime.now(timezone.utc) + timedelta(days=days)).isoformat()
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO theses
                    (id, title, summary, seed_kind, seed_query, seed_fingerprint,
                     consensus_tickers_json, model_name, acceleration,
                     acceleration_basis, article_count_recent, article_count_base,
                     evidence_json, status, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    thesis_id,
                    thesis.get("title", ""),
                    thesis.get("summary", ""),
                    thesis.get("seed_kind", "auto"),
                    thesis.get("seed_query", ""),
                    thesis.get("seed_fingerprint"),
                    json.dumps(thesis.get("consensus_tickers", [])),
                    thesis.get("model_name"),
                    thesis.get("acceleration"),
                    thesis.get("acceleration_basis", "sov"),
                    int(thesis.get("article_count_recent", 0)),
                    int(thesis.get("article_count_base", 0)),
                    json.dumps(thesis.get("evidence", [])),
                    thesis.get("status", "active"),
                    expires_at,
                ),
            )
        return thesis_id

    def insert_thesis_nodes(self, thesis_id: str, rows: list[dict]) -> dict[str, str]:
        """Insert chain nodes; return {node_key: node_id}.

        The model emits a flat list keyed by node_key with parent_key
        references, so parent_id is resolved from that map rather than from
        insertion order — a child may be emitted before its parent.
        """
        if not rows:
            return {}
        key_to_id = {r["node_key"]: str(uuid.uuid4()) for r in rows}
        with self.connection() as conn:
            for r in rows:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO thesis_nodes
                        (id, thesis_id, parent_id, node_key, order_depth, claim,
                         mechanism, bottleneck_type, falsifier, lead_time,
                         confidence, is_leaf, searched, evidence_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        key_to_id[r["node_key"]],
                        thesis_id,
                        key_to_id.get(r.get("parent_key")),
                        r["node_key"],
                        int(r.get("order_depth", 1)),
                        r.get("claim", ""),
                        r.get("mechanism", ""),
                        r.get("bottleneck_type", ""),
                        r.get("falsifier", ""),
                        r.get("lead_time", ""),
                        float(r.get("confidence", 0.5)),
                        int(bool(r.get("is_leaf", False))),
                        int(bool(r.get("searched", False))),
                        json.dumps(r.get("sources", [])),
                    ),
                )
        return key_to_id

    def upsert_thesis_candidates(self, rows: list[dict]) -> int:
        """Insert or replace candidate companies. Returns rows written."""
        if not rows:
            return 0
        n = 0
        with self.connection() as conn:
            for r in rows:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO thesis_candidates
                        (id, thesis_id, node_id, company_name, ticker, ticker_guess,
                         market, resolution_status, alt_tickers_json, listing_status,
                         parent_company, us_proxy, role_in_chain, exposure,
                         exposure_rationale, exposure_basis, substitutability,
                         evidence_json, conviction, crowding, rumour_stage, edge_score,
                         data_coverage, promoted, last_scored_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        r.get("id") or str(uuid.uuid4()),
                        r["thesis_id"],
                        r.get("node_id"),
                        r.get("company_name", ""),
                        r.get("ticker"),
                        r.get("ticker_guess"),
                        r.get("market"),
                        r.get("resolution_status", "pending"),
                        json.dumps(r.get("alt_tickers", [])),
                        r.get("listing_status", "unresolved"),
                        r.get("parent_company"),
                        r.get("us_proxy"),
                        r.get("role_in_chain", ""),
                        float(r.get("exposure") or 0.0),
                        r.get("exposure_rationale", ""),
                        r.get("exposure_basis", ""),
                        r.get("substitutability", "medium"),
                        json.dumps(r.get("evidence_urls", [])),
                        float(r.get("conviction") or 0.0),
                        r.get("crowding"),
                        r.get("rumour_stage"),
                        r.get("edge_score"),
                        float(r.get("data_coverage") or 0.0),
                        int(bool(r.get("promoted", False))),
                        r.get("last_scored_at"),
                    ),
                )
                n += 1
        return n

    def get_active_theses(self, limit: int = 20) -> list[dict]:
        """Active, unexpired theses, most accelerated first."""
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM theses
                WHERE status = 'active'
                  AND (expires_at IS NULL OR expires_at >= datetime('now'))
                ORDER BY COALESCE(acceleration, 0) DESC, generated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def find_articles_for_claim(
        self,
        embedding: "np.ndarray",
        limit: int = 3,
        min_similarity: float = 0.45,
        since_iso: Optional[str] = None,
    ) -> list[dict]:
        """Articles from our own corpus that support one bottleneck claim.

        Nearest-neighbour over the embeddings ingest already paid for, so this
        costs no API call — which is why every chain node can be grounded, not
        just the leaves the web-search budget reaches.

        `min_similarity` is not optional cosmetics. Top-k over a corpus this
        small returns k rows whatever the query, so without a floor a claim
        with no coverage at all still gets three confident-looking citations.
        """
        if embedding is None:
            return []

        cols = ("id, headline, url, source_name, published_at, importance_score")
        base_filter = (
            "embedding IS NOT NULL AND duplicate_of IS NULL "
            "AND (event_type IS NULL OR event_type != 'noise')"
        )

        if getattr(self, "has_sqlite_vec", False):
            sql = (
                f"SELECT {cols}, vec_distance_cosine(embedding, ?) AS distance "
                f"FROM articles WHERE {base_filter}"
            )
            params: list = [embedding.astype(np.float32).tobytes()]
            if since_iso:
                sql += " AND published_at >= ?"
                params.append(since_iso)
            # Over-fetch, then apply the floor in Python: LIMIT has to be
            # applied by the index scan before similarity is known.
            sql += " ORDER BY distance LIMIT ?"
            params.append(max(limit * 5, 20))
            with self.connection() as conn:
                rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
            for r in rows:
                r["similarity"] = 1.0 - (r.pop("distance") or 1.0)
        else:
            sql = f"SELECT {cols}, embedding FROM articles WHERE {base_filter}"
            params = []
            if since_iso:
                sql += " AND published_at >= ?"
                params.append(since_iso)
            sql += " ORDER BY published_at DESC LIMIT 3000"
            with self.connection() as conn:
                raw = [dict(r) for r in conn.execute(sql, params).fetchall()]
            query = embedding.astype(np.float32)
            query_norm = float(np.linalg.norm(query)) or 1.0
            rows = []
            for r in raw:
                vec = np.frombuffer(r.pop("embedding"), dtype=np.float32)
                if vec.shape != query.shape:
                    continue
                denom = (float(np.linalg.norm(vec)) or 1.0) * query_norm
                r["similarity"] = float(np.dot(vec, query) / denom)
                rows.append(r)

        rows = [r for r in rows if r["similarity"] >= min_similarity]
        rows.sort(key=lambda r: -r["similarity"])
        return rows[:limit]

    def count_theses_since(self, cutoff_utc: str) -> int:
        """How many theses were generated at or after `cutoff_utc`.

        `cutoff_utc` must be SQLite's own `CURRENT_TIMESTAMP` format —
        "YYYY-MM-DD HH:MM:SS", space-separated and UTC — because that is what
        the column holds. An ISO string with a "T" separator compares greater
        than every real row of the same day and silently reports zero.
        """
        with self.connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM theses WHERE generated_at >= ?",
                (cutoff_utc,),
            ).fetchone()
        return int(row["c"] if row else 0)

    def get_thesis_detail(self, thesis_id: str) -> Optional[dict]:
        """A thesis with its chain nodes and candidates attached."""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM theses WHERE id = ?", (thesis_id,)
            ).fetchone()
            if not row:
                return None
            thesis = dict(row)
            thesis["nodes"] = [
                dict(r) for r in conn.execute(
                    "SELECT * FROM thesis_nodes WHERE thesis_id = ? "
                    "ORDER BY order_depth, node_key",
                    (thesis_id,),
                ).fetchall()
            ]
            thesis["candidates"] = [
                dict(r) for r in conn.execute(
                    "SELECT * FROM thesis_candidates WHERE thesis_id = ? "
                    "ORDER BY edge_score IS NULL, edge_score DESC, conviction DESC",
                    (thesis_id,),
                ).fetchall()
            ]
        return thesis

    def get_scoreable_candidates(self, limit: int = 60) -> list[dict]:
        """Candidates on active theses that have a ticker worth scoring."""
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT c.*, t.title AS thesis_title
                FROM thesis_candidates c
                JOIN theses t ON t.id = c.thesis_id
                WHERE t.status = 'active'
                  AND (t.expires_at IS NULL OR t.expires_at >= datetime('now'))
                  AND c.ticker IS NOT NULL AND TRIM(c.ticker) != ''
                ORDER BY c.conviction DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def update_candidate_score(
        self,
        candidate_id: str,
        stage: Optional[str],
        crowding: Optional[float],
        edge_score: Optional[float],
        coverage: float,
    ) -> None:
        """Write the latest score back onto the candidate row.

        Denormalised on purpose so the list endpoint stays one query; the
        per-day history lives in thesis_candidate_snapshots.
        """
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE thesis_candidates
                SET rumour_stage = ?, crowding = ?, edge_score = ?,
                    data_coverage = ?, last_scored_at = ?
                WHERE id = ?
                """,
                (stage, crowding, edge_score, coverage,
                 datetime.now(timezone.utc).isoformat(), candidate_id),
            )

    def upsert_thesis_snapshots(self, rows: list[dict]) -> int:
        """Write one score snapshot per candidate per day. Idempotent."""
        if not rows:
            return 0
        n = 0
        with self.connection() as conn:
            for r in rows:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO thesis_candidate_snapshots
                        (candidate_id, ticker, as_of_date, price, crowding,
                         rumour_stage, edge_score, data_coverage, components_json,
                         ret_1m, ret_3m, dist_from_52w_high, volume_ratio,
                         mentions_30d, mention_accel)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        r["candidate_id"], r.get("ticker"), r["as_of_date"],
                        r.get("price"), r.get("crowding"), r.get("rumour_stage"),
                        r.get("edge_score"), r.get("data_coverage"),
                        json.dumps(r.get("components", {})),
                        r.get("ret_1m"), r.get("ret_3m"), r.get("dist_from_52w_high"),
                        r.get("volume_ratio"), int(r.get("mentions_30d") or 0),
                        r.get("mention_accel"),
                    ),
                )
                n += 1
        return n

    def get_latest_snapshot(
        self, candidate_id: str, before_date: Optional[str] = None
    ) -> Optional[dict]:
        """Most recent snapshot for a candidate, optionally strictly before a date."""
        with self.connection() as conn:
            if before_date:
                row = conn.execute(
                    "SELECT * FROM thesis_candidate_snapshots "
                    "WHERE candidate_id = ? AND as_of_date < ? "
                    "ORDER BY as_of_date DESC LIMIT 1",
                    (candidate_id, before_date),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM thesis_candidate_snapshots "
                    "WHERE candidate_id = ? ORDER BY as_of_date DESC LIMIT 1",
                    (candidate_id,),
                ).fetchone()
        return dict(row) if row else None

    def get_stage_transitions(self, days: int = 3) -> list[dict]:
        """Candidates whose rumour stage changed between their last two snapshots.

        The EARLY -> BUILDING -> CROWDED walk is the trade timing: the first
        step confirms the entry, the last one is the exit.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT s.candidate_id, s.ticker, s.as_of_date,
                       s.rumour_stage AS new_stage, s.crowding, s.edge_score,
                       prev.rumour_stage AS old_stage, prev.as_of_date AS prev_date,
                       c.company_name, c.thesis_id, t.title AS thesis_title
                FROM thesis_candidate_snapshots s
                JOIN thesis_candidates c ON c.id = s.candidate_id
                JOIN theses t ON t.id = c.thesis_id
                JOIN thesis_candidate_snapshots prev
                  ON prev.candidate_id = s.candidate_id
                 AND prev.as_of_date = (
                        SELECT MAX(as_of_date) FROM thesis_candidate_snapshots p2
                        WHERE p2.candidate_id = s.candidate_id
                          AND p2.as_of_date < s.as_of_date)
                WHERE s.as_of_date >= ?
                  AND s.rumour_stage IS NOT NULL
                  AND prev.rumour_stage IS NOT NULL
                  AND s.rumour_stage != prev.rumour_stage
                ORDER BY s.as_of_date DESC
                """,
                (cutoff,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_active_thesis_context(self, ticker: str, limit: int = 3) -> str:
        """Render any active thesis chains this ticker appears in, for the debate.

        Returns '' when the ticker is in no chain, so callers can splice the
        result in unconditionally.
        """
        if not ticker:
            return ""
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT t.title, c.role_in_chain, c.exposure, c.substitutability,
                       c.rumour_stage, n.claim, n.mechanism, n.falsifier,
                       n.order_depth
                FROM thesis_candidates c
                JOIN theses t ON t.id = c.thesis_id
                LEFT JOIN thesis_nodes n ON n.id = c.node_id
                WHERE c.ticker = ? AND t.status = 'active'
                  AND (t.expires_at IS NULL OR t.expires_at >= datetime('now'))
                ORDER BY c.edge_score IS NULL, c.edge_score DESC
                LIMIT ?
                """,
                (ticker.upper().strip(), limit),
            ).fetchall()
        if not rows:
            return ""
        parts = []
        for r in rows:
            parts.append(
                f"- Thesis: {r['title']}\n"
                f"  Chain position (hop {r['order_depth'] or '?'}): "
                f"{r['claim'] or 'n/a'}\n"
                f"  Mechanism: {r['mechanism'] or 'n/a'}\n"
                f"  Role of this company: {r['role_in_chain'] or 'n/a'} "
                f"(est. revenue exposure {r['exposure'] or 0:.0f}%, "
                f"substitutability {r['substitutability'] or 'unknown'})\n"
                f"  Crowding stage: {r['rumour_stage'] or 'UNKNOWN'}\n"
                f"  This link fails if: {r['falsifier'] or 'n/a'}"
            )
        return "\n".join(parts)

    def get_thesis_calls_due_review(
        self, min_age_days: int = 45, benchmark: str = "SPY", limit: int = 40
    ) -> list[dict]:
        """Matured EARLY calls, with everything needed to judge them.

        Pairs each candidate's first EARLY snapshot against its most recent one
        and carries the benchmark's close on both dates, so the caller can ask
        whether the call beat the market rather than merely rose.

        Candidates already written to reflection_log are excluded by the tag the
        writer stamps, which is what keeps the job idempotent across restarts --
        it re-runs daily and must not re-teach the same lesson every morning.
        """
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT c.id AS candidate_id, c.ticker, c.company_name,
                       c.thesis_id, c.role_in_chain, t.title AS thesis_title,
                       n.claim, n.falsifier,
                       f.as_of_date AS flagged_date, f.price AS flagged_price,
                       f.edge_score AS flagged_edge,
                       l.as_of_date AS latest_date, l.price AS latest_price,
                       l.rumour_stage AS latest_stage,
                       (SELECT close FROM price_history
                         WHERE ticker = ? AND date <= f.as_of_date
                           AND close IS NOT NULL
                         ORDER BY date DESC LIMIT 1) AS bench_start,
                       (SELECT close FROM price_history
                         WHERE ticker = ? AND date <= l.as_of_date
                           AND close IS NOT NULL
                         ORDER BY date DESC LIMIT 1) AS bench_end
                FROM thesis_candidates c
                JOIN theses t ON t.id = c.thesis_id
                LEFT JOIN thesis_nodes n ON n.id = c.node_id
                JOIN thesis_candidate_snapshots f
                  ON f.candidate_id = c.id
                 AND f.as_of_date = (
                        SELECT MIN(as_of_date) FROM thesis_candidate_snapshots e
                         WHERE e.candidate_id = c.id AND e.rumour_stage = 'EARLY')
                JOIN thesis_candidate_snapshots l
                  ON l.candidate_id = c.id
                 AND l.as_of_date = (
                        SELECT MAX(as_of_date) FROM thesis_candidate_snapshots m
                         WHERE m.candidate_id = c.id)
                WHERE c.ticker IS NOT NULL AND TRIM(c.ticker) != ''
                  AND f.price > 0 AND l.price > 0
                  AND julianday(l.as_of_date) - julianday(f.as_of_date) >= ?
                  AND NOT EXISTS (
                        SELECT 1 FROM reflection_log r
                         WHERE r.tags LIKE '%"thesis_review": "' || c.id || '"%')
                ORDER BY f.as_of_date ASC
                LIMIT ?
                """,
                (benchmark, benchmark, min_age_days, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Company name -> ticker resolution cache ──────────────────────────

    def get_cached_resolution(
        self, name_key: str, ttl_days: int = 30, negative_ttl_days: int = 7
    ) -> Optional[dict]:
        """Cached name->ticker lookup, or None when absent or stale.

        Negative results expire faster than positive ones: an unresolved name
        is usually a transient Yahoo failure. ticker_info shows what happens
        without a TTL at all — a cached 'Unknown' there is permanent, because
        updated_at is written and never read.
        """
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM company_resolution WHERE company_name_key = ?",
                (name_key,),
            ).fetchone()
        if not row:
            return None
        rec = dict(row)
        ttl = ttl_days if rec.get("ticker") else negative_ttl_days
        try:
            resolved = datetime.fromisoformat(rec["resolved_at"])
        except (TypeError, ValueError):
            return None
        if resolved.tzinfo is None:
            resolved = resolved.replace(tzinfo=timezone.utc)
        age = datetime.now(timezone.utc) - resolved
        return rec if age <= timedelta(days=ttl) else None

    def upsert_resolutions(self, rows: list[dict]) -> int:
        """Cache name->ticker resolutions."""
        if not rows:
            return 0
        now = datetime.now(timezone.utc).isoformat()
        with self.connection() as conn:
            for r in rows:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO company_resolution
                        (company_name_key, company_name, ticker, listing_status,
                         exchange, us_proxy, resolved_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (r["company_name_key"], r.get("company_name"), r.get("ticker"),
                     r.get("listing_status"), r.get("exchange"), r.get("us_proxy"),
                     now),
                )
        return len(rows)

    def promote_thesis_ticker(
        self, ticker: str, thesis_id: str, rationale: str = ""
    ) -> None:
        """Mark a thesis-discovered ticker as hot so ingest starts tracking it.

        Deliberately NOT upsert_hot_ticker(): that method rewrites
        mention_count, avg_sentiment and rationale wholesale, and
        SectorAnalyzer re-runs it every 15 minutes — a thesis rationale written
        through it would be clobbered within the quarter-hour. This writes only
        the columns the thesis engine owns and leaves the analyzer's fields
        untouched on conflict.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO hot_tickers
                    (ticker, mention_count, avg_sentiment, sectors_json, rationale,
                     first_detected_at, last_detected_at, thesis_id, source)
                VALUES (?, 0, 0.0, '[]', ?, ?, ?, ?, 'thesis')
                ON CONFLICT(ticker) DO UPDATE SET
                    thesis_id = excluded.thesis_id,
                    source = 'thesis',
                    last_detected_at = excluded.last_detected_at
                """,
                (ticker.upper().strip(), rationale, now, now, thesis_id),
            )
