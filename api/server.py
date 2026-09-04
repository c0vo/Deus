"""
Deus — REST and SSE APIs

Implements all required REST and SSE endpoints for Milestone 1.
"""

from __future__ import annotations

import json
import asyncio
import time
from typing import Optional
from datetime import datetime, date, timezone, timedelta

from fastapi import APIRouter, Request, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import yfinance as yf
import pandas as pd
import numpy as np

from config.cache import TTLCache
from config.logging_config import get_logger
from config.settings import settings
from data.database import Database
from pipeline.predictor import StockPredictor, HORIZON_LABELS
from pipeline.chat_orchestrator import ChatOrchestrator, build_chat_prompt
from pipeline.web_search import enrich_chat_context
from pipeline.embedder import Embedder
from pipeline.sector_analyzer import SectorAnalyzer
from pipeline.ipo_detector import IPODetector
from pipeline.geo_tagger import country_name
from pipeline.event_tracker import EventTracker
from pipeline.trend_forecaster import TrendForecaster
from pipeline.trending import get_trending_with_summaries
from pipeline.darkpool import DarkPoolTracker
from pipeline.market_regime import (
    METRIC_DIX, METRIC_GEX, METRIC_PUT_CALL, MarketRegimeTracker,
)
from api.sse_manager import event_bus
from config.llm import is_llm_configured, response_cost, stream_complete
from config.usage import track_llm

router = APIRouter()

log = get_logger(__name__)

def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _sse_event(event: str, data="") -> str:
    """Format payloads as valid SSE, including multi-line model output."""
    if not isinstance(data, str):
        data = json.dumps(data, default=str)
    lines = data.splitlines() or [""]
    return f"event: {event}\n" + "".join(f"data: {line}\n" for line in lines) + "\n"


# Response cache for the market grid. The ticker tape polls this from every
# open tab, so without single-flight caching each tab paid the full build cost
# independently.
_markets_cache = TTLCache(ttl_seconds=15)

# /api/status is seven aggregate counts plus a stat(); the header polls it from
# every open tab every 30s. /api/brain/stream rebuilds its snapshot on every
# connect, which means on every page load and every SSE reconnect.
_status_cache = TTLCache(ttl_seconds=15)
_brain_snapshot_cache = TTLCache(ttl_seconds=15)

# Off-exchange data lands once a day and the dashboard panel polls every 10
# minutes, so this holds far longer than the 15s grid caches. The window is a
# cache key rather than a filter on one cached series: the three presets are
# few enough to each keep their own entry.
_darkpool_cache = TTLCache(ttl_seconds=60)

# Analyst consensus is captured once a day and technical ratings once a day, so
# this can hold far longer than the panels above. Five minutes rather than an
# hour only because the watchlist panel is expanded on demand and a stale card
# after a manual backfill would look broken.
_analyst_cache = TTLCache(ttl_seconds=300)

# Theses are generated once a day and re-scored once a day, so this can hold far
# longer than the price-driven grids.
_thesis_cache = TTLCache(ttl_seconds=120)

# Generation runs a reasoning call plus several web searches, and this endpoint
# lives in the read-mostly API process. One at a time, so two open tabs cannot
# start two runs.
_thesis_stream_lock = asyncio.Semaphore(1)

# Sized off measured throughput, not guessed. deepseek-v4-pro at xhigh streams
# roughly 30 tokens/sec here, so decompose alone can spend ~400s of the
# thesis_max_output_tokens budget before the searches, extraction, ticker
# resolution and scoring that follow it. The previous 600s was set when that
# budget was 4000 and no longer leaves room; a timeout here reads to the user
# as another silent stall.
THESIS_STREAM_TIMEOUT_SECONDS = 1200


def _prediction_to_badge(pred: dict | None) -> dict | None:
    if not pred:
        return None
    return {
        "direction": pred.get("predicted_direction", "UP"),
        "confidence": _safe_float(pred.get("confidence")),
        "horizon_days": pred.get("horizon_days"),
        "created_at": pred.get("created_at"),
    }


# ── Request Models ────────────────────────────────────────────────────

class WatchlistRequest(BaseModel):
    ticker: str

class ChatRequest(BaseModel):
    message: str

class ReflectionRequest(BaseModel):
    lesson_learned: str
    was_successful: bool = True
    prediction_id: Optional[int] = None
    scope: str = "ticker"           # 'ticker', 'sector', or 'market'
    sector: Optional[str] = None    # required when scope='sector'
    tags: Optional[str] = None      # comma-separated free-form tags


# ── Smart Money (insider / institutional / KR flows) ─────────────────

@router.get("/api/brain/smart-money")
async def get_smart_money(
    request: Request,
    days: int = Query(30, ge=1, le=365),
    limit: int = Query(40, ge=1, le=200),
):
    """Recent disclosed positioning across all tracked tickers.

    Insider rows are open-market buys/sells only — grants, option exercises and
    tax withholding are compensation mechanics and would drown the signal.

    `limit` caps only the transaction list the UI renders. Totals and per-ticker
    rolls aggregate the entire window in SQL, so the headline figures stay true
    to the window instead of describing whichever page happened to be fetched.
    """
    db = getattr(request.app.state, "db", None) or Database()
    transactions = db.get_recent_insider_activity(days=days, limit=limit)
    stakes = db.get_recent_stakes(days=max(days, 90), limit=25)

    # Per-ticker net, so the UI can rank who is being bought and who is being sold.
    tickers = db.get_insider_window_rollup(days=days)
    for row in tickers:
        row["net_value"] = row["buy_value"] - row["sell_value"]
    tickers.sort(key=lambda r: r["net_value"], reverse=True)

    buy_value = sum(r["buy_value"] for r in tickers)
    sell_value = sum(r["sell_value"] for r in tickers)
    denom = buy_value + sell_value

    return {"data": {
        "window_days": days,
        "totals": {
            "buy_value": buy_value,
            "sell_value": sell_value,
            "net_value": buy_value - sell_value,
            "buy_ratio": (buy_value / denom) if denom else None,
            "transaction_count": sum(r["buy_count"] + r["sell_count"] for r in tickers),
        },
        "by_ticker": tickers,
        "transactions": transactions,
        "stakes": stakes,
    }}


@router.get("/api/insider/{ticker}")
async def get_insider_for_ticker(
    request: Request,
    ticker: str,
    days: int = Query(90, ge=1, le=730),
):
    """Insider and >5%-stake summary for one ticker."""
    db = getattr(request.app.state, "db", None) or Database()
    from pipeline.insider_tracker import InsiderTracker
    summary = InsiderTracker(db).get_summary(ticker.upper().strip(), days=days)
    return {"data": summary}


@router.get("/api/flows/kr/{ticker}")
async def get_kr_flows(
    request: Request,
    ticker: str,
    days: int = Query(60, ge=1, le=730),
):
    """Daily Korean institutional and foreign net trading for one ticker."""
    db = getattr(request.app.state, "db", None) or Database()
    from data.tickers import to_krx_code
    from pipeline.kr_flows import KrFlowTracker

    code = to_krx_code(ticker)
    if not code:
        raise HTTPException(status_code=400, detail=f"{ticker} is not a Korean listing")

    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    series = [r for r in db.get_kr_flow_series(code) if r["trade_date"] >= cutoff]
    return {"data": {
        "ticker": ticker.upper(),
        "krx_code": code,
        "summary": KrFlowTracker(db).get_summary(ticker, days=days),
        "series": series,
    }}


def _build_darkpool_payload(db: Database, symbol: str, days: int) -> dict:
    """
    Assemble the Dark Pool card's payload. Synchronous by design.

    Six SQLite round-trips, so this runs in a worker thread rather than on the
    event loop — one executor hop for the whole build, not one per query.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")

    series = []
    for row in db.get_offexchange_series(symbol):
        if str(row["session_date"]) < cutoff:
            continue
        total = row["total_volume"] or 0.0
        consolidated = row["consolidated_volume"] or 0.0
        series.append({
            "session_date": row["session_date"],
            "off_exchange_volume": total,
            "consolidated_volume": consolidated or None,
            "off_exchange_share": (total / consolidated) if consolidated > 0 else None,
            "short_ratio": (row["short_volume"] / total) if total > 0 else None,
        })

    regime = {
        metric: db.get_recent_market_regime(metric, days=days)
        for metric in (METRIC_DIX, METRIC_GEX, METRIC_PUT_CALL)
    }

    return {"data": {
        "ticker": symbol,
        "summary": DarkPoolTracker(db).get_summary(symbol, days=20),
        "series": series,
        "regime": regime,
        "regime_summary": MarketRegimeTracker(db).get_summary(days=60),
    }}


@router.get("/api/darkpool/{ticker}")
async def get_darkpool_for_ticker(
    request: Request,
    ticker: str,
    days: int = Query(60, ge=1, le=730),
):
    """Off-exchange (dark pool) volume for one ticker, plus market-wide regime.

    Each session carries both ratios the dashboard plots: the share of the
    day's tape that printed off-exchange, and the share of that off-exchange
    volume that was short. Sessions with no matching price_history row report a
    null share rather than being dropped, so a gap in prices does not silently
    shorten the series.

    Read-only and cached, like every other panel endpoint — the dashboard polls
    this on a timer from every open tab, and the underlying tables only change
    once a day.
    """
    db = getattr(request.app.state, "db", None) or Database()
    symbol = ticker.upper().strip()

    return await _darkpool_cache.get_or_build(
        (symbol, days),
        lambda: asyncio.to_thread(_build_darkpool_payload, db, symbol, days),
    )


def _build_analyst_payload(db: Database, symbol: str) -> dict:
    """
    Assemble the analyst panel's payload. Synchronous by design.

    Three SQLite round-trips, so this runs in a worker thread — one executor hop
    for the whole build, not one per query.

    Consensus and technical ratings are returned together because they are one
    card in the UI, and because they fail independently: a ticker can have no
    analyst coverage but a perfectly good technical rating, or deep price history
    with no sell-side following. Each side reports its own absence rather than
    the endpoint 404ing on either.
    """
    from pipeline.analyst_ratings import AnalystRatingsTracker
    from pipeline.technical_rating import TechnicalRatingTracker

    consensus = AnalystRatingsTracker(db).get_summary(symbol)
    ratings = TechnicalRatingTracker(db).get_summary(symbol)

    # Prefer the live tape for the implied-upside figure. The consensus row
    # carries the spot from the morning snapshot, which is hours stale by the
    # time anyone opens the panel.
    live = db.get_latest_prices([symbol]).get(symbol) or {}
    spot = live.get("price") or (consensus.get("latest") or {}).get("spot_price")

    target_mean = (consensus.get("latest") or {}).get("target_mean")
    return {"data": {
        "ticker": symbol,
        "spot_price": spot,
        "consensus": consensus,
        "upside_pct": ((target_mean - spot) / spot * 100.0)
                      if (spot and target_mean and spot > 0) else None,
        "technical": ratings["timeframes"],
        "min_bars": ratings["min_bars"],
    }}


@router.get("/api/analysts/{ticker}")
async def get_analysts_for_ticker(request: Request, ticker: str):
    """Analyst consensus, price targets and technical ratings for one ticker.

    The consensus buckets and targets are fetched sell-side opinion; the
    technical ratings are computed locally from stored OHLCV under TradingView's
    published methodology. Both are daily, so this is cached hard.

    A ticker with neither returns 200 with `consensus.covered` false and an empty
    `technical` map — absence of coverage is a real answer, not an error.
    """
    db = getattr(request.app.state, "db", None) or Database()
    symbol = ticker.upper().strip()

    return await _analyst_cache.get_or_build(
        symbol,
        lambda: asyncio.to_thread(_build_analyst_payload, db, symbol),
    )

# ── Thesis Engine ────────────────────────────────────────────────────
#
# Route order matters: the literal paths below are declared before
# /api/thesis/{thesis_id}, or the parameterised route swallows them. Same trap
# /api/predict/history/recent and /api/reflections/sectors already dodge.


def _build_thesis_list_payload(db: Database, limit: int) -> dict:
    """Active theses, each with its top candidates.

    Synchronous on purpose — several blocking SQLite reads, wrapped in one
    executor hop by the caller rather than one hop per query.
    """
    theses = db.get_active_theses(limit=limit)
    out = []
    for t in theses:
        detail = db.get_thesis_detail(t["id"]) or {}
        candidates = detail.get("candidates", [])
        out.append({
            **t,
            "consensus_tickers": _safe_json_list(t.get("consensus_tickers_json")),
            "evidence": _safe_json_list(t.get("evidence_json")),
            "node_count": len(detail.get("nodes", [])),
            "candidate_count": len(candidates),
            "top_candidates": [_thesis_candidate_dto(c) for c in candidates[:5]],
        })
    return {"data": out}


def _build_thesis_detail_payload(db: Database, thesis_id: str) -> Optional[dict]:
    detail = db.get_thesis_detail(thesis_id)
    if not detail:
        return None
    detail["consensus_tickers"] = _safe_json_list(detail.get("consensus_tickers_json"))
    detail["evidence"] = _safe_json_list(detail.get("evidence_json"))
    detail["candidates"] = [_thesis_candidate_dto(c) for c in detail["candidates"]]
    detail["nodes"] = [_thesis_node_dto(n) for n in detail.get("nodes", [])]
    return {"data": detail}


def _thesis_node_dto(n: dict) -> dict:
    """Chain node with its per-hop citations parsed out of the JSON column."""
    return {**n, "sources": _safe_json_list(n.get("evidence_json"))}


def _safe_json_list(raw) -> list:
    try:
        value = json.loads(raw or "[]")
        return value if isinstance(value, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def _thesis_candidate_dto(c: dict) -> dict:
    """Candidate row plus its parsed evidence URLs."""
    return {**c, "evidence_urls": _safe_json_list(c.get("evidence_json"))}


@router.get("/api/thesis/candidates")
async def get_thesis_candidates(
    request: Request,
    stage: str = Query("", description="Filter by rumour stage, e.g. EARLY"),
    limit: int = Query(40, ge=1, le=200),
):
    """Cross-thesis candidate list, least crowded first."""
    db = getattr(request.app.state, "db", None) or Database()
    wanted = (stage or "").upper().strip()

    def build() -> dict:
        rows = db.get_scoreable_candidates(limit=limit * 3)
        dtos = [_thesis_candidate_dto(r) for r in rows]
        if wanted:
            dtos = [d for d in dtos if (d.get("rumour_stage") or "") == wanted]
        # Highest edge first; unscored names sort last rather than as zero.
        dtos.sort(key=lambda d: (d.get("edge_score") is None,
                                 -(d.get("edge_score") or 0.0)))
        return {"data": dtos[:limit]}

    return await _thesis_cache.get_or_build(
        ("candidates", wanted, limit), lambda: asyncio.to_thread(build)
    )


@router.get("/api/thesis/transitions")
async def get_thesis_transitions(
    request: Request,
    days: int = Query(3, ge=1, le=30),
):
    """Candidates whose rumour stage moved between their last two snapshots.

    The dashboard widget also receives these live over SSE, but only at the
    instant the daily re-score publishes them. Without this endpoint the strip
    is empty on every page load until the next 09:10 job happens to fire.
    """
    db = getattr(request.app.state, "db", None) or Database()
    return await _thesis_cache.get_or_build(
        ("transitions", days),
        lambda: asyncio.to_thread(
            lambda: {"data": db.get_stage_transitions(days=days)}
        ),
    )


@router.get("/api/thesis/stream")
async def thesis_stream(request: Request, seed: str = Query("", max_length=300)):
    """Build a thesis from a user-supplied topic, streaming as it reasons.

    Deliberately does NOT publish to event_bus: this page is already receiving
    the events directly, and publishing as well would double-deliver to any
    dashboard listening on the broadcast topic.
    """
    topic = (seed or "").strip()
    if not topic:
        raise HTTPException(status_code=400, detail="A seed topic is required")

    db = getattr(request.app.state, "db", None) or Database()
    queue: asyncio.Queue = asyncio.Queue()
    background_task = None

    async def event_generator():
        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=5.0)
                except asyncio.TimeoutError:
                    if await request.is_disconnected():
                        break
                    continue
                event, data = item["event"], item["data"]
                yield _sse_event(event, data)
                if event in ("done", "error"):
                    break
        except asyncio.CancelledError:
            raise
        finally:
            nonlocal background_task
            if background_task and not background_task.done():
                background_task.cancel()

    async def progress_callback(message: str):
        await queue.put({"event": "agent_update", "data": message})

    async def chunk_callback(token: str):
        await queue.put({"event": "reasoning_chunk", "data": json.dumps({"text": token})})

    async def research_callback(event_type: str, data: dict):
        await queue.put({"event": event_type, "data": json.dumps(data)})

    async def node_callback(node: dict):
        await queue.put({"event": "chain_node", "data": json.dumps(node)})

    async def candidate_callback(candidate: dict):
        await queue.put({"event": "candidate", "data": json.dumps(candidate, default=str)})

    async def run_thesis():
        if _thesis_stream_lock.locked():
            await queue.put({"event": "error",
                             "data": "Another thesis is already being generated."})
            await queue.put({"event": "done", "data": ""})
            return
        async with _thesis_stream_lock:
            try:
                from pipeline.thesis_engine import ThesisEngine

                result = await asyncio.wait_for(
                    ThesisEngine(db).generate_from_text(
                        topic,
                        progress_callback=progress_callback,
                        chunk_callback=chunk_callback,
                        research_callback=research_callback,
                        node_callback=node_callback,
                        candidate_callback=candidate_callback,
                    ),
                    timeout=THESIS_STREAM_TIMEOUT_SECONDS,
                )
                await queue.put({"event": "verdict", "data": json.dumps({
                    "thesis_id": result.get("thesis_id"),
                    "nodes": result.get("chain_nodes", []),
                    "candidates": result.get("scored", []),
                    "errors": result.get("errors", []),
                }, default=str)})
                if not result.get("thesis_id"):
                    # A run that reasoned and then died — most often the chain
                    # never parsed — reaches here with errors and no thesis.
                    # Without this the page just stops mid-stream and looks
                    # like it is still thinking.
                    reasons = result.get("errors") or [
                        "the run finished without producing a thesis"
                    ]
                    # Flattened: _sse_event splits a multi-line payload across
                    # several data: lines, and the page's parser only reads the
                    # first one. Validation errors are routinely multi-line.
                    await queue.put({
                        "event": "error",
                        "data": " ".join("; ".join(reasons).split()),
                    })
            except asyncio.CancelledError:
                pass
            except asyncio.TimeoutError:
                await queue.put({"event": "error", "data": "Thesis generation timed out."})
            except Exception as e:
                log.error("api.thesis_stream_failed", error=str(e))
                await queue.put({"event": "error", "data": str(e)})
            finally:
                await queue.put({"event": "done", "data": ""})

    background_task = asyncio.create_task(run_thesis())
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/api/thesis")
async def get_theses(request: Request, limit: int = Query(10, ge=1, le=50)):
    db = getattr(request.app.state, "db", None) or Database()
    return await _thesis_cache.get_or_build(
        ("list", limit),
        lambda: asyncio.to_thread(_build_thesis_list_payload, db, limit),
    )


@router.get("/api/thesis/{thesis_id}")
async def get_thesis(request: Request, thesis_id: str):
    db = getattr(request.app.state, "db", None) or Database()
    payload = await _thesis_cache.get_or_build(
        ("detail", thesis_id),
        lambda: asyncio.to_thread(_build_thesis_detail_payload, db, thesis_id),
    )
    if payload is None:
        raise HTTPException(status_code=404, detail="Thesis not found")
    return payload


@router.post("/api/thesis/candidates/{candidate_id}/track")
async def track_thesis_candidate(request: Request, candidate_id: str):
    """Promote a candidate onto the watchlist by hand."""
    db = getattr(request.app.state, "db", None) or Database()

    def promote() -> Optional[str]:
        with db.connection() as conn:
            row = conn.execute(
                "SELECT ticker FROM thesis_candidates WHERE id = ?", (candidate_id,)
            ).fetchone()
        if not row or not row["ticker"]:
            return None
        db.add_tracked_ticker(row["ticker"])
        return row["ticker"]

    ticker = await asyncio.to_thread(promote)
    if not ticker:
        raise HTTPException(status_code=404,
                            detail="Candidate not found or has no resolved ticker")
    return {"success": True, "ticker": ticker}
# ── Watchlist CRUD ───────────────────────────────────────────────────

@router.get("/api/watchlist")
async def get_watchlist(request: Request):
    db = getattr(request.app.state, "db", None) or Database()
    tracked = db.get_tracked_tickers()
    return {"data": tracked}

@router.post("/api/watchlist")
async def add_watchlist(request: Request, payload: WatchlistRequest):
    ticker = payload.ticker.upper().strip()
    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker cannot be empty")
    db = getattr(request.app.state, "db", None) or Database()
    success = db.add_tracked_ticker(ticker)
    return {"success": success, "ticker": ticker}

@router.delete("/api/watchlist/{ticker}")
async def remove_watchlist(request: Request, ticker: str):
    ticker = ticker.upper().strip()
    db = getattr(request.app.state, "db", None) or Database()
    success = db.remove_tracked_ticker(ticker)
    return {"success": success, "ticker": ticker}


# ── Live Market Grid ─────────────────────────────────────────────────

@router.get("/api/markets")
async def get_markets(request: Request):
    """
    The market grid behind the ticker tape.

    Reads only — no network calls, no model loading, no training. Quotes come
    from `latest_prices`, refreshed by the worker (pipeline/price_feed.py), and
    predictions come from rows the worker has already generated. Every open tab
    polls this continuously, so anything expensive here is paid forever.
    """
    db = getattr(request.app.state, "db", None) or Database()
    tickers = await asyncio.to_thread(db.get_tracked_tickers)
    if not tickers:
        tickers = ["AAPL", "MSFT", "GOOGL"]

    # Keyed on the watchlist so adding or removing a ticker invalidates the
    # entry immediately rather than showing a stale grid for the whole TTL.
    cache_key = tuple(sorted(tickers))
    return await _markets_cache.get_or_build(
        cache_key, lambda: _build_markets_payload(db, tickers)
    )


def _load_quotes(db: Database, tickers: list[str]) -> dict[str, dict]:
    """
    Last-known quotes for the grid.

    Falls back to stored OHLCV for anything the worker has not refreshed yet,
    so a cold start shows real numbers instead of a grid of zeros.
    """
    quotes = db.get_latest_prices(tickers)
    missing = [t for t in tickers if t not in quotes]
    if missing:
        quotes.update(db.get_closes_from_history(missing))
    return quotes


async def _build_markets_payload(db: Database, tickers: list[str]) -> dict:
    """Assemble the market grid from state the worker has already computed."""
    quotes = await asyncio.to_thread(_load_quotes, db, tickers)

    def load_ticker_rows(ticker: str):
        # Grouped into one executor hop: three separate to_thread calls would
        # each open their own connection for a few milliseconds of work.
        return (
            db.get_recent_predictions(ticker, limit=20, active_only=True),
            db.get_cached_advisory(ticker, days=5),
            db.get_ticker_sector(ticker),
        )

    async def process_ticker(ticker: str) -> dict:
        recent_preds, cached_advisory, sector = await asyncio.to_thread(
            load_ticker_rows, ticker
        )

        predictions = {}
        for pred in recent_preds:
            label = HORIZON_LABELS.get(pred.get("horizon_days"))
            if label and label not in predictions:
                predictions[label] = _prediction_to_badge(pred)

        for horizon_days, label in HORIZON_LABELS.items():
            if label not in predictions:
                # No live prediction for this horizon. Training and prediction
                # both belong to the worker (see train_missing_models in
                # orchestrator/scheduler.py) — a page load must never trigger a
                # five-fold model fit.
                predictions[label] = {
                    "direction": "TRAINING",
                    "confidence": 0.0,
                    "horizon_days": horizon_days,
                }

        quote = quotes.get(ticker) or {}
        current_price = _safe_float(quote.get("price"))
        daily_change_pct = _safe_float(quote.get("daily_change_pct"))

        return {
            "ticker": ticker,
            "sector": sector or "Unknown",
            "price": current_price,
            "current_price": current_price,
            "daily_change_pct": daily_change_pct,
            "price_updated_at": quote.get("updated_at"),
            "predictions": predictions,
            "cached_prediction": recent_preds[0] if recent_preds else None,
            "cached_advisory": cached_advisory,
            "cached_debate": cached_advisory,
        }

    data = await asyncio.gather(*(process_ticker(t) for t in tickers))
    return {"data": data}


@router.get("/api/news/general")
async def get_general_news(
    request: Request,
    hours: int = 168,
    limit: int = 20,
    min_importance: float = 0.0
):
    """
    Return macro, geopolitical, and general news not tied to a specific ticker.
    Only returns articles that have been classified and ranked past the threshold.
    """
    db = getattr(request.app.state, "db", None) or Database()
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    with db.connection() as conn:
        rows = conn.execute(
            """
            SELECT
                a.id,
                a.headline,
                a.summary,
                a.classification_summary,
                a.source_name,
                a.url,
                a.published_at,
                a.importance_score,
                a.sentiment_score,
                a.urgency,
                a.suggested_direction,
                a.event_type,
                a.affected_sectors
            FROM articles a
            WHERE LOWER(a.event_type) IN ('macro', 'geopolitical', 'general')
              AND a.published_at >= ?
              AND a.importance_score >= ?
              AND a.event_type != 'noise'
            ORDER BY a.importance_score DESC NULLS LAST, a.published_at DESC
            LIMIT ?
            """,
            (cutoff, min_importance, limit),
        ).fetchall()
    return {"data": [dict(row) for row in rows]}


@router.get("/api/news/{ticker}")
async def get_ticker_news(request: Request, ticker: str, hours: int = 168, limit: int = 10):
    db = getattr(request.app.state, "db", None) or Database()
    ticker = ticker.upper().strip()
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    with db.connection() as conn:
        rows = conn.execute(
            """
            SELECT
                a.id,
                a.headline,
                a.summary,
                a.classification_summary,
                a.source_name,
                a.url,
                a.published_at,
                a.importance_score,
                COALESCE(tm.sentiment_score, a.sentiment_score) AS sentiment_score,
                COALESCE(tm.urgency, a.urgency) AS urgency,
                a.suggested_direction
            FROM ticker_mentions tm
            JOIN articles a ON a.id = tm.article_id
            WHERE tm.ticker = ?
              AND a.published_at >= ?
              AND (a.event_type IS NULL OR a.event_type != 'noise')
            ORDER BY a.importance_score DESC NULLS LAST, a.published_at DESC
            LIMIT ?
            """,
            (ticker, cutoff, limit),
        ).fetchall()
    return {"data": [dict(row) for row in rows]}


# ── Recent Debates (cross-ticker) ─────────────────────────────────────

@router.get("/api/predict/history/recent")
async def get_recent_debates(
    request: Request,
    limit: int = Query(50, ge=1, le=200)
):
    """Retrieve recent debate entries across all tickers."""
    db = getattr(request.app.state, "db", None) or Database()
    return {"data": db.get_recent_debates(limit=limit)}


# ── SSE Predict & Debate Streaming ────────────────────────────────────

@router.get("/api/predict/{ticker}/history")
async def get_predict_history(request: Request, ticker: str):
    """Retrieve historical debate dates available for a ticker."""
    ticker = ticker.upper().strip()
    db = getattr(request.app.state, "db", None) or Database()
    with db.connection() as conn:
        rows = conn.execute(
            "SELECT date FROM predictions_cache WHERE ticker = ? ORDER BY date DESC",
            (ticker,)
        ).fetchall()
        return {"data": [row["date"] for row in rows]}

@router.get("/api/predict/{ticker}/history/{date}")
async def get_predict_history_by_date(request: Request, ticker: str, date: str):
    """Retrieve historical debate details for a ticker on a specific date."""
    ticker = ticker.upper().strip()
    db = getattr(request.app.state, "db", None) or Database()
    with db.connection() as conn:
        row = conn.execute(
            "SELECT advisory_json FROM predictions_cache WHERE ticker = ? AND date = ?",
            (ticker, date)
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"No debate history found for {ticker} on {date}")
        try:
            return {"data": json.loads(row["advisory_json"])}
        except Exception:
            raise HTTPException(status_code=500, detail="Error decoding cached advisory")

@router.get("/api/predict/{ticker}/stream")
async def predict_stream(request: Request, ticker: str, refresh: bool = False):
    ticker = ticker.upper().strip()
    db = getattr(request.app.state, "db", None) or Database()
    queue = asyncio.Queue()
    background_task = None  # track for cleanup on disconnect

    async def event_generator():
        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=5.0)
                except asyncio.TimeoutError:
                    # Check if client disconnected
                    if await request.is_disconnected():
                        break
                    continue
                event = item["event"]
                data = item["data"]
                yield _sse_event(event, data)
                if event in ("done", "error"):
                    break
        except asyncio.CancelledError:
            raise
        finally:
            # Cancel any running background task on generator exit
            nonlocal background_task
            if background_task and not background_task.done():
                background_task.cancel()

    # 1. Check cache first if refresh is False
    if not refresh:
        cached = db.get_cached_advisory(ticker, days=5)
        if cached:
            async def run_cached_prediction():
                try:
                    await queue.put({"event": "agent_update", "data": f"Loading cached debate advisory from {cached.get('_cache_date', 'database')}..."})
                    await asyncio.sleep(0.3)

                    debate_history = cached.get("debate_history", [])
                    bull_count = 0
                    bear_count = 0
                    for line in debate_history:
                        if line.startswith("Bull: "):
                            speaker = "Bull"
                            text = line[6:]
                            bull_count += 1
                            round_num = bull_count
                        elif line.startswith("Bear: "):
                            speaker = "Bear"
                            text = line[6:]
                            bear_count += 1
                            round_num = bear_count
                        else:
                            continue

                        await queue.put({"event": "agent_update", "data": f"Retrieving {speaker} Round {round_num} argument..."})
                        words = text.split(" ")
                        chunk_size = 5
                        for i in range(0, len(words), chunk_size):
                            chunk = " ".join(words[i:i+chunk_size]) + (" " if i + chunk_size < len(words) else "")
                            await queue.put({
                                "event": "debate_chunk",
                                "data": json.dumps({"speaker": speaker, "round": round_num, "text": chunk})
                            })
                            await asyncio.sleep(0.01)
                        await asyncio.sleep(0.1)

                    bull_report = "\n\n".join([line[6:] for line in debate_history if line.startswith("Bull: ")])
                    bear_report = "\n\n".join([line[6:] for line in debate_history if line.startswith("Bear: ")])

                    verdict_data = {
                        "ticker": ticker,
                        # The ML baseline, not the trade call — see the live
                        # path below. Advisories cached before the trader
                        # started reporting its own call have neither field.
                        "predicted_direction": cached.get("ml_prediction", {}).get("predicted_direction", "UNKNOWN"),
                        "confidence": cached.get("ml_prediction", {}).get("confidence", 0.0),
                        "advisory_direction": cached.get("trader_direction"),
                        "advisory_conviction": cached.get("trader_conviction"),
                        "final_advisory": cached.get("final_advisory"),
                        "bull_report": bull_report,
                        "bear_report": bear_report,
                        "ml_prediction": cached.get("ml_prediction"),
                        "debate_history": debate_history,
                        "executive_summary": cached.get("executive_summary")
                    }
                    await queue.put({"event": "verdict", "data": json.dumps(verdict_data)})
                except asyncio.CancelledError:
                    pass  # cancelled due to client disconnect
                except Exception as e:
                    await queue.put({"event": "error", "data": str(e)})
                finally:
                    await queue.put({"event": "done", "data": ""})

            background_task = asyncio.create_task(run_cached_prediction())
            return StreamingResponse(event_generator(), media_type="text/event-stream")

    # 2. Live prediction if refresh is True or no cache exists
    async def progress_callback(msg: str):
        await queue.put({"event": "agent_update", "data": msg})

    async def debate_chunk_callback(speaker: str, round_num: int, token: str):
        await queue.put({
            "event": "debate_chunk",
            "data": json.dumps({"speaker": speaker, "round": round_num, "text": token})
        })

    async def research_callback(event_type: str, data: dict):
        await queue.put({
            "event": event_type,
            "data": json.dumps(data)
        })

    async def run_prediction():
        try:
            predictor = StockPredictor(db)
            result = await predictor.predict_with_agents(
                ticker,
                progress_callback=progress_callback,
                debate_chunk_callback=debate_chunk_callback,
                research_callback=research_callback
            )

            debate_history = result.get("debate_history", [])
            bull_report = "\n\n".join([line[6:] for line in debate_history if line.startswith("Bull: ")])
            bear_report = "\n\n".join([line[6:] for line in debate_history if line.startswith("Bear: ")])

            verdict_data = {
                "ticker": ticker,
                # `predicted_direction`/`confidence` are the GradientBoosting
                # baseline and go stale to UNKNOWN/0.0 whenever no model exists
                # for the horizon. The trade call the debate actually reached
                # is `advisory_*`, which is what the arena headlines.
                "predicted_direction": result.get("ml_prediction", {}).get("predicted_direction", "UNKNOWN"),
                "confidence": result.get("ml_prediction", {}).get("confidence", 0.0),
                "advisory_direction": result.get("trader_direction"),
                "advisory_conviction": result.get("trader_conviction"),
                "final_advisory": result.get("final_advisory"),
                "bull_report": bull_report,
                "bear_report": bear_report,
                "ml_prediction": result.get("ml_prediction"),
                "debate_history": debate_history,
                "executive_summary": result.get("executive_summary")
            }
            await queue.put({"event": "verdict", "data": json.dumps(verdict_data)})
        except asyncio.CancelledError:
            pass  # cancelled due to client disconnect
        except Exception as e:
            import traceback
            traceback.print_exc()
            await queue.put({"event": "error", "data": str(e)})
        finally:
            await queue.put({"event": "done", "data": ""})

    background_task = asyncio.create_task(run_prediction())
    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ── SSE Chat Streaming ───────────────────────────────────────────────

@router.post("/api/chat/stream")
async def chat_stream(request: Request, payload: ChatRequest):
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    db = getattr(request.app.state, "db", None) or Database()
    queue = asyncio.Queue()

    # Simple conversation / greeting routing check
    greetings = {"hi", "hello", "hey", "greetings", "howdy", "yo", "sup", "good morning", "good afternoon", "good evening"}
    clean_msg = message.lower().strip().strip("?!.")
    words = clean_msg.split()
    is_greeting = (len(words) <= 3 and any(w in greetings for w in words)) or clean_msg in {"how are you", "who are you", "what is your name", "what can you do"}

    async def run_chat():
        try:
            if is_greeting:
                await queue.put({
                    "event": "step",
                    "data": json.dumps({"step": "classification", "intent": "greeting", "reasoning": "Greeting bypass route."})
                })
                greeting_text = "Hello! I am your Deus financial analyst. I can help you analyze stocks, review news, run predictions, or discuss market trends. How can I assist you today?"
                for token in greeting_text.split(" "):
                    await queue.put({"event": "token", "data": token + " "})
                    await asyncio.sleep(0.02)
                return

            orchestrator = ChatOrchestrator(db)
            state = {"query": message, "context": "", "routing_decision": "", "final_answer": ""}

            router_res = await orchestrator.router_node(state)
            decision = router_res.get("routing_decision", "shallow")
            state["routing_decision"] = decision

            await queue.put({
                "event": "step",
                "data": json.dumps({"step": "classification", "intent": decision, "reasoning": f"Routed to {decision} agent"})
            })

            rag_res = await orchestrator.rag_node(state)
            context = rag_res.get("context", "")
            top_articles = rag_res.get("top_articles", [])
            state["context"] = context

            await queue.put({
                "event": "step",
                "data": json.dumps({"step": "retrieval", "context": context})
            })

            web_sources = []
            # ── Web search enrichment for complex queries ──
            if decision == "complex":
                await queue.put({
                    "event": "step",
                    "data": json.dumps({
                        "step": "web_search",
                        "intent": "searching",
                        "reasoning": "Running live web search for latest news..."
                    })
                })

                rag_count = len(top_articles)

                async def _chat_research_callback(event_type: str, data: dict):
                    if event_type == "research_source":
                        data["total"] = data.get("total", 0) + rag_count
                        data["index"] = data.get("index", 0) + rag_count
                    elif event_type == "research_complete":
                        data["sources_found"] = data.get("sources_found", 0) + rag_count
                    elif event_type == "research_start":
                        data["total"] = settings.web_search_max_results + rag_count

                    await queue.put({
                        "event": event_type,
                        "data": json.dumps(data)
                    })

                enriched, web_sources = await enrich_chat_context(
                    query=message,
                    db_context=context,
                    max_results=settings.web_search_max_results,
                    research_callback=_chat_research_callback,
                )
                if enriched != context:
                    context = enriched
                    state["context"] = context
                    await queue.put({
                        "event": "step",
                        "data": json.dumps({
                            "step": "web_search",
                            "intent": "merged",
                            "reasoning": f"Merged {len(web_sources)} web search source(s) into context"
                        })
                    })

            # Emit combined sources (RAG DB top_articles + Web Search sources)
            all_articles = list(top_articles) + web_sources
            if all_articles:
                await queue.put({
                    "event": "sources",
                    "data": json.dumps({"articles": all_articles})
                })

            # Was: a hardcoded pair of slugs shadowing the settings whenever
            # they happened to be empty, so the configured chat models were
            # never actually reached. The settings are the only source now.
            model = (
                settings.model_chat_shallow if decision == "shallow"
                else settings.model_chat_complex
            )
            if not is_llm_configured() or not model:
                await queue.put({"event": "error", "data": "❌ LLM not configured."})
                return

            prompt = build_chat_prompt(message, context)

            # This is the primary user-facing chat path and it recorded nothing
            # at all until it was instrumented. Streamed responses carry usage
            # on a trailing chunk, so keep the last one seen and log it after
            # the stream drains.
            with track_llm(db, model, "chat_stream") as usage:
                collected = []
                async for chunk in stream_complete(
                    model=model,
                    prompt=prompt,
                    reasoning=None if decision == "shallow" else "medium",
                ):
                    if chunk.text:
                        collected.append(chunk.text)
                        await queue.put({"event": "token", "data": json.dumps({"text": chunk.text})})
                    if chunk.usage is not None:
                        usage.prompt_tokens = getattr(chunk.usage, "prompt_tokens", None)
                        usage.candidate_tokens = getattr(chunk.usage, "completion_tokens", None)
                        usage.cost = response_cost(chunk.usage)

                usage.response_text = "".join(collected)

        except asyncio.CancelledError:
            pass  # cancelled due to client disconnect
        except Exception as e:
            import traceback
            traceback.print_exc()
            await queue.put({"event": "error", "data": str(e)})
        finally:
            await queue.put({"event": "done", "data": ""})

    background_task = asyncio.create_task(run_chat())

    async def event_generator():
        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=5.0)
                except asyncio.TimeoutError:
                    if await request.is_disconnected():
                        break
                    continue
                event = item["event"]
                data = item["data"]
                yield _sse_event(event, data)
                if event in ("done", "error"):
                    break
        except asyncio.CancelledError:
            raise
        finally:
            if background_task and not background_task.done():
                background_task.cancel()

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ── News Geography ───────────────────────────────────────────────────

@router.get("/api/brain/news-geo")
async def get_news_geo(request: Request, hours: int = 24):
    """Per-country news volume plus recent tagged stories, for the globe."""
    db = getattr(request.app.state, "db", None) or Database()
    data = db.get_news_geo(hours=hours)
    # Attach display names so the client does not need its own country table.
    for entry in data["countries"]:
        entry["name"] = country_name(entry["country"])
    data["untagged_backlog"] = db.get_geo_backlog_count()
    return {"data": data}


# ── News Briefings ───────────────────────────────────────────────────

@router.get("/api/briefing")
async def get_briefing(request: Request, hours: int = 24, limit: int = 10):
    db = getattr(request.app.state, "db", None) or Database()
    briefings = db.get_briefing_by_sector(hours=hours, limit=limit)
    return {"data": briefings}


@router.get("/api/trending")
async def get_trending(request: Request, hours: int = 24, limit: int = 15, refresh: bool = False):
    db = getattr(request.app.state, "db", None) or Database()
    data, was_cached = await get_trending_with_summaries(
        db, hours=hours, limit=limit, refresh=refresh
    )
    if was_cached:
        return {"data": data, "cached": True}
    return {"data": data}


# ── Charts & Technical Indicators ─────────────────────────────────────

@router.get("/api/charts/{ticker}")
async def get_charts(ticker: str, days: int = 90):
    ticker = ticker.upper().strip()
    try:
        # Fetch extra days to compute indicators reliably
        fetch_days = days + 50
        t = yf.Ticker(ticker)
        period = "2y" if fetch_days > 252 else ("1y" if fetch_days > 90 else "6mo")
        df = await asyncio.to_thread(t.history, period=period)
        if df.empty:
            df = await asyncio.to_thread(t.history, period="max")
        if df.empty:
            return {"data": []}

        df = df.dropna(subset=['Close'])
        df.columns = [c.lower() for c in df.columns]

        # SMA 20
        df['sma20'] = df['close'].rolling(window=20).mean()

        # MACD (12, 26, 9)
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # Bollinger Bands (20, 2)
        std20 = df['close'].rolling(window=20).std()
        df['upper_bb'] = df['sma20'] + (std20 * 2)
        df['lower_bb'] = df['sma20'] - (std20 * 2)

        # RSI (14)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi14'] = 100 - (100 / (1 + rs))

        # Clean NaN and Inf values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.where(pd.notnull(df), None)

        # Subset to requested days
        df_subset = df.iloc[-days:]
        history = []
        for index, row in df_subset.iterrows():
            history.append({
                "time": index.strftime("%Y-%m-%d"),
                "open": _safe_float(row["open"]),
                "high": _safe_float(row["high"]),
                "low": _safe_float(row["low"]),
                "close": _safe_float(row["close"]),
                "volume": int(_safe_float(row["volume"])),
                "sma20": None if row["sma20"] is None else _safe_float(row["sma20"]),
                "macd": None if row["macd"] is None else _safe_float(row["macd"]),
                "macd_signal": None if row["macd_signal"] is None else _safe_float(row["macd_signal"]),
                "macd_hist": None if row["macd_hist"] is None else _safe_float(row["macd_hist"]),
                "upper_bb": None if row["upper_bb"] is None else _safe_float(row["upper_bb"]),
                "lower_bb": None if row["lower_bb"] is None else _safe_float(row["lower_bb"]),
                "rsi14": None if row["rsi14"] is None else _safe_float(row["rsi14"])
            })
        return {"data": history}
    except Exception as e:
        return {"data": [], "error": str(e)}



# ── Accuracy & Reflections ───────────────────────────────────────────

@router.get("/api/accuracy")
async def get_accuracy(request: Request, ticker: Optional[str] = None):
    db = getattr(request.app.state, "db", None) or Database()
    if ticker:
        ticker = ticker.upper().strip()
    acc = db.get_prediction_accuracy(ticker)
    recent = db.get_recent_predictions(ticker, limit=10)
    total = acc.get("total", 0) or 0
    correct = acc.get("correct", 0) or 0
    incorrect = acc.get("incorrect", 0) or 0
    accuracy_pct = _safe_float(acc.get("accuracy_pct"))
    return {
        "accuracy": (accuracy_pct / 100) if accuracy_pct > 1 else accuracy_pct,
        "accuracy_pct": accuracy_pct,
        "total": total,
        "correct": correct,
        "incorrect": incorrect,
        "correct_count": correct,
        "incorrect_count": incorrect,
        "raw": acc,
        "recent": recent
    }

@router.get("/api/reflections/sectors")
async def get_reflection_sectors(request: Request):
    """Get unique sector names used in reflections and ticker_info."""
    db = getattr(request.app.state, "db", None) or Database()
    sectors = set()
    with db.connection() as conn:
        rows = conn.execute(
            "SELECT DISTINCT sector FROM reflection_log WHERE sector IS NOT NULL AND sector != ''"
        ).fetchall()
        for r in rows:
            sectors.add(r["sector"])
        ti_rows = conn.execute(
            "SELECT DISTINCT sector FROM ticker_info WHERE sector IS NOT NULL AND sector != ''"
        ).fetchall()
        for r in ti_rows:
            sectors.add(r["sector"])
    return {"data": sorted(sectors)}


@router.get("/api/reflections")
async def get_all_reflections(
    request: Request,
    scope: Optional[str] = Query(None),
    sector: Optional[str] = Query(None),
    ticker: Optional[str] = Query(None),
    was_successful: Optional[bool] = Query(None),
    limit: int = Query(50, ge=1, le=200),
):
    """Get all reflections with optional filters."""
    db = getattr(request.app.state, "db", None) or Database()

    conditions = []
    params = []

    if scope:
        conditions.append("scope = ?")
        params.append(scope)
    if sector:
        conditions.append("sector = ?")
        params.append(sector)
    if ticker:
        conditions.append("ticker = ?")
        params.append(ticker.upper().strip())
    if was_successful is not None:
        conditions.append("was_successful = ?")
        params.append(1 if was_successful else 0)

    where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""

    with db.connection() as conn:
        rows = conn.execute(
            f"""
            SELECT id, ticker, prediction_id, date, lesson_learned,
                   was_successful, scope, sector, tags
            FROM reflection_log{where_clause}
            ORDER BY date DESC
            LIMIT ?
            """,
            params + [limit]
        ).fetchall()
        return {"data": [dict(r) for r in rows]}


@router.get("/api/reflections/{ticker}")
async def get_reflections(request: Request, ticker: str, limit: int = 50):
    db = getattr(request.app.state, "db", None) or Database()
    ticker = ticker.upper().strip()
    with db.connection() as conn:
        rows = conn.execute(
            """
            SELECT id, ticker, prediction_id, date, lesson_learned, was_successful
            FROM reflection_log
            WHERE ticker = ?
            ORDER BY date DESC
            LIMIT ?
            """,
            (ticker, limit)
        ).fetchall()
        return {"data": [dict(r) for r in rows]}

@router.post("/api/reflections/{ticker}")
async def create_reflection(request: Request, ticker: str, payload: ReflectionRequest):
    db = getattr(request.app.state, "db", None) or Database()
    # Only uppercase ticker for ticker-scoped reflections
    ticker_val = ticker.upper().strip() if payload.scope == "ticker" else ticker
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    db.insert_reflection(
        ticker=ticker_val,
        prediction_id=payload.prediction_id,
        date=today_str,
        lesson_learned=payload.lesson_learned,
        was_successful=payload.was_successful,
        scope=payload.scope,
        sector=payload.sector,
        tags=payload.tags,
    )
    return {"success": True, "ticker": ticker_val, "lesson_learned": payload.lesson_learned}


@router.delete("/api/reflections/{reflection_id}")
async def delete_reflection(request: Request, reflection_id: int):
    """Delete a reflection by ID."""
    db = getattr(request.app.state, "db", None) or Database()
    with db.connection() as conn:
        existing = conn.execute(
            "SELECT id FROM reflection_log WHERE id = ?", (reflection_id,)
        ).fetchone()
        if not existing:
            raise HTTPException(status_code=404, detail="Reflection not found")
        conn.execute("DELETE FROM reflection_log WHERE id = ?", (reflection_id,))
    return {"success": True}


# ── System Status & Usage ─────────────────────────────────────────────

@router.get("/api/status")
async def get_status(request: Request):
    db = getattr(request.app.state, "db", None) or Database()

    def build_status() -> dict:
        stats = db.get_stats()
        with db.connection() as conn:
            stats["total_predictions"] = conn.execute("SELECT COUNT(*) AS c FROM predictions").fetchone()["c"]
            stats["total_reflections"] = conn.execute("SELECT COUNT(*) AS c FROM reflection_log").fetchone()["c"]
        stats["db_size_bytes"] = int(_safe_float(stats.get("db_size_mb")) * 1024 * 1024)
        stats["watchlist_size"] = len(db.get_tracked_tickers())
        return stats

    return await _status_cache.get_or_build(
        "status", lambda: asyncio.to_thread(build_status)
    )

@router.get("/api/usage")
async def get_usage(request: Request):
    db = getattr(request.app.state, "db", None) or Database()
    # Explicit window: by_model is built from `details`, so the totals must be
    # computed over the same period or the table can never sum to the headline.
    # The lifetime figures ride along as all_time_* for the dashboard header.
    usage = db.get_usage_stats(days=7)
    by_model = {}
    for row in usage.get("details", []):
        model = row.get("model_name") or "unknown"
        current = by_model.setdefault(model, {"tokens": 0, "cost": 0.0})
        current["tokens"] += row.get("tokens") or 0
        current["cost"] += row.get("cost") or 0.0
    usage["by_model"] = by_model
    # `unpriced_calls` comes straight from get_usage_stats. It counts successful
    # calls the provider reported no cost for, which are stored at $0.00 — the
    # one way the totals here can understate real spend.
    return usage


@router.get("/api/brain")
async def get_brain_dashboard(request: Request, q: Optional[str] = None):
    """Provides backend telemetry for Bloomberg /brain dashboard, including semantic search."""
    db = getattr(request.app.state, "db", None) or Database()
    
    # 1. Ingested articles (recent 20)
    with db.connection() as conn:
        articles_rows = conn.execute(
            """
            SELECT id, headline, source_name, published_at, importance_score, event_type, sentiment_score 
            FROM articles 
            ORDER BY published_at DESC LIMIT 20
            """
        ).fetchall()
        articles = [dict(row) for row in articles_rows]

    # 2. Embedding status/logs
    with db.connection() as conn:
        total = conn.execute("SELECT COUNT(*) AS c FROM articles").fetchone()["c"]
        embedded = conn.execute("SELECT COUNT(*) AS c FROM articles WHERE embedding IS NOT NULL").fetchone()["c"]
        pending = total - embedded
    dedup = db.get_dedup_stats()
    embedding_status = {
        "total_articles": total,
        "embedded_articles": embedded,
        "pending_articles": pending,
        "success_rate_pct": (embedded / total * 100) if total > 0 else 100.0,
        "duplicate_articles": dedup["duplicates"],
        "unique_articles": total - dedup["duplicates"],
        "dedup_pending": dedup["unchecked"],
    }

    # 3. Sentiment analysis distribution
    with db.connection() as conn:
        total_sent = conn.execute("SELECT COUNT(*) AS c FROM articles WHERE sentiment_score IS NOT NULL").fetchone()["c"]
        bullish = conn.execute("SELECT COUNT(*) AS c FROM articles WHERE sentiment_score > 0.15").fetchone()["c"]
        bearish = conn.execute("SELECT COUNT(*) AS c FROM articles WHERE sentiment_score < -0.15").fetchone()["c"]
        neutral = total_sent - bullish - bearish
        sentiment_distribution = {
            "bullish": bullish,
            "bearish": bearish,
            "neutral": neutral,
            "total": total_sent
        }

    # 4. Semantic search results (if q is provided)
    semantic_results = []
    if q:
        embedder = Embedder()
        await embedder.initialize()
        query_vec = await embedder.get_embedding(q)
        if query_vec is not None:
            def vector_search():
                if getattr(db, 'has_sqlite_vec', False):
                    query_bytes = query_vec.astype(np.float32).tobytes()
                    with db.connection() as conn:
                        rows = conn.execute(
                            """
                            SELECT id, headline, source_name, published_at, importance_score, vec_distance_cosine(embedding, ?) as distance 
                            FROM articles 
                            WHERE embedding IS NOT NULL 
                            ORDER BY distance LIMIT 10
                            """,
                            (query_bytes,)
                        ).fetchall()
                        return [dict(row) for row in rows]
                else:
                    # Fallback to numpy similarity calculation
                    def calculate_similarity(v1, v2):
                        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
                    
                    embeddings = db.get_all_embeddings(exclude_noise=True)
                    scored = []
                    for article_id, vec in embeddings:
                        sim = calculate_similarity(query_vec, vec)
                        scored.append((sim, article_id))
                    scored.sort(key=lambda x: x[0], reverse=True)
                    top_10 = scored[:10]
                    if not top_10:
                        return []
                    ids = [x[1] for x in top_10]
                    sims = {x[1]: x[0] for x in top_10}
                    with db.connection() as conn:
                        placeholders = ','.join('?' for _ in ids)
                        rows = conn.execute(
                            f"SELECT id, headline, source_name, published_at, importance_score FROM articles WHERE id IN ({placeholders})",
                            ids
                        ).fetchall()
                        res = []
                        for row in rows:
                            d = dict(row)
                            d["distance"] = 1.0 - sims[d["id"]]
                            res.append(d)
                        return res
            try:
                loop = asyncio.get_running_loop()
                semantic_results = await loop.run_in_executor(None, vector_search)
                semantic_results.sort(key=lambda x: x.get("distance", 1.0))
            except Exception as e:
                log.error("api.semantic_search_failed", error=str(e))

    return {
        "articles": articles,
        "embedding_status": embedding_status,
        "sentiment_distribution": sentiment_distribution,
        "semantic_results": semantic_results
    }


# ── THE_BRAIN Intelligence Endpoints ─────────────────────────────────

@router.get("/api/brain/sector-heatmap")
async def get_sector_heatmap(
    request: Request,
    period: str = Query("1d", pattern=r"^(1d|7d|1m|1y)$")
):
    """Get sector sentiment matrix for the brain dashboard.

    Args:
        period: Time window - 1d (24h), 7d (168h), 1m (720h / ~30d), 1y (8760h / ~365d).
    """
    db = getattr(request.app.state, "db", None) or Database()

    period_hours = {"1d": 24, "7d": 168, "1m": 720, "1y": 8760}
    hours = period_hours[period]

    analyzer = SectorAnalyzer(db)
    data = analyzer._compute_sector_snapshot(hours=hours)
    return {"data": data}


@router.get("/api/brain/sector-shifts")
async def get_sector_shifts(request: Request):
    """Get active sector rotation signals."""
    db = getattr(request.app.state, "db", None) or Database()
    signals = db.get_active_rotation_signals()
    return {"data": signals}


@router.get("/api/brain/ipos")
async def get_ipos(request: Request):
    """Get tracked IPO watchlist."""
    db = getattr(request.app.state, "db", None) or Database()
    detector = IPODetector(db)
    ipos = detector.get_ipo_watchlist()
    return {"data": ipos}


@router.delete("/api/brain/ipos/{ipo_id}")
async def delete_ipo(request: Request, ipo_id: int):
    """Remove an IPO from the watchlist."""
    db = getattr(request.app.state, "db", None) or Database()
    with db.connection() as conn:
        existing = conn.execute("SELECT id FROM ipo_tracker WHERE id = ?", (ipo_id,)).fetchone()
        if not existing:
            raise HTTPException(status_code=404, detail="IPO not found")
        conn.execute("DELETE FROM ipo_tracker WHERE id = ?", (ipo_id,))
    return {"success": True}


@router.get("/api/brain/events")
async def get_events(
    request: Request,
    ticker: Optional[str] = Query(None),
    days_ahead: int = Query(14, ge=1, le=90),
):
    """Get upcoming earnings and major events."""
    db = getattr(request.app.state, "db", None) or Database()
    tracker = EventTracker(db)
    if ticker:
        events = tracker.get_ticker_events(ticker, days_ahead=days_ahead)
    else:
        events = tracker.get_all_upcoming_events(days_ahead=days_ahead)
    return {"data": events}


@router.get("/api/brain/events/{ticker}")
async def get_ticker_events(
    request: Request,
    ticker: str,
    days_ahead: int = Query(30, ge=1, le=90),
):
    """Get upcoming events for a specific ticker."""
    db = getattr(request.app.state, "db", None) or Database()
    tracker = EventTracker(db)
    events = tracker.get_ticker_events(ticker.upper(), days_ahead=days_ahead)
    return {"data": events}


@router.delete("/api/brain/events/{event_id}")
async def delete_event(request: Request, event_id: int):
    """Dismiss/remove an upcoming event."""
    db = getattr(request.app.state, "db", None) or Database()
    with db.connection() as conn:
        existing = conn.execute("SELECT id FROM ticker_events WHERE id = ?", (event_id,)).fetchone()
        if not existing:
            raise HTTPException(status_code=404, detail="Event not found")
        conn.execute("DELETE FROM ticker_events WHERE id = ?", (event_id,))
    return {"success": True}


# ── Calendar ─────────────────────────────────────────────────────────────────
# A merged view over ticker_events and ipo_tracker. It exists rather than
# reusing /api/brain/events + /api/brain/ipos because those are shaped for the
# dashboard cards: events has no from-date and caps at 90 days ahead, ipos
# hardcodes LIMIT 30 and a status-priority ordering. A month grid needs an
# arbitrary date window, no limit, and date ordering — and the two tables have
# colliding integer PKs, so the merged rows carry prefixed ids.

_CALENDAR_REFRESH_COOLDOWN_S = 60
_last_calendar_refresh = 0.0

IPO_CONFIDENCE = {
    "priced": "confirmed",
    "listed": "confirmed",
    "upcoming": "estimated",
}


def _ipo_to_calendar_item(ipo: dict) -> dict:
    meta = {}
    if ipo.get("metadata_json"):
        try:
            meta = json.loads(ipo["metadata_json"])
        except (json.JSONDecodeError, TypeError):
            meta = {}

    detail_parts = [p for p in (meta.get("exchange"), ipo.get("offering_price")) if p]
    return {
        "id": f"ipo:{ipo['id']}",
        "kind": "ipo",
        "date": ipo.get("ipo_date"),
        "ticker": ipo.get("ticker"),
        "title": ipo.get("company_name"),
        "event_type": "ipo",
        "confidence": IPO_CONFIDENCE.get(ipo.get("status") or "", "rumored"),
        "source": meta.get("source") or "llm_extracted",
        "sector": ipo.get("sector"),
        "detail": " · ".join(str(p) for p in detail_parts),
        "raw_id": ipo["id"],
    }


def _event_to_calendar_item(event: dict) -> dict:
    return {
        "id": f"event:{event['id']}",
        "kind": "event",
        "date": event.get("event_date"),
        "ticker": event.get("ticker"),
        "title": event.get("event_title") or f"{event.get('ticker', '')} {event.get('event_type', '')}".strip(),
        "event_type": event.get("event_type"),
        "confidence": event.get("confidence"),
        "source": event.get("source"),
        "sector": event.get("sector"),
        "detail": event.get("notes") or "",
        "raw_id": event["id"],
    }


@router.get("/api/calendar")
async def get_calendar(
    request: Request,
    from_date: Optional[str] = Query(None, alias="from", pattern=r"^\d{4}-\d{2}-\d{2}$"),
    to_date: Optional[str] = Query(None, alias="to", pattern=r"^\d{4}-\d{2}-\d{2}$"),
):
    """Events and IPOs in a date window, for the calendar page."""
    today = datetime.now(timezone.utc).date()
    if not from_date:
        from_date = today.replace(day=1).isoformat()
    if not to_date:
        next_month = (today.replace(day=28) + timedelta(days=4)).replace(day=1)
        to_date = (next_month - timedelta(days=1)).isoformat()

    if from_date > to_date:
        raise HTTPException(status_code=400, detail="`from` must not be after `to`")
    span = (date.fromisoformat(to_date) - date.fromisoformat(from_date)).days
    if span > 400:
        raise HTTPException(status_code=400, detail="Range must be 400 days or fewer")

    db = getattr(request.app.state, "db", None) or Database()

    events = EventTracker(db).get_events_in_range(from_date, to_date)
    items = [_event_to_calendar_item(e) for e in events]

    with db.connection() as conn:
        ipo_rows = conn.execute(
            """
            SELECT * FROM ipo_tracker
            WHERE ipo_date IS NOT NULL
              AND ipo_date >= ? AND ipo_date <= ?
            ORDER BY ipo_date ASC
            """,
            (from_date, to_date),
        ).fetchall()
    items.extend(_ipo_to_calendar_item(dict(r)) for r in ipo_rows)

    # Sorted server-side so the client never re-sorts.
    items.sort(key=lambda i: (i["date"] or "", i["kind"], i["ticker"] or ""))

    return {
        "data": {
            "from": from_date,
            "to": to_date,
            "items": items,
            "counts": {
                "event": sum(1 for i in items if i["kind"] == "event"),
                "ipo": sum(1 for i in items if i["kind"] == "ipo"),
            },
            "sources": {"finnhub": bool(settings.finnhub_api_key)},
        }
    }


@router.post("/api/calendar/refresh")
async def refresh_calendar(request: Request):
    """
    Pull the Finnhub earnings and IPO calendars on demand.

    Deliberately does NOT call scan_upcoming_events() — that would fire the
    DeepSeek news extraction pass and spend tokens on a button press.
    """
    global _last_calendar_refresh

    now = time.monotonic()
    if now - _last_calendar_refresh < _CALENDAR_REFRESH_COOLDOWN_S:
        raise HTTPException(
            status_code=429,
            detail=f"Try again in {int(_CALENDAR_REFRESH_COOLDOWN_S - (now - _last_calendar_refresh))}s",
        )
    _last_calendar_refresh = now

    db = getattr(request.app.state, "db", None) or Database()
    started = time.monotonic()

    earnings = await EventTracker(db)._scan_earnings_calendar()
    ipos = await IPODetector(db).scan_finnhub_ipos()

    return {
        "data": {
            "earnings_added": len(earnings),
            "ipos_added": len(ipos),
            "finnhub_configured": bool(settings.finnhub_api_key),
            "duration_ms": int((time.monotonic() - started) * 1000),
        }
    }


@router.get("/api/brain/macro-themes")
async def get_macro_themes(request: Request, refresh: bool = False):
    """Get LLM-generated macro themes from recent high-importance news."""
    # Fast path: return in-memory cache if warm (refreshed every 4h by scheduler)
    if not refresh:
        cached = TrendForecaster.get_cached_macro_themes()
        if cached is not None:
            return {"data": cached, "cached": True}

    # Cold cache or forced refresh → generate fresh and cache
    db = getattr(request.app.state, "db", None) or Database()
    forecaster = TrendForecaster(db)
    themes = await forecaster.generate_and_cache_macro_themes()
    return {"data": themes}


@router.get("/api/brain/trend-forecast/{sector}")
async def get_trend_forecast(request: Request, sector: str, refresh: bool = False):
    """Get LLM outlook for a specific sector (cached from 4h scheduler cycle)."""
    db = getattr(request.app.state, "db", None) or Database()
    forecaster = TrendForecaster(db)

    # Read from DB cache (refreshed every 4h by scheduler)
    if not refresh:
        cached = forecaster.get_active_forecasts(sector=sector)
        if cached:
            row = cached[0]
            return {"data": {
                "ticker": row.get("ticker"),
                "sector": row.get("sector"),
                "forecast_type": row.get("forecast_type"),
                "scenario_label": row.get("scenario_label"),
                "time_horizon": row.get("time_horizon"),
                "confidence": row.get("confidence"),
                "narrative": row.get("narrative"),
                "key_drivers": json.loads(row.get("key_drivers_json") or "[]"),
                "supporting_evidence": row.get("supporting_evidence"),
            }}

    # Cold cache or forced refresh → generate fresh and persist
    forecast = await forecaster.get_sector_outlook(sector)
    if forecast:
        forecaster._store_forecast(forecast)
    return {"data": forecast}


@router.get("/api/brain/pipeline-metrics")
async def get_pipeline_metrics(request: Request, limit: int = Query(20, ge=1, le=100)):
    """Get recent pipeline cycle timing/profiling data."""
    db = getattr(request.app.state, "db", None) or Database()
    metrics = db.get_recent_pipeline_metrics(limit=limit)
    return {"data": metrics}


@router.get("/api/brain/stream")
async def brain_stream(request: Request):
    """
    Multiplexed SSE endpoint for THE_BRAIN real-time dashboard.

    On connect, sends a full snapshot of all current state.
    Then streams incremental updates via the SSE event bus.
    """
    db = getattr(request.app.state, "db", None) or Database()

    topics = [
        "pipeline_status", "new_articles", "sector_heatmap",
        "rotation_signal", "ipo_alert", "trend_forecast",
        "hot_tickers", "market_ticker", "sentiment_distribution",
        "embedding_status", "events_updated", "thesis_update",
    ]
    subscriber = event_bus.subscribe(topics)

    def _build_brain_snapshot():
        """
        Build a full snapshot for late-joining SSE clients.

        Synchronous on purpose — every statement below is blocking SQLite, so
        the caller runs it in a thread rather than pinning the event loop for
        the duration of a dozen aggregates plus a sector recompute.
        """
        with db.connection() as conn:
            # Last 50 articles
            articles = conn.execute(
                "SELECT id, headline, source_name, published_at, importance_score, "
                "event_type, sentiment_score, urgency, suggested_direction, "
                "classification_summary, affected_sectors, url FROM articles "
                "ORDER BY published_at DESC LIMIT 50"
            ).fetchall()

            total = conn.execute("SELECT COUNT(*) AS c FROM articles").fetchone()["c"]
            embedded = conn.execute(
                "SELECT COUNT(*) AS c FROM articles WHERE embedding IS NOT NULL"
            ).fetchone()["c"]

            total_sent = conn.execute(
                "SELECT COUNT(*) AS c FROM articles WHERE sentiment_score IS NOT NULL"
            ).fetchone()["c"]
            bullish = conn.execute(
                "SELECT COUNT(*) AS c FROM articles WHERE sentiment_score > 0.15"
            ).fetchone()["c"]
            bearish = conn.execute(
                "SELECT COUNT(*) AS c FROM articles WHERE sentiment_score < -0.15"
            ).fetchone()["c"]

        analyzer = SectorAnalyzer(db)
        sector_data = analyzer._compute_sector_snapshot(hours=24)

        metrics = db.get_recent_pipeline_metrics(limit=10)

        return {
            "articles": [dict(r) for r in articles],
            "embedding_status": {
                "total_articles": total,
                "embedded_articles": embedded,
                "pending_articles": total - embedded,
                "success_rate_pct": (embedded / total * 100) if total > 0 else 100.0,
                "duplicate_articles": db.get_dedup_stats()["duplicates"],
            },
            "sentiment_distribution": {
                "bullish": bullish,
                "bearish": bearish,
                "neutral": total_sent - bullish - bearish,
                "total": total_sent,
            },
            "sector_heatmap": sector_data,
            "pipeline_metrics": [dict(r) for r in metrics],
        }

    async def event_generator():
        try:
            # 1. Send full snapshot first. Cached and single-flighted: this
            # runs on every page load and every SSE reconnect, so a reconnect
            # storm would otherwise multiply the cost by the number of tabs.
            snapshot = await _brain_snapshot_cache.get_or_build(
                "snapshot", lambda: asyncio.to_thread(_build_brain_snapshot)
            )
            yield _sse_event("snapshot", json.dumps(snapshot))

            # 2. Stream incremental updates
            while True:
                try:
                    event_type, data = await asyncio.wait_for(
                        subscriber.queue.get(), timeout=30.0
                    )
                    yield _sse_event(event_type, json.dumps(data))
                except asyncio.TimeoutError:
                    yield _sse_event("heartbeat", "")
        except asyncio.CancelledError:
            pass
        finally:
            event_bus.unsubscribe(subscriber.id)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
