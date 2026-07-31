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

from config.logging_config import get_logger
from config.settings import settings
from data.database import Database
from pipeline.predictor import StockPredictor
from pipeline.chat_orchestrator import ChatOrchestrator, build_chat_prompt
from pipeline.web_search import enrich_chat_context
from pipeline.embedder import GeminiEmbedder
from pipeline.sector_analyzer import SectorAnalyzer
from pipeline.ipo_detector import IPODetector
from pipeline.geo_tagger import country_name
from pipeline.event_tracker import EventTracker
from pipeline.trend_forecaster import TrendForecaster
from api.sse_manager import event_bus
from config.llm import get_client, parse_structured, DEFAULT_SAFETY_SETTINGS
from data.models import TickerNote, notes_to_dict
from config.usage import track_llm
from google.genai import types

router = APIRouter()

log = get_logger(__name__)

HORIZON_LABELS = {
    5: "5d",
    21: "1m",
    63: "3m",
    252: "1y",
}


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


# Spot training runs a 1200-day feature loop plus a RandomizedSearchCV on the
# API event loop. /api/markets is polled every 10s by the globally-mounted
# ticker tape and can request every (ticker, horizon) pair at once, so without
# a cap a cold model directory would launch dozens of concurrent searches and
# stall the API and Telegram bot. Queued tasks keep their active_trainings slot,
# so the cap throttles rather than drops them.
MAX_CONCURRENT_SPOT_TRAININGS = 2
_spot_training_semaphore = asyncio.Semaphore(MAX_CONCURRENT_SPOT_TRAININGS)


async def background_train_and_predict(ticker: str, horizon_days: int, app_state):
    db = getattr(app_state, "db", None) or Database()
    predictor = StockPredictor(db)
    try:
        async with _spot_training_semaphore:
            log.info(f"Spot training ML model for {ticker} ({horizon_days}d) in background...")
            await predictor.train_model(ticker, scope="per_ticker", horizon_days=horizon_days)
            await predictor.predict(ticker, horizon_days=horizon_days, fast_fallback=False)
            log.info(f"Spot training complete and prediction saved for {ticker} ({horizon_days}d)")
        # Clear any earlier failure: without this a pair that failed once stays
        # pinned to the fast heuristic for the whole process lifetime, and that
        # fabricated confidence gets written to `predictions` and later scored
        # as if it were a real model output.
        getattr(app_state, "failed_trainings", set()).discard((ticker, horizon_days))
    except Exception as e:
        log.error(f"Error in spot training for {ticker} ({horizon_days}d): {e}")
        failed_trainings = getattr(app_state, "failed_trainings", set())
        failed_trainings.add((ticker, horizon_days))
    finally:
        active_trainings = getattr(app_state, "active_trainings", set())
        active_trainings.discard((ticker, horizon_days))


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
    """
    db = getattr(request.app.state, "db", None) or Database()
    transactions = db.get_recent_insider_activity(days=days, limit=limit)
    stakes = db.get_recent_stakes(days=max(days, 90), limit=25)

    buy_value = sum(abs(t["value_usd"] or 0) for t in transactions
                    if t["transaction_code"] == "P")
    sell_value = sum(abs(t["value_usd"] or 0) for t in transactions
                     if t["transaction_code"] == "S")
    denom = buy_value + sell_value

    # Per-ticker net, so the UI can rank who is being bought and who is being sold.
    by_ticker: dict[str, dict] = {}
    for t in transactions:
        row = by_ticker.setdefault(t["ticker"], {
            "ticker": t["ticker"], "buy_value": 0.0, "sell_value": 0.0,
            "buy_count": 0, "sell_count": 0, "buyers": set(),
        })
        value = abs(t["value_usd"] or 0)
        if t["transaction_code"] == "P":
            row["buy_value"] += value
            row["buy_count"] += 1
            if t["insider_name"]:
                row["buyers"].add(t["insider_name"])
        else:
            row["sell_value"] += value
            row["sell_count"] += 1

    tickers = []
    for row in by_ticker.values():
        row["distinct_buyers"] = len(row.pop("buyers"))
        row["net_value"] = row["buy_value"] - row["sell_value"]
        tickers.append(row)
    tickers.sort(key=lambda r: r["net_value"], reverse=True)

    return {"data": {
        "window_days": days,
        "totals": {
            "buy_value": buy_value,
            "sell_value": sell_value,
            "net_value": buy_value - sell_value,
            "buy_ratio": (buy_value / denom) if denom else None,
            "transaction_count": len(transactions),
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
    db = getattr(request.app.state, "db", None) or Database()
    tickers = db.get_tracked_tickers()
    if not tickers:
        tickers = ["AAPL", "MSFT", "GOOGL"]

    if not hasattr(request.app.state, "active_trainings"):
        request.app.state.active_trainings = set()
    if not hasattr(request.app.state, "failed_trainings"):
        request.app.state.failed_trainings = set()
    active_trainings = request.app.state.active_trainings
    failed_trainings = request.app.state.failed_trainings

    def fetch_market_snapshot(ticker: str) -> dict:
        try:
            t = yf.Ticker(ticker)
            df = t.history(period="5d")
            if not df.empty:
                closes = [float(c) for c in df["Close"].dropna().tolist()]
                current = closes[-1] if closes else 0.0
                previous = closes[-2] if len(closes) >= 2 else current
            else:
                current = _safe_float(t.fast_info.get("lastPrice", 0.0))
                previous = _safe_float(t.fast_info.get("previousClose", current))
            daily_change = ((current - previous) / previous * 100) if previous else 0.0
            return {"current_price": current, "daily_change_pct": daily_change}
        except Exception:
            return {"current_price": 0.0, "daily_change_pct": 0.0}

    snapshots = await asyncio.gather(*(asyncio.to_thread(fetch_market_snapshot, t) for t in tickers))

    async def process_ticker(ticker: str, snapshot: dict) -> dict:
        recent_preds = db.get_recent_predictions(ticker, limit=20)
        predictions = {}
        for pred in recent_preds:
            label = HORIZON_LABELS.get(pred.get("horizon_days"))
            if label and label not in predictions:
                predictions[label] = _prediction_to_badge(pred)

        predictor = StockPredictor(db)
        for horizon_days, label in HORIZON_LABELS.items():
            if label not in predictions:
                model, model_type = predictor._load_model(ticker, horizon_days)
                if model:
                    try:
                        new_pred = await predictor.predict(ticker, horizon_days=horizon_days, fast_fallback=False)
                        predictions[label] = _prediction_to_badge(new_pred)
                    except Exception:
                        pass
                else:
                    if (ticker, horizon_days) in failed_trainings:
                        try:
                            new_pred = await predictor.predict(ticker, horizon_days=horizon_days, fast_fallback=True)
                            predictions[label] = _prediction_to_badge(new_pred)
                        except Exception:
                            pass
                    else:
                        predictions[label] = {
                            "direction": "TRAINING",
                            "confidence": 0.0,
                            "horizon_days": horizon_days
                        }
                        if (ticker, horizon_days) not in active_trainings:
                            active_trainings.add((ticker, horizon_days))
                            asyncio.create_task(background_train_and_predict(ticker, horizon_days, request.app.state))

        cached_advisory = db.get_cached_advisory(ticker, days=5)
        sector = await asyncio.to_thread(db.get_ticker_sector, ticker)
        current_price = _safe_float(snapshot.get("current_price"))
        daily_change_pct = _safe_float(snapshot.get("daily_change_pct"))

        return {
            "ticker": ticker,
            "sector": sector or "Unknown",
            "price": current_price,
            "current_price": current_price,
            "daily_change_pct": daily_change_pct,
            "predictions": predictions,
            "cached_prediction": recent_preds[0] if recent_preds else None,
            "cached_advisory": cached_advisory,
            "cached_debate": cached_advisory,
        }

    data = await asyncio.gather(*(process_ticker(t, s) for t, s in zip(tickers, snapshots)))
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
                        "predicted_direction": cached.get("ml_prediction", {}).get("predicted_direction", "UNKNOWN"),
                        "confidence": cached.get("ml_prediction", {}).get("confidence", 0.0),
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
                "predicted_direction": result.get("ml_prediction", {}).get("predicted_direction", "UNKNOWN"),
                "confidence": result.get("ml_prediction", {}).get("confidence", 0.0),
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

            client = get_client()
            if not client:
                await queue.put({"event": "error", "data": "❌ LLM not configured."})
                return

            model = (
                settings.gemini_model_chat_shallow if decision == "shallow"
                else settings.gemini_model_chat_complex
            )
            if not model:
                model = "gemini-3.1-flash-lite" if decision == "shallow" else "gemini-3-flash-preview"

            prompt = build_chat_prompt(message, context)

            config = types.GenerateContentConfig(
                safety_settings=DEFAULT_SAFETY_SETTINGS
            )
            if decision == "complex":
                config.thinking_config = types.ThinkingConfig(thinking_level=types.ThinkingLevel.MEDIUM)

            # This is the primary user-facing chat path and it recorded nothing
            # at all until now. Streamed responses carry usage_metadata on the
            # trailing chunks, so keep the last one seen and log it after the
            # stream drains.
            with track_llm(db, model, "chat_stream") as usage:
                response = await client.aio.models.generate_content_stream(
                    model=model,
                    contents=prompt,
                    config=config
                )

                collected = []
                final_usage = None
                async for chunk in response:
                    if chunk.text:
                        collected.append(chunk.text)
                        await queue.put({"event": "token", "data": json.dumps({"text": chunk.text})})
                    if getattr(chunk, "usage_metadata", None):
                        final_usage = chunk.usage_metadata

                if final_usage is not None:
                    usage.prompt_tokens = final_usage.prompt_token_count
                    usage.candidate_tokens = final_usage.candidates_token_count
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


_trending_cache = {}

@router.get("/api/trending")
async def get_trending(request: Request, hours: int = 24, limit: int = 15, refresh: bool = False):
    global _trending_cache
    
    current_time = time.time()
    cache_key = f"{hours}_{limit}"
    
    if not refresh and cache_key in _trending_cache:
        cached_data, cache_time = _trending_cache[cache_key]
        if current_time - cache_time < 86400:
            return {"data": cached_data, "cached": True}

    db = getattr(request.app.state, "db", None) or Database()
    trending_tickers = db.get_top_trending_tickers(hours=hours, limit=limit)
    if not trending_tickers:
        return {"data": []}

    # 1. Collect all context first
    all_ticker_contexts = {}
    for t in trending_tickers:
        ticker_name = t['ticker']
        summaries = db.get_recent_summaries_for_ticker(ticker_name, hours=hours)
        if summaries:
            all_ticker_contexts[ticker_name] = summaries

    # 2. Make a single batch LLM call
    ai_summaries = {}
    client = get_client()
    if client and all_ticker_contexts:
        prompt = (
            f"You are a Professional, precise, and highly analytical Wall Street analyst.\n"
            f"Below is a list of trending tickers and their recent news summaries.\n"
            f"For EACH ticker, write a concise 1-sentence explanation of exactly why it is trending based ONLY on the context.\n"
            f"You MUST include an exact quote from the context if available. Do NOT use emojis.\n"
            f"Do NOT use markdown or HTML tags in your summaries. Just plain text.\n"
            f"Return one entry per ticker in the required response schema.\n\n"
        )
        for tk, sums in all_ticker_contexts.items():
            prompt += f"Ticker: {tk}\nContext:\n" + "\n".join(f"- {s}" for s in sums) + "\n\n"

        try:
            loop = asyncio.get_running_loop()
            def ask_batch_llm():
                start_time = time.time()
                is_error = False
                error_msg = None
                response_text = None
                try:
                    response = client.models.generate_content(
                        model=settings.gemini_model_chat,
                        contents=prompt,
                        config={
                            'safety_settings': DEFAULT_SAFETY_SETTINGS,
                            'response_mime_type': 'application/json',
                            'response_schema': list[TickerNote],
                            'thinking_config': types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW)
                        }
                    )
                    response_text = response.text
                except Exception as e:
                    is_error = True
                    error_msg = str(e)
                    raise
                finally:
                    latency_ms = int((time.time() - start_time) * 1000)
                    if not is_error and response and response.usage_metadata:
                        db.log_llm_usage(
                            model_name=settings.gemini_model_chat,
                            operation="trending_summary_batch",
                            prompt_tokens=response.usage_metadata.prompt_token_count,
                            candidate_tokens=response.usage_metadata.candidates_token_count,
                            latency_ms=latency_ms,
                            is_error=False,
                            prompt_text=prompt,
                            response_text=response_text
                        )
                    elif is_error:
                        db.log_llm_usage(
                            model_name=settings.gemini_model_chat,
                            operation="trending_summary_batch",
                            prompt_tokens=0,
                            candidate_tokens=0,
                            latency_ms=latency_ms,
                            is_error=True,
                            error_message=error_msg,
                            prompt_text=prompt,
                            response_text=None
                        )
                if isinstance(response.parsed, list):
                    return notes_to_dict(response.parsed)
                return notes_to_dict(parse_structured(response.text, list[TickerNote]))

            ai_summaries = await loop.run_in_executor(None, ask_batch_llm)
        except Exception as e:
            log.error("api.trending_batch_summary_failed", error=str(e))

    # 3. Construct data payload with top articles per ticker
    data = []
    for t in trending_tickers:
        ticker = t["ticker"]
        articles = db.get_recent_articles_for_ticker(ticker, hours=hours, limit=5)
        data.append({
            "ticker": ticker,
            "mention_count": t["mention_count"],
            "avg_sentiment": t["avg_sentiment"],
            "summary": ai_summaries.get(ticker.upper()) or "No AI summary available.",
            "articles": articles,
        })

    _trending_cache[cache_key] = (data, current_time)

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
    stats = db.get_stats()
    with db.connection() as conn:
        stats["total_predictions"] = conn.execute("SELECT COUNT(*) AS c FROM predictions").fetchone()["c"]
        stats["total_reflections"] = conn.execute("SELECT COUNT(*) AS c FROM reflection_log").fetchone()["c"]
    stats["db_size_bytes"] = int(_safe_float(stats.get("db_size_mb")) * 1024 * 1024)
    stats["watchlist_size"] = len(db.get_tracked_tickers())
    return stats

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
        embedder = GeminiEmbedder()
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
    forecast = await forecaster._generate_sector_outlook(sector)
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
        "embedding_status", "events_updated",
    ]
    subscriber = event_bus.subscribe(topics)

    async def _build_brain_snapshot():
        """Build a full snapshot for late-joining SSE clients."""
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
            # 1. Send full snapshot first
            snapshot = await _build_brain_snapshot()
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
