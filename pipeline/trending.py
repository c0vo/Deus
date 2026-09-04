"""
Trending tickers with AI "why is this trending" summaries.

Shared by the /api/trending endpoint and the Telegram /trending command. Both
ran identical code and made the same batched Gemini call, but only the API side
cached it, so every /trending in Telegram paid for a summary the dashboard had
already bought. Consolidating here gives them one cache: a /trending in the bot
warms the dashboard and vice versa.

Lives in pipeline/ rather than being imported from api.server because that
module pulls in yfinance, pandas, numpy, StockPredictor, ChatOrchestrator and
several analyzers at import time — far too much to drag into a bot command.
"""

from __future__ import annotations

import asyncio
import time

from config.llm import complete, is_llm_configured, parse_structured
from config.logging_config import get_logger
from config.settings import settings
from config.usage import track_llm
from data.models import TickerNote, notes_to_dict

log = get_logger(__name__)

# Keyed "{hours}_{limit}". Both callers default to limit=15, so the common case
# shares the "24_15" entry.
_trending_cache: dict[str, tuple[list[dict], float]] = {}
TRENDING_TTL = 86400  # 24h — mention counts move slowly and the call is costly.

SUMMARY_PROMPT_HEADER = (
    "You are a Professional, precise, and highly analytical Wall Street analyst.\n"
    "Below is a list of trending tickers and their recent news summaries.\n"
    "For EACH ticker, write a concise 1-sentence explanation of exactly why it is trending based ONLY on the context.\n"
    "You MUST include an exact quote from the context if available. Do NOT use emojis.\n"
    "Do NOT use markdown or HTML tags in your summaries. Just plain text.\n"
    "Return one entry per ticker in the required response schema.\n\n"
)


def _build_prompt(ticker_contexts: dict[str, list[str]]) -> str:
    prompt = SUMMARY_PROMPT_HEADER
    for ticker, summaries in ticker_contexts.items():
        prompt += f"Ticker: {ticker}\nContext:\n" + "\n".join(f"- {s}" for s in summaries) + "\n\n"
    return prompt


async def _summarize(db, ticker_contexts: dict[str, list[str]]) -> dict[str, str]:
    """One batched call covering every trending ticker. Never raises."""
    if not is_llm_configured() or not settings.model_trending or not ticker_contexts:
        return {}

    prompt = _build_prompt(ticker_contexts)

    try:
        with track_llm(db, settings.model_trending, "trending_summary_batch",
                       prompt_text=prompt, store_text=True) as u:
            u.response = response = await complete(
                model=settings.model_trending,
                prompt=prompt,
                schema=list[TickerNote],
                reasoning="low",
            )

        if isinstance(response.parsed, list):
            return notes_to_dict(response.parsed)
        return notes_to_dict(parse_structured(response.text, list[TickerNote]))
    except Exception as e:
        # A failed summary degrades the response to mention counts only; it is
        # not worth failing the whole request over.
        log.error("trending.batch_summary_failed", error=str(e))
        return {}


async def get_trending_with_summaries(
    db,
    hours: int = 24,
    limit: int = 15,
    refresh: bool = False,
) -> tuple[list[dict], bool]:
    """
    Top trending tickers, each with an AI explanation of why it is trending.

    Returns (rows, was_cached). Each row carries ticker, mention_count,
    avg_sentiment, summary, and up to five recent articles. An empty list means
    nothing was trending in the window; that case is not cached, since it is
    usually a cold database rather than a real answer.
    """
    now = time.time()
    cache_key = f"{hours}_{limit}"

    if not refresh:
        cached = _trending_cache.get(cache_key)
        if cached and (now - cached[1]) < TRENDING_TTL:
            return cached[0], True

    trending_tickers = db.get_top_trending_tickers(hours=hours, limit=limit)
    if not trending_tickers:
        return [], False

    ticker_contexts: dict[str, list[str]] = {}
    for t in trending_tickers:
        summaries = db.get_recent_summaries_for_ticker(t["ticker"], hours=hours)
        if summaries:
            ticker_contexts[t["ticker"]] = summaries

    ai_summaries = await _summarize(db, ticker_contexts)

    data = []
    for t in trending_tickers:
        ticker = t["ticker"]
        data.append({
            "ticker": ticker,
            "mention_count": t["mention_count"],
            "avg_sentiment": t["avg_sentiment"],
            # Distinguishes "we had no news to summarise" from "the call failed",
            # which the Telegram renderer surfaces differently.
            "has_context": ticker in ticker_contexts,
            "summary": ai_summaries.get(ticker.upper()) or "No AI summary available.",
            "articles": db.get_recent_articles_for_ticker(ticker, hours=hours, limit=5),
        })

    # Evict expired keys before inserting: /trending accepts an arbitrary hours
    # argument, so the key space would otherwise grow without bound.
    for stale in [k for k, (_, ts) in _trending_cache.items() if (now - ts) >= TRENDING_TTL]:
        del _trending_cache[stale]
    _trending_cache[cache_key] = (data, now)

    return data, False
