"""
Project Scrooge V2 — Web Search Provider

Abstracts web search behind a common interface so the debate agents can
enrich their news context with real-time search results (e.g. Tavily).
Search results are then distilled into concise, factual summaries via
DeepSeek (non-thinking, fast) — the same cheap model used for news
classification — so the debate agents receive actionable insights
rather than raw article dumps.
Gracefully degrades if no provider is configured or summarization fails.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from config.llm import get_deepseek_client, is_deepseek_configured
from config.logging_config import get_logger
from config.settings import settings

log = get_logger(__name__)


@dataclass
class WebSearchResult:
    """A single hit from a web search provider."""

    title: str
    url: str
    content: str
    source: str = ""
    published_date: Optional[str] = None
    score: float = 0.0


# ── Abstract interface ───────────────────────────────────────────────


class WebSearchProvider(ABC):
    """Interface every web-search backend must implement."""

    @abstractmethod
    async def search(self, query: str, max_results: int = 5) -> list[WebSearchResult]:
        ...


# ── Tavily implementation ────────────────────────────────────────────


class TavilySearchProvider(WebSearchProvider):
    """Web search backed by the Tavily API (tavily-python SDK)."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._client: Optional["AsyncTavilyClient"] = None  # type: ignore  # noqa: F821

    # ------------------------------------------------------------------
    # Lazy-import the SDK so a missing / broken tavily-python doesn't
    # prevent the module from being loaded at all.
    # ------------------------------------------------------------------
    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from tavily import AsyncTavilyClient

            self._client = AsyncTavilyClient(api_key=self._api_key)
        except ImportError:
            log.error("tavily.not_installed", msg="tavily-python is not installed. pip install tavily-python")
            raise
        return self._client

    async def search(self, query: str, max_results: int = 5) -> list[WebSearchResult]:
        client = self._get_client()
        try:
            response = await client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results,
            )
        except Exception:
            log.warning("tavily.search_failed", query=query, exc_info=True)
            return []

        raw_results = response.get("results", []) if isinstance(response, dict) else []
        parsed: list[WebSearchResult] = []
        for item in raw_results:
            try:
                parsed.append(
                    WebSearchResult(
                        title=item.get("title", "") or "",
                        url=item.get("url", "") or "",
                        content=item.get("content", "") or "",
                        source=item.get("source", "") or self._extract_domain(item.get("url", "")),
                        published_date=item.get("published_date"),
                        score=float(item.get("score", 0.0) or 0.0),
                    )
                )
            except Exception:
                continue  # skip malformed items
        return parsed

    @staticmethod
    def _extract_domain(url: str) -> str:
        """Pull a readable domain from a URL for use as a source label."""
        if "://" in url:
            url = url.split("://", 1)[1]
        return url.split("/", 1)[0].replace("www.", "")


# ── Factory ──────────────────────────────────────────────────────────


def create_search_provider() -> Optional[WebSearchProvider]:
    """Return a configured provider based on *settings*, or *None*."""
    provider_name = (settings.web_search_provider or "").strip().lower()

    if provider_name == "tavily":
        if not settings.tavily_api_key:
            log.info("web_search.disabled", reason="TAVILY_API_KEY is not set")
            return None
        return TavilySearchProvider(api_key=settings.tavily_api_key)

    # Future: elif provider_name == "brave": ...

    log.info("web_search.unknown_provider", provider=provider_name)
    return None


# ── Convenience helpers ──────────────────────────────────────────────


def build_ticker_search_query(ticker: str) -> str:
    """Build a natural-language search query for a ticker's latest news.

    Focuses on recent events, catalysts, and developments rather than
    routine price data — the goal is to give debate agents facts about
    upcoming earnings, product launches, analyst moves, and regulatory
    news they can cite in their arguments.
    """
    month_str = datetime.now(timezone.utc).strftime("%B %Y")
    # The natural-language framing steers Tavily toward news articles about
    # actual events rather than static data-aggregator pages.
    return f"what is happening with {ticker} stock {month_str} news catalysts"


SEARCH_SUMMARIZATION_PROMPT = """You are a financial news analyst. Below are raw web search results about {ticker}.

For EACH article, write a concise 2-3 sentence factual summary. Focus on:
- Key events, catalysts, earnings figures, or regulatory developments
- Specific upcoming dates (product launches, earnings calls, IPO dates, etc.)
- Analyst ratings, price targets, or material financial data
- Concrete numbers and figures mentioned

Keep each summary purely factual — no speculation, no editorializing.
Include the source name at the start of each summary in brackets.

Respond ONLY with a JSON array of strings, e.g.:
["[Source1] Summary of first article...", "[Source2] Summary of second article..."]

Articles to summarize:
"""


def _fallback_format(results: list[WebSearchResult]) -> str:
    """Simple fallback formatting when DeepSeek is unavailable."""
    if not results:
        return ""
    lines: list[str] = []
    for i, r in enumerate(results, 1):
        date_str = r.published_date or "recent"
        snippet = r.content[:400].rsplit(" ", 1)[0] if len(r.content) > 400 else r.content
        lines.append(
            f"{i}. [{r.source}] {r.title} ({date_str})\n"
            f"   {snippet}"
        )
    return "\n\n".join(lines)


async def summarize_search_results(ticker: str, results: list[WebSearchResult]) -> str:
    """Distill raw search results into concise DeepSeek-generated summaries.

    Uses the same non-thinking DeepSeek model as the news classifier for
    speed and low cost.  Falls back to simple content truncation if the
    LLM is unavailable or summarization fails.
    """
    if not results:
        return ""

    client = get_deepseek_client()
    if not client or not is_deepseek_configured():
        log.info("web_search.no_llm_for_summary", fallback="truncated_content")
        return _fallback_format(results)

    # Build batch prompt
    parts: list[str] = []
    for i, r in enumerate(results, 1):
        content = r.content[:1500] if r.content else ""
        parts.append(
            f"[Article {i}]\n"
            f"Source: {r.source}\n"
            f"Title: {r.title}\n"
            f"Published: {r.published_date or 'unknown'}\n"
            f"Content:\n{content}\n"
        )

    prompt = (
        SEARCH_SUMMARIZATION_PROMPT.format(ticker=ticker)
        + "\n---\n".join(parts)
    )

    try:
        response = await client.chat.completions.create(
            model=settings.deepseek_model_classifier,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
            extra_body={"thinking": {"type": "disabled"}},
        )
        text = response.choices[0].message.content.strip()
        summaries = json.loads(text)

        if isinstance(summaries, dict):
            # DeepSeek sometimes wraps the array in {"summaries": [...]}
            for val in summaries.values():
                if isinstance(val, list):
                    summaries = val
                    break
            else:
                raise ValueError("Unexpected JSON structure")

        if not isinstance(summaries, list):
            raise ValueError(f"Expected a list, got {type(summaries).__name__}")

        return "\n\n".join(
            f"{i}. {s}" for i, s in enumerate(summaries, 1) if isinstance(s, str)
        )
    except Exception as exc:
        log.warning("web_search.summary_failed", ticker=ticker, error=str(exc), fallback="truncated_content")
        return _fallback_format(results)


async def search_ticker_news(ticker: str, max_results: int = 5) -> list[WebSearchResult]:
    """One-shot convenience: create a provider and search for *ticker* news."""
    provider = create_search_provider()
    if provider is None:
        return []
    query = build_ticker_search_query(ticker)
    return await provider.search(query, max_results=max_results)


# ── Chat context enrichment ──────────────────────────────────────────


def _merge_chat_context(db_news: str, web_context: str) -> str:
    """Merge DB news and web results into a single labelled context block.

    Same pattern as StockPredictor._merge_db_and_web — ported here so
    the chat orchestrator can use it without importing predictor.
    """
    parts: list[str] = []

    if db_news:
        parts.append(
            "========================================\n"
            "IN-HOUSE NEWS (classified & ranked)\n"
            "========================================\n"
            f"{db_news}"
        )

    if web_context:
        parts.append(
            "========================================\n"
            "LIVE WEB SEARCH RESULTS\n"
            "========================================\n"
            f"{web_context}"
        )

    if not parts:
        return "No recent news context available."

    return "\n\n".join(parts)


async def enrich_chat_context(
    query: str,
    db_context: str,
    max_results: int = 5,
    research_callback=None,
) -> tuple[str, list[dict]]:
    """Enrich DB news context with live web search for complex chat queries.

    Searches the raw user query directly via Tavily — no ticker extraction
    needed. Falls back to _fallback_format (truncated raw snippets) to
    avoid the per-ticker DeepSeek summarization prompt, which expects a
    ticker symbol.

    Emits streaming research events to research_callback if provided.
    Returns (enriched_context, web_sources_list).
    """
    async def _emit(event_type: str, data: dict):
        if research_callback:
            try:
                await research_callback(event_type, data)
            except Exception:
                pass

    provider = create_search_provider()
    if provider is None:
        return db_context, []

    await _emit("research_start", {"query": query})

    try:
        raw_results = await provider.search(query, max_results=max_results)
    except Exception:
        log.warning("web_search.chat_enrich_failed", query=query[:80], exc_info=True)
        await _emit("research_complete", {"sources_found": 0})
        return db_context, []

    if not raw_results:
        await _emit("research_complete", {"sources_found": 0})
        return db_context, []

    total = len(raw_results)
    web_sources: list[dict] = []

    for i, r in enumerate(raw_results, 1):
        domain = r.source or TavilySearchProvider._extract_domain(r.url)
        web_sources.append({
            "title": r.title,
            "url": r.url,
            "domain": domain,
            "summary": r.content[:250] if r.content else "",
            "score": r.score,
            "source_type": "web",
        })
        await _emit("research_source", {
            "title": r.title,
            "url": r.url,
            "domain": domain,
            "index": i,
            "total": total,
        })

    await _emit("research_complete", {"sources_found": total})

    web_context = _fallback_format(raw_results)
    if not web_context:
        return db_context, web_sources

    return _merge_chat_context(db_context, web_context), web_sources

