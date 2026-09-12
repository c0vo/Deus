"""
Deus — grade the evidence first, search the web only when it is thin.

Two things used to ask a model to explain something without ever checking
whether it had anything to explain it with:

  * `MarketScanner._generate_and_send_alert` handed the model a two-day news
    query and, when that came back empty, *told it to reason from general
    knowledge*. The result was a guaranteed generic answer — "profit-taking
    after a strong run, likely a short-term overreaction" — for every ticker on
    every red day, indistinguishable from a real catalyst.
  * the chat graph searched the web only when the router happened to say
    "complex", so a shallow question about a ticker the database has never
    heard of got the same bluff.

This module is the shared fix, and it is deliberately two steps:

  `grade_context()`  — does the retrieved context actually answer this, with a
                       dated, concrete fact? Empty context is answered without
                       an LLM call. A grading failure reports *insufficient*,
                       so the fallback is to go and look rather than to bluff.
  `explain_move()`   — the price-alert consumer: numbered evidence, optional
                       Tavily top-up, one structured call, and an honesty rule
                       with teeth.

The honesty rule is the whole point. If no numbered item explains the move, the
answer says so and the cause is *templated from the index context* — "market-wide:
SPY -2.10%, QQQ -2.80%" or "idiosyncratic" — rather than written by the model.
An unexplained move is real information; a fabricated explanation for one is
worse than silence, because it reads exactly like a real one.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from dataclasses import dataclass, field
from typing import Literal, Optional

from pydantic import BaseModel, Field

from config.llm import complete, config_error, is_llm_configured, parse_structured
from config.logging_config import get_logger
from config.settings import settings
from config.usage import track_llm
from data.database import Database
from pipeline.web_search import (
    TavilySearchProvider,
    WebSearchResult,
    _fallback_format,
    build_move_search_query,
    create_search_provider,
)

log = get_logger(__name__)

# How far back a price move may reach for its own explanation. A catalyst from
# last week is not why a stock is down today; two sessions is wide enough to
# cover an after-hours print that moved the next open.
MOVE_EVIDENCE_HOURS = 48
MOVE_EVIDENCE_LIMIT = 8

# Cited sources carried on a GroundedAnswer. The Telegram message shows at most
# three; keeping a couple more means the dashboard row can show what the message
# had no room for.
MAX_CITED_SOURCES = 5

# The sentence a grounded answer must open with when nothing was found. Shared
# so the chat prompt's rule 8 and the test that guards it cannot drift apart.
HONESTY_SENTENCE = (
    "No specific, dated catalyst was found in the database or on the web; "
)


# ── Grading ──────────────────────────────────────────────────────────


class GradeVerdict(BaseModel):
    """Whether retrieved context can answer a question *specifically*."""

    sufficient: bool = Field(
        description="true only if the context contains a dated, concrete fact "
                    "that bears directly on the question"
    )
    specificity: Literal["specific", "generic", "none"] = Field(
        description="specific = names a dated event; generic = true of any "
                    "stock in any week; none = nothing relevant at all"
    )
    reason: str = Field(default="", description="One sentence, under 200 characters")


GRADER_PROMPT = """You are grading retrieved context, not answering the question.

QUESTION ({purpose}):
{query}

RETRIEVED CONTEXT:
{context}

Decide whether this context lets an analyst answer the question SPECIFICALLY.

- sufficient = true requires at least one concrete, dated fact in the context
  that bears directly on the question: a named event, a number, a filing, a
  guidance change, a named counterparty.
- sufficient = false if the only available answer would be true of any company
  in any week — "profit taking", "broad risk-off sentiment", "valuation
  concerns", "mixed analyst views". That is the failure this grade exists to
  catch.
- specificity: "specific" when a dated event is named, "generic" when the
  context is on-topic but says nothing datable, "none" when the context is
  empty or about something else.
- reason: one short sentence naming what is present or missing. Do not answer
  the question.

Err toward false. A false negative costs one web search; a false positive
produces a confident answer with nothing behind it.
"""


async def grade_context(
    db: Optional[Database],
    query: str,
    context: str,
    *,
    purpose: str = "chat",
) -> GradeVerdict:
    """
    Grade whether `context` can answer `query` specifically.

    Never raises and never returns None: every caller uses the verdict to pick a
    branch, so a failure has to be expressible as a verdict. Every failure mode
    — empty context, unset model, unparseable response — reports *insufficient*,
    which routes toward searching rather than toward answering anyway.
    """
    if not (context or "").strip():
        return GradeVerdict(
            sufficient=False, specificity="none",
            reason="No context was retrieved.",
        )

    model = settings.model_grader or settings.model_router
    if not is_llm_configured() or not model:
        return GradeVerdict(
            sufficient=False, specificity="none",
            reason=config_error("MODEL_GRADER"),
        )

    prompt = GRADER_PROMPT.format(
        purpose=purpose, query=query, context=context[:8000]
    )

    try:
        with track_llm(db, model, "context_grader") as u:
            u.response = resp = await complete(
                model=model,
                prompt=prompt,
                schema=GradeVerdict,
                reasoning="none",
                temperature=0.0,
            )

        verdict = resp.parsed if isinstance(resp.parsed, GradeVerdict) else None
        if verdict is None:
            # Every schema call site keeps this fallback: `strict` is off on the
            # json_schema, so a model can still wrap the object in prose.
            verdict = parse_structured(resp.text, GradeVerdict)
        if not isinstance(verdict, GradeVerdict):
            raise ValueError("grader returned no verdict")
        log.info("grader.verdict", purpose=purpose,
                 sufficient=verdict.sufficient, specificity=verdict.specificity)
        return verdict
    except Exception as e:
        log.warning("grader.failed", purpose=purpose, error=str(e))
        return GradeVerdict(
            sufficient=False, specificity="none",
            reason="Grading failed; treating the context as insufficient.",
        )


# ── Session context ──────────────────────────────────────────────────


@dataclass
class SessionContext:
    """Whether a move is the market's or the company's own."""

    label: Literal["market-wide", "idiosyncratic", "unknown"]
    line: str
    moves: dict[str, float] = field(default_factory=dict)


def classify_session_context(
    pct: float, index_context: Optional[dict[str, float]] = None
) -> SessionContext:
    """
    Read a ticker's move against SPY/QQQ on the same session.

    The judgment a reader makes first and an LLM has no way to make at all:
    "down 3% on a day the market is down 2.5%" is a different alert from "down
    3% on a flat tape", and only the second one is about the company.

    A same-direction index move counts as an explanation when it is at least
    half the ticker's move (and at least a full percent, so a 0.4% ticker drift
    is not declared market-wide by a 0.3% index wobble).
    """
    moves = {
        str(k).upper(): float(v)
        for k, v in (index_context or {}).items()
        if v is not None
    }
    if not moves:
        return SessionContext("unknown", "No index context available.", {})

    rendered = ", ".join(f"{k} {v:+.2f}%" for k, v in moves.items())
    aligned = max(
        (abs(v) for v in moves.values() if (v < 0) == (pct < 0)), default=0.0
    )
    if aligned >= max(1.0, abs(pct) * 0.5):
        return SessionContext("market-wide", f"{rendered} -> market-wide", moves)
    return SessionContext("idiosyncratic", f"{rendered} -> idiosyncratic", moves)


# ── Move explanation ─────────────────────────────────────────────────


class MoveExplanation(BaseModel):
    """One structured read on why a price moved."""

    catalyst_found: bool = Field(
        description="true only if a numbered evidence item explains this move"
    )
    catalyst_kind: Literal["company", "sector", "macro", "technical", "unknown"] = (
        Field(default="unknown", description="What kind of catalyst it is")
    )
    cause: str = Field(default="", description="One or two sentences, no speculation")
    sustainability: str = Field(
        default="", description="Whether the move looks durable or an overreaction"
    )
    what_to_watch: str = Field(
        default="", description="The next checkable thing, one sentence"
    )
    source_indices: list[int] = Field(
        default_factory=list,
        description="Numbers of the evidence items the cause rests on",
    )


MOVE_EXPLANATION_PROMPT = """You are a precise markets analyst explaining one price move.

MOVE: {ticker} is {direction} {abs_pct:.2f}% today, at ${price:.2f}.
SESSION: {session_line}
SCHEDULED TODAY: {macro_line}

NUMBERED EVIDENCE (the only facts you may cite):
{evidence}

RULES — the first one decides the answer:

1. If no numbered item above explains THIS move on THIS day, set
   catalyst_found=false, leave `cause` empty, and cite nothing.
   Do NOT speculate. Do NOT fall back on general knowledge, on what usually
   moves stocks, or on what you remember about this company. "Profit taking
   after a strong run", "broad risk-off sentiment" and "valuation concerns" are
   not answers — they are what this rule exists to stop.
2. If one or more items do explain it, set catalyst_found=true and list their
   numbers in source_indices. Every claim in `cause` must be traceable to a
   cited item. Quote the concrete figure or event, not a paraphrase of it.
3. An unexplained move is a legitimate, useful answer. Reporting one honestly
   is worth more than a plausible story, which is indistinguishable from a real
   catalyst to the person reading the alert.
4. `sustainability`: one sentence on whether the move looks durable or an
   overreaction, grounded in the cited evidence or in the session context above.
5. `what_to_watch`: one sentence naming the next checkable thing — a date, a
   level, a release, a filing.
6. Plain text only. No HTML, no Markdown, no emoji. Under 400 characters per
   field.
"""

NO_EVIDENCE_PLACEHOLDER = "(none — no articles in the window and no web results)"


@dataclass
class GroundedAnswer:
    """An answer plus where its grounding came from."""

    text: str = ""
    sources: list[dict] = field(default_factory=list)
    grounded_by: Literal["db", "web", "none"] = "none"
    grade: Optional[GradeVerdict] = None
    explanation: Optional[MoveExplanation] = None
    session: Optional[SessionContext] = None


def _evidence_from_articles(rows: list[dict]) -> list[dict]:
    """Normalise stored articles into citable evidence items."""
    items: list[dict] = []
    for row in rows or []:
        headline = (row.get("headline") or "").strip()
        if not headline:
            continue
        published = str(row.get("published_at") or "")[:10]
        items.append({
            "title": headline,
            "url": row.get("url") or "",
            "source": row.get("source_name") or "in-house",
            "published_at": published,
            "summary": (row.get("classification_summary")
                        or row.get("summary") or "")[:400],
            "kind": "db",
        })
    return items


def _evidence_from_web(hits: list[WebSearchResult]) -> list[dict]:
    items: list[dict] = []
    for hit in hits or []:
        if not (hit.title or hit.content):
            continue
        items.append({
            "title": hit.title or hit.url,
            "url": hit.url or "",
            "source": hit.source or TavilySearchProvider._extract_domain(hit.url or ""),
            "published_at": (hit.published_date or "")[:10],
            "summary": (hit.content or "")[:400],
            "kind": "web",
        })
    return items


def _number_evidence(items: list[dict]) -> str:
    """The numbered block the model cites by index."""
    if not items:
        return NO_EVIDENCE_PLACEHOLDER
    lines: list[str] = []
    for i, item in enumerate(items, 1):
        stamp = item.get("published_at") or "undated"
        lines.append(
            f"{i}. [{stamp} | {item.get('source')}] {item.get('title')}\n"
            f"   {item.get('summary') or '(no summary)'}\n"
            f"   {item.get('url') or '(no url)'}"
        )
    return "\n".join(lines)


def _templated_cause(session: SessionContext, has_evidence: bool) -> str:
    """
    The cause text for a move nothing explains — written here, never by a model.

    Names the index reading explicitly so the reader can tell "the whole tape is
    down" from "only this name is down", which is the actionable half of an
    otherwise unexplained alert.
    """
    searched = (
        "No catalyst in the last 48h of news or on the web"
        if has_evidence
        else "No news in the last 48h and no web results"
    )
    if session.label == "market-wide":
        return f"{searched}. Market-wide: {session.line.split(' -> ')[0]}."
    if session.label == "idiosyncratic":
        return (
            f"{searched}. Idiosyncratic: {session.line.split(' -> ')[0]}, "
            f"so the index does not explain it."
        )
    return f"{searched}, and no index context was available to place the move."


async def explain_move(
    db: Database,
    ticker: str,
    pct: float,
    price: float,
    *,
    index_context: Optional[dict[str, float]] = None,
    macro_today: Optional[list[str]] = None,
    model: str = "",
    allow_web: bool = False,
) -> GroundedAnswer:
    """
    Explain one price move, or state plainly that nothing explains it.

    `allow_web` is the cost control: a tracked ticker's drop is worth a Tavily
    search, a default-watchlist name's 5% wobble is not. Unset either way, the
    function still returns a usable answer — the templated one.
    """
    session = classify_session_context(pct, index_context)
    macro_lines = [line for line in (macro_today or []) if line]
    direction = "up" if pct >= 0 else "down"
    question = (
        f"Why is {ticker} {direction} {abs(pct):.2f}% today?"
    )

    rows = await asyncio.to_thread(
        db.get_recent_articles_for_ticker, ticker,
        MOVE_EVIDENCE_HOURS, MOVE_EVIDENCE_LIMIT,
    )
    items = _evidence_from_articles(rows)
    grounded_by: Literal["db", "web", "none"] = "db" if items else "none"

    grade = await grade_context(
        db, question, _number_evidence(items) if items else "",
        purpose="price_move",
    )

    if not grade.sufficient and allow_web:
        web_items = await _search_the_web(ticker, pct)
        if web_items:
            items.extend(web_items)
            grounded_by = "web"

    if not items:
        # Nothing to cite means nothing to reason over, so there is no call to
        # make: the templated answer is the only honest one and it is free.
        log.info("explain_move.no_evidence", ticker=ticker, pct=round(pct, 2))
        return _unexplained(session, grade, has_evidence=False, macro=macro_lines)

    explanation = await _ask_for_explanation(
        db, ticker, pct, price, items, session, macro_lines, model
    )

    if explanation is None:
        return _unexplained(session, grade, has_evidence=True, macro=macro_lines)

    cited = [
        items[i - 1]
        for i in explanation.source_indices
        if isinstance(i, int) and 1 <= i <= len(items)
    ][:MAX_CITED_SOURCES]

    if not explanation.catalyst_found or not cited:
        # A claimed catalyst with no citable evidence behind it is the same
        # failure as no catalyst at all, so it is demoted rather than trusted.
        if explanation.catalyst_found:
            log.warning("explain_move.uncited_catalyst", ticker=ticker,
                        indices=explanation.source_indices)
        return _unexplained(
            session, grade, has_evidence=True, macro=macro_lines,
            explanation=explanation,
        )

    cause = (explanation.cause or "").strip()
    if not cause:
        return _unexplained(
            session, grade, has_evidence=True, macro=macro_lines,
            explanation=explanation,
        )

    text = " ".join(p for p in (cause, explanation.sustainability.strip()) if p)
    log.info("explain_move.grounded", ticker=ticker, grounded_by=grounded_by,
             kind=explanation.catalyst_kind, sources=len(cited))
    return GroundedAnswer(
        text=text, sources=cited, grounded_by=grounded_by,
        grade=grade, explanation=explanation, session=session,
    )


def _unexplained(
    session: SessionContext,
    grade: GradeVerdict,
    *,
    has_evidence: bool,
    macro: list[str],
    explanation: Optional[MoveExplanation] = None,
) -> GroundedAnswer:
    """
    The honest answer: a templated cause, no citations, grounded_by="none".

    `sustainability` and `what_to_watch` survive from the model when it produced
    them — they are reads on the move rather than claims about its cause — but
    the cause itself is replaced, and no source is cited behind it.
    """
    cause = _templated_cause(session, has_evidence)
    kept = MoveExplanation(
        catalyst_found=False,
        catalyst_kind="macro" if session.label == "market-wide" else "unknown",
        cause=cause,
        sustainability=(explanation.sustainability if explanation else ""),
        what_to_watch=(explanation.what_to_watch if explanation else ""),
        source_indices=[],
    )
    text = cause
    if macro:
        text += f" Scheduled today: {'; '.join(macro[:2])}."
    return GroundedAnswer(
        text=text, sources=[], grounded_by="none",
        grade=grade, explanation=kept, session=session,
    )


async def _search_the_web(ticker: str, pct: float) -> list[dict]:
    """Tavily, phrased as the question a human would type. Never raises."""
    provider = create_search_provider()
    if provider is None:
        return []
    query = build_move_search_query(ticker, pct)
    try:
        hits = await provider.search(
            query, max_results=settings.alert_web_search_max_results
        )
    except Exception as e:
        log.warning("explain_move.web_search_failed", ticker=ticker, error=str(e))
        return []
    log.info("explain_move.web_searched", ticker=ticker, hits=len(hits))
    return _evidence_from_web(hits)


async def _ask_for_explanation(
    db: Database,
    ticker: str,
    pct: float,
    price: float,
    items: list[dict],
    session: SessionContext,
    macro: list[str],
    model: str,
) -> Optional[MoveExplanation]:
    """One structured call. Returns None when it cannot be made or parsed."""
    model = model or settings.model_market_scanner
    if not is_llm_configured() or not model:
        log.info("explain_move.model_unset", hint=config_error("MODEL_MARKET_SCANNER"))
        return None

    prompt = MOVE_EXPLANATION_PROMPT.format(
        ticker=ticker,
        direction="up" if pct >= 0 else "down",
        abs_pct=abs(pct),
        price=price,
        session_line=session.line,
        macro_line="; ".join(macro) if macro else "nothing scheduled",
        evidence=_number_evidence(items),
    )

    try:
        with track_llm(db, model, "explain_move",
                       prompt_text=prompt, store_text=True) as u:
            u.response = resp = await complete(
                model=model,
                prompt=prompt,
                schema=MoveExplanation,
                reasoning="low",
                temperature=0.0,
            )
        if isinstance(resp.parsed, MoveExplanation):
            return resp.parsed
        parsed = parse_structured(resp.text, MoveExplanation)
        return parsed if isinstance(parsed, MoveExplanation) else None
    except Exception as e:
        log.error("explain_move.llm_failed", ticker=ticker, error=str(e))
        return None


def move_window_day() -> dt.date:
    """Today in UTC — the day a search query is dated with. Kept as a seam."""
    return dt.datetime.now(dt.timezone.utc).date()
