"""
Deus — Weekly tip: measured precedents plus the coming week's known risks.

The weekly job this replaces printed five-day percentages and nothing else: no
calendar, no precedent, nothing persisted. What a Sunday-evening message is
actually for is the two things you cannot get from a price screen — what has
historically happened in the week ahead, and what is already scheduled to happen
in it.

The design constraint is that both halves must be checkable, which shapes the
whole module:

* **Facts first, model second.** `gather_facts()` does all the work — the
  seasonality sentences come from `pipeline.seasonality`, the events from the
  macro calendar, the earnings from `ticker_events`, the warnings from the
  article corpus. The model's only job is selection and phrasing, over a FACTS
  block it is told is its entire world.
* **Numbers are verified after the fact.** Any tip citing a number that does not
  appear in FACTS is dropped, not corrected. A model that invents "down 7% in 8
  of the last 10 Septembers" is not having an off day, it is doing the one thing
  this feature exists to avoid.
* **Something always goes out.** An unset `MODEL_WEEKLY_TIP`, a failed call or
  zero surviving tips all degrade to the facts-only render with the reason
  stated. A silent Sunday is indistinguishable from a dead worker, which is the
  same reasoning as `EMPTY_BRIEFING_TEXT`.

Nothing here triggers an expensive generation as a side effect: the macro themes
are read from where `TrendForecaster` stored them and never regenerated, and the
theme clustering runs its numpy path without the LLM naming call.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import re
from typing import Literal, Optional

from pydantic import BaseModel, Field

from api.sse_manager import event_bus
from config.llm import complete, is_llm_configured, parse_structured
from config.logging_config import get_logger
from config.settings import settings
from config.usage import track_llm
from data.database import Database
from data.taxonomy import is_global
from pipeline.event_tracker import EventTracker
from pipeline.ipo_detector import IPODetector
from pipeline.macro_calendar import EASTERN
from pipeline.seasonality import (
    WEEK_DAYS,
    build_ticker_seasonality,
    upcoming_effects,
)
from pipeline.theme_detector import ThemeDetector
from pipeline.trend_forecaster import TrendForecaster

log = get_logger(__name__)

# How many tickers get a full seasonality build. Each one is a history read plus
# a dozen passes over it; a watchlist is normally two or three names, and the cap
# only matters for someone tracking forty.
MAX_SEASONALITY_TICKERS = 10

# Caps on the facts block. The model reads all of it, so every extra row is
# prompt cost for a Sunday message nobody scrolls past the first screen of.
MAX_EFFECTS = 12
MAX_EVENTS = 12
MAX_EARNINGS = 10
MAX_IPOS = 5
MAX_THEMES = 3
MAX_HEADLINES = 5
MAX_TIPS = 5

# Sessions used for the week-over-week number. Five trading days back is "a week
# ago" in the only calendar the prices have.
WOW_SESSIONS = 5

# The importance floor for a headline to count as a warning. Above the briefing's
# own floor of 7.0: over a 168-hour window that one is far too generous, and the
# tip is meant to carry the handful of stories still worth naming a week later.
NEWS_MIN_IMPORTANCE = 7.5
NEWS_WINDOW_HOURS = 168

# How old the stored macro themes may be and still go in as current. The trend
# job rewrites them every four hours, so a day-old row means six runs in a row
# produced nothing, and quoting it would present a stale list as this week's.
MACRO_THEMES_MAX_AGE_HOURS = 24

# Numeric tokens, for the post-validation pass. Deliberately permissive about the
# sign so "+8.8%", "-0.6%" and "45%" all come out whole rather than as a bare
# digit run.
NUMBER_RE = re.compile(r"[+\-]?\d+(?:\.\d+)?%?")

# Models write U+2212 MINUS SIGN and the dashes as readily as ASCII hyphen, and
# a tip dropped for spelling its minus differently would be the validator
# failing, not the model. Both sides of the comparison are normalised.
_DASHES = {"−": "-", "–": "-", "—": "-"}


class Tip(BaseModel):
    """One actionable observation about the coming week."""

    title: str = Field(description="Short headline for the tip, under 70 characters")
    precedent: str = Field(
        description="The historical or scheduled fact this rests on, quoting the "
                    "numbers from FACTS verbatim"
    )
    evidence: str = Field(
        description="Which FACTS entries support it — name the tickers, dates or "
                    "headlines"
    )
    action: str = Field(
        description="What to actually do or watch for, in one or two sentences"
    )
    severity: Literal["info", "watch", "warning"] = Field(
        description="info = context only, watch = worth checking during the week, "
                    "warning = a known risk with a date on it"
    )


WEEKLY_TIP_PROMPT = """You are writing the weekly tip for a self-hosted market dashboard. \
Your reader already owns the positions listed and wants to know two things about the \
week ahead: what has historically happened in weeks like it, and what is already \
scheduled to happen in it.

Write between 2 and 5 tips. Rules, in order of importance:

1. Use ONLY the FACTS block below. It is your entire world. You have no other \
knowledge of this market, this week, or these companies, and you must not draw on any.
2. Every number you cite must appear VERBATIM in FACTS. Do not round it, do not \
restate a percentage as a fraction, do not compute a new figure from two given ones. \
A tip citing a number that is not in FACTS is discarded automatically.
3. Fewer tips beats invented ones. If FACTS only supports two tips, write two. If it \
supports none, return an empty list.
4. A seasonality claim must carry its sample size, because that is what makes it \
checkable — "median -0.6% and up in 45% of 33 years" is a precedent, "September is \
historically weak" is a rumour.
5. Never predict a price or a direction with a number attached. Describe the \
precedent and the risk; the reader decides.
6. Prefer a tip that ties a seasonal precedent to a dated event in the same week over \
either one alone.
7. Plain text only — no markdown, no HTML, no bullet characters. Each field is one to \
three sentences.

FACTS:
"""


class WeeklyTipComposer:
    """Builds, writes and publishes the weekly tip digest."""

    def __init__(self, db: Database, price_feed=None) -> None:
        self.db = db
        # Only `ensure_deep_history` needs the feed, and that is the scheduler's
        # monthly job rather than anything on this path — so /tip can construct a
        # composer without one.
        self.price_feed = price_feed

    # ── Facts ────────────────────────────────────────────────────────────

    async def gather_facts(self, now: Optional[dt.datetime] = None) -> dict:
        """
        Assemble everything the tip is allowed to talk about.

        One `asyncio.to_thread` hop for the whole build: it is a dozen SQLite
        reads plus a numpy clustering pass, and the API process shares this
        database. The anchor is an Eastern-time date because every date this
        reasons about — a macro release, an expiration, a month end — belongs to
        a US session, and the worker runs in Asia/Seoul where a naive
        `date.today()` is already tomorrow.
        """
        if now is None:
            anchor = dt.datetime.now(EASTERN)
        elif now.tzinfo is None:
            anchor = now.replace(tzinfo=EASTERN)
        else:
            anchor = now.astimezone(EASTERN)

        return await asyncio.to_thread(self._gather_sync, anchor)

    def _gather_sync(self, anchor: dt.datetime) -> dict:
        """The blocking half of `gather_facts`. Never raises on a single source."""
        today = anchor.date()
        week_end = today + dt.timedelta(days=WEEK_DAYS - 1)

        benchmarks = [
            t.strip().upper()
            for t in (settings.seasonality_benchmarks or "").split(",")
            if t.strip()
        ]
        try:
            tracked = [t.strip().upper() for t in (self.db.get_tracked_tickers() or [])]
        except Exception as e:
            log.warning("weekly_tip.tracked_failed", error=str(e))
            tracked = []

        universe: list[str] = []
        for symbol in benchmarks + tracked:
            if symbol and symbol not in universe:
                universe.append(symbol)

        macro = self._safe(
            "macro_events",
            lambda: self.db.get_macro_events(today.isoformat(), week_end.isoformat()),
            [],
        )

        stats_by_ticker = {}
        history_depth = {}
        for symbol in universe[:MAX_SEASONALITY_TICKERS]:
            rows = self._safe("price_history", lambda s=symbol: self.db.get_price_history(s), [])
            stats = build_ticker_seasonality(symbol, rows)
            if stats is None:
                continue
            stats_by_ticker[symbol] = stats
            history_depth[symbol] = {
                "first_date": stats.first_date,
                "last_date": stats.last_date,
                "years": stats.years_of_history,
                "bars": stats.bars,
            }

        effects = upcoming_effects(today, stats_by_ticker, macro)

        return {
            "generated_at": anchor.isoformat(),
            "period_start": today.isoformat(),
            "period_end": week_end.isoformat(),
            "model": settings.model_weekly_tip or "",
            "tip_status": "pending",
            "benchmarks": benchmarks,
            "tracked": tracked,
            "history_depth": history_depth,
            "seasonality": [
                {
                    "name": e.name, "window": e.window,
                    "stat_line": e.stat_line, "numbers": e.numbers,
                }
                for e in effects[:MAX_EFFECTS]
            ],
            "events": self._events(macro),
            "earnings": self._earnings(today, week_end),
            "ipos": self._ipos(today, week_end),
            "news_warnings": self._news_warnings(),
            "performance": self._performance(universe),
        }

    @staticmethod
    def _safe(label: str, build, fallback):
        """Run one fact source, logging and degrading instead of aborting the tip."""
        try:
            return build()
        except Exception as e:
            log.warning("weekly_tip.source_failed", source=label, error=str(e))
            return fallback

    @staticmethod
    def _events(macro: list[dict]) -> list[dict]:
        return [
            {
                "date": row.get("date"),
                "time_et": row.get("time_et"),
                "name": row.get("name"),
                "kind": row.get("kind"),
                "importance": row.get("importance") or 1,
                "confirmed": (row.get("source") or "seed") in ("seed", "manual"),
            }
            for row in (macro or [])[:MAX_EVENTS]
        ]

    def _earnings(self, today: dt.date, week_end: dt.date) -> list[dict]:
        rows = self._safe(
            "ticker_events",
            lambda: EventTracker(self.db).get_tracked_events_calendar(WEEK_DAYS),
            [],
        )
        # get_tracked_events_calendar windows on the UTC date, which is up to a
        # day ahead of the Eastern one this tip is written against — so the
        # window is re-applied here rather than trusted.
        inside = [
            row for row in rows or []
            if today.isoformat() <= str(row.get("event_date") or "")[:10] <= week_end.isoformat()
        ]
        return [
            {
                "ticker": row.get("ticker"),
                "date": str(row.get("event_date"))[:10],
                "event_type": row.get("event_type"),
                "title": row.get("event_title"),
                "confidence": row.get("confidence"),
            }
            for row in inside[:MAX_EARNINGS]
        ]

    def _ipos(self, today: dt.date, week_end: dt.date) -> list[dict]:
        rows = self._safe(
            "ipo_watchlist", lambda: IPODetector(self.db).get_ipo_watchlist(), []
        )
        # The watchlist deliberately keeps undated (TBA) rows and a day of
        # already-listed ones. Neither belongs in "happening this week", so the
        # filter is on a real date inside the window.
        inside = [
            row for row in rows or []
            if today.isoformat() <= str(row.get("ipo_date") or "")[:10] <= week_end.isoformat()
        ]
        return [
            {
                "company_name": row.get("company_name"),
                "ticker": row.get("ticker"),
                "date": str(row.get("ipo_date"))[:10],
                "status": row.get("status"),
            }
            for row in inside[:MAX_IPOS]
        ]

    def _news_warnings(self) -> dict:
        # Stored only, never generated: calling generate_and_cache_macro_themes()
        # would turn a digest job into an unannounced model call. No stored
        # themes, or none newer than MACRO_THEMES_MAX_AGE_HOURS, is a missing
        # section rather than an old list presented as current.
        stored = self._safe(
            "macro_themes",
            lambda: TrendForecaster(self.db).get_stored_macro_themes(
                max_age_hours=MACRO_THEMES_MAX_AGE_HOURS
            ),
            None,
        )
        themes = [
            {
                "title": str(t.get("title") or "")[:160],
                "explanation": str(t.get("explanation") or "")[:400],
            }
            for t in (stored["themes"] if stored else [])[:MAX_THEMES]
            if isinstance(t, dict)
        ]

        seeds = self._safe(
            "theme_clusters", lambda: ThemeDetector(self.db).cluster_recent(), []
        )
        accelerating = []
        for seed in (seeds or [])[:MAX_THEMES]:
            if seed.acceleration is None:
                continue
            # seed.title is empty until name_clusters() runs, and that is an LLM
            # call. The top headline is the honest label for an unnamed cluster.
            label = (seed.headlines[0] if seed.headlines else "").strip()
            if not label:
                continue
            accelerating.append({
                "label": label[:160],
                "acceleration": round(float(seed.acceleration), 2),
                "articles_recent": seed.count_recent,
                "articles_baseline": seed.count_base,
            })

        candidates = self._safe(
            "briefing_candidates",
            lambda: self.db.get_briefing_candidates(
                hours=NEWS_WINDOW_HOURS,
                min_importance=NEWS_MIN_IMPORTANCE,
                limit=40,
            ),
            [],
        )
        headlines = [
            {
                "headline": str(row.get("headline") or "")[:200],
                "importance": round(float(row.get("importance_score") or 0.0), 1),
                "source": row.get("source_name"),
                "published_at": str(row.get("published_at") or "")[:10],
            }
            for row in (candidates or []) if is_global(row)
        ][:MAX_HEADLINES]

        return {
            "macro_themes": themes,
            "accelerating_themes": accelerating,
            "headlines": headlines,
        }

    def _performance(self, universe: list[str]) -> list[dict]:
        """Week-over-week move per ticker, and the same number relative to SPY."""
        moves: dict[str, float] = {}
        for symbol in universe[:MAX_SEASONALITY_TICKERS]:
            rows = self._safe(
                "wow_history",
                lambda s=symbol: self.db.get_price_history(s, limit=WOW_SESSIONS + 1),
                [],
            )
            closes = [
                float(r["close"]) for r in rows or []
                if r.get("close") is not None
            ]
            if len(closes) < 2:
                continue
            moves[symbol] = (closes[-1] / closes[0] - 1.0) * 100.0

        spy = moves.get("SPY")
        return [
            {
                "ticker": symbol,
                "pct": round(pct, 2),
                "vs_spy": None if spy is None or symbol == "SPY" else round(pct - spy, 2),
            }
            for symbol, pct in moves.items()
        ]

    # ── Facts as text ────────────────────────────────────────────────────

    @staticmethod
    def render_facts_text(facts: dict) -> str:
        """
        The FACTS block, in plain text.

        Serves two purposes at once and has to serve both exactly: it is what the
        model is shown, and it is the corpus the post-validation pass checks
        every cited number against. A number that reaches the model but not this
        string would be unquotable; one that reaches this string but not the
        model would be unverifiable.
        """
        lines: list[str] = [
            f"WEEK: {facts.get('period_start')} to {facts.get('period_end')} "
            f"(US Eastern dates)",
        ]

        depth = facts.get("history_depth") or {}
        if depth:
            lines.append("")
            lines.append("PRICE HISTORY HELD:")
            for ticker, info in depth.items():
                lines.append(
                    f"- {ticker}: {info.get('bars')} daily bars, "
                    f"{info.get('first_date')} to {info.get('last_date')} "
                    f"({info.get('years')} years)"
                )

        seasonality = facts.get("seasonality") or []
        lines.append("")
        lines.append("SEASONAL PRECEDENTS (measured on the bars above):")
        if seasonality:
            for row in seasonality:
                lines.append(f"- [{row.get('name')} | {row.get('window')}] {row.get('stat_line')}")
        else:
            lines.append("- none computable from the history held.")

        events = facts.get("events") or []
        lines.append("")
        lines.append("SCHEDULED MACRO EVENTS THIS WEEK:")
        if events:
            for row in events:
                when = f" {row.get('time_et')} ET" if row.get("time_et") else ""
                tag = "confirmed" if row.get("confirmed") else "estimated"
                lines.append(
                    f"- {row.get('date')}{when} {row.get('name')} "
                    f"(kind={row.get('kind')}, importance={row.get('importance')}, {tag})"
                )
        else:
            lines.append("- nothing scheduled.")

        earnings = facts.get("earnings") or []
        if earnings:
            lines.append("")
            lines.append("TRACKED-TICKER EVENTS THIS WEEK:")
            for row in earnings:
                title = f" — {row.get('title')}" if row.get("title") else ""
                lines.append(
                    f"- {row.get('date')} {row.get('ticker')} "
                    f"{row.get('event_type')}{title} ({row.get('confidence')})"
                )

        ipos = facts.get("ipos") or []
        if ipos:
            lines.append("")
            lines.append("IPOS THIS WEEK:")
            for row in ipos:
                symbol = f" ({row.get('ticker')})" if row.get("ticker") else ""
                lines.append(
                    f"- {row.get('date')} {row.get('company_name')}{symbol} "
                    f"[{row.get('status')}]"
                )

        warnings = facts.get("news_warnings") or {}
        themes = warnings.get("macro_themes") or []
        if themes:
            lines.append("")
            lines.append("CURRENT MACRO THEMES (from recent news):")
            for row in themes:
                lines.append(f"- {row.get('title')}: {row.get('explanation')}")

        accelerating = warnings.get("accelerating_themes") or []
        if accelerating:
            lines.append("")
            lines.append("FASTEST-ACCELERATING STORY CLUSTERS (share of coverage):")
            for row in accelerating:
                lines.append(
                    f"- {row.get('label')} "
                    f"(acceleration {row.get('acceleration')}x, "
                    f"{row.get('articles_recent')} recent vs "
                    f"{row.get('articles_baseline')} baseline articles)"
                )

        headlines = warnings.get("headlines") or []
        if headlines:
            lines.append("")
            lines.append("HIGH-IMPORTANCE MACRO HEADLINES FROM THE LAST 7 DAYS:")
            for row in headlines:
                lines.append(
                    f"- [{row.get('importance')}] {row.get('headline')} "
                    f"({row.get('source')}, {row.get('published_at')})"
                )

        performance = facts.get("performance") or []
        if performance:
            lines.append("")
            lines.append("LAST WEEK'S MOVE PER TICKER:")
            for row in performance:
                relative = (
                    f", {row.get('vs_spy'):+.2f}pp vs SPY"
                    if row.get("vs_spy") is not None else ""
                )
                lines.append(f"- {row.get('ticker')}: {row.get('pct'):+.2f}%{relative}")

        return "\n".join(lines)

    # ── Composition ──────────────────────────────────────────────────────

    async def compose(self, facts: dict) -> list[Tip]:
        """
        Turn the facts into tips with one LLM call, then verify every number.

        Writes the outcome into `facts["tip_status"]` — `not_configured`,
        `failed`, `empty` or `ok` — because that is what decides the renderer's
        header and it ends up in `facts_json`, where it answers "why was last
        Sunday's tip facts-only?" without a log dig.
        """
        model = settings.model_weekly_tip
        facts_text = self.render_facts_text(facts)

        if not model or not is_llm_configured():
            facts["tip_status"] = "not_configured"
            log.info("weekly_tip.model_not_configured", model_set=bool(model))
            return []

        prompt = WEEKLY_TIP_PROMPT + facts_text
        try:
            with track_llm(self.db, model, "weekly_tip",
                           prompt_text=prompt, store_text=True) as u:
                u.response = response = await complete(
                    model=model,
                    prompt=prompt,
                    schema=list[Tip],
                    temperature=0.3,
                    reasoning="low",
                )
        except Exception as e:
            facts["tip_status"] = "failed"
            log.error("weekly_tip.compose_failed", model=model, error=str(e))
            return []

        if isinstance(response.parsed, list):
            candidates = [t for t in response.parsed if isinstance(t, Tip)]
        else:
            # Every schema call site keeps this fallback: `strict` is off on the
            # json_schema, so a model can still wrap the JSON in prose.
            candidates = list(parse_structured(response.text, list[Tip]) or [])

        kept = [t for t in candidates if self._is_supported(t, facts_text)]
        facts["tips_returned"] = len(candidates)
        facts["tips_dropped"] = len(candidates) - len(kept)
        facts["tip_status"] = "ok" if kept else "empty"
        log.info("weekly_tip.composed", model=model,
                 returned=len(candidates), kept=len(kept))
        return kept[:MAX_TIPS]

    @staticmethod
    def _normalize(text: str) -> str:
        out = str(text or "")
        for bad, good in _DASHES.items():
            out = out.replace(bad, good)
        return out

    @classmethod
    def _is_supported(cls, tip: Tip, facts_text: str) -> bool:
        """
        Drop a tip unless every number in it occurs in the facts text.

        Substring matching rather than token-set matching, which is the lenient
        choice on purpose: "14" appearing inside the date "2026-09-14" should
        count, and a tip dropped because the validator was cleverer than the
        model is a worse failure than one slightly loose number getting through.
        What it does catch is the case that matters — a statistic, a percentage
        or a sample size that exists nowhere in our data.
        """
        haystack = cls._normalize(facts_text)
        claimed: set[str] = set()
        for field_value in (tip.title, tip.precedent, tip.evidence, tip.action):
            claimed.update(NUMBER_RE.findall(cls._normalize(field_value)))

        missing = sorted(token for token in claimed if token not in haystack)
        if missing:
            log.warning("weekly_tip.unsupported_claim_dropped",
                        title=str(tip.title or "")[:80], numbers=missing)
            return False
        return True

    # ── Persistence ──────────────────────────────────────────────────────

    async def persist_and_publish(
        self, facts: dict, tips: list[Tip], body_html: str
    ) -> Optional[int]:
        """
        Store the digest and tell the dashboard. Never raises.

        `body_text` is the facts block rather than a plain-text copy of the
        message: the digest row is then self-verifying — every number in
        `body_html` can be checked against the same string the model was shown,
        which is exactly what §8's verification step does.
        """
        payload = dict(facts)
        payload["tips"] = [t.model_dump() for t in tips]

        digest_id: Optional[int] = None
        try:
            digest_id = await asyncio.to_thread(
                self.db.insert_digest,
                "weekly_tip",
                body_html,
                body_text=self.render_facts_text(facts),
                facts_json=payload,
                model=facts.get("model") or None,
                period_start=facts.get("period_start"),
                period_end=facts.get("period_end"),
            )
        except Exception as e:
            log.error("weekly_tip.persist_failed", error=str(e))

        try:
            await event_bus.publish("weekly_tip", {
                "id": digest_id,
                "kind": "weekly_tip",
                "period_start": facts.get("period_start"),
                "period_end": facts.get("period_end"),
                "model": facts.get("model") or None,
                "tip_status": facts.get("tip_status"),
                "tip_count": len(tips),
                "titles": [t.title for t in tips],
            })
        except Exception as e:
            log.warning("weekly_tip.publish_failed", error=str(e))

        return digest_id
