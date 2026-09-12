"""
Deus — Macro calendar service.

Two ways a macro event gets into the database, in order of authority:

1. **`seed()`** — the official schedule compiled into `data/macro_calendar.py`,
   read off federalreserve.gov, bls.gov, bea.gov, census.gov and nyse.com. This
   is the source of truth and runs on every worker start; it is an upsert, so
   restarting does not duplicate anything.
2. **`refresh_from_web()`** — a monthly Tavily search plus one extraction call,
   for dates published *after* the seed's verification date. BLS, BEA and Census
   publish the following year in the autumn, so without this the calendar simply
   runs dry in January. It is a top-up and is treated as such: a web row can
   never overwrite a seed or manual row (`Database.upsert_macro_event`), because
   an LLM paraphrasing a third-party calendar is exactly how a wrong FOMC date
   would get in.

`upcoming()` and `context_lines()` are the read side. `context_lines()` exists
so a price-drop alert can say "CPI (September 2026) today 08:30 ET" instead of
inventing a macro explanation, and so the weekly tip and daily stance can splice
the same phrasing into their prompts.

Nothing here raises: every public method is reachable from a scheduled job, and
a Tavily outage or an unset model must degrade to zero new rows, not kill the
job that called it.
"""

from __future__ import annotations

import datetime as dt
from typing import Literal, Optional
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field

from config.llm import complete, is_llm_configured, parse_structured
from config.logging_config import get_logger
from config.settings import settings
from config.usage import track_llm
from data.database import Database
from data.macro_calendar import (
    KNOWN_GAPS,
    MACRO_EVENT_KINDS,
    VERIFIED_AGAINST,
    importance_for,
    seed_rows,
)
from pipeline.date_utils import normalize_date
from pipeline.web_search import WebSearchResult, create_search_provider

log = get_logger(__name__)

# Every date in macro_events is an Eastern-time date — an 08:30 CPI print and a
# 16:00 quarter end are both stamped with the US session they belong to. So
# "today" here has to be the Eastern day, not the host's.
#
# The worker runs in Asia/Seoul, thirteen to fourteen hours ahead, which means a
# naive date.today() is already tomorrow for most of the US session: at 03:00 KST
# it would drop an FOMC decision that is being announced at that very moment.
# Same class of bug as the 'localtime' note on get_last_alert_abs_pct_today.
EASTERN = ZoneInfo("America/New_York")


def today_et() -> dt.date:
    """The current Eastern-time date — the calendar's own notion of today."""
    return dt.datetime.now(EASTERN).date()


# How far ahead a web-extracted date may be. Anything past this is a model
# hallucinating a year rather than reading a published schedule — no agency
# publishes more than about thirteen months out.
MAX_WEB_HORIZON_DAYS = 400

# Searches per refresh. Two schedule queries plus one per month of the near
# window: enough to catch a newly published year without turning a monthly job
# into a Tavily bill.
WEB_SEARCH_MONTHS_AHEAD = 3

# Per-search result count. Deliberately not settings.web_search_max_results —
# that one is tuned for chat answers, and a schedule page needs breadth rather
# than depth.
WEB_SEARCH_RESULTS_PER_QUERY = 5

# The Literal in MacroEventExtract has to be written out: a json_schema enum
# needs literal members at class-definition time, so it cannot be generated from
# MACRO_EVENT_KINDS. The test asserts the two agree, which is what keeps them so.
MacroEventKind = Literal[
    "fomc", "fomc_minutes", "cpi", "ppi", "pce", "nfp", "gdp", "retail_sales",
    "fed_speech", "jackson_hole", "opex", "quad_witching", "month_end",
    "quarter_end", "holiday", "early_close", "earnings_season", "other",
]


class MacroEventExtract(BaseModel):
    """One macro event read out of web search results."""

    date: str = Field(description="Release or event date, YYYY-MM-DD")
    name: str = Field(description="Short event name, e.g. 'CPI (October 2027)'")
    kind: MacroEventKind = Field(description="Event category")
    time_et: Optional[str] = Field(
        default=None, description="Release time in Eastern time as HH:MM, or null"
    )
    importance: int = Field(
        default=1, ge=1, le=3,
        description="3 = moves the whole market, 2 = often moves it, 1 = context",
    )


EXTRACTION_PROMPT = """You are reading web search results for official US macro event schedules.

Extract ONLY scheduled economic releases and market-structure dates that the text
states explicitly, with a specific calendar date. Rules:

- A date you cannot read directly out of the text is not an answer. Omit it.
  Returning five real dates beats returning twenty with three invented ones.
- Do NOT compute, infer or remember dates. No "CPI is usually released mid-month".
- Dates must be YYYY-MM-DD. Ignore anything vaguer than a single day
  ("mid-November", "Q3", "to be announced").
- Ignore anything already in the past and anything more than a year ahead.
- Times are US Eastern, 24-hour HH:MM. Use null if the text does not give one.
- `kind` must be one of the listed categories; use "other" for a real scheduled
  macro event that fits none of them.
- Importance: 3 for FOMC decisions, CPI and the jobs report; 2 for PCE, GDP, PPI,
  retail sales, quad witching and Jackson Hole; 1 for everything else.

Return one entry per event.

SEARCH RESULTS:
"""


class MacroCalendar:
    """Reads and writes the `macro_events` table."""

    def __init__(self, db: Database) -> None:
        self.db = db

    # ── Write side ───────────────────────────────────────────────────────

    def seed(self) -> dict:
        """
        Upsert the compiled official schedule. Synchronous and idempotent.

        Called from `PipelineOrchestrator.start()` before the scheduler starts,
        so the calendar is populated on a fresh database before any job that
        reads it runs. Re-running changes no row count — matching is on
        (date, kind, name) — which is what makes it safe on every boot.
        """
        rows = seed_rows()
        counts = {"rows": len(rows), "inserted": 0, "updated": 0, "skipped": 0}
        for row in rows:
            try:
                counts[self.db.upsert_macro_event(row)] += 1
            except Exception as e:
                log.error("macro_calendar.seed_row_failed",
                          date=row.get("date"), kind=row.get("kind"), error=str(e))
        # The gap list is logged rather than silently carried so that "why is
        # there no 2027 CPI?" is answerable from the worker log instead of
        # looking like a seeding bug.
        log.info("macro_calendar.seeded", verified_against=VERIFIED_AGAINST,
                 known_gaps=len(KNOWN_GAPS), **counts)
        return counts

    async def refresh_from_web(self) -> dict:
        """
        Top up the calendar with dates published since the seed was verified.

        Returns counts for every stage, all zero when Tavily or the extraction
        model is unconfigured. Never raises: this runs unattended on a monthly
        cron and an exception here would abort the job, not just the refresh.
        """
        counts = {"queries": 0, "results": 0, "extracted": 0, "rejected": 0,
                  "inserted": 0, "updated": 0, "skipped": 0}

        provider = create_search_provider()
        if provider is None:
            log.info("macro_calendar.refresh_skipped", reason="no_search_provider")
            return counts
        if not is_llm_configured() or not settings.model_extract:
            log.info("macro_calendar.refresh_skipped", reason="model_extract_unset")
            return counts

        today = today_et()
        results: list[WebSearchResult] = []
        for query in self._build_queries(today):
            counts["queries"] += 1
            try:
                hits = await provider.search(
                    query, max_results=WEB_SEARCH_RESULTS_PER_QUERY
                )
            except Exception as e:
                log.warning("macro_calendar.search_failed", query=query, error=str(e))
                continue
            results.extend(hits)

        counts["results"] = len(results)
        if not results:
            log.info("macro_calendar.refresh_no_results", queries=counts["queries"])
            return counts

        extracted = await self._extract(results)
        counts["extracted"] = len(extracted)

        for candidate in extracted:
            row = self._validate(candidate, today)
            if row is None:
                counts["rejected"] += 1
                continue
            try:
                counts[self.db.upsert_macro_event(row)] += 1
            except Exception as e:
                log.error("macro_calendar.upsert_failed",
                          date=row["date"], kind=row["kind"], error=str(e))
                counts["rejected"] += 1

        log.info("macro_calendar.refreshed", **counts)
        return counts

    @staticmethod
    def _build_queries(today: dt.date) -> list[str]:
        """Search queries for a refresh, dated so Tavily prefers live schedules."""
        queries = [
            f"FOMC meeting schedule {today.year}",
            f"FOMC meeting schedule {today.year + 1}",
        ]
        cursor = today
        for _ in range(WEB_SEARCH_MONTHS_AHEAD):
            queries.append(
                f"economic calendar {cursor.strftime('%B %Y')} "
                "CPI PPI jobs report GDP release dates"
            )
            # First of next month, without dateutil: day 28 plus four days is
            # always inside the following month for every month length.
            cursor = (cursor.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
        return queries

    async def _extract(self, results: list[WebSearchResult]) -> list[MacroEventExtract]:
        """One batched extraction call over every search hit. Never raises."""
        blocks: list[str] = []
        for i, hit in enumerate(results, 1):
            blocks.append(
                f"[Result {i}]\n"
                f"Source: {hit.source or hit.url}\n"
                f"Title: {hit.title}\n"
                f"Content:\n{(hit.content or '')[:1500]}\n"
            )
        prompt = EXTRACTION_PROMPT + "\n---\n".join(blocks)

        try:
            with track_llm(self.db, settings.model_extract,
                           "macro_calendar_extract",
                           prompt_text=prompt, store_text=True) as u:
                u.response = response = await complete(
                    model=settings.model_extract,
                    prompt=prompt,
                    schema=list[MacroEventExtract],
                    temperature=0.0,
                    reasoning="none",
                )

            if isinstance(response.parsed, list):
                return [r for r in response.parsed if isinstance(r, MacroEventExtract)]
            # Every schema call site keeps this fallback: `strict` is off on the
            # json_schema, so a model can still answer with prose around the JSON.
            parsed = parse_structured(response.text, list[MacroEventExtract])
            return list(parsed) if parsed else []
        except Exception as e:
            log.error("macro_calendar.extract_failed", error=str(e))
            return []

    def _validate(self, candidate: MacroEventExtract, today: dt.date) -> Optional[dict]:
        """
        Turn one extracted event into an insertable row, or None.

        Three independent rejections, each of which has a realistic failure mode:
        an unparseable or relative date, a kind outside the enum (the schema
        constrains it but `strict` is off, so the fallback parser can still yield
        one), and a date outside [today, today + MAX_WEB_HORIZON_DAYS] — which is
        what catches a model answering with last year's schedule or a fabricated
        one two years out.
        """
        iso = normalize_date(candidate.date or "")
        if not iso:
            log.debug("macro_calendar.rejected", reason="unparseable_date",
                      raw=candidate.date)
            return None
        try:
            day = dt.date.fromisoformat(iso)
        except ValueError:
            log.debug("macro_calendar.rejected", reason="bad_iso_date", raw=iso)
            return None

        if candidate.kind not in MACRO_EVENT_KINDS:
            log.debug("macro_calendar.rejected", reason="unknown_kind",
                      kind=candidate.kind)
            return None

        if day < today or day > today + dt.timedelta(days=MAX_WEB_HORIZON_DAYS):
            log.debug("macro_calendar.rejected", reason="out_of_window", date=iso)
            return None

        name = (candidate.name or "").strip()
        if not name:
            log.debug("macro_calendar.rejected", reason="empty_name", date=iso)
            return None

        return {
            "date": iso,
            "time_et": self._normalize_time(candidate.time_et),
            "name": name[:200],
            "kind": candidate.kind,
            # The kind decides importance, not the model: letting it self-rate
            # makes the dashboard's importance dots disagree with the seed rows
            # for the same kind of release.
            "importance": importance_for(candidate.kind),
            "source": "web",
            "notes": f"Extracted from web search on {today.isoformat()}.",
        }

    @staticmethod
    def _normalize_time(raw: str | None) -> Optional[str]:
        """Accept only a well-formed 24h HH:MM. Anything else becomes None."""
        if not raw:
            return None
        text = raw.strip()
        hh, _, mm = text.partition(":")
        if len(text) == 5 and hh.isdigit() and mm.isdigit():
            if 0 <= int(hh) < 24 and 0 <= int(mm) < 60:
                return text
        return None

    # ── Read side ────────────────────────────────────────────────────────

    def upcoming(self, days: int = 14) -> list[dict]:
        """
        Macro events from today through today + `days`, inclusive.

        Consumed by /api/brain/macro-events, the Telegram /macro command, and
        the scheduler's start-up check for an exhausted calendar.
        """
        today = today_et()
        return self.db.get_macro_events(
            today.isoformat(), (today + dt.timedelta(days=max(days, 0))).isoformat()
        )

    def context_lines(self, date_et: dt.date | str | None = None) -> list[str]:
        """
        One line per macro event landing today or tomorrow, for prompt context.

        Reads like "CPI (September 2026) today 08:30 ET" and "FOMC rate decision
        tomorrow 14:00 ET". Two days rather than a week because the consumers are
        all answering "does something scheduled explain this, or is it about to?"
        — a release eight days out is not context for today's move.

        `date_et` is the Eastern-time "today" the caller is reasoning about, and
        defaults to it. It stays a parameter because the callers that matter —
        a price alert explaining a specific session, a weekly tip built for a
        named week — are reasoning about a particular day rather than right now.
        """
        if date_et is None:
            anchor = today_et()
        elif isinstance(date_et, str):
            try:
                anchor = dt.date.fromisoformat(date_et[:10])
            except ValueError:
                log.warning("macro_calendar.bad_context_date", raw=date_et)
                return []
        else:
            anchor = date_et

        tomorrow = anchor + dt.timedelta(days=1)
        rows = self.db.get_macro_events(anchor.isoformat(), tomorrow.isoformat())

        lines: list[str] = []
        for row in rows:
            when = "today" if row["date"] == anchor.isoformat() else "tomorrow"
            line = f"{row['name']} {when}"
            if row.get("time_et"):
                line += f" {row['time_et']} ET"
            lines.append(line)
        return lines
