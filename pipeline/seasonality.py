"""
Deus — Seasonality statistics measured on our own price history.

"September is historically the worst month" is the kind of thing everybody
repeats and nobody checks. A weekly tip that repeats it without a number is
indistinguishable from a horoscope, so every statistic here is computed from
`price_history` — the same daily bars the predictor and the technical ratings
read — and comes back as a finished sentence carrying its own sample size. An
effect whose statistic cannot be computed from the bars we actually hold is
omitted rather than asserted qualitatively. That rule is the module's reason to
exist; breaking it turns the weekly tip back into folklore.

Three layers:

1. **Reshaping** — `monthly_returns` turns daily closes into calendar-month
   returns. Incomplete periods are dropped: a September still in progress is
   not a September, and a month with no successor in the data cannot be proven
   finished.
2. **Distributions** — `month_of_year_stats`, `window_stats`,
   `santa_rally_stats`, `event_drift_stats`, `short_week_stats` and
   `period_end_stats` all return the same `MonthStats` container, or `None`
   when the sample is too thin to quote. `None` is the signal that an effect
   must be left out.
3. **Selection** — `upcoming_effects` is the only function here that knows
   about "this week": given today, the per-ticker statistics and the next seven
   days of the macro calendar, it decides which effects are genuinely in play
   and formats each one.

`ensure_deep_history` is the plumbing half. Ten years of bars is the difference
between "up in 45% of 33 years" and "up in 40% of 5 years", and
`PriceFeed.refresh_history()` only ever stores three months — so the deep pull
has to be asked for explicitly, which is what this does.
"""

from __future__ import annotations

import asyncio
import bisect
import datetime as dt
import math
import statistics
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from config.logging_config import get_logger
from config.settings import settings
from data.macro_calendar import HISTORICAL_FOMC_DECISIONS, third_friday
from pipeline.macro_calendar import today_et

log = get_logger(__name__)

MONTH_NAMES = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)

# Below this there is no distribution worth printing — a "hit rate" over three
# observations is noise wearing a percentage sign. Every statistics function
# takes an override so a test can work with a short synthetic series.
MIN_OBSERVATIONS = 5

# An event-drift statistic is only honest when most of the event dates fall
# inside the bars we hold. Eight is roughly a year of FOMC decisions: fewer than
# that and the statistic is describing one particular year, not a regularity.
MIN_EVENTS_IN_DATA = 8

QUARTER_END_MONTHS = (3, 6, 9, 12)

# The "month-end window": the last three sessions of a month, which is where
# index rebalancing and pension flows land.
PERIOD_END_SESSIONS = 3

# Expiration and FOMC windows are both quoted over a trading week.
OPEX_WEEK_SESSIONS = 5
PRE_FOMC_SESSIONS = 5

# Santa Claus rally, as defined by Hirsch: the last five sessions of the year
# plus the first two of the next.
SANTA_SESSIONS_BEFORE = 5
SANTA_SESSIONS_AFTER = 2

# A week accommodates weekends and the occasional multi-day market closure, but
# not a monthly observation. Session-window statistics must never turn sparse
# history into a made-up daily statistic.
MAX_SESSION_GAP_DAYS = 7

# Same pacing as scripts/manual/backfill_price_history.py. A "max" range pull is
# the single heaviest thing we ask Yahoo for, and doing several back-to-back is
# how the endpoint starts returning 429 to everything else the worker needs.
DEEP_HISTORY_DELAY_SECONDS = 2.0


# ── Containers ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MonthStats:
    """
    The distribution of one recurring calendar window's returns, in percent.

    `years` is the observation count. For a month-of-year or seasonal-half
    statistic that is literally a number of years; for an event-drift statistic
    it is a number of events, and the caller names the unit when it formats the
    sentence. `worst` and `best` carry their own label — a year, a season like
    "2008/09", or an event date — so the extreme is always attributable.
    """

    years: int
    mean: float
    median: float
    hit_rate: float          # fraction of observations that closed positive, 0..1
    worst: tuple[str, float]
    best: tuple[str, float]
    since: int               # earliest year in the sample, 0 when unknown


@dataclass(frozen=True)
class TickerSeasonality:
    """Every statistic we can compute for one ticker, built once per run."""

    ticker: str
    bars: int
    first_date: str
    last_date: str
    years_of_history: float
    months: dict[int, MonthStats] = field(default_factory=dict)
    summer_half: Optional[MonthStats] = None      # May–Oct, the "sell in May" half
    winter_half: Optional[MonthStats] = None      # Nov–Apr
    santa: Optional[MonthStats] = None
    fomc_run_up: Optional[MonthStats] = None
    opex_week: Optional[MonthStats] = None
    quad_week: Optional[MonthStats] = None
    month_end: Optional[MonthStats] = None
    quarter_end: Optional[MonthStats] = None
    short_week: Optional[MonthStats] = None


@dataclass(frozen=True)
class Effect:
    """
    One seasonal effect that applies to the week being written about.

    `stat_line` is the whole point: a finished sentence, already carrying the
    ticker, the median, the hit rate, the sample size and both extremes, so the
    composer can hand it to a model as a quotable fact and the renderer can
    print it unchanged. `numbers` is the same content as machine-readable
    values, for the dashboard card and for anything that wants to sort or plot
    rather than read.
    """

    name: str
    window: str
    stat_line: str
    numbers: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class _Bar:
    date: dt.date
    close: float


# ── Parsing and small helpers ────────────────────────────────────────────

def _parse_bars(rows: Optional[Iterable[dict]]) -> list[_Bar]:
    """
    Normalise `Database.get_price_history` rows into sorted, deduplicated bars.

    Defensive rather than trusting: the table's primary key already prevents
    duplicate dates and the query already drops null closes, but these rows also
    arrive hand-built from tests and from callers that assembled them elsewhere.
    """
    by_date: dict[dt.date, float] = {}
    for row in rows or ():
        raw_date = row.get("date")
        raw_close = row.get("close")
        if raw_date is None or raw_close is None:
            continue
        try:
            day = dt.date.fromisoformat(str(raw_date)[:10])
            close = float(raw_close)
        except (TypeError, ValueError):
            continue
        if close <= 0 or not math.isfinite(close):
            continue
        by_date[day] = close
    return [_Bar(day, by_date[day]) for day in sorted(by_date)]


def _pct(start: float, end: float) -> float:
    """Percentage change between two closes."""
    return (end / start - 1.0) * 100.0


def _next_month(key: tuple[int, int]) -> tuple[int, int]:
    year, month = key
    return (year + 1, 1) if month == 12 else (year, month + 1)


def _season_label(start_year: int, end_year: int) -> str:
    """'2008' for a window inside one year, '2008/09' for one that wraps it."""
    if end_year == start_year:
        return str(start_year)
    return f"{start_year}/{str(end_year)[-2:]}"


def _summarize(
    observations: Sequence[tuple[str, float]],
    *,
    min_observations: int = MIN_OBSERVATIONS,
) -> Optional[MonthStats]:
    """Reduce labelled observations to a `MonthStats`, or None if too few."""
    pairs = [
        (str(label), float(value))
        for label, value in observations
        if value is not None and math.isfinite(float(value))
    ]
    # Two is the floor below which "worst" and "best" are the same number
    # wearing two hats, whatever the caller asked for.
    if len(pairs) < max(int(min_observations), 2):
        return None

    values = [value for _, value in pairs]
    years = [int(label[:4]) for label, _ in pairs if label[:4].isdigit()]
    return MonthStats(
        years=len(pairs),
        mean=statistics.fmean(values),
        median=statistics.median(values),
        hit_rate=sum(1 for v in values if v > 0) / len(values),
        worst=min(pairs, key=lambda p: p[1]),
        best=max(pairs, key=lambda p: p[1]),
        since=min(years) if years else 0,
    )


def _last_index_before(
    dates: Sequence[dt.date], day: dt.date, *, inclusive: bool
) -> Optional[int]:
    """
    Index of the last session on or before `day` (`inclusive`), or strictly before.

    Takes the already-sorted date list rather than the bars so the caller builds
    it once: `event_drift_stats` asks this question 87 times for FOMC history,
    and rebuilding an 8,000-entry list each time is the difference between
    instant and noticeable.
    """
    pos = bisect.bisect_right(dates, day) if inclusive else bisect.bisect_left(dates, day)
    return pos - 1 if pos > 0 else None


def _is_contiguous_session_window(
    bars: Sequence[_Bar], start: int, end: int
) -> bool:
    """Whether consecutive selected bars plausibly represent trading sessions."""
    return all(
        0 < (bars[index].date - bars[index - 1].date).days <= MAX_SESSION_GAP_DAYS
        for index in range(start + 1, end + 1)
    )


# ── Layer 1: reshaping ───────────────────────────────────────────────────

def monthly_returns(rows: Optional[Iterable[dict]]) -> dict[tuple[int, int], float]:
    """
    Month-end-to-month-end returns in percent, keyed by (year, month).

    Two deliberate exclusions:

    * **Non-consecutive months.** A gap in the stored history would otherwise
      report a three-month move as one month's return, which is how a seasonality
      table quietly becomes fiction.
    * **The trailing month.** Nothing in the bars proves the last month in the
      data finished — the series ends on the last session fetched, not on a
      month boundary — so it never enters a distribution. A month is treated as
      complete exactly when a later month exists in the data.
    """
    bars = _parse_bars(rows)
    if len(bars) < 2:
        return {}

    last_close_of_month: dict[tuple[int, int], float] = {}
    for bar in bars:
        last_close_of_month[(bar.date.year, bar.date.month)] = bar.close

    keys = sorted(last_close_of_month)
    returns: dict[tuple[int, int], float] = {}
    for previous, current in zip(keys, keys[1:]):
        if _next_month(previous) != current:
            continue
        returns[current] = _pct(
            last_close_of_month[previous], last_close_of_month[current]
        )
    returns.pop(keys[-1], None)
    return returns


# ── Layer 2: distributions ───────────────────────────────────────────────

def month_of_year_stats(
    rows: Optional[Iterable[dict]],
    month: int,
    *,
    min_observations: int = MIN_OBSERVATIONS,
) -> Optional[MonthStats]:
    """Distribution of one calendar month's returns across every year held."""
    returns = monthly_returns(rows)
    observations = [
        (str(year), value)
        for (year, mon), value in sorted(returns.items())
        if mon == month
    ]
    return _summarize(observations, min_observations=min_observations)


def window_stats(
    rows: Optional[Iterable[dict]],
    months: Sequence[int],
    *,
    min_observations: int = MIN_OBSERVATIONS,
) -> Optional[MonthStats]:
    """
    Distribution of a compounded run of calendar months — the Sell-in-May halves.

    `months` is read as a sequence that may wrap the year: the window rolls into
    the next calendar year the first time the month number stops increasing, so
    `(11, 12, 1, 2, 3, 4)` is one November-to-April season rather than six
    unrelated months. A season is only counted when every one of its months is
    present, which keeps a partial year out of the distribution.
    """
    returns = monthly_returns(rows)
    if not returns or not months:
        return None

    observations: list[tuple[str, float]] = []
    for start_year in sorted({year for year, _ in returns}):
        factor = 1.0
        year = start_year
        previous_month: Optional[int] = None
        complete = True
        for month in months:
            if previous_month is not None and month <= previous_month:
                year += 1
            previous_month = month
            value = returns.get((year, month))
            if value is None:
                complete = False
                break
            factor *= 1.0 + value / 100.0
        if complete:
            observations.append(
                (_season_label(start_year, year), (factor - 1.0) * 100.0)
            )
    return _summarize(observations, min_observations=min_observations)


def santa_rally_stats(
    rows: Optional[Iterable[dict]],
    *,
    sessions_before: int = SANTA_SESSIONS_BEFORE,
    sessions_after: int = SANTA_SESSIONS_AFTER,
    min_observations: int = MIN_OBSERVATIONS,
) -> Optional[MonthStats]:
    """
    Distribution of the Santa Claus rally window, measured session by session.

    The window is the last `sessions_before` sessions of a year plus the first
    `sessions_after` of the next, so it cannot be derived from monthly returns —
    it needs the daily bars. A year only counts when its own last bar is in
    December and the following year's first bar is in January: otherwise the
    "last five sessions of the year" are just the last five sessions we happen
    to have, which is a different statistic wearing the same name.
    """
    bars = _parse_bars(rows)
    if len(bars) < sessions_before + sessions_after + 1:
        return None

    indices_by_year: dict[int, list[int]] = {}
    for i, bar in enumerate(bars):
        indices_by_year.setdefault(bar.date.year, []).append(i)

    observations: list[tuple[str, float]] = []
    for year in sorted(indices_by_year):
        this_year = indices_by_year.get(year, [])
        next_year = indices_by_year.get(year + 1, [])
        if len(this_year) < sessions_before or len(next_year) < sessions_after:
            continue
        if bars[this_year[-1]].date.month != 12 or bars[next_year[0]].date.month != 1:
            continue
        # The window's base is the close before it opens, not the first close
        # inside it — otherwise the first session's move is silently excluded.
        base = this_year[-sessions_before] - 1
        if base < 0:
            continue
        end = next_year[sessions_after - 1]
        if not _is_contiguous_session_window(bars, base, end):
            continue
        observations.append(
            (_season_label(year, year + 1), _pct(bars[base].close, bars[end].close))
        )
    return _summarize(observations, min_observations=min_observations)


def event_drift_stats(
    rows: Optional[Iterable[dict]],
    event_dates: Iterable[str | dt.date],
    days_before: int = 1,
    *,
    include_event_day: bool = False,
    min_observations: int = MIN_OBSERVATIONS,
    min_events: int = MIN_EVENTS_IN_DATA,
) -> Optional[MonthStats]:
    """
    Distribution of the move running into a recurring scheduled event.

    The window is the `days_before` sessions ending on the last session
    **strictly before** each event date — so `days_before=1` is the single
    session before an FOMC decision, and `days_before=5` is the run-up week.
    With `include_event_day=True` the window instead ends on the event's own
    session, which is what an expiration week needs: the move that matters there
    runs *into* the Friday print, not up to the Thursday before it.

    Using "the last session at or before the date" rather than requiring an
    exact match is what makes this robust to an event landing on a holiday —
    April 2022's expiration moved to the Thursday because Good Friday was the
    third Friday, and this picks up the Thursday without a special case.

    Returns None unless at least `min_events` of the dates fall inside the bars
    we hold. Eight FOMC decisions spread over a decade of bars is a regularity;
    two is one particular year.
    """
    bars = _parse_bars(rows)
    if len(bars) <= days_before:
        return None

    parsed: list[dt.date] = []
    for raw in event_dates or ():
        if isinstance(raw, dt.date):
            parsed.append(raw)
            continue
        try:
            parsed.append(dt.date.fromisoformat(str(raw)[:10]))
        except (TypeError, ValueError):
            continue

    dates = [bar.date for bar in bars]
    first, last = dates[0], dates[-1]
    inside = sorted({day for day in parsed if first <= day <= last})
    if len(inside) < int(min_events):
        return None

    observations: list[tuple[str, float]] = []
    for day in inside:
        end = _last_index_before(dates, day, inclusive=include_event_day)
        if end is None:
            continue
        start = end - days_before
        if start < 0:
            continue
        if not _is_contiguous_session_window(bars, start, end):
            continue
        observations.append((day.isoformat(), _pct(bars[start].close, bars[end].close)))
    return _summarize(observations, min_observations=min_observations)


def short_week_stats(
    rows: Optional[Iterable[dict]],
    *,
    min_observations: int = MIN_OBSERVATIONS,
) -> Optional[MonthStats]:
    """
    Distribution of holiday-shortened weeks, detected from the bars themselves.

    No holiday calendar is consulted, and that is on purpose: the seeded NYSE
    holiday list covers two years, while the bars cover decades. A Monday-keyed
    week with two to four sessions where the neighbouring weeks have five *is*
    a shortened week, whatever caused it.

    Two exclusions keep data artefacts out: a week is only measured against the
    immediately preceding calendar week (so a gap in the history is skipped
    rather than reported as a week's move), and the final week present is never
    measured, because it is truncated by the end of the data rather than by a
    holiday.
    """
    bars = _parse_bars(rows)
    if len(bars) < 10:
        return None

    # ISO weeks start on Monday, so the Monday of a bar's week is a complete
    # week key — no ISO year/week pair needed, and no year-boundary edge case.
    weeks: dict[dt.date, list[int]] = {}
    for i, bar in enumerate(bars):
        monday = bar.date - dt.timedelta(days=bar.date.weekday())
        weeks.setdefault(monday, []).append(i)

    mondays = sorted(weeks)
    observations: list[tuple[str, float]] = []
    for previous, current in zip(mondays, mondays[1:]):
        if (current - previous).days != 7 or current == mondays[-1]:
            continue
        sessions = len(weeks[current])
        if not 2 <= sessions <= 4:
            continue
        observations.append((
            current.isoformat(),
            _pct(bars[weeks[previous][-1]].close, bars[weeks[current][-1]].close),
        ))
    return _summarize(observations, min_observations=min_observations)


def period_end_stats(
    rows: Optional[Iterable[dict]],
    *,
    sessions: int = PERIOD_END_SESSIONS,
    months: Optional[Sequence[int]] = None,
    min_observations: int = MIN_OBSERVATIONS,
) -> Optional[MonthStats]:
    """
    Distribution of the last `sessions` sessions of a month.

    `months` restricts it to the quarter-end months, which is the same window
    measured on the four dates institutional rebalancing actually targets. The
    trailing month in the data is excluded for the reason `monthly_returns`
    gives: nothing proves it finished.
    """
    bars = _parse_bars(rows)
    if len(bars) <= sessions:
        return None

    last_index_of_month: dict[tuple[int, int], int] = {}
    for i, bar in enumerate(bars):
        last_index_of_month[(bar.date.year, bar.date.month)] = i

    keys = sorted(last_index_of_month)
    observations: list[tuple[str, float]] = []
    for year, month in keys[:-1]:
        if months and month not in months:
            continue
        end = last_index_of_month[(year, month)]
        start = end - sessions
        if start < 0:
            continue
        if not _is_contiguous_session_window(bars, start, end):
            continue
        observations.append(
            (f"{year}-{month:02d}", _pct(bars[start].close, bars[end].close))
        )
    return _summarize(observations, min_observations=min_observations)


# ── Per-ticker assembly ──────────────────────────────────────────────────

def _expiration_dates(first: dt.date, last: dt.date, *, quarterly_only: bool) -> list[dt.date]:
    """Third Fridays inside a date range, generated rather than remembered."""
    months = QUARTER_END_MONTHS if quarterly_only else range(1, 13)
    dates: list[dt.date] = []
    for year in range(first.year, last.year + 1):
        for month in months:
            day = third_friday(year, month)
            if first <= day <= last:
                dates.append(day)
    return dates


def build_ticker_seasonality(
    ticker: str,
    rows: Optional[Iterable[dict]],
    *,
    min_observations: int = MIN_OBSERVATIONS,
) -> Optional[TickerSeasonality]:
    """
    Compute every statistic available for one ticker's stored bars.

    Returns None when there is nothing to measure. Individual statistics come
    back as None independently — a five-year series has real month-of-year
    numbers and no usable Sell-in-May sample, and saying so per statistic is
    what stops a thin history from being quoted as a deep one.
    """
    bars = _parse_bars(rows)
    if len(bars) < 2:
        return None

    # Re-serialised from the parsed bars so every statistic below sees the same
    # deduplicated, sorted, close-only view rather than re-validating the caller's
    # rows eleven times.
    clean = [{"date": b.date.isoformat(), "close": b.close} for b in bars]
    first, last = bars[0].date, bars[-1].date
    kwargs = {"min_observations": min_observations}

    months = {}
    for month in range(1, 13):
        stats = month_of_year_stats(clean, month, **kwargs)
        if stats is not None:
            months[month] = stats

    return TickerSeasonality(
        ticker=ticker.upper(),
        bars=len(bars),
        first_date=first.isoformat(),
        last_date=last.isoformat(),
        years_of_history=round((last - first).days / 365.25, 1),
        months=months,
        summer_half=window_stats(clean, (5, 6, 7, 8, 9, 10), **kwargs),
        winter_half=window_stats(clean, (11, 12, 1, 2, 3, 4), **kwargs),
        santa=santa_rally_stats(clean, **kwargs),
        fomc_run_up=event_drift_stats(
            clean, HISTORICAL_FOMC_DECISIONS, PRE_FOMC_SESSIONS, **kwargs
        ),
        opex_week=event_drift_stats(
            clean,
            _expiration_dates(first, last, quarterly_only=False),
            OPEX_WEEK_SESSIONS,
            include_event_day=True,
            **kwargs,
        ),
        quad_week=event_drift_stats(
            clean,
            _expiration_dates(first, last, quarterly_only=True),
            OPEX_WEEK_SESSIONS,
            include_event_day=True,
            **kwargs,
        ),
        month_end=period_end_stats(clean, **kwargs),
        quarter_end=period_end_stats(clean, months=QUARTER_END_MONTHS, **kwargs),
        short_week=short_week_stats(clean, **kwargs),
    )


# ── Layer 3: which effects apply to the coming week ──────────────────────

WEEK_DAYS = 7


def _pct_str(value: float) -> str:
    """Signed one-decimal percent, ASCII minus so a model can copy it back."""
    return f"{value:+.1f}%"


def _stat_line(ticker: str, subject: str, stats: MonthStats, *, unit: str) -> str:
    """
    Format one statistic as a self-contained, quotable sentence.

    Deliberately fewer numbers than `MonthStats` holds: median, hit rate, sample
    size and the two extremes are what make a precedent checkable, and every
    extra figure is one more thing a model can misquote. The rest travel in
    `Effect.numbers`, where nothing can paraphrase them.
    """
    since = f" since {stats.since}" if stats.since else ""
    return (
        f"{ticker} {subject}: median {_pct_str(stats.median)}, "
        f"up in {stats.hit_rate * 100:.0f}% of {stats.years} {unit}{since}; "
        f"worst {stats.worst[0]} {_pct_str(stats.worst[1])}, "
        f"best {stats.best[0]} {_pct_str(stats.best[1])}"
    )


def _numbers(stats: MonthStats) -> dict[str, float]:
    return {
        "median_pct": round(stats.median, 2),
        "mean_pct": round(stats.mean, 2),
        "hit_rate_pct": round(stats.hit_rate * 100, 1),
        "observations": stats.years,
        "since": stats.since,
        "worst_pct": round(stats.worst[1], 2),
        "best_pct": round(stats.best[1], 2),
    }


def _effect(
    name: str, window: str, ticker: str, subject: str, stats: Optional[MonthStats],
    *, unit: str,
) -> Optional[Effect]:
    """Build an Effect, or None when the statistic does not exist."""
    if stats is None:
        return None
    return Effect(
        name=name,
        window=window,
        stat_line=_stat_line(ticker, subject, stats, unit=unit),
        numbers=_numbers(stats),
    )


def _window_days(today: dt.date, days: int = WEEK_DAYS) -> list[dt.date]:
    return [today + dt.timedelta(days=i) for i in range(days)]


def _is_month_end(day: dt.date) -> bool:
    return (day + dt.timedelta(days=1)).month != day.month


def upcoming_effects(
    today: dt.date,
    stats_by_ticker: dict[str, TickerSeasonality],
    macro_next7: Optional[Sequence[dict]] = None,
    *,
    days: int = WEEK_DAYS,
) -> list[Effect]:
    """
    The seasonal effects that actually apply to the `days` starting at `today`.

    Two independent triggers, because neither alone is enough: the macro
    calendar says an expiration or a holiday is scheduled, and the date itself
    says the window crosses a month end or reaches the Santa period. An effect
    fires on either, and is then dropped anyway unless the statistic exists for
    at least one ticker — which is the rule that keeps "September is bad" out of
    the message when we hold four years of bars.

    Tickers are visited in the order `stats_by_ticker` gives them, so a caller
    that puts its benchmarks first gets SPY before its own holdings. Effects
    come back grouped by effect, one entry per ticker, which is how the renderer
    prints one heading over several lines.
    """
    window = _window_days(today, days)
    if not window:
        return []

    span = f"{window[0].isoformat()} to {window[-1].isoformat()}"
    kinds = {
        str(row.get("kind") or "").lower()
        for row in (macro_next7 or ())
        if str(row.get("date") or "")[:10] >= window[0].isoformat()
        and str(row.get("date") or "")[:10] <= window[-1].isoformat()
    }
    fomc_dates = sorted(
        str(row.get("date"))[:10]
        for row in (macro_next7 or ())
        if str(row.get("kind") or "").lower() == "fomc"
        and window[0].isoformat() <= str(row.get("date") or "")[:10] <= window[-1].isoformat()
    )

    effects: list[Effect] = []

    def add(name, win, subject, pick, *, unit):
        for ticker, stats in stats_by_ticker.items():
            effect = _effect(name, win, ticker, subject, pick(stats), unit=unit)
            if effect is not None:
                effects.append(effect)

    # ── Month of the year ────────────────────────────────────────────────
    # Always in play: the window is inside at least one month, and two when it
    # straddles a month boundary.
    for month in dict.fromkeys(day.month for day in window):
        label = MONTH_NAMES[month - 1]
        # September gets its own name because its reputation is the reason this
        # module exists — the point is to print the checked number next to it.
        name = "September effect" if month == 9 else f"{label} seasonality"
        year = next(day.year for day in window if day.month == month)
        add(name, f"{label} {year}", label,
            lambda s, m=month: s.months.get(m), unit="years")

    # ── Quarter end, else month end ─────────────────────────────────────
    # Quarter end subsumes month end: printing both would quote the same three
    # sessions twice, and the quarterly number is the more specific claim.
    quarter_end_day = next(
        (d for d in window if _is_month_end(d) and d.month in QUARTER_END_MONTHS), None
    )
    month_end_day = next((d for d in window if _is_month_end(d)), None)
    if quarter_end_day or "quarter_end" in kinds:
        when = quarter_end_day.isoformat() if quarter_end_day else span
        add("Quarter-end rebalancing", when,
            f"over the last {PERIOD_END_SESSIONS} sessions of a quarter",
            lambda s: s.quarter_end, unit="quarter-ends")
    elif month_end_day or "month_end" in kinds:
        when = month_end_day.isoformat() if month_end_day else span
        add("Month-end window", when,
            f"over the last {PERIOD_END_SESSIONS} sessions of a month",
            lambda s: s.month_end, unit="month-ends")

    # ── Expiration ──────────────────────────────────────────────────────
    expiry = next(
        (d for d in window if d == third_friday(d.year, d.month)), None
    )
    if expiry or kinds & {"opex", "quad_witching"}:
        quarterly = bool(
            "quad_witching" in kinds
            or (expiry is not None and expiry.month in QUARTER_END_MONTHS)
        )
        when = expiry.isoformat() if expiry else span
        if quarterly:
            add("Quad witching week", when,
                f"over the {OPEX_WEEK_SESSIONS} sessions into a quarterly expiration",
                lambda s: s.quad_week, unit="expirations")
        else:
            add("Monthly expiration week", when,
                f"over the {OPEX_WEEK_SESSIONS} sessions into a monthly expiration",
                lambda s: s.opex_week, unit="expirations")

    # ── Holiday-shortened week ──────────────────────────────────────────
    if kinds & {"holiday", "early_close"}:
        add("Holiday-shortened week", span, "in a holiday-shortened week",
            lambda s: s.short_week, unit="short weeks")

    # ── Pre-FOMC run-up ─────────────────────────────────────────────────
    if fomc_dates:
        add("FOMC decision week", f"FOMC {fomc_dates[0]}",
            f"over the {PRE_FOMC_SESSIONS} sessions before an FOMC decision",
            lambda s: s.fomc_run_up, unit="decisions")

    # ── Seasonal halves ─────────────────────────────────────────────────
    # Keyed to the turn itself rather than to the whole half: "sell in May" is
    # only news in the week it starts.
    if any(day.month == 5 and day.day == 1 for day in window):
        add("Sell in May", "May to October", "from May through October",
            lambda s: s.summer_half, unit="years")
    if any(day.month == 11 and day.day == 1 for day in window):
        add("Seasonally strong half", "November to April",
            "from November through April", lambda s: s.winter_half, unit="years")

    # ── Santa Claus rally ───────────────────────────────────────────────
    if any(day.month == 12 and day.day >= 20 for day in window):
        add("Santa Claus rally", span,
            f"over the last {SANTA_SESSIONS_BEFORE} sessions of the year plus "
            f"the first {SANTA_SESSIONS_AFTER}",
            lambda s: s.santa, unit="years")

    return effects


# ── Deep history ─────────────────────────────────────────────────────────

def _too_shallow(start: Optional[str], cutoff: dt.date) -> bool:
    """True when a ticker's earliest stored bar is younger than the cutoff."""
    if not start:
        return True
    try:
        return dt.date.fromisoformat(str(start)[:10]) > cutoff
    except (TypeError, ValueError):
        return True


async def ensure_deep_history(
    db,
    price_feed,
    tickers: Iterable[str],
    min_years: Optional[int] = None,
) -> dict:
    """
    Pull `history_range="max"` bars for tickers whose stored history is too thin.

    `PriceFeed.refresh_history()` stores three months, which is the right window
    for indicators and useless for seasonality. This is the opposite trade:
    expensive, rare, and only for the tickers that need it — a ticker already
    holding more than `min_years` is skipped entirely, so the monthly job
    normally fetches nothing.

    Serialised with a two-second gap rather than run concurrently. A "max" pull
    is the heaviest request we make of Yahoo, and several at once is how the
    endpoint starts throttling the quote refresh the dashboard depends on.
    Never raises: this runs from a cron job and from the startup catch-up.
    """
    horizon = int(min_years if min_years is not None else settings.seasonality_min_years)

    wanted: list[str] = []
    for raw in tickers or ():
        symbol = (raw or "").strip().upper()
        if symbol and symbol not in wanted:
            wanted.append(symbol)

    result = {"checked": len(wanted), "needed": 0, "refreshed": 0, "rows": 0, "failed": 0}
    if not wanted:
        return result

    cutoff = today_et() - dt.timedelta(days=round(365.25 * horizon))
    try:
        starts = await asyncio.to_thread(db.get_price_history_starts, wanted)
    except Exception as e:
        log.error("seasonality.history_depth_query_failed", error=str(e))
        return result

    need = [t for t in wanted if _too_shallow(starts.get(t), cutoff)]
    result["needed"] = len(need)

    for i, ticker in enumerate(need):
        if i:
            await asyncio.sleep(DEEP_HISTORY_DELAY_SECONDS)
        try:
            stored = await price_feed.refresh_history_for([ticker], history_range="max")
        except Exception as e:
            log.warning("seasonality.deep_history_failed", ticker=ticker, error=str(e))
            result["failed"] += 1
            continue
        if stored:
            result["refreshed"] += 1
            result["rows"] += int(stored)
        else:
            result["failed"] += 1

    log.info("seasonality.deep_history", min_years=horizon, **result)
    return result
