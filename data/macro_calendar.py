"""
US macro event calendar — the seeded official schedule.

Why this is checked in rather than fetched: the Finnhub economic calendar is
paywalled, and every free "economic calendar" aggregator is a scrape of the
primary sources with its own transcription errors. The primary sources are
free, stable and authoritative, so the schedule lives here as literal data read
off them once, with a monthly Tavily/LLM top-up in pipeline/macro_calendar.py
for whatever gets published after the verification date.

**Every date below was read off an official page on VERIFIED_AGAINST. None was
typed from memory.** A date that cannot be cited is a gap (see KNOWN_GAPS), not
a guess — an invented FOMC date is worse than a missing one, because the rest of
the system treats this table as ground truth.

Two kinds of row:

- **Fetched** — FOMC decisions, BLS/BEA/Census release dates, NYSE holidays and
  early closes. Literal tables, each with the page it came from in a comment.
- **Derived** — options expiration, quad witching, month/quarter end, FOMC
  minutes. Computed by the pure generators below, which is why the NYSE holiday
  sets are full-year: a March month-end rollback needs March's holidays even
  though no March 2026 row is ever emitted.

This module imports nothing from the project — same rule data/taxonomy.py
follows — so data/, pipeline/, bot/ and api/ can all use it without cycles, and
the generators are unit-testable with no database, no settings and no network.

Traps worth knowing before editing (full list in the research README):

- **September 2026's FOMC decision is the 16th, not the 17th.** The meeting runs
  Sept 15-16 and the decision lands on day two.
- **June 2027 options expiration is Thursday the 17th**, not Friday the 18th:
  the third Friday is Juneteenth observed. The same shift moves June 2027 quad
  witching. **May 2027 month end is the 28th** because Monday the 31st is
  Memorial Day. Those are the only two collisions in range.
- **There is no July half-day in either year, and no Christmas Eve half-day in
  2027.** Code that hardcodes "always a half-day before Independence Day" is
  wrong for both. The only early closes in range are 2026-11-27, 2026-12-24 and
  2027-11-26.
- **FOMC minutes dates are derived at decision + 21 days** because the Fed does
  not publish them in advance. The rule was validated against all five 2026
  minutes dates the Fed does publish, but December 2025's landed at +20 days
  (year-end pull-forward), so December is the likeliest to shift by a day.
- **2027 is thin on purpose.** BLS, BEA and Census had published no 2027
  calendar as of the verification date — confirmed by HTTP 404s, not assumed.
"""

from __future__ import annotations

import datetime as dt

# The day every fetched table below was read off its official page. Bump it only
# together with a re-read of the sources, never on its own.
VERIFIED_AGAINST = "2026-09-12"

# ── Source pages ─────────────────────────────────────────────────────────
#
# Each table below cites one of these. They are also what a refresh should
# re-read first: the LLM/Tavily path in pipeline/macro_calendar.py is a top-up
# for dates published after VERIFIED_AGAINST, not a replacement for these.

SRC_FOMC = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
SRC_CPI = "https://www.bls.gov/schedule/news_release/cpi.htm"
SRC_PPI = "https://www.bls.gov/schedule/news_release/ppi.htm"
SRC_NFP = "https://www.bls.gov/schedule/news_release/empsit.htm"
SRC_BEA = "https://www.bea.gov/news/schedule"
SRC_CENSUS = "https://www.census.gov/retail/release_schedule.html"
SRC_NYSE = "https://www.nyse.com/markets/hours-calendars"
SRC_JACKSON_HOLE = "https://www.kansascityfed.org/research/jackson-hole-economic-symposium/"

# ── Taxonomy ─────────────────────────────────────────────────────────────

# The closed set every macro_events.kind must be in. The LLM extraction path
# validates against this, and "other" is what an unrecognised but real event
# falls back to rather than being invented a new kind for.
MACRO_EVENT_KINDS: frozenset[str] = frozenset({
    "fomc", "fomc_minutes", "cpi", "ppi", "pce", "nfp", "gdp", "retail_sales",
    "fed_speech", "jackson_hole", "opex", "quad_witching", "month_end",
    "quarter_end", "holiday", "early_close", "earnings_season", "other",
})

# 3 = moves the whole tape on release; 2 = moves it often; 1 = structural or
# scheduling context. Anything not listed is 1.
IMPORTANCE_BY_KIND: dict[str, int] = {
    "fomc": 3, "cpi": 3, "nfp": 3,
    "pce": 2, "gdp": 2, "ppi": 2,
    "quad_witching": 2, "retail_sales": 2, "jackson_hole": 2,
}


def importance_for(kind: str) -> int:
    """Importance for a kind. Single source of truth for seed and web rows."""
    return IMPORTANCE_BY_KIND.get(kind, 1)


# Nothing before this date is emitted. The August 2026 CPI/PPI/NFP releases, the
# five already-published 2026 FOMC minutes dates and the seven pre-September
# 2026 NYSE holidays are all deliberately excluded as already past — but the
# holidays stay in the sets below because the derived-date generators need them.
SEED_FLOOR = dt.date(2026, 9, 1)

# ── Fetched: FOMC decisions ──────────────────────────────────────────────
# Source: SRC_FOMC. (decision_date, SEP released, meeting label).
#
# Cross-month meetings resolve to the SECOND day — the decision day. September
# 2026 is the one most easily got wrong: the meeting is Sept 15-16, so 09-16.
FOMC_DECISIONS: tuple[tuple[str, bool, str], ...] = (
    ("2026-09-16", True, "September 2026"),
    ("2026-10-28", False, "October 2026"),
    ("2026-12-09", True, "December 2026"),
    ("2027-01-27", False, "January 2027"),
    ("2027-03-17", True, "March 2027"),
    ("2027-04-28", False, "April 2027"),
    ("2027-06-09", True, "June 2027"),
    ("2027-07-28", False, "July 2027"),
    ("2027-09-15", True, "September 2027"),
    ("2027-10-27", False, "October 2027"),
    ("2027-12-08", True, "December 2027"),
)

# ── Fetched: BLS releases ────────────────────────────────────────────────
# Every date double-sourced: the per-release page AND /schedule/2026/MM_sched.htm.
# All 08:30 ET. 2027 is absent because BLS had published no 2027 calendar.
CPI_RELEASES: tuple[tuple[str, str], ...] = (
    ("2026-10-14", "September 2026"),
    ("2026-11-10", "October 2026"),
    ("2026-12-10", "November 2026"),
)

# 2026-11-13 is the one to not "correct": the November calendar first rendered
# PPI as the 12th, and both a targeted re-read and ppi.htm say the 13th.
PPI_RELEASES: tuple[tuple[str, str], ...] = (
    ("2026-10-15", "September 2026"),
    ("2026-11-13", "October 2026"),
    ("2026-12-15", "November 2026"),
)

NFP_RELEASES: tuple[tuple[str, str], ...] = (
    ("2026-10-02", "September 2026"),
    ("2026-11-06", "October 2026"),
    ("2026-12-04", "November 2026"),
)

# ── Fetched: BEA releases ────────────────────────────────────────────────
# Source: SRC_BEA. GDP and Personal Income and Outlays are CO-RELEASED, so each
# date legitimately carries both a gdp and a pce row — not a duplicate.
# (date, GDP label, PCE reference month).
BEA_RELEASES: tuple[tuple[str, str, str], ...] = (
    ("2026-09-30", "Q2 2026 (third estimate)", "August 2026"),
    ("2026-10-29", "Q3 2026 (advance estimate)", "September 2026"),
    ("2026-11-25", "Q3 2026 (second estimate)", "October 2026"),
    ("2026-12-23", "Q3 2026 (third estimate)", "November 2026"),
)

# ── Fetched: Census advance retail sales (MARTS) ─────────────────────────
# Source: SRC_CENSUS — the HTML page, NOT retail/marts/www/martsdates.pdf,
# which is a live URL serving a schedule last revised in October 2023.
RETAIL_SALES_RELEASES: tuple[tuple[str, str], ...] = (
    ("2026-09-16", "August 2026"),
    ("2026-10-15", "September 2026"),
    ("2026-11-17", "October 2026"),
    ("2026-12-16", "November 2026"),
)

# ── Fetched: NYSE holidays ───────────────────────────────────────────────
# Source: SRC_NYSE. FULL years, including months below SEED_FLOOR: the derived
# generators test every third Friday and every month end against these, so
# trimming them to the emitted window would silently produce wrong opex and
# month-end dates.
NYSE_HOLIDAYS: dict[int, tuple[tuple[str, str], ...]] = {
    2026: (
        ("2026-01-01", "New Year's Day"),
        ("2026-01-19", "Martin Luther King, Jr. Day"),
        ("2026-02-16", "Washington's Birthday"),
        ("2026-04-03", "Good Friday"),
        ("2026-05-25", "Memorial Day"),
        ("2026-06-19", "Juneteenth National Independence Day"),
        ("2026-07-03", "Independence Day (observed)"),
        ("2026-09-07", "Labor Day"),
        ("2026-11-26", "Thanksgiving Day"),
        ("2026-12-25", "Christmas Day"),
    ),
    2027: (
        ("2027-01-01", "New Year's Day"),
        ("2027-01-18", "Martin Luther King, Jr. Day"),
        ("2027-02-15", "Washington's Birthday"),
        ("2027-03-26", "Good Friday"),
        ("2027-05-31", "Memorial Day"),
        ("2027-06-18", "Juneteenth National Independence Day (observed)"),
        ("2027-07-05", "Independence Day (observed)"),
        ("2027-09-06", "Labor Day"),
        ("2027-11-25", "Thanksgiving Day"),
        ("2027-12-24", "Christmas Day (observed)"),
    ),
}

# Years the holiday tables cover. A derived generator asked for any other year
# returns unadjusted dates, so seed_rows() only walks these.
SEEDED_YEARS: tuple[int, ...] = tuple(sorted(NYSE_HOLIDAYS))

# The three closures that are NOT the usual pattern, annotated so nobody
# "fixes" them back into half-days.
_HOLIDAY_NOTES: dict[str, str] = {
    "2026-07-03": ("July 4 2026 falls on a Saturday, so this is a full closure "
                   "with NO adjacent half-day."),
    "2027-07-05": ("July 4 2027 falls on a Sunday, so this is a full closure "
                   "with NO July half-day."),
    "2027-12-24": ("Dec 25 2027 falls on a Saturday, so Dec 24 is a full "
                   "closure, NOT a Christmas Eve half-day."),
}

# ── Fetched: NYSE early closes ───────────────────────────────────────────
# Source: SRC_NYSE. These three are the complete set in range — see the July /
# Christmas Eve note above.
NYSE_EARLY_CLOSES: tuple[tuple[str, str], ...] = (
    ("2026-11-27", "day after Thanksgiving"),
    ("2026-12-24", "Christmas Eve"),
    ("2027-11-26", "day after Thanksgiving"),
)

EARLY_CLOSE_NOTE = (
    "Equities close 13:00 ET; eligible options 13:15 ET; "
    "NYSE American/Arca/National/Texas late sessions close 17:00 ET"
)

# ── Fetched: Jackson Hole ────────────────────────────────────────────────
# Source: SRC_JACKSON_HOLE. Deliberately EMPTY, and the emptiness is a verified
# finding rather than a failed fetch: the 2026 symposium ran 2026-08-27 to
# 2026-08-29 (Chair remarks Friday the 28th, 10:00 ET), which is below
# SEED_FLOOR, and the Kansas City Fed had not announced 2027 dates. The table
# stays so a refresh has somewhere to land. (year -> (start, end, theme)).
JACKSON_HOLE: dict[int, tuple[str, str, str]] = {}

# ── Fetched: historical FOMC decisions, 2015-2025 ────────────────────────
#
# Not emitted as events — these are the sample for pre-FOMC drift statistics
# (W5's seasonality module is the consumer). Only REGULARLY SCHEDULED decisions
# are included, which is why 2020 has seven: the scheduled March 17-18 2020
# meeting is marked "(cancelled)" and that month's rate actions came from
# unscheduled sessions. Also excluded: 2019-10-04 and 2025-08-22 (unscheduled
# conference calls / notation votes).
#
# Sources: fomchistorical{2015..2020}.htm for those years — the series stops at
# 2020 on a five-year publication lag, 2021.htm is a 404 — and the past-years
# sections of SRC_FOMC for 2021-2025.
_HISTORICAL_FOMC_BY_YEAR: dict[int, tuple[str, ...]] = {
    2015: ("01-28", "03-18", "04-29", "06-17", "07-29", "09-17", "10-28", "12-16"),
    2016: ("01-27", "03-16", "04-27", "06-15", "07-27", "09-21", "11-02", "12-14"),
    2017: ("02-01", "03-15", "05-03", "06-14", "07-26", "09-20", "11-01", "12-13"),
    2018: ("01-31", "03-21", "05-02", "06-13", "08-01", "09-26", "11-08", "12-19"),
    2019: ("01-30", "03-20", "05-01", "06-19", "07-31", "09-18", "10-30", "12-11"),
    2020: ("01-29", "04-29", "06-10", "07-29", "09-16", "11-05", "12-16"),
    2021: ("01-27", "03-17", "04-28", "06-16", "07-28", "09-22", "11-03", "12-15"),
    2022: ("01-26", "03-16", "05-04", "06-15", "07-27", "09-21", "11-02", "12-14"),
    2023: ("02-01", "03-22", "05-03", "06-14", "07-26", "09-20", "11-01", "12-13"),
    2024: ("01-31", "03-20", "05-01", "06-12", "07-31", "09-18", "11-07", "12-18"),
    2025: ("01-29", "03-19", "05-07", "06-18", "07-30", "09-17", "10-29", "12-10"),
}

HISTORICAL_FOMC_DECISIONS: tuple[str, ...] = tuple(sorted(
    f"{year}-{day}"
    for year, days in _HISTORICAL_FOMC_BY_YEAR.items()
    for day in days
))

# ── Known gaps ───────────────────────────────────────────────────────────
#
# Recorded rather than filled. Logged once by MacroCalendar.seed() so that
# "why does 2027 have no CPI?" is answerable from the worker log instead of
# being mistaken for a bug in the seeding.
KNOWN_GAPS: tuple[str, ...] = (
    "BLS 2027 release schedule unpublished as of 2026-09-12 "
    "(/schedule/2027/home.htm and /schedule/news_release/2027_sched.htm both 404) "
    "— no 2027 CPI/PPI/NFP dates exist",
    "BLS December 2026 reference month releases in January 2027, inside that "
    "unpublished calendar",
    "BEA 2027 release schedule unpublished — /news/schedule/full-2027 renders the "
    "page template with no rows, and the machine-readable feed's latest entry of "
    "any kind is 2026-12-23",
    "BEA Q4 2026 GDP advance estimate and December 2026 PCE fall in 2027 and are "
    "unpublished",
    "Census 2027 advance retail sales schedule unpublished — both the retail "
    "schedule page and the calendar list view stop at 2026-12-31",
    "Census December 2026 MARTS listed as 'To be announced at a later date' "
    "(the normal annual-revision pattern, not an error)",
    "Kansas City Fed Jackson Hole 2027 dates unannounced",
    "FOMC minutes dates unpublished for the 2026-09-16, 2026-10-28 and 2026-12-09 "
    "meetings and all eight 2027 meetings — derived at decision + 21 days",
    "fomchistorical2021.htm returns 404; the Fed's historical series stops at 2020 "
    "on a five-year lag, so 2021-2025 decisions came from fomccalendars.htm",
    "2020 yields 7 scheduled FOMC decisions, not 8 — the March 17-18 2020 meeting "
    "is marked '(cancelled)' and that month's actions were unscheduled",
    "census.gov/retail/marts/www/martsdates.pdf is a live URL serving a schedule "
    "last revised 2023-10-31; discarded in favour of the HTML page",
)


# ── Derived: pure generators ─────────────────────────────────────────────
#
# Pure functions of (year) over the holiday tables above. No I/O, no settings,
# no clock — so a test can pin them to a known year and assert exact dates.


def holidays(year: int) -> frozenset[dt.date]:
    """NYSE full-closure dates for `year`. Empty for years with no table."""
    return frozenset(
        dt.date.fromisoformat(iso) for iso, _ in NYSE_HOLIDAYS.get(year, ())
    )


def third_friday(year: int, month: int) -> dt.date:
    """Calendar third Friday, before any holiday adjustment."""
    first = dt.date(year, month, 1)
    return first + dt.timedelta(days=(4 - first.weekday()) % 7 + 14)


def opex(year: int) -> dict[int, dt.date]:
    """
    Options expiration date for each month of `year`, keyed by month number.

    The third Friday, rolled back to Thursday when that Friday is an NYSE
    holiday — June 2027 is the only such case in range (Juneteenth observed on
    Friday the 18th moves expiration to Thursday the 17th).

    All twelve months, including the four quarterly ones: a quad-witching date
    *is* an expiration date. seed_rows() is where the one-row-per-family rule
    suppresses the duplicate monthly row, so that de-dup decision stays in one
    place instead of being baked into the calendar arithmetic.
    """
    closed = holidays(year)
    dates: dict[int, dt.date] = {}
    for month in range(1, 13):
        day = third_friday(year, month)
        if day in closed:
            day -= dt.timedelta(days=1)
        dates[month] = day
    return dates


def quad_witching(year: int) -> dict[int, dt.date]:
    """
    Quarterly expiration of index futures, index options, stock options and
    single-stock futures — March, June, September and December expiration,
    keyed by month number. Same holiday adjustment as opex().
    """
    monthly = opex(year)
    return {month: monthly[month] for month in (3, 6, 9, 12)}


def last_trading_day(year: int, month: int) -> dt.date:
    """Last weekday of the month that is not an NYSE holiday."""
    closed = holidays(year)
    nxt = dt.date(year + 1, 1, 1) if month == 12 else dt.date(year, month + 1, 1)
    day = nxt - dt.timedelta(days=1)
    while day.weekday() > 4 or day in closed:
        day -= dt.timedelta(days=1)
    return day


def month_end(year: int) -> dict[int, dt.date]:
    """
    Last trading day of each month of `year`, keyed by month number.

    May 2027 is the only holiday-shifted case in range: the last weekday is
    Monday 2027-05-31, which is Memorial Day, so month end rolls back to Friday
    2027-05-28. All twelve months for the same reason opex() returns all
    twelve — quarter_end() is the subset, and seed_rows() suppresses the
    duplicate.
    """
    return {month: last_trading_day(year, month) for month in range(1, 13)}


def quarter_end(year: int) -> dict[int, dt.date]:
    """Last trading day of each calendar quarter, keyed by month number."""
    ends = month_end(year)
    return {month: ends[month] for month in (3, 6, 9, 12)}


# The Fed's own wording: "The minutes of regularly scheduled meetings are
# released three weeks after the date of the policy decision." It does not
# publish the dates themselves in advance, so they are computed.
FOMC_MINUTES_LAG_DAYS = 21


def fomc_minutes(decision_date: dt.date | str) -> dt.date:
    """
    Release date of the minutes for a decision taken on `decision_date`.

    Decision + 21 days, per the Fed's stated policy. Validated rather than
    assumed: all five 2026 minutes dates the Fed actually publishes equal
    decision + 21 exactly. One known deviation — December 2025's minutes came
    at +20 days, a year-end pull-forward — so a December-meeting result is the
    one worth re-checking against the page before relying on it.
    """
    if isinstance(decision_date, str):
        decision_date = dt.date.fromisoformat(decision_date)
    return decision_date + dt.timedelta(days=FOMC_MINUTES_LAG_DAYS)


# ── Seed rows ────────────────────────────────────────────────────────────

_MINUTES_NOTE = (
    "Derived: decision day {decision} + {lag} days. The Fed states 'The minutes "
    "of regularly scheduled meetings are released three weeks after the date of "
    "the policy decision' but does not publish the date. Rule validated against "
    "all five published 2026 minutes dates. Caveat: the December 2025 minutes "
    "came at +20 days (year-end pull-forward), so December is the likeliest to "
    "shift."
)

_BLS_NOTE = (
    "All times Eastern per the BLS calendar note; double-sourced against "
    "/schedule/2026/MM_sched.htm"
)

_PPI_NOV_NOTE = (
    " The November calendar first rendered this as Nov 12; a targeted re-read "
    "and ppi.htm both confirm Nov 13."
)

_RETAIL_NOTE = (
    "Advance report (MARTS). The full MRTS shares this release date but lags "
    "one reference month."
)


def _row(date: dt.date | str, time_et: str | None, name: str, kind: str,
         source_url: str, notes: str) -> dict:
    """
    One seed row, shaped exactly like the macro_events columns.

    `source_url` and the fetched/derived distinction are folded into `notes`
    because the table has no column for either — provenance belongs with the
    row, and notes is the only place it fits without widening the schema.
    """
    iso = date if isinstance(date, str) else date.isoformat()
    return {
        "date": iso,
        "time_et": time_et,
        "name": name,
        "kind": kind,
        "importance": importance_for(kind),
        "source": "seed",
        "notes": f"{notes} Source: {source_url}",
    }


def seed_rows(start: str | None = None, end: str | None = None) -> list[dict]:
    """
    Every seeded macro event in `[start, end]`, sorted by date.

    Both bounds are inclusive ISO dates and both are optional; `start` can only
    narrow the window, never widen it past SEED_FLOOR, because there is no data
    below the floor to widen into.

    Rows are `source='seed'`, which is what stops the monthly web refresh from
    overwriting them — see Database.upsert_macro_event.

    One row per date per effect family: a quarterly expiration emits
    `quad_witching` and no `opex`, a quarter end emits `quarter_end` and no
    `month_end`. Rows of DIFFERENT kinds may share a date and are kept —
    2026-09-30 legitimately carries gdp, pce and quarter_end — so uniqueness is
    on (date, kind, name), not on date.
    """
    floor = SEED_FLOOR
    if start:
        parsed_start = dt.date.fromisoformat(start)
        if parsed_start > floor:
            floor = parsed_start
    ceiling = dt.date.fromisoformat(end) if end else None

    rows: list[dict] = []

    def emit(row: dict) -> None:
        day = dt.date.fromisoformat(row["date"])
        if day < floor:
            return
        if ceiling is not None and day > ceiling:
            return
        rows.append(row)

    # FOMC decisions and their derived minutes.
    for decision, has_sep, label in FOMC_DECISIONS:
        emit(_row(
            decision, "14:00", "FOMC rate decision", "fomc", SRC_FOMC,
            ("SEP released" if has_sep else "No SEP")
            + "; statement 14:00 ET, press conference 14:30 ET.",
        ))
        emit(_row(
            fomc_minutes(decision), "14:00",
            f"FOMC minutes ({label} meeting)", "fomc_minutes", SRC_FOMC,
            _MINUTES_NOTE.format(decision=decision, lag=FOMC_MINUTES_LAG_DAYS),
        ))

    # BLS.
    for day, reference in CPI_RELEASES:
        emit(_row(day, "08:30", f"CPI ({reference})", "cpi", SRC_CPI, _BLS_NOTE))
    for day, reference in PPI_RELEASES:
        note = _BLS_NOTE + (_PPI_NOV_NOTE if day == "2026-11-13" else "")
        emit(_row(day, "08:30", f"PPI ({reference})", "ppi", SRC_PPI, note))
    for day, reference in NFP_RELEASES:
        emit(_row(day, "08:30", f"Employment Situation / NFP ({reference})",
                  "nfp", SRC_NFP, _BLS_NOTE))

    # BEA — one date, two rows, because the two reports are co-released.
    for day, gdp_label, pce_reference in BEA_RELEASES:
        emit(_row(day, "08:30", f"GDP {gdp_label}", "gdp", SRC_BEA,
                  "Co-released with Personal Income and Outlays."))
        emit(_row(day, "08:30",
                  f"Personal Income and Outlays / PCE ({pce_reference})",
                  "pce", SRC_BEA, "Co-released with GDP."))

    # Census.
    for day, reference in RETAIL_SALES_RELEASES:
        emit(_row(day, "08:30", f"Advance Retail Sales ({reference})",
                  "retail_sales", SRC_CENSUS, _RETAIL_NOTE))

    # NYSE closures. Holidays carry no time.
    for year in SEEDED_YEARS:
        for day, name in NYSE_HOLIDAYS[year]:
            emit(_row(day, None, f"NYSE holiday: {name}", "holiday", SRC_NYSE,
                      _HOLIDAY_NOTES.get(day, "US equity markets closed.")))
    for day, occasion in NYSE_EARLY_CLOSES:
        emit(_row(day, "13:00", f"NYSE early close ({occasion})",
                  "early_close", SRC_NYSE, EARLY_CLOSE_NOTE))

    # Jackson Hole, if a future refresh ever fills the table.
    for year, (start_day, end_day, theme) in sorted(JACKSON_HOLE.items()):
        emit(_row(start_day, None,
                  f"Jackson Hole Economic Policy Symposium ({year})",
                  "jackson_hole", SRC_JACKSON_HOLE,
                  f"Runs {start_day} to {end_day}. Theme: {theme}."))

    # Derived market-structure dates, timed at the 16:00 cash close.
    for year in SEEDED_YEARS:
        expirations = opex(year)
        quarterlies = quad_witching(year)
        for month, day in expirations.items():
            shifted = day != third_friday(year, month)
            note = ("The third Friday falls on an NYSE holiday, so expiration "
                    "moves back to Thursday." if shifted
                    else "Third Friday of the month.")
            if month in quarterlies:
                emit(_row(day, "16:00",
                          f"Quad witching (Q{(month - 1) // 3 + 1} {year})",
                          "quad_witching", SRC_NYSE,
                          note + " Quarterly expiration of index futures, index "
                          "options, stock options and single-stock futures. The "
                          "monthly opex row is suppressed on this date."))
            else:
                emit(_row(day, "16:00", "Monthly options expiration", "opex",
                          SRC_NYSE, note))

        ends = month_end(year)
        quarter_ends = quarter_end(year)
        for month, day in ends.items():
            naive = (dt.date(year + 1, 1, 1) if month == 12
                     else dt.date(year, month + 1, 1)) - dt.timedelta(days=1)
            while naive.weekday() > 4:
                naive -= dt.timedelta(days=1)
            note = ("Last trading day of the month." if day == naive else
                    f"The last weekday {naive.isoformat()} is an NYSE holiday, "
                    "so this rolls back to the previous trading day.")
            if month in quarter_ends:
                emit(_row(day, "16:00",
                          f"Quarter end (Q{(month - 1) // 3 + 1} {year})",
                          "quarter_end", SRC_NYSE,
                          note + " The month-end row is suppressed on this date."))
            else:
                emit(_row(day, "16:00", "Last trading day of month",
                          "month_end", SRC_NYSE, note))

    rows.sort(key=lambda r: (r["date"], r["kind"], r["name"]))
    return rows
