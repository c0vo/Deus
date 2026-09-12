"""
Tests for the macro event calendar — the seeded official schedule, the derived
date arithmetic, and the web top-up.

Network-free and LLM-free by construction. `data/macro_calendar.py` imports
nothing from the project, so the generators are pure functions over literal
tables; the service layer runs against a temporary SQLite file with
`pipeline.macro_calendar.complete` patched and a stub search provider.

The point of the seed tests is not coverage, it is that a wrong date here
silently becomes ground truth for the alerts, the weekly tip and the daily
stance. So the spot checks pin the two traps the research pass actually caught:
September 2026's FOMC decision is the 16th, and June 2027's expiration moves to
Thursday because the third Friday is Juneteenth observed.
"""

from __future__ import annotations

import datetime as dt
import typing
from collections import Counter, defaultdict
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from data.database import Database
from data.macro_calendar import (
    FOMC_DECISIONS,
    HISTORICAL_FOMC_DECISIONS,
    MACRO_EVENT_KINDS,
    SEED_FLOOR,
    VERIFIED_AGAINST,
    fomc_minutes,
    importance_for,
    month_end,
    opex,
    quad_witching,
    quarter_end,
    seed_rows,
    third_friday,
)
from pipeline.macro_calendar import (
    MacroCalendar,
    MacroEventExtract,
    MacroEventKind,
    today_et,
)
from pipeline.web_search import WebSearchResult


@pytest.fixture
def db(tmp_path):
    database = Database(db_path=str(tmp_path / "test.db"))
    database.initialize()
    return database


def _seed_row(date: str, kind: str, name: str, **extra) -> dict:
    row = {
        "date": date,
        "time_et": "08:30",
        "name": name,
        "kind": kind,
        "importance": importance_for(kind),
        "source": "seed",
        "notes": "",
    }
    row.update(extra)
    return row


# ── Seed data integrity ──────────────────────────────────────────────────


def test_every_seed_date_is_iso_parseable():
    for row in seed_rows():
        # Raises on anything that is not YYYY-MM-DD, which is the shape every
        # SQLite date comparison in the queries relies on.
        dt.date.fromisoformat(row["date"])


def test_every_seed_kind_is_in_the_enum():
    for row in seed_rows():
        assert row["kind"] in MACRO_EVENT_KINDS, row


def test_seed_keys_are_unique_on_date_kind_name():
    keys = [(r["date"], r["kind"], r["name"]) for r in seed_rows()]
    assert len(keys) == len(set(keys))


def test_seed_importance_matches_the_kind_mapping():
    for row in seed_rows():
        assert row["importance"] == importance_for(row["kind"]), row


def test_seed_rows_are_all_source_seed():
    # What stops the monthly web refresh from overwriting them.
    assert {r["source"] for r in seed_rows()} == {"seed"}


def test_no_seed_row_predates_the_floor():
    for row in seed_rows():
        assert dt.date.fromisoformat(row["date"]) >= SEED_FLOOR, row


def test_seed_times_are_well_formed_or_absent():
    for row in seed_rows():
        time_et = row["time_et"]
        if time_et is None:
            continue
        hh, _, mm = time_et.partition(":")
        assert len(time_et) == 5 and hh.isdigit() and mm.isdigit(), row
        assert 0 <= int(hh) < 24 and 0 <= int(mm) < 60, row


def test_seed_rows_are_sorted_by_date():
    dates = [r["date"] for r in seed_rows()]
    assert dates == sorted(dates)


def test_one_row_per_date_per_effect_family():
    """Quad witching supersedes opex; quarter end supersedes month end."""
    by_date: dict[str, set[str]] = defaultdict(set)
    for row in seed_rows():
        by_date[row["date"]].add(row["kind"])

    for date, kinds in by_date.items():
        assert not {"opex", "quad_witching"} <= kinds, date
        assert not {"month_end", "quarter_end"} <= kinds, date


def test_different_kinds_may_share_a_date():
    """
    BEA co-releases GDP and Personal Income and Outlays, and 2026-09-30 is also
    a quarter end. All three rows are real — suppressing them would be the bug.
    """
    kinds = {r["kind"] for r in seed_rows() if r["date"] == "2026-09-30"}
    assert {"gdp", "pce", "quarter_end"} <= kinds


def test_seed_row_notes_cite_an_official_source():
    for row in seed_rows():
        assert "Source: https://" in (row["notes"] or ""), row


def test_seed_window_bounds_are_inclusive_and_cannot_widen_past_the_floor():
    windowed = seed_rows(start="2026-10-01", end="2026-10-31")
    assert windowed, "October 2026 has seeded events"
    assert all(r["date"].startswith("2026-10") for r in windowed)

    # A start below the floor is clamped rather than honoured: there is no data
    # down there to widen into.
    assert seed_rows(start="2000-01-01") == seed_rows()


def test_verified_against_is_a_real_date():
    dt.date.fromisoformat(VERIFIED_AGAINST)


def test_today_et_is_the_eastern_date_not_the_hosts():
    """
    Every stored date is an Eastern-time date, so "today" has to be too. On the
    Asia/Seoul worker a naive date.today() is thirteen to fourteen hours ahead —
    already tomorrow for most of the US session.
    """
    import zoneinfo

    expected = dt.datetime.now(zoneinfo.ZoneInfo("America/New_York")).date()
    assert today_et() == expected
    assert abs((today_et() - dt.datetime.now(dt.timezone.utc).date()).days) <= 1


# ── Fetched dates that are easy to get wrong ─────────────────────────────


def test_september_2026_fomc_decision_is_the_sixteenth():
    """The meeting runs Sept 15-16; the decision is day two. Not the 17th."""
    decisions = {d for d, _, _ in FOMC_DECISIONS}
    assert "2026-09-16" in decisions
    assert "2026-09-17" not in decisions


def test_historical_fomc_decisions_cover_2015_to_2025():
    years = Counter(d[:4] for d in HISTORICAL_FOMC_DECISIONS)
    assert set(years) == {str(y) for y in range(2015, 2026)}
    assert len(HISTORICAL_FOMC_DECISIONS) == len(set(HISTORICAL_FOMC_DECISIONS))
    # 2020 has seven, not eight: the scheduled March 17-18 2020 meeting is marked
    # cancelled on the Fed's page and that month's actions were unscheduled.
    assert years["2020"] == 7
    assert all(years[str(y)] == 8 for y in range(2015, 2026) if y != 2020)


def test_no_july_or_christmas_eve_half_day_in_2027():
    """
    Both "always a half-day before Independence Day / Christmas" assumptions are
    wrong in range. The only early closes are these three.
    """
    early = {r["date"] for r in seed_rows() if r["kind"] == "early_close"}
    assert early == {"2026-11-27", "2026-12-24", "2027-11-26"}


# ── Derived date arithmetic ──────────────────────────────────────────────


def test_opex_2026_spot_checks():
    expirations = opex(2026)
    assert expirations[9] == dt.date(2026, 9, 18)
    assert expirations[12] == dt.date(2026, 12, 18)
    # June 2026 is the year's one shift, and it is below SEED_FLOOR so it never
    # reaches the table — which is exactly why the holiday sets are kept
    # full-year rather than trimmed to the emitted window. Juneteenth 2026 falls
    # on Friday the 19th, the third Friday, so expiration is Thursday the 18th.
    assert third_friday(2026, 6) == dt.date(2026, 6, 19)
    assert expirations[6] == dt.date(2026, 6, 18)
    assert [
        month for month in range(1, 13)
        if expirations[month] != third_friday(2026, month)
    ] == [6]


def test_opex_2027_moves_june_back_for_juneteenth():
    expirations = opex(2027)
    assert third_friday(2027, 6) == dt.date(2027, 6, 18)  # Juneteenth observed
    assert expirations[6] == dt.date(2027, 6, 17)  # so expiration is Thursday
    assert expirations[3] == dt.date(2027, 3, 19)
    # June is the only shift in 2027.
    assert [
        month for month in range(1, 13)
        if expirations[month] != third_friday(2027, month)
    ] == [6]


def test_quad_witching_is_the_quarterly_subset_of_opex():
    for year in (2026, 2027):
        quarterly = quad_witching(year)
        assert set(quarterly) == {3, 6, 9, 12}
        assert all(quarterly[m] == opex(year)[m] for m in quarterly)
    # And inherits the Juneteenth shift.
    assert quad_witching(2027)[6] == dt.date(2027, 6, 17)


def test_may_2027_month_end_rolls_back_for_memorial_day():
    ends = month_end(2027)
    assert ends[5] == dt.date(2027, 5, 28)  # Monday the 31st is Memorial Day
    assert ends[3] == dt.date(2027, 3, 31)


def test_month_end_is_never_a_weekend():
    for year in (2026, 2027):
        for day in month_end(year).values():
            assert day.weekday() <= 4, day


def test_quarter_end_is_the_quarterly_subset_of_month_end():
    for year in (2026, 2027):
        quarterly = quarter_end(year)
        assert set(quarterly) == {3, 6, 9, 12}
        assert all(quarterly[m] == month_end(year)[m] for m in quarterly)


def test_fomc_minutes_is_decision_plus_21_days():
    assert fomc_minutes("2026-09-16") == dt.date(2026, 10, 7)
    assert fomc_minutes(dt.date(2026, 12, 9)) == dt.date(2026, 12, 30)


def test_every_decision_has_a_minutes_row():
    rows = seed_rows()
    decisions = {r["date"] for r in rows if r["kind"] == "fomc"}
    minutes = {r["date"] for r in rows if r["kind"] == "fomc_minutes"}
    for decision in decisions:
        assert fomc_minutes(decision).isoformat() in minutes


# ── Schema / enum agreement ──────────────────────────────────────────────


def test_extraction_literal_matches_the_kind_enum():
    """
    The Literal has to be written out for the json_schema, so this is what keeps
    it from drifting away from MACRO_EVENT_KINDS.
    """
    assert set(typing.get_args(MacroEventKind)) == MACRO_EVENT_KINDS


def test_extraction_schema_rejects_an_unknown_kind():
    with pytest.raises(Exception):
        MacroEventExtract(date="2027-01-13", name="Something", kind="not_a_kind")


# ── seed() against a real database ───────────────────────────────────────


def _count(db: Database) -> int:
    with db.connection() as conn:
        return conn.execute("SELECT COUNT(*) AS c FROM macro_events").fetchone()["c"]


def test_seed_inserts_every_row(db):
    counts = MacroCalendar(db).seed()
    assert counts["inserted"] == counts["rows"] == len(seed_rows())
    assert counts["skipped"] == 0
    assert _count(db) == len(seed_rows())


def test_seed_is_idempotent(db):
    calendar = MacroCalendar(db)
    calendar.seed()
    first = _count(db)

    second_counts = calendar.seed()

    assert _count(db) == first
    assert second_counts["inserted"] == 0
    assert second_counts["updated"] == second_counts["rows"]


def test_seed_repairs_a_corrected_date_without_duplicating(db):
    """A re-seed after a schedule correction updates in place, on the unique key."""
    calendar = MacroCalendar(db)
    calendar.seed()
    with db.connection() as conn:
        conn.execute(
            "UPDATE macro_events SET importance = 1, notes = 'stale' WHERE kind = 'cpi'"
        )

    calendar.seed()

    rows = [r for r in db.get_macro_events("2026-01-01", "2028-01-01")
            if r["kind"] == "cpi"]
    assert rows
    assert all(r["importance"] == 3 for r in rows)
    assert all(r["notes"] != "stale" for r in rows)


# ── upsert_macro_event authority rules ───────────────────────────────────


def test_web_row_cannot_override_a_seed_row_for_the_same_date_and_kind(db):
    db.upsert_macro_event(_seed_row("2027-02-10", "cpi", "CPI (January 2027)"))

    # Different name on purpose: this is the realistic case, and inserting it
    # would put two CPI rows on one day rather than overwriting anything.
    status = db.upsert_macro_event(
        _seed_row("2027-02-10", "cpi", "CPI report", source="web")
    )

    assert status == "skipped"
    rows = db.get_macro_events_on("2027-02-10")
    assert len(rows) == 1
    assert rows[0]["source"] == "seed"
    assert rows[0]["name"] == "CPI (January 2027)"


def test_web_row_overrides_a_seed_row_when_explicitly_allowed(db):
    db.upsert_macro_event(_seed_row("2027-02-10", "cpi", "CPI (January 2027)"))

    status = db.upsert_macro_event(
        _seed_row("2027-02-10", "cpi", "CPI report", source="web"),
        allow_override_seed=True,
    )

    assert status == "inserted"
    assert len(db.get_macro_events_on("2027-02-10")) == 2


def test_web_row_lands_where_no_authoritative_row_exists(db):
    status = db.upsert_macro_event(
        _seed_row("2027-03-10", "cpi", "CPI (February 2027)", source="web")
    )
    assert status == "inserted"

    # And a second pass updates its own row rather than duplicating it.
    assert db.upsert_macro_event(
        _seed_row("2027-03-10", "cpi", "CPI (February 2027)", source="web")
    ) == "updated"
    assert len(db.get_macro_events_on("2027-03-10")) == 1


def test_seed_row_outranks_an_existing_web_row(db):
    db.upsert_macro_event(_seed_row("2027-04-13", "cpi", "CPI", source="web"))

    assert db.upsert_macro_event(
        _seed_row("2027-04-13", "cpi", "CPI", source="seed")
    ) == "updated"
    assert db.get_macro_events_on("2027-04-13")[0]["source"] == "seed"


def test_manual_rows_are_authoritative_too(db):
    db.upsert_macro_event(_seed_row("2027-05-12", "fomc", "FOMC", source="manual"))
    assert db.upsert_macro_event(
        _seed_row("2027-05-12", "fomc", "FOMC decision", source="web")
    ) == "skipped"


def test_upsert_sets_updated_at_on_an_update(db):
    db.upsert_macro_event(_seed_row("2027-06-09", "fomc", "FOMC"))
    with db.connection() as conn:
        conn.execute("UPDATE macro_events SET updated_at = '2000-01-01 00:00:00'")

    db.upsert_macro_event(_seed_row("2027-06-09", "fomc", "FOMC", notes="fresh"))

    row = db.get_macro_events_on("2027-06-09")[0]
    assert row["updated_at"] != "2000-01-01 00:00:00"
    assert row["notes"] == "fresh"


# ── Window queries ───────────────────────────────────────────────────────


def test_get_macro_events_window_is_inclusive_on_both_ends(db):
    for day in ("2027-01-10", "2027-01-15", "2027-01-20", "2027-01-25"):
        db.upsert_macro_event(_seed_row(day, "cpi", f"CPI {day}"))

    rows = db.get_macro_events("2027-01-15", "2027-01-20")

    assert [r["date"] for r in rows] == ["2027-01-15", "2027-01-20"]


def test_get_macro_events_orders_importance_first_within_a_day(db):
    db.upsert_macro_event(_seed_row("2027-01-15", "month_end", "Month end"))
    db.upsert_macro_event(_seed_row("2027-01-15", "cpi", "CPI (December 2026)"))
    db.upsert_macro_event(_seed_row("2027-01-15", "gdp", "GDP Q4 2026"))

    # So a caller that truncates drops the month-end marker, not the CPI print.
    assert [r["kind"] for r in db.get_macro_events("2027-01-15", "2027-01-15")] == [
        "cpi", "gdp", "month_end",
    ]


def test_get_macro_events_on_returns_only_that_day(db):
    db.upsert_macro_event(_seed_row("2027-01-15", "cpi", "CPI"))
    db.upsert_macro_event(_seed_row("2027-01-16", "ppi", "PPI"))

    rows = db.get_macro_events_on("2027-01-15")

    assert len(rows) == 1 and rows[0]["kind"] == "cpi"


def test_upcoming_window_excludes_the_past_and_beyond_the_horizon(db):
    today = today_et()
    db.upsert_macro_event(_seed_row((today - dt.timedelta(days=1)).isoformat(),
                                    "cpi", "Yesterday"))
    db.upsert_macro_event(_seed_row(today.isoformat(), "cpi", "Today"))
    db.upsert_macro_event(_seed_row((today + dt.timedelta(days=5)).isoformat(),
                                    "ppi", "In five days"))
    db.upsert_macro_event(_seed_row((today + dt.timedelta(days=40)).isoformat(),
                                    "gdp", "In forty days"))

    names = [r["name"] for r in MacroCalendar(db).upcoming(days=14)]

    assert names == ["Today", "In five days"]


# ── context_lines ────────────────────────────────────────────────────────


def test_context_lines_wording_for_today_and_tomorrow(db):
    anchor = dt.date(2026, 9, 16)
    db.upsert_macro_event(_seed_row("2026-09-16", "cpi", "CPI (August 2026)"))
    db.upsert_macro_event(_seed_row("2026-09-17", "fomc", "FOMC rate decision",
                                    time_et="14:00"))

    lines = MacroCalendar(db).context_lines(anchor)

    assert lines == [
        "CPI (August 2026) today 08:30 ET",
        "FOMC rate decision tomorrow 14:00 ET",
    ]


def test_context_lines_omits_the_time_when_there_is_none(db):
    db.upsert_macro_event(_seed_row("2026-11-26", "holiday",
                                    "NYSE holiday: Thanksgiving Day",
                                    time_et=None))

    assert MacroCalendar(db).context_lines("2026-11-26") == [
        "NYSE holiday: Thanksgiving Day today"
    ]


def test_context_lines_stops_at_tomorrow(db):
    db.upsert_macro_event(_seed_row("2026-09-18", "cpi", "Two days out"))

    assert MacroCalendar(db).context_lines(dt.date(2026, 9, 16)) == []


def test_context_lines_defaults_to_today_and_survives_a_bad_date(db):
    today = today_et()
    db.upsert_macro_event(_seed_row(today.isoformat(), "cpi", "CPI now"))

    assert MacroCalendar(db).context_lines() == ["CPI now today 08:30 ET"]
    assert MacroCalendar(db).context_lines("not-a-date") == []


# ── refresh_from_web ─────────────────────────────────────────────────────


class _StubProvider:
    """Stands in for Tavily. Records the queries it was asked."""

    def __init__(self, hits: list[WebSearchResult]) -> None:
        self.hits = hits
        self.queries: list[str] = []

    async def search(self, query: str, max_results: int = 5):
        self.queries.append(query)
        # Only the first query returns hits, so the extraction prompt is built
        # once from a known set rather than five copies of it.
        return self.hits if len(self.queries) == 1 else []


def _hit(title: str = "FOMC calendar") -> WebSearchResult:
    return WebSearchResult(
        title=title,
        url="https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
        content="Schedule text.",
        source="federalreserve.gov",
    )


def _llm_response(extracts: list[MacroEventExtract]):
    # Shaped like an LLMResponse as far as track_llm and the caller look at it:
    # `.parsed` for the happy path, `.text` for the parse_structured fallback.
    return SimpleNamespace(parsed=extracts, text="", usage=None, cost=None)


@pytest.fixture
def web_ready(monkeypatch):
    """Make the refresh path believe Tavily and the model are configured."""
    monkeypatch.setattr("pipeline.macro_calendar.is_llm_configured", lambda: True)
    monkeypatch.setattr("pipeline.macro_calendar.settings.model_extract",
                        "test/extract-model", raising=False)


async def test_refresh_returns_zeros_without_a_search_provider(db, monkeypatch):
    monkeypatch.setattr("pipeline.macro_calendar.create_search_provider",
                        lambda: None)

    counts = await MacroCalendar(db).refresh_from_web()

    assert counts["queries"] == 0 and counts["inserted"] == 0
    assert _count(db) == 0


async def test_refresh_returns_zeros_when_the_model_is_unset(db, monkeypatch):
    monkeypatch.setattr("pipeline.macro_calendar.create_search_provider",
                        lambda: _StubProvider([_hit()]))
    monkeypatch.setattr("pipeline.macro_calendar.settings.model_extract", "",
                        raising=False)

    counts = await MacroCalendar(db).refresh_from_web()

    assert counts == {"queries": 0, "results": 0, "extracted": 0, "rejected": 0,
                      "inserted": 0, "updated": 0, "skipped": 0}


async def test_refresh_inserts_a_valid_extracted_event(db, monkeypatch, web_ready):
    provider = _StubProvider([_hit()])
    monkeypatch.setattr("pipeline.macro_calendar.create_search_provider",
                        lambda: provider)
    target = (today_et() + dt.timedelta(days=30)).isoformat()
    extracts = [MacroEventExtract(date=target, name="CPI (next month)",
                                  kind="cpi", time_et="08:30", importance=1)]

    with patch("pipeline.macro_calendar.complete",
               AsyncMock(return_value=_llm_response(extracts))) as mock:
        counts = await MacroCalendar(db).refresh_from_web()

    assert mock.await_count == 1
    assert counts["inserted"] == 1 and counts["rejected"] == 0
    row = db.get_macro_events_on(target)[0]
    assert row["source"] == "web"
    # Importance comes from the kind, not from the model's self-rating, so the
    # dashboard's dots agree with the seed rows for the same release.
    assert row["importance"] == 3
    assert any("FOMC meeting schedule" in q for q in provider.queries)
    assert any("economic calendar" in q for q in provider.queries)


async def test_refresh_drops_bad_dates(db, monkeypatch, web_ready):
    monkeypatch.setattr("pipeline.macro_calendar.create_search_provider",
                        lambda: _StubProvider([_hit()]))
    today = today_et()
    extracts = [
        # Already past.
        MacroEventExtract(date=(today - dt.timedelta(days=10)).isoformat(),
                          name="Old CPI", kind="cpi"),
        # Beyond any published schedule — a model inventing a year.
        MacroEventExtract(date=(today + dt.timedelta(days=900)).isoformat(),
                          name="Far CPI", kind="cpi"),
        # Not a date at all.
        MacroEventExtract(date="to be announced", name="Vague CPI", kind="cpi"),
        # No name to render.
        MacroEventExtract(date=(today + dt.timedelta(days=20)).isoformat(),
                          name="   ", kind="cpi"),
    ]

    with patch("pipeline.macro_calendar.complete",
               AsyncMock(return_value=_llm_response(extracts))):
        counts = await MacroCalendar(db).refresh_from_web()

    assert counts["extracted"] == 4
    assert counts["rejected"] == 4
    assert counts["inserted"] == 0
    assert _count(db) == 0


async def test_refresh_drops_a_kind_outside_the_enum(db, monkeypatch, web_ready):
    monkeypatch.setattr("pipeline.macro_calendar.create_search_provider",
                        lambda: _StubProvider([_hit()]))
    # model_construct bypasses validation, which is how a bad kind could still
    # reach _validate: `strict` is off on the json_schema, so the request only
    # guides decoding.
    rogue = MacroEventExtract.model_construct(
        date=(today_et() + dt.timedelta(days=20)).isoformat(),
        name="Mystery release", kind="bank_holiday", time_et=None, importance=1,
    )

    with patch("pipeline.macro_calendar.complete",
               AsyncMock(return_value=_llm_response([rogue]))):
        counts = await MacroCalendar(db).refresh_from_web()

    assert counts["rejected"] == 1 and counts["inserted"] == 0


async def test_refresh_never_overrides_a_seed_row(db, monkeypatch, web_ready):
    monkeypatch.setattr("pipeline.macro_calendar.create_search_provider",
                        lambda: _StubProvider([_hit()]))
    target = (today_et() + dt.timedelta(days=25)).isoformat()
    db.upsert_macro_event(_seed_row(target, "cpi", "CPI (official)"))
    extracts = [MacroEventExtract(date=target, name="CPI report", kind="cpi")]

    with patch("pipeline.macro_calendar.complete",
               AsyncMock(return_value=_llm_response(extracts))):
        counts = await MacroCalendar(db).refresh_from_web()

    assert counts["skipped"] == 1 and counts["inserted"] == 0
    rows = db.get_macro_events_on(target)
    assert len(rows) == 1 and rows[0]["name"] == "CPI (official)"


async def test_refresh_survives_a_failing_extraction_call(db, monkeypatch, web_ready):
    monkeypatch.setattr("pipeline.macro_calendar.create_search_provider",
                        lambda: _StubProvider([_hit()]))

    with patch("pipeline.macro_calendar.complete",
               AsyncMock(side_effect=RuntimeError("upstream 500"))):
        counts = await MacroCalendar(db).refresh_from_web()

    # Never raises out of a scheduled job, and reports the stage it reached.
    assert counts["results"] == 1
    assert counts["extracted"] == 0 and counts["inserted"] == 0


async def test_refresh_survives_a_failing_search(db, monkeypatch, web_ready):
    class _Exploding:
        async def search(self, query, max_results=5):
            raise RuntimeError("tavily down")

    monkeypatch.setattr("pipeline.macro_calendar.create_search_provider",
                        lambda: _Exploding())

    counts = await MacroCalendar(db).refresh_from_web()

    assert counts["queries"] > 0 and counts["results"] == 0
    assert counts["inserted"] == 0


async def test_refresh_normalizes_a_malformed_time(db, monkeypatch, web_ready):
    monkeypatch.setattr("pipeline.macro_calendar.create_search_provider",
                        lambda: _StubProvider([_hit()]))
    target = (today_et() + dt.timedelta(days=15)).isoformat()
    extracts = [MacroEventExtract(date=target, name="Jobs report", kind="nfp",
                                  time_et="8:30 a.m. ET")]

    with patch("pipeline.macro_calendar.complete",
               AsyncMock(return_value=_llm_response(extracts))):
        await MacroCalendar(db).refresh_from_web()

    assert db.get_macro_events_on(target)[0]["time_et"] is None
