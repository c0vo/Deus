"""
Tests for the events calendar: Finnhub payload mapping and date-range queries.

Network-free by construction — the Finnhub helpers under test are pure
functions over a response dict, and the range queries run against a temporary
SQLite file. Nothing here touches a live API or an LLM.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta

import pytest

from data.database import Database
from pipeline.event_tracker import EventTracker, EARNINGS_HOUR_LABELS
from pipeline.ipo_detector import IPODetector, FINNHUB_IPO_STATUS


@pytest.fixture
def db(tmp_path):
    database = Database(db_path=str(tmp_path / "test.db"))
    database.initialize()
    return database


def _iso(days_from_now: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=days_from_now)).strftime("%Y-%m-%d")


def _insert_event(db: Database, ticker: str, event_date: str, event_type: str = "earnings"):
    with db.connection() as conn:
        conn.execute(
            """
            INSERT INTO ticker_events (ticker, event_type, event_date, event_title, confidence, source)
            VALUES (?, ?, ?, ?, 'confirmed', 'finnhub')
            """,
            (ticker, event_type, event_date, f"{ticker} test event"),
        )


# ── Earnings titles and session labels ───────────────────────────────────────

def test_earnings_title_uses_period_when_available():
    entry = {"symbol": "IREN", "date": "2026-08-26", "quarter": 4, "year": 2027, "hour": "amc"}
    assert EventTracker._earnings_title("IREN", entry) == "IREN Q4 FY2027 earnings"


def test_earnings_title_falls_back_without_period():
    assert EventTracker._earnings_title("MU", {"date": "2026-09-21"}) == "MU earnings"
    assert EventTracker._earnings_title("MU", {"quarter": 3, "year": None}) == "MU earnings"


def test_earnings_hour_labels_cover_finnhub_vocabulary():
    assert EARNINGS_HOUR_LABELS["amc"] == "After market close"
    assert EARNINGS_HOUR_LABELS["bmo"] == "Before market open"
    # An unknown or absent hour must degrade to an empty note, not a KeyError.
    assert EARNINGS_HOUR_LABELS.get("", "") == ""
    assert EARNINGS_HOUR_LABELS.get("unexpected", "") == ""


# ── Finnhub IPO mapping ──────────────────────────────────────────────────────

def test_finnhub_ipo_maps_expected_to_upcoming():
    record = IPODetector._finnhub_to_record({
        "symbol": "APMD", "name": "Apnimed, Inc.", "date": "2026-07-31",
        "price": "14.00-16.00", "numberOfShares": 10_000_000,
        "totalSharesValue": 184_000_000, "exchange": "NASDAQ Global",
        "status": "expected",
    })
    assert record["company_name"] == "Apnimed, Inc."
    assert record["ticker"] == "APMD"
    assert record["status"] == "upcoming"
    # Price stays a string: it can be a range, and the frontend types it so.
    assert record["expected_price"] == "14.00-16.00"
    assert record["source"] == "finnhub"
    assert "NASDAQ Global" in record["notes"]
    assert json.loads(record["metadata_json"])["source"] == "finnhub"


def test_finnhub_ipo_skips_filed_rows():
    """A `filed` row's date is the S-1 filing date, not a listing date."""
    assert IPODetector._finnhub_to_record({
        "name": "Somebody Inc", "date": "2026-06-01", "status": "filed",
    }) is None
    assert "filed" not in FINNHUB_IPO_STATUS


def test_finnhub_ipo_requires_a_company_name():
    assert IPODetector._finnhub_to_record({"name": "", "status": "expected"}) is None
    assert IPODetector._finnhub_to_record({"status": "expected"}) is None


def test_finnhub_ipo_tolerates_missing_optional_fields():
    record = IPODetector._finnhub_to_record({
        "name": "Minimal Co", "date": "2026-08-15", "status": "priced",
    })
    assert record["ticker"] is None
    assert record["expected_price"] is None
    assert record["notes"] == ""
    # Finnhub supplies neither, so the LLM path can still contribute them.
    assert record["sector"] is None
    assert record["estimated_valuation"] is None


# ── Range and upcoming queries ───────────────────────────────────────────────

def test_upcoming_events_exclude_the_past(db):
    _insert_event(db, "PAST", _iso(-10))
    _insert_event(db, "SOON", _iso(3))

    tickers = [e["ticker"] for e in EventTracker(db).get_all_upcoming_events(days_ahead=30)]
    assert tickers == ["SOON"]


def test_upcoming_events_exclude_empty_dates(db):
    """
    event_date is DATE NOT NULL but SQLite stores '' happily, and '' sorts below
    every real date — so an undated row satisfied `event_date <= end` and leaked
    into every calendar. The `>= today` bound is what excludes it.
    """
    with db.connection() as conn:
        conn.execute(
            "INSERT INTO ticker_events (ticker, event_type, event_date, event_title) "
            "VALUES ('BAD', 'earnings', '', 'undated')"
        )
    _insert_event(db, "GOOD", _iso(5))

    tickers = [e["ticker"] for e in EventTracker(db).get_all_upcoming_events(days_ahead=30)]
    assert tickers == ["GOOD"]


def test_upcoming_events_respect_the_horizon(db):
    _insert_event(db, "NEAR", _iso(5))
    _insert_event(db, "FAR", _iso(60))

    tracker = EventTracker(db)
    assert [e["ticker"] for e in tracker.get_all_upcoming_events(days_ahead=14)] == ["NEAR"]
    assert {e["ticker"] for e in tracker.get_all_upcoming_events(days_ahead=90)} == {"NEAR", "FAR"}


def test_ticker_events_exclude_the_past(db):
    _insert_event(db, "AAPL", _iso(-5))
    _insert_event(db, "AAPL", _iso(7))

    events = EventTracker(db).get_ticker_events("AAPL", days_ahead=30)
    assert len(events) == 1
    assert events[0]["event_date"] == _iso(7)


def test_events_in_range_includes_the_past(db):
    """The month grid navigates backwards, so this query must NOT clamp to today."""
    _insert_event(db, "PAST", _iso(-10))
    _insert_event(db, "SOON", _iso(3))

    events = EventTracker(db).get_events_in_range(_iso(-30), _iso(30))
    assert {e["ticker"] for e in events} == {"PAST", "SOON"}


def test_events_in_range_is_bounded_and_ordered(db):
    _insert_event(db, "OUT_LOW", _iso(-40))
    _insert_event(db, "B_IN", _iso(10))
    _insert_event(db, "A_IN", _iso(2))
    _insert_event(db, "OUT_HIGH", _iso(40))

    events = EventTracker(db).get_events_in_range(_iso(-30), _iso(30))
    assert [e["ticker"] for e in events] == ["A_IN", "B_IN"]


def test_events_in_range_excludes_empty_dates(db):
    with db.connection() as conn:
        conn.execute(
            "INSERT INTO ticker_events (ticker, event_type, event_date, event_title) "
            "VALUES ('BAD', 'earnings', '', 'undated')"
        )
    _insert_event(db, "GOOD", _iso(1))

    events = EventTracker(db).get_events_in_range(_iso(-30), _iso(30))
    assert [e["ticker"] for e in events] == ["GOOD"]


def test_initialize_purges_undated_rows(tmp_path):
    """The one-time migration cleans rows written before the no-date guard."""
    path = str(tmp_path / "migrate.db")
    db = Database(db_path=path)
    db.initialize()
    with db.connection() as conn:
        conn.execute(
            "INSERT INTO ticker_events (ticker, event_type, event_date, event_title) "
            "VALUES ('LEGACY', 'earnings', '', 'undated')"
        )

    Database(db_path=path).initialize()

    with db.connection() as conn:
        remaining = conn.execute("SELECT COUNT(*) FROM ticker_events").fetchone()[0]
    assert remaining == 0
