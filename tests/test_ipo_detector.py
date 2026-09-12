"""
Tests for the IPO watchlist: date filtering, ordering, retirement and storage.

Network-free and LLM-free by construction — everything under test is either a
pure string function (``normalize_date``) or SQL over a temporary SQLite file.
``_store_ipo``, ``retire_stale`` and ``get_ipo_watchlist`` are all synchronous,
so none of this needs an event loop either.

The bug these cover: an undated IPO was stored with ``ipo_date = ''``, which is
neither NULL nor comparable to a real date. It therefore failed the watchlist's
``ipo_date >= cutoff`` filter while satisfying ``retire_stale``'s
``ipo_date < backdate_cutoff`` delete — so undated offerings vanished instead of
rendering as TBA, while offerings whose date had already passed kept showing
because the cutoff was 90 days wide and the sort put status ahead of date.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from config.settings import settings
from data.database import Database
from pipeline.date_utils import normalize_date
from pipeline.ipo_detector import IPODetector


@pytest.fixture
def db(tmp_path):
    database = Database(db_path=str(tmp_path / "test.db"))
    database.initialize()
    return database


@pytest.fixture
def detector(db):
    return IPODetector(db)


def _iso(days_from_now: int) -> str:
    """A UTC calendar date offset from today, matching SQLite's date('now')."""
    return (
        datetime.now(timezone.utc) + timedelta(days=days_from_now)
    ).strftime("%Y-%m-%d")


def _insert_ipo(
    db: Database,
    company_name: str,
    ipo_date,
    status: str,
    *,
    detected_at: str | None = None,
    metadata_json: str = "{}",
) -> int:
    """Insert one ipo_tracker row verbatim, bypassing _store_ipo's coercion.

    ``ipo_date`` is written exactly as given so a test can seed the legacy ''
    shape that the migration and ``retire_stale`` are meant to clean up.
    """
    with db.connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO ipo_tracker
                (company_name, ticker, status, ipo_date, detected_at, metadata_json)
            VALUES (?, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP), ?)
            """,
            (company_name, None, status, ipo_date, detected_at, metadata_json),
        )
        return cursor.lastrowid


@pytest.fixture
def seeded(db):
    """The six-row fixture every watchlist assertion below reads.

    Covers each combination that used to be mishandled: a date in the past, a
    date in the future, '' and NULL for "not announced yet", and a listing
    inside and outside the one-day grace window.
    """
    ids = {
        "past_upcoming": _insert_ipo(db, "Past Upcoming Co", _iso(-10), "upcoming"),
        "future_priced": _insert_ipo(db, "Future Priced Co", _iso(7), "priced"),
        # detected_at pinned so the TBA pair has a deterministic final tiebreak.
        "empty_rumored": _insert_ipo(
            db, "Empty Dated Rumor Co", "", "rumored",
            detected_at="2026-01-01 00:00:00",
        ),
        "null_upcoming": _insert_ipo(
            db, "Null Dated Upcoming Co", None, "upcoming",
            detected_at="2026-01-02 00:00:00",
        ),
        "listed_yesterday": _insert_ipo(db, "Listed Yesterday Co", _iso(-1), "listed"),
        "listed_5d": _insert_ipo(db, "Listed Five Days Ago Co", _iso(-5), "listed"),
    }
    return ids


def _names(rows) -> list[str]:
    return [r["company_name"] for r in rows]


# ── normalize_date sentinels ─────────────────────────────────────────────────

@pytest.mark.parametrize(
    "raw",
    ["TBD", "tbd", "TBA", "tba", "N/A", "n/a", "NA", "na",
     "Unknown", "unknown", "UNKNOWN", "null", "None", "none", "  TBD  ", ""],
)
def test_normalize_date_returns_empty_for_no_date_sentinels(raw):
    """A placeholder means "no date known" and must not reach dateparser.

    dateparser reads "unknown" as today, which would stamp a fabricated listing
    date onto an offering whose date has not been announced.
    """
    assert normalize_date(raw) == ""


def test_normalize_date_still_parses_real_dates():
    assert normalize_date("2026-12-01") == "2026-12-01"
    assert normalize_date("December 1, 2026") == "2026-12-01"


# ── get_ipo_watchlist filtering ──────────────────────────────────────────────

def test_watchlist_excludes_past_dated_rows(detector, seeded):
    names = _names(detector.get_ipo_watchlist())
    assert "Past Upcoming Co" not in names
    assert "Listed Five Days Ago Co" not in names


def test_watchlist_keeps_listing_inside_the_grace_window(detector, seeded):
    """ipo_show_listed_days=1 means yesterday's listing still reads as ✅ listed."""
    assert settings.ipo_show_listed_days == 1
    assert "Listed Yesterday Co" in _names(detector.get_ipo_watchlist())


def test_watchlist_includes_both_undated_rows(detector, seeded):
    """'' and NULL both mean TBA and must survive the date filter."""
    names = _names(detector.get_ipo_watchlist())
    assert "Empty Dated Rumor Co" in names
    assert "Null Dated Upcoming Co" in names


def test_watchlist_undated_rows_render_as_tba(detector, seeded):
    """The card and /ipos both key off falsiness, so '' and NULL are equivalent."""
    rows = {r["company_name"]: r for r in detector.get_ipo_watchlist()}
    assert not rows["Empty Dated Rumor Co"]["ipo_date"]
    assert not rows["Null Dated Upcoming Co"]["ipo_date"]


def test_watchlist_excludes_withdrawn(detector, db, seeded):
    _insert_ipo(db, "Pulled Deal Co", _iso(5), "withdrawn")
    assert "Pulled Deal Co" not in _names(detector.get_ipo_watchlist())


def test_watchlist_honours_limit(detector, seeded):
    assert len(detector.get_ipo_watchlist(limit=2)) == 2


# ── get_ipo_watchlist ordering ───────────────────────────────────────────────

def test_watchlist_future_priced_is_first(detector, seeded):
    """The real future offering leads the board.

    It outranks every TBA row, any stale `upcoming` row (those are now removed
    by the date filter outright) and — the point of the three-way bucket —
    yesterday's completed listing, which is the clutter that was reported.
    """
    names = _names(detector.get_ipo_watchlist())
    assert names[0] == "Future Priced Co"
    assert names.index("Future Priced Co") < names.index("Empty Dated Rumor Co")
    assert names.index("Future Priced Co") < names.index("Null Dated Upcoming Co")
    assert names.index("Future Priced Co") < names.index("Listed Yesterday Co")


def test_watchlist_grace_window_listing_sorts_last(detector, seeded):
    """A passed date is a receipt, not a watchlist item — it goes to the bottom.

    Below the TBA rows too: an unannounced offering is still ahead of us, while
    this one has already happened.
    """
    names = _names(detector.get_ipo_watchlist())
    assert names[-1] == "Listed Yesterday Co"


def test_watchlist_today_dated_row_ranks_as_forward_looking(detector, db, seeded):
    """The bucket boundary is `>= date('now')`, so today is not yet past."""
    _insert_ipo(db, "Lists Today Co", _iso(0), "priced")
    names = _names(detector.get_ipo_watchlist())
    assert names[0] == "Lists Today Co"      # soonest date in bucket 0
    assert names.index("Lists Today Co") < names.index("Null Dated Upcoming Co")


def test_watchlist_drops_grace_window_listing_when_disabled(
    detector, seeded, monkeypatch
):
    """ipo_show_listed_days=0 removes yesterday's listing from the board."""
    monkeypatch.setattr(settings, "ipo_show_listed_days", 0)
    assert "Listed Yesterday Co" not in _names(detector.get_ipo_watchlist())


def test_watchlist_orders_future_then_tba_then_grace_window(detector, seeded):
    """Full contract: bucket, then date ascending, then status, then detected_at."""
    assert _names(detector.get_ipo_watchlist()) == [
        "Future Priced Co",         # bucket 0 — dated today or later
        "Null Dated Upcoming Co",   # bucket 1 — TBA, upcoming > rumored
        "Empty Dated Rumor Co",     # bucket 1 — TBA, rumored last
        "Listed Yesterday Co",      # bucket 2 — already happened
    ]


def test_watchlist_status_priority_breaks_same_date_ties(detector, db):
    """priced > upcoming > listed > rumored, applied only within one date."""
    same_day = _iso(3)
    for name, status in [
        ("Rumor Co", "rumored"),
        ("Listed Co", "listed"),
        ("Upcoming Co", "upcoming"),
        ("Priced Co", "priced"),
    ]:
        _insert_ipo(db, name, same_day, status)

    assert _names(IPODetector(db).get_ipo_watchlist()) == [
        "Priced Co", "Upcoming Co", "Listed Co", "Rumor Co",
    ]


# ── retire_stale ─────────────────────────────────────────────────────────────

def _statuses(db) -> dict[str, str]:
    with db.connection() as conn:
        return {
            r["company_name"]: r["status"]
            for r in conn.execute("SELECT company_name, status FROM ipo_tracker")
        }


def test_retire_stale_flips_past_dated_upcoming_to_listed(detector, db, seeded):
    detector.retire_stale(listed_retention_days=settings.ipo_listed_retention_days)
    assert _statuses(db)["Past Upcoming Co"] == "listed"


def test_retire_stale_flips_past_dated_priced_to_listed(detector, db):
    _insert_ipo(db, "Priced Last Week Co", _iso(-7), "priced")
    detector.retire_stale(listed_retention_days=settings.ipo_listed_retention_days)
    assert _statuses(db)["Priced Last Week Co"] == "listed"


def test_retire_stale_leaves_future_dated_rows_alone(detector, db, seeded):
    detector.retire_stale(listed_retention_days=settings.ipo_listed_retention_days)
    assert _statuses(db)["Future Priced Co"] == "priced"


def test_retire_stale_leaves_undated_rows_alone(detector, db, seeded):
    """A TBA offering has not happened yet — it must not be flipped to listed."""
    detector.retire_stale(listed_retention_days=settings.ipo_listed_retention_days)
    statuses = _statuses(db)
    assert statuses["Null Dated Upcoming Co"] == "upcoming"
    assert statuses["Empty Dated Rumor Co"] == "rumored"


def test_retire_stale_nulls_empty_string_dates(detector, db, seeded):
    detector.retire_stale(listed_retention_days=settings.ipo_listed_retention_days)
    with db.connection() as conn:
        row = conn.execute(
            "SELECT ipo_date FROM ipo_tracker WHERE company_name = ?",
            ("Empty Dated Rumor Co",),
        ).fetchone()
    assert row["ipo_date"] is None


def test_retire_stale_does_not_delete_undated_rows(detector, db, seeded):
    """The old backdate rule deleted them: '' < any cutoff is true in SQLite."""
    detector.retire_stale(listed_retention_days=settings.ipo_listed_retention_days)
    names = set(_statuses(db))
    assert "Empty Dated Rumor Co" in names
    assert "Null Dated Upcoming Co" in names


def test_retire_stale_still_deletes_long_past_and_withdrawn(detector, db):
    _insert_ipo(db, "Ancient Listing Co", _iso(-200), "listed")
    _insert_ipo(
        db, "Withdrawn Long Ago Co", _iso(5), "withdrawn",
        detected_at="2020-01-01 00:00:00",
    )
    removed = detector.retire_stale(
        listed_retention_days=settings.ipo_listed_retention_days
    )
    assert removed == 2
    assert _statuses(db) == {}


def test_retire_stale_is_idempotent(detector, db, seeded):
    retention = settings.ipo_listed_retention_days
    detector.retire_stale(listed_retention_days=retention)
    before = _statuses(db)
    assert detector.retire_stale(listed_retention_days=retention) == 0
    assert _statuses(db) == before


# ── _store_ipo date handling ─────────────────────────────────────────────────

def _ipo_date(db, company_name: str):
    with db.connection() as conn:
        row = conn.execute(
            "SELECT ipo_date FROM ipo_tracker WHERE company_name = ?",
            (company_name,),
        ).fetchone()
    return row["ipo_date"] if row else None


def test_store_ipo_insert_writes_null_not_empty_string(detector, db):
    detector._store_ipo({
        "company_name": "Undated Newcomer Co",
        "status": "rumored",
        "expected_date": normalize_date("TBD"),
        "notes": "",
    })
    assert _ipo_date(db, "Undated Newcomer Co") is None


def test_store_ipo_update_keeps_existing_date_when_extraction_has_none(detector, db):
    """The regression this guards: the UPDATE used to overwrite with ''."""
    known = _iso(14)
    _insert_ipo(db, "Known Date Co", known, "upcoming")

    detector._store_ipo({
        "company_name": "Known Date Co",
        "status": "priced",
        "expected_date": "",          # re-extraction could not read a date
        "notes": "priced the offering",
    })

    assert _ipo_date(db, "Known Date Co") == known
    assert _statuses(db)["Known Date Co"] == "priced"


def test_store_ipo_update_accepts_a_newly_learned_date(detector, db):
    """COALESCE must not be so sticky that a real date can never be set."""
    _insert_ipo(db, "Newly Dated Co", None, "rumored")
    announced = _iso(21)

    detector._store_ipo({
        "company_name": "Newly Dated Co",
        "status": "upcoming",
        "expected_date": announced,
        "notes": "",
    })

    assert _ipo_date(db, "Newly Dated Co") == announced


def test_store_ipo_update_replaces_a_stale_date_with_a_revised_one(detector, db):
    _insert_ipo(db, "Rescheduled Co", _iso(5), "upcoming")
    revised = _iso(30)

    detector._store_ipo({
        "company_name": "Rescheduled Co",
        "status": "upcoming",
        "expected_date": revised,
        "notes": "delayed",
    })

    assert _ipo_date(db, "Rescheduled Co") == revised


# ── _store_ipo status precedence ─────────────────────────────────────────────

def test_store_ipo_update_does_not_downgrade_a_listed_row(detector, db):
    """A retrospective article must not walk a completed listing back.

    Without this, an LLM extraction reading "Acme began trading last week" as
    status='upcoming' would undo retire_stale's flip and put the row back on the
    watchlist claiming to be a future event — the reported bug, restored once an
    hour. Everything except status still updates.
    """
    _insert_ipo(db, "Already Listed Co", _iso(-2), "listed")
    revised = _iso(-3)

    detector._store_ipo({
        "company_name": "Already Listed Co",
        "status": "upcoming",
        "expected_date": revised,
        "notes": "retrospective coverage",
    })

    assert _statuses(db)["Already Listed Co"] == "listed"
    assert _ipo_date(db, "Already Listed Co") == revised


def test_store_ipo_update_does_not_downgrade_a_withdrawn_row(detector, db):
    _insert_ipo(db, "Pulled Deal Co", _iso(10), "withdrawn")

    detector._store_ipo({
        "company_name": "Pulled Deal Co",
        "status": "rumored",
        "expected_date": "",
        "notes": "still being talked about",
    })

    assert _statuses(db)["Pulled Deal Co"] == "withdrawn"


def test_store_ipo_finnhub_may_revise_a_terminal_status(detector, db):
    """Finnhub knows the status for certain, so it is the one writer allowed to."""
    _insert_ipo(db, "Revived Deal Co", _iso(10), "withdrawn")

    detector._store_ipo({
        "company_name": "Revived Deal Co",
        "status": "priced",
        "expected_date": _iso(12),
        "notes": "",
        "source": "finnhub",
    })

    assert _statuses(db)["Revived Deal Co"] == "priced"


def test_store_ipo_update_still_advances_a_non_terminal_status(detector, db):
    """The guard covers listed/withdrawn only — rumored → priced must work."""
    _insert_ipo(db, "Progressing Co", _iso(10), "rumored")

    detector._store_ipo({
        "company_name": "Progressing Co",
        "status": "priced",
        "expected_date": _iso(10),
        "notes": "",
    })

    assert _statuses(db)["Progressing Co"] == "priced"


def test_store_ipo_update_keeps_status_when_extraction_omits_it(detector, db):
    """An absent status must not be written as NULL over a known one."""
    _insert_ipo(db, "Statusless Update Co", _iso(10), "upcoming")

    detector._store_ipo({
        "company_name": "Statusless Update Co",
        "expected_date": _iso(11),
        "notes": "",
    })

    assert _statuses(db)["Statusless Update Co"] == "upcoming"


def test_store_ipo_undated_llm_row_reaches_the_watchlist(detector, db):
    """End to end: an undated extraction is stored, survives retirement, shows."""
    detector._store_ipo({
        "company_name": "Quiet Filing Co",
        "status": "upcoming",
        "expected_date": normalize_date("N/A"),
        "notes": "",
    })
    detector.retire_stale(
        listed_retention_days=settings.ipo_listed_retention_days
    )
    rows = {r["company_name"]: r for r in detector.get_ipo_watchlist()}
    assert "Quiet Filing Co" in rows
    assert not rows["Quiet Filing Co"]["ipo_date"]


# ── migration ────────────────────────────────────────────────────────────────

def test_initialize_nulls_legacy_empty_ipo_dates(tmp_path):
    """The one-time cleanup for rows already on disk with ipo_date = ''."""
    path = str(tmp_path / "legacy.db")
    first = Database(db_path=path)
    first.initialize()
    _insert_ipo(first, "Legacy Empty Date Co", "", "upcoming")

    reopened = Database(db_path=path)
    reopened.initialize()

    assert _ipo_date(reopened, "Legacy Empty Date Co") is None


def test_ipo_tracker_date_index_exists(db):
    with db.connection() as conn:
        names = {
            r["name"]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = 'ipo_tracker'"
            )
        }
    assert "idx_ipo_tracker_date" in names
