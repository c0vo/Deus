"""Tests for thesis accountability — grading matured EARLY calls.

The engine is only worth trusting if it can be scored after the fact, and the
scoring has exactly two ways to lie: crediting a name for a move the whole
market made, and re-teaching the same lesson every morning. Both are covered
here alongside the age gate that decides when a call is old enough to judge.
"""

import json
import pytest


@pytest.fixture
def db(tmp_path):
    from data.database import Database
    database = Database(db_path=str(tmp_path / "test.db"))
    database.initialize()
    return database


def _seed(db, snapshots, spy=(("2026-01-10", 500.0), ("2026-06-10", 550.0))):
    """One thesis, one hop-3 node, and a candidate per distinct id in snapshots.

    `snapshots` is a list of (candidate_id, ticker, date, price, stage).
    """
    with db.connection() as conn:
        conn.execute(
            "INSERT INTO theses (id, title, summary, status, generated_at) "
            "VALUES ('t1', 'HBM supply crunch', 's', 'active', '2026-01-01')"
        )
        conn.execute(
            "INSERT INTO thesis_nodes (id, thesis_id, node_key, order_depth, claim, "
            "mechanism, falsifier, confidence) VALUES "
            "('n1','t1','packaging',3,'Packaging is the chokepoint',"
            "'CoWoS capacity is fixed','Capacity doubles ahead of schedule',0.8)"
        )
        seen = {}
        for cid, ticker, _d, _p, _s in snapshots:
            seen[cid] = ticker
        for cid, ticker in seen.items():
            conn.execute(
                "INSERT INTO thesis_candidates (id, thesis_id, node_id, company_name, "
                "ticker, role_in_chain, listing_status) VALUES (?,?,?,?,?,?,?)",
                (cid, "t1", "n1", f"{ticker} Inc", ticker,
                 "advanced packaging", "us_listed"),
            )
        for cid, ticker, d, price, stage in snapshots:
            conn.execute(
                "INSERT INTO thesis_candidate_snapshots (candidate_id, ticker, "
                "as_of_date, price, rumour_stage, edge_score) VALUES (?,?,?,?,?,?)",
                (cid, ticker, d, price, stage, 0.5),
            )
        for d, close in spy:
            conn.execute(
                "INSERT INTO price_history (ticker, date, open, high, low, close, "
                "volume) VALUES ('SPY',?,?,?,?,?,0)", (d, close, close, close, close),
            )


class TestDueForReview:
    """Which calls the job picks up."""

    def test_matured_early_call_is_returned_with_both_endpoints(self, db):
        _seed(db, [
            ("c1", "AMKR", "2026-01-10", 100.0, "EARLY"),
            ("c1", "AMKR", "2026-06-10", 150.0, "BUILDING"),
        ])
        rows = db.get_thesis_calls_due_review(min_age_days=45)
        assert len(rows) == 1
        r = rows[0]
        assert r["ticker"] == "AMKR"
        assert (r["flagged_date"], r["flagged_price"]) == ("2026-01-10", 100.0)
        assert (r["latest_date"], r["latest_price"]) == ("2026-06-10", 150.0)
        # The falsifier rides along so the lesson can name what would have
        # broken the link rather than only that the price moved.
        assert r["falsifier"] == "Capacity doubles ahead of schedule"

    def test_call_younger_than_the_gate_is_not_yet_judged(self, db):
        _seed(db, [
            ("c3", "NEW", "2026-06-05", 100.0, "EARLY"),
            ("c3", "NEW", "2026-06-10", 120.0, "EARLY"),
        ])
        assert db.get_thesis_calls_due_review(min_age_days=45) == []

    def test_candidate_never_flagged_early_is_ignored(self, db):
        """The engine's claim is about EARLY calls, so only those are graded."""
        _seed(db, [
            ("c4", "LATE", "2026-01-10", 100.0, "CROWDED"),
            ("c4", "LATE", "2026-06-10", 150.0, "CROWDED"),
        ])
        assert db.get_thesis_calls_due_review(min_age_days=45) == []


class TestBenchmark:
    """Excess return, which is the only number that means anything."""

    def test_benchmark_closes_are_joined_on_both_dates(self, db):
        _seed(db, [
            ("c1", "AMKR", "2026-01-10", 100.0, "EARLY"),
            ("c1", "AMKR", "2026-06-10", 150.0, "BUILDING"),
        ])
        r = db.get_thesis_calls_due_review(min_age_days=45)[0]
        assert (r["bench_start"], r["bench_end"]) == (500.0, 550.0)

    def test_a_name_that_lagged_the_market_is_not_a_win(self, db):
        """+3% while SPY did +10% is a losing call, not a winning one.

        Grading on the absolute move would score this as a success and teach
        the debate the opposite of what happened.
        """
        _seed(db, [
            ("c2", "SOXX", "2026-01-10", 100.0, "EARLY"),
            ("c2", "SOXX", "2026-06-10", 103.0, "EARLY"),
        ])
        r = db.get_thesis_calls_due_review(min_age_days=45)[0]
        ret = (r["latest_price"] - r["flagged_price"]) / r["flagged_price"]
        bench = (r["bench_end"] - r["bench_start"]) / r["bench_start"]
        assert ret > 0 and (ret - bench) < 0

    def test_missing_benchmark_bars_are_reported_as_absent(self, db):
        """No SPY history must surface as None, not as a silent zero return."""
        _seed(db, [
            ("c1", "AMKR", "2026-01-10", 100.0, "EARLY"),
            ("c1", "AMKR", "2026-06-10", 150.0, "BUILDING"),
        ], spy=())
        r = db.get_thesis_calls_due_review(min_age_days=45)[0]
        assert r["bench_start"] is None and r["bench_end"] is None


class TestIdempotency:
    def test_a_reviewed_call_is_not_reviewed_again(self, db):
        """The job runs daily; without this guard it re-teaches every morning."""
        _seed(db, [
            ("c1", "AMKR", "2026-01-10", 100.0, "EARLY"),
            ("c1", "AMKR", "2026-06-10", 150.0, "BUILDING"),
            ("c2", "SOXX", "2026-01-10", 100.0, "EARLY"),
            ("c2", "SOXX", "2026-06-10", 103.0, "EARLY"),
        ])
        assert len(db.get_thesis_calls_due_review(min_age_days=45)) == 2

        db.insert_reflection("AMKR", None, "2026-06-11", "lesson", True,
                             "ticker", None, json.dumps({"thesis_review": "c1"}))

        remaining = db.get_thesis_calls_due_review(min_age_days=45)
        assert [r["ticker"] for r in remaining] == ["SOXX"]

    def test_stamp_matches_the_whole_id_not_a_prefix(self, db):
        """'c1' must not suppress 'c10' — the tag is matched quoted for this."""
        _seed(db, [
            ("c1", "AMKR", "2026-01-10", 100.0, "EARLY"),
            ("c1", "AMKR", "2026-06-10", 150.0, "BUILDING"),
            ("c10", "OTHER", "2026-01-10", 100.0, "EARLY"),
            ("c10", "OTHER", "2026-06-10", 150.0, "BUILDING"),
        ])
        db.insert_reflection("AMKR", None, "2026-06-11", "lesson", True,
                             "ticker", None, json.dumps({"thesis_review": "c1"}))
        remaining = db.get_thesis_calls_due_review(min_age_days=45)
        assert [r["ticker"] for r in remaining] == ["OTHER"]


class TestLessonsReachTheDebate:
    def test_written_lesson_is_visible_to_get_relevant_reflections(self, db):
        """scope='ticker' is what makes the loop close.

        The reader filters on it, so a lesson written at any other scope would
        be recorded and then never read by the debate it exists to inform.
        """
        db.insert_reflection("AMKR", None, "2026-06-11",
                             "Thesis flagged AMKR EARLY; it returned +40% excess.",
                             True, "ticker", None,
                             json.dumps({"thesis_review": "c1"}))
        lessons = db.get_relevant_reflections("AMKR")
        assert any("EARLY" in row["lesson_learned"]
                   for row in lessons["ticker_lessons"])
