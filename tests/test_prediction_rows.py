"""Tests for data.prediction_rows — which row speaks for a horizon, and accuracy splits."""

import pytest

from data.prediction_rows import (
    accuracy_breakdown,
    accuracy_by_horizon,
    latest_by_horizon,
)


def pred(horizon, model_type, created_at, **extra):
    return {"horizon_days": horizon, "model_type": model_type,
            "created_at": created_at, **extra}


class TestLatestByHorizon:

    def test_model_row_beats_a_newer_debate_row(self):
        rows = [  # newest first, as get_recent_predictions returns them
            pred(5, "multi_agent", "2026-09-13 09:00:00"),
            pred(5, "universal", "2026-09-13 08:30:00"),
            pred(5, "universal", "2026-09-12 08:30:00"),
        ]
        assert latest_by_horizon(rows)[5] is rows[1]

    def test_prior_row_counts_as_the_model_speaking(self):
        rows = [
            pred(252, "llm_only", "2026-09-13 09:00:00"),
            pred(252, "prior", "2026-09-12 08:30:00"),
        ]
        assert latest_by_horizon(rows)[252]["model_type"] == "prior"

    def test_falls_back_to_newest_of_any_type(self):
        rows = [
            pred(21, "multi_agent", "2026-09-13 09:00:00"),
            pred(21, "per_ticker", "2026-09-10 08:30:00"),
        ]
        assert latest_by_horizon(rows)[21] is rows[0]

    def test_model_types_filter_applies_to_both_passes(self):
        rows = [
            pred(5, "multi_agent", "2026-09-13 09:00:00"),
            pred(5, "llm_only", "2026-09-12 08:30:00"),
            pred(63, "multi_agent", "2026-09-13 09:00:00"),
        ]
        picked = latest_by_horizon(rows, {"universal", "prior", "llm_only"})
        assert picked[5]["model_type"] == "llm_only"
        assert 63 not in picked

    def test_rows_without_a_horizon_are_ignored(self):
        assert latest_by_horizon([{"model_type": "universal"}]) == {}

    def test_accepts_a_generator(self):
        rows = (pred(h, "universal", "2026-09-13") for h in (5, 21))
        assert sorted(latest_by_horizon(rows)) == [5, 21]


class FakeDb:
    """The two accuracy queries, answered from a fixed list of resolved rows."""

    def __init__(self, resolved):
        self.resolved = resolved  # (ticker, horizon_days, model_type, is_correct)

    def get_prediction_accuracy_breakdown(self):
        groups = {}
        for _, horizon, model_type, correct in self.resolved:
            total_correct = groups.setdefault((horizon, model_type), [0, 0])
            total_correct[0] += 1
            total_correct[1] += correct
        return [{"horizon_days": h, "model_type": m, "total": t, "correct": c,
                 "accuracy_pct": c / t * 100}
                for (h, m), (t, c) in sorted(groups.items())]

    def get_prediction_accuracy(self, ticker=None, horizon_days=None, model_type=None):
        hits = [c for t, h, m, c in self.resolved
                if (not ticker or t == ticker)
                and (horizon_days is None or h == horizon_days)
                and (not model_type or m == model_type)]
        return {"total": len(hits), "correct": sum(hits),
                "incorrect": len(hits) - sum(hits)}


RESOLVED = [
    ("AAPL", 5, "universal", 1), ("AAPL", 5, "universal", 0),
    ("AAPL", 5, "prior", 1),
    ("MSFT", 5, "universal", 1),
    ("MSFT", 252, "prior", 1),
]


class TestAccuracyBreakdown:

    def test_all_tickers_uses_the_grouped_counts(self):
        rows = accuracy_breakdown(FakeDb(RESOLVED))
        assert [(r["horizon_days"], r["model_type"], r["total"], r["correct"]) for r in rows] == [
            (5, "prior", 1, 1),
            (5, "universal", 3, 2),
            (252, "prior", 1, 1),
        ]
        assert rows[1]["accuracy_pct"] == pytest.approx(200 / 3)
        assert set(rows[0]) == {"model_type", "horizon_days", "total", "correct", "accuracy_pct"}

    def test_one_ticker_recounts_and_drops_empty_combinations(self):
        rows = accuracy_breakdown(FakeDb(RESOLVED), "AAPL")
        assert [(r["horizon_days"], r["model_type"], r["total"], r["correct"]) for r in rows] == [
            (5, "prior", 1, 1),
            (5, "universal", 2, 1),
        ]

    def test_by_horizon_sums_model_types(self):
        by_horizon = accuracy_by_horizon(accuracy_breakdown(FakeDb(RESOLVED)))
        assert by_horizon == [
            {"horizon_days": 5, "total": 4, "correct": 3, "accuracy_pct": 75.0},
            {"horizon_days": 252, "total": 1, "correct": 1, "accuracy_pct": 100.0},
        ]

    def test_no_resolved_rows(self):
        assert accuracy_breakdown(FakeDb([])) == []
        assert accuracy_by_horizon([]) == []
