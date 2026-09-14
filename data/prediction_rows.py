"""
Which stored prediction row speaks for a horizon, and how the calls scored.

A ticker holds several live prediction rows per horizon at once: the pooled
model's own call (`universal`, or `prior` when walk-forward evaluation found no
measurable edge at that horizon and the row carries the base rate), the
debate's `multi_agent` row that restates the same baseline beside its advisory,
and rows from retired tiers (`per_ticker`, `sector`, `fast_heuristic`,
`llm_only`). Taking "the newest row of any type" let a debate row, or the
UNKNOWN placeholder a debate wrote while no model existed, stand in for the
model's call and hide its statistics.

The market grid, the forecast narratives, the morning stance note and
Telegram's /markets all choose through `latest_by_horizon`; /api/accuracy and
/accuracy split the record through `accuracy_breakdown`. One rule in one place,
so no two surfaces disagree about which call is current or how it scored. The
worker's prediction refresh (train_missing_models) judges freshness on the same
row, so it never re-predicts a horizon whose shown call is still current.

This module imports nothing from the project, so data/, pipeline/, bot/ and
api/ can all use it without cycles, and the stance note without loading the ML
stack.
"""

from __future__ import annotations

from typing import Iterable, Optional

# The pooled model's own rows. `prior` is still the model speaking: it says the
# horizon has no measurable edge, and its probability is the base rate.
MODEL_ROW_TYPES = frozenset({"universal", "prior"})


def latest_by_horizon(rows: Iterable[dict],
                      model_types: Optional[Iterable[str]] = None) -> dict[int, dict]:
    """
    The row that speaks for each horizon, keyed by `horizon_days`.

    `rows` must be newest first, as `Database.get_recent_predictions` returns
    them. Per horizon the newest `universal` or `prior` row wins; a horizon with
    neither falls back to its newest row of any type. `model_types`, when given,
    restricts both passes to those types.
    """
    allowed = frozenset(model_types) if model_types is not None else None
    model_rows: dict[int, dict] = {}
    newest: dict[int, dict] = {}
    for row in rows:
        horizon = row.get("horizon_days")
        model_type = row.get("model_type")
        if horizon is None or (allowed is not None and model_type not in allowed):
            continue
        newest.setdefault(horizon, row)
        if model_type in MODEL_ROW_TYPES:
            model_rows.setdefault(horizon, row)
    return {horizon: model_rows.get(horizon, row) for horizon, row in newest.items()}


def _scored(total, correct) -> dict:
    total = int(total or 0)
    correct = int(correct or 0)
    return {
        "total": total,
        "correct": correct,
        "accuracy_pct": (correct / total * 100) if total else 0.0,
    }


def accuracy_breakdown(db, ticker: Optional[str] = None) -> list[dict]:
    """
    Resolved accuracy per horizon and model type, ordered by horizon then type:
    `{model_type, horizon_days, total, correct, accuracy_pct}`.

    Synchronous SQLite reads, so callers run it in a worker thread. Without a
    ticker it is one grouped query. With one, the combinations come from that
    query and each is counted again for the ticker, dropping any the ticker has
    no resolved row for.

    The split is the point: a `prior` row's hit rate is the base rate the model
    rows have to beat, and a 1y call averaged with a 5d call describes neither.
    """
    out = []
    for row in db.get_prediction_accuracy_breakdown():
        horizon, model_type = row["horizon_days"], row["model_type"]
        if ticker:
            acc = db.get_prediction_accuracy(ticker, horizon, model_type)
            total, correct = acc.get("total"), acc.get("correct")
        else:
            total, correct = row.get("total"), row.get("correct")
        if not total:
            continue
        out.append({"model_type": model_type, "horizon_days": horizon,
                    **_scored(total, correct)})
    return out


def accuracy_by_horizon(breakdown: Iterable[dict]) -> list[dict]:
    """`accuracy_breakdown` rows summed per horizon: `{horizon_days, total, correct, accuracy_pct}`."""
    sums: dict[int, list[int]] = {}
    for row in breakdown:
        running = sums.setdefault(row["horizon_days"], [0, 0])
        running[0] += row["total"]
        running[1] += row["correct"]
    return [{"horizon_days": horizon, **_scored(*sums[horizon])} for horizon in sorted(sums)]
