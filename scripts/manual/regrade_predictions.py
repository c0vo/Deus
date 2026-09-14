"""
Re-grade every resolved prediction with the session-based rule.

resolve_predictions used to grade a Yahoo chart window that ended on the night
the job ran and was sized in calendar days, so the "before" price was rarely the
price the prediction was made at. It now grades with
orchestrator.scheduler.grade_prediction: the base is the session the
prediction's features described (or, for older rows, the last New York close
before the prediction was made) and the outcome is the close exactly
horizon_days stored sessions later.

This applies that rule to the rows graded the old way and prints accuracy
before and after by model_type x horizon_days, and how many rows would change.
A row that cannot be graded from price_history — no stored bars for its ticker,
or no bar yet horizon_days sessions after its base — keeps its old grade and is
counted as ungradable. "acc after" therefore still mixes new grades with those
old ones; "acc gradable" is the accuracy over the regraded rows alone, and
"always UP" is what calling UP every time would have scored on those same rows,
the bar a directional call has to clear.

Dry run by default; --apply writes the new grades with
Database.regrade_prediction (resolved_at is left alone). Uses Database(), so it
honours DB_PATH, unless --db names another file.

Usage:
    python scripts/manual/regrade_predictions.py
    python scripts/manual/regrade_predictions.py --apply
    python scripts/manual/regrade_predictions.py --db storage/phone_snapshot_2026-09-13.db
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.logging_config import get_logger
from data.database import Database
from orchestrator.scheduler import grade_prediction

log = get_logger(__name__)

# A stored percentage within this of the new one is the same measurement.
PCT_TOLERANCE = 0.01


def resolved_predictions(db: Database) -> list[dict]:
    with db.connection() as conn:
        rows = conn.execute(
            "SELECT * FROM predictions WHERE is_correct IS NOT NULL ORDER BY created_at"
        ).fetchall()
    return [dict(r) for r in rows]


def regrade(db: Database) -> tuple[dict, list[tuple[dict, dict]]]:
    stats: dict[tuple[str, int], dict] = {}
    changes: list[tuple[dict, dict]] = []
    for row in resolved_predictions(db):
        key = (str(row.get("model_type")), int(row.get("horizon_days") or 0))
        s = stats.setdefault(key, {"resolved": 0, "before": 0, "after": 0, "regraded": 0,
                                   "correct_gradable": 0, "up_gradable": 0,
                                   "ungradable": 0, "changed": 0, "flipped": 0})
        old_correct = int(row["is_correct"])
        s["resolved"] += 1
        s["before"] += old_correct

        try:
            grade = grade_prediction(row, db)
        except Exception as e:
            log.warning("regrade.grade_failed", id=row.get("id"), ticker=row.get("ticker"), error=str(e))
            grade = None
        if grade is None:
            s["ungradable"] += 1
            s["after"] += old_correct
            continue

        new_correct = int(grade["is_correct"])
        s["regraded"] += 1
        s["after"] += new_correct
        s["correct_gradable"] += new_correct
        s["up_gradable"] += int(grade["actual_direction"] == "UP")
        old_pct = row.get("actual_change_pct")
        pct_moved = old_pct is None or abs(float(old_pct) - grade["actual_change_pct"]) > PCT_TOLERANCE
        if new_correct != old_correct or grade["actual_direction"] != row.get("actual_direction") or pct_moved:
            s["changed"] += 1
            s["flipped"] += int(new_correct != old_correct)
            changes.append((row, grade))
    return stats, changes


def _share(count: int, total: int) -> str:
    return f"{count / total:.1%}" if total else "n/a"


def _report_row(label: str, horizon: str, s: dict) -> tuple[str, ...]:
    return (label, horizon, str(s["resolved"]), _share(s["before"], s["resolved"]),
            _share(s["after"], s["resolved"]), _share(s["correct_gradable"], s["regraded"]),
            _share(s["up_gradable"], s["regraded"]), str(s["regraded"]), str(s["changed"]),
            str(s["flipped"]), str(s["ungradable"]))


def print_report(stats: dict) -> None:
    headers = ("model_type", "horizon", "resolved", "acc before", "acc after", "acc gradable",
               "always UP", "regraded", "changed", "flipped", "ungradable")
    body = []
    totals = {"resolved": 0, "before": 0, "after": 0, "regraded": 0, "correct_gradable": 0,
              "up_gradable": 0, "ungradable": 0, "changed": 0, "flipped": 0}
    for (model_type, horizon), s in sorted(stats.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        for k in totals:
            totals[k] += s[k]
        body.append(_report_row(model_type, f"{horizon}d", s))
    if totals["resolved"]:
        body.append(_report_row("all", "", totals))
    widths = [max(len(headers[i]), *(len(r[i]) for r in body)) if body else len(headers[i])
              for i in range(len(headers))]
    print(" | ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("-+-".join("-" * w for w in widths))
    for r in body:
        print(" | ".join(c.ljust(w) for c, w in zip(r, widths)))
    print(f"\nOn the {totals['regraded']} rows price_history can grade, the calls were right "
          f"{_share(totals['correct_gradable'], totals['regraded'])} of the time; always calling UP "
          f"would have been right {_share(totals['up_gradable'], totals['regraded'])}. "
          f"{totals['changed']} of {totals['resolved']} resolved rows would change "
          f"({totals['flipped']} flip between correct and incorrect); "
          f"{totals['ungradable']} cannot be graded and keep their old grade, which only "
          f"\"acc before\" and \"acc after\" count.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-grade resolved predictions on stored sessions.")
    parser.add_argument("--apply", action="store_true", help="write the new grades (default: dry run)")
    parser.add_argument("--db", default=None, help="database path (default DB_PATH)")
    args = parser.parse_args()

    db = Database(args.db) if args.db else Database()
    stats, changes = regrade(db)
    if not stats:
        print("No resolved predictions.")
        return
    print_report(stats)

    if not args.apply:
        print("\nDry run: nothing written. Re-run with --apply to store the new grades.")
        return
    for row, grade in changes:
        db.regrade_prediction(row["id"], grade["actual_direction"], grade["actual_change_pct"],
                              grade["is_correct"])
    print(f"\nApplied {len(changes)} regrades.")


if __name__ == "__main__":
    main()
