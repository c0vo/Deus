"""
One-shot backfill of the market-wide regime series (DIX/GEX, OCC put/call).

The two feeds behave very differently and that shapes this script:

  DIX/GEX  one CSV carrying the whole history back to 2011. A single request
           does the entire backfill, so the scheduled job already re-ingests it
           in full every day and there is nothing here to tune.
  OCC      one request per session. This is the part that needs a manual run,
           for the same reason as the dark-pool backfill: ~1,300 sequential
           requests is not something a scheduled job should attempt on boot.

Like the dark-pool script this walks OCC backward from today, so an interrupted
run still leaves the most recent sessions in place.

Usage:
    python scripts/manual/backfill_market_regime.py
    python scripts/manual/backfill_market_regime.py --days 2000 --skip-dix
"""

import argparse
import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.settings import settings
from data.database import Database
from pipeline.market_regime import (
    METRIC_DIX,
    METRIC_GEX,
    METRIC_PUT_CALL,
    OccVolumeProvider,
    SqueezeMetricsProvider,
)

DEFAULT_DAYS = 2000
FLOOR_MISS_STREAK = 15


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS,
                        help=f"How far back to walk OCC (default {DEFAULT_DAYS}).")
    parser.add_argument("--skip-dix", action="store_true",
                        help="Skip the DIX/GEX CSV (it is one request anyway).")
    parser.add_argument("--skip-occ", action="store_true",
                        help="Skip the OCC put/call walk (the slow part).")
    args = parser.parse_args()

    db = Database(settings.db_path)
    db.initialize()

    if not args.skip_dix:
        print("Fetching the full DIX/GEX history (one request)...")
        rows = await SqueezeMetricsProvider().fetch()
        written = db.upsert_market_regime(rows)
        sessions = written // 2 if written else 0
        print(f"  {written} rows written (~{sessions} sessions, dix + gex).\n")

    if not args.skip_occ:
        print(f"Walking OCC put/call back up to {args.days} days...")
        occ = OccVolumeProvider()
        today = datetime.now(timezone.utc).date()
        stop_at = today - timedelta(days=args.days)

        stored = misses = 0
        streak = 0
        day = today
        batch: list[dict] = []

        async with occ.session() as http:
            while day >= stop_at:
                if day.weekday() >= 5:
                    day -= timedelta(days=1)
                    continue

                row = await occ.fetch_day(day, http)
                if row is None:
                    misses += 1
                    streak += 1
                    if streak >= FLOOR_MISS_STREAK:
                        print(f"\n  {streak} consecutive weekday misses ending {day} — "
                              f"treating this as the end of available history.")
                        break
                    day -= timedelta(days=1)
                    continue

                streak = 0
                row.pop("exchanges", None)
                batch.append(row)
                stored += 1

                # Flush periodically so an interrupted run keeps its progress.
                if len(batch) >= 50:
                    db.upsert_market_regime(batch)
                    batch = []
                    print(f"  {day}  ({stored} sessions)")

                await asyncio.sleep(0.3)
                day -= timedelta(days=1)

        if batch:
            db.upsert_market_regime(batch)
        print(f"\n  {stored} OCC sessions stored, {misses} absent.")

    print("\nStored coverage:")
    for metric in (METRIC_DIX, METRIC_GEX, METRIC_PUT_CALL):
        with db.connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n, MIN(session_date) AS lo, MAX(session_date) AS hi"
                " FROM market_regime_daily WHERE metric = ?",
                (metric,),
            ).fetchone()
        if row and row["n"]:
            print(f"  {metric:<20} {row['n']:>6} sessions  {row['lo']} .. {row['hi']}")
        else:
            print(f"  {metric:<20}      0 sessions")


if __name__ == "__main__":
    asyncio.run(main())
