"""
One-shot backfill of FINRA off-exchange (dark pool) volume.

Deliberately manual rather than a first-run branch inside the scheduled job:
this is ~1,300 sequential requests over 30-60 minutes, and doing that during
orchestrator startup would stall the whole pipeline on a cold install. The
scheduled job's own cold-start window is DARKPOOL_BACKFILL_DAYS (default 30),
which is enough for it to self-heal but not enough to train on.

Walks BACKWARD from today, which does two useful things: the most valuable
(most recent) sessions land first so an interrupted run still leaves something
usable, and it discovers the CDN's retention floor rather than assuming one.
FINRA keeps a rolling window of roughly eight years and the floor moves forward
over time, so a hardcoded start date would quietly start failing.

Because one file carries every US symbol, re-running this for a newly tracked
ticker costs a full re-download. That is the deliberate trade for not storing
~24M rows of symbols nobody watches.

Usage:
    python scripts/manual/backfill_darkpool.py
    python scripts/manual/backfill_darkpool.py --days 2000
    python scripts/manual/backfill_darkpool.py --tickers NVDA,AMD
"""

import argparse
import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.settings import settings
from data.database import Database
from data.tickers import US, classify_market
from pipeline.darkpool import FinraShortVolumeClient

# 5.5 years: the predictor trains on 5 years of prices and needs a 20-session
# warm-up before the first feature vector, so this covers the full window with
# room for the trailing z-score lookbacks.
DEFAULT_DAYS = 2000

# Consecutive weekday misses that mean we are past the retention floor rather
# than in a holiday week. No real holiday run comes close to ten weekdays.
FLOOR_MISS_STREAK = 10


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS,
                        help=f"How far back to walk (default {DEFAULT_DAYS}).")
    parser.add_argument("--tickers", type=str, default="",
                        help="Comma-separated override; defaults to tracked tickers.")
    args = parser.parse_args()

    db = Database(settings.db_path)
    db.initialize()

    if args.tickers:
        requested = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        requested = db.get_tracked_tickers()

    wanted = {t for t in requested if classify_market(t) == US}
    skipped = sorted(set(requested) - wanted)
    if not wanted:
        print(f"No US tickers to backfill (requested: {requested or 'none'}).")
        return

    print(f"Backfilling {len(wanted)} ticker(s): {', '.join(sorted(wanted))}")
    if skipped:
        print(f"Skipping non-US: {', '.join(skipped)}")
    print(f"Walking back up to {args.days} days from today.\n")

    client = FinraShortVolumeClient()
    today = datetime.now(timezone.utc).date()
    stop_at = today - timedelta(days=args.days)

    sessions = rows_written = misses = 0
    streak = 0
    day = today

    async with client.session() as http:
        while day >= stop_at:
            if day.weekday() >= 5:
                day -= timedelta(days=1)
                continue

            parsed = await client.fetch_day(day, http)
            if parsed is None:
                misses += 1
                streak += 1
                if streak >= FLOOR_MISS_STREAK:
                    print(f"\n{streak} consecutive weekday misses ending {day} — "
                          f"treating this as the CDN retention floor and stopping.")
                    break
                day -= timedelta(days=1)
                continue

            streak = 0
            tracked = [r for r in parsed if r["ticker"] in wanted]
            rows_written += db.upsert_offexchange_volume(tracked)
            sessions += 1

            if sessions % 25 == 0:
                print(f"  {day}  ({sessions} sessions, {rows_written} rows)")

            await asyncio.sleep(0.3)
            day -= timedelta(days=1)

    print(f"\nDone. {sessions} sessions stored, {rows_written} rows written, "
          f"{misses} absent (weekends excluded, holidays counted).")

    earliest = None
    with db.connection() as conn:
        row = conn.execute(
            "SELECT MIN(session_date) AS lo, MAX(session_date) AS hi,"
            " COUNT(*) AS n FROM offexchange_volume"
        ).fetchone()
        if row:
            earliest = row["lo"]
            print(f"Table now holds {row['n']} rows spanning {row['lo']} .. {row['hi']}.")

    if earliest:
        # Scans the whole stored series rather than the tail: price_history is
        # filled on demand by the predictor, so the most recent sessions are
        # routinely unpriced even when years of overlap exist further back.
        # Checking only the tail reports "no data" on a healthy table.
        print("\nSanity check — off-exchange share of consolidated volume "
              "should sit around 0.30-0.50 for large caps:")
        for ticker in sorted(wanted):
            shares = [
                r["total_volume"] / r["consolidated_volume"]
                for r in db.get_offexchange_series(ticker)
                if r.get("consolidated_volume") and r.get("total_volume")
            ]
            if shares:
                avg = sum(shares) / len(shares)
                flag = "" if 0.20 <= avg <= 0.65 else "   <-- outside the expected band"
                print(f"  {ticker:<8} {avg:.1%} over {len(shares):>4} priced sessions{flag}")
            else:
                print(f"  {ticker:<8} no price_history overlap — run a price fetch first")


if __name__ == "__main__":
    asyncio.run(main())
