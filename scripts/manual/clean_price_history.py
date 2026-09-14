"""
Find, and optionally repair, rows in price_history that are not daily bars.

Yahoo's chart endpoint silently answers `range=max&interval=1d` with MONTHLY
bars. seasonality.ensure_deep_history asked PriceFeed for exactly that, so on
the phone some symbols (SPY, QQQ) hold decades of monthly rows in front of their
daily history, with the daily rows starting only where a normal refresh first
covered them. Every rolling window that spans the join reads a month as a day.
PriceFeed now requests "max" as an explicit period1/period2 window, which
returns true daily bars, and refuses any non-daily response — so this has to run
once per database, not on a schedule.

For each ticker it finds the rows pipeline.features.clean_daily_bars would
discard — dated on a weekend, or before the trailing run of daily bars — and
prints the counts. With --apply it deletes them and re-fetches the ticker's full
daily history. The delete comes first on purpose: a monthly row stamped on a
weekday holiday (1 January, say) would survive a re-fetch, because no daily bar
replaces it, and sit inside the daily run where the gap rule cannot see it.

Crypto pairs (-USD) trade at weekends, so their weekend rows are kept.

Dry run by default; only --apply writes. Uses Database(), so it honours DB_PATH.

Usage:
    python scripts/manual/clean_price_history.py
    python scripts/manual/clean_price_history.py --tickers SPY,QQQ
    python scripts/manual/clean_price_history.py --apply
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd

from config.logging_config import get_logger
from data.database import Database
from pipeline.features import clean_daily_bars
from pipeline.price_feed import PriceFeed

log = get_logger(__name__)

# Seconds between re-fetches. A full daily history is the heaviest request made
# of Yahoo, and a burst of them is how the quote refresh gets throttled.
REFETCH_DELAY = 2.0


def trades_on_weekends(ticker: str) -> bool:
    return ticker.upper().endswith("-USD")


def stored_tickers(db: Database) -> list[str]:
    with db.connection() as conn:
        rows = conn.execute(
            "SELECT DISTINCT ticker FROM price_history ORDER BY ticker"
        ).fetchall()
    return [r["ticker"] for r in rows]


def find_bad_rows(db: Database, ticker: str) -> tuple[list[str], int, str | None]:
    """(stored dates to remove, how many of them are weekend rows, daily segment start)."""
    rows = db.get_price_history(ticker)
    if not rows:
        return [], 0, None
    stored = [str(r["date"]) for r in rows]
    days = pd.DatetimeIndex(pd.to_datetime([s[:10] for s in stored]))
    frame = pd.DataFrame({"close": [float(r["close"]) for r in rows]}, index=days)

    kept = clean_daily_bars(frame, keep_weekends=trades_on_weekends(ticker))
    keep = set(kept.index)
    bad = [(s, day) for s, day in zip(stored, days) if day not in keep]
    weekend = sum(1 for _, day in bad if day.dayofweek >= 5)
    start = kept.index[0].strftime("%Y-%m-%d") if len(kept) else None
    return [s for s, _ in bad], weekend, start


def delete_rows(db: Database, ticker: str, dates: list[str]) -> int:
    with db.connection() as conn:
        cursor = conn.executemany(
            "DELETE FROM price_history WHERE ticker = ? AND date = ?",
            [(ticker, d) for d in dates],
        )
        return cursor.rowcount or 0


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers",
                        help="comma-separated subset (default: every ticker in price_history)")
    parser.add_argument("--apply", action="store_true",
                        help="delete the rows and re-fetch each affected ticker's daily history")
    args = parser.parse_args()

    db = Database()
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = stored_tickers(db)

    plan: list[tuple[str, list[str]]] = []
    for ticker in tickers:
        bad, weekend, start = find_bad_rows(db, ticker)
        if not bad:
            continue
        plan.append((ticker, bad))
        print(f"  {ticker}: {len(bad)} row(s) to remove: {weekend} weekend-dated, "
              f"{len(bad) - weekend} before the daily segment starting {start}")

    if not plan:
        print(f"price_history is clean ({len(tickers)} ticker(s) checked).")
        return 0

    total = sum(len(bad) for _, bad in plan)
    print(f"\n{len(plan)} of {len(tickers)} ticker(s) affected, {total} row(s) in all.")
    if not args.apply:
        print("Dry run: nothing changed. Re-run with --apply to delete these rows "
              "and re-fetch full daily history for the affected tickers.")
        return 0

    # Only a writing run may touch the schema (price_splits must exist for the
    # re-fetch to store split events).
    db.initialize()
    feed = PriceFeed(db)
    unresolved = 0
    for i, (ticker, bad) in enumerate(plan):
        if i:
            await asyncio.sleep(REFETCH_DELAY)
        deleted = await asyncio.to_thread(delete_rows, db, ticker, bad)
        stored = await feed.refresh_history_for([ticker], history_range="max")
        remaining, _, start = find_bad_rows(db, ticker)
        ok = bool(stored) and not remaining
        unresolved += 0 if ok else 1
        print(f"  {ticker}: deleted {deleted}, re-fetched {stored} daily bar(s); "
              f"{len(remaining)} bad row(s) remain, daily from {start}"
              f"{'' if ok else '  <- CHECK'}")
        log.info("clean_price_history.ticker", ticker=ticker, deleted=deleted,
                 refetched=stored, remaining=len(remaining), segment_start=start)

    print(f"\nDone: {len(plan) - unresolved} repaired, {unresolved} need a look.")
    return 1 if unresolved else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
