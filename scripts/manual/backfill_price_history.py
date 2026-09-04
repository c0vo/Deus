"""
One-shot deep backfill of daily OHLCV into price_history, then rate every ticker.

Deliberately manual rather than a first-run branch in the scheduled job, for the
same reason as backfill_darkpool.py: this pulls the full available history for
every tracked ticker and doing that during orchestrator startup would stall the
pipeline on a cold install. The scheduled job (PriceFeed.refresh_history) pulls
HISTORY_RANGE — enough to stay current and self-heal a gap, not enough to warm
up a 200-period moving average on monthly bars.

Why it exists: the technical rating needs SMA/EMA(200) on daily, weekly AND
monthly bars. The arithmetic is unforgiving —

    daily    200 bars  ~ 10 months of dailies
    weekly   200 bars  ~ 3.9 years of dailies
    monthly  200 bars  ~ 17 years of dailies

so without a deep pull the medium and long ratings simply do not exist. Tickers
listed recently will never have a long rating no matter how often this runs;
that is a fact about the ticker, not a failure of the backfill.

Unlike the dark-pool and consensus backfills, this one is genuinely useful
retroactively: a technical rating is a pure function of price, so deepening
price_history lets --rate populate every past session too.

Usage:
    python scripts/manual/backfill_price_history.py
    python scripts/manual/backfill_price_history.py --range 10y
    python scripts/manual/backfill_price_history.py --tickers NVDA,AMD
    python scripts/manual/backfill_price_history.py --no-rate
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.logging_config import get_logger
from data.database import Database
from pipeline.technical_rating import (
    MIN_BARS,
    TIMEFRAMES,
    TechnicalRatingTracker,
    compute_rating,
    resample_ohlcv,
)

log = get_logger(__name__)

# Everything Yahoo will give. There is no reason to ask for less: the response is
# a few hundred KB, the upsert is idempotent, and the monthly rating needs ~17
# years, which no fixed shorter window reliably covers.
DEFAULT_RANGE = "max"

# Seconds between tickers. yfinance blocks IPs on bursts, and a block here also
# breaks the predictor, the option snapshot and the consensus job — they all read
# the same host.
TICKER_DELAY = 2.0


def fetch_history(ticker: str, period: str) -> list[dict]:
    """Full daily OHLCV for one ticker as storage rows. Blocking."""
    import yfinance as yf

    # auto_adjust=False keeps the raw close. The rating compares price against
    # its own moving averages, so a split-adjusted series is fine either way —
    # but latest_prices and the dark-pool volume join are both unadjusted, and
    # mixing the two conventions in one table is the kind of thing that is
    # invisible until a ratio comes out an order of magnitude wrong.
    frame = yf.Ticker(ticker).history(period=period, auto_adjust=False)
    if frame is None or frame.empty:
        return []

    frame.columns = [c.lower() for c in frame.columns]
    frame = frame.dropna(subset=["close"])
    return [
        {
            "date": stamp.strftime("%Y-%m-%d"),
            "open": float(row.open),
            "high": float(row.high),
            "low": float(row.low),
            "close": float(row.close),
            "volume": float(row.volume or 0),
        }
        for stamp, row in frame.iterrows()
    ]


def report_depth(db: Database, ticker: str) -> None:
    """Say which timeframes this ticker can now support, and which it cannot.

    Printed per ticker because it is the one thing the operator actually needs
    from this script: a row count means nothing, "long: 42/280 bars" explains
    exactly why a panel will be blank.
    """
    bars = TechnicalRatingTracker.load_bars(db, ticker)
    if bars is None or bars.empty:
        print(f"  {ticker}: no bars stored")
        return

    parts = []
    for timeframe, rule in TIMEFRAMES.items():
        count = len(resample_ohlcv(bars, rule))
        ok = "ok" if count >= MIN_BARS else f"{count}/{MIN_BARS}"
        parts.append(f"{timeframe} {ok}")
    print(f"  {ticker}: {len(bars)} daily bars from "
          f"{bars.index[0].date()} — {', '.join(parts)}")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--range", default=DEFAULT_RANGE,
                        help=f"yfinance period string (default {DEFAULT_RANGE})")
    parser.add_argument("--tickers",
                        help="comma-separated override for the watchlist")
    parser.add_argument("--no-rate", action="store_true",
                        help="fetch bars only; skip recomputing ratings")
    args = parser.parse_args()

    db = Database()
    db.initialize()

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = db.get_tracked_tickers()

    if not tickers:
        print("No tickers to backfill.")
        return 1

    print(f"Backfilling {len(tickers)} ticker(s) at period={args.range}\n")

    stored = failed = 0
    loop = asyncio.get_running_loop()

    for ticker in tickers:
        try:
            rows = await loop.run_in_executor(None, fetch_history, ticker, args.range)
        except Exception as e:
            # One delisted or throttled symbol must not cost the rest of the
            # watchlist its backfill.
            print(f"  {ticker}: FAILED — {type(e).__name__}: {e}")
            failed += 1
            await asyncio.sleep(TICKER_DELAY)
            continue

        if not rows:
            print(f"  {ticker}: no data returned")
            failed += 1
            await asyncio.sleep(TICKER_DELAY)
            continue

        db.upsert_price_history(ticker, rows)
        stored += len(rows)
        report_depth(db, ticker)
        await asyncio.sleep(TICKER_DELAY)

    print(f"\nStored {stored} bars; {failed} ticker(s) failed.")

    if args.no_rate:
        return 0

    print("\nRecomputing technical ratings...")
    totals = await TechnicalRatingTracker(db).compute_all(tickers)
    print(f"  {totals['rows']} rating rows across "
          f"{totals['tickers']} ticker(s); "
          f"{totals['insufficient']} still short on history, "
          f"{totals['failed']} failed.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
