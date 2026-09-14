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

The direction model (pipeline.features) trains on more than the watchlist, so
`--universe` selects its symbol sets instead of the tracked list:

    core      tracked + the default watchlist (minus BTC-USD) + the eleven
              sector ETFs + the market inputs (SPY, ^GSPC, ^VIX, ^TNX)
    backbone  core + TRAINING_BACKBONE

`--splits` also stores each ticker's split history in price_splits. Run it
whenever bars are backfilled for the model: the features use the split table to
find where a stored history switches from one split adjustment to another.

Usage:
    python scripts/manual/backfill_price_history.py
    python scripts/manual/backfill_price_history.py --range 10y
    python scripts/manual/backfill_price_history.py --tickers NVDA,AMD
    python scripts/manual/backfill_price_history.py --universe core --splits --range max
    python scripts/manual/backfill_price_history.py --no-rate
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.logging_config import get_logger
from data.database import Database
from data.watchlist import (
    DEFAULT_WATCHLIST,
    MARKET_INPUT_TICKERS,
    SECTOR_ETFS,
    TRAINING_BACKBONE,
)
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

UNIVERSES = ("core", "backbone")

# Trades around the clock and has no place in an equity model's training rows.
_EXCLUDED_FROM_UNIVERSE = {"BTC-USD"}


def resolve_universe(db: Database, name: str) -> list[str]:
    """The symbols a named model universe backfills, sorted."""
    tracked = db.get_tracked_tickers() or []
    symbols = {t.strip().upper() for t in (*tracked, *DEFAULT_WATCHLIST) if t and t.strip()}
    symbols -= _EXCLUDED_FROM_UNIVERSE
    symbols |= set(SECTOR_ETFS) | set(MARKET_INPUT_TICKERS)
    if name == "backbone":
        symbols |= set(TRAINING_BACKBONE)
    return sorted(symbols)


def fetch_history(ticker: str, period: str,
                  with_splits: bool = False) -> tuple[list[dict], list[dict]]:
    """Full daily OHLCV for one ticker as storage rows, plus its splits. Blocking.

    Returns (rows, splits); splits is empty unless `with_splits`.
    """
    import yfinance as yf

    handle = yf.Ticker(ticker)
    # auto_adjust=False keeps the close unadjusted for DIVIDENDS, which matches
    # what PriceFeed stores from the chart endpoint. It does not undo split
    # adjustment: Yahoo serves every bar already split-adjusted as of the day it
    # is fetched, which is why the split events are stored separately (see
    # price_splits) rather than assumed absent. Mixing dividend conventions in
    # one table is the kind of thing that is invisible until a ratio comes out
    # wrong, so both writers agree on this one.
    frame = handle.history(period=period, auto_adjust=False)
    if frame is None or frame.empty:
        return [], []

    frame.columns = [c.lower() for c in frame.columns]
    splits: list[dict] = []
    if with_splits:
        # history() already carries the split column (actions=True), so the
        # events come off the same download; .splits is the fallback for a
        # frame without it and costs a second request.
        events = frame["stock splits"] if "stock splits" in frame.columns else handle.splits
        if events is not None and len(events):
            splits = [
                {"date": stamp.strftime("%Y-%m-%d"), "ratio": float(ratio)}
                for stamp, ratio in events.items()
                if ratio and float(ratio) > 0 and float(ratio) != 1.0
            ]

    frame = frame.dropna(subset=["close"])
    rows = [
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
    return rows, splits


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
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--range", default=DEFAULT_RANGE,
                        help=f"yfinance period string (default {DEFAULT_RANGE})")
    which = parser.add_mutually_exclusive_group()
    which.add_argument("--tickers",
                       help="comma-separated override for the watchlist")
    which.add_argument("--universe", choices=UNIVERSES,
                       help="backfill a model universe instead of the tracked list")
    parser.add_argument("--splits", action="store_true",
                        help="also store each ticker's split history in price_splits")
    parser.add_argument("--no-rate", action="store_true",
                        help="fetch bars only; skip recomputing ratings")
    args = parser.parse_args()

    db = Database()
    db.initialize()

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    elif args.universe:
        tickers = resolve_universe(db, args.universe)
    else:
        tickers = db.get_tracked_tickers()

    if not tickers:
        print("No tickers to backfill.")
        return 1

    print(f"Backfilling {len(tickers)} ticker(s) at period={args.range}"
          f"{' with splits' if args.splits else ''}\n")

    stored = failed = split_rows = 0
    loop = asyncio.get_running_loop()

    for ticker in tickers:
        try:
            rows, splits = await loop.run_in_executor(
                None, fetch_history, ticker, args.range, args.splits)
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
        if splits:
            split_rows += db.upsert_price_splits(ticker, splits)
            print(f"  {ticker}: {len(splits)} split(s), latest "
                  f"{splits[-1]['date']} x{splits[-1]['ratio']:g}")
        report_depth(db, ticker)
        await asyncio.sleep(TICKER_DELAY)

    print(f"\nStored {stored} bars and {split_rows} split rows; "
          f"{failed} ticker(s) failed.")

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
