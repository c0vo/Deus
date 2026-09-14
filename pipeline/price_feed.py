"""
Price Feed Component

Keeps the `latest_prices` and `price_history` tables warm so neither the API
nor the dark-pool join ever waits on Yahoo.

/api/markets used to call yfinance once per tracked ticker per request, and the
ticker tape polls it from every open browser tab. That meant a dozen live HTTP
round-trips every few seconds — enough to make the endpoint take seconds on a
phone, and enough for Yahoo to start throttling, at which point the blocked
calls tied up executor threads and slowed the rest of the API with them.

This runs in the worker process instead, on a timer. It reuses the same Yahoo
chart endpoint as market_scanner.py rather than yfinance: it is a plain async
httpx call, so the whole watchlist refreshes concurrently without occupying a
thread each.

Two jobs live here, against that same endpoint:

  refresh()          last-known quote per ticker -> latest_prices, every few
                     minutes, so the dashboard tape stays live.
  refresh_history()  daily OHLCV bars and split events per ticker ->
                     price_history / price_splits, once a day.

refresh_history exists because price_history had no owner. Its only writer was
Predictor._fetch_and_cache_prices, which runs when a prediction or a retrain
happens, so a ticker's volume history froze the moment it stopped being
predicted. That silently broke the dark-pool card: off-exchange share is
off-exchange volume / consolidated volume, the FINRA half of which updates
daily for the whole watchlist while the consolidated half comes from here. A
ticker last predicted six weeks ago kept accruing FINRA rows that could never
join to a price row, so its card showed a dash and an empty chart while looking
perfectly healthy.

Two properties of Yahoo's chart endpoint matter to everything reading these
tables:

  * Its bars are split-adjusted as of the moment they are fetched. Every write
    here is INSERT OR REPLACE over a trailing window, so after a split the
    stored history holds re-fetched rows at the new scale next to older rows
    still at the old one. The split events are stored alongside so the feature
    builder can find that seam; the rows themselves are never rewritten.
  * `range=max` silently downgrades `interval=1d` to MONTHLY bars. A "max" pull
    is therefore requested as an explicit period1/period2 window, which returns
    true dailies, and any response whose dataGranularity is not 1d is refused
    rather than stored — monthly rows mixed into a daily table poison every
    rolling window that spans them.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import httpx

from config.logging_config import get_logger
from data.database import Database
from data.watchlist import (
    DEFAULT_WATCHLIST,
    MARKET_INPUT_TICKERS,
    SECTOR_ETFS,
    TRAINING_BACKBONE,
)

log = get_logger(__name__)

# Mirrors the fallback list /api/markets uses when nothing is tracked yet, so
# that path has real quotes too.
FALLBACK_TICKERS = ["AAPL", "MSFT", "GOOGL"]

YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
MAX_CONCURRENT_FETCHES = 4

# Daily bars pulled per ticker. Deliberately far wider than the one new session
# a day this needs: the upsert is INSERT OR REPLACE, so re-reading a bar costs
# nothing and every run also repairs what earlier runs missed. A ticker just
# added to the watchlist, or a deployment that was powered off for a fortnight,
# closes its own gap on the next run instead of needing a manual backfill.
#
# Widened from 3mo when technical ratings landed. The rating needs 200-period
# moving averages on daily, weekly and monthly bars, and 3mo is ~63 sessions —
# not enough for a single timeframe, let alone the resampled ones. 2y keeps the
# daily rating working from a cold start; the weekly and monthly legs need the
# deep one-off pull in scripts/manual/backfill_price_history.py.
HISTORY_RANGE = "2y"

# The one range value Yahoo answers with monthly bars at interval=1d. Requests
# for it are rewritten to an explicit epoch window; see _chart_params.
MAX_RANGE = "max"
DAILY_GRANULARITY = "1d"

# The quote payload is a handful of numbers; three months of daily bars is
# ~60 rows x 6 arrays, and Termux fetches it over a phone connection.
HISTORY_TIMEOUT = 20.0


def _at(values: Optional[list], index: int) -> Optional[Any]:
    """One element of a Yahoo indicator array, or None if absent.

    The OHLCV arrays run parallel to `timestamp`, but Yahoo pads untraded
    sessions with null and occasionally returns one array shorter than the
    others, so neither the index nor the value can be assumed present.
    """
    if not values or index >= len(values):
        return None
    return values[index]


def _chart_params(history_range: str) -> dict[str, Any]:
    """Query parameters for a daily-bar chart request over `history_range`.

    Every range is passed through except "max", which Yahoo answers with one
    bar per MONTH despite interval=1d. period1=0/period2=now asks for the same
    span explicitly and gets true daily bars back.
    """
    params: dict[str, Any] = {"interval": DAILY_GRANULARITY, "events": "splits"}
    if history_range == MAX_RANGE:
        params["period1"] = 0
        params["period2"] = int(datetime.now(timezone.utc).timestamp())
    else:
        params["range"] = history_range
    return params


def _parse_splits(result: dict, offset: timedelta) -> list[dict]:
    """The chart payload's split events as `{date, ratio}` rows, oldest first.

    `ratio` is numerator / denominator — new shares per old share, so 4.0 for a
    4:1 split and 0.1 for a 1:10 reverse split. The event's own `date` stamp is
    used, not its dict key: on a coarse-granularity response the key is the
    containing bar's timestamp, not the split's. The date is read in the
    exchange's timezone for the same reason the bars are.
    """
    events = (result.get("events") or {}).get("splits") or {}
    if isinstance(events, dict):
        events = list(events.values())

    splits = []
    for event in events:
        if not isinstance(event, dict):
            continue
        stamp = event.get("date")
        try:
            ratio = float(event.get("numerator")) / float(event.get("denominator"))
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        if stamp is None or not ratio > 0:
            continue
        day = datetime.fromtimestamp(stamp, tz=timezone.utc) + offset
        splits.append({"date": day.strftime("%Y-%m-%d"), "ratio": ratio})
    splits.sort(key=lambda s: s["date"])
    return splits


class PriceFeed:
    """Refreshes last-known quotes and daily bars for every tracked ticker."""

    def __init__(self, db: Database):
        self.db = db

    def universe(self) -> list[str]:
        """
        Every symbol this feed keeps warm: tracked plus the default watchlist.

        The default names used to be absent, which is what made the market
        scanner fetch Yahoo itself: SPY and QQQ were never in `latest_prices`
        unless somebody happened to track them, so there was no stored session
        context to read a drop against. Now the two tables the scanner and the
        dashboard read are the only thing either of them needs.

        FALLBACK_TICKERS is still applied when nothing is tracked, because
        /api/markets falls back to exactly that list and GOOGL is not in the
        default watchlist — without it that cold-start path shows a dash.
        """
        tracked = self.db.get_tracked_tickers() or []
        symbols = set(tracked) | set(DEFAULT_WATCHLIST)
        if not tracked:
            symbols |= set(FALLBACK_TICKERS)
        return sorted(symbols)

    def history_universe(self) -> list[str]:
        """
        Every symbol whose daily bars are kept: universe() plus model inputs.

        Wider than universe() because the direction model trains on more than
        the watchlist (the backbone names and the sector ETFs) and reads market
        series nobody tracks (SPY or ^GSPC, ^VIX, ^TNX). Deliberately not used
        by refresh(): quotes are polled on a short timer for the dashboard tape
        and bars once a day, and polling quotes for several times as many
        symbols would buy the tape nothing it renders.
        """
        symbols = (
            set(self.universe())
            | set(TRAINING_BACKBONE)
            | set(SECTOR_ETFS)
            | set(MARKET_INPUT_TICKERS)
        )
        return sorted(symbols)

    async def refresh(self) -> int:
        """Fetch the whole universe and persist the results. Returns the count stored."""
        tickers = self.universe()
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_FETCHES)

        async with httpx.AsyncClient(timeout=10) as client:
            results = await asyncio.gather(
                *(self._fetch_quote(client, semaphore, t) for t in tickers),
                return_exceptions=True,
            )

        quotes = [r for r in results if isinstance(r, dict)]
        failures = len(results) - len(quotes)

        stored = await asyncio.to_thread(self.db.upsert_latest_prices, quotes)
        log.info("price_feed.refreshed", stored=stored, failed=failures)
        return stored

    async def refresh_history(self) -> int:
        """Fetch daily OHLCV and splits for history_universe() into the database.

        Returns the number of bar rows written. Each ticker is stored as its own
        upsert rather than one batch at the end, so a symbol Yahoo refuses
        cannot cost the rest of the watchlist its update.
        """
        tickers = self.history_universe()
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_FETCHES)

        async with httpx.AsyncClient(timeout=HISTORY_TIMEOUT) as client:
            results = await asyncio.gather(
                *(self._fetch_history(client, semaphore, t) for t in tickers),
                return_exceptions=True,
            )

        stored, failed, splits = await self._persist_history(tickers, results)
        log.info("price_feed.history_refreshed",
                 tickers=len(tickers) - failed, rows=stored, failed=failed,
                 splits=splits)
        return stored

    async def _persist_history(self, tickers: list[str],
                               results: list[Any]) -> tuple[int, int, int]:
        """Store each ticker's fetched (rows, splits); returns (rows, failed, splits).

        A result that is None, an exception from gather(), or an empty bar list
        counts as a failed ticker and writes nothing.
        """
        stored = failed = split_rows = 0
        for ticker, fetched in zip(tickers, results):
            if not isinstance(fetched, tuple) or len(fetched) != 2 or not fetched[0]:
                failed += 1
                continue
            rows, splits = fetched
            await asyncio.to_thread(self.db.upsert_price_history, ticker, rows)
            stored += len(rows)
            if splits:
                await asyncio.to_thread(self.db.upsert_price_splits, ticker, splits)
                split_rows += len(splits)
        return stored, failed, split_rows

    async def refresh_history_for(
        self, tickers: list[str], history_range: str = "1y"
    ) -> int:
        """Fetch daily OHLCV and splits for an explicit ticker list.

        refresh_history() covers a fixed universe over HISTORY_RANGE. Thesis
        candidates are off-watchlist by construction, and the crowding score
        needs a year of bars to know where a name sits against its 52-week
        high, so both of those assumptions have to be overridable.
        `history_range="max"` fetches the full daily history (see _chart_params).
        """
        symbols = [t.strip().upper() for t in tickers if t and t.strip()]
        if not symbols:
            return 0
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_FETCHES)

        async with httpx.AsyncClient(timeout=HISTORY_TIMEOUT) as client:
            results = await asyncio.gather(
                *(self._fetch_history(client, semaphore, t, history_range)
                  for t in symbols),
                return_exceptions=True,
            )

        stored, failed, splits = await self._persist_history(symbols, results)
        log.info("price_feed.history_refreshed_for",
                 tickers=len(symbols) - failed, rows=stored, failed=failed,
                 splits=splits)
        return stored

    async def _fetch_quote(
        self,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        ticker: str,
    ) -> Optional[dict]:
        """Fetch one ticker's last two daily closes, plus the last bar's volume."""
        async with semaphore:
            try:
                resp = await client.get(
                    YAHOO_CHART_URL.format(ticker=ticker),
                    params={"range": "5d", "interval": "1d"},
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                resp.raise_for_status()
                data = resp.json()

                chart = data.get("chart", {}).get("result", [])
                if not chart:
                    return None

                quote = chart[0].get("indicators", {}).get("quote", [{}])[0]
                closes = quote.get("close") or []
                # Index-aligned rather than compacted: the volume has to come
                # from the same bar as the close, and compacting the close array
                # first (as this did) throws away the index that says which one.
                traded = [i for i, c in enumerate(closes) if c is not None]
                if not traded:
                    return None

                last = traded[-1]
                current = float(closes[last])
                previous = (
                    float(closes[traded[-2]]) if len(traded) >= 2 else current
                )
                change_pct = (
                    ((current - previous) / previous * 100) if previous else 0.0
                )
                # The live session's running volume, which is what an anomalous
                # volume alert compares against the 20-session average. None
                # when Yahoo left it null — 0 would read as "no volume today".
                volume = _at(quote.get("volume"), last)
                return {
                    "ticker": ticker,
                    "price": current,
                    "previous_close": previous,
                    "daily_change_pct": change_pct,
                    "volume": float(volume) if volume is not None else None,
                }
            except Exception as e:
                log.warning("price_feed.fetch_failed", ticker=ticker, error=str(e))
                return None

    async def _fetch_history(
        self,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        ticker: str,
        history_range: str = HISTORY_RANGE,
    ) -> Optional[tuple[list[dict], list[dict]]]:
        """Fetch one ticker's daily OHLCV bars and split events, oldest first.

        Returns (rows, splits), or None when the fetch failed or was refused.
        """
        fetched = await self._fetch_chart(client, semaphore, ticker, history_range)
        return (fetched[1], fetched[2]) if fetched else None

    async def _fetch_chart(
        self,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        ticker: str,
        history_range: str = HISTORY_RANGE,
    ) -> Optional[tuple[dict, list[dict], list[dict]]]:
        """Fetch one ticker's chart payload as (meta, rows, splits), oldest first.

        The `meta` block carries symbol, longName, exchangeName and currency,
        which is everything needed to confirm a symbol actually exists and is
        the company it claims to be — so ticker resolution and the price
        warm-up are one request rather than two. `splits` are the split events
        inside the requested window, as `{date, ratio}`.

        Returns None on failure rather than raising: this backs a scheduled job
        over the whole watchlist, and one delisted or throttled symbol must not
        abort the others. A response whose bars are not daily is a failure too
        (see the module docstring).
        """
        async with semaphore:
            try:
                resp = await client.get(
                    YAHOO_CHART_URL.format(ticker=ticker),
                    params=_chart_params(history_range),
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                resp.raise_for_status()
                data = resp.json()

                chart = data.get("chart", {}).get("result", [])
                if not chart:
                    return None
                result = chart[0]

                # Absent on old or hand-built payloads; only an explicit
                # non-daily answer is refused.
                granularity = (result.get("meta") or {}).get("dataGranularity")
                if granularity and granularity != DAILY_GRANULARITY:
                    log.warning("price_feed.non_daily_granularity",
                                ticker=ticker, granularity=granularity,
                                history_range=history_range)
                    return None

                stamps = result.get("timestamp") or []
                quote = (result.get("indicators", {}).get("quote") or [{}])[0]

                # Yahoo stamps each daily bar at that exchange's opening bell,
                # so the session date has to be read in the exchange's own
                # timezone. Reading it in UTC — or worse in the host's local
                # zone, which is Asia/Seoul on the deployment target — happens
                # to land on the right day for a US 09:30 open and will not for
                # every listing. This date is the join key against FINRA's
                # session_date, so an off-by-one yields a NULL consolidated
                # volume and a blank card, never an error anyone would notice.
                offset = timedelta(
                    seconds=result.get("meta", {}).get("gmtoffset") or 0
                )

                rows = []
                for i, stamp in enumerate(stamps):
                    close = _at(quote.get("close"), i)
                    if stamp is None or close is None:
                        continue  # untraded session, or a bar Yahoo left null
                    volume = _at(quote.get("volume"), i)
                    session = datetime.fromtimestamp(stamp, tz=timezone.utc) + offset
                    rows.append({
                        "date": session.strftime("%Y-%m-%d"),
                        "open": _at(quote.get("open"), i),
                        "high": _at(quote.get("high"), i),
                        "low": _at(quote.get("low"), i),
                        "close": float(close),
                        # volume is the whole point of this table for the
                        # dark-pool ratio, and 0 reads downstream as "unusable"
                        # exactly like a missing row — so a null is left as 0
                        # rather than becoming something that would divide.
                        "volume": int(volume) if volume is not None else 0,
                    })

                rows.sort(key=lambda r: r["date"])
                return result.get("meta", {}) or {}, rows, _parse_splits(result, offset)
            except Exception as e:
                log.warning("price_feed.history_fetch_failed",
                            ticker=ticker, error=str(e))
                return None
