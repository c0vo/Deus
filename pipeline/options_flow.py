"""
Daily option-chain snapshots — options positioning, not options flow.

The distinction matters and is worth stating plainly, because the two are
routinely conflated. Real "options flow" — sweeps, blocks, and the buy/sell
aggressor tagging that makes them readable as sentiment — requires OPRA
trade-by-trade data with exchange and condition codes plus the NBBO at trade
time. That is a licensed feed with a redistributor fee in the four figures per
month, and no free tier exists at any provider. It is not obtainable here.

What IS free is positioning: how much volume and open interest sit on puts
versus calls, and what the implied-volatility surface looks like. That comes
from yfinance's chain endpoint and answers a narrower question — "how is the
options market currently positioned" rather than "who just bought what" — but
it is the part with published predictive evidence behind it anyway.

Two constraints shape everything below:

  No history.  yfinance returns only the current chain. There is no way to ask
               for last Tuesday, so this panel accrues one session per day and
               a missed session is gone permanently. That is why the collector
               runs from day one even though no feature reads the table until
               FEATURE_SCHEMA_VERSION 4.

  No Greeks.   Delta/gamma/theta/vega are absent, so skew is approximated by
               moneyness buckets rather than by delta. Cboe publishes Greeks
               but explicitly prohibits programmatic extraction, so that route
               is closed regardless of it being technically reachable.

Implied volatility from this source needs care. Yahoo's IV is unusable on
deep-ITM and illiquid strikes — a deep-ITM AAPL 260 call with spot at 313
quotes 133% IV — so every IV figure here is computed under the filter in
_usable_iv(): out-of-the-money, non-zero bid, some open interest or volume,
and within MONEYNESS_BAND of spot. Without it a single garbage quote drags the
average far enough to invert the sign of any skew measure.

This is not a NewsSource — see the pipeline.darkpool docstring for why.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo

from config.logging_config import get_logger
from config.settings import settings
from data.database import Database
from data.tickers import US, classify_market

log = get_logger(__name__)

# Days-to-expiry the snapshot aims for, one chain fetched per target.
#
# Targeting DTE rather than taking the first N expirations is the difference
# between a term structure and a rounding error: liquid names list weeklies, so
# "the first four" on AAPL spans 1 to 12 days and the near/far legs measure
# essentially the same thing. These four span a quarter. The ~30d leg is the
# reference for ATM IV and skew, matching the convention VIX-style measures use.
TARGET_DTES = (7, 30, 60, 90)
SKEW_TARGET_DTE = 30

# Seconds between tickers. yfinance is an unofficial scraper of Yahoo endpoints
# and blocks IPs on bursts; a block here would also break the predictor's price
# fetching, which shares the same host. Serial with a delay, never gather().
TICKER_DELAY = 2.0

# Strikes further than this from spot are excluded from IV aggregates. Wide
# enough to keep a real skew measurement, tight enough to drop the illiquid
# wings where Yahoo's IV is noise.
MONEYNESS_BAND = 0.25

# Moneyness offset used for the skew legs: IV of puts ~10% below spot minus IV
# of calls ~10% above. A delta-based definition would be better but there are
# no Greeks in this feed.
SKEW_OFFSET = 0.10

# The session a post-close snapshot belongs to is the US trading day that just
# ended, which is not the UTC date for most of the scheduled window.
_ET = ZoneInfo("America/New_York")


def _num(value: Any) -> float:
    """NaN-safe float coercion.

    yfinance returns pandas NaN — not None — for volume on contracts that did
    not trade, and NaN is truthy in Python. So `row.get("volume") or 0.0` keeps
    the NaN, one NaN poisons the running total, and every volume figure for the
    ticker lands as NaN with the ratios silently falling back to None.
    """
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if out != out else out  # NaN is the only value unequal to itself


def _mean(values: list[float]) -> Optional[float]:
    return (sum(values) / len(values)) if values else None


def _dte(expiry: str, today: Optional[date] = None) -> int:
    """Days to expiry for a 'YYYY-MM-DD' string, or -1 if unparseable."""
    try:
        return (date.fromisoformat(expiry) - (today or datetime.now(_ET).date())).days
    except (TypeError, ValueError):
        return -1


class OptionsSnapshotTracker:
    """Captures one option-chain aggregate row per ticker per session."""

    def __init__(self, db: Database, target_dtes: tuple[int, ...] = TARGET_DTES):
        self.db = db
        self.target_dtes = target_dtes

    @property
    def enabled(self) -> bool:
        return settings.options_snapshot_enabled

    # ── Capture ──────────────────────────────────────────────────────────

    async def snapshot_all(self, tickers: list[str]) -> dict[str, int]:
        """Snapshot every US ticker in the list, one at a time.

        Serial by design. yfinance is synchronous and blocking, so each call is
        pushed to the default executor; running them concurrently would both
        saturate that pool — which the predictor also uses — and present Yahoo
        with exactly the burst pattern that gets an IP blocked.
        """
        totals = {"tickers": 0, "rows": 0, "failed": 0}
        if not self.enabled:
            log.info("options.skipped_disabled")
            return totals

        # Yahoo keeps serving the last session's chain over the weekend, so a
        # Saturday run would store Friday's numbers under a Saturday date and
        # do it again on Sunday — two junk rows per week, 29% of the panel,
        # each looking like a real session to anything reading the table later.
        session = datetime.now(timezone.utc).astimezone(_ET).date()
        if session.weekday() >= 5:
            log.info("options.skipped_weekend", session=session.isoformat())
            return totals

        loop = asyncio.get_running_loop()
        rows: list[dict[str, Any]] = []

        for ticker in tickers:
            if classify_market(ticker) != US:
                continue
            try:
                row = await loop.run_in_executor(None, self._snapshot_sync, ticker)
            except Exception as e:
                log.error("options.snapshot_failed", ticker=ticker,
                          error=str(e) or repr(e), error_type=type(e).__name__)
                totals["failed"] += 1
                continue

            totals["tickers"] += 1
            if row is None:
                totals["failed"] += 1
            else:
                rows.append(row)
            await asyncio.sleep(TICKER_DELAY)

        totals["rows"] = self.db.upsert_option_chain_daily(rows)
        log.info("options.snapshot_complete", **totals)
        return totals

    def _snapshot_sync(self, ticker: str) -> Optional[dict[str, Any]]:
        """Blocking yfinance read for one ticker. Runs in an executor."""
        import yfinance as yf

        ticker = ticker.upper().strip()
        handle = yf.Ticker(ticker)

        try:
            expirations = list(handle.options or [])
        except Exception as e:
            log.warning("options.expirations_failed", ticker=ticker,
                        error=str(e) or repr(e), error_type=type(e).__name__)
            return None
        if not expirations:
            log.info("options.no_chain", ticker=ticker)
            return None

        chains: list[tuple[int, Any, Any]] = []
        spot: Optional[float] = None

        for expiry in self.select_expirations(expirations, self.target_dtes):
            try:
                chain = handle.option_chain(expiry)
            except Exception as e:
                log.warning("options.chain_failed", ticker=ticker, expiry=expiry,
                            error=str(e) or repr(e), error_type=type(e).__name__)
                continue
            if spot is None:
                # The chain response carries the underlying quote, so this costs
                # nothing extra — no second request for a price.
                underlying = getattr(chain, "underlying", None) or {}
                raw = underlying.get("regularMarketPrice")
                spot = float(raw) if raw else None
            chains.append((_dte(expiry), chain.calls, chain.puts))

        if not chains:
            return None

        return self.aggregate(ticker, spot, chains)

    @staticmethod
    def select_expirations(expirations: list[str],
                           targets: tuple[int, ...] = TARGET_DTES) -> list[str]:
        """Pick the listed expiry nearest each target DTE, deduplicated.

        Returns fewer than len(targets) entries when the same expiry is nearest
        to two targets, which is normal for thinly listed names — one request
        saved rather than the same chain fetched twice.
        """
        dated = [(e, _dte(e)) for e in expirations]
        dated = [(e, d) for e, d in dated if d >= 0]
        if not dated:
            return []

        chosen: list[str] = []
        for target in targets:
            nearest = min(dated, key=lambda pair: abs(pair[1] - target))[0]
            if nearest not in chosen:
                chosen.append(nearest)
        return chosen

    # ── Aggregation ──────────────────────────────────────────────────────

    @classmethod
    def aggregate(cls, ticker: str, spot: Optional[float],
                  chains: list[tuple[int, Any, Any]]) -> dict[str, Any]:
        """Fold captured chains into one storage row.

        Chains arrive as (dte, calls, puts). Takes DataFrames — or anything
        exposing the same columns — so the aggregation is testable against
        synthetic frames without a network call.
        """
        chains = sorted(chains, key=lambda c: c[0])
        near_dte = chains[0][0]
        far_dte = chains[-1][0]
        # The reference leg for ATM IV and skew: whichever captured expiry sits
        # closest to SKEW_TARGET_DTE. Front weeklies are the wrong place to
        # measure skew — their wings have no bid, so both legs filter out and
        # the result is None on most sessions.
        ref_dte = min((c[0] for c in chains), key=lambda d: abs(d - SKEW_TARGET_DTE))

        call_volume = put_volume = call_oi = put_oi = 0.0
        contracts = 0
        near_ivs: list[float] = []
        far_ivs: list[float] = []
        ref_ivs: list[float] = []
        skew_put_ivs: list[float] = []
        skew_call_ivs: list[float] = []

        for dte, calls, puts in chains:
            for frame, is_call in ((calls, True), (puts, False)):
                for row in cls._rows(frame):
                    contracts += 1
                    volume = _num(row.get("volume"))
                    oi = _num(row.get("openInterest"))
                    if is_call:
                        call_volume += volume
                        call_oi += oi
                    else:
                        put_volume += volume
                        put_oi += oi

                    iv = cls._usable_iv(row, spot, is_call)
                    if iv is None:
                        continue
                    if dte == near_dte:
                        near_ivs.append(iv)
                    if dte == far_dte:
                        far_ivs.append(iv)
                    if dte != ref_dte:
                        continue

                    ref_ivs.append(iv)
                    if not spot:
                        continue
                    moneyness = (_num(row.get("strike")) - spot) / spot
                    if not is_call and moneyness <= -SKEW_OFFSET:
                        skew_put_ivs.append(iv)
                    elif is_call and moneyness >= SKEW_OFFSET:
                        skew_call_ivs.append(iv)

        put_leg, call_leg = _mean(skew_put_ivs), _mean(skew_call_ivs)
        now = datetime.now(timezone.utc)

        return {
            "ticker": ticker,
            # The US session that just closed, not the UTC date. The job runs at
            # 06:00 KST = 21:00 UTC the previous day, so a UTC date would label
            # roughly half the year's snapshots with the wrong session.
            "session_date": now.astimezone(_ET).date().isoformat(),
            "spot_price": spot,
            "call_volume": call_volume,
            "put_volume": put_volume,
            "call_oi": call_oi,
            "put_oi": put_oi,
            "put_call_volume_ratio": (put_volume / call_volume) if call_volume > 0 else None,
            "put_call_oi_ratio": (put_oi / call_oi) if call_oi > 0 else None,
            "atm_iv": _mean(ref_ivs),
            "iv_skew": (put_leg - call_leg) if (put_leg is not None and call_leg is not None) else None,
            "near_term_iv": _mean(near_ivs),
            "far_term_iv": _mean(far_ivs),
            "expirations_seen": len(chains),
            "contracts_seen": contracts,
            "published_at": now.isoformat(),
            "source": "yfinance",
        }

    @staticmethod
    def _rows(frame: Any) -> list[dict]:
        """DataFrame -> list of plain dicts, tolerating an empty or absent frame."""
        if frame is None:
            return []
        try:
            if hasattr(frame, "to_dict"):
                return frame.to_dict("records")
            return list(frame)
        except Exception:
            return []

    @staticmethod
    def _usable_iv(row: dict, spot: Optional[float], is_call: bool) -> Optional[float]:
        """Implied vol for one contract, or None if the quote cannot be trusted.

        Four filters, each earning its place against observed Yahoo data:

          out-of-the-money   ITM quotes carry most of the garbage. The deep-ITM
                             AAPL 260 call quoting 133% IV against a 313 spot is
                             the canonical example.
          non-zero bid       A contract nobody will bid on has no real price, so
                             its back-solved vol is meaningless.
          traded or held     Zero volume AND zero open interest means the quote
                             is a market-maker placeholder.
          near the money     The far wings are illiquid enough that their IV is
                             noise even when the filters above pass.
        """
        iv = _num(row.get("impliedVolatility"))
        if not (0.0 < iv < 5.0):
            return None
        if row.get("inTheMoney"):
            return None
        if _num(row.get("bid")) <= 0:
            return None
        if _num(row.get("volume")) <= 0 and _num(row.get("openInterest")) <= 0:
            return None

        strike = _num(row.get("strike"))
        if spot and strike and abs(strike - spot) / spot > MONEYNESS_BAND:
            return None
        return iv

    # ── Reporting ────────────────────────────────────────────────────────

    def get_summary(self, ticker: str) -> dict[str, Any]:
        """Latest stored snapshot for one ticker, with its trailing context."""
        series = self.db.get_option_chain_series(ticker)
        if not series:
            return {"ticker": ticker.upper(), "sessions": 0, "latest": None}
        return {
            "ticker": ticker.upper(),
            "sessions": len(series),
            "latest": series[-1],
            "first_session": series[0]["session_date"],
        }
