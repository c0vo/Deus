"""
Multi-timeframe technical ratings — the Webull/TradingView bull-bear gauge.

This is arithmetic, not analysis. Twenty-six indicators each cast one vote of
+1/0/-1 under a fixed rule, the votes are averaged per group, and the result is
bucketed into Strong Sell .. Strong Buy. There is no discretion anywhere in it
and no LLM involved: the same bars always produce the same rating, which is why
every rule below is a named constant or an explicit expression rather than a
branch buried in a loop.

The formula is TradingView's own `TechnicalRating` Pine library v3 (MPL-2.0) —
the code behind the Screener's Rating column, not a reproduction of it:

    https://www.tradingview.com/script/jDWyb5PG-TechnicalRating

Four rules differ from the version most third-party ports implement, each
verified against that source. They are called out at their definitions below,
but collected here because getting any of them wrong yields a plausible number
rather than a visible failure:

  Ichimoku    is the full four-part cloud with 26-bar displaced leading spans,
              NOT base-line-versus-price.
  CCI         is computed on `close`. TradingView's standalone CCI indicator
              defaults to hlc3; the rating does not.
  ADX         requires a RISING adx (adx > adx[1]) on both legs. It is not a
              DI crossover, and TradingView's own help page contradicts its
              source here — the source wins.
  BullBear    uses `bear` for the buy leg and `bull` for the sell leg
              separately. The rating never uses their sum.

Indicators are hand-rolled in pandas rather than taken from a library. `ta`
lacks VWMA, Hull MA and Bull Bear Power outright; `pandas-ta`'s upstream repo
was deleted and it pins a numba that will not install here; TA-Lib ships
manylinux aarch64 wheels, which do not match Termux's bionic libc, so pip falls
back to an sdist build that fails there. Each indicator is a handful of lines,
so a dependency would buy nothing and cost the Android deployment.

Unlike every other collector in this package, nothing here touches the network.
A rating is a pure function of stored OHLCV, which also makes it the only one of
the three analyst-panel signals that can be backfilled.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from config.logging_config import get_logger
from config.settings import settings
from data.database import Database

log = get_logger(__name__)

# ── Rating buckets ───────────────────────────────────────────────────────────
# Strict inequalities on both sides, so a score of exactly +/-0.1 or +/-0.5 is
# the *weaker* label. Matches `ratingStatus` in the Pine library.
STRONG_BUY = "STRONG_BUY"
BUY = "BUY"
NEUTRAL = "NEUTRAL"
SELL = "SELL"
STRONG_SELL = "STRONG_SELL"

RATING_STRONG = 0.5
RATING_WEAK = 0.1

# ── Timeframes ───────────────────────────────────────────────────────────────
# The same formula on resampled bars, which is what TradingView does per chart
# interval. Keys are the stored `timeframe` column; the value is the pandas
# resample rule, or None for the daily bars as they are already stored.
TIMEFRAMES: dict[str, Optional[str]] = {
    "short": None,      # daily
    "medium": "W-FRI",  # weekly, Friday-anchored to match a US trading week
    "long": "ME",       # monthly, month-end
}

# Bars required before a rating is emitted at all.
#
# The binding constraint is SMA/EMA(200); Ichimoku's displaced span needs 78.
# TradingView renders partial ratings by dropping indicators that have not
# warmed up, which quietly shrinks the denominator and makes two ratings on the
# same page incomparable. This refuses instead: below the threshold there is no
# row, and `bars_available` says why.
#
# Consequence worth stating plainly: `long` needs ~200 months of daily history,
# so it will be absent for most tickers and permanently absent for anything
# listed recently. That is correct behaviour, not a gap to paper over.
MIN_BARS = 280

MA_LENGTHS = (10, 20, 30, 50, 100, 200)

# Trend gate for the two oscillators that have one. Pine: priceAvg = ema(close, 50).
TREND_EMA = 50

# The session a rating belongs to is the US trading day, not the UTC date — the
# job runs at 08:05 KST, which is the previous evening in New York.
_ET = ZoneInfo("America/New_York")

# Seconds between tickers. Nothing here hits the network, but each ticker is a
# few hundred pandas operations across three timeframes, and the worker shares
# its executor with the predictor. A short yield keeps a 200-name watchlist from
# monopolising the loop.
TICKER_DELAY = 0.05


# ── Pine primitives ──────────────────────────────────────────────────────────
# These three are where silent numeric error lives. Each has an obvious-looking
# wrong variant that produces plausible output, so each is asserted against a
# hand-computed value in tests/test_analyst_ratings.py.

def _rma(series: pd.Series, length: int) -> pd.Series:
    """Wilder's smoothing — ta.rma. alpha = 1/length, seeded on SMA(length).

    The seed matters. Pine primes the recursion with a simple mean of the first
    `length` valid samples; starting from the first value instead shifts RSI by
    several points for hundreds of bars, and a plain rolling mean is a different
    function entirely (~6 RSI points of mean absolute error).
    """
    series = series.astype(float)
    values = series.to_numpy()
    valid = np.flatnonzero(~np.isnan(values))
    if len(valid) < length:
        return pd.Series(np.nan, index=series.index, dtype=float)

    seeded = series.copy()
    first = valid[length - 1]
    seeded.iloc[:first] = np.nan
    seeded.iloc[first] = values[valid[:length]].mean()
    return seeded.ewm(alpha=1.0 / length, adjust=False).mean()


def _wma(series: pd.Series, length: int) -> pd.Series:
    """Linearly weighted MA — ta.wma. The NEWEST bar carries the largest weight.

    Pine's weight is (length - i) * length with i=0 being the current bar, so
    the ramp ascends oldest-to-newest. Reversing it is the easy mistake and it
    moves WMA(9) by whole points, which then propagates through the Hull MA.
    """
    values = series.to_numpy(dtype=float)
    if len(values) < length:
        return pd.Series(np.nan, index=series.index, dtype=float)
    weights = np.arange(1.0, length + 1.0)
    out = np.full(len(values), np.nan)
    out[length - 1:] = (sliding_window_view(values, length) @ weights) / weights.sum()
    return pd.Series(out, index=series.index)


def _ta_dev(series: pd.Series, length: int) -> pd.Series:
    """Mean absolute deviation about the window's own mean — ta.dev.

    Not a standard deviation. Substituting rolling().std() in the CCI
    denominator is wrong by ~19 CCI points, which is enough to flip its vote.
    """
    values = series.to_numpy(dtype=float)
    if len(values) < length:
        return pd.Series(np.nan, index=series.index, dtype=float)
    windows = sliding_window_view(values, length)
    out = np.full(len(values), np.nan)
    out[length - 1:] = np.abs(windows - windows.mean(axis=1, keepdims=True)).mean(axis=1)
    return pd.Series(out, index=series.index)


def _ema(series: pd.Series, length: int) -> pd.Series:
    """Pine's recursive EMA. adjust=False is mandatory, not stylistic.

    adjust=True computes the re-weighted finite-window form and differs by up to
    ~0.5 at span 13, compounding through MACD, Bull Bear Power and the trend
    gate that gates two more votes.
    """
    return series.ewm(span=length, adjust=False).mean()


def _donchian(high: pd.Series, low: pd.Series, length: int) -> pd.Series:
    """Midpoint of the `length`-bar range — the Ichimoku line construction."""
    return (high.rolling(length).max() + low.rolling(length).min()) / 2.0


def _at(series: pd.Series, offset: int = 0) -> float:
    """series[offset] in Pine terms: 0 is the last bar, 1 the one before it."""
    try:
        return float(series.iloc[-1 - offset])
    except (IndexError, ValueError, TypeError):
        return float("nan")


def _vote(bull: Any, bear: Any) -> int:
    """+1 / -1 / 0.

    NaN comparisons are False in both directions, so an indicator that has not
    warmed up abstains rather than voting. Both legs true is impossible in every
    rule below, but if a future edit makes it possible, neutral is the safe read.
    """
    bull, bear = bool(bull), bool(bear)
    if bull and not bear:
        return 1
    if bear and not bull:
        return -1
    return 0


def label_for(score: Optional[float]) -> Optional[str]:
    """Bucket a -1..+1 score. `ratingStatus` in the Pine library."""
    if score is None or score != score:
        return None
    if score < -RATING_STRONG:
        return STRONG_SELL
    if score < -RATING_WEAK:
        return SELL
    if score > RATING_STRONG:
        return STRONG_BUY
    if score > RATING_WEAK:
        return BUY
    return NEUTRAL


def resample_ohlcv(bars: pd.DataFrame, rule: Optional[str]) -> pd.DataFrame:
    """Daily bars -> weekly/monthly bars. Pass-through when rule is None.

    The partial trailing period is kept on purpose: the current week is the week
    traders are in, and dropping it would rate the market as of last Friday.
    """
    if rule is None:
        return bars
    out = bars.resample(rule).agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    })
    return out.dropna(subset=["close"])


# ── The rating ───────────────────────────────────────────────────────────────

def compute_rating(bars: pd.DataFrame) -> Optional[dict[str, Any]]:
    """One rating from one set of OHLCV bars, or None if history is too short.

    Takes a DataFrame indexed by date with open/high/low/close/volume columns,
    so the whole calculation is testable against a synthetic frame with no
    database and no network — the convention the other collectors follow in
    tests/test_flow_sources.py.

    Returns the 15 moving-average votes and 11 oscillator votes averaged into
    their two group scores, plus the summary. Note that the summary is the mean
    of the two GROUP scores, not of all 26 votes: the MA group is over-weighted
    relative to a flat average, which is TradingView's definition.
    """
    if bars is None or len(bars) < MIN_BARS:
        return None

    close, high, low = bars["close"], bars["high"], bars["low"]
    volume = bars["volume"]

    ma_votes = _moving_average_votes(close, high, low, volume)
    osc_votes = _oscillator_votes(close, high, low)

    ma_score = float(np.mean(list(ma_votes.values())))
    osc_score = float(np.mean(list(osc_votes.values())))
    summary = (ma_score + osc_score) / 2.0

    all_votes = list(ma_votes.values()) + list(osc_votes.values())
    return {
        "summary_score": summary,
        "summary_label": label_for(summary),
        "ma_score": ma_score,
        "ma_label": label_for(ma_score),
        "osc_score": osc_score,
        "osc_label": label_for(osc_score),
        "buy_votes": sum(1 for v in all_votes if v > 0),
        "neutral_votes": sum(1 for v in all_votes if v == 0),
        "sell_votes": sum(1 for v in all_votes if v < 0),
        "bars_available": int(len(bars)),
        # Kept for the panel tooltip and for debugging a rating that looks
        # wrong: without the per-indicator breakdown, a bad vote is invisible.
        "votes": {**ma_votes, **osc_votes},
    }


def _moving_average_votes(close: pd.Series, high: pd.Series, low: pd.Series,
                          volume: pd.Series) -> dict[str, int]:
    """15 votes: 6 SMA + 6 EMA + Hull MA + VWMA + Ichimoku.

    Every MA except Ichimoku votes on its position relative to price, which is
    why they move together and why this group swings harder than the
    oscillators. That is inherent to the methodology, not a bug to damp.
    """
    votes: dict[str, int] = {}
    spot = _at(close)

    for length in MA_LENGTHS:
        sma = close.rolling(length).mean()
        votes[f"SMA{length}"] = _vote(spot > _at(sma), spot < _at(sma))
        ema = _ema(close, length)
        votes[f"EMA{length}"] = _vote(spot > _at(ema), spot < _at(ema))

    # Pine integer-divides, so ta.hma(close, 9) uses wma(..., 9 // 2 == 4) and
    # a final smoothing of round(sqrt(9)) == 3.
    hma = _wma(2 * _wma(close, 4) - _wma(close, 9), 3)
    votes["HullMA9"] = _vote(spot > _at(hma), spot < _at(hma))

    # sma(close * volume, n) / sma(volume, n) reduces to the ratio of sums.
    vwma = (close * volume).rolling(20).sum() / volume.rolling(20).sum()
    votes["VWMA20"] = _vote(spot > _at(vwma), spot < _at(vwma))

    # Ichimoku: the full cloud, not the base line. All four conditions must
    # hold, and the leading spans are read 26 bars back — the displacement is
    # the whole point of the indicator and dropping it inverts the signal in
    # exactly the trending markets it is meant to confirm.
    conversion = _donchian(high, low, 9)
    base = _donchian(high, low, 26)
    lead1 = ((conversion + base) / 2.0).shift(26)
    lead2 = _donchian(high, low, 52).shift(26)

    l1, l2 = _at(lead1), _at(lead2)
    b, c = _at(base), _at(conversion)
    votes["Ichimoku"] = _vote(
        (l1 > l2) and (b > l1) and (c > b) and (spot > c),
        (l1 < l2) and (b < l1) and (c < b) and (spot < c),
    )
    return votes


def _oscillator_votes(close: pd.Series, high: pd.Series,
                      low: pd.Series) -> dict[str, int]:
    """11 votes. Most are conditional on a level AND a direction of travel.

    The recurring shape is "oversold and turning up" rather than merely
    "oversold": an indicator pinned at an extreme votes neutral until it starts
    to recover, which is what keeps the group from screaming buy all the way
    down a trend.
    """
    votes: dict[str, int] = {}

    trend_ema = _ema(close, TREND_EMA)
    up_trend = _at(close) > _at(trend_ema)
    down_trend = _at(close) < _at(trend_ema)

    # RSI(14) — Wilder-smoothed. The two guards are in TradingView's own source:
    # an all-down window is 0 rather than a division by zero.
    change = close.diff()
    gain = _rma(change.clip(lower=0), 14)
    loss = _rma((-change).clip(lower=0), 14)
    rsi = (100 - 100 / (1 + gain / loss)).where(loss != 0, 100.0).where(gain != 0, 0.0)
    votes["RSI14"] = _vote(
        _at(rsi) < 30 and _at(rsi, 1) < _at(rsi),
        _at(rsi) > 70 and _at(rsi, 1) > _at(rsi),
    )

    # Stochastic %K(14,3,3) — the rating uses the SMOOTHED %K, not raw.
    highest, lowest = high.rolling(14).max(), low.rolling(14).min()
    raw_k = 100 * (close - lowest) / (highest - lowest)
    k = raw_k.rolling(3).mean()
    d = k.rolling(3).mean()
    votes["Stoch14"] = _vote(
        _at(k) < 20 and _at(d) < 20 and _at(k) > _at(d),
        _at(k) > 80 and _at(d) > 80 and _at(k) < _at(d),
    )

    # CCI(20) on close, over mean absolute deviation.
    cci = (close - close.rolling(20).mean()) / (0.015 * _ta_dev(close, 20))
    votes["CCI20"] = _vote(
        _at(cci) < -100 and _at(cci) > _at(cci, 1),
        _at(cci) > 100 and _at(cci) < _at(cci, 1),
    )

    # ADX(14,14) with +DI/-DI. Only the larger directional move counts, and only
    # if it is positive — the classic implementation bug is letting both fire.
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)
    plus_dm.iloc[0] = minus_dm.iloc[0] = np.nan

    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    true_range.iloc[0] = np.nan

    smoothed_tr = _rma(true_range, 14)
    # Pine's fixnan() carries the last good value forward across the division's
    # zero-range holes; ffill is the same operation.
    di_plus = (100 * _rma(plus_dm, 14) / smoothed_tr).ffill()
    di_minus = (100 * _rma(minus_dm, 14) / smoothed_tr).ffill()
    di_sum = di_plus + di_minus
    dx = (di_plus - di_minus).abs() / di_sum.where(di_sum != 0, 1.0)
    adx = 100 * _rma(dx, 14)

    rising_adx = _at(adx) > 20 and _at(adx) > _at(adx, 1)
    votes["ADX14"] = _vote(
        rising_adx and _at(di_plus) > _at(di_minus),
        rising_adx and _at(di_plus) < _at(di_minus),
    )

    # Awesome Oscillator on median price, not close. Votes on a zero cross or on
    # two consecutive bars of acceleration in the same direction.
    median_price = (high + low) / 2.0
    ao = median_price.rolling(5).mean() - median_price.rolling(34).mean()
    ao0, ao1, ao2 = _at(ao), _at(ao, 1), _at(ao, 2)
    votes["AO"] = _vote(
        (ao1 < 0 <= ao0) or (ao0 > 0 and ao1 > 0 and ao0 > ao1 and ao2 > ao1),
        (ao1 > 0 >= ao0) or (ao0 < 0 and ao1 < 0 and ao0 < ao1 and ao2 < ao1),
    )

    # Momentum(10) — ta.mom is a plain difference, and the vote is on its slope.
    mom = close.diff(10)
    votes["Mom10"] = _vote(_at(mom) > _at(mom, 1), _at(mom) < _at(mom, 1))

    macd = _ema(close, 12) - _ema(close, 26)
    signal = _ema(macd, 9)
    votes["MACD"] = _vote(_at(macd) > _at(signal), _at(macd) < _at(signal))

    # Stochastic RSI(14,14,3,3) — a stochastic OF the RSI series, so RSI is the
    # source and the high and the low.
    rsi_high, rsi_low = rsi.rolling(14).max(), rsi.rolling(14).min()
    k_rsi = (100 * (rsi - rsi_low) / (rsi_high - rsi_low)).rolling(3).mean()
    d_rsi = k_rsi.rolling(3).mean()
    # The trend gate reads inverted and is meant to: TradingView treats Stoch
    # RSI as mean-reversion, so it buys an oversold reading only against a
    # DOWNtrend. Do not "fix" this to match Bull Bear Power below.
    votes["StochRSI"] = _vote(
        down_trend and _at(k_rsi) < 20 and _at(d_rsi) < 20 and _at(k_rsi) > _at(d_rsi),
        up_trend and _at(k_rsi) > 80 and _at(d_rsi) > 80 and _at(k_rsi) < _at(d_rsi),
    )

    # Williams %R(14). Pine writes `100 *`, and the negative range falls out.
    wr = 100 * (close - highest) / (highest - lowest)
    votes["WilliamsR14"] = _vote(
        _at(wr) < -80 and _at(wr) > _at(wr, 1),
        _at(wr) > -20 and _at(wr) < _at(wr, 1),
    )

    # Bull Bear Power(13). The legs are used separately, never summed: buying
    # needs the BEAR leg recovering in an uptrend, selling needs the BULL leg
    # fading in a downtrend.
    ema13 = _ema(close, 13)
    bull_power = high - ema13
    bear_power = low - ema13
    votes["BullBearPower"] = _vote(
        up_trend and _at(bear_power) < 0 and _at(bear_power) > _at(bear_power, 1),
        down_trend and _at(bull_power) > 0 and _at(bull_power) < _at(bull_power, 1),
    )

    # Ultimate Oscillator(7,14,28). Each average is a ratio of sums, which is
    # not the mean of the per-bar ratios, and the 4:2:1 weights divide by 7.
    true_low = pd.concat([low, prev_close], axis=1).min(axis=1)
    true_high = pd.concat([high, prev_close], axis=1).max(axis=1)
    buying_pressure = close - true_low
    uo_range = true_high - true_low

    def _avg(length: int) -> pd.Series:
        return buying_pressure.rolling(length).sum() / uo_range.rolling(length).sum()

    uo = 100 * (4 * _avg(7) + 2 * _avg(14) + _avg(28)) / 7
    votes["UO"] = _vote(_at(uo) > 70, _at(uo) < 30)

    return votes


# ── Tracker ──────────────────────────────────────────────────────────────────

class TechnicalRatingTracker:
    """Computes and stores one rating per ticker per timeframe per session."""

    def __init__(self, db: Database):
        self.db = db

    @property
    def enabled(self) -> bool:
        return settings.technical_rating_enabled

    async def compute_all(self, tickers: list[str]) -> dict[str, int]:
        """Rate every ticker across every timeframe.

        Unlike the network-bound collectors this is CPU work, so it runs in the
        executor rather than being rate-limited. A ticker whose history is too
        short is not a failure and is counted separately — on a fresh database
        that is every ticker, and it must not look like an outage.
        """
        totals = {"tickers": 0, "rows": 0, "insufficient": 0, "failed": 0}
        if not self.enabled:
            log.info("techrating.skipped_disabled")
            return totals

        loop = asyncio.get_running_loop()
        rows: list[dict[str, Any]] = []

        for ticker in tickers:
            try:
                built = await loop.run_in_executor(None, self._rate_sync, ticker)
            except Exception as e:
                log.error("techrating.failed", ticker=ticker,
                          error=str(e) or repr(e), error_type=type(e).__name__)
                totals["failed"] += 1
                continue

            totals["tickers"] += 1
            if not built:
                totals["insufficient"] += 1
            rows.extend(built)
            await asyncio.sleep(TICKER_DELAY)

        totals["rows"] = self.db.upsert_technical_ratings(rows)
        log.info("techrating.complete", **totals)
        return totals

    def _rate_sync(self, ticker: str) -> list[dict[str, Any]]:
        """Blocking rate of one ticker across all timeframes. Runs in an executor."""
        ticker = ticker.upper().strip()
        bars = self.load_bars(self.db, ticker)
        if bars is None or bars.empty:
            log.info("techrating.no_bars", ticker=ticker)
            return []

        now = datetime.now(timezone.utc)
        session = now.astimezone(_ET).date().isoformat()
        out: list[dict[str, Any]] = []

        for timeframe, rule in TIMEFRAMES.items():
            frame = resample_ohlcv(bars, rule)
            rating = compute_rating(frame)
            if rating is None:
                log.debug("techrating.insufficient_history", ticker=ticker,
                          timeframe=timeframe, bars=len(frame))
                continue
            rating.pop("votes", None)
            out.append({
                "ticker": ticker,
                "session_date": session,
                "timeframe": timeframe,
                "published_at": now.isoformat(),
                **rating,
            })
        return out

    @staticmethod
    def load_bars(db: Database, ticker: str) -> Optional[pd.DataFrame]:
        """Stored daily OHLCV as a date-indexed frame, or None if there is none."""
        rows = db.get_price_history(ticker)
        if not rows:
            return None
        frame = pd.DataFrame(rows)
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.set_index("date").sort_index()
        return frame.dropna(subset=["close"])

    # ── Reporting ────────────────────────────────────────────────────────

    def get_summary(self, ticker: str) -> dict[str, Any]:
        """Latest rating per timeframe for one ticker."""
        ticker = ticker.upper().strip()
        latest = self.db.get_latest_technical_ratings(ticker)
        return {
            "ticker": ticker,
            "timeframes": {row["timeframe"]: row for row in latest},
            "min_bars": MIN_BARS,
        }

    def get_report(self, ticker: str) -> str:
        """Plain-text block for the Bull/Bear debate prompt.

        Names the absence explicitly when there is no rating. A blank section
        invites the model to invent one, which is the failure mode the other
        trackers' reports are written to avoid.
        """
        summary = self.get_summary(ticker)
        frames = summary["timeframes"]
        if not frames:
            return (f"No technical rating available for {ticker} — fewer than "
                    f"{MIN_BARS} stored bars on every timeframe.")

        labels = {"short": "Short term (daily)",
                  "medium": "Medium term (weekly)",
                  "long": "Long term (monthly)"}
        lines = [f"Technical rating for {ticker} "
                 f"(26 indicators, TradingView methodology):"]
        for timeframe in TIMEFRAMES:
            row = frames.get(timeframe)
            if not row:
                lines.append(f"  {labels[timeframe]}: not enough history.")
                continue
            lines.append(
                f"  {labels[timeframe]}: {row['summary_label']} "
                f"({row['summary_score']:+.2f}) — "
                f"{row['buy_votes']} buy / {row['neutral_votes']} neutral / "
                f"{row['sell_votes']} sell; "
                f"MAs {row['ma_label']}, oscillators {row['osc_label']}."
            )
        lines.append("  Scale: -1 strong sell to +1 strong buy. This is a "
                     "momentum/trend read, not a valuation view.")
        return "\n".join(lines)
