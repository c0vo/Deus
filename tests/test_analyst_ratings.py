"""Tests for analyst consensus and the TradingView-methodology technical rating.

No network and no API keys. The analyst payloads are verbatim slices of real
yfinance responses; the rating is checked against hand-computed values and
synthetic frames.

Every assertion here encodes a failure mode that produces a plausible number
rather than a visible error — which is the whole risk with this feature. A
wrongly seeded Wilder average, a reversed WMA weight ramp or a standard
deviation where the formula wants mean absolute deviation all yield a rating
that looks perfectly reasonable and is wrong.
"""

import numpy as np
import pandas as pd
import pytest

from pipeline.analyst_ratings import (
    AnalystRatingsTracker,
    key_from_mean,
    _num,
)
from pipeline.technical_rating import (
    BUY,
    MIN_BARS,
    NEUTRAL,
    STRONG_BUY,
    STRONG_SELL,
    SELL,
    TIMEFRAMES,
    _rma,
    _ta_dev,
    _wma,
    compute_rating,
    label_for,
    resample_ohlcv,
)


# ── Pine primitives ─────────────────────────────────────────────────────────
# The three functions where a wrong-but-plausible variant exists.

def test_rma_is_seeded_on_the_sma_not_the_first_value():
    """Pine primes Wilder's recursion with mean(first n), not with x[0].

    Seeding on the first value instead leaves the average biased for hundreds of
    bars — long enough that a 200-bar rating never recovers from it.
    """
    s = pd.Series([1.0, 2.0, 3.0, 4.0])
    out = _rma(s, 3)

    assert np.isnan(out.iloc[0]) and np.isnan(out.iloc[1])
    assert out.iloc[2] == pytest.approx(2.0)            # seed = mean(1,2,3)
    assert out.iloc[3] == pytest.approx(2.0 + (4.0 - 2.0) / 3.0)

    # The wrong variant: ewm straight off the raw series with no seeding.
    naive = s.ewm(alpha=1 / 3, adjust=False).mean()
    assert naive.iloc[3] != pytest.approx(out.iloc[3])


def test_rma_is_not_a_rolling_mean():
    """Substituting rolling().mean() shifts RSI by several points, not decimals."""
    s = pd.Series(np.linspace(1.0, 50.0, 60))
    assert _rma(s, 14).iloc[-1] != pytest.approx(s.rolling(14).mean().iloc[-1])


def test_rma_returns_all_nan_when_history_is_shorter_than_the_window():
    assert _rma(pd.Series([1.0, 2.0]), 14).isna().all()


def test_wma_weights_the_newest_bar_heaviest():
    """Pine's ramp ascends oldest-to-newest. Reversing it is the easy mistake.

    Both orderings return a number in the right range, so nothing fails loudly —
    it just propagates a wrong Hull MA into the moving-average vote.
    """
    s = pd.Series([1.0, 2.0, 3.0])
    assert _wma(s, 3).iloc[-1] == pytest.approx((1 * 1 + 2 * 2 + 3 * 3) / 6.0)
    # The reversed ramp, for contrast.
    assert _wma(s, 3).iloc[-1] != pytest.approx((3 * 1 + 2 * 2 + 1 * 3) / 6.0)


def test_ta_dev_is_mean_absolute_deviation_not_stdev():
    """CCI divides by ta.dev. rolling().std() is wrong by ~19 CCI points."""
    s = pd.Series([1.0, 2.0, 6.0])
    assert _ta_dev(s, 3).iloc[-1] == pytest.approx(2.0)   # (2+1+3)/3
    assert _ta_dev(s, 3).iloc[-1] != pytest.approx(s.rolling(3).std().iloc[-1])


# ── Buckets ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("score,expected", [
    (-1.0, STRONG_SELL),
    (-0.51, STRONG_SELL),
    (-0.5, SELL),        # boundary: exactly -0.5 is the weaker label
    (-0.2, SELL),
    (-0.1, NEUTRAL),     # boundary
    (0.0, NEUTRAL),
    (0.1, NEUTRAL),      # boundary
    (0.2, BUY),
    (0.5, BUY),          # boundary: exactly +0.5 is NOT strong buy
    (0.51, STRONG_BUY),
    (1.0, STRONG_BUY),
])
def test_label_boundaries_are_strict(score, expected):
    """TradingView's ratingStatus uses strict inequalities on both sides.

    Flipping one to >= moves every rating that lands exactly on a boundary, and
    boundaries are hit often because the scores are ratios of small integers —
    12/15 and 0.5 are both reachable exactly.
    """
    assert label_for(score) == expected


def test_label_of_missing_score_is_none():
    assert label_for(None) is None
    assert label_for(float("nan")) is None


# ── The rating ──────────────────────────────────────────────────────────────

def _synthetic(n: int, step: float = 1.0, start: float = 100.0) -> pd.DataFrame:
    """A clean monotonic trend. Direction follows the sign of `step`.

    Compounding rather than a straight line, and that detail matters: on a
    perfectly linear ramp the Hull MA equals price exactly — it is zero-lag by
    construction on a straight line — so it abstains and the moving-average
    group scores 14/15 instead of 15/15. That is a property of the indicator
    rather than anything a real series would show, so the fixture is convex.

    Monday-anchored so the first W-FRI bucket is a whole Mon-Fri week.
    """
    rate = step / start
    close = start * np.power(1.0 + rate, np.arange(n, dtype=float))
    tick = np.abs(close) * abs(rate) * 0.5
    return pd.DataFrame(
        {
            "open": close,
            "high": close + tick,
            "low": close - tick,
            "close": close,
            "volume": np.full(n, 1_000_000.0),
        },
        index=pd.date_range("2015-01-05", periods=n, freq="B"),
    )


def test_short_history_returns_none_rather_than_a_rating():
    """Refusing beats shrinking the denominator.

    TradingView drops un-warmed indicators, which silently makes two ratings on
    the same screen incomparable. This declines instead, so a thin history is
    visible as an absence.
    """
    assert compute_rating(_synthetic(MIN_BARS - 1)) is None
    assert compute_rating(_synthetic(MIN_BARS)) is not None
    assert compute_rating(None) is None


def test_every_indicator_votes_exactly_once():
    """26 votes, always. A miscounted group changes every score silently."""
    r = compute_rating(_synthetic(400))
    assert r["buy_votes"] + r["neutral_votes"] + r["sell_votes"] == 26
    assert len(r["votes"]) == 26


def test_group_denominators_are_15_and_11():
    """Scores must land on multiples of 1/15 and 1/11.

    This is the cheapest possible check that no indicator was dropped or
    double-counted: any other denominator produces a score that fails it.
    """
    r = compute_rating(_synthetic(400))
    assert (r["ma_score"] * 15) == pytest.approx(round(r["ma_score"] * 15))
    assert (r["osc_score"] * 11) == pytest.approx(round(r["osc_score"] * 11))


def test_uptrend_puts_every_moving_average_behind_price():
    """A monotonic rise is the one case where all 15 MA votes must agree.

    Ichimoku included — it is the only MA vote with a compound rule, so if the
    four-part cloud condition or its 26-bar displacement is wrong, this is where
    it shows up as 14/15 instead of 15/15.
    """
    r = compute_rating(_synthetic(400, step=1.0))
    assert r["ma_score"] == pytest.approx(1.0)
    assert r["ma_label"] == STRONG_BUY
    assert r["votes"]["Ichimoku"] == 1


def test_downtrend_inverts_the_moving_average_group():
    """A compounding decline flips 14 of the 15 MA votes to sell.

    Not 15: the Hull MA votes buy, and that is the indicator working as designed
    rather than a defect. Its nested-WMA construction removes lag by overshooting
    curvature, so on a decline that is decelerating — which compounding decay is —
    it sits just below price. Asserting -1.0 here would be asserting a Hull MA
    that lags like an SMA, which is the one thing it is built not to do.
    """
    r = compute_rating(_synthetic(400, step=-0.2, start=200.0))

    assert r["ma_score"] == pytest.approx(-13 / 15)
    assert r["ma_label"] == STRONG_SELL
    assert r["votes"]["Ichimoku"] == -1
    assert r["votes"]["HullMA9"] == 1
    assert sum(1 for k, v in r["votes"].items()
               if k.startswith(("SMA", "EMA")) and v == -1) == 12


def test_summary_is_the_mean_of_group_scores_not_of_all_votes():
    """The MA group is deliberately over-weighted relative to a flat average.

    With 15 MA votes and 11 oscillator votes, averaging all 26 would weight the
    oscillators 11/26; TradingView weights them 1/2. On a strong trend the two
    definitions differ by enough to change the label.
    """
    r = compute_rating(_synthetic(400))
    assert r["summary_score"] == pytest.approx((r["ma_score"] + r["osc_score"]) / 2)

    flat = np.mean([r["votes"][k] for k in r["votes"]])
    assert r["summary_score"] != pytest.approx(flat)


def test_bars_available_reports_the_frame_actually_rated():
    r = compute_rating(_synthetic(321))
    assert r["bars_available"] == 321


# ── Resampling ──────────────────────────────────────────────────────────────

def test_resample_keeps_ohlc_semantics():
    """open is the period's first, close its last, high/low its extremes.

    Getting close from 'first' would rate the market as of Monday every week.
    """
    daily = _synthetic(20)
    weekly = resample_ohlcv(daily, "W-FRI")

    assert len(weekly) < len(daily)
    assert weekly["close"].iloc[0] == daily["close"].iloc[4]     # first Friday
    assert weekly["open"].iloc[0] == daily["open"].iloc[0]
    assert weekly["high"].iloc[0] == daily["high"].iloc[:5].max()
    assert weekly["volume"].iloc[0] == daily["volume"].iloc[:5].sum()


def test_daily_timeframe_passes_bars_through_untouched():
    daily = _synthetic(30)
    assert resample_ohlcv(daily, TIMEFRAMES["short"]) is daily


def test_weekly_needs_five_times_the_daily_history():
    """Why `long` is absent for most tickers, asserted rather than assumed.

    280 weekly bars is ~5.4 years of dailies and 280 monthly bars is ~23 years,
    so a recently listed name legitimately has no medium or long rating.
    """
    daily = _synthetic(400)                                  # ~1.6 years
    assert compute_rating(daily) is not None
    assert compute_rating(resample_ohlcv(daily, "W-FRI")) is None
    assert compute_rating(resample_ohlcv(daily, "ME")) is None


# ── Analyst consensus parsing ───────────────────────────────────────────────

# Verbatim from yfinance 1.4.1 for AAPL.
AAPL_INFO = {
    "recommendationKey": "buy",
    "recommendationMean": 2.08696,
    "numberOfAnalystOpinions": 41,
    "targetMeanPrice": 322.81854,
    "targetHighPrice": 400.0,
    "targetLowPrice": 215.0,
    "targetMedianPrice": 330.0,
    "currentPrice": 313.33,
}

AAPL_TREND = [
    {"period": "0m", "strongBuy": 6, "buy": 21, "hold": 15, "sell": 2, "strongSell": 2},
    {"period": "-1m", "strongBuy": 6, "buy": 22, "hold": 14, "sell": 2, "strongSell": 2},
    {"period": "-2m", "strongBuy": 6, "buy": 22, "hold": 16, "sell": 1, "strongSell": 2},
    {"period": "-3m", "strongBuy": 7, "buy": 23, "hold": 15, "sell": 1, "strongSell": 2},
]


def test_build_row_maps_every_field():
    row = AnalystRatingsTracker.build_row(
        "AAPL", AAPL_INFO, AnalystRatingsTracker.latest_trend(pd.DataFrame(AAPL_TREND)))

    assert row["strong_buy"] == 6 and row["hold"] == 15 and row["strong_sell"] == 2
    assert row["analyst_count"] == 41
    assert row["recommendation_key"] == "buy"
    assert row["target_mean"] == pytest.approx(322.81854)
    assert row["target_high"] == 400.0 and row["target_low"] == 215.0
    assert row["spot_price"] == pytest.approx(313.33)
    assert row["published_at"] and row["session_date"]


def test_latest_trend_selects_the_current_month_by_label():
    """Yahoo labels the current month '0m'. Reading position 0 instead works
    only while the frame happens to arrive newest-first, and it has arrived both
    ways across yfinance releases — a positional read then silently rates the
    ticker on three-month-old opinion."""
    reversed_frame = pd.DataFrame(list(reversed(AAPL_TREND)))
    assert AnalystRatingsTracker.latest_trend(reversed_frame)["buy"] == 21


def test_latest_trend_tolerates_missing_and_empty_inputs():
    assert AnalystRatingsTracker.latest_trend(None) == {}
    assert AnalystRatingsTracker.latest_trend(pd.DataFrame()) == {}


def test_uncovered_ticker_yields_no_row_at_all():
    """A row of nulls would make "no analysts follow this" indistinguishable
    from "the fetch failed", and small caps genuinely have no coverage."""
    assert AnalystRatingsTracker.build_row("NOBODY", {}, {}) is None
    assert AnalystRatingsTracker.build_row(
        "NOBODY", {"currentPrice": 10.0},
        {"strong_buy": 0, "buy": 0, "hold": 0, "sell": 0, "strong_sell": 0},
    ) is None


def test_a_ticker_with_targets_but_no_buckets_still_stores():
    row = AnalystRatingsTracker.build_row("X", {"targetMeanPrice": 50.0}, {})
    assert row is not None and row["target_mean"] == 50.0


def test_nan_targets_become_none_not_zero():
    """pandas NaN is truthy and unequal to itself.

    `value or None` keeps it, and a NaN target then reaches the panel as a
    real-looking price target. A missing target is not a target of zero.
    """
    nan = float("nan")
    row = AnalystRatingsTracker.build_row(
        "X", {"targetMeanPrice": nan, "targetHighPrice": nan, "currentPrice": 10.0},
        {"strong_buy": 1, "buy": 0, "hold": 0, "sell": 0, "strong_sell": 0})

    assert row["target_mean"] is None
    assert row["target_high"] is None


def test_num_maps_junk_to_none_rather_than_zero():
    assert _num(float("nan")) is None
    assert _num(None) is None
    assert _num("garbage") is None
    assert _num("12.5") == 12.5


def test_analyst_count_falls_back_to_the_bucket_sum():
    """Yahoo's count and the bucket total legitimately disagree — the buckets
    count ratings, the count counts analysts. A panel reading "based on 0
    analysts" beside 44 ratings looks broken, so absence falls back."""
    row = AnalystRatingsTracker.build_row(
        "X", {"targetMeanPrice": 50.0},
        {"strong_buy": 2, "buy": 3, "hold": 1, "sell": 0, "strong_sell": 0})
    assert row["analyst_count"] == 6


def test_price_targets_fallback_is_used_when_info_is_thin():
    """analyst_price_targets is a subset of .info and omits the analyst count,
    so it is a fallback rather than the primary read."""
    row = AnalystRatingsTracker.build_row(
        "X", {}, {},
        targets_fallback={"current": 9.0, "high": 20.0, "low": 5.0,
                          "mean": 12.0, "median": 11.0})
    assert row["target_mean"] == 12.0
    assert row["spot_price"] == 9.0


@pytest.mark.parametrize("mean,expected", [
    (1.0, "strong_buy"), (1.49, "strong_buy"),
    (1.5, "buy"), (2.49, "buy"),
    (2.5, "hold"), (3.49, "hold"),
    (3.5, "sell"), (4.49, "sell"),
    (4.5, "strong_sell"), (5.0, "strong_sell"),
    (None, None),
])
def test_recommendation_key_derived_from_the_mean(mean, expected):
    """Yahoo's scale is inverted: LOW is bullish. Reading it the intuitive way
    round labels every strong buy a strong sell."""
    assert key_from_mean(mean) == expected


@pytest.mark.asyncio
async def test_snapshot_skips_non_us_symbols():
    """Korean listings, indices and crypto have no sell-side coverage on Yahoo.

    Mirrors test_flow_sources.test_sync_range_filters_to_tracked_tickers: the
    filter has to happen before any request is made, not after.
    """
    stored = []

    class _DB:
        def upsert_analyst_consensus(self, rows):
            stored.extend(rows)
            return len(rows)

    tracker = AnalystRatingsTracker(db=_DB())
    calls = []
    tracker._snapshot_sync = lambda t: calls.append(t) or None

    totals = await tracker.snapshot_all(["005930.KS", "BTC-USD", "^VIX"])

    assert calls == []
    assert totals["tickers"] == 0
    assert stored == []
