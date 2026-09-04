"""Tests for the crowding / rumour-stage scorer.

The behaviour that matters most here is what happens when data is *absent*.
Most thesis candidates are freshly discovered names with no price history, no
mention history and no dark-pool rows, and "nobody is talking about it" is
legitimately the EARLY signal — so absence must neither raise nor silently
manufacture a confident score.
"""

import sqlite3
from datetime import date, timedelta

import pytest

from pipeline.crowding import (
    COMPONENT_WEIGHTS,
    MIN_COVERAGE_FOR_HIGH_STAGE,
    STAGE_BUILDING,
    STAGE_CROWDED,
    STAGE_EARLY,
    STAGE_POST_NEWS,
    STAGE_UNKNOWN,
    CrowdingScorer,
    clamp01,
    edge_score,
    score_conviction,
)


class FakeDB:
    """Minimal stand-in exposing only what CrowdingScorer touches."""

    def __init__(self, price_rows=None, mentions=None, offexch=None, insider=None):
        self._price_rows = price_rows or []
        self._mentions = mentions or {}
        self._offexch = offexch or []
        self._insider = insider or []

    def connection(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute(
            "CREATE TABLE price_history (ticker TEXT, date TEXT, close REAL, volume INTEGER)"
        )
        conn.executemany(
            "INSERT INTO price_history VALUES (?,?,?,?)",
            [("TEST", r["date"], r["close"], r["volume"]) for r in self._price_rows],
        )

        class Ctx:
            def __enter__(self_inner):
                return conn

            def __exit__(self_inner, *a):
                conn.close()
                return False

        return Ctx()

    def get_ticker_mention_counts(self, tickers, **kw):
        return {t: dict(self._mentions) for t in tickers}

    def get_offexchange_series(self, ticker):
        return list(self._offexch)

    def get_insider_series(self, ticker):
        return list(self._insider)


def make_prices(n=300, start=100.0, growth=0.0, volume=1_000_000):
    """n daily bars compounding at `growth` per bar."""
    rows, px = [], start
    d = date(2025, 1, 1)
    for i in range(n):
        px *= 1.0 + growth
        rows.append({"date": (d + timedelta(days=i)).isoformat(),
                     "close": round(px, 4), "volume": volume})
    return rows


# ── Weights and helpers ──────────────────────────────────────────────────


def test_component_weights_sum_to_one():
    assert pytest.approx(sum(COMPONENT_WEIGHTS.values()), abs=1e-9) == 1.0


def test_clamp01_bounds():
    assert clamp01(-5) == 0.0
    assert clamp01(0.42) == 0.42
    assert clamp01(99) == 1.0


# ── The missing-data contract ────────────────────────────────────────────


def test_no_data_at_all_is_unknown_not_early():
    """The single most important case.

    A candidate with no data must not be reported as EARLY — that would be
    indistinguishable from a genuinely quiet name, which is the entire
    distinction the feature rests on.
    """
    result = CrowdingScorer(FakeDB()).score("TEST")
    assert result["stage"] == STAGE_UNKNOWN
    assert result["crowding"] is None
    assert result["coverage"] == 0.0


def test_missing_data_never_raises():
    result = CrowdingScorer(FakeDB(price_rows=make_prices(5))).score("TEST")
    assert result["stage"] == STAGE_UNKNOWN


def test_coverage_reflects_only_observed_components():
    """Price-only data yields exactly the price components' weight."""
    db = FakeDB(price_rows=make_prices(300, growth=0.002))  # trades a real range
    result = CrowdingScorer(db).score("TEST")
    expected = sum(
        COMPONENT_WEIGHTS[k]
        for k in ("price_runup_1m", "price_runup_3m", "dist_from_52w_high", "volume_surge")
    )
    assert result["coverage"] == pytest.approx(expected, abs=1e-6)
    assert result["crowding"] is not None


def test_dist_from_high_absent_when_the_name_never_moved():
    """A flat stock sits at its 52-week high trivially.

    Reporting that as "fully crowded" would mark an ignored, motionless ticker
    as a finished move, so the component withholds itself below a 5% range.
    """
    flat = CrowdingScorer(FakeDB(price_rows=make_prices(300, growth=0.0))).score("TEST")
    assert flat["components"]["dist_from_52w_high"] is None

    ranged = CrowdingScorer(FakeDB(price_rows=make_prices(300, growth=0.002))).score("TEST")
    assert ranged["components"]["dist_from_52w_high"] is not None


def test_crowding_renormalises_rather_than_zero_filling():
    """A flat-priced name scores near 0 on observed components only.

    Under zero-filling the absent mention/smart-money components would drag the
    result toward 0 as if they had been measured and found quiet; renormalising
    means the score reports only what was actually seen.
    """
    db = FakeDB(price_rows=make_prices(300, growth=0.0))
    result = CrowdingScorer(db).score("TEST")
    observed = {k: v for k, v in result["components"].items() if v is not None}
    weight_seen = sum(COMPONENT_WEIGHTS[k] for k in observed)
    manual = sum(COMPONENT_WEIGHTS[k] * v for k, v in observed.items()) / weight_seen
    assert result["crowding"] == pytest.approx(manual, abs=1e-4)


def test_thin_coverage_cannot_reach_crowded():
    """Below the coverage floor the stage is capped at BUILDING."""
    db = FakeDB(price_rows=make_prices(300, growth=0.02))  # a violent run-up
    result = CrowdingScorer(db).score("TEST")
    if result["coverage"] < MIN_COVERAGE_FOR_HIGH_STAGE:
        assert result["stage"] in (STAGE_EARLY, STAGE_BUILDING)


# ── Directional behaviour ────────────────────────────────────────────────


def test_flat_quiet_name_scores_early():
    db = FakeDB(price_rows=make_prices(300, growth=0.0))
    result = CrowdingScorer(db).score("TEST")
    assert result["stage"] == STAGE_EARLY
    assert result["crowding"] < 0.25


def test_run_up_name_scores_more_crowded_than_flat_one():
    flat = CrowdingScorer(FakeDB(price_rows=make_prices(300, growth=0.0))).score("TEST")
    hot = CrowdingScorer(FakeDB(price_rows=make_prices(300, growth=0.01))).score("TEST")
    assert hot["crowding"] > flat["crowding"]


def test_heavy_mentions_raise_crowding():
    prices = make_prices(300, growth=0.0)
    quiet = CrowdingScorer(FakeDB(price_rows=prices)).score("TEST")
    loud = CrowdingScorer(FakeDB(
        price_rows=prices,
        mentions={"recent": 40, "base": 10, "total_recent": 200, "total_base": 400},
    )).score("TEST")
    assert loud["crowding"] > quiet["crowding"]
    assert loud["coverage"] > quiet["coverage"]


def test_mention_components_absent_below_baseline_floor():
    """A ticker with one historical mention yields no acceleration component."""
    db = FakeDB(
        price_rows=make_prices(300),
        mentions={"recent": 1, "base": 1, "total_recent": 100, "total_base": 400},
    )
    result = CrowdingScorer(db).score("TEST")
    assert result["components"]["mention_accel"] is None
    # Absolute attention is still observable even when the ratio is not.
    assert result["components"]["mention_absolute"] is not None


def test_insider_buying_pushes_against_crowding():
    """Insider purchases read as 'the crowd has not arrived'."""
    prices = make_prices(300, growth=0.0)
    buys = [{"filed_at": "2099-01-01", "is_discretionary": 1, "transaction_code": "P"}] * 5
    sells = [{"filed_at": "2099-01-01", "is_discretionary": 1, "transaction_code": "S"}] * 5
    buying = CrowdingScorer(FakeDB(price_rows=prices, insider=buys)).score("TEST")
    selling = CrowdingScorer(FakeDB(price_rows=prices, insider=sells)).score("TEST")
    assert buying["components"]["insider_net"] < selling["components"]["insider_net"]


# ── Stage classification ─────────────────────────────────────────────────


def test_post_news_requires_a_prior_snapshot():
    """POST_NEWS is a transition, not a threshold.

    Nothing can be POST_NEWS on its first ever scoring run, because rolling
    over is only observable against a previous reading.
    """
    raw = {"mention_accel": 1.0, "ret_1m": -0.1}
    assert CrowdingScorer._classify(0.9, 1.0, True, raw, None) == STAGE_CROWDED
    prior = {"mention_accel": 5.0}
    assert CrowdingScorer._classify(0.9, 1.0, True, raw, prior) == STAGE_POST_NEWS


def test_post_news_needs_attention_falling_and_price_down():
    raw_rising = {"mention_accel": 9.0, "ret_1m": -0.1}
    prior = {"mention_accel": 5.0}
    # Attention still climbing -> still CROWDED, not POST_NEWS.
    assert CrowdingScorer._classify(0.9, 1.0, True, raw_rising, prior) == STAGE_CROWDED


def test_stage_thresholds():
    raw = {}
    assert CrowdingScorer._classify(0.10, 1.0, True, raw, None) == STAGE_EARLY
    assert CrowdingScorer._classify(0.40, 1.0, True, raw, None) == STAGE_BUILDING
    assert CrowdingScorer._classify(0.60, 1.0, True, raw, None) == STAGE_CROWDED
    assert CrowdingScorer._classify(None, 0.0, False, raw, None) == STAGE_UNKNOWN


# ── Conviction and edge ──────────────────────────────────────────────────


def test_pure_play_beats_conglomerate_at_equal_confidence():
    """The distinction the whole feature depends on.

    A supplier with most of its revenue exposed to the bottleneck must outrank
    a conglomerate where the same tailwind is diluted away.
    """
    pure = score_conviction(0.7, exposure_pct=70.0, substitutability="duopoly",
                            evidence_count=3)
    diluted = score_conviction(0.7, exposure_pct=8.0, substitutability="duopoly",
                               evidence_count=3)
    assert pure > diluted


def test_sole_source_beats_commoditized():
    a = score_conviction(0.6, 50.0, "sole_source", 2)
    b = score_conviction(0.6, 50.0, "commoditized", 2)
    assert a > b


def test_conviction_is_bounded():
    assert 0.0 <= score_conviction(0.0, 0.0, "commoditized", 0) <= 1.0
    assert 0.0 <= score_conviction(1.0, 100.0, "sole_source", 99) <= 1.0


def test_edge_score_prefers_quiet_high_conviction():
    """The formula that encodes buy-the-rumour."""
    early = edge_score(conviction=0.8, crowding=0.1)
    late = edge_score(conviction=0.8, crowding=0.9)
    assert early > late
    assert early == pytest.approx(0.72, abs=1e-6)


def test_edge_score_is_none_when_crowding_unknown():
    """An unmeasurable edge is absent from the ranking, not optimistic."""
    assert edge_score(conviction=0.9, crowding=None) is None
