"""
Tests for pipeline.model_eval and pipeline.model_configs, on synthetic data only.

The panel is built so the right answer is known: every ticker trades every
weekday (rows of different tickers share dates), column 0 of X drives the
forward return and columns 1-2 are noise, and the last `horizon` rows of each
ticker have no label. A harness that works finds column 0, ships "model" on it,
ships "prior" on pure noise and scores chance once the labels are shuffled.

The fold tests are the leakage contract: no training label may be known on or
after `test_start - embargo`, and unlabeled rows must never be scored.

The bootstrap, evidence and fold-plan tests are the statistical contract: an
interval over overlapping labels must widen with its block length and, widened
for what blocks miss, hold its level on noise; a fold too short to hold two
blocks must give no interval rather than a narrow one; the ship rule must refuse
confirm folds with too few independent label windows, and the fold plan must
give them enough windows whenever history allows.

The same-date tests are the regression test for the across-dates bias: on a
random walk, a rule built from the price path must not look skilful, which it
does when a fold's rows are ranked across dates.

The training-target tests are the labelling contract: a config's label changes
only what a model is fitted on (and on which rows); every metric, the
calibration and the artifact describe P(close higher). The leaderboard tests
keep a selection AUC drawn from a few weeks of folds from choosing PRODUCTION.

No network, no database; the slowest pieces are three small HGB evaluations.
"""

import functools
import importlib.util
import json
import math
import pickle
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from pipeline import model_eval, model_training
from pipeline.features import FEATURE_GROUPS, FEATURE_NAMES
from pipeline.model_configs import (
    CALENDAR_DAYS_PER_SESSION,
    CONFIGS,
    DEFAULT_HGB,
    FACTOR_FEATURES,
    FEATURE_GROUP_NAMES,
    PLAN_MAX_TEST_DAYS,
    PRODUCTION,
    RUN1_GROUPS,
    RUN2_GROUPS,
    STRONG_HGB,
    SWEEP_GRID_SMALL,
    V3_LIKE_FEATURES,
    ModelConfig,
    config_for_horizon,
    fold_plan,
    get_config,
    planned_test_days,
    resolve_columns,
    sweep_configs,
    train_stride_for,
)
from pipeline.model_eval import (
    MIN_CONFIRM_N_EFF,
    Calibrator,
    EvalResult,
    baseline_logreg_factory,
    baseline_momentum,
    baseline_prior,
    block_bootstrap_auc,
    block_length_for,
    coverage_folds,
    deadband_mask,
    evaluate,
    fit_calibrator,
    gbm_factory,
    hgb_factory,
    metrics_block,
    per_ticker_factory,
    same_date_auc,
    ship_decision,
    ship_reasons,
    shuffle_control,
    summarize_for_report,
    walkforward_folds,
    weights_for,
)

HORIZON = 5
N_FOLDS = 4
# A noise panel that fails several ship conditions at once (AUC, its CI, Brier
# skill), so the "prior" assertion does not hinge on one borderline number.
NOISE_SEED = 2


def make_panel(*, n_tickers=6, start="2019-01-01", end="2023-12-29", horizon=HORIZON,
               signal=1.0, seed=0):
    """Weekday panel; fwd_ret = 0.01 * (signal * X[:, 0] + noise); last `horizon` rows unlabeled."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, end).to_numpy()
    label_one = np.full(days.size, np.datetime64("NaT"), dtype="datetime64[ns]")
    label_one[:-horizon] = days[horizon:]
    dates = np.tile(days, n_tickers)
    label_dates = np.tile(label_one, n_tickers)
    tickers = np.repeat([f"T{i}" for i in range(n_tickers)], days.size)
    X = rng.normal(size=(dates.size, 3))
    fwd = 0.01 * (signal * X[:, 0] + rng.normal(size=dates.size))
    fwd[np.isnat(label_dates)] = np.nan
    y = np.where(np.isnan(fwd), np.nan, (fwd > 0).astype(float))
    return {"dates": dates, "label_dates": label_dates, "tickers": tickers, "X": X,
            "fwd": fwd, "y": y}


class _TrainingLabelRecorder:
    """Answers its training up-rate for every row and appends the labels of each fit to `log`."""

    def __init__(self, log):
        self.log = log

    def fit(self, X, y, sample_weight=None):
        self.log.append(np.asarray(y).copy())
        self.rate_ = float(np.mean(y))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = np.full(len(X), self.rate_)
        return np.column_stack([1.0 - p, p])


@pytest.fixture(scope="module")
def panel():
    return make_panel()


@pytest.fixture(scope="module")
def folds(panel):
    return walkforward_folds(panel["dates"], panel["label_dates"], HORIZON, n_folds=N_FOLDS)


@pytest.fixture(scope="module")
def signal_result(panel, folds):
    return evaluate(hgb_factory(DEFAULT_HGB), panel["X"], panel["y"], panel["fwd"],
                    panel["dates"], panel["tickers"], folds, label_dates=panel["label_dates"])


def _days(values):
    return pd.DatetimeIndex(values).to_numpy().astype("datetime64[D]")


# ---------------------------------------------------------------------------
# (a) walkforward_folds / coverage_folds
# ---------------------------------------------------------------------------

class TestWalkforwardFolds:

    def test_folds_are_chronological_contiguous_and_end_at_last_label(self, panel, folds):
        assert len(folds) == N_FOLDS
        assert [f.index for f in folds] == list(range(N_FOLDS))
        for earlier, later in zip(folds, folds[1:]):
            assert earlier.test_end < later.test_start
            assert later.test_start - earlier.test_end == pd.Timedelta(days=1)
        for fold in folds:
            assert fold.test_end - fold.test_start == pd.Timedelta(days=126 - 1)
        last_labeled = _days(panel["dates"][~np.isnat(panel["label_dates"])]).max()
        assert np.datetime64(folds[-1].test_end.date(), "D") == last_labeled

    def test_test_blocks_are_disjoint_and_hold_every_labeled_row(self, panel, folds):
        seen = np.concatenate([f.test_idx for f in folds])
        assert np.unique(seen).size == seen.size
        day = _days(panel["dates"])
        labeled = ~np.isnat(panel["label_dates"])
        for fold in folds:
            start = np.datetime64(fold.test_start.date(), "D")
            end = np.datetime64(fold.test_end.date(), "D")
            expected = np.flatnonzero(labeled & (day >= start) & (day <= end))
            assert np.array_equal(np.sort(fold.test_idx), expected)

    def test_purge_embargo(self, panel, folds):
        embargo = math.ceil(1.5 * HORIZON)
        label_day = _days(panel["label_dates"])
        labeled = ~np.isnat(panel["label_dates"])
        for fold in folds:
            assert fold.embargo_days == embargo
            cutoff = np.datetime64(fold.test_start.date(), "D") - np.timedelta64(embargo, "D")
            assert (label_day[fold.train_idx] < cutoff).all()
            # and nothing labeled before the cutoff is thrown away
            assert np.array_equal(fold.train_idx, np.flatnonzero(labeled & (label_day < cutoff)))

    def test_explicit_embargo_moves_the_cutoff(self, panel):
        wide = walkforward_folds(panel["dates"], panel["label_dates"], HORIZON, n_folds=2,
                                 embargo_days=60)
        label_day = _days(panel["label_dates"])
        for fold in wide:
            assert fold.embargo_days == 60
            cutoff = np.datetime64(fold.test_start.date(), "D") - np.timedelta64(60, "D")
            assert (label_day[fold.train_idx] < cutoff).all()

    def test_same_date_rows_never_straddle_a_split(self, panel, folds):
        day = _days(panel["dates"])
        for fold in folds:
            assert not set(day[fold.train_idx]) & set(day[fold.test_idx])

    def test_unlabeled_rows_appear_nowhere(self, panel):
        label_dates = panel["label_dates"].copy()
        holes = np.arange(3000, 3050)
        label_dates[holes] = np.datetime64("NaT")
        unlabeled = set(np.flatnonzero(np.isnat(label_dates)))
        for fold in walkforward_folds(panel["dates"], label_dates, HORIZON, n_folds=N_FOLDS):
            assert not unlabeled & set(fold.train_idx)
            assert not unlabeled & set(fold.test_idx)

    def test_min_train_days_drops_the_oldest_folds(self, panel):
        kept = walkforward_folds(panel["dates"], panel["label_dates"], HORIZON, n_folds=6,
                                 min_train_days=1200)
        assert 2 <= len(kept) < 6
        day = _days(panel["dates"])
        for fold in kept:
            span = (day[fold.train_idx].max() - day[fold.train_idx].min()).astype(int)
            assert span >= 1200

    def test_too_short_history_raises(self):
        short = make_panel(start="2022-01-03", end="2023-03-31")
        with pytest.raises(ValueError, match="usable fold"):
            walkforward_folds(short["dates"], short["label_dates"], HORIZON)

    def test_no_labels_raises(self, panel):
        with pytest.raises(ValueError, match="no labeled rows"):
            walkforward_folds(panel["dates"],
                              np.full(panel["dates"].size, np.datetime64("NaT"), dtype="datetime64[ns]"),
                              HORIZON)

    def test_tz_aware_and_string_dates_give_the_same_folds(self, panel, folds):
        tz_dates = pd.Series(pd.DatetimeIndex(panel["dates"]).tz_localize("UTC"))
        str_labels = pd.Series(panel["label_dates"]).dt.strftime("%Y-%m-%d").to_numpy()
        again = walkforward_folds(tz_dates, str_labels, HORIZON, n_folds=N_FOLDS)
        assert [(f.test_start, f.test_end) for f in again] == [(f.test_start, f.test_end) for f in folds]
        for a, b in zip(again, folds):
            assert np.array_equal(a.train_idx, b.train_idx)
            assert np.array_equal(a.test_idx, b.test_idx)

    def test_coverage_folds_stay_inside_the_coverage_era(self, panel):
        coverage = pd.Timestamp("2023-09-01")
        cov = coverage_folds(panel["dates"], panel["label_dates"], HORIZON, "2023-09-01T00:00:00Z")
        assert len(cov) == 3
        day = _days(panel["dates"])
        label_day = _days(panel["label_dates"])
        for fold in cov:
            assert fold.test_start >= coverage
            assert fold.test_end - fold.test_start == pd.Timedelta(days=20)
            cutoff = np.datetime64(fold.test_start.date(), "D") - np.timedelta64(8, "D")
            assert (label_day[fold.train_idx] < cutoff).all()
            assert day[fold.train_idx].min() < np.datetime64("2020-01-01")   # any earlier date

    def test_coverage_folds_empty_when_not_measurable(self, panel):
        assert coverage_folds(panel["dates"], panel["label_dates"], HORIZON, "2023-12-10") == []
        assert coverage_folds(panel["dates"], panel["label_dates"], HORIZON, None) == []


# ---------------------------------------------------------------------------
# (b) metrics_block
# ---------------------------------------------------------------------------

class TestMetricsBlock:

    def test_hand_computed_case(self):
        y = [1, 0, 1, 0]
        p = [0.8, 0.6, 0.4, 0.2]
        dates = ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]
        m = metrics_block(y, p, None, 0.5, 2, dates)
        assert m["auc"] == pytest.approx(1.0)           # both same-date pairs ordered
        assert m["auc_pooled"] == pytest.approx(0.75)   # across dates: 3 of 4 pos/neg pairs ordered
        assert m["brier"] == pytest.approx(0.2)         # (.04 + .36 + .36 + .04) / 4
        assert m["brier_prior"] == pytest.approx(0.25)
        assert m["brier_skill"] == pytest.approx(0.2)
        assert m["log_loss"] == pytest.approx(-(math.log(0.8) + math.log(0.4)) / 2)
        assert m["acc"] == pytest.approx(0.5)
        assert m["acc_majority"] == pytest.approx(0.5)
        assert m["acc_skill"] == pytest.approx(0.0)
        assert m["hi_conf_10"] == {"acc": pytest.approx(0.5), "n": 4}   # 0.6 and 0.4 count
        assert m["hi_conf_15"] == {"acc": pytest.approx(1.0), "n": 2}
        assert math.isnan(m["decile_spread"])           # under 20 rows
        assert (m["n"], m["n_dates"], m["n_eff"]) == (4, 2, 1.0)

    def test_perfect_forecast(self):
        y = np.array([0, 1] * 15)
        m = metrics_block(y, y.astype(float), np.where(y == 1, 0.02, -0.02), 0.5, 5,
                          np.repeat(pd.bdate_range("2024-01-01", periods=15), 2))
        assert m["auc"] == 1.0
        assert m["brier"] == 0.0
        assert m["brier_skill"] == 1.0
        assert m["acc"] == 1.0
        assert m["log_loss"] < 1e-5
        assert m["decile_spread"] == pytest.approx(0.04)

    def test_constant_prior_forecast_has_zero_skill(self):
        rng = np.random.default_rng(3)
        y = (rng.random(200) < 0.6).astype(int)
        prior = 0.6
        m = metrics_block(y, np.full(200, prior), rng.normal(size=200), prior, 5,
                          np.repeat(pd.bdate_range("2024-01-01", periods=40), 5))
        assert m["brier_skill"] == pytest.approx(0.0, abs=1e-12)
        assert m["acc_skill"] == pytest.approx(0.0, abs=1e-12)
        assert m["auc"] == pytest.approx(0.5)
        assert m["decile_spread"] == 0.0              # ties fall on both sides

    def test_single_class_auc_is_half(self):
        m = metrics_block([1, 1, 1], [0.2, 0.5, 0.9], None, 0.5, 1, ["2024-01-02"] * 3)
        assert m["auc"] == 0.5

    def test_decile_spread_top_minus_bottom(self):
        p = (np.arange(20) + 0.5) / 20
        fwd = np.arange(20, dtype=float)
        dates = ["2024-01-02"] * 20
        assert metrics_block((fwd > 9).astype(int), p, fwd, 0.5, 1, dates)["decile_spread"] == pytest.approx(18.0)
        assert math.isnan(metrics_block((fwd[:19] > 9).astype(int), p[:19], fwd[:19], 0.5, 1,
                                        dates[:19])["decile_spread"])

    def test_decile_spread_ranks_within_each_date(self):
        # date two's scores are all higher and its returns all lower: ranked across
        # dates the spread would be negative; within each date it is +1.0
        p = np.r_[np.linspace(0.1, 0.2, 10), np.linspace(0.8, 0.9, 10)]
        fwd = np.r_[np.linspace(0.0, 1.0, 10), np.linspace(-9.0, -8.0, 10)]
        dates = ["2024-01-02"] * 10 + ["2024-01-03"] * 10
        m = metrics_block((fwd > -5).astype(int), p, fwd, 0.5, 1, dates)
        assert m["decile_spread"] == pytest.approx(1.0)
        # one row per date holds no same-date ranking at all
        lone = metrics_block((fwd > -5).astype(int), p, fwd, 0.5, 1,
                             pd.bdate_range("2024-01-01", periods=20))
        assert math.isnan(lone["decile_spread"]) and lone["auc"] == 0.5

    def test_reliability_bins_skip_empty_ones(self):
        m = metrics_block([0, 1, 0, 1, 1], [0.05, 0.15, 0.15, 0.95, 1.0], None, 0.5, 1,
                          ["2024-01-02"] * 5)
        bins = {b["bin"]: b for b in m["reliability"]}
        assert sorted(bins) == [0, 1, 9]
        assert bins[1]["n"] == 2 and bins[1]["y_rate"] == pytest.approx(0.5)
        assert bins[9]["p_mean"] == pytest.approx(0.975)
        assert sum(b["n"] for b in m["reliability"]) == 5

    def test_p_rank_drives_only_the_ranking_metrics(self):
        y = np.array([0, 1] * 10)
        m = metrics_block(y, np.full(20, 0.5), y * 1.0, 0.5, 1,
                          ["2024-01-02"] * 10 + ["2024-01-03"] * 10, p_rank=y + 0.1)
        assert m["auc"] == 1.0 and m["auc_pooled"] == 1.0
        assert m["decile_spread"] == pytest.approx(1.0)
        assert m["brier"] == pytest.approx(0.25)

    def test_per_row_prior(self):
        y = np.array([1, 1, 0, 0])
        m = metrics_block(y, [0.9, 0.9, 0.1, 0.1], None, np.array([0.7, 0.7, 0.3, 0.3]), 1,
                          ["2024-01-02"] * 4)
        assert m["acc_majority"] == 1.0
        assert m["brier_prior"] == pytest.approx(0.09)


# ---------------------------------------------------------------------------
# (b2) same-date ranking: the across-dates bias the ship rule avoids
# ---------------------------------------------------------------------------

def _path_rule_on_a_random_walk(seed=0, n_dates=1260, n_tickers=12, horizon=5, lookback=60):
    """
    A random walk with a market factor, scored by a rule built from its own
    past: lean against the market's and the ticker's last `lookback` sessions.
    Nothing here is predictable, but the rule's score late in a test block
    holds the returns that decided the labels of the block's earlier rows.
    Returns (y, score, fwd, dates) in date order, one row per (date, ticker).
    """
    rng = np.random.default_rng(seed)
    n = n_dates + lookback + horizon + 1
    market = rng.standard_normal(n)
    level = np.cumsum(0.8 * market[:, None] + 0.6 * rng.standard_normal((n, n_tickers)), axis=0)
    market_level = np.cumsum(market)
    rows = np.arange(lookback, lookback + n_dates)
    fwd = level[rows + horizon] - level[rows]
    score = (-(market_level[rows] - market_level[rows - lookback])[:, None]
             - 0.5 * (level[rows] - level[rows - lookback]))
    dates = np.repeat(pd.bdate_range("2010-01-04", periods=n_dates).to_numpy(), n_tickers)
    return (fwd > 0).astype(int).ravel(), score.ravel(), fwd.ravel(), dates


class TestSameDateRanking:

    def test_same_date_auc_counts_only_pairs_on_one_date(self):
        rng = np.random.default_rng(8)
        dates = np.repeat(pd.bdate_range("2024-01-01", periods=30).to_numpy(), 7)
        y = (rng.random(dates.size) < 0.5).astype(float)
        p = np.round(rng.random(dates.size), 1)                      # plenty of ties

        def brute(days):
            num = den = 0.0
            for d in days:
                m = dates == d
                up, down = p[m & (y == 1)], p[m & (y == 0)]
                num += (up[:, None] > down[None, :]).sum() + 0.5 * (up[:, None] == down[None, :]).sum()
                den += up.size * down.size
            return num, den

        days = np.unique(dates)
        num, den = brute(days)
        assert same_date_auc(y, p, dates) == pytest.approx(num / den, abs=1e-12)
        # a date drawn twice counts its own pairs twice and is never paired with its copy
        stat = model_eval._SameDateAUC(y, p, model_eval._chronological_codes(dates))
        weights = np.ones(stat.n_dates)
        weights[0] = 2.0
        first_num, first_den = brute(days[:1])
        assert stat.auc(weights) == pytest.approx((num + first_num) / (den + first_den), abs=1e-12)

    def test_a_date_without_both_outcomes_carries_no_weight(self):
        dates = ["2024-01-02"] * 4 + ["2024-01-03"] * 4
        y = [1, 1, 1, 1, 1, 0, 1, 0]
        p = [0.9, 0.1, 0.5, 0.3, 0.8, 0.2, 0.7, 0.4]
        # the first date went up across the board, so only the second date's four
        # pairs count, and it orders all four; ranked across dates the ups at 0.1
        # and 0.3 would sit below downs
        assert same_date_auc(y, p, dates) == pytest.approx(1.0)
        assert roc_auc_score(y, p) < 1.0
        assert same_date_auc([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1], ["2024-01-02", "2024-01-02",
                                                                   "2024-01-03", "2024-01-03"]) == 0.5

    def test_a_rule_built_from_the_path_is_not_rewarded_on_a_random_walk(self):
        # Regression test for the across-dates bias. Ranked across the dates of
        # six-month blocks, a rule that leans against recent moves looks skilful
        # on returns nothing can predict; ranked against same-date rows only, it
        # scores chance. The trend-following mirror image is punished across dates.
        y, score, fwd, dates = _path_rule_on_a_random_walk()
        per_block = 126 * 12
        pooled, same, spread = [], [], []
        for start in range(0, y.size, per_block):
            s = slice(start, start + per_block)
            m = metrics_block(y[s], np.full(y[s].size, 0.5), fwd[s], 0.5, 5, dates[s], p_rank=score[s])
            pooled.append(m["auc_pooled"])
            same.append(m["auc"])
            spread.append(m["decile_spread"])
        assert np.mean(pooled) > 0.53
        assert abs(np.mean(same) - 0.5) < 0.015
        trend = [metrics_block(y[start:start + per_block], np.full(per_block, 0.5), None, 0.5, 5,
                               dates[start:start + per_block],
                               p_rank=-score[start:start + per_block])["auc_pooled"]
                 for start in range(0, y.size, per_block)]
        assert np.mean(trend) < 0.47

    def test_the_interval_of_a_path_rule_covers_chance_only_on_the_same_date(self):
        # ten six-month folds: across dates the rule's interval clears 0.50, a false
        # "edge"; ranked against same-date rows it brackets chance
        y, score, _fwd, dates = _path_rule_on_a_random_walk(seed=2)
        groups = np.repeat(np.arange(10), 126 * 12)
        lo, hi = block_bootstrap_auc(y, score, dates, n_boot=300, seed=0, groups=groups,
                                     block_length=5, horizon=5)
        assert lo < 0.5 < hi
        pooled_lo, _ = block_bootstrap_auc(y, score, dates, n_boot=300, seed=0, groups=groups,
                                           block_length=5, horizon=5, same_date=False)
        assert pooled_lo > 0.5


# ---------------------------------------------------------------------------
# (c) block_bootstrap_auc
# ---------------------------------------------------------------------------

def _scored_rows(seed=0, n_dates=300, per_date=5):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n_dates * per_date)
    y = (x + 1.5 * rng.normal(size=x.size) > 0).astype(int)
    p = 1 / (1 + np.exp(-x))
    dates = np.repeat(pd.bdate_range("2020-01-01", periods=n_dates).to_numpy(), per_date)
    return y, p, dates


def _overlapping_label_rows(seed=0, n_dates=500, n_tickers=8, horizon=20):
    """
    Rows shaped like the panel's: every label is the sign of the next `horizon`
    sessions' return (so neighbouring dates share horizon - 1 of them), every
    ticker shares a market factor, and the scores are a slow random walk with a
    trace of the future return in them. Adjacent dates are near-duplicates.
    """
    rng = np.random.default_rng(seed)
    n = n_dates + horizon
    ret = 0.7 * rng.normal(size=(n, 1)) + rng.normal(size=(n, n_tickers))
    csum = np.vstack([np.zeros(n_tickers), np.cumsum(ret, axis=0)])
    fwd = csum[horizon + 1:horizon + 1 + n_dates] - csum[1:1 + n_dates]
    score = 0.05 * np.cumsum(rng.normal(size=(n_dates, n_tickers)), axis=0) + 0.02 * fwd
    dates = pd.bdate_range("2015-01-01", periods=n_dates).to_numpy()
    return (fwd > 0).astype(int).ravel(), score.ravel(), np.repeat(dates, n_tickers)


def _noise_with_overlapping_labels(seed, n_dates=400, n_tickers=10, horizon=10):
    """Persistent forecasts that say nothing, labels overlapping `horizon` sessions, a market factor."""
    rng = np.random.default_rng(seed)
    n = n_dates + horizon
    ret = 0.7 * rng.normal(size=(n, 1)) + rng.normal(size=(n, n_tickers))
    csum = np.vstack([np.zeros(n_tickers), np.cumsum(ret, axis=0)])
    fwd = csum[horizon + 1:horizon + 1 + n_dates] - csum[1:1 + n_dates]
    score = np.cumsum(rng.normal(size=(n_dates, n_tickers)), axis=0)
    dates = np.repeat(pd.bdate_range("2012-01-02", periods=n_dates).to_numpy(), n_tickers)
    y = (fwd > 0).astype(int).ravel()
    return y, score.ravel(), dates, np.repeat([0, 1], y.size // 2)


class TestBlockBootstrapAUC:

    def test_widened_interval_holds_its_level_on_overlapping_label_noise(self):
        # Blocks of h sessions miss part of the dependence between overlapping
        # labels, and two folds hold few blocks: the bare 90% percentile interval
        # misses 0.50 about twice as often as it should. Widened by
        # interval_inflation it holds its level (20 of 200 when written).
        raw_miss = wide_miss = 0
        for seed in range(200):
            y, p, dates, groups = _noise_with_overlapping_labels(1000 + seed)
            raw = block_bootstrap_auc(y, p, dates, seed=seed, groups=groups, block_length=10)
            wide = block_bootstrap_auc(y, p, dates, seed=seed, groups=groups, block_length=10, horizon=10)
            assert wide[0] <= raw[0] and wide[1] >= raw[1]
            raw_miss += not raw[0] <= 0.5 <= raw[1]
            wide_miss += not wide[0] <= 0.5 <= wide[1]
        assert wide_miss <= 30                     # nominal 20 (10%)
        assert raw_miss >= wide_miss + 12

    def test_inflation_factor(self):
        # single dates see about 1/h of the variance of an h-session label
        assert model_eval.interval_inflation(21, 1, 5000) == pytest.approx(math.sqrt(21), rel=0.03)
        # blocks of h keep 2/3 of it; many blocks leave only that taper
        assert model_eval.interval_inflation(10, 10, 10**6) == pytest.approx(math.sqrt(1.5), rel=1e-3)
        # fewer blocks, wider; under MIN_INTERVAL_BLOCKS blocks, no interval at all
        factors = [model_eval.interval_inflation(21, 21, n) for n in (5000, 360, 120)]
        assert factors[0] < factors[1] < factors[2]
        assert math.isnan(model_eval.interval_inflation(21, 21, 83))
        for dof, table in ((3, 2.353363), (10, 1.812461), (30, 1.697261)):
            assert model_eval._t_quantile(0.95, dof) == pytest.approx(table, rel=1e-3)

    def test_horizon_widens_the_interval_around_the_point_estimate(self):
        y, p, dates = _overlapping_label_rows()
        groups = np.repeat([0, 1], y.size // 2)
        lo, hi = block_bootstrap_auc(y, p, dates, groups=groups, block_length=20)
        wide_lo, wide_hi = block_bootstrap_auc(y, p, dates, groups=groups, block_length=20, horizon=20)
        point = np.mean([same_date_auc(y[groups == g], p[groups == g], dates[groups == g]) for g in (0, 1)])
        factor = model_eval.interval_inflation(20, 20, np.unique(dates).size)
        assert factor > 1.2
        assert point - wide_lo == pytest.approx(factor * (point - lo))
        assert wide_hi - point == pytest.approx(factor * (hi - point))

    @pytest.mark.parametrize("joint", [True, False])
    def test_interval_widens_with_block_length_on_overlapping_labels(self, joint):
        y, p, dates = _overlapping_label_rows()
        groups = np.repeat([0, 1], y.size // 2)
        widths = []
        for length in (1, 5, 20, 60):
            lo, hi = block_bootstrap_auc(y, p, dates, n_boot=400, seed=1, groups=groups,
                                         block_length=length, joint=joint)
            widths.append(hi - lo)
        assert all(narrow < wide for narrow, wide in zip(widths, widths[1:]))
        # a block per label window: single dates had understated the spread severalfold
        assert widths[2] > 2.0 * widths[0]

    def test_group_too_short_for_two_blocks_gives_nan(self):
        y, p, dates = _overlapping_label_rows()
        groups = np.repeat([0, 1], y.size // 2)                 # 250 dates per group
        fits = block_bootstrap_auc(y, p, dates, groups=groups, block_length=125)
        assert all(math.isfinite(v) for v in fits)
        too_short = block_bootstrap_auc(y, p, dates, groups=groups, block_length=126)
        assert all(math.isnan(v) for v in too_short)
        assert all(math.isnan(v) for v in block_bootstrap_auc(y, p, dates, block_length=251))

    def test_a_short_group_makes_the_ship_rule_answer_prior(self):
        y, p, dates = _overlapping_label_rows()
        lo, hi = block_bootstrap_auc(y, p, dates, groups=np.repeat([0, 1], y.size // 2),
                                     block_length=200)
        confirm = {**PASSING, "auc_ci_low": lo, "auc_ci_high": hi}
        assert ship_decision(confirm) == "prior"
        assert any("CI" in reason for reason in ship_reasons(confirm))

    def test_circular_blocks(self):
        rng = np.random.default_rng(0)
        weights = model_eval._circular_block_weights(rng, 10, 4, 50)
        assert weights.shape == (50, 10) and (weights.sum(axis=1) == 10).all()
        # one block as long as the series wraps the whole circle once, wherever it starts
        assert (model_eval._circular_block_weights(rng, 10, 10, 20) == 1.0).all()
        # a block of one is the plain date bootstrap
        assert (model_eval._circular_block_weights(rng, 7, 1, 20).sum(axis=1) == 7).all()

    def test_block_length_follows_the_horizon(self):
        assert block_length_for(1) >= 1
        for h in (5, 21, 63, 252):
            assert block_length_for(h) == math.ceil(model_eval.BLOCK_LENGTH_MULT * h) >= h

    def test_integer_day_numbers_and_datetimes_agree(self):
        y, p, dates = _overlapping_label_rows(seed=3)
        days = pd.DatetimeIndex(dates).to_numpy().astype("datetime64[D]").astype(np.int64)
        kwargs = dict(n_boot=100, seed=2, block_length=20, groups=np.repeat([0, 1], y.size // 2))
        assert block_bootstrap_auc(y, p, dates, **kwargs) == block_bootstrap_auc(y, p, days, **kwargs)

    def test_interval_contains_the_point_estimate(self):
        y, p, dates = _scored_rows()
        point = same_date_auc(y, p, dates)
        lo, hi = block_bootstrap_auc(y, p, dates, n_boot=200, seed=0)
        assert 0.0 <= lo < point < hi <= 1.0
        assert hi - lo < 0.15
        # the across-dates statistic, kept for diagnostics, brackets sklearn's AUC
        lo, hi = block_bootstrap_auc(y, p, dates, n_boot=200, seed=0, same_date=False)
        assert lo < roc_auc_score(y, p) < hi

    def test_grouped_interval_brackets_the_fold_mean(self):
        y, p, dates = _scored_rows(seed=1)
        groups = np.repeat([0, 1, 2], y.size // 3)
        point = np.mean([same_date_auc(y[groups == g], p[groups == g], dates[groups == g])
                         for g in range(3)])
        lo, hi = block_bootstrap_auc(y, p, dates, seed=0, groups=groups)
        assert lo < point < hi

    def test_same_seed_same_interval(self):
        y, p, dates = _scored_rows(seed=2)
        assert block_bootstrap_auc(y, p, dates, seed=7) == block_bootstrap_auc(y, p, dates, seed=7)

    def test_single_class_gives_nan(self):
        lo, hi = block_bootstrap_auc([1, 1, 1, 1], [0.1, 0.4, 0.6, 0.9],
                                     ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"])
        assert math.isnan(lo) and math.isnan(hi)

    def test_unweighted_fast_auc_matches_sklearn(self):
        y, p, dates = _scored_rows(seed=4)
        p = np.round(p, 2)                              # force ties
        fast = model_eval._DateWeightedAUC(y.astype(float), p, model_eval._date_codes(dates))
        assert fast.auc() == pytest.approx(roc_auc_score(y, p), abs=1e-12)


# ---------------------------------------------------------------------------
# (d) calibration
# ---------------------------------------------------------------------------

def _under_confident(seed=0, n=2000):
    """Raw scores squeezed towards 0.5 relative to the true probability."""
    rng = np.random.default_rng(seed)
    true_p = rng.uniform(0.05, 0.95, size=n)
    y = (rng.random(n) < true_p).astype(int)
    return 0.5 + 0.3 * (true_p - 0.5), y


class TestCalibration:

    @pytest.mark.parametrize("method", ["platt", "isotonic"])
    def test_monotone_and_better_brier(self, method):
        p_raw, y = _under_confident()
        cal = fit_calibrator(p_raw, y, method)
        assert cal.method == method
        grid = np.linspace(0.0, 1.0, 101)
        assert (np.diff(cal.transform(grid)) >= -1e-12).all()
        brier_raw = np.mean((p_raw - y) ** 2)
        brier_cal = np.mean((cal.transform(p_raw) - y) ** 2)
        assert brier_cal < brier_raw

    def test_thin_or_one_class_evidence_answers_the_base_rate(self):
        p_raw, y = _under_confident()
        thin = fit_calibrator(p_raw[:49], y[:49], "platt", base_rate=0.55)
        assert thin.method == "base_rate"
        assert np.allclose(thin.transform(p_raw), 0.55)
        assert fit_calibrator(p_raw, np.ones_like(y), "isotonic", base_rate=0.6).method == "base_rate"
        identity = fit_calibrator(p_raw, y, "identity")
        assert identity.method == "identity"
        assert np.array_equal(identity.transform(p_raw), p_raw)
        assert fit_calibrator([], [], "platt").method == "identity"      # no rows, no anchor

    @pytest.mark.parametrize("method", ["platt", "isotonic"])
    def test_shrinks_towards_the_base_rate_as_evidence_thins(self, method):
        p_raw, y = _under_confident(seed=2)
        spread = {}
        for n_eff in (1.0, 20.0, 2000.0):
            cal = fit_calibrator(p_raw, y, method, base_rate=0.5, n_eff=n_eff)
            spread[n_eff] = float(np.std(cal.transform(p_raw)))
        assert spread[1.0] < spread[20.0] < spread[2000.0]
        assert spread[1.0] < 0.1 * spread[2000.0]

    def test_no_skill_scores_stay_near_the_base_rate(self):
        # an overconfident model with nothing to say, calibrated on thin
        # overlapping-label evidence, must not turn its noise into forecasts
        rng = np.random.default_rng(7)
        p_raw = 1.0 / (1.0 + np.exp(-rng.normal(0.0, 2.0, 5000)))
        y = (rng.random(5000) < 0.58).astype(int)
        cal = fit_calibrator(p_raw, y, "platt", base_rate=0.58, n_eff=15.0)
        p_new = 1.0 / (1.0 + np.exp(-rng.normal(0.0, 2.0, 5000)))
        y_new = (rng.random(5000) < 0.58).astype(int)
        skill = 1.0 - np.mean((cal.transform(p_new) - y_new) ** 2) / np.mean((0.58 - y_new) ** 2)
        assert skill > -0.01
        assert np.max(np.abs(cal.transform(p_new) - 0.58)) < 0.1
        raw_skill = 1.0 - np.mean((p_new - y_new) ** 2) / np.mean((0.58 - y_new) ** 2)
        assert raw_skill < -0.3                                         # what identity would serve (-0.44)

    def test_genuine_signal_keeps_most_of_its_slope(self):
        # well-calibrated scores (true slope 1) backed by 60 independent windows:
        # the prior of CALIBRATION_PRIOR_N_EFF windows trims the slope, it must
        # not erase it, and the calibrated forecasts must beat the base rate
        rng = np.random.default_rng(11)

        def sample(n):
            logit = math.log(0.56 / 0.44) + rng.normal(0.0, 0.8, n)
            return 1 / (1 + np.exp(-logit)), (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

        p_raw, y = sample(6000)
        cal = fit_calibrator(p_raw, y, "platt", base_rate=0.56, n_eff=60.0)
        assert 0.6 < cal.slope < 1.0
        p_new, y_new = sample(6000)
        skill = 1.0 - np.mean((cal.transform(p_new) - y_new) ** 2) / np.mean((0.56 - y_new) ** 2)
        assert skill > 0.5 * (1.0 - np.mean((p_new - y_new) ** 2) / np.mean((0.56 - y_new) ** 2)) > 0

    def test_scores_are_read_against_their_own_models_base_rate(self):
        # two fold models, trained on windows with different up-rates, whose raw
        # scores carry the same deviation: one slope must explain both
        rng = np.random.default_rng(4)
        deviation = rng.normal(0.0, 1.0, 4000)
        row_rate = np.where(np.arange(4000) < 2000, 0.45, 0.65)
        logit = np.log(row_rate / (1 - row_rate))
        y = (rng.random(4000) < 1 / (1 + np.exp(-(logit + 0.5 * deviation)))).astype(int)
        p_raw = 1 / (1 + np.exp(-(logit + deviation)))
        cal = fit_calibrator(p_raw, y, "platt", base_rate=0.55, oof_base_rate=row_rate)
        assert cal.slope == pytest.approx(0.5, abs=0.08)
        assert cal.transform([0.55])[0] == pytest.approx(0.55)

    def test_scores_centred_on_another_targets_rate_are_read_against_it(self):
        # a model fitted on a within-date target centres its scores on 0.5 while
        # the outcome it is judged on goes up 62% of the time: read against 0.5,
        # its deviations keep their slope and the forecasts keep the base rate
        rng = np.random.default_rng(3)
        deviation = rng.normal(0.0, 0.8, 6000)
        base = 0.62
        y = (rng.random(6000) < 1 / (1 + np.exp(-(math.log(base / (1 - base)) + 0.5 * deviation)))).astype(int)
        p_raw = 1 / (1 + np.exp(-deviation))
        cal = fit_calibrator(p_raw, y, "platt", base_rate=base, oof_base_rate=base, n_eff=500.0,
                             score_base_rate=0.5, oof_score_rate=0.5)
        assert cal.slope == pytest.approx(0.5, abs=0.1)
        assert cal.transform([0.5])[0] == pytest.approx(base)
        assert abs(cal.transform(p_raw).mean() - y.mean()) < 0.015
        isotonic = fit_calibrator(p_raw, y, "isotonic", base_rate=base, oof_base_rate=base, n_eff=500.0,
                                  score_base_rate=0.5, oof_score_rate=0.5)
        assert abs(isotonic.transform(p_raw).mean() - y.mean()) < 0.015
        # read against the base rate instead, the 0.5-centred scores drag every forecast down
        unanchored = fit_calibrator(p_raw, y, "platt", base_rate=base, oof_base_rate=base, n_eff=500.0)
        assert unanchored.transform(p_raw).mean() < y.mean() - 0.02

    def test_a_calibrator_pickled_before_score_rates_still_transforms(self):
        p_raw, _ = _under_confident(seed=5)
        current = Calibrator("platt", base_rate=0.57, slope=0.6)
        old = Calibrator.__new__(Calibrator)
        old.__dict__.update({k: v for k, v in current.__dict__.items() if k != "score_base_rate"})
        assert np.array_equal(old.transform(p_raw), current.transform(p_raw))
        assert "score_base_rate" not in repr(old)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            fit_calibrator([0.5] * 60, [0, 1] * 30, "beta")

    @pytest.mark.parametrize("method", ["platt", "isotonic", "identity"])
    def test_pickle_round_trip(self, method):
        p_raw, y = _under_confident(seed=1)
        cal = fit_calibrator(p_raw, y, method)
        clone = pickle.loads(pickle.dumps(cal))
        assert isinstance(clone, Calibrator)
        assert np.allclose(clone.transform(p_raw), cal.transform(p_raw))


# ---------------------------------------------------------------------------
# (e) evaluate, (f) shuffle control
# ---------------------------------------------------------------------------

class TestEvaluate:

    def test_signal_panel_has_skill_and_ships(self, signal_result):
        assert isinstance(signal_result, EvalResult)
        assert signal_result.confirm["auc_mean"] > 0.6
        assert signal_result.confirm["auc_ci_low"] > 0.5
        assert signal_result.confirm["brier_skill_mean"] > 0
        assert ship_decision(signal_result.confirm) == "model"

    def test_fold_bookkeeping_and_blocks(self, panel, folds, signal_result):
        r = signal_result
        assert (r.horizon, r.calibrator_method, r.confirm_folds) == (HORIZON, "platt", 2)
        assert (r.overall["n_folds"], r.confirm["n_folds"], r.selection["n_folds"]) == (N_FOLDS, 2, N_FOLDS - 2)
        assert r.confirm["auc_mean"] == pytest.approx(np.mean([row["auc"] for row in r.folds[-2:]]))
        assert r.selection["auc_mean"] == pytest.approx(np.mean([row["auc"] for row in r.folds[:-2]]))
        assert r.overall["n_total"] == sum(f.test_idx.size for f in folds)
        assert (r.n_tickers, len(r.per_ticker)) == (6, 6)
        used = np.unique(np.concatenate([np.concatenate([f.train_idx, f.test_idx]) for f in folds]))
        assert r.n_rows == used.size
        for row, fold in zip(r.folds, folds):
            assert row["prior_rate"] == pytest.approx(panel["y"][fold.train_idx].mean())
            assert (row["n_train"], row["n_test"]) == (fold.train_idx.size, fold.test_idx.size)
        for key in ("auc_ci_low", "auc_ci_high", "auc_pooled_mean", "brier_skill_mean",
                    "decile_spread_mean", "hi_conf_10_acc_pooled", "hi_conf_15_n_total", "reliability",
                    "n_eff_total", "block_length", "horizon"):
            assert key in r.overall and key in r.confirm
        assert r.overall["auc_ci_low"] < r.overall["auc_mean"] < r.overall["auc_ci_high"]
        assert r.confirm["block_length"] == block_length_for(HORIZON)
        confirm_dates = np.unique(_days(panel["dates"][np.concatenate([f.test_idx for f in folds[-2:]])]))
        assert r.confirm["n_eff_total"] == pytest.approx(confirm_dates.size / HORIZON)
        assert r.confirm["ci_inflation"] == pytest.approx(model_eval.interval_inflation(
            HORIZON, block_length_for(HORIZON), confirm_dates.size))
        assert r.confirm["n_eff_total"] >= MIN_CONFIRM_N_EFF

    def test_oof_arrays_line_up_with_the_panel(self, panel, folds, signal_result):
        oof = signal_result.oof
        assert np.array_equal(oof["idx"], np.concatenate([f.test_idx for f in folds]))
        assert np.array_equal(oof["y"], panel["y"][oof["idx"]].astype(int))
        assert np.array_equal(np.unique(oof["fold"]), np.arange(N_FOLDS))
        assert np.array_equal(oof["label_day"], _days(panel["label_dates"][oof["idx"]]))
        assert len({oof[k].size for k in ("idx", "p_raw", "p_cal", "y", "fold", "prior",
                                          "label_day")}) == 1

    def test_calibration_uses_only_earlier_folds_with_known_outcomes(self, panel, folds, signal_result):
        oof, rows = signal_result.oof, signal_result.folds
        # no earlier fold, no evidence: the first fold answers its base rate
        assert (rows[0]["calibrator"], rows[0]["n_calib"]) == ("base_rate", 0)
        assert np.allclose(oof["p_cal"][oof["fold"] == 0], rows[0]["prior_rate"])
        oof_days = _days(panel["dates"][oof["idx"]])
        for k in range(1, N_FOLDS):
            cutoff = (np.datetime64(folds[k].test_start.date(), "D")
                      - np.timedelta64(folds[k].embargo_days, "D"))
            earlier = (oof["fold"] < k) & (oof["label_day"] < cutoff)
            # contiguous blocks: the previous block's last sessions are labelled too late
            assert 0 < earlier.sum() < (oof["fold"] < k).sum()
            n_eff = np.unique(oof_days[earlier]).size / HORIZON
            refit = fit_calibrator(oof["p_raw"][earlier], oof["y"][earlier], "platt",
                                   base_rate=rows[k]["prior_rate"], oof_base_rate=oof["prior"][earlier],
                                   n_eff=n_eff)
            assert (rows[k]["calibrator"], rows[k]["n_calib"]) == ("platt", int(earlier.sum()))
            assert rows[k]["calib_n_eff"] == pytest.approx(n_eff)
            assert rows[k]["calib_slope"] == pytest.approx(refit.slope)
            assert np.allclose(refit.transform(oof["p_raw"][oof["fold"] == k]),
                               oof["p_cal"][oof["fold"] == k])

    def test_calibration_purge_drops_rows_labelled_inside_the_next_block(self):
        horizon, embargo, n_tickers = 5, 8, 6
        days = pd.bdate_range("2021-01-04", periods=260).to_numpy()
        label_one = np.full(days.size, np.datetime64("NaT"), dtype="datetime64[ns]")
        label_one[:-horizon] = days[horizon:]
        dates, label_dates = np.tile(days, n_tickers), np.tile(label_one, n_tickers)
        rng = np.random.default_rng(11)
        X = rng.normal(size=(dates.size, 2))
        y = (X[:, 0] + rng.normal(size=dates.size) > 0).astype(float)
        labeled = ~np.isnat(label_dates)

        def fold(index, first, last):
            cutoff = days[first] - np.timedelta64(embargo, "D")
            return model_eval.Fold(
                index=index, test_start=pd.Timestamp(days[first]), test_end=pd.Timestamp(days[last]),
                train_idx=np.flatnonzero(labeled & (label_dates < cutoff)),
                test_idx=np.flatnonzero(labeled & (dates >= days[first]) & (dates <= days[last])),
                horizon=horizon, embargo_days=embargo)

        def calibration_rows(earlier, later):
            cutoff = days[later] - np.timedelta64(embargo, "D")
            return int((label_dates[earlier.test_idx] < cutoff).sum())

        args = (X, y, None, dates, np.repeat([f"T{i}" for i in range(n_tickers)], days.size))
        for first_session, expected_method in ((180, "platt"), (200, "base_rate")):
            folds = [fold(0, first_session, 209), fold(1, 210, 235)]
            expected = calibration_rows(folds[0], 210)
            # the earlier block's last sessions are labelled on or after the later block's cutoff
            assert expected < folds[0].test_idx.size
            purged = evaluate(baseline_logreg_factory(), *args, folds, confirm_folds=1,
                              importance=False, label_dates=label_dates)
            unpurged = evaluate(baseline_logreg_factory(), *args, folds, confirm_folds=1,
                                importance=False)
            assert purged.folds[1]["n_calib"] == expected
            assert purged.folds[1]["calibrator"] == expected_method
            assert unpurged.folds[1]["n_calib"] == folds[0].test_idx.size
            assert unpurged.folds[1]["calibrator"] == "platt"
            assert np.isnat(unpurged.oof["label_day"]).all()
            earlier = purged.oof["fold"] == 0
            known = purged.oof["label_day"][earlier] < (
                np.datetime64(days[210], "D") - np.timedelta64(embargo, "D"))
            known_days = np.unique(dates[purged.oof["idx"][earlier][known]])
            refit = fit_calibrator(purged.oof["p_raw"][earlier][known], purged.oof["y"][earlier][known],
                                   "platt", base_rate=purged.folds[1]["prior_rate"],
                                   oof_base_rate=purged.folds[0]["prior_rate"],
                                   n_eff=known_days.size / horizon)
            assert refit.method == expected_method
            assert np.allclose(refit.transform(purged.oof["p_raw"][purged.oof["fold"] == 1]),
                               purged.oof["p_cal"][purged.oof["fold"] == 1])

    def test_permutation_importance_finds_the_signal(self, signal_result):
        importance = signal_result.importance
        assert [row["feature"] for row in importance] == [0, 1, 2]     # column order
        assert max(importance, key=lambda row: row["importance"])["feature"] == 0
        assert importance[0]["importance"] > 0.1

    def test_to_dict_is_json_serialisable(self, signal_result):
        payload = signal_result.to_dict()
        assert "oof" not in payload
        decoded = json.loads(json.dumps(payload, allow_nan=False))
        assert decoded["confirm"]["auc_mean"] == pytest.approx(signal_result.confirm["auc_mean"])
        assert decoded["folds"][0]["hi_conf_10"]["n"] == signal_result.folds[0]["hi_conf_10"]["n"]
        with_oof = signal_result.to_dict(include_oof=True)
        json.dumps(with_oof, allow_nan=False)
        assert len(with_oof["oof"]["p_raw"]) == signal_result.oof["p_raw"].size

    def test_summary_names_features_and_decision(self, signal_result):
        text = summarize_for_report(signal_result, ["signal", "noise_a", "noise_b"])
        assert "ship decision `model`" in text
        assert "| 1 | signal |" in text
        assert "Reliability" in text

    def test_pure_noise_ships_prior(self, folds):
        noise = make_panel(signal=0.0, seed=NOISE_SEED)
        result = evaluate(hgb_factory(DEFAULT_HGB), noise["X"], noise["y"], noise["fwd"],
                          noise["dates"], noise["tickers"], folds, importance=False,
                          label_dates=noise["label_dates"])
        assert result.importance == []
        assert result.confirm["auc_mean"] < 0.55
        assert ship_decision(result.confirm) == "prior"

    def test_shuffled_labels_score_chance(self, panel, folds):
        auc = shuffle_control(hgb_factory(DEFAULT_HGB), panel["X"], panel["y"], panel["fwd"],
                              panel["dates"], panel["tickers"], folds, seed=0)
        assert abs(auc - 0.5) <= 0.06

    def test_confirm_folds_clamped_when_folds_are_few(self, panel):
        two = walkforward_folds(panel["dates"], panel["label_dates"], HORIZON, n_folds=2)
        result = evaluate(baseline_logreg_factory(), panel["X"], panel["y"], panel["fwd"],
                          panel["dates"], panel["tickers"], two, confirm_folds=2, importance=False)
        assert result.confirm_folds == 1
        assert (result.confirm["n_folds"], result.selection["n_folds"]) == (1, 1)
        assert result.confirm["auc_mean"] > 0.6

    def test_weights_and_train_mask_reach_training_only(self, panel, folds):
        vol_scaled = panel["fwd"] / 0.01
        keep = deadband_mask(vol_scaled, 0.5)
        weights = weights_for("abs_ret_scaled", panel["fwd"], vol_scaled, panel["dates"],
                              half_life_years=2.0)
        result = evaluate(baseline_logreg_factory(), panel["X"], panel["y"], panel["fwd"],
                          panel["dates"], panel["tickers"], folds, sample_weight=weights,
                          train_mask=keep, calibration="isotonic", importance=False)
        for row, fold in zip(result.folds, folds):
            assert row["n_train"] == int(keep[fold.train_idx].sum()) < fold.train_idx.size
            assert row["n_test"] == fold.test_idx.size
            assert row["prior_rate"] == pytest.approx(panel["y"][fold.train_idx].mean())
        assert result.confirm["auc_mean"] > 0.6

    def test_single_class_training_predicts_its_up_rate(self, panel, folds):
        y = panel["y"].copy()
        y[folds[0].train_idx] = 1.0
        result = evaluate(baseline_logreg_factory(), panel["X"], y, panel["fwd"], panel["dates"],
                          panel["tickers"], folds[:2], confirm_folds=1, importance=False)
        assert np.allclose(result.oof["p_raw"][result.oof["fold"] == 0], 1.0)

    def test_unlabeled_row_inside_a_fold_raises(self, panel, folds):
        y = panel["y"].copy()
        y[folds[0].test_idx[0]] = np.nan
        with pytest.raises(ValueError, match="unlabeled"):
            evaluate(baseline_logreg_factory(), panel["X"], y, panel["fwd"], panel["dates"],
                     panel["tickers"], folds, importance=False)

    def test_y_fit_trains_the_models_while_everything_scored_reads_y(self, panel, folds):
        y_fit = (np.random.default_rng(5).random(panel["y"].size) < 0.3).astype(float)
        fitted = []
        result = evaluate(functools.partial(_TrainingLabelRecorder, fitted), panel["X"], panel["y"],
                          panel["fwd"], panel["dates"], panel["tickers"], folds, importance=False,
                          label_dates=panel["label_dates"], y_fit=y_fit)
        assert len(fitted) == len(folds)
        oof = result.oof
        assert np.array_equal(oof["y"], panel["y"][oof["idx"]].astype(int))
        for pos, (labels, fold, row) in enumerate(zip(fitted, folds, result.folds)):
            assert np.array_equal(labels, y_fit[fold.train_idx].astype(int))
            assert row["prior_rate"] == pytest.approx(panel["y"][fold.train_idx].mean())
            in_fold = oof["fold"] == pos
            assert np.allclose(oof["prior"][in_fold], panel["y"][fold.train_idx].mean())
            assert np.allclose(oof["score_rate"][in_fold], y_fit[fold.train_idx].mean())
        # a model that answers its own target's up-rate says nothing: calibrated, it is
        # the absolute base rate, not the 30% its training target went up
        assert np.allclose(oof["p_raw"], oof["score_rate"])
        assert np.allclose(oof["p_cal"], oof["prior"])
        assert result.confirm["brier_skill_mean"] == pytest.approx(0.0, abs=1e-9)

    def test_y_fit_equal_to_y_changes_nothing(self, panel, folds):
        args = (panel["X"], panel["y"], panel["fwd"], panel["dates"], panel["tickers"], folds)
        plain = evaluate(baseline_logreg_factory(), *args, importance=False, label_dates=panel["label_dates"])
        same = evaluate(baseline_logreg_factory(), *args, importance=False, label_dates=panel["label_dates"],
                        y_fit=panel["y"].copy())
        for key in ("p_raw", "p_cal", "prior", "score_rate"):
            assert np.array_equal(plain.oof[key], same.oof[key])
        assert np.array_equal(plain.oof["prior"], plain.oof["score_rate"])
        assert [r["calib_slope"] for r in plain.folds] == [r["calib_slope"] for r in same.folds]

    def test_y_fit_rows_without_a_label_must_be_masked_out_of_training(self, panel, folds):
        y_fit = panel["y"].copy()
        y_fit[folds[1].train_idx[:10]] = np.nan
        y_fit[folds[1].train_idx[10]] = 1.0 - y_fit[folds[1].train_idx[10]]
        args = (panel["X"], panel["y"], panel["fwd"], panel["dates"], panel["tickers"], folds[:2])
        with pytest.raises(ValueError, match="y_fit"):
            evaluate(baseline_logreg_factory(), *args, confirm_folds=1, importance=False, y_fit=y_fit)
        result = evaluate(baseline_logreg_factory(), *args, confirm_folds=1, importance=False,
                          y_fit=y_fit, train_mask=np.isfinite(y_fit))
        assert result.folds[1]["n_train"] == folds[1].train_idx.size - 10
        assert result.folds[1]["prior_rate"] == pytest.approx(panel["y"][folds[1].train_idx].mean())


class TestBaselines:

    def test_prior_baseline_has_no_skill(self, panel, folds):
        result = baseline_prior(panel["y"], panel["fwd"], panel["dates"], folds, HORIZON)
        json.dumps(result, allow_nan=False)
        assert (result["name"], len(result["folds"]), result["confirm_folds"]) == ("prior", N_FOLDS, 2)
        for row in result["folds"]:
            assert row["auc"] == 0.5
            assert row["brier_skill"] == pytest.approx(0.0, abs=1e-12)
            assert row["acc_skill"] == pytest.approx(0.0, abs=1e-12)
        assert {"auc_mean", "auc_ci_low", "brier_skill_mean", "decile_spread_mean"} <= set(result["confirm"])
        assert ship_decision(result["confirm"]) == "prior"

    def test_momentum_baseline_reads_its_column(self, panel, folds):
        args = (panel["y"], panel["fwd"], panel["dates"], folds, HORIZON)
        assert baseline_momentum(panel["X"], *args, 0)["confirm"]["auc_mean"] > 0.6
        assert abs(baseline_momentum(panel["X"], *args, 1)["confirm"]["auc_mean"] - 0.5) < 0.05
        blank = panel["X"].copy()
        blank[:, 0] = np.nan
        result = baseline_momentum(blank, *args, 0)
        assert result["overall"]["auc_mean"] == 0.5
        assert all(row["hi_conf_10"]["n"] == 0 for row in result["folds"])


class TestPerTickerClassifier:

    @staticmethod
    def _data():
        rng = np.random.default_rng(5)
        x = rng.normal(size=(420, 2))
        y = np.r_[(x[:400, 0] > 0).astype(int), np.ones(15, dtype=int), np.zeros(5, dtype=int)]
        x[::7, 1] = np.nan                                   # the imputer has to cope
        codes = np.r_[np.zeros(400), np.ones(20)]
        return np.column_stack([x, codes]), y

    def test_routes_rows_by_ticker_code(self):
        X, y = self._data()
        factory = per_ticker_factory(gbm_factory({"n_estimators": 20, "max_depth": 2, "random_state": 0}),
                                     ticker_col=2, min_rows=50)
        model = factory().fit(X, y, sample_weight=np.ones(y.size))
        proba = model.predict_proba(X)
        assert proba.shape == (420, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert roc_auc_score(y[:400], proba[:400, 1]) > 0.9       # ticker 0 has its own model
        assert np.allclose(proba[400:, 1], 0.75)                   # ticker 1: too few rows, own up-rate
        unseen = X[:3].copy()
        unseen[:, 2] = 7
        assert np.allclose(model.predict_proba(unseen)[:, 1], model.prior_)
        assert np.allclose(pickle.loads(pickle.dumps(model)).predict_proba(X), proba)


# ---------------------------------------------------------------------------
# (g) weights and dead-band
# ---------------------------------------------------------------------------

class TestWeightsAndDeadband:

    def test_abs_ret_scaled_clips_and_fills(self):
        v = np.array([0.0, 0.05, -0.5, 5.0, np.nan, -2.0])
        w = weights_for("abs_ret_scaled", np.zeros(6), v, pd.bdate_range("2024-01-01", periods=6))
        assert np.allclose(w, [0.1, 0.1, 0.5, 3.0, 1.0, 2.0])

    def test_decay_halves_every_half_life(self):
        dates = np.array(["2026-01-01", "2023-01-01", "2020-01-01"], dtype="datetime64[D]")
        decay = weights_for("none", np.zeros(3), None, dates, half_life_years=3.0)
        assert decay == pytest.approx([1.0, 0.5, 0.25], rel=2e-3)
        both = weights_for("abs_ret_scaled", np.zeros(3), np.array([2.0, 2.0, 0.0]), dates,
                           half_life_years=3.0)
        assert both == pytest.approx([2.0, 1.0, 0.025], rel=2e-3)

    def test_none_without_decay_is_unweighted(self):
        assert weights_for("none", np.zeros(4), None, pd.bdate_range("2024-01-01", periods=4)) is None

    @pytest.mark.parametrize("scheme,vol,half_life", [
        ("sqrt", np.ones(3), None),
        ("abs_ret_scaled", None, None),
        ("none", None, 0.0),
    ])
    def test_invalid_arguments_raise(self, scheme, vol, half_life):
        with pytest.raises(ValueError):
            weights_for(scheme, np.zeros(3), vol, pd.bdate_range("2024-01-01", periods=3),
                        half_life_years=half_life)

    def test_deadband_mask(self):
        v = np.array([0.05, -0.05, 0.1, -0.2, np.nan])
        assert deadband_mask(v, 0.1).tolist() == [False, False, True, True, True]
        assert deadband_mask(v, 0.0).all()


# ---------------------------------------------------------------------------
# (h) hgb_factory
# ---------------------------------------------------------------------------

class TestHgbFactory:

    def test_drops_max_features_when_sklearn_lacks_it(self, monkeypatch):
        monkeypatch.setattr(model_eval, "_HGB_ACCEPTED_PARAMS",
                            model_eval._HGB_ACCEPTED_PARAMS - {"max_features"})
        model = hgb_factory({**DEFAULT_HGB, "max_features": 0.5})()
        assert isinstance(model, HistGradientBoostingClassifier)
        assert model.get_params().get("max_features") != 0.5
        assert model.max_iter == DEFAULT_HGB["max_iter"]

    def test_keeps_supported_params_and_categoricals(self):
        if "max_features" not in model_eval._HGB_ACCEPTED_PARAMS:
            pytest.skip("this scikit-learn has no max_features")
        model = hgb_factory({**DEFAULT_HGB, "max_features": 0.5}, categorical_idx=[1])()
        assert model.max_features == 0.5
        assert list(model.categorical_features) == [1]

    def test_no_categoricals_means_none(self):
        assert hgb_factory(DEFAULT_HGB, [])().categorical_features is None
        assert hgb_factory(DEFAULT_HGB)().categorical_features is None

    def test_typo_still_raises(self):
        with pytest.raises(TypeError):
            hgb_factory({"max_leaf_node": 7})()

    def test_factories_pickle(self):
        for factory in (hgb_factory(DEFAULT_HGB, [0]), baseline_logreg_factory(0.5),
                        per_ticker_factory(gbm_factory({"n_estimators": 5}), 3)):
            pickle.loads(pickle.dumps(factory))()

    def test_every_hgb_config_builds(self):
        for cfg in CONFIGS.values():
            if cfg.model == "hgb":
                assert isinstance(hgb_factory(cfg.params)(), HistGradientBoostingClassifier)


# ---------------------------------------------------------------------------
# (i) ship_decision
# ---------------------------------------------------------------------------

PASSING = {"auc_mean": 0.53, "auc_ci_low": 0.5001, "brier_skill_mean": 1e-6,
           "decile_spread_mean": 1e-6, "n_eff_total": MIN_CONFIRM_N_EFF, "horizon": 63}


class TestShipDecision:

    def test_passes_exactly_at_the_auc_floor_and_the_evidence_minimum(self):
        assert ship_decision(PASSING) == "model"
        assert ship_reasons(PASSING) == []

    @pytest.mark.parametrize("key,value", [
        ("auc_mean", 0.5299),
        ("auc_ci_low", 0.50),
        ("brier_skill_mean", 0.0),
        ("decile_spread_mean", 0.0),
        ("n_eff_total", MIN_CONFIRM_N_EFF - 0.01),
        ("auc_mean", float("nan")),
        ("auc_ci_low", None),
        ("brier_skill_mean", float("nan")),
        ("decile_spread_mean", None),
        ("n_eff_total", None),
    ])
    def test_any_failing_or_missing_condition_serves_the_prior(self, key, value):
        assert ship_decision({**PASSING, key: value}) == "prior"
        assert len(ship_reasons({**PASSING, key: value})) == 1

    def test_too_few_independent_windows_refuses_even_a_strong_confirm_block(self):
        strong = {**PASSING, "auc_mean": 0.64, "auc_ci_low": 0.60, "brier_skill_mean": 0.05,
                  "decile_spread_mean": 0.02, "n_eff_total": 2.8}
        assert ship_decision(strong) == "prior"
        assert ship_reasons(strong) == [
            f"n_eff 2.8 < {MIN_CONFIRM_N_EFF:g}: too few independent 63-session windows to measure"]

    def test_reasons_name_every_failed_condition(self):
        failing = {"auc_mean": 0.512, "auc_ci_low": 0.49, "brier_skill_mean": -0.021,
                   "decile_spread_mean": -0.004, "n_eff_total": 2.8, "horizon": 63}
        reasons = ship_reasons(failing)
        assert len(reasons) == 5
        assert reasons[0].startswith("n_eff 2.8 < ") and "63-session windows" in reasons[0]
        assert reasons[1] == "AUC 0.512 < 0.53"
        assert reasons[2] == "AUC CI low 0.490 <= 0.50"
        assert reasons[3] == "Brier skill -0.0210 <= 0"
        assert reasons[4] == "decile spread -0.0040 <= 0"

    def test_missing_keys(self):
        assert ship_decision({}) == "prior"
        assert len(ship_reasons({})) == 5
        assert ship_decision({k: v for k, v in PASSING.items() if k != "decile_spread_mean"}) == "prior"
        assert ship_reasons({k: v for k, v in PASSING.items() if k != "n_eff_total"})[0].startswith(
            "n_eff not reported")

    def test_min_auc_is_configurable(self):
        assert ship_decision(PASSING, min_auc=0.55) == "prior"
        assert ship_decision({**PASSING, "auc_mean": 0.515}, min_auc=0.51) == "model"
        assert ship_reasons(PASSING, min_n_eff=MIN_CONFIRM_N_EFF + 1)[0].startswith("n_eff")


# ---------------------------------------------------------------------------
# model_configs
# ---------------------------------------------------------------------------

FAKE_GROUPS = {
    "price": ["ret_1d", "rsi_14", "mom_12_1"],
    "market": ["vix_level", "spy_ret_5d"],
    "regime": ["dix_z60"],
    "smart_money": ["offexch_share_z20"],
    "sentiment": ["sent_1d"],
    "calendar": ["dow"],
    "context": ["sector_id"],
}


class TestModelConfigs:

    def test_expected_configs_exist_and_serialise(self):
        expected = {"run0", "run1", "run1b", "run2", "run3", "run4a", "run4b", "run4c", "run4d",
                    "run6a", "run6b", "run6c", "run6d", "run6e", "run6f", "run6g", "logreg_cs",
                    "logreg", "prior", "momentum"}
        assert expected <= set(CONFIGS)
        for name, cfg in CONFIGS.items():
            assert cfg.name == name
            assert set(cfg.feature_groups) <= set(FEATURE_GROUP_NAMES)
            json.dumps(cfg.to_dict())

    def test_experiment_definitions(self):
        assert CONFIGS["run0"].model == "gbm_per_ticker"
        assert CONFIGS["run0"].features == V3_LIKE_FEATURES
        assert CONFIGS["run1b"].universe == "backbone"
        assert CONFIGS["run2"].feature_groups == RUN2_GROUPS
        assert CONFIGS["run3"].test_window == "coverage_era"
        assert "sentiment" in CONFIGS["run3"].feature_groups
        assert CONFIGS["run4a"].weight_scheme == "abs_ret_scaled"
        assert CONFIGS["run4b"].deadband == 0.1
        assert CONFIGS["run4c"].decay_half_life_years == 3.0
        assert CONFIGS["run4d"].label == "excess_spy"
        assert (CONFIGS["logreg"].model, CONFIGS["logreg"].feature_groups) == ("logreg", RUN2_GROUPS)
        assert CONFIGS["run2"].params == DEFAULT_HGB
        assert CONFIGS["run1"].params is not CONFIGS["run2"].params
        assert all(cfg.min_train_days == 504 for cfg in CONFIGS.values())

    def test_production_mapping(self):
        assert set(PRODUCTION) == {5, 21, 63, 252}
        for h, name in PRODUCTION.items():
            assert get_config(name) is CONFIGS[name]
            assert config_for_horizon(h) is CONFIGS[name]
            assert config_for_horizon(h).model in model_training.PRODUCTION_MODELS
        with pytest.raises(KeyError):
            config_for_horizon(10)
        with pytest.raises(KeyError, match="unknown model config"):
            get_config("run99")

    def test_production_sweep_picks_are_the_run1_sweep_points(self):
        # picked by selection AUC on the 2026-09-14 snapshot protocol; 252d has no pick
        picks = {5: "run5_lr0.06_leaves31_msl100_it400", 21: "run5_lr0.03_leaves15_msl100_it200",
                 63: "run5_lr0.06_leaves7_msl300_it200"}
        assert PRODUCTION == {**picks, 252: "run1"}
        sweep = {cfg.name: cfg for cfg in sweep_configs(get_config("run1"), SWEEP_GRID_SMALL)}
        for name in picks.values():
            registered = get_config(name)
            assert registered == sweep[name]
            assert registered.to_dict() == sweep[name].to_dict()
        assert get_config("run5_lr0.06_leaves7_msl300_it200").params == {
            **DEFAULT_HGB, "learning_rate": 0.06, "max_leaf_nodes": 7, "min_samples_leaf": 300,
            "max_iter": 200}

    def test_v3_like_features_are_unique(self):
        assert len(V3_LIKE_FEATURES) == len(set(V3_LIKE_FEATURES)) >= 30

    @pytest.mark.parametrize("field_name,value", [
        ("feature_groups", ("prices",)),
        ("model", "xgboost"),
        ("weight_scheme", "sqrt"),
        ("universe", "world"),
        ("test_window", "recent"),
        ("n_folds", 1),
        ("deadband", -0.1),
        ("test_days", 0),
        ("confirm_folds", 0),
        ("confirm_folds", 6),
        ("label", "rank"),
        ("train_stride", 0),
        ("train_stride", "fast"),
        ("train_stride", True),
        ("train_stride", 2.5),
    ])
    def test_invalid_config_raises(self, field_name, value):
        with pytest.raises(ValueError):
            ModelConfig(**{"name": "bad", "feature_groups": ("price",), field_name: value})

    def test_run6_family_definitions(self):
        family = ("run6a", "run6b", "run6c", "run6d", "run6e", "run6f", "run6g", "logreg_cs")
        for name in family:
            cfg = CONFIGS[name]
            assert (cfg.train_stride, cfg.test_days, cfg.test_window) == ("auto", None, "all")
            assert cfg.label == ("abs" if name == "run6f" else "cs_median")
            assert cfg.universe == ("backbone" if name == "run6d" else "core")
            assert cfg.params == ({"C": 0.1} if name == "logreg_cs" else
                                  DEFAULT_HGB if name == "run6a" else STRONG_HGB)
        assert CONFIGS["run6a"].feature_groups == CONFIGS["run6b"].feature_groups == RUN1_GROUPS
        assert CONFIGS["run6f"].feature_groups == CONFIGS["run6d"].feature_groups == RUN1_GROUPS
        assert CONFIGS["run6e"].feature_groups == RUN2_GROUPS
        assert CONFIGS["run6g"].feature_groups == ("price", "market", "calendar")
        assert CONFIGS["run6c"].features == CONFIGS["logreg_cs"].features == FACTOR_FEATURES
        assert CONFIGS["logreg_cs"].model == "logreg"
        assert STRONG_HGB["max_features"] == 0.5 and STRONG_HGB["min_samples_leaf"] == 1000

    def test_new_configs_resolve_to_feature_columns(self):
        assert len(set(FACTOR_FEATURES)) == len(FACTOR_FEATURES) == 12
        assert set(FACTOR_FEATURES) <= set(FEATURE_NAMES)
        assert not set(FACTOR_FEATURES) & set(FEATURE_GROUPS["context"])
        for name in ("run6a", "run6b", "run6c", "run6d", "run6e", "run6f", "run6g", "logreg_cs"):
            columns = resolve_columns(CONFIGS[name], FEATURE_GROUPS)
            assert columns and len(set(columns)) == len(columns) and set(columns) <= set(FEATURE_NAMES)
        assert resolve_columns(CONFIGS["run6c"], FEATURE_GROUPS) == [c for c in FEATURE_NAMES
                                                                    if c in FACTOR_FEATURES]
        without_context = resolve_columns(CONFIGS["run6g"], FEATURE_GROUPS)
        assert not set(without_context) & set(FEATURE_GROUPS["context"])
        assert set(without_context) == set(FEATURE_GROUPS["price"] + FEATURE_GROUPS["market"]
                                            + FEATURE_GROUPS["calendar"])

    def test_train_stride_for(self):
        auto = CONFIGS["run6b"]
        assert [train_stride_for(auto, h) for h in (1, 4, 5, 21, 63, 252)] == [1, 1, 1, 4, 12, 50]
        assert {train_stride_for(CONFIGS["run2"], h) for h in (5, 21, 63, 252)} == {1}
        assert train_stride_for(ModelConfig(name="every_third", feature_groups=("price",),
                                            train_stride=3), 252) == 3
        assert json.loads(json.dumps(auto.to_dict()))["train_stride"] == "auto"

    def test_resolve_columns_by_group_and_by_name(self):
        assert resolve_columns(CONFIGS["run1"], FAKE_GROUPS) == [
            "ret_1d", "rsi_14", "mom_12_1", "vix_level", "spy_ret_5d", "dow", "sector_id"]
        named = ModelConfig(name="named", feature_groups=("price",), features=("vix_level", "ret_1d"))
        assert resolve_columns(named, FAKE_GROUPS) == ["ret_1d", "vix_level"]
        assert resolve_columns(CONFIGS["prior"], FAKE_GROUPS) == []
        with pytest.raises(KeyError):
            resolve_columns(CONFIGS["run0"], FAKE_GROUPS)
        assert resolve_columns(CONFIGS["run0"], FAKE_GROUPS, strict=False) == [
            "ret_1d", "rsi_14", "vix_level", "spy_ret_5d", "dix_z60", "offexch_share_z20", "sent_1d"]

    def test_default_geometry_is_planned_per_horizon(self):
        assert all(cfg.test_days is None for name, cfg in CONFIGS.items() if name != "run3")
        assert CONFIGS["run3"].test_days == 21
        json.dumps(CONFIGS["run2"].to_dict())

    def test_sweep_grid_is_the_full_product(self):
        sweep = sweep_configs(CONFIGS["run2"])
        assert len(sweep) == math.prod(len(v) for v in SWEEP_GRID_SMALL.values()) == 36
        assert len({cfg.name for cfg in sweep}) == 36
        for cfg in sweep:
            assert cfg.feature_groups == RUN2_GROUPS
            assert cfg.params["l2_regularization"] == DEFAULT_HGB["l2_regularization"]
            assert cfg.params["learning_rate"] in SWEEP_GRID_SMALL["learning_rate"]


# ---------------------------------------------------------------------------
# fold plan: horizon-aware geometry, through to evaluate_config
# ---------------------------------------------------------------------------

def _label_panel(start: str, end: str, horizon: int, n_tickers: int = 2, seed: int = 0) -> pd.DataFrame:
    """The panel columns build_design needs for a calendar-only or prior config, weekdays only."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, end)
    label = pd.Series(days).shift(-horizon).to_numpy(dtype="datetime64[ns]")
    parts = []
    for i in range(n_tickers):
        fwd = rng.normal(0.001 * horizon, 0.02 * math.sqrt(horizon), days.size)
        fwd[-horizon:] = np.nan
        parts.append(pd.DataFrame({
            "ticker": f"T{i}", "date": days, "dow": days.dayofweek.to_numpy(dtype=float),
            "days_to_month_end": rng.integers(0, 22, days.size).astype(float), "vol_21": 0.02,
            f"fwd_ret_{horizon}": fwd, f"y_{horizon}": np.where(np.isnan(fwd), np.nan, (fwd > 0) * 1.0),
            f"label_date_{horizon}": label,
        }))
    return pd.concat(parts, ignore_index=True).sort_values(["date", "ticker"], kind="stable")


class TestFoldPlan:

    @pytest.mark.parametrize("horizon", [5, 21, 63])
    def test_planned_confirm_folds_hold_the_evidence_minimum(self, horizon):
        plan = fold_plan(get_config("run2"), horizon)
        assert (plan.n_folds, plan.confirm_folds) == (6, 2)
        confirm_sessions = plan.confirm_folds * plan.test_days / CALENDAR_DAYS_PER_SESSION
        assert confirm_sessions >= MIN_CONFIRM_N_EFF * horizon
        # the smallest whole half-year block that does it
        assert plan.test_days == 126 or (plan.confirm_folds * (plan.test_days - 126)
                                         / CALENDAR_DAYS_PER_SESSION < MIN_CONFIRM_N_EFF * horizon)
        # and every confirm fold holds two bootstrap blocks
        assert plan.test_days / CALENDAR_DAYS_PER_SESSION >= 2 * block_length_for(horizon)

    def test_the_one_year_horizon_is_capped_below_the_minimum(self):
        plan = fold_plan(get_config("run2"), 252)
        assert plan.test_days == PLAN_MAX_TEST_DAYS == planned_test_days(252)
        assert plan.confirm_folds * plan.test_days / CALENDAR_DAYS_PER_SESSION < MIN_CONFIRM_N_EFF * 252

    def test_explicit_test_days_are_kept_at_every_horizon(self):
        assert {tuple(fold_plan(CONFIGS["run3"], h)) for h in (5, 21, 63, 252)} == {(3, 21, 2)}
        fixed = ModelConfig(name="fixed", feature_groups=("price",), n_folds=4, test_days=90,
                            confirm_folds=1)
        assert tuple(fold_plan(fixed, 63)) == (4, 90, 1)

    @pytest.mark.parametrize("horizon", [5, 21, 63])
    def test_long_history_gives_the_confirm_folds_enough_windows(self, horizon):
        cfg = get_config("run2")
        plan = fold_plan(cfg, horizon)
        panel = _label_panel("1999-01-04", "2026-06-30", horizon, n_tickers=1)
        dates = panel["date"].to_numpy()
        folds = walkforward_folds(dates, panel[f"label_date_{horizon}"].to_numpy(), horizon,
                                  n_folds=plan.n_folds, test_days=plan.test_days,
                                  min_train_days=cfg.min_train_days)
        assert len(folds) == plan.n_folds
        confirm = np.concatenate([f.test_idx for f in folds[-plan.confirm_folds:]])
        assert np.unique(dates[confirm]).size / horizon >= MIN_CONFIRM_N_EFF

    def test_evaluate_config_uses_the_plan_end_to_end(self):
        cfg = ModelConfig(name="base_rate_63", feature_groups=(), model="prior")
        panel = _label_panel("1999-01-04", "2026-06-30", 63)
        ev = model_training.evaluate_config(panel, None, cfg, 63, importance=False, n_boot=50)
        assert ev["status"] == "evaluated" and ev["plan"] == fold_plan(cfg, 63)
        assert len(ev["folds"]) == 6
        confirm = ev["result"]["confirm"]
        assert confirm["n_folds"] == 2 and confirm["n_eff_total"] >= MIN_CONFIRM_N_EFF
        assert ev["folds"][-1].test_end - ev["folds"][-1].test_start == pd.Timedelta(days=504 - 1)
        # a constant forecast fails on skill, not on evidence
        assert ev["decision"] == "prior"
        assert ev["reasons"] and not any(reason.startswith("n_eff") for reason in ev["reasons"])

    def test_short_history_is_cleanly_not_measurable(self):
        cfg = ModelConfig(name="calendar_63", feature_groups=("calendar",), model="logreg")
        panel = _label_panel("2019-01-02", "2022-06-30", 63)
        ev = model_training.evaluate_config(panel, None, cfg, 63, importance=False)
        assert ev["status"] == "not_measurable"
        assert (ev["decision"], ev["reasons"], ev["result"]) == (None, None, None)
        assert "usable fold" in ev["error"] and "504-day test block" in ev["error"]

    def test_too_short_confirm_blocks_say_so_in_the_reasons(self):
        cfg = ModelConfig(name="short_blocks_63", feature_groups=(), model="prior", test_days=126)
        panel = _label_panel("1999-01-04", "2026-06-30", 63)
        ev = model_training.evaluate_config(panel, None, cfg, 63, importance=False, n_boot=50)
        assert ev["status"] == "evaluated" and ev["decision"] == "prior"
        assert ev["reasons"][0].startswith("n_eff ") and "63-session windows" in ev["reasons"][0]


# ---------------------------------------------------------------------------
# training targets and the training stride: build_design, evaluate_config, fit_final
# ---------------------------------------------------------------------------

CS_HORIZON = 5
N_CS_TICKERS = 10
# Until THIN_UNTIL only the first N_EARLY_TICKERS tickers trade: too few for a median.
N_EARLY_TICKERS = 4
THIN_UNTIL = "2013-06-28"


def _cross_section_panel(seed: int = 0) -> tuple[pd.DataFrame, pd.DatetimeIndex, np.ndarray]:
    """
    Weekday panel for build_design, plus its dates and the market close series.

    Every forward return is the market's h-session log return (drifting up, so
    about 60% of rows close higher) plus a days_to_month_end effect every
    ticker shares and noise: the ranking signal lives within each date.
    """
    rng = np.random.default_rng(seed)
    h = CS_HORIZON
    days = pd.bdate_range("2012-01-02", "2020-12-31")
    label = pd.Series(days).shift(-h).to_numpy(dtype="datetime64[ns]")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0012, 0.01, days.size)))
    market = np.full(days.size, np.nan)
    market[:-h] = np.log(close[h:] / close[:-h])
    parts = []
    for i in range(N_CS_TICKERS):
        dtm = rng.integers(0, 22, days.size).astype(float)
        fwd = market + 0.006 * (dtm - 10.5) / 6.0 + rng.normal(0.0, 0.01, days.size)
        frame = pd.DataFrame({
            "ticker": f"T{i:02d}", "date": days, "dow": days.dayofweek.to_numpy(dtype=float),
            "days_to_month_end": dtm, "vol_21": 0.02,
            f"fwd_ret_{h}": fwd, f"y_{h}": np.where(np.isnan(fwd), np.nan, (fwd > 0) * 1.0),
            f"label_date_{h}": label,
        })
        if i >= N_EARLY_TICKERS:
            frame = frame[frame["date"] > pd.Timestamp(THIN_UNTIL)]
        parts.append(frame)
    panel = pd.concat(parts, ignore_index=True).sort_values(["date", "ticker"], kind="stable")
    return panel.reset_index(drop=True), days, close


def _target_config(name: str = "cs_every_4th", **overrides) -> ModelConfig:
    fields = dict(name=name, feature_groups=("calendar",), model="logreg", params={"C": 1.0},
                  label="cs_median", train_stride=4)
    fields.update(overrides)
    return ModelConfig(**fields)


def _recording_factory(log):
    """A stand-in for model_training.factory_for whose estimators record their training labels."""
    return lambda cfg, design: functools.partial(_TrainingLabelRecorder, log)


@pytest.fixture(scope="module")
def cross_section():
    panel, days, close = _cross_section_panel()
    return {"panel": panel, "days": days, "close": close}


@pytest.fixture(scope="module")
def target_evals(cross_section):
    """cs_median on every 4th session, and the same model on the absolute label and every session."""
    configs = {"cs": _target_config(),
               "abs": _target_config("abs_every_session", label="abs", train_stride=1)}
    return {key: model_training.evaluate_config(cross_section["panel"], None, cfg, CS_HORIZON,
                                                n_threads=1, importance=False, n_boot=50)
            for key, cfg in configs.items()}


class TestTrainingTargets:

    def test_abs_label_trains_on_y_itself(self, cross_section):
        design = model_training.build_design(cross_section["panel"],
                                             _target_config("plain", label="abs", train_stride=1), CS_HORIZON)
        assert design.y_train is design.y and design.fwd_train is design.fwd and design.trains_on_y
        assert design.train_mask is None and design.train_stride == 1 and design.label == "abs"

    def test_cs_median_is_balanced_within_each_date_and_drops_thin_dates_from_training_only(self, cross_section):
        panel = cross_section["panel"]
        design = model_training.build_design(panel, _target_config(train_stride=1), CS_HORIZON)
        labeled = panel[panel[f"y_{CS_HORIZON}"].notna()]
        # every labeled row stays in the design, scored on its absolute label
        assert design.n_rows == len(labeled)
        assert np.array_equal(design.y, labeled[f"y_{CS_HORIZON}"].to_numpy(dtype=float))
        assert np.array_equal(design.fwd, labeled[f"fwd_ret_{CS_HORIZON}"].to_numpy(dtype=float))
        assert not design.trains_on_y and design.label == "cs_median"
        rows = pd.DataFrame({"date": design.dates, "y_train": design.y_train, "train": design.train_mask})
        by_date = rows.groupby("date")
        n = by_date.size()
        thick = n >= model_training.MIN_CS_TICKERS
        assert thick.any() and (~thick).any()
        assert (by_date["y_train"].sum()[thick] == n[thick] // 2).all()
        assert by_date["y_train"].apply(lambda s: s.isna().all())[~thick].all()
        assert not by_date["train"].any()[~thick].any()
        assert by_date["train"].all()[thick].all()
        assert np.nanmean(design.y_train) == pytest.approx(0.5)
        assert design.y.mean() > 0.55

    def test_cs_median_ties_are_down_and_thin_dates_have_no_label(self):
        dates = np.repeat(np.array(["2020-01-02", "2020-01-03"], dtype="datetime64[ns]"), [9, 8])
        fwd = np.r_[np.arange(9) * 0.01, [0.01] * 3 + [0.02] * 5]
        out = model_training.cross_sectional_return(fwd, dates, min_rows=8)
        assert out[4] == 0.0 and (out[:9] > 0).sum() == 4                 # the median row is not up
        assert not (out[9:] > 0).any()                                    # five rows tie on the median
        assert np.isnan(model_training.cross_sectional_return(fwd, dates, min_rows=9)[9:]).all()

    def test_stride_keeps_whole_cross_sections_of_every_kth_session(self, cross_section):
        panel = cross_section["panel"]
        design = model_training.build_design(panel, _target_config("abs_every_4th", label="abs"),
                                             CS_HORIZON)
        assert design.trains_on_y and design.train_stride == 4
        sessions = np.unique(panel["date"].to_numpy(dtype="datetime64[ns]"))
        assert np.array_equal(design.train_mask, np.isin(design.dates, sessions[::4]))
        kept = pd.Series(design.train_mask).groupby(design.dates).agg(["all", "any"])
        assert (kept["all"] == kept["any"]).all() and kept["all"].any()

    def test_training_filters_never_touch_test_rows_calibration_or_the_base_rate(self, target_evals):
        cs, plain = target_evals["cs"], target_evals["abs"]
        assert cs["status"] == plain["status"] == "evaluated" and len(cs["folds"]) == len(plain["folds"])
        for fold_cs, fold_plain in zip(cs["folds"], plain["folds"]):
            assert np.array_equal(fold_cs.test_idx, fold_plain.test_idx)
            assert np.array_equal(fold_cs.train_idx, fold_plain.train_idx)
        for row_cs, row_plain, fold in zip(cs["result"].folds, plain["result"].folds, cs["folds"]):
            assert row_cs["n_test"] == row_plain["n_test"] == fold.test_idx.size
            assert row_cs["n_calib"] == row_plain["n_calib"]
            assert row_cs["prior_rate"] == row_plain["prior_rate"]
            assert row_cs["n_train"] == int(cs["design"].train_mask[fold.train_idx].sum())
            assert row_cs["n_train"] < 0.3 * row_plain["n_train"]
        assert np.array_equal(cs["result"].oof["idx"], plain["result"].oof["idx"])
        assert cs["result"].confirm["n_eff_total"] == plain["result"].confirm["n_eff_total"]

    def test_cs_median_is_judged_on_absolute_labels(self, target_evals):
        cs, plain = target_evals["cs"], target_evals["abs"]
        design, result = cs["design"], cs["result"]
        assert np.array_equal(design.y, plain["design"].y) and np.array_equal(design.fwd, plain["design"].fwd)
        assert np.array_equal(result.oof["y"], design.y[result.oof["idx"]].astype(int))
        assert cs["baselines"]["prior"]["confirm"] == plain["baselines"]["prior"]["confirm"]
        for pos, fold in enumerate(cs["folds"]):
            in_fold = result.oof["fold"] == pos
            assert np.allclose(result.oof["prior"][in_fold], design.y[fold.train_idx].mean())
            assert np.allclose(result.oof["score_rate"][in_fold], np.nanmean(design.y_train[fold.train_idx]))
        assert (result.oof["prior"] > result.oof["score_rate"] + 0.04).all()
        # calibrated on close-higher outcomes, the forecasts sit at the absolute base rate,
        # not at the 50% the training target goes up
        assert abs(result.oof["p_cal"].mean() - result.oof["prior"].mean()) < 0.02

    def test_excess_spy_trains_on_the_excess_and_is_judged_on_the_raw_sign(self, cross_section, monkeypatch):
        panel, days, close = cross_section["panel"], cross_section["days"], cross_section["close"]
        gap = days[300].to_datetime64()
        market = pd.DataFrame({"close": close}, index=days).drop(index=days[300])
        inputs = types.SimpleNamespace(market_symbol="SPY", market={"SPY": market}, coverage_start=None)
        cfg = _target_config("excess_every_session", label="excess_spy", train_stride=1)
        design = model_training.build_design(panel, cfg, CS_HORIZON, inputs=inputs)
        plain = model_training.build_design(panel, _target_config("plain", label="abs", train_stride=1),
                                            CS_HORIZON)
        assert design.n_rows == plain.n_rows
        assert np.array_equal(design.y, plain.y) and np.array_equal(design.fwd, plain.fwd)
        start = pd.Series(close, index=days).reindex(design.dates).to_numpy()
        end = pd.Series(close, index=days).reindex(design.label_dates).to_numpy()
        no_market = (design.dates == gap) | (design.label_dates == gap)
        assert no_market.any()
        assert np.isnan(design.y_train[no_market]).all() and not design.train_mask[no_market].any()
        assert design.train_mask[~no_market].all()
        assert np.allclose(design.fwd_train[~no_market], (design.fwd - np.log(end / start))[~no_market])
        assert np.array_equal(design.y_train[~no_market], (design.fwd_train[~no_market] > 0).astype(float))

        fitted = []
        monkeypatch.setattr(model_training, "factory_for", _recording_factory(fitted))
        folds = model_training.make_folds(design, cfg)
        result = model_training.evaluate_design(cfg, design, folds, n_threads=1, importance=False, n_boot=20)
        assert len(fitted) == len(folds)
        for labels, fold in zip(fitted, folds):
            rows = fold.train_idx[design.train_mask[fold.train_idx]]
            assert np.array_equal(labels, design.y_train[rows].astype(int))
        assert np.array_equal(result.oof["y"], design.y[result.oof["idx"]].astype(int))

    def test_fit_final_fits_the_target_and_calibrates_on_absolute_labels(self, target_evals, monkeypatch):
        ev = target_evals["cs"]
        design, cfg, result = ev["design"], ev["config"], ev["result"]
        monkeypatch.setattr(model_training, "ship_decision", lambda confirm, **kwargs: "model")
        artifact = model_training.fit_final(design, cfg, result, n_threads=1)
        base_rate = float(design.y.mean())
        assert artifact.status == "model" and artifact.n_rows == design.n_rows
        assert artifact.prior_up_rate == pytest.approx(base_rate)
        assert base_rate > np.nanmean(design.y_train) + 0.04
        calibrator = artifact.calibrator
        assert calibrator.base_rate == pytest.approx(base_rate)
        assert calibrator.score_base_rate == pytest.approx(np.nanmean(design.y_train))
        assert calibrator.slope > 0
        probabilities = np.array([artifact.predict_proba_up(row) for row in design.X[-300:]])
        assert abs(probabilities.mean() - base_rate) < 0.03
        assert probabilities.std() > 0.01

        fitted = []
        monkeypatch.setattr(model_training, "factory_for", _recording_factory(fitted))
        model_training.fit_final(design, cfg, result, n_threads=1)
        assert np.array_equal(fitted[-1], design.y_train[np.flatnonzero(design.train_mask)].astype(int))


# ---------------------------------------------------------------------------
# leaderboard pick rule (scripts/manual/train_experiments.py)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def experiments():
    path = Path(__file__).resolve().parent.parent / "scripts" / "manual" / "train_experiments.py"
    spec = importlib.util.spec_from_file_location("train_experiments_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _entry(config: str = "run6b", *, horizon: int = 21, selection_auc: float = 0.56,
           selection_n_eff=30.0, model: str = "hgb", **extra) -> dict:
    summary = {"selection_auc": selection_auc, "confirm_auc": 0.51, "confirm_n_eff": 16.0}
    if selection_n_eff is not None:
        summary["selection_n_eff"] = selection_n_eff
    return {"config": config, "model": model, "horizon": horizon, "universe": "core",
            "status": "evaluated", "decision": "prior", "reasons": [], "auc_statistic": "same_date",
            "test_window": "all", "summary": summary, **extra}


class TestLeaderboardPick:

    def test_thin_selection_folds_are_never_pick_candidates(self, experiments):
        assert experiments.is_pick_candidate(_entry())
        assert experiments.is_pick_candidate(_entry(selection_n_eff=MIN_CONFIRM_N_EFF))
        thin = _entry("run3", horizon=5, selection_auc=0.557, selection_n_eff=2.9, test_window="coverage_era")
        assert not experiments.is_pick_candidate(thin)
        assert experiments.pick_exclusion(thin).startswith(f"selection n_eff 2.9 < {MIN_CONFIRM_N_EFF:g}")

    def test_older_entries_are_recomputed_from_their_folds_or_excluded(self, experiments):
        folds = [{"n_dates": n} for n in (173, 170, 175, 174, 172, 171)]
        old = _entry(selection_n_eff=None, result={"confirm_folds": 2, "folds": folds})
        assert experiments.entry_selection_n_eff(old) == pytest.approx((173 + 170 + 175 + 174) / 21)
        assert experiments.is_pick_candidate(old)
        # the same folds read at one year hold under three windows
        assert not experiments.is_pick_candidate({**old, "horizon": 252})
        stored = _entry(selection_n_eff=None, result={"selection": {"n_eff_total": 12.5}})
        assert experiments.entry_selection_n_eff(stored) == 12.5
        unknown = _entry(selection_n_eff=None)
        assert experiments.entry_selection_n_eff(unknown) is None
        assert not experiments.is_pick_candidate(unknown) and "re-run" in experiments.pick_exclusion(unknown)

    def test_older_coverage_era_entries_are_excluded_outright(self, experiments):
        folds = [{"n_dates": 400}, {"n_dates": 15}, {"n_dates": 15}]
        old = _entry("run3", horizon=5, selection_n_eff=None, test_window="coverage_era",
                     result={"confirm_folds": 2, "folds": folds})
        assert experiments.entry_selection_n_eff(old) == pytest.approx(80.0)
        assert not experiments.is_pick_candidate(old) and "coverage-era" in experiments.pick_exclusion(old)
        legacy = {k: v for k, v in old.items() if k != "test_window"}
        assert experiments.legacy_test_window(legacy, []) == "coverage_era"
        sweep_point = {"config": "run5_lr0.03_leaves7_msl100_it100"}
        assert experiments.legacy_test_window(sweep_point, [CONFIGS["run3"].to_dict()]) == "coverage_era"
        assert experiments.legacy_test_window(sweep_point, []) is None

    def test_baselines_are_never_picked(self, experiments):
        for config, model in (("logreg_cs", "logreg"), ("logreg", "logreg"), ("momentum", "momentum"),
                              ("prior", "prior")):
            assert experiments.pick_exclusion(_entry(config, model=model)) == "baseline"

    def test_leaderboard_picks_on_evidence_and_lists_what_it_left_out(self, experiments, tmp_path, monkeypatch):
        monkeypatch.setattr(experiments, "EXPERIMENTS_DIR", tmp_path)
        monkeypatch.setattr(experiments, "LEADERBOARD_PATH", tmp_path / "LEADERBOARD.md")
        runs = {
            "20260914-000000-run3": [_entry("run3", horizon=5, selection_auc=0.557, selection_n_eff=6.0,
                                            test_window="coverage_era")],
            "20260914-000100-run1": [_entry("run1", horizon=5, selection_auc=0.525, selection_n_eff=69.0)],
        }
        for name, entries in runs.items():
            (tmp_path / name).mkdir()
            (tmp_path / name / "metrics.json").write_text(json.dumps({"kind": "run", "entries": entries}),
                                                          encoding="utf-8")
        text = experiments.write_leaderboard()
        assert "**Pick: `run1`**" in text
        assert f"- `run3` (20260914-000000-run3): selection n_eff 6.0 < {MIN_CONFIRM_N_EFF:g}" in text
        assert (tmp_path / "LEADERBOARD.md").read_text(encoding="utf-8") == text
