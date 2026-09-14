"""
Deus - named configurations for the direction-model experiments.

A config names everything that differs between two experiment runs: the
feature columns, the estimator and its parameters, how training rows are
weighted or filtered, the label, the ticker universe and the folds. The
experiment runner evaluates configs by name; the weekly retrain asks
``config_for_horizon``.

Feature-group names are a contract with ``pipeline/features.py``
(``FEATURE_GROUPS``). This module does not import it: ``resolve_columns`` takes
that dict as an argument, so configs stay importable (and testable) on their own.

Field meanings the runner has to honour:

* ``model``: "hgb" -> ``model_eval.hgb_factory(params, categorical_idx)``;
  "gbm_per_ticker" -> ``model_eval.per_ticker_factory(gbm_factory(params), ticker_col)``;
  "logreg" -> ``model_eval.baseline_logreg_factory(**params)``;
  "prior" -> ``model_eval.baseline_prior``;
  "momentum" -> ``model_eval.baseline_momentum`` on the column ``params["feature"]``.
* ``label`` chooses the TRAINING target only: "abs" is fwd_ret > 0; "excess_spy" is
  fwd_ret - SPY fwd_ret > 0; "cs_median" is fwd_ret above the median of every row on the
  same date (ties are 0), with dates holding fewer than
  ``model_training.MIN_CS_TICKERS`` labeled rows left out of training. Metrics,
  calibration, the ship rule and the artifact always read the absolute label and
  return (y_h, fwd_ret_h): every surface presents the probability as P(close higher).
  Pass the training target as ``evaluate(y_fit=...)``.
* ``weight_scheme`` / ``decay_half_life_years`` -> ``model_eval.weights_for``, on the return
  the training target is the sign of.
* ``deadband`` > 0 -> ``evaluate(train_mask=model_eval.deadband_mask(vol_scaled_ret, deadband))``,
  on that same return.
* ``train_stride``: train only on every k-th session of the panel's session calendar,
  whole cross-sections at a time (``train_stride_for``; "auto" is max(1, h // 5)). A
  training-row filter like the dead band: test rows and calibration evidence keep
  every session.
* ``test_window``: "all" -> ``walkforward_folds(n_folds, test_days, min_train_days)``;
  "coverage_era" -> ``coverage_folds(..., coverage_start, n_folds, test_days)``, where an
  empty list means "not measurable yet".
* Fold geometry comes from ``fold_plan(config, horizon)``, never from the fields
  directly: ``test_days=None`` (the default) sizes the test blocks per horizon
  so the confirm folds can hold MIN_CONFIRM_N_EFF independent label windows.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass, field, replace
from typing import NamedTuple

from pipeline.model_eval import MIN_CONFIRM_N_EFF

FEATURE_GROUP_NAMES = ("price", "market", "regime", "smart_money", "sentiment", "calendar", "context")
MODEL_KINDS = ("hgb", "gbm_per_ticker", "logreg", "prior", "momentum")
WEIGHT_SCHEMES = ("none", "abs_ret_scaled")
LABELS = ("abs", "excess_spy", "cs_median")
UNIVERSES = ("tracked", "core", "backbone")
TEST_WINDOWS = ("all", "coverage_era")
# train_stride "auto": one kept training date per this many sessions of label
# horizon, so each h-session label window still holds about five kept dates.
TRAIN_STRIDE_AUTO = "auto"
TRAIN_STRIDE_AUTO_SESSIONS = 5

# Calendar days per trading session: fold boundaries are calendar days, labels
# are sessions (252 a year).
CALENDAR_DAYS_PER_SESSION = 365.25 / 252
# Horizon-planned test blocks come in whole half-years of calendar days, from
# one half-year up to three years. Six three-year blocks plus the training
# minimum already need ~20 years of rows; a longer block would leave no folds.
PLAN_TEST_DAYS_STEP = 126
PLAN_MAX_TEST_DAYS = 756


@dataclass(frozen=True)
class ModelConfig:
    """One experiment configuration. Validated on construction; ``params`` is copied."""

    name: str
    feature_groups: tuple[str, ...]
    model: str = "hgb"
    params: dict = field(default_factory=dict, hash=False)
    weight_scheme: str = "none"
    decay_half_life_years: float | None = None
    deadband: float = 0.0
    label: str = "abs"
    # Sessions between kept training dates: an integer >= 1 or "auto" (see
    # train_stride_for). 1 trains on every session.
    train_stride: int | str = 1
    universe: str = "core"
    test_window: str = "all"
    n_folds: int = 6
    # Calendar days per test block. None: planned per horizon (fold_plan); a
    # number fixes the block length at every horizon.
    test_days: int | None = None
    confirm_folds: int = 2
    min_train_days: int = 504
    # Explicit column names; when non-empty they replace the group selection.
    features: tuple[str, ...] = ()
    description: str = ""

    def __post_init__(self):
        # Own copies, so configs derived with dataclasses.replace never share a
        # params dict, and list arguments still hash and compare as tuples.
        object.__setattr__(self, "params", dict(self.params or {}))
        object.__setattr__(self, "feature_groups", tuple(self.feature_groups))
        object.__setattr__(self, "features", tuple(self.features))
        unknown = [g for g in self.feature_groups if g not in FEATURE_GROUP_NAMES]
        if unknown:
            raise ValueError(f"config {self.name!r}: unknown feature groups {unknown}; "
                             f"known: {FEATURE_GROUP_NAMES}")
        for attr, allowed in (("model", MODEL_KINDS), ("weight_scheme", WEIGHT_SCHEMES),
                              ("label", LABELS), ("universe", UNIVERSES),
                              ("test_window", TEST_WINDOWS)):
            if getattr(self, attr) not in allowed:
                raise ValueError(f"config {self.name!r}: {attr}={getattr(self, attr)!r} "
                                 f"is not one of {allowed}")
        if (self.n_folds < 2 or (self.test_days is not None and self.test_days < 1)
                or self.min_train_days < 0 or not 1 <= self.confirm_folds < self.n_folds):
            raise ValueError(f"config {self.name!r}: needs n_folds >= 2, test_days >= 1 (or None), "
                             f"1 <= confirm_folds < n_folds and min_train_days >= 0")
        if self.deadband < 0:
            raise ValueError(f"config {self.name!r}: deadband must be >= 0")
        stride = self.train_stride
        if stride != TRAIN_STRIDE_AUTO and not (isinstance(stride, int) and not isinstance(stride, bool)
                                                and stride >= 1):
            raise ValueError(f"config {self.name!r}: train_stride must be an integer >= 1 or "
                             f"{TRAIN_STRIDE_AUTO!r}, got {stride!r}")
        if self.decay_half_life_years is not None and self.decay_half_life_years <= 0:
            raise ValueError(f"config {self.name!r}: decay_half_life_years must be > 0")

    def to_dict(self) -> dict:
        """Plain JSON-ready dict (the ``config_json`` stored with metrics)."""
        out = asdict(self)
        out["feature_groups"] = list(self.feature_groups)
        out["features"] = list(self.features)
        return out


# Pooled HistGradientBoosting defaults: shallow trees, large leaves and L2 so
# ~60 noisy features over ~50k correlated rows cannot memorise the panel.
DEFAULT_HGB = dict(learning_rate=0.03, max_iter=200, max_leaf_nodes=15, min_samples_leaf=300,
                   l2_regularization=1.0, early_stopping=False, random_state=42)

# Heavier regularisation for the run6 family: fewer, smaller trees on leaves of a
# thousand rows, stronger L2, and half the columns considered at each split
# (max_features; model_eval drops it on a scikit-learn older than 1.4).
STRONG_HGB = dict(learning_rate=0.03, max_iter=150, max_leaf_nodes=7, min_samples_leaf=1000,
                  l2_regularization=5.0, max_features=0.5, early_stopping=False, random_state=42)

# A compact set of classic cross-sectional factors from the price and market
# groups: momentum and reversal, volatility and tail shape, distance from highs
# and drawdown, market sensitivity, trading activity and size. No context column
# (sector_id, is_etf, hist_len_years), so the model cannot tell tickers apart by
# identity.
FACTOR_FEATURES: tuple[str, ...] = (
    "mom_12_1", "ret_252d", "ret_21d", "ret_5d", "vol_63", "dist_52w_high", "dd_63",
    "beta_63", "rel_spy_63d", "volume_z20", "log_dollar_vol", "skew_63",
)

# The v3 per-ticker model (predictor.py's fold model), for run0.
GBM_V3_PARAMS = dict(n_estimators=100, max_depth=4, learning_rate=0.05, subsample=0.9,
                     random_state=42)

# The v4 columns closest to the old 38-feature v3 vector: its price technicals
# made stationary, 5-day regime changes, the dark-pool, DIX/GEX/put-call and
# insider features, and the news block. 13D/13G stakes, KR flows and
# llm_historical_accuracy have no v4 counterpart.
V3_LIKE_FEATURES: tuple[str, ...] = (
    # price
    "ret_1d", "ret_5d", "dist_sma20", "rsi_14", "vol_21", "volume_z20", "macd_hist_pct",
    "bb_width", "atr_pct",
    # market
    "vix_level", "vix_chg_5d", "spy_ret_5d", "tnx_chg_5d", "rel_sector_21d",
    # regime
    "dix_z60", "gex_z60", "pcr_z60",
    # smart money
    "offexch_short_ratio_z20", "offexch_short_ratio_mom", "offexch_share_z20",
    "insider_buy_ratio_90d", "insider_net_30d", "insider_cluster_30d", "days_since_insider_buy",
    # sentiment
    "sent_1d", "sent_3d", "sent_7d", "sent_mom", "news_n_3d", "news_n_14d", "news_velocity",
    "imp_avg_7d", "imp_max_7d", "bull_ratio_7d", "urg_max_7d",
)

RUN1_GROUPS = ("price", "market", "calendar", "context")
RUN2_GROUPS = RUN1_GROUPS + ("regime", "smart_money")
RUN3_GROUPS = RUN2_GROUPS + ("sentiment",)

_RUN1 = ModelConfig(
    name="run1", feature_groups=RUN1_GROUPS, params=DEFAULT_HGB,
    description="Pooled HGB on price, market, calendar and context features; core universe; "
                "unweighted. Does pooling plus stationary features produce skill?")
_RUN2 = replace(
    _RUN1, name="run2", feature_groups=RUN2_GROUPS,
    description="run1 plus regime (DIX/GEX/put-call) and smart money (off-exchange volume, "
                "insider filings). Do they add skill?")
_RUN6A = replace(
    _RUN1, name="run6a", label="cs_median", train_stride=TRAIN_STRIDE_AUTO,
    description="run1 trained to beat the same date's median return (cs_median) on every "
                "max(1, h // 5)-th session. The ship metric ranks tickers within a date; a "
                "within-date-balanced target drops the market direction the model cannot learn.")
_RUN6B = replace(
    _RUN6A, name="run6b", params=STRONG_HGB,
    description="run6a with STRONG_HGB (7-leaf trees on 1000-row leaves, L2 5, half the "
                "columns per split).")

CONFIGS: dict[str, ModelConfig] = {c.name: c for c in (
    ModelConfig(
        name="run0", feature_groups=("price", "market", "regime", "smart_money", "sentiment"),
        features=V3_LIKE_FEATURES, model="gbm_per_ticker", params=GBM_V3_PARAMS,
        description="The v3 approach under honest folds: one GradientBoosting(100 trees, depth 4) "
                    "per ticker on the v4 columns closest to the old 38 features."),
    _RUN1,
    replace(_RUN1, name="run1b", universe="backbone",
            description="run1 on the backbone universe (core plus liquid sector-diverse large "
                        "caps). Do more tickers help or dilute?"),
    _RUN2,
    replace(_RUN2, name="run3", feature_groups=RUN3_GROUPS, test_window="coverage_era",
            n_folds=3, test_days=21,
            description="run2 plus sentiment, tested only on dates with news coverage "
                        "(3 x 21-day blocks). Not measurable until the confirm blocks hold "
                        "enough independent label windows, and a prior while the CI spans 0.50."),
    replace(_RUN2, name="run4a", weight_scheme="abs_ret_scaled",
            description="run2 weighting rows by clip(|vol-scaled forward return|, 0.1, 3)."),
    replace(_RUN2, name="run4b", deadband=0.1,
            description="run2 dropping training rows whose vol-scaled forward move is under 0.1."),
    replace(_RUN2, name="run4c", decay_half_life_years=3.0,
            description="run2 with a 3-year half-life on training-row weights."),
    replace(_RUN2, name="run4d", label="excess_spy",
            description="run2 trained on the sign of the return in excess of SPY; judged, "
                        "calibrated and served on the raw sign like every config."),
    _RUN6A,
    _RUN6B,
    replace(_RUN6B, name="run6c", feature_groups=("price", "market"), features=FACTOR_FEATURES,
            description="run6b on the twelve FACTOR_FEATURES only: can a compact factor set, "
                        "with nothing that identifies a ticker, rank the cross-section?"),
    replace(_RUN6B, name="run6d", universe="backbone",
            description="run6b on the backbone universe: does a wider cross-section help a "
                        "within-date target?"),
    replace(_RUN6B, name="run6e", feature_groups=RUN2_GROUPS,
            description="run6b plus regime and smart money (run2's groups)."),
    replace(_RUN6B, name="run6f", label="abs",
            description="run6b on the raw sign (abs): STRONG_HGB and the training stride "
                        "without the cs_median target."),
    replace(_RUN6B, name="run6g", feature_groups=("price", "market", "calendar"),
            description="run6b without the context group: sector_id with hist_len_years "
                        "all but names a ticker, so a pooled model can memorise which tickers "
                        "boomed recently."),
    replace(_RUN6B, name="logreg_cs", model="logreg", params={"C": 0.1},
            feature_groups=("price", "market"), features=FACTOR_FEATURES,
            description="Linear baseline for the run6 family: median-impute, standardise, L2 "
                        "logistic regression (C=0.1) on FACTOR_FEATURES, cs_median target, same "
                        "training stride."),
    replace(_RUN2, name="logreg", model="logreg", params={"C": 0.1},
            description="Median-impute, standardise, L2 logistic regression (C=0.1) on run2's "
                        "features. HGB ships only if it beats this on the confirm folds."),
    ModelConfig(
        name="prior", feature_groups=(), model="prior",
        description="Training up-rate for every row: zero skill by construction."),
    ModelConfig(
        name="momentum", feature_groups=("price",), model="momentum",
        params={"feature": "mom_12_1"},
        description="55% up when 12-1 momentum is positive, 45% when not, 50% when unknown."),
)}

# Which config the weekly retrain ships per horizon (days). Picked by selection AUC on the
# 2026-09-14 snapshot protocol (252d has no pick and keeps plain run1); all four served the
# base rate on the confirm folds at the time, and the weekly retrain re-tests them. The run5
# sweep points are registered in CONFIGS after sweep_configs.
PRODUCTION: dict[int, str] = {5: "run5_lr0.06_leaves31_msl100_it400",
                              21: "run5_lr0.03_leaves15_msl100_it200",
                              63: "run5_lr0.06_leaves7_msl300_it200",
                              252: "run1"}

# run5: small chronological grid over the run4 winner's HGB parameters.
SWEEP_GRID_SMALL = dict(learning_rate=[0.03, 0.06], max_leaf_nodes=[7, 15, 31],
                        min_samples_leaf=[100, 300], max_iter=[100, 200, 400])

_PARAM_ABBREVIATIONS = {"learning_rate": "lr", "max_leaf_nodes": "leaves",
                        "min_samples_leaf": "msl", "max_iter": "it"}


def get_config(name: str) -> ModelConfig:
    """The named config; KeyError listing the known names otherwise."""
    try:
        return CONFIGS[name]
    except KeyError:
        raise KeyError(f"unknown model config {name!r}; known: {', '.join(sorted(CONFIGS))}") from None


def config_for_horizon(h: int) -> ModelConfig:
    """The production config for a horizon in days."""
    if h not in PRODUCTION:
        raise KeyError(f"no production config for {h}d; configured horizons: {sorted(PRODUCTION)}")
    return get_config(PRODUCTION[h])


class FoldPlan(NamedTuple):
    """Walk-forward geometry for one config at one horizon (see ``fold_plan``)."""

    n_folds: int
    test_days: int
    confirm_folds: int


def planned_test_days(horizon: int, confirm_folds: int = 2) -> int:
    """
    Calendar days per test block so ``confirm_folds`` blocks hold MIN_CONFIRM_N_EFF windows.

    The confirm folds need MIN_CONFIRM_N_EFF * h sessions between them, i.e.
    MIN_CONFIRM_N_EFF * h / confirm_folds sessions per block, converted at
    CALENDAR_DAYS_PER_SESSION and rounded up to whole half-years. At the
    default minimum of 10 windows that is 126 days at 5d, 252 at 21d and 504 at
    63d. The 252d horizon would need five years per block; it is capped at
    PLAN_MAX_TEST_DAYS instead, so its confirm folds hold about four windows,
    fail the evidence minimum and serve the base rate. That is the honest
    answer: a few years of history cannot say whether a one-year call is right
    more often than the base rate.
    """
    sessions = MIN_CONFIRM_N_EFF * int(horizon) / max(int(confirm_folds), 1)
    days = math.ceil(sessions * CALENDAR_DAYS_PER_SESSION / PLAN_TEST_DAYS_STEP) * PLAN_TEST_DAYS_STEP
    return int(min(max(days, PLAN_TEST_DAYS_STEP), PLAN_MAX_TEST_DAYS))


def fold_plan(config: ModelConfig, horizon: int) -> FoldPlan:
    """
    The fold geometry for ``config`` at ``horizon``: (n_folds, test_days, confirm_folds).

    Everything that builds folds goes through here - the experiment runner,
    ``model_training.make_folds``/``evaluate_config`` and so the weekly retrain
    - so an experiment and production measure a horizon on identical folds.
    A config with ``test_days=None`` gets ``planned_test_days``; an explicit
    ``test_days`` (run3's short coverage-era blocks, test fixtures) is kept at
    every horizon, and where its confirm folds hold too few independent windows
    the ship rule says so and the horizon serves the base rate.
    ``min_train_days`` and the embargo are untouched;
    when history is too short for the planned blocks, ``walkforward_folds``
    drops the oldest folds and raises below two.
    """
    test_days = (planned_test_days(horizon, config.confirm_folds) if config.test_days is None
                 else int(config.test_days))
    return FoldPlan(int(config.n_folds), test_days, int(config.confirm_folds))


def train_stride_for(config: ModelConfig, horizon: int) -> int:
    """
    Sessions between kept training dates for ``config`` at ``horizon``.

    An integer ``train_stride`` is used as is. "auto" is
    max(1, h // TRAIN_STRIDE_AUTO_SESSIONS): 1 at 5d, 4 at 21d, 12 at 63d and 50
    at 252d. An h-session label overlaps the next h - 1 dates' labels, so
    neighbouring training dates are near-duplicates; keeping one date in k (with
    its whole cross-section) cuts the rows a fit sees k-fold while each label
    window still holds about five kept dates.
    """
    if config.train_stride == TRAIN_STRIDE_AUTO:
        return max(1, int(horizon) // TRAIN_STRIDE_AUTO_SESSIONS)
    return int(config.train_stride)


def sweep_configs(base: ModelConfig, grid: dict | None = None, *, prefix: str = "run5") -> list[ModelConfig]:
    """
    One config per point of ``grid`` (cartesian product) on top of ``base.params``.

    Names encode the point, e.g. ``run5_lr0.03_leaves7_msl100_it100``. Choose
    between them on ``EvalResult.selection`` (folds before the confirm folds),
    then apply ``ship_decision`` to the winner's ``confirm`` only.
    """
    grid = SWEEP_GRID_SMALL if grid is None else grid
    keys = list(grid)
    out = []
    for values in itertools.product(*(grid[k] for k in keys)):
        point = dict(zip(keys, values))
        suffix = "_".join(f"{_PARAM_ABBREVIATIONS.get(k, k)}{v}" for k, v in point.items())
        out.append(replace(base, name=f"{prefix}_{suffix}", params={**base.params, **point},
                           description=f"{base.name} with {point}"))
    return out


# The sweep points PRODUCTION names, registered so get_config resolves them. Each is taken
# from sweep_configs(run1, SWEEP_GRID_SMALL) by name rather than retyped, so it stays the
# config the experiments evaluated. A sweep name does not say its base: a sweep of run2
# generates the same names on other features, and these are run1's.
_RUN1_SWEEP = {cfg.name: cfg for cfg in sweep_configs(_RUN1, SWEEP_GRID_SMALL)}
CONFIGS.update({name: _RUN1_SWEEP[name] for name in (
    "run5_lr0.06_leaves31_msl100_it400",    # 5d
    "run5_lr0.03_leaves15_msl100_it200",    # 21d
    "run5_lr0.06_leaves7_msl300_it200",     # 63d
)})


def resolve_columns(config: ModelConfig, feature_groups: dict[str, list[str]], *,
                    strict: bool = True) -> list[str]:
    """
    The config's feature columns, in the order ``feature_groups`` lists them.

    ``feature_groups`` is ``pipeline.features.FEATURE_GROUPS`` (group -> column
    names, in ``FEATURE_NAMES`` order). Explicit ``config.features`` win over
    groups. With ``strict`` an unknown group or column raises ``KeyError``
    (a drifted contract with features.py); otherwise it is skipped.
    """
    ordered = [column for columns in feature_groups.values() for column in columns]
    if config.features:
        known = set(ordered)
        missing = [c for c in config.features if c not in known]
        if missing and strict:
            raise KeyError(f"config {config.name!r}: columns not in the feature set: {missing}")
        wanted = set(config.features)
        return [c for c in ordered if c in wanted]
    missing_groups = [g for g in config.feature_groups if g not in feature_groups]
    if missing_groups and strict:
        raise KeyError(f"config {config.name!r}: feature groups not in the feature set: "
                       f"{missing_groups}")
    selected = set(config.feature_groups)
    return [c for group, columns in feature_groups.items() if group in selected for c in columns]
