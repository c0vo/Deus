"""
Deus — training the pooled direction model, one horizon at a time.

The one code path from stored data to a saved artifact. The weekly retrain
(StockPredictor.train_pooled, run by the scheduler), the one-shot
scripts/manual/train_pooled_models.py and the desktop experiment runner
(scripts/manual/train_experiments.py) all go through it, so a number in
LEADERBOARD.md and the same number in the Telegram table cannot have been
produced two different ways.

Stages, each timed:

  load       features.load_panel_inputs: one query per table per ticker
  panel      features.build_panel: every ticker's feature frame plus labels
  evaluate   build_design -> make_folds -> model_eval.evaluate (purged
             walk-forward folds), with the prior and momentum baselines
             scored on the same folds
  final_fit  fit_final: ship_decision on the confirm folds; a "model"
             horizon refits on every training row and calibrates on the
             out-of-fold predictions, a "prior" horizon saves the base rate

A config's label chooses only what the estimator is fitted on. Metrics,
calibration, the ship rule and the artifact always read the absolute label
(y_h, fwd_ret_h), so a probability means P(close higher) on every surface.

Everything here is synchronous and CPU-bound. Async callers run it in a worker
thread (asyncio.to_thread) and every fit runs under a thread cap, because on
the phone the SoC is shared with the API process.
"""

from __future__ import annotations

import functools
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from config.logging_config import get_logger
from config.settings import settings
from data.watchlist import DEFAULT_WATCHLIST, MARKET_INPUT_TICKERS, SECTOR_ETFS, TRAINING_BACKBONE
from pipeline import features
from pipeline.features import CATEGORICAL_FEATURES, FEATURE_GROUPS, FEATURE_NAMES, HORIZONS, PanelInputs
from pipeline.model_artifact import STATUS_MODEL, STATUS_PRIOR, PooledArtifact
from pipeline.model_configs import (
    UNIVERSES,
    ModelConfig,
    config_for_horizon,
    fold_plan,
    get_config,
    resolve_columns,
    train_stride_for,
)
from pipeline.model_eval import (
    EvalResult,
    Fold,
    baseline_logreg_factory,
    baseline_momentum,
    baseline_prior,
    coverage_folds,
    deadband_mask,
    evaluate,
    fit_calibrator,
    gbm_factory,
    hgb_factory,
    is_measurable,
    per_ticker_factory,
    ship_decision,
    ship_reasons,
    walkforward_folds,
    weights_for,
)

log = get_logger(__name__)

# Model kinds that produce an estimator the live predictor can score a row
# with. gbm_per_ticker needs the ticker's training code at inference and exists
# only to evaluate the old approach; momentum is a baseline rule.
SERVABLE_MODELS = ("hgb", "logreg")
# A config whose model is "prior" may be shipped deliberately: it saves the
# base rate with its baseline metrics.
PRODUCTION_MODELS = SERVABLE_MODELS + ("prior",)

MOMENTUM_FEATURE = "mom_12_1"
MOMENTUM_FALLBACK_FEATURE = "ret_63d"

TOP_FEATURES = 8
IMPORTANCE_TOP_N = 25
PER_TICKER_COLUMN = "ticker"

# Label "cs_median": a date needs this many labeled rows before "above that
# date's median" is a training label. Thinner dates (early history, before most
# of the universe listed) stay in evaluation and are left out of training.
MIN_CS_TICKERS = 8

# The confirm-fold figures an artifact carries (PooledArtifact.metrics).
ARTIFACT_METRIC_KEYS = (
    "auc_mean", "auc_ci_low", "auc_ci_high", "brier_skill_mean",
    "hi_conf_10_acc_pooled", "hi_conf_10_n_total", "decile_spread_mean",
)


# ── Universe and panel ───────────────────────────────────────────────────────

def training_tickers(db: Any, universe: str) -> list[str]:
    """The symbols whose rows the pooled model trains on, sorted.

    tracked   the user's watchlist
    core      tracked + DEFAULT_WATCHLIST + the eleven sector ETFs
    backbone  core + TRAINING_BACKBONE

    Crypto pairs (-USD), index symbols (^...) and the market inputs (SPY,
    ^GSPC, ^VIX, ^TNX) are never training rows: the market series are what
    every row is read against, and a crypto pair trades a different calendar.
    """
    if universe not in UNIVERSES:
        raise ValueError(f"unknown universe {universe!r}; known: {UNIVERSES}")
    symbols = set(db.get_tracked_tickers() or [])
    if universe in ("core", "backbone"):
        symbols |= set(DEFAULT_WATCHLIST) | set(SECTOR_ETFS)
    if universe == "backbone":
        symbols |= set(TRAINING_BACKBONE)

    excluded = {s.upper() for s in MARKET_INPUT_TICKERS}
    out = set()
    for raw in symbols:
        symbol = str(raw or "").strip().upper()
        if not symbol or symbol.endswith("-USD") or symbol.startswith("^") or symbol in excluded:
            continue
        out.add(symbol)
    return sorted(out)


def _load_panel_timed(db: Any, tickers: list[str],
                      horizons: tuple[int, ...] = HORIZONS) -> tuple[PanelInputs, pd.DataFrame, dict]:
    started = time.perf_counter()
    # load_panel_inputs brings the market inputs (SPY, ^GSPC, ^VIX, ^TNX and
    # each ticker's sector ETF) along with the tickers themselves.
    inputs = features.load_panel_inputs(db, tickers)
    loaded = time.perf_counter()
    panel = features.build_panel(inputs, tickers, horizons)
    built = time.perf_counter()
    timing = {"load": loaded - started, "panel": built - loaded}
    log.info("predictor.panel_loaded", tickers=len(tickers),
             tickers_with_rows=int(panel["ticker"].nunique()) if len(panel) else 0,
             rows=int(len(panel)), columns=int(panel.shape[1]),
             market_symbol=inputs.market_symbol,
             load_seconds=round(timing["load"], 2), panel_seconds=round(timing["panel"], 2))
    return inputs, panel, timing


def load_panel(db: Any, tickers: list[str],
               horizons: tuple[int, ...] = HORIZONS) -> tuple[PanelInputs, pd.DataFrame]:
    """Load every input for `tickers` and build the training panel (logs timing and shape)."""
    inputs, panel, _timing = _load_panel_timed(db, list(tickers), tuple(horizons))
    return inputs, panel


# ── Design matrix ────────────────────────────────────────────────────────────

@dataclass
class Design:
    """One config's rows for one horizon, as the evaluation consumes them.

    Rows are the panel rows with a label for the horizon (`rows` holds their
    panel index labels). `y` and `fwd` are the absolute label and forward log
    return (y_h, fwd_ret_h), whatever the config trains on: every metric, the
    calibration, the ship rule and the artifact read them, because every
    surface presents the probability as P(close higher). `vol_scaled` is `fwd`
    divided by vol_21 * sqrt(h).

    `y_train` is the target the estimator is fitted on, chosen by the config's
    label (`label`): `y` itself for "abs", the sign of the excess over the
    market index for "excess_spy", above the date's median for "cs_median".
    `fwd_train` is the return it is the sign of, and training weights and the
    dead band are computed from it. Both are NaN on rows without a training
    label (no market return, a date too thin for a median).

    `train_mask` (True = train on the row; None = every row) combines the dead
    band, the rows without a training label and the training-date stride
    (`train_stride` sessions). It never touches evaluation: test rows,
    calibration evidence and base rates keep every row.
    """

    X: np.ndarray
    y: np.ndarray
    fwd: np.ndarray
    vol_scaled: np.ndarray
    dates: np.ndarray
    label_dates: np.ndarray
    tickers: np.ndarray
    columns: list[str]
    feature_index: list[int]
    cat_idx: list[int]
    weights: Optional[np.ndarray]
    train_mask: Optional[np.ndarray]
    horizon: int
    rows: Optional[np.ndarray] = None
    y_train: Optional[np.ndarray] = None
    fwd_train: Optional[np.ndarray] = None
    label: str = "abs"
    train_stride: int = 1

    @property
    def n_rows(self) -> int:
        return int(self.y.size)

    @property
    def fit_labels(self) -> np.ndarray:
        """The target the estimator is fitted on: `y_train`, or `y` when none was set."""
        return self.y if self.y_train is None else self.y_train

    @property
    def trains_on_y(self) -> bool:
        """Whether the fit target is the evaluation label itself (label "abs")."""
        return self.y_train is None or self.y_train is self.y or bool(
            np.array_equal(self.y_train, self.y, equal_nan=True))


def _market_forward_return(inputs: Optional[PanelInputs], dates: np.ndarray,
                           label_dates: np.ndarray) -> np.ndarray:
    """log(C[label_date] / C[date]) on the market index, NaN where either close is missing."""
    symbol = inputs.market_symbol if inputs is not None else None
    frame = inputs.market.get(symbol) if symbol else None
    if frame is None or len(frame) == 0:
        raise ValueError("label 'excess_spy' needs PanelInputs with a market index series "
                         f"(market_symbol={symbol!r})")
    close = frame["close"].astype(float)
    close.index = pd.DatetimeIndex(close.index).astype("datetime64[ns]")
    close = close[~close.index.duplicated(keep="last")]
    start = close.reindex(pd.DatetimeIndex(dates).astype("datetime64[ns]")).to_numpy(dtype=float)
    end = close.reindex(pd.DatetimeIndex(label_dates).astype("datetime64[ns]")).to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(end / start)
    return np.where(np.isfinite(out), out, np.nan)


def cross_sectional_return(fwd: np.ndarray, dates: np.ndarray, *,
                           min_rows: int = MIN_CS_TICKERS) -> np.ndarray:
    """`fwd` minus the median of the finite `fwd` values sharing its date.

    NaN where `fwd` is NaN or the date holds fewer than `min_rows` finite
    values. Its sign is the "cs_median" label: above the date's median is up, a
    row exactly on it is not.
    """
    fwd = np.asarray(fwd, dtype=float)
    if fwd.size == 0:
        return fwd.copy()
    codes = pd.factorize(np.asarray(dates))[0]
    if codes.min() < 0:                      # rows without a date share one group
        codes = np.where(codes < 0, codes.max() + 1, codes)
    finite = np.isfinite(fwd)
    median = (pd.Series(np.where(finite, fwd, np.nan)).groupby(codes).transform("median")
              .to_numpy(dtype=float))
    count = np.bincount(codes, weights=finite.astype(float))[codes]
    return np.where(finite & (count >= int(min_rows)), fwd - median, np.nan)


def session_stride_mask(dates: np.ndarray, calendar: np.ndarray, stride: int) -> np.ndarray:
    """True for rows dated on every `stride`-th session of `calendar`, counting from its first.

    `calendar` is every session date the panel holds (duplicates are fine), so
    all rows of a date are kept or dropped together and the kept dates do not
    depend on which rows carry a label.
    """
    stride = int(stride)
    dates = np.asarray(dates, dtype="datetime64[ns]")
    if stride <= 1:
        return np.ones(dates.shape, dtype=bool)
    sessions = np.unique(np.asarray(calendar, dtype="datetime64[ns]"))
    return np.searchsorted(sessions, dates) % stride == 0


def training_return(label: str, fwd: np.ndarray, dates: np.ndarray, label_dates: np.ndarray,
                    inputs: Optional[PanelInputs] = None) -> np.ndarray:
    """The return a config's training label is the sign of (NaN where it has none)."""
    if label == "abs":
        return fwd
    if label == "excess_spy":
        return fwd - _market_forward_return(inputs, dates, label_dates)
    if label == "cs_median":
        return cross_sectional_return(fwd, dates)
    raise ValueError(f"unknown label {label!r}")


def _vol_scaled(ret: np.ndarray, vol_21: np.ndarray, horizon: int) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = ret / (vol_21 * math.sqrt(horizon))
    return np.where(np.isfinite(out), out, np.nan)


def build_design(panel: pd.DataFrame, cfg: ModelConfig, horizon: int, *,
                 inputs: Optional[PanelInputs] = None) -> Design:
    """The config's columns, evaluation and training labels, weights and training mask for one horizon."""
    h = int(horizon)
    y_col, fwd_col, label_col = f"y_{h}", f"fwd_ret_{h}", f"label_date_{h}"
    missing = [c for c in (y_col, fwd_col, label_col) if c not in panel.columns]
    if missing:
        raise KeyError(f"panel has no label columns for {h}d: {missing}")

    columns = resolve_columns(cfg, FEATURE_GROUPS)
    frame = panel.loc[panel[y_col].notna() & panel[label_col].notna()]

    dates = frame["date"].to_numpy(dtype="datetime64[ns]")
    label_dates = frame[label_col].to_numpy(dtype="datetime64[ns]")
    fwd = frame[fwd_col].to_numpy(dtype=float)
    y = frame[y_col].to_numpy(dtype=float)
    keep = np.isfinite(fwd) & np.isfinite(y)
    if not keep.all():
        frame = frame.loc[keep]
        dates, label_dates, fwd, y = dates[keep], label_dates[keep], fwd[keep], y[keep]

    vol_21 = frame["vol_21"].to_numpy(dtype=float)
    vol_scaled = _vol_scaled(fwd, vol_21, h)

    # The training target. Evaluation keeps y and fwd; a row without a training
    # label stays in evaluation and is masked out of training.
    if cfg.label == "abs":
        y_train, fwd_train, vol_scaled_train = y, fwd, vol_scaled
    else:
        fwd_train = training_return(cfg.label, fwd, dates, label_dates, inputs)
        y_train = np.where(np.isfinite(fwd_train), (fwd_train > 0).astype(float), np.nan)
        vol_scaled_train = _vol_scaled(fwd_train, vol_21, h)

    weights = weights_for(cfg.weight_scheme, fwd_train, vol_scaled_train, dates,
                          half_life_years=cfg.decay_half_life_years) if len(y) else None
    stride = train_stride_for(cfg, h)
    trainable = np.isfinite(y_train)
    if cfg.deadband > 0:
        trainable &= deadband_mask(vol_scaled_train, cfg.deadband)
    if stride > 1:
        trainable &= session_stride_mask(dates, panel["date"].to_numpy(dtype="datetime64[ns]"), stride)
    train_mask = None if trainable.all() else trainable

    return Design(
        X=frame[columns].to_numpy(dtype=float) if columns else np.empty((len(frame), 0)),
        y=y,
        fwd=fwd,
        vol_scaled=vol_scaled,
        dates=dates,
        label_dates=label_dates,
        tickers=frame["ticker"].astype(str).to_numpy(dtype=object),
        columns=list(columns),
        feature_index=[FEATURE_NAMES.index(c) for c in columns],
        cat_idx=[columns.index(c) for c in CATEGORICAL_FEATURES if c in columns],
        weights=weights,
        train_mask=train_mask,
        horizon=h,
        rows=frame.index.to_numpy(),
        y_train=y_train,
        fwd_train=fwd_train,
        label=cfg.label,
        train_stride=stride,
    )


def make_folds(design: Design, cfg: ModelConfig, *, coverage_start: Any = None) -> list[Fold]:
    """Walk-forward or coverage-era folds per `cfg.test_window`, on `fold_plan(cfg, h)`'s geometry.

    Lets model_eval's ValueError (fewer than two walk-forward folds) or empty
    list (coverage era not measurable yet) reach the caller.
    """
    plan = fold_plan(cfg, design.horizon)
    if cfg.test_window == "coverage_era":
        return coverage_folds(design.dates, design.label_dates, design.horizon, coverage_start,
                              n_folds=plan.n_folds, test_days=plan.test_days)
    return walkforward_folds(design.dates, design.label_dates, design.horizon,
                             n_folds=plan.n_folds, test_days=plan.test_days,
                             min_train_days=cfg.min_train_days)


def design_matrix(cfg: ModelConfig, design: Design) -> np.ndarray:
    """X as the config's estimator reads it: per-ticker models get a ticker-code column."""
    if cfg.model == "gbm_per_ticker":
        codes = pd.factorize(pd.Series(design.tickers))[0].astype(float)
        return np.column_stack([design.X, codes])
    return design.X


def design_column_names(cfg: ModelConfig, design: Design) -> list[str]:
    names = list(design.columns)
    if cfg.model == "gbm_per_ticker":
        names.append(PER_TICKER_COLUMN)
    return names


class ObservedColumnsHGB(ClassifierMixin, BaseEstimator):
    """HistGradientBoostingClassifier fitted on the columns that have an observed value.

    scikit-learn 1.9 cannot bin a numeric column that is NaN on every training
    row (its binning takes a two-wide sliding window over zero distinct values
    and raises), and a walk-forward fold routinely holds one: any source whose
    history starts after the fold's training window ends — the dark-pool and
    regime series, and every sentiment column before news coverage began. A
    column with no observed value has nothing to teach, so it is left out of
    the fit and ignored at predict time, with the categorical indices remapped
    onto the columns that remain. Everything else is model_eval.hgb_factory.
    """

    def __init__(self, params: Optional[dict] = None, categorical_idx: tuple = ()):
        self.params = params
        self.categorical_idx = categorical_idx

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        self.n_features_in_ = int(X.shape[1])
        observed = ~np.isnan(X).all(axis=0) if X.shape[0] else np.zeros(X.shape[1], dtype=bool)
        self.columns_ = np.flatnonzero(observed)
        if self.columns_.size == 0 or np.unique(y).size < 2:
            weights = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
            self.model_ = None
            self.prior_ = float(np.average(y, weights=weights)) if y.size else 0.5
            self.classes_ = np.array([0, 1])
            return self
        position = {int(c): i for i, c in enumerate(self.columns_)}
        categorical = [position[int(c)] for c in (self.categorical_idx or ()) if int(c) in position]
        self.model_ = hgb_factory(self.params or {}, categorical)()
        _fit_with_weights(self.model_, X[:, self.columns_], y, sample_weight)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        if self.model_ is None:
            p = np.full(X.shape[0], self.prior_, dtype=float)
            return np.column_stack([1.0 - p, p])
        return self.model_.predict_proba(X[:, self.columns_])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def factory_for(cfg: ModelConfig, design: Design) -> Callable[[], Any]:
    """The estimator factory a config names."""
    if cfg.model == "hgb":
        return functools.partial(ObservedColumnsHGB, params=dict(cfg.params),
                                 categorical_idx=tuple(int(i) for i in design.cat_idx))
    if cfg.model == "logreg":
        return baseline_logreg_factory(**cfg.params)
    if cfg.model == "gbm_per_ticker":
        return per_ticker_factory(gbm_factory(cfg.params), ticker_col=design.X.shape[1])
    raise ValueError(f"config {cfg.name!r}: model {cfg.model!r} has no estimator factory")


# ── Evaluation ───────────────────────────────────────────────────────────────

def _momentum_column(panel: pd.DataFrame, design: Design,
                     preferred: str = MOMENTUM_FEATURE) -> tuple[np.ndarray, str]:
    """The design rows' momentum reading, falling back to ret_63d when the preferred column is empty."""
    for name in dict.fromkeys((preferred, MOMENTUM_FALLBACK_FEATURE)):
        if name not in panel.columns:
            continue
        values = panel.loc[design.rows, name].to_numpy(dtype=float)
        if np.isfinite(values).any():
            return values, name
    return np.full(design.n_rows, np.nan), preferred


def _confirm_block(result: Any) -> dict:
    if isinstance(result, EvalResult):
        return result.confirm or {}
    if isinstance(result, dict):
        return result.get("confirm") or {}
    return {}


def evaluate_config(panel: pd.DataFrame, inputs: Optional[PanelInputs], cfg: ModelConfig,
                    horizon: int, *, n_threads: int = 2, importance: bool = True,
                    n_boot: int = 200) -> dict:
    """Score one config at one horizon on purged walk-forward folds.

    Returns a dict:

      config, config_name, horizon
      plan       fold_plan(cfg, horizon): the geometry every fold below uses
      status     "evaluated" or "not_measurable" (too little history for two
                 folds, or no coverage-era blocks yet; `error` says which)
      decision   ship_decision on the confirm folds ("model"/"prior"), None
                 when not measurable
      reasons    ship_reasons on the confirm folds: why the decision is
                 "prior" in plain words (empty for "model", None when not
                 measurable)
      result     model_eval.EvalResult for hgb/logreg/gbm_per_ticker; the
                 baseline dict itself for the prior and momentum configs
      baselines  {"prior": ..., "momentum": ...} on the same folds
      error      the not-measurable reason, else None
      timing     seconds per stage
      design, folds  the in-memory inputs, for fit_final and reporting
    """
    h = int(horizon)
    started = time.perf_counter()
    timing: dict[str, float] = {}
    plan = fold_plan(cfg, h)
    out: dict = {"config": cfg, "config_name": cfg.name, "horizon": h, "plan": plan,
                 "status": "evaluated", "decision": None, "reasons": None, "result": None,
                 "baselines": {}, "error": None, "timing": timing, "design": None, "folds": None}

    def finish() -> dict:
        timing["total"] = time.perf_counter() - started
        return out

    t0 = time.perf_counter()
    design = build_design(panel, cfg, h, inputs=inputs)
    timing["design"] = time.perf_counter() - t0
    out["design"] = design
    if design.n_rows == 0:
        out.update(status="not_measurable", error=f"no labeled rows for the {h}d horizon")
        return finish()

    t0 = time.perf_counter()
    coverage_start = inputs.coverage_start if inputs is not None else None
    try:
        folds = make_folds(design, cfg, coverage_start=coverage_start)
    except ValueError as exc:
        out.update(status="not_measurable", error=str(exc))
        timing["folds"] = time.perf_counter() - t0
        return finish()
    timing["folds"] = time.perf_counter() - t0
    if not folds:
        out.update(status="not_measurable",
                   error=(f"fewer than two {plan.test_days}-day test blocks fit after the news "
                          f"coverage start ({coverage_start}) for the {h}d horizon"))
        return finish()
    out["folds"] = folds

    t0 = time.perf_counter()
    momentum, momentum_feature = _momentum_column(panel, design)
    confirm = plan.confirm_folds
    out["baselines"] = {
        "prior": baseline_prior(design.y, design.fwd, design.dates, folds, h,
                                confirm_folds=confirm, n_boot=n_boot),
        "momentum": {**baseline_momentum(momentum.reshape(-1, 1), design.y, design.fwd,
                                         design.dates, folds, h, 0, confirm_folds=confirm,
                                         n_boot=n_boot),
                     "feature": momentum_feature},
    }
    timing["baselines"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    if cfg.model == "prior":
        result: Any = out["baselines"]["prior"]
    elif cfg.model == "momentum":
        preferred = str(cfg.params.get("feature") or MOMENTUM_FEATURE)
        if preferred == MOMENTUM_FEATURE:
            result = out["baselines"]["momentum"]
        else:
            values, used = _momentum_column(panel, design, preferred)
            result = {**baseline_momentum(values.reshape(-1, 1), design.y, design.fwd,
                                          design.dates, folds, h, 0, confirm_folds=confirm,
                                          n_boot=n_boot),
                      "feature": used}
    else:
        result = evaluate_design(cfg, design, folds, n_threads=n_threads,
                                 importance=importance, n_boot=n_boot)
    timing["evaluate"] = time.perf_counter() - t0
    out["result"] = result
    out["reasons"] = ship_reasons(_confirm_block(result))
    out["decision"] = ship_decision(_confirm_block(result))
    return finish()


def evaluate_design(cfg: ModelConfig, design: Design, folds: list[Fold], *,
                    n_threads: int = 2, importance: bool = True, n_boot: int = 200) -> EvalResult:
    """model_eval.evaluate with everything the design and config specify.

    Fold models are fitted on the design's training target over its training
    mask; everything scored reads the absolute label and return.
    """
    return evaluate(
        factory_for(cfg, design), design_matrix(cfg, design), design.y, design.fwd,
        design.dates, design.tickers, folds,
        sample_weight=design.weights, train_mask=design.train_mask,
        label_dates=design.label_dates, horizon=design.horizon,
        confirm_folds=fold_plan(cfg, design.horizon).confirm_folds,
        importance=importance, n_threads=n_threads, n_boot=n_boot,
        y_fit=None if design.trains_on_y else design.y_train,
    )


# ── Final fit ────────────────────────────────────────────────────────────────

def _fit_with_weights(estimator: Any, X: np.ndarray, y: np.ndarray,
                      sample_weight: Optional[np.ndarray]) -> Any:
    if sample_weight is None:
        return estimator.fit(X, y)
    if isinstance(estimator, Pipeline):
        return estimator.fit(X, y, **{f"{estimator.steps[-1][0]}__sample_weight": sample_weight})
    return estimator.fit(X, y, sample_weight=sample_weight)


def _confirm_summary(confirm: dict) -> dict:
    out = {key: confirm.get(key) for key in ARTIFACT_METRIC_KEYS}
    out["prior_rate"] = confirm.get("prior_rate_mean")
    return out


def ranked_importance(importance: list[dict], names: list[str]) -> list[dict]:
    """Permutation importances with column names, largest first, NaN last."""
    ranked = []
    for entry in importance or []:
        index = int(entry.get("feature", -1))
        value = float(entry.get("importance", math.nan))
        ranked.append({
            "feature": names[index] if 0 <= index < len(names) else f"col_{index}",
            "importance": value,
            "std": float(entry.get("std", math.nan)),
        })
    ranked.sort(key=lambda r: r["importance"] if math.isfinite(r["importance"]) else -math.inf,
                reverse=True)
    return ranked


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _last_date(dates: np.ndarray) -> str:
    if dates is None or len(dates) == 0:
        return ""
    return str(pd.Timestamp(np.max(dates)).date())


def prior_artifact(design: Optional[Design], cfg: ModelConfig, horizon: int, *,
                   universe: Optional[str] = None, metrics: Optional[dict] = None) -> PooledArtifact:
    """The no-edge artifact: no estimator, the labeled rows' up-rate as the answer."""
    columns = list(design.columns) if design is not None else []
    has_rows = design is not None and design.n_rows > 0
    return PooledArtifact(
        model=None,
        calibrator=None,
        feature_names=columns,
        feature_index=[FEATURE_NAMES.index(c) for c in columns],
        categorical_idx=list(design.cat_idx) if design is not None else [],
        horizon=int(horizon),
        schema_version=features.FEATURE_SCHEMA_VERSION,
        status=STATUS_PRIOR,
        prior_up_rate=float(np.mean(design.y)) if has_rows else 0.5,
        trained_at=_utc_now_iso(),
        train_end=_last_date(design.dates) if has_rows else "",
        config_name=cfg.name,
        universe=universe or cfg.universe,
        n_tickers=int(pd.Series(design.tickers).nunique()) if has_rows else 0,
        n_rows=design.n_rows if design is not None else 0,
        metrics=dict(metrics or {}),
        top_features=[],
    )


def fit_final(design: Design, cfg: ModelConfig, result: EvalResult, *, n_threads: int = 2,
              universe: Optional[str] = None) -> PooledArtifact:
    """Apply the ship rule and, for a "model" horizon, fit the artifact on every labeled row.

    The final estimator sees every row the design's training mask keeps (the
    dead band, rows without a training label and the training stride are left
    out, and weights rescaled to mean 1, as each fold did), fitted on the
    design's training target. Its calibrator is fitted on the evaluation's
    out-of-fold predictions against the absolute label — the only honest sample
    of how this model's raw scores map to P(close higher) — by the same
    fit_calibrator the folds used: anchored on the absolute up-rate of every
    labeled row, each OOF score read against its own fold's up-rate, and the
    evidence counted as distinct OOF dates / horizon. When the training target
    is not the absolute label, raw scores are centred on that target's rate
    instead (about 0.5 for "cs_median"), so the final model's scores are read
    against the target's up-rate and each OOF score against its fold's. The
    artifact's prior_up_rate is the absolute up-rate either way. A "prior"
    horizon keeps the evaluation metrics that explain the decision.
    """
    status = ship_decision(result.confirm)
    metrics = _confirm_summary(result.confirm or {})
    names = design_column_names(cfg, design)
    top = [r["feature"] for r in ranked_importance(result.importance, names)
           if math.isfinite(r["importance"]) and r["importance"] > 0 and r["feature"] in design.columns]
    top = top[:TOP_FEATURES]

    if status != STATUS_MODEL:
        artifact = prior_artifact(design, cfg, design.horizon, universe=universe, metrics=metrics)
        artifact.top_features = top
        return artifact
    if cfg.model not in SERVABLE_MODELS:
        raise ValueError(f"config {cfg.name!r}: model {cfg.model!r} passed the ship rule but "
                         f"cannot be served; production configs use {SERVABLE_MODELS}")

    rows = (np.arange(design.n_rows) if design.train_mask is None
            else np.flatnonzero(design.train_mask))
    target = design.fit_labels
    if not np.isfinite(target[rows]).all():
        raise ValueError(f"config {cfg.name!r}: the training mask keeps rows without a training "
                         f"label at {design.horizon}d")
    y_fit = target[rows].astype(int)
    if rows.size == 0 or np.unique(y_fit).size < 2:
        log.warning("predictor.final_fit_single_class", horizon_days=design.horizon,
                    config=cfg.name, rows=int(rows.size))
        artifact = prior_artifact(design, cfg, design.horizon, universe=universe, metrics=metrics)
        artifact.top_features = top
        return artifact

    weights = None
    if design.weights is not None:
        weights = design.weights[rows]
        weights = weights / weights.mean()

    model = factory_for(cfg, design)()
    with threadpool_limits(limits=int(n_threads)):
        _fit_with_weights(model, design.X[rows], y_fit, weights)
    oof_dates = np.unique(np.asarray(design.dates)[result.oof["idx"]]).size
    scores = {}
    if not design.trains_on_y:
        labeled = target[np.isfinite(target)]
        scores = {"score_base_rate": float(labeled.mean()) if labeled.size else float(np.mean(design.y)),
                  "oof_score_rate": result.oof.get("score_rate", result.oof["prior"])}
    calibrator = fit_calibrator(result.oof["p_raw"], result.oof["y"], result.calibrator_method,
                                base_rate=float(np.mean(design.y)),
                                oof_base_rate=result.oof["prior"],
                                n_eff=oof_dates / design.horizon, **scores)

    return PooledArtifact(
        model=model,
        calibrator=calibrator,
        feature_names=list(design.columns),
        feature_index=list(design.feature_index),
        categorical_idx=list(design.cat_idx),
        horizon=design.horizon,
        schema_version=features.FEATURE_SCHEMA_VERSION,
        status=STATUS_MODEL,
        prior_up_rate=float(np.mean(design.y)),
        trained_at=_utc_now_iso(),
        train_end=_last_date(design.dates[rows]),
        config_name=cfg.name,
        universe=universe or cfg.universe,
        n_tickers=int(pd.Series(design.tickers).nunique()),
        n_rows=design.n_rows,
        metrics=metrics,
        top_features=top,
    )


# ── Metrics row ──────────────────────────────────────────────────────────────

def _importance_json(result: Any, cfg: ModelConfig, design: Optional[Design]) -> dict:
    if not isinstance(result, EvalResult) or design is None or not result.importance:
        return {"top": [], "groups": {}}
    ranked = ranked_importance(result.importance, design_column_names(cfg, design))
    groups: dict[str, float] = {}
    by_name = {r["feature"]: r["importance"] for r in ranked}
    for group, columns in FEATURE_GROUPS.items():
        values = [by_name[c] for c in columns if c in by_name and math.isfinite(by_name[c])]
        if values:
            groups[group] = float(sum(values))
    return {"top": ranked[:IMPORTANCE_TOP_N], "groups": groups}


def _folds_json(result: Any) -> list:
    if isinstance(result, EvalResult):
        return result.to_dict()["folds"]
    if isinstance(result, dict):
        return list(result.get("folds") or [])
    return []


def metrics_row(eval_dict: dict, artifact: PooledArtifact, *, run_id: Optional[str],
                scope: str = "universal") -> dict:
    """One model_metrics row for a trained horizon.

    Every metric column comes from the confirm folds — the numbers the ship
    decision was taken on — so a row's status can always be re-derived from
    the row. A prior horizon stores them too: they are why it is prior. The
    why in words goes into `note`, kept inside config_json since the table has
    no column for it: the not-measurable reason when no folds could be built,
    else ship_reasons joined with "; " for a prior horizon. config_json also
    records the fold plan, the confirm folds' n_eff, the reasons as a list and
    `measurable` — False when the confirm folds hold fewer than
    MIN_CONFIRM_N_EFF independent label windows or no interval could be drawn,
    so the table can say "not measurable" rather than print an AUC.
    """
    cfg: ModelConfig = eval_dict["config"]
    result = eval_dict.get("result")
    confirm = _confirm_block(result)
    design: Optional[Design] = eval_dict.get("design")
    evaluated = eval_dict.get("status") == "evaluated"
    reasons = list(eval_dict.get("reasons") or ship_reasons(confirm)) if evaluated else []
    if not evaluated:
        note = eval_dict.get("error")
    else:
        note = "; ".join(reasons) if artifact.status != STATUS_MODEL and reasons else None

    config_json = cfg.to_dict()
    config_json["universe"] = artifact.universe
    plan = eval_dict.get("plan") or fold_plan(cfg, artifact.horizon)
    config_json["fold_plan"] = dict(plan._asdict())
    config_json["train_stride_sessions"] = (design.train_stride if design is not None
                                            else train_stride_for(cfg, artifact.horizon))
    config_json["confirm_n_eff"] = confirm.get("n_eff_total")
    config_json["ship_reasons"] = reasons
    config_json["measurable"] = bool(evaluated and is_measurable(confirm))
    if note:
        config_json["note"] = note

    n_dates = int(len(np.unique(design.dates))) if design is not None and design.n_rows else 0
    row = {
        "run_id": run_id,
        "scope": scope,
        "horizon_days": int(artifact.horizon),
        "schema_version": int(artifact.schema_version),
        "config_name": artifact.config_name,
        "status": artifact.status,
        "n_rows": int(artifact.n_rows),
        "n_dates": n_dates,
        "n_tickers": int(artifact.n_tickers),
        "train_end": artifact.train_end or None,
        "auc_mean": confirm.get("auc_mean"),
        "auc_std": confirm.get("auc_std"),
        "auc_ci_low": confirm.get("auc_ci_low"),
        "auc_ci_high": confirm.get("auc_ci_high"),
        "logloss_mean": confirm.get("log_loss_mean"),
        "brier_mean": confirm.get("brier_mean"),
        "brier_skill_mean": confirm.get("brier_skill_mean"),
        "acc_mean": confirm.get("acc_mean"),
        "acc_majority_mean": confirm.get("acc_majority_mean"),
        "hi_conf_acc": confirm.get("hi_conf_10_acc_pooled"),
        "hi_conf_n": confirm.get("hi_conf_10_n_total"),
        "decile_spread_mean": confirm.get("decile_spread_mean"),
        "prior_up_rate": float(artifact.prior_up_rate),
        "config_json": config_json,
        "folds_json": _folds_json(result),
        "importance_json": _importance_json(result, cfg, design),
    }
    if note:
        row["note"] = note
    return row


# ── One horizon, end to end ──────────────────────────────────────────────────

def train_horizon(db: Any, horizon: int, *, config_name: Optional[str] = None,
                  universe: Optional[str] = None,
                  panel_bundle: Optional[tuple[PanelInputs, pd.DataFrame]] = None,
                  n_threads: int = 2, run_id: Optional[str] = None) -> tuple[PooledArtifact, dict, dict]:
    """Evaluate, decide and fit one horizon. Returns (artifact, metrics_row, eval_dict).

    `config_name` defaults to model_configs.PRODUCTION[horizon]; `universe` to
    settings.predictor_universe, then the config's own. `panel_bundle` is an
    (inputs, panel) pair from load_panel, shared across horizons by the weekly
    retrain so the panel is built once; it must have been loaded for the same
    universe. A horizon whose folds cannot be built is saved as the prior with
    the reason in the metrics row's note. Nothing is written here — the caller
    saves the artifact and inserts the row.
    """
    h = int(horizon)
    started = time.perf_counter()
    cfg = get_config(config_name) if config_name else config_for_horizon(h)
    if cfg.model not in PRODUCTION_MODELS:
        raise ValueError(f"config {cfg.name!r}: model {cfg.model!r} cannot be served in "
                         f"production; use one of {PRODUCTION_MODELS}")
    universe = universe or settings.predictor_universe or cfg.universe
    if universe not in UNIVERSES:
        raise ValueError(f"unknown universe {universe!r}; known: {UNIVERSES}")

    timing = {"load": 0.0, "panel": 0.0}
    if panel_bundle is None:
        inputs, panel, load_timing = _load_panel_timed(db, training_tickers(db, universe))
        timing.update(load_timing)
    else:
        inputs, panel = panel_bundle

    eval_dict = evaluate_config(panel, inputs, cfg, h, n_threads=n_threads, importance=True)
    timing["evaluate"] = eval_dict["timing"]["total"]

    t0 = time.perf_counter()
    design = eval_dict["design"]
    if eval_dict["status"] != "evaluated":
        artifact = prior_artifact(design, cfg, h, universe=universe)
        log.info("predictor.horizon_not_measurable", horizon_days=h, config=cfg.name,
                 reason=eval_dict["error"])
    elif cfg.model == "prior":
        artifact = prior_artifact(design, cfg, h, universe=universe,
                                  metrics=_confirm_summary(_confirm_block(eval_dict["result"])))
    else:
        artifact = fit_final(design, cfg, eval_dict["result"], n_threads=n_threads,
                             universe=universe)
    timing["final_fit"] = time.perf_counter() - t0

    row = metrics_row(eval_dict, artifact, run_id=run_id)
    timing["total"] = time.perf_counter() - started
    eval_dict["stage_timing"] = timing
    auc = row.get("auc_mean")
    log.info("predictor.train_timing", horizon_days=h, config=cfg.name, universe=universe,
             status=artifact.status, rows=artifact.n_rows, tickers=artifact.n_tickers,
             auc=round(float(auc), 4) if auc is not None and math.isfinite(float(auc)) else None,
             **{f"{k}_seconds": round(v, 2) for k, v in timing.items()})
    return artifact, row, eval_dict
