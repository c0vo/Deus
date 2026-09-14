"""
Null calibration and power study for the pooled model's ship rule (desktop tool).

The walk-forward ship rule (pipeline.model_eval.ship_decision) has to answer
"prior" when there is nothing to find and "model" when there is. This measures
both on simulated panels whose answer is known, through the production code
path: simulated OHLCV -> features.build_panel -> model_training.evaluate_config
(evaluate_design for the positive control, model_eval.evaluate for the probe)
with each horizon's production config, its fold plan, its HGB parameters through
model_training.factory_for, the calibrator and the ship rule.

Panels. 20 tickers over 4,500 sessions, half listed from the first session and
half at staggered later dates; one market factor (betas 0.5-1.5) and three
sector factors (with a sector ETF each), GARCH(1,1) volatility clustering in
every factor and ticker, Student-t (df 4) shocks, SPY, ^VIX (from the market's
conditional volatility) and ^TNX series, no alternative data. Every return is a
log return with a drift of 0.035 x its conditional sd.

Two nulls, the same panels with different labels:

* exact - features from one path, fwd_ret_h / y_h from an independent path on
  the same structure (dates, listings, betas, sectors, volatility levels). The
  labels overlap across neighbouring dates and move together across tickers as
  real ones do, and nothing a feature row holds can predict them. The
  calibration test of the rule.
* path - features and labels from the same path. Given the shock magnitudes the
  signs are coin flips, so the labels are no more predictable than on the exact
  null (the up-probability moves only with the dispersion of volatility inside
  the window, a second-order effect, and not at all with zero drift). What the
  path null adds is the dependence real data has: a forecast made late in a
  test block holds the returns that decided the block's earlier labels. Ranking
  a fold's rows across dates (``auc_pooled``) is biased under that dependence -
  forecasts leaning against recent moves score above 0.50, trend-followers
  below - which is why the ship rule ranks tickers on the same date only. Both
  statistics are reported side by side.

Probes. (1) No lookahead: every bar after a cut date is replaced and the panel
rebuilt; every feature row before the cut must be bit-identical. (2) Fixed
rules, no fitting: confirm-fold AUC across dates and on the same date, on both
nulls. (3) Permuted training labels: the production HGB fitted on labels whose
dates are permuted in contiguous blocks within its training rows only, scored
on untouched test rows; a harness that leaks nothing through training scores
0.50.

Positive control. Path panels with one extra column, the forward h-session
return divided by vol_21 * sqrt(h) plus Gaussian noise; the noise scale is set
on two pilot panels so that the column's same-date AUC is 0.55 or 0.60.

Reported per horizon: the share of panels whose confirm AUC CI low > 0.50
(target about 5%), the false-ship rate (target 5% or less), the mean CI width
beside 2 x 1.645 x the empirical sd of the confirm AUC across panels (an honest
interval has a ratio near 1), mean n_eff and mean confirm Brier skill; for the
positive control the ship rate (power). Every rate carries its Monte Carlo
standard error sqrt(p(1-p)/n). The variant tables re-score the same
out-of-fold predictions under other bootstrap block lengths and calibration
priors without refitting: that is how the constants in pipeline/model_eval.py
were chosen.

Writes storage/experiments/null_ship_rate.md and null_ship_rate.json (every
panel's numbers). Seeds are fixed: the same command gives the same tables.
Every worker fits with one thread, so --workers is the number of cores used.

Usage:
    python scripts/manual/null_ship_rate.py --smoke           # one panel of each study, about 2 min
    python scripts/manual/null_ship_rate.py --quick           # about 8 min on 2 workers
    python scripts/manual/null_ship_rate.py                   # the acceptance run, about 40 min on 2 workers
    python scripts/manual/null_ship_rate.py --from-json storage/experiments/null_ship_rate.json
    python scripts/manual/null_ship_rate.py --horizons 5,21 --tag short
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import logging
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from pipeline import features, model_configs, model_eval, model_training
from pipeline.features import HORIZONS, PanelInputs
from pipeline.model_eval import (
    BLOCK_LENGTH_MULT,
    CALIBRATION_PRIOR_N_EFF,
    MIN_CONFIRM_N_EFF,
    MIN_SHIP_AUC,
    block_bootstrap_auc,
    block_length_for,
    same_date_auc,
    ship_decision,
    ship_reasons,
)

OUT_DIR = REPO_ROOT / "storage" / "experiments"

# ── Panel shape ──────────────────────────────────────────────────────────────

N_TICKERS = 20
N_SESSIONS = 4500
START = "2008-01-02"
GARCH_BURN = 300
T_DF = 4
# Drift per unit of conditional daily volatility: P(up over 5 sessions) ~ 54%.
SHARPE_PER_SESSION = 0.035
MARKET_GARCH = (0.011, 0.08, 0.90)          # long-run daily vol, alpha, beta
SECTOR_GARCH = (0.006, 0.05, 0.93)
IDIO_GARCH = ((0.008, 0.022), 0.06, 0.92)   # long-run vol drawn per ticker
SECTORS = (("technology", "XLK"), ("energy", "XLE"), ("healthcare", "XLV"))
# Independent random streams per seed: the panel's structure (listings, betas,
# sectors), the path its features are built from, and the path an exact null
# reads its labels from.
STREAM_STRUCTURE, STREAM_PATH, STREAM_LABEL_PATH = 0, 1, 2
LABEL_MODES = ("exact", "path")
# The nulls a task can run on: name -> (labels, drift). The zero-drift path null
# is unpredictable whatever the volatility does; it runs the fixed rules only,
# to show the across-dates bias is not a drift effect.
NULLS = {"exact": ("exact", SHARPE_PER_SESSION), "path": ("path", SHARPE_PER_SESSION),
         "path, zero drift": ("path", 0.0)}

# ── Study size ───────────────────────────────────────────────────────────────

N_BOOT = 200
SIGNAL_AUCS = (0.55, 0.60)
PILOT_SEEDS = (90001, 90002)
LOOKAHEAD_SEEDS = (1, 2)
LOOKAHEAD_CUT_SESSION = 3000
# Training dates are permuted in runs of this many sessions by the probe.
PERMUTED_BLOCK_SESSIONS = 126
# Panels per study; per-horizon dicts give the horizons each seed runs.
FULL = {
    "hgb_exact": {5: 24, 21: 24, 63: 12, 252: 12},
    "hgb_path": {5: 24, 21: 24},
    "momentum": {5: 96, 21: 96, 63: 96, 252: 96},     # per null; about 3 s a panel
    "power": {5: 12, 21: 12},                          # per target AUC
    "probe": {5: 8, 21: 8},
}
QUICK = {
    "hgb_exact": {5: 4, 21: 4, 63: 2, 252: 2},
    "hgb_path": {5: 4, 21: 4},
    "momentum": {5: 8, 21: 8, 63: 8, 252: 8},
    "power": {5: 3, 21: 3},
    "probe": {5: 2, 21: 2},
}
# One panel of everything: proves the pipeline end to end, measures nothing.
SMOKE = {
    "hgb_exact": {5: 1, 21: 1, 63: 1, 252: 1},
    "hgb_path": {5: 1},
    "momentum": {5: 2, 21: 2, 63: 2, 252: 2},
    "power": {5: 1},
    "probe": {5: 1},
}
# Intervals re-scored on the confirm folds: (label, block length in multiples of
# h, widened by model_eval.interval_inflation or the bare percentile interval).
CI_VARIANTS = (("L = 1h, percentile only", 1.0, False), ("L = 1h", 1.0, True), ("L = 2h", 2.0, True),
               ("L = 3h", 3.0, True))
# Calibration variants: (label, prior weight; None = the unregularised Platt used before).
CAL_VARIANTS = (("unregularised Platt (before)", None), ("prior 5", 5.0), ("prior 20", 20.0),
                ("prior 50", 50.0))
# Fixed rules for the diagnosis: (panel column, sign). Market-level columns are
# the same for every ticker on a date, so their same-date AUC is exactly 0.50.
FIXED_RULES = (("spy_dist_sma200", -1), ("dist_sma200", -1), ("ret_21d", -1), ("mom_12_1", 1),
               ("vol_21", 1))


def production_variant_labels() -> tuple[str, str]:
    """The block-length and calibration variant labels that match model_eval's constants."""
    return f"L = {BLOCK_LENGTH_MULT:g}h", f"prior {CALIBRATION_PRIOR_N_EFF:g}"


# ── Simulation ───────────────────────────────────────────────────────────────

def _garch(rng: np.random.Generator, n: int, k: int, sigma_lr, alpha: float, beta: float):
    """(shocks, conditional sd), both (n, k): GARCH(1,1) with unit-variance Student-t innovations."""
    sigma_lr = np.broadcast_to(np.asarray(sigma_lr, dtype=float), (k,))
    z = rng.standard_t(T_DF, size=(n, k)) * math.sqrt((T_DF - 2) / T_DF)
    omega = sigma_lr ** 2 * (1.0 - alpha - beta)
    var = sigma_lr ** 2
    shocks, sd = np.empty((n, k)), np.empty((n, k))
    for t in range(n):
        sd[t] = np.sqrt(var)
        shocks[t] = sd[t] * z[t]
        var = omega + alpha * shocks[t] ** 2 + beta * var
    return shocks, sd


def _bars(rng: np.random.Generator, days: pd.DatetimeIndex, ret: np.ndarray, sd: np.ndarray,
          volume_scale: float, sd_ref: float) -> pd.DataFrame:
    """OHLCV whose close follows the log returns `ret`; nothing in a bar uses a later session."""
    close = 50.0 * np.exp(np.cumsum(ret))
    prev = np.r_[close[0] * np.exp(-ret[0]), close[:-1]]
    open_ = prev * np.exp(0.3 * ret + 0.15 * sd * rng.standard_normal(ret.size))
    wick = np.abs(rng.standard_normal((ret.size, 2))) * 0.5 * sd[:, None]
    return pd.DataFrame({
        "open": open_,
        "high": np.maximum(open_, close) * np.exp(wick[:, 0]),
        "low": np.minimum(open_, close) * np.exp(-wick[:, 1]),
        "close": close,
        "volume": volume_scale * np.exp(0.5 * np.log(sd / sd_ref) + 0.25 * rng.standard_normal(ret.size)),
    }, index=days)


def _level_frame(days: pd.DatetimeIndex, level: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"open": level, "high": level, "low": level, "close": level,
                         "volume": np.zeros(level.size)}, index=days)


@dataclasses.dataclass(frozen=True)
class PanelStructure:
    """What every path of one panel shares: session dates, listings, loadings, sectors, scales."""

    days: pd.DatetimeIndex
    listing: np.ndarray        # first session index per ticker
    beta: np.ndarray           # market loading
    loading: np.ndarray        # sector loading
    sector: np.ndarray         # index into SECTORS
    idio_vol: np.ndarray       # long-run idiosyncratic daily vol
    volume_scale: np.ndarray

    @property
    def tickers(self) -> list[str]:
        return [f"SIM{j:02d}" for j in range(self.beta.size)]


@dataclasses.dataclass
class SimPath:
    """One draw of every return series on a PanelStructure: log returns and conditional sds."""

    ret: np.ndarray            # (sessions, tickers)
    sd: np.ndarray
    market_ret: np.ndarray     # (sessions,)
    market_sd: np.ndarray
    market_next_sd: np.ndarray
    sector_ret: np.ndarray     # (sessions, sectors)
    sector_sd: np.ndarray


def panel_structure(seed: int, n_tickers: int = N_TICKERS, n_sessions: int = N_SESSIONS) -> PanelStructure:
    """Half the tickers listed from the first session, half at staggered later dates."""
    rng = np.random.default_rng([int(seed), STREAM_STRUCTURE])
    lo, hi = IDIO_GARCH[0]
    return PanelStructure(
        days=pd.DatetimeIndex(pd.bdate_range(START, periods=n_sessions)).astype("datetime64[ns]"),
        listing=np.where(np.arange(n_tickers) < n_tickers // 2, 0,
                         rng.integers(0, int(0.6 * n_sessions), n_tickers)),
        beta=rng.uniform(0.5, 1.5, n_tickers),
        loading=rng.uniform(0.3, 0.8, n_tickers),
        sector=rng.integers(0, len(SECTORS), n_tickers),
        idio_vol=rng.uniform(lo, hi, n_tickers),
        volume_scale=rng.uniform(5e5, 5e6, n_tickers),
    )


def simulate_path(structure: PanelStructure, rng: np.random.Generator, drift: float) -> SimPath:
    """
    Market, sector and idiosyncratic GARCH(1,1) shocks with Student-t innovations.

    Every return is a log return whose drift is ``drift`` x its conditional sd.
    Given every shock's magnitude, the signs are independent coin flips, so
    with drift 0 the sign of any forward return is a fair coin whatever came
    before: a strict null even with volatility clustering. With drift > 0 the
    up-probability over h sessions is Phi(drift x sum(sd) / sqrt(sum(sd^2))),
    which moves only with the dispersion of volatility inside the window.
    """
    n_sessions, k = len(structure.days), structure.beta.size
    n = n_sessions + GARCH_BURN
    m_shock, m_sd = _garch(rng, n, 1, *MARKET_GARCH)
    s_shock, s_sd = _garch(rng, n, len(SECTORS), *SECTOR_GARCH)
    i_shock, i_sd = _garch(rng, n, k, structure.idio_vol, *IDIO_GARCH[1:])
    beta, loading, sector = structure.beta, structure.loading, structure.sector
    sd = np.sqrt((beta * m_sd) ** 2 + (loading * s_sd[:, sector]) ** 2 + i_sd ** 2)
    ret = drift * sd + beta * m_shock + loading * s_shock[:, sector] + i_shock
    sector_sd = np.sqrt(m_sd ** 2 + s_sd ** 2)
    burn = slice(GARCH_BURN, None)
    return SimPath(
        ret=ret[burn], sd=sd[burn],
        market_ret=(drift * m_sd + m_shock)[burn, 0], market_sd=m_sd[burn, 0],
        # the next session's conditional sd is set by today's shock: known at the close
        market_next_sd=np.r_[m_sd[GARCH_BURN + 1:, 0], m_sd[-1, 0]],
        sector_ret=(drift * sector_sd + m_shock + s_shock)[burn], sector_sd=sector_sd[burn],
    )


def inputs_from_path(structure: PanelStructure, path: SimPath, rng: np.random.Generator) -> PanelInputs:
    """PanelInputs (bars, SPY, sector ETFs, ^VIX, ^TNX; no alternative data) for one path."""
    days, n_sessions = structure.days, len(structure.days)
    inputs = PanelInputs(market_symbol="SPY")
    for j, ticker in enumerate(structure.tickers):
        first = int(structure.listing[j])
        sd_ref = math.sqrt((structure.beta[j] * MARKET_GARCH[0]) ** 2
                           + (structure.loading[j] * SECTOR_GARCH[0]) ** 2 + structure.idio_vol[j] ** 2)
        inputs.bars[ticker] = _bars(rng, days[first:], path.ret[first:, j], path.sd[first:, j],
                                    float(structure.volume_scale[j]), sd_ref)
        inputs.sector_of[ticker] = SECTORS[int(structure.sector[j])][0]
    inputs.market["SPY"] = _bars(rng, days, path.market_ret, path.market_sd, 5e7, MARKET_GARCH[0])
    for k, (_, etf) in enumerate(SECTORS):
        inputs.market[etf] = _bars(rng, days, path.sector_ret[:, k], path.sector_sd[:, k], 1e7,
                                   math.hypot(MARKET_GARCH[0], SECTOR_GARCH[0]))
    inputs.market["^VIX"] = _level_frame(
        days, 100.0 * math.sqrt(252) * path.market_next_sd * np.exp(0.08 * rng.standard_normal(n_sessions)))
    tnx = np.empty(n_sessions)
    level = 3.0
    for t, shock in enumerate(0.04 * rng.standard_normal(n_sessions)):
        level += 0.002 * (3.0 - level) + shock
        tnx[t] = level
    inputs.market["^TNX"] = _level_frame(days, tnx)
    return inputs


def simulate_inputs(seed: int, *, drift: float = SHARPE_PER_SESSION, n_tickers: int = N_TICKERS,
                    n_sessions: int = N_SESSIONS) -> tuple[PanelInputs, list[str]]:
    """PanelInputs for one simulated panel: the structure of ``seed`` and its feature path."""
    structure = panel_structure(seed, n_tickers, n_sessions)
    rng = np.random.default_rng([int(seed), STREAM_PATH])
    return inputs_from_path(structure, simulate_path(structure, rng, drift), rng), structure.tickers


def independent_labels(seed: int, panel: pd.DataFrame, *, drift: float = SHARPE_PER_SESSION,
                       horizons=HORIZONS, n_tickers: int = N_TICKERS,
                       n_sessions: int = N_SESSIONS) -> pd.DataFrame:
    """
    ``panel`` with every label column recomputed from an independent path.

    The label path shares the feature path's structure - session dates,
    listings, market betas, sector memberships and loadings, volatility levels
    - but draws its own market, sector and idiosyncratic shocks. Its labels
    therefore overlap across neighbouring dates and move together across
    tickers exactly as real labels do, while nothing a feature row holds can
    predict them: the exact null. Labels use features._labels, the production
    definition.
    """
    structure = panel_structure(seed, n_tickers, n_sessions)
    path = simulate_path(structure, np.random.default_rng([int(seed), STREAM_LABEL_PATH]), drift)
    horizons = tuple(int(h) for h in horizons)
    frames = []
    for j, ticker in enumerate(structure.tickers):
        first = int(structure.listing[j])
        close = pd.Series(50.0 * np.exp(np.cumsum(path.ret[first:, j])), index=structure.days[first:])
        labels = features._labels(close, horizons).rename_axis("date").reset_index()
        labels.insert(0, "ticker", ticker)
        frames.append(labels)
    label_cols = features.label_columns(horizons)
    merged = panel.drop(columns=label_cols).merge(pd.concat(frames, ignore_index=True),
                                                  on=["ticker", "date"], how="left",
                                                  validate="one_to_one")
    return merged[list(panel.columns)]


def build_null_panel(seed: int, *, labels: str = "exact",
                     drift: float = SHARPE_PER_SESSION) -> tuple[PanelInputs, pd.DataFrame]:
    """
    One simulated panel. ``labels`` "exact": labels from an independent path
    (the calibration test of the ship rule); "path": labels from the path the
    features are built from (the realistic panel the power study uses).
    """
    if labels not in LABEL_MODES:
        raise ValueError(f"labels must be one of {LABEL_MODES}, got {labels!r}")
    inputs, tickers = simulate_inputs(seed, drift=drift)
    panel = features.build_panel(inputs, tickers, HORIZONS)
    if labels == "exact":
        panel = independent_labels(seed, panel, drift=drift)
    return inputs, panel


# ── Probes ───────────────────────────────────────────────────────────────────

def check_no_lookahead(seed: int, cut_session: int = LOOKAHEAD_CUT_SESSION) -> dict:
    """
    Replace every bar after a cut date - tickers, SPY, sector ETFs, ^VIX, ^TNX -
    with another path's (rescaled, so the seam itself jumps) and rebuild the
    panel: every row dated on or before the cut must keep bit-identical
    features, and every label resolved by the cut its value.
    """
    structure = panel_structure(seed)
    rng = np.random.default_rng([int(seed), STREAM_PATH])
    base = inputs_from_path(structure, simulate_path(structure, rng, SHARPE_PER_SESSION), rng)
    other_rng = np.random.default_rng([int(seed), 77])
    other = inputs_from_path(structure, simulate_path(structure, other_rng, 0.2), other_rng)
    cut = structure.days[cut_session]
    spliced = PanelInputs(market_symbol=base.market_symbol, sector_of=dict(base.sector_of))
    for kind in ("bars", "market"):
        for key, frame in getattr(base, kind).items():
            later = getattr(other, kind)[key]
            getattr(spliced, kind)[key] = pd.concat([frame.loc[:cut], later.loc[later.index > cut] * 1.37])
    before = features.build_panel(base, structure.tickers, HORIZONS)
    after = features.build_panel(spliced, structure.tickers, HORIZONS)
    a = before[before["date"] <= cut].reset_index(drop=True)
    b = after[after["date"] <= cut].reset_index(drop=True)
    same_rows = a[["ticker", "date"]].equals(b[["ticker", "date"]])
    identical = same_rows and a[features.FEATURE_NAMES].equals(b[features.FEATURE_NAMES])
    resolved = same_rows and all(
        a.loc[a[f"label_date_{h}"] <= cut, f"y_{h}"].equals(b.loc[b[f"label_date_{h}"] <= cut, f"y_{h}"])
        for h in HORIZONS)
    altered = not before.loc[before["date"] > cut, "ret_1d"].reset_index(drop=True).equals(
        after.loc[after["date"] > cut, "ret_1d"].reset_index(drop=True))
    return {"seed": int(seed), "cut": str(cut.date()), "rows_before_cut": int(len(a)),
            "same_rows": bool(same_rows), "features_identical": bool(identical),
            "resolved_labels_identical": bool(resolved), "future_altered": bool(altered)}


class BlockPermutedLabels(ClassifierMixin, BaseEstimator):
    """
    The leakage probe's estimator: ``inner()`` fitted on labels whose training
    dates are permuted in contiguous runs of ``block`` sessions.

    X carries two trailing columns, the row's day number and ticker code, which
    are stripped before the inner model sees X. Only the rows handed to fit
    (a fold's training rows) are permuted; predict_proba only strips the
    columns, so the test rows and their labels are untouched.
    """

    def __init__(self, inner=None, block: int = PERMUTED_BLOCK_SESSIONS, seed: int = 0):
        self.inner = inner
        self.block = block
        self.seed = seed

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=float)
        day, ticker = X[:, -2].astype(np.int64), X[:, -1].astype(np.int64)
        dates, date_pos = np.unique(day, return_inverse=True)
        table = np.full((dates.size, int(ticker.max()) + 1), np.nan)
        table[date_pos, ticker] = np.asarray(y, dtype=float)
        n_blocks = -(-dates.size // self.block)
        order = np.random.default_rng([int(self.seed), dates.size]).permutation(n_blocks)
        shuffled = np.concatenate([np.arange(k * self.block, min((k + 1) * self.block, dates.size))
                                   for k in order])
        permuted = table[shuffled][date_pos, ticker]
        keep = np.isfinite(permuted)
        self.model_ = self.inner()
        self.model_.fit(X[keep, :-2], permuted[keep].astype(int))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(np.asarray(X, dtype=float)[:, :-2])


# ── Positive control ─────────────────────────────────────────────────────────

def _signal(design: model_training.Design, scale: float, seed: int) -> np.ndarray:
    noise = np.random.default_rng([int(seed), int(design.horizon), 17]).standard_normal(design.n_rows)
    return design.vol_scaled + scale * noise


def with_signal(design: model_training.Design, scale: float, seed: int) -> model_training.Design:
    """The design plus a last column carrying a noisy copy of the vol-scaled forward return."""
    return dataclasses.replace(
        design, X=np.column_stack([design.X, _signal(design, scale, seed)]),
        columns=list(design.columns) + ["signal"],
        feature_index=list(design.feature_index) + [len(features.FEATURE_NAMES)])


def calibrate_signal_scales(horizons: list[int]) -> dict[str, float]:
    """Noise scale per (horizon, target AUC) so the signal column's same-date AUC is the target."""
    pilots = [build_null_panel(seed, labels="path") + (seed,) for seed in PILOT_SEEDS]
    scales = {}
    for h in horizons:
        cfg = model_configs.config_for_horizon(h)
        designs = [(model_training.build_design(panel, cfg, h, inputs=inputs), seed)
                   for inputs, panel, seed in pilots]

        def auc(scale: float) -> float:
            values = []
            for design, seed in designs:
                score = _signal(design, scale, seed)
                ok = np.isfinite(score)
                values.append(same_date_auc(design.y[ok], score[ok], design.dates[ok]))
            return float(np.mean(values))

        for target in SIGNAL_AUCS:
            lo, hi = math.log(0.1), math.log(200.0)
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if auc(math.exp(mid)) > target else (lo, mid)
            scales[f"{h}:{target:.2f}"] = math.exp(0.5 * (lo + hi))
    return scales


# ── Scoring one panel ────────────────────────────────────────────────────────

def _pooled_spread(p: np.ndarray, fwd: np.ndarray) -> float:
    """The decile spread before same-date ranking: one fold's rows ranked across dates."""
    if p.size < model_eval.MIN_DECILE_ROWS:
        return math.nan
    top, bottom = p >= np.quantile(p, 0.9), p <= np.quantile(p, 0.1)
    if not (np.isfinite(fwd[top]).any() and np.isfinite(fwd[bottom]).any()):
        return math.nan
    return float(np.nanmean(fwd[top]) - np.nanmean(fwd[bottom]))


def _ranking_record(confirm: dict, y: np.ndarray, p: np.ndarray, fwd: np.ndarray, day: np.ndarray,
                    pos: np.ndarray, h: int) -> dict:
    """The confirm folds under the ship rule, under the across-dates rule it replaced, and per block length."""
    record = {
        "auc": confirm.get("auc_mean"), "ci_low": confirm.get("auc_ci_low"),
        "ci_high": confirm.get("auc_ci_high"), "brier_skill": confirm.get("brier_skill_mean"),
        "spread": confirm.get("decile_spread_mean"), "n_eff": confirm.get("n_eff_total"),
        "prior_rate": confirm.get("prior_rate_mean"), "decision": ship_decision(confirm),
        "reasons": ship_reasons(confirm),
    }
    groups = np.unique(pos)
    pooled_auc = float(np.mean([model_eval._auc(y[pos == g], p[pos == g]) for g in groups]))
    pooled_spread = float(np.nanmean([_pooled_spread(p[pos == g], fwd[pos == g]) for g in groups]))
    lo, hi = block_bootstrap_auc(y, p, day, n_boot=N_BOOT, seed=0, groups=pos,
                                 block_length=block_length_for(h), joint=True, same_date=False)
    record["pooled"] = {
        "auc": pooled_auc, "ci_low": lo, "ci_high": hi, "spread": pooled_spread,
        "decision": ship_decision({**confirm, "auc_mean": pooled_auc, "auc_ci_low": lo,
                                   "auc_ci_high": hi, "decile_spread_mean": pooled_spread}),
    }
    record["ci_variants"] = {}
    for label, mult, widened in CI_VARIANTS:
        lo, hi = block_bootstrap_auc(y, p, day, n_boot=N_BOOT, seed=0, groups=pos,
                                     block_length=max(1, math.ceil(mult * h)), joint=True,
                                     horizon=h if widened else None)
        record["ci_variants"][label] = {
            "ci_low": lo, "ci_high": hi,
            "decision": ship_decision({**confirm, "auc_ci_low": lo, "auc_ci_high": hi})}
    return record


def _old_platt(p_prev: np.ndarray, y_prev: np.ndarray):
    """The calibrator before the shrinkage fix: unregularised Platt, identity under 50 rows or one class."""
    if p_prev.size < model_eval.MIN_CALIBRATION_ROWS or np.unique(y_prev).size < 2:
        return lambda p: np.clip(p, 0.0, 1.0)
    model = LogisticRegression(C=1e4, max_iter=1000)
    model.fit(model_eval._logit(p_prev).reshape(-1, 1), y_prev.astype(int))
    return lambda p: model.predict_proba(model_eval._logit(p).reshape(-1, 1))[:, 1]


def _calibration_variants(result: model_eval.EvalResult, folds: list, design, h: int,
                          confirm: dict) -> dict:
    """Confirm-fold Brier skill under each calibration variant, replaying evaluate's purge."""
    oof = result.oof
    day = model_eval._day_numbers(design.dates)
    label_day = model_eval._day_numbers(oof["label_day"])
    y = oof["y"].astype(float)
    out = {}
    for label, weight in CAL_VARIANTS:
        skills = []
        for pos in range(len(folds) - result.confirm_folds, len(folds)):
            fold = folds[pos]
            here = oof["fold"] == pos
            base = float(oof["prior"][here][0])
            cutoff = int(model_eval._day_numbers([fold.test_start])[0]) - int(fold.embargo_days)
            prev = (oof["fold"] < pos) & (label_day != model_eval._NAT_DAY) & (label_day < cutoff)
            p_prev, y_prev, p_here = oof["p_raw"][prev], y[prev], oof["p_raw"][here]
            if weight is None:
                p_cal = _old_platt(p_prev, y_prev)(p_here)
            else:
                saved = model_eval.CALIBRATION_PRIOR_N_EFF
                model_eval.CALIBRATION_PRIOR_N_EFF = weight
                try:
                    calibrator = model_eval.fit_calibrator(
                        p_prev, y_prev, "platt", base_rate=base, oof_base_rate=oof["prior"][prev],
                        n_eff=np.unique(day[oof["idx"][prev]]).size / h)
                finally:
                    model_eval.CALIBRATION_PRIOR_N_EFF = saved
                p_cal = calibrator.transform(p_here)
            y_here = y[here]
            skills.append(1.0 - np.mean((p_cal - y_here) ** 2) / np.mean((base - y_here) ** 2))
        skill = float(np.mean(skills))
        out[label] = {"brier_skill": skill,
                      "decision": ship_decision({**confirm, "brier_skill_mean": skill})}
    return out


def _confirm_rows(result: model_eval.EvalResult, n_folds: int):
    m = result.oof["fold"] >= n_folds - result.confirm_folds
    return result.oof["idx"][m], result.oof["fold"][m], m


def _fitted_record(result: model_eval.EvalResult, folds: list, design, h: int, *,
                   calibration: bool = True) -> dict:
    idx, pos, m = _confirm_rows(result, len(folds))
    day = model_eval._day_numbers(design.dates)[idx]
    record = _ranking_record(result.confirm, result.oof["y"][m].astype(float), result.oof["p_raw"][m],
                             design.fwd[idx], day, pos, h)
    if calibration:
        record["cal_variants"] = _calibration_variants(result, folds, design, h, result.confirm)
    record["calib_slopes"] = [row.get("calib_slope") for row in result.folds]
    return record


def _momentum_record(ev: dict, panel: pd.DataFrame, h: int) -> dict:
    design, folds = ev["design"], ev["folds"]
    momentum = ev["baselines"]["momentum"]
    k = int(momentum["confirm_folds"])
    column, _ = model_training._momentum_column(panel, design)
    idx = np.concatenate([f.test_idx for f in folds[-k:]])
    pos = np.concatenate([np.full(f.test_idx.size, i) for i, f in enumerate(folds[-k:])])
    v = column[idx]
    p = np.where(np.isnan(v), 0.5, np.where(v > 0, 0.55, 0.45))
    day = model_eval._day_numbers(design.dates)[idx]
    return _ranking_record(momentum["confirm"], design.y[idx], p, design.fwd[idx], day, pos, h)


def _fixed_rules(panel: pd.DataFrame, design, folds: list, h: int, confirm_folds: int) -> dict:
    """Per fixed rule, the confirm folds' mean AUC across dates and on the same date."""
    out = {}
    for column, sign in FIXED_RULES:
        values = sign * panel.loc[design.rows, column].to_numpy(dtype=float)
        pooled, same = [], []
        for fold in folds[-confirm_folds:]:
            idx = fold.test_idx[np.isfinite(values[fold.test_idx])]
            pooled.append(model_eval._auc(design.y[idx], values[idx]))
            same.append(same_date_auc(design.y[idx], values[idx], design.dates[idx]))
        out[f"{'-' if sign < 0 else '+'}{column}"] = {"pooled": float(np.mean(pooled)),
                                                      "same_date": float(np.mean(same))}
    return out


def _score_horizon(task: dict, inputs: PanelInputs, panel: pd.DataFrame, h: int) -> dict:
    kind, seed = task["kind"], int(task["seed"])
    cfg = model_configs.config_for_horizon(h)
    plan = model_configs.fold_plan(cfg, h)
    record: dict[str, Any] = {"plan": list(plan), "base_rate": float(np.nanmean(panel[f"y_{h}"])),
                              "status": "evaluated"}
    if kind in ("hgb", "momentum"):
        model_cfg = cfg if kind == "hgb" else model_configs.get_config("momentum")
        ev = model_training.evaluate_config(panel, inputs, model_cfg, h, n_threads=1,
                                            importance=False, n_boot=N_BOOT)
        if ev["status"] != "evaluated":
            return {**record, "status": ev["status"], "error": ev["error"]}
        record["momentum"] = _momentum_record(ev, panel, h)
        if kind == "hgb":
            record["hgb"] = _fitted_record(ev["result"], ev["folds"], ev["design"], h)
        else:
            record["rules"] = _fixed_rules(panel, ev["design"], ev["folds"], h, plan.confirm_folds)
        return record

    design = model_training.build_design(panel, cfg, h, inputs=inputs)
    if kind == "power":
        design = with_signal(design, task["scales"][f"{h}:{task['target_auc']:.2f}"], seed)
    try:
        folds = model_training.make_folds(design, cfg)
    except ValueError as exc:
        return {**record, "status": "not_measurable", "error": str(exc)}
    if kind == "power":
        confirm_idx = np.concatenate([f.test_idx for f in folds[-plan.confirm_folds:]])
        signal = design.X[confirm_idx, -1]
        record["feature_auc"] = same_date_auc(design.y[confirm_idx], signal, design.dates[confirm_idx])
        record["feature_auc_pooled"] = model_eval._auc(design.y[confirm_idx], signal)
        result = model_training.evaluate_design(cfg, design, folds, n_threads=1, importance=False,
                                                n_boot=N_BOOT)
        record["hgb"] = _fitted_record(result, folds, design, h)
        return record

    # kind == "probe": the production estimator on block-permuted training labels
    day = model_eval._day_numbers(design.dates)
    codes = pd.factorize(pd.Series(design.tickers))[0]
    factory = functools.partial(BlockPermutedLabels, inner=model_training.factory_for(cfg, design),
                                block=PERMUTED_BLOCK_SESSIONS, seed=seed)
    result = model_eval.evaluate(factory, np.column_stack([design.X, day, codes]), design.y, design.fwd,
                                 design.dates, design.tickers, folds, label_dates=design.label_dates,
                                 horizon=h, confirm_folds=plan.confirm_folds, importance=False,
                                 n_threads=1, n_boot=N_BOOT)
    record["hgb"] = _fitted_record(result, folds, design, h, calibration=False)
    return record


def score_panel(task: dict) -> dict:
    """One panel, every horizon the task names. Runs in a worker process."""
    started = time.perf_counter()
    labels, drift = NULLS[task["null"]]
    inputs, panel = build_null_panel(int(task["seed"]), labels=labels, drift=drift)
    out: dict[str, Any] = {"kind": task["kind"], "seed": int(task["seed"]), "null": task["null"],
                           "target_auc": task.get("target_auc"), "rows": int(len(panel)),
                           "horizons": {}}
    for h in task["horizons"]:
        out["horizons"][str(h)] = _score_horizon(task, inputs, panel, int(h))
    out["seconds"] = time.perf_counter() - started
    return out


def _init_worker() -> None:
    logging.getLogger().setLevel(logging.ERROR)


def build_tasks(sizes: dict, horizons: list[int], scales: dict) -> list[dict]:
    """One task per (study, seed): the seed runs every horizon whose panel count reaches it."""

    def seeds(counts: dict) -> dict[int, list[int]]:
        wanted = {h: n for h, n in counts.items() if h in horizons}
        return {seed: [h for h in horizons if wanted.get(h, 0) >= seed]
                for seed in range(1, 1 + max(wanted.values(), default=0))}

    tasks = []
    for null, key in (("exact", "hgb_exact"), ("path", "hgb_path")):
        tasks += [{"kind": "hgb", "null": null, "seed": s, "horizons": hs}
                  for s, hs in seeds(sizes[key]).items()]
    for target in SIGNAL_AUCS:
        tasks += [{"kind": "power", "null": "path", "seed": s, "horizons": hs, "target_auc": target,
                   "scales": scales} for s, hs in seeds(sizes["power"]).items()]
    tasks += [{"kind": "probe", "null": "path", "seed": s, "horizons": hs}
              for s, hs in seeds(sizes["probe"]).items()]
    for null in NULLS:
        tasks += [{"kind": "momentum", "null": null, "seed": s, "horizons": hs}
                  for s, hs in seeds(sizes["momentum"]).items()]
    return [t for t in tasks if t["horizons"]]


# ── Aggregation ──────────────────────────────────────────────────────────────

def _finite(values) -> np.ndarray:
    v = np.asarray([math.nan if x is None else x for x in values], dtype=float)
    return v[np.isfinite(v)]


def _rate(flags: list[bool]) -> str:
    n = len(flags)
    if not n:
        return "n/a"
    p = float(np.mean(flags))
    return f"{100 * p:.1f}% +/- {100 * math.sqrt(p * (1 - p) / n):.1f}"


def _mean(values, digits: int = 3, signed: bool = False) -> str:
    v = _finite(values)
    if not v.size:
        return "n/a"
    return f"{v.mean():+.{digits}f}" if signed else f"{v.mean():.{digits}f}"


def _mean_se(values, digits: int = 3) -> str:
    v = _finite(list(values))
    if v.size < 2:
        return _mean(v, digits)
    return f"{v.mean():.{digits}f} +/- {v.std(ddof=1) / math.sqrt(v.size):.{digits}f}"


def _records(panels: list[dict], kind: str, h: int, key: str, *, null: Optional[str] = None,
             target: Optional[float] = None) -> list[dict]:
    out = []
    for panel in panels:
        if panel["kind"] != kind or (null is not None and panel["null"] != null):
            continue
        if target is not None and panel.get("target_auc") != target:
            continue
        record = panel["horizons"].get(str(h)) or {}
        if record.get("status") == "evaluated" and record.get(key):
            out.append(record[key])
    return out


def _above(low) -> bool:
    return low is not None and math.isfinite(low) and low > 0.5


def _below(high) -> bool:
    return high is not None and math.isfinite(high) and high < 0.5


def _width_ratio(records: list[dict], ci_of) -> tuple[float, float, float]:
    """(mean CI width, 2 x 1.645 x sd of the point AUC across panels, their ratio)."""
    aucs = _finite(r["auc"] for r in records)
    sd = float(aucs.std(ddof=1)) if aucs.size > 1 else math.nan
    widths = _finite((ci_of(r)[1] - ci_of(r)[0]) if ci_of(r)[0] is not None and ci_of(r)[1] is not None
                     else math.nan for r in records)
    target = 2 * 1.645 * sd
    width = float(widths.mean()) if widths.size else math.nan
    return width, target, (width / target if math.isfinite(target) and target > 0 else math.nan)


def _not_measurable(panels: list[dict], kind: str, null: Optional[str], h: int) -> int:
    return sum(1 for p in panels if p["kind"] == kind and (null is None or p["null"] == null)
               and (p["horizons"].get(str(h)) or {}).get("status") == "not_measurable")


NULL_HEADER = ("| Horizon | Null | Panels | Test blocks | CI low > 0.50 | CI high < 0.50 | False ship | "
               "Mean CI width | 2 x 1.645 x sd | Width ratio | CI n/a | n_eff | Brier skill |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|---|")


def _null_row(h: int, null: str, records: list[dict], plan, not_measurable: int) -> str:
    width, target, ratio = _width_ratio(records, lambda r: (r["ci_low"], r["ci_high"]))
    nan_ci = sum(1 for r in records if r["ci_low"] is None or not math.isfinite(r["ci_low"]))
    panels = f"{len(records)}{f' (+{not_measurable} n/m)' if not_measurable else ''}"
    return (f"| {h}d | {null} | {panels} | {plan[1]}d x {plan[0]} | "
            f"{_rate([_above(r['ci_low']) for r in records])} | "
            f"{_rate([_below(r['ci_high']) for r in records])} | "
            f"{_rate([r['decision'] == 'model' for r in records])} | {width:.3f} | {target:.3f} | "
            f"{ratio:.2f} | {nan_ci} | {_mean([r['n_eff'] for r in records], 1)} | "
            f"{_mean((r['brier_skill'] for r in records), 4, signed=True)} |")


def _plan_of(panels: list[dict], h: int):
    for panel in panels:
        record = panel["horizons"].get(str(h))
        if record:
            return record["plan"]
    return (0, 0, 0)


def _horizons_with(panels: list[dict], kind: str, horizons: list[int]) -> list[int]:
    return [h for h in horizons if any(p["kind"] == kind and str(h) in p["horizons"] for p in panels)]


def render_report(panels: list[dict], horizons: list[int], scales: dict, lookahead: list[dict],
                  args, wall: float) -> str:
    block_label, cal_label = production_variant_labels()
    cpu = sum(p["seconds"] for p in panels)

    def count(kind: str, null: Optional[str] = None) -> int:
        return len({p["seed"] for p in panels if p["kind"] == kind and (null is None or p["null"] == null)})

    lines = [
        "# Null calibration and power of the ship rule", "",
        f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC by scripts/manual/null_ship_rate.py"
        f"{' --quick' if args.quick else ''}: {len(panels)} panel runs, {wall / 60:.1f} min wall, "
        f"{cpu / 60:.1f} min CPU on {args.workers} workers, one thread per fit.", "",
        f"- Panels: {N_TICKERS} tickers x {N_SESSIONS:,} sessions (half listed from the start, half "
        f"staggered over the first 60%), about {np.mean([p['rows'] for p in panels]):,.0f} panel rows; "
        f"market beta 0.5-1.5, three sector factors, GARCH(1,1), Student-t df {T_DF}, log-return drift "
        f"{SHARPE_PER_SESSION} x conditional sd per session.",
        "- Nulls: `exact` = labels from an independent path on the same structure (the calibration "
        "test); `path` = labels from the path the features are built from (realistic dependence "
        "between a forecast and earlier labels). No feature can predict a label on either.",
        "- Base up-rates (drift 0.035): " + ", ".join(
            f"{h}d {_mean((p['horizons'][str(h)]['base_rate'] for p in panels if str(h) in p['horizons'] and NULLS[p['null']][1] > 0), 3)}"
            for h in horizons) + ".",
        "- Production path: each horizon's production config (" + ", ".join(
            f"{h}d {model_configs.PRODUCTION[h]}" for h in horizons) + ") and its HGB parameters via "
        "model_training.factory_for, model_training.evaluate_config, the production fold plan, "
        f"{N_BOOT} bootstrap resamples.",
        f"- Ship rule under test: confirm n_eff >= {MIN_CONFIRM_N_EFF:g}, same-date AUC >= "
        f"{MIN_SHIP_AUC:.2f}, its 90% CI low > 0.50 (circular blocks of {block_label} drawn jointly "
        "across the confirm folds, the percentile interval widened by model_eval.interval_inflation), "
        f"Brier skill > 0 (calibration {cal_label}), same-date decile spread > 0.",
        f"- Seeds: HGB nulls 1..{max(count('hgb', 'exact'), count('hgb', 'path'))}, momentum nulls "
        f"1..{count('momentum', 'exact')}, power 1..{count('power')}, probe 1..{count('probe')}, "
        f"pilot panels {PILOT_SEEDS} for the signal noise scale.",
        "- Rates are percent +/- Monte Carlo standard error sqrt(p(1-p)/n); means are +/- their "
        "standard error. Width ratio = mean reported 90% CI width / (2 x 1.645 x empirical sd of the "
        "confirm AUC across panels); an honest interval sits near 1. CI n/a = the interval could not be "
        "drawn (a confirm fold shorter than two blocks), which the ship rule treats as a failure.", "",
        "## Diagnosis: why the rule ranks tickers on the same date", "",
        "The quick study behind this change found the production HGB clearing its interval on null "
        "panels far more often than 5%. The probes below separate the candidate causes. Probe 1: no "
        "feature sees a later bar. Probe 3: a model fitted on permuted training labels clears no "
        "interval, so training leaks nothing. Probe 2: fixed rules that no fitting could have tuned score far from "
        "0.50 when a fold's rows are ranked across dates on the path null - identically with zero "
        "drift, where the labels are strictly unpredictable whatever the volatility does, so this is no "
        "signal in the simulator - and at 0.50 on the exact null or when only same-date rows are "
        "compared. The cause is the across-dates ranking itself: a forecast made late in a test block "
        "holds the returns that decided the labels of the block's earlier rows. A forecast that leans "
        "against recent moves is low just after rows that went up and high just after rows that went "
        "down, so the across-dates AUC rewards it, and punishes a trend-follower the same way, whether "
        "or not either can predict anything. Comparing only rows that share a forecast date removes the "
        "effect by construction, so `auc` and the decile spread now do that; the across-dates figure is "
        "kept as `auc_pooled` for reports. Real prices carry the same dependence, so the rule it "
        "replaces would have been fooled on real data too. Once the bias was gone the bare percentile "
        "interval was still too narrow - blocks of h sessions miss part of the dependence between "
        "overlapping labels, and a confirm period holds few blocks - so it is now widened by "
        "model_eval.interval_inflation; the last variant table shows both.", "",
        "### Probe 1: no lookahead", "",
        "| Seed | Cut | Rows on or before the cut | Same rows | Features bit-identical | "
        "Resolved labels identical | Future bars altered |", "|---|---|---|---|---|---|---|"]
    for r in lookahead:
        lines.append(f"| {r['seed']} | {r['cut']} | {r['rows_before_cut']:,} | {r['same_rows']} | "
                     f"{r['features_identical']} | {r['resolved_labels_identical']} | {r['future_altered']} |")

    rule_horizons = _horizons_with(panels, "momentum", horizons)
    lines += ["", "### Probe 2: fixed rules on the confirm folds (no fitting)", "",
              "Mean confirm AUC over panels (+/- standard error). Market-level columns are the same for "
              "every ticker on a date, so their same-date AUC is exactly 0.50.", "",
              "| Horizon | Rule | Path: across dates | Path: same date | Path, zero drift: across dates | "
              "Exact: across dates | Exact: same date |", "|---|---|---|---|---|---|---|"]
    for h in rule_horizons:
        path_rules = _records(panels, "momentum", h, "rules", null="path")
        zero_rules = _records(panels, "momentum", h, "rules", null="path, zero drift")
        exact_rules = _records(panels, "momentum", h, "rules", null="exact")
        for column, sign in FIXED_RULES:
            name = f"{'-' if sign < 0 else '+'}{column}"
            lines.append(f"| {h}d | {name} | {_mean_se(r[name]['pooled'] for r in path_rules)} | "
                         f"{_mean_se(r[name]['same_date'] for r in path_rules)} | "
                         f"{_mean_se(r[name]['pooled'] for r in zero_rules)} | "
                         f"{_mean_se(r[name]['pooled'] for r in exact_rules)} | "
                         f"{_mean_se(r[name]['same_date'] for r in exact_rules)} |")

    lines += ["", "### Production HGB: across dates (before) vs same date (the rule)", "",
              "The same fitted models and confirm rows, scored both ways. `Before` is the rule as it "
              "stood: across-dates AUC and decile spread, blocks of 1h.", "",
              "| Horizon | Null | Panels | Before: mean AUC | Before: CI low > 0.50 | Before: false ship | "
              "Same date: mean AUC | Same date: CI low > 0.50 | Same date: false ship |",
              "|---|---|---|---|---|---|---|---|---|"]
    for h in _horizons_with(panels, "hgb", horizons):
        for null in ("exact", "path"):
            recs = _records(panels, "hgb", h, "hgb", null=null)
            if not recs:
                continue
            lines.append(f"| {h}d | {null} | {len(recs)} | {_mean_se(r['pooled']['auc'] for r in recs)} | "
                         f"{_rate([_above(r['pooled']['ci_low']) for r in recs])} | "
                         f"{_rate([r['pooled']['decision'] == 'model' for r in recs])} | "
                         f"{_mean_se(r['auc'] for r in recs)} | {_rate([_above(r['ci_low']) for r in recs])} | "
                         f"{_rate([r['decision'] == 'model' for r in recs])} |")

    lines += ["", "### Probe 3: permuted training labels", "",
              f"Path panels; each fold's training dates are permuted in runs of {PERMUTED_BLOCK_SESSIONS} "
              "sessions, test rows untouched.", "",
              "| Horizon | Panels | Across dates: mean AUC | Across dates: CI low > 0.50 | Same date: mean AUC "
              "| Same date: CI low > 0.50 | Same date: false ship |", "|---|---|---|---|---|---|---|"]
    for h in _horizons_with(panels, "probe", horizons):
        recs = _records(panels, "probe", h, "hgb")
        lines.append(f"| {h}d | {len(recs)} | {_mean_se(r['pooled']['auc'] for r in recs)} | "
                     f"{_rate([_above(r['pooled']['ci_low']) for r in recs])} | {_mean_se(r['auc'] for r in recs)} | "
                     f"{_rate([_above(r['ci_low']) for r in recs])} | "
                     f"{_rate([r['decision'] == 'model' for r in recs])} |")

    lines += ["", "## Null calibration of the ship rule", "", "### Momentum baseline (no fitting)", "",
              *NULL_HEADER]
    for h in rule_horizons:
        for null in NULLS:
            recs = _records(panels, "momentum", h, "momentum", null=null)
            if recs:
                lines.append(_null_row(h, null, recs, _plan_of(panels, h),
                                       _not_measurable(panels, "momentum", null, h)))
    lines += ["", "The momentum rule forecasts 55%/45% whatever the base rate, so its Brier skill is "
              "negative and it rarely passes the whole rule; its CI-low rate is the test of the interval.",
              "", "### Production HGB", "", *NULL_HEADER]
    for h in _horizons_with(panels, "hgb", horizons):
        for null in ("exact", "path"):
            recs = _records(panels, "hgb", h, "hgb", null=null)
            if recs:
                lines.append(_null_row(h, null, recs, _plan_of(panels, h),
                                       _not_measurable(panels, "hgb", null, h)))

    lines += ["", "## Power: production HGB with a signal column (path panels)", "",
              "| Horizon | Target AUC | Noise scale | Feature AUC, same date (confirm) | Feature AUC across "
              "dates (confirm) | Panels | HGB confirm AUC | CI low > 0.50 | Ship rate (power) | Brier skill |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for h in _horizons_with(panels, "power", horizons):
        for target in SIGNAL_AUCS:
            recs = _records(panels, "power", h, "hgb", target=target)
            rows = [p["horizons"][str(h)] for p in panels if p["kind"] == "power"
                    and p.get("target_auc") == target and str(h) in p["horizons"]]
            lines.append(f"| {h}d | {target:.2f} | {scales.get(f'{h}:{target:.2f}', math.nan):.2f} | "
                         f"{_mean(r.get('feature_auc') for r in rows)} | "
                         f"{_mean(r.get('feature_auc_pooled') for r in rows)} | {len(recs)} | "
                         f"{_mean(r['auc'] for r in recs)} | {_rate([_above(r['ci_low']) for r in recs])} | "
                         f"{_rate([r['decision'] == 'model' for r in recs])} | "
                         f"{_mean((r['brier_skill'] for r in recs), 4, signed=True)} |")

    lines += ["", "## Variant: the interval (same-date AUC)", "",
              "The same confirm-fold predictions re-scored with blocks of L sessions drawn jointly across "
              "the confirm folds, as the bare percentile interval or widened by "
              "model_eval.interval_inflation (block taper, finite blocks, Student t). The production "
              f"choice is `{block_label}`, widened.", ""]
    for h in horizons:
        mom = _records(panels, "momentum", h, "momentum", null="exact")
        exact = _records(panels, "hgb", h, "hgb", null="exact")
        path = _records(panels, "hgb", h, "hgb", null="path")
        if not (mom or exact or path):
            continue
        lines += [f"### {h}d", "", "| Blocks | Momentum exact: CI low > 0.50 | width ratio | HGB exact: CI low "
                  "> 0.50 | false ship | width ratio | CI n/a | HGB path: CI low > 0.50 | false ship | "
                  "Power ship @0.55 | Power ship @0.60 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for key, mult, widened in CI_VARIANTS:

            def ci(r, key=key):
                return r["ci_variants"][key]["ci_low"], r["ci_variants"][key]["ci_high"]

            power = [_rate([r["ci_variants"][key]["decision"] == "model"
                            for r in _records(panels, "power", h, "hgb", target=t)]) for t in SIGNAL_AUCS]
            nan_ci = sum(1 for r in exact if not _finite([ci(r)[0]]).size)
            mark = " (production)" if key == block_label and widened else ""
            lines.append(
                f"| {key} ({max(1, math.ceil(mult * h))} sessions){mark} | "
                f"{_rate([_above(ci(r)[0]) for r in mom])} | {_width_ratio(mom, ci)[2]:.2f} | "
                f"{_rate([_above(ci(r)[0]) for r in exact])} | "
                f"{_rate([r['ci_variants'][key]['decision'] == 'model' for r in exact])} | "
                f"{_width_ratio(exact, ci)[2]:.2f} | {nan_ci} | {_rate([_above(ci(r)[0]) for r in path])} | "
                f"{_rate([r['ci_variants'][key]['decision'] == 'model' for r in path])} | "
                f"{power[0]} | {power[1]} |")
        lines.append("")

    lines += ["## Variant: calibration", "",
              "Confirm-fold Brier skill of the same HGB predictions under each calibrator, fitted per fold "
              "on the purged earlier out-of-fold rows exactly as evaluate does. Null = exact-null panels: "
              "mean and 5th percentile. The ship columns apply the whole rule with that Brier skill. The "
              f"production choice is `{cal_label}`.", ""]
    for h in horizons:
        null = _records(panels, "hgb", h, "hgb", null="exact")
        if not null:
            continue
        lines += [f"### {h}d", "", "| Calibration | Null Brier skill (mean / p5) | Null false ship | "
                  "Brier skill @0.55 | Brier skill @0.60 | Ship @0.55 | Ship @0.60 |",
                  "|---|---|---|---|---|---|---|"]
        for label, _ in CAL_VARIANTS:
            skill = _finite(r["cal_variants"][label]["brier_skill"] for r in null)
            p5 = f"{np.percentile(skill, 5):+.4f}" if skill.size else "n/a"
            cells = []
            for t in SIGNAL_AUCS:
                recs = _records(panels, "power", h, "hgb", target=t)
                cells.append((_mean((r["cal_variants"][label]["brier_skill"] for r in recs), 4, signed=True),
                              _rate([r["cal_variants"][label]["decision"] == "model" for r in recs])))
            mark = " (production)" if label == cal_label else ""
            lines.append(f"| {label}{mark} | {_mean(skill, 4, signed=True)} / {p5} | "
                         f"{_rate([r['cal_variants'][label]['decision'] == 'model' for r in null])} | "
                         f"{cells[0][0]} | {cells[1][0]} | {cells[0][1]} | {cells[1][1]} |")
        lines.append("")
    return "\n".join(lines) + "\n"


# ── Main ─────────────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--quick", action="store_true", help="a few panels per study, for iterating")
    parser.add_argument("--smoke", action="store_true", help="one panel per study: proves the pipeline runs")
    parser.add_argument("--workers", type=int, default=2,
                        help="worker processes; each fits with one thread, so this is the cores used")
    parser.add_argument("--horizons", default=None, help="comma-separated, e.g. 5,21 (default all)")
    parser.add_argument("--tag", default="", help="suffix for the output files")
    parser.add_argument("--from-json", default=None,
                        help="re-render the report from a saved null_ship_rate JSON without computing")
    args = parser.parse_args(argv)
    horizons = [int(x) for x in args.horizons.split(",")] if args.horizons else list(HORIZONS)
    sizes = SMOKE if args.smoke else QUICK if args.quick else FULL
    args.quick = args.quick or args.smoke
    logging.getLogger().setLevel(logging.ERROR)
    stem = "null_ship_rate" + (f"_{args.tag}" if args.tag else "")
    if args.from_json:
        saved = json.loads(Path(args.from_json).read_text(encoding="utf-8"))
        args.quick = bool(saved.get("quick"))
        args.workers = saved.get("workers", args.workers)
        report = render_report(saved["panels"], saved["horizons"], saved["scales"], saved["lookahead"],
                               args, float(saved.get("wall_seconds") or 0.0))
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / f"{stem}.md").write_text(report, encoding="utf-8")
        print(report)
        print(f"re-rendered {OUT_DIR / (stem + '.md')} from {args.from_json}")
        return
    # One OpenMP/BLAS thread per worker process, inherited when the pool spawns them.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"

    started = time.perf_counter()
    print(f"no-lookahead probe on seeds {LOOKAHEAD_SEEDS} ...", flush=True)
    lookahead = [check_no_lookahead(seed) for seed in LOOKAHEAD_SEEDS]
    print(f"calibrating the signal noise on pilot panels {PILOT_SEEDS} ...", flush=True)
    scales = calibrate_signal_scales([h for h in horizons if h in sizes["power"]])
    print("  " + ", ".join(f"{k} -> {v:.2f}" for k, v in scales.items()), flush=True)

    tasks = build_tasks(sizes, horizons, scales)
    panels = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
        futures = [pool.submit(score_panel, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), start=1):
            panels.append(future.result())
            if done % max(1, len(tasks) // 20) == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} panels, {(time.perf_counter() - started) / 60:.1f} min",
                      flush=True)
    panels.sort(key=lambda p: (p["kind"], p["null"], p.get("target_auc") or 0, p["seed"]))
    wall = time.perf_counter() - started

    report = render_report(panels, horizons, scales, lookahead, args, wall)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{stem}.md").write_text(report, encoding="utf-8")
    (OUT_DIR / f"{stem}.json").write_text(
        json.dumps({"scales": scales, "horizons": horizons, "quick": args.quick, "workers": args.workers,
                    "wall_seconds": wall, "lookahead": lookahead, "panels": panels},
                   default=lambda o: None), encoding="utf-8")
    print()
    print(report)
    print(f"wrote {OUT_DIR / (stem + '.md')} in {wall / 60:.1f} min")


if __name__ == "__main__":
    main()
