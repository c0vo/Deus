"""
Deus - walk-forward evaluation harness for the pooled direction model.

Everything here is a pure function over numpy / pandas objects: no database, no
files, no settings. The predictor (weekly retrain) and the desktop experiment
runner both call it, so the numbers in the Telegram table and the numbers in
LEADERBOARD.md come from the same code path.

Why each piece exists:

* ``walkforward_folds`` purges and embargoes. Labels span ``h`` sessions, so a
  plain ``TimeSeriesSplit`` lets the last training rows' label windows reach into
  the validation block. Here a training row must have its *label date* strictly
  before ``test_start - embargo``, and splits are by date across all tickers.
* ``metrics_block`` reports skill against the base rate (Brier skill, accuracy
  over always-majority), not raw accuracy: a 252d model that is "80% accurate"
  in a market that went up 80% of the time has no skill.
* Ranking skill (``auc``, ``decile_spread``) compares only rows that share a
  date. Ranking a fold's rows across dates is biased even when nothing is
  predictable: a forecast made late in a test block already contains the
  returns that decided the labels of the block's earlier rows, so a forecast
  that leans against recent moves scores above 0.50 and one that follows them
  below: 0.57-0.59 for a fixed rule leaning against the market's distance from
  its 200-day average on simulated random walks, 0.535 for the production
  model at 21d (scripts/manual/null_ship_rate.py). Pairs on one date share a
  forecast date, so no such contamination exists. The across-dates figure is
  still reported as ``auc_pooled``; no decision reads it. Probability quality
  (Brier skill) is a per-row score and needs no such care.
* ``evaluate`` calibrates each fold with a calibrator fitted only on earlier
  folds' out-of-fold predictions, so the calibrated metrics are leakage-free.
* ``ship_decision`` reads the last folds only (``confirm``), which play the role
  of a holdout that hyperparameter selection (``selection``) never looked at.

Dates may be anything ``pandas.to_datetime`` understands (datetime64 of any
unit, tz-aware or naive, ISO strings); internally they become integer day
numbers, which behave the same under pandas 2.x and 3.x.
"""

from __future__ import annotations

import functools
import inspect
import math
import time
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Callable, Literal, Optional

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from config.logging_config import get_logger

log = get_logger(__name__)

# Probabilities are clipped to [EPS, 1 - EPS] wherever a log is taken.
EPS = 1e-6
# Rows needed before a calibrator is fitted; below this it answers the base rate.
MIN_CALIBRATION_ROWS = 50
# Prior weight, in independent label windows, of "the scores say nothing beyond
# the base rate". A calibrator's slope is shrunk towards the base rate as if
# this many windows had shown no relationship (see fit_calibrator).
CALIBRATION_PRIOR_N_EFF = 20.0
# Answer the base rate rather than invert a model whose earlier out-of-fold
# scores ranked the wrong way round.
CALIBRATION_NONNEGATIVE_SLOPE = False
# Minimum rows for a decile spread (a top and bottom decile of one or two rows
# each is noise, not a ranking).
MIN_DECILE_ROWS = 20
# Thread cap around every fit and predict. The phone's SoC is shared with the
# API process and the rest of the worker; HistGradientBoosting would otherwise
# take every core through OpenMP.
DEFAULT_THREADS = 2
CALIBRATION_METHODS = ("platt", "isotonic", "identity")
# (threshold on |p - 0.5|, metric key)
HI_CONF_LEVELS = ((0.10, "hi_conf_10"), (0.15, "hi_conf_15"))
# |0.6 - 0.5| is 0.09999999999999998 in floating point; without a tolerance a
# 60% call would not count as a 10-point call.
_HI_CONF_TOL = 1e-9
_NAT_DAY = np.iinfo(np.int64).min
# Bootstrap block length per session of label horizon (see block_length_for).
BLOCK_LENGTH_MULT = 1.0
# Fewest bootstrap blocks an interval is drawn from: its spread is a variance
# estimated from that many blocks (see interval_inflation).
MIN_INTERVAL_BLOCKS = 4.0
# Draw bootstrap blocks once over every date of an aggregate (so a block may
# straddle two adjacent folds) rather than separately within each fold.
BOOTSTRAP_JOINT = True
# The ship rule's floor on the confirm folds' fold-averaged AUC.
MIN_SHIP_AUC = 0.53
# Independent label windows (distinct confirm dates / horizon) the confirm
# folds must hold before the ship rule believes anything they say.
MIN_CONFIRM_N_EFF = 10.0


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------

def _day_numbers(values) -> np.ndarray:
    """
    Days since 1970-01-01 as int64, with NaT mapped to ``_NAT_DAY``.

    Accepts datetime64 of any unit, tz-aware input (converted to UTC, then made
    naive), pandas Series/Index, ISO strings and date objects. Integers are
    rejected: ``to_datetime`` would silently read them as nanoseconds.
    """
    if not isinstance(values, (pd.Series, pd.Index)):
        values = np.asarray(values)
        if values.ndim == 0:
            values = values.reshape(1)
    if np.issubdtype(np.asarray(values).dtype, np.integer):
        raise TypeError("dates must be datetime-like, not integers")
    idx = pd.DatetimeIndex(pd.to_datetime(values))
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx.to_numpy().astype("datetime64[D]").astype(np.int64)


def _day_to_timestamp(day: int) -> pd.Timestamp:
    return pd.Timestamp(np.datetime64(int(day), "D"))


def _iso(day) -> Optional[str]:
    return None if int(day) == _NAT_DAY else str(np.datetime64(int(day), "D"))


def _date_codes(dates) -> np.ndarray:
    """Integer code per distinct date value; used only to group rows by date."""
    if not isinstance(dates, (pd.Series, pd.Index)):
        dates = np.asarray(dates)
    codes = np.asarray(pd.factorize(dates)[0], dtype=np.int64)
    if codes.size and codes.min() < 0:          # NaT / None rows form one group
        codes = np.where(codes < 0, codes.max() + 1, codes)
    return codes


# ---------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------

@dataclass
class Fold:
    """One walk-forward split. Indices are row positions into the panel arrays."""

    index: int
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_idx: np.ndarray
    test_idx: np.ndarray
    horizon: int = 0
    embargo_days: int = 0


def _default_embargo(horizon: int, embargo_days: Optional[int]) -> int:
    # ~1.5 calendar days per session: about one label window of extra gap on top
    # of the purge, in the calendar days the fold boundaries are measured in.
    return int(math.ceil(1.5 * horizon)) if embargo_days is None else int(embargo_days)


def _fold_inputs(dates, label_dates) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    day = _day_numbers(dates)
    label_day = _day_numbers(label_dates)
    if day.shape != label_day.shape:
        raise ValueError(
            f"dates ({day.size}) and label_dates ({label_day.size}) differ in length"
        )
    labeled = (day != _NAT_DAY) & (label_day != _NAT_DAY)
    return day, label_day, labeled


def _check_fold_args(horizon: int, n_folds: int, test_days: int, embargo: int) -> None:
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if n_folds < 1:
        raise ValueError(f"n_folds must be >= 1, got {n_folds}")
    if test_days < 1:
        raise ValueError(f"test_days must be >= 1, got {test_days}")
    if embargo < 0:
        raise ValueError(f"embargo_days must be >= 0, got {embargo}")


def _scan_blocks(day, label_day, labeled, horizon, *, n_folds, test_days, embargo,
                 min_train_days, first_test_day=None) -> list[Fold]:
    """
    Walk test blocks backwards from the last labeled date; return them oldest first.

    Walking stops at the first block whose training set is empty or spans fewer
    than ``min_train_days`` (every older block trains on a subset of that
    history), or that starts before ``first_test_day``. A block with no labeled
    rows (a data gap) is skipped without stopping.
    """
    last = int(day[labeled].max())
    found = []
    for k in range(n_folds):
        end = last - k * test_days
        start = end - test_days + 1
        if first_test_day is not None and start < first_test_day:
            break
        # The purge and embargo: every training label is known strictly before
        # start - embargo. `day < start` follows for sane data; it is kept so the
        # sides stay disjoint even if a label date precedes its row date.
        train = labeled & (label_day < start - embargo) & (day < start)
        if not train.any():
            break
        if int(day[train].max() - day[train].min()) < min_train_days:
            break
        test = labeled & (day >= start) & (day <= end)
        if not test.any():
            continue
        found.append((start, end, np.flatnonzero(train), np.flatnonzero(test)))
    found.reverse()
    return [
        Fold(index=i, test_start=_day_to_timestamp(s), test_end=_day_to_timestamp(e),
             train_idx=tr, test_idx=te, horizon=int(horizon), embargo_days=int(embargo))
        for i, (s, e, tr, te) in enumerate(found)
    ]


def walkforward_folds(dates, label_dates, horizon: int, *, n_folds: int = 6,
                      test_days: int = 126, embargo_days: int | None = None,
                      min_train_days: int = 504) -> list[Fold]:
    """
    Purged, embargoed, expanding-window walk-forward folds, split by date.

    * Test blocks are consecutive ``test_days``-calendar-day windows ending at the
      last labeled row date, walking backwards; test rows have
      ``test_start <= date <= test_end``.
    * Train rows have ``label_date < test_start - embargo``, with ``embargo`` =
      ``embargo_days`` or ``ceil(1.5 * horizon)`` calendar days.
    * Rows with a NaT date or label date appear in no fold.
    * Rows of different tickers on the same date always land on the same side.
    * A fold is kept only if its training rows span >= ``min_train_days``
      calendar days, so fewer than ``n_folds`` may come back (oldest dropped
      first). Fewer than two raises ``ValueError``.

    Folds are returned oldest first, ``index`` 0..n-1.
    """
    embargo = _default_embargo(horizon, embargo_days)
    _check_fold_args(horizon, n_folds, test_days, embargo)
    if n_folds < 2:
        raise ValueError(f"walkforward_folds needs n_folds >= 2, got {n_folds}")
    day, label_day, labeled = _fold_inputs(dates, label_dates)
    if not labeled.any():
        raise ValueError("walkforward_folds: no labeled rows (every label_date is NaT)")

    folds = _scan_blocks(day, label_day, labeled, horizon, n_folds=n_folds,
                         test_days=test_days, embargo=embargo,
                         min_train_days=min_train_days)
    if len(folds) < 2:
        first, last = int(day[labeled].min()), int(day[labeled].max())
        raise ValueError(
            f"walkforward_folds(h={horizon}): {len(folds)} usable fold(s), need >= 2. "
            f"Labeled rows span {_iso(first)}..{_iso(last)} ({last - first} days); each "
            f"fold needs >= {min_train_days} days of training rows whose labels end "
            f"{embargo} days before a {test_days}-day test block."
        )
    log.debug("model_eval.folds", horizon=horizon, n_folds=len(folds), embargo_days=embargo,
              first_test=str(folds[0].test_start.date()),
              last_test=str(folds[-1].test_end.date()))
    return folds


def coverage_folds(dates, label_dates, horizon: int, coverage_start, *, n_folds: int = 3,
                   test_days: int = 21, embargo_days: int | None = None) -> list[Fold]:
    """
    Folds whose test blocks lie entirely on or after ``coverage_start``.

    For features that only exist since a recent date (news sentiment). Same
    purge and embargo as ``walkforward_folds``; training rows may come from any
    earlier date and there is no minimum training span. Returns an empty list,
    rather than raising, when fewer than two blocks fit, so a runner can report
    the horizon as "not measurable yet".
    """
    embargo = _default_embargo(horizon, embargo_days)
    _check_fold_args(horizon, n_folds, test_days, embargo)
    if coverage_start is None:
        return []
    first = int(_day_numbers([coverage_start])[0])
    day, label_day, labeled = _fold_inputs(dates, label_dates)
    if first == _NAT_DAY or not labeled.any():
        return []
    folds = _scan_blocks(day, label_day, labeled, horizon, n_folds=n_folds,
                         test_days=test_days, embargo=embargo, min_train_days=0,
                         first_test_day=first)
    if len(folds) < 2:
        log.debug("model_eval.coverage_folds_unmeasurable", horizon=horizon,
                  coverage_start=_iso(first), n_folds=len(folds))
        return []
    return folds


# ---------------------------------------------------------------------------
# AUC and its block bootstrap
# ---------------------------------------------------------------------------

def _auc(y, score) -> float:
    """ROC AUC, or 0.5 when only one class is present (AUC is undefined there)."""
    y = np.asarray(y)
    if y.size == 0 or np.unique(y).size < 2:
        return 0.5
    return float(roc_auc_score(y, score))


class _SameDateAUC:
    """
    AUC over the pairs of rows that share a date, under per-date multiplicities.

    For each date d, U_d counts that date's (up, down) row pairs whose scores
    are ordered the right way (ties one half) and P_d = n_up * n_down is the
    number of such pairs; the AUC is sum(w_d * U_d) / sum(w_d * P_d) for date
    weights w (1 each for the point estimate). A date whose rows all went the
    same way holds no pairs and so no weight. Both sums are formed once, so a
    bootstrap resample costs two dot products.

    Why only same-date pairs: see the module docstring. Under any null in which
    the tickers of one date share an up-probability, the expected U_d is exactly
    P_d / 2 whatever the forecasts depend on, so the estimate is unbiased even
    for forecasts built from the price path the labels come from.

    ``codes`` are chronological date codes shared by every group of one
    bootstrap; ``date_codes`` keeps the ones this row set holds, in date order.
    """

    def __init__(self, y: np.ndarray, p: np.ndarray, codes: np.ndarray):
        y = np.asarray(y, dtype=float).ravel()
        uniq, local = np.unique(np.asarray(codes).ravel(), return_inverse=True)
        local = np.asarray(local, dtype=np.int64).ravel()
        k = int(uniq.size)
        ranks = (pd.Series(np.asarray(p, dtype=float).ravel()).groupby(local)
                 .rank(method="average").to_numpy(dtype=float))
        n = np.bincount(local, minlength=k).astype(float)
        n_up = np.bincount(local, weights=y, minlength=k)
        rank_up = np.bincount(local, weights=ranks * y, minlength=k)
        self.pairs = n_up * (n - n_up)
        self.concordant = rank_up - n_up * (n_up + 1.0) / 2.0
        self.date_codes = np.asarray(uniq, dtype=np.int64)
        self.n_dates = k
        self.two_class = bool(self.pairs.sum() > 0)

    def auc(self, date_weights: Optional[np.ndarray] = None) -> float:
        if date_weights is None:
            pairs, concordant = float(self.pairs.sum()), float(self.concordant.sum())
        else:
            pairs = float(np.dot(date_weights, self.pairs))
            concordant = float(np.dot(date_weights, self.concordant))
        return concordant / pairs if pairs > 0 else math.nan


def same_date_auc(y, score, dates) -> float:
    """
    AUC counting only pairs of rows on the same date; 0.5 when no date holds
    both an up and a down row (no ranking evidence, as ``_auc`` answers for one
    class). This is the ``auc`` every metrics block and the ship rule use.
    """
    y = np.asarray(y, dtype=float).ravel()
    if y.size == 0:
        return 0.5
    value = _SameDateAUC(y, np.asarray(score, dtype=float).ravel(), _date_codes(dates)).auc()
    return value if math.isfinite(value) else 0.5


class _DateWeightedAUC:
    """
    AUC of one fixed row set under per-date multiplicities, sorted once.

    A date bootstrap re-weights whole dates; sorting by score once and scoring
    each resample with bincount/cumsum keeps 200 resamples of a 50k-row panel
    well under a second on the phone. Ties count one half, which is what the
    trapezoidal ROC AUC does, so with unit weights this equals roc_auc_score.

    ``codes`` are chronological date codes shared by every group of one
    bootstrap; ``date_codes`` keeps the ones this row set holds, in date order,
    so a draw over all dates can be narrowed to this set's local dates.
    """

    def __init__(self, y: np.ndarray, p: np.ndarray, codes: np.ndarray):
        order = np.argsort(p, kind="mergesort")
        self.y = y[order].astype(float)
        _, tie = np.unique(p[order], return_inverse=True)
        self.tie = np.asarray(tie).ravel()
        self.n_ties = int(self.tie.max()) + 1 if self.tie.size else 0
        uniq, local = np.unique(codes[order], return_inverse=True)
        self.date_of_row = np.asarray(local).ravel()
        self.date_codes = np.asarray(uniq, dtype=np.int64)
        self.n_dates = int(uniq.size)
        self.two_class = bool(self.y.size and 0.0 < self.y.mean() < 1.0)

    def auc(self, date_weights: Optional[np.ndarray] = None) -> float:
        w = 1.0 if date_weights is None else date_weights[self.date_of_row]
        pos = np.bincount(self.tie, weights=w * self.y, minlength=self.n_ties)
        neg = np.bincount(self.tie, weights=w * (1.0 - self.y), minlength=self.n_ties)
        n_pos, n_neg = pos.sum(), neg.sum()
        if n_pos <= 0 or n_neg <= 0:
            return math.nan
        below = np.cumsum(neg) - neg
        return float(np.dot(pos, below + 0.5 * neg) / (n_pos * n_neg))


def block_length_for(horizon: int) -> int:
    """Bootstrap block length in sessions for an ``horizon``-session label: ceil(BLOCK_LENGTH_MULT * h)."""
    return max(1, int(math.ceil(BLOCK_LENGTH_MULT * int(horizon))))


def _t_quantile(p: float, dof: float) -> float:
    """Student-t quantile by the four-term Cornish-Fisher expansion (the 95th percentile within 0.1% for dof >= 3)."""
    z = NormalDist().inv_cdf(p)
    g1 = (z ** 3 + z) / 4.0
    g2 = (5 * z ** 5 + 16 * z ** 3 + 3 * z) / 96.0
    g3 = (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / 384.0
    g4 = (79 * z ** 9 + 776 * z ** 7 + 1482 * z ** 5 - 1920 * z ** 3 - 945 * z) / 92160.0
    return z + g1 / dof + g2 / dof ** 2 + g3 / dof ** 3 + g4 / dof ** 4


def interval_inflation(horizon: int, block_length: int, n_dates: int, alpha: float = 0.10) -> float:
    """
    How far a block-bootstrap percentile interval over ``n_dates`` dates must be
    widened, around its point estimate, to cover at its nominal level when the
    labels overlap ``horizon`` sessions. The product of three factors:

    * taper: a block of L sessions keeps a share (1 - lag / L) of each lag's
      covariance, and labels overlapping h sessions correlate about (1 - lag / h)
      at each lag, so the bootstrap sees a share 1 - h / (3L) of the true
      variance when L >= h (and (L / h)(1 - L / (3h)) when L < h: about 1/h for
      single dates, which is why a date bootstrap stops widening with h);
    * finite blocks: a variance from k = n_dates / L blocks is short by (k - 1) / k;
    * the variance is itself estimated from k blocks, so the quantile is Student
      t with k - 1 degrees of freedom rather than normal.

    On simulated null panels (scripts/manual/null_ship_rate.py) the bare
    interval was 0.70-0.85 as wide as the spread of the confirm AUC it brackets
    and its low end cleared 0.50 in 6-10% of panels (nominal 5%); widened,
    0.9-1.1 as wide and 3-5%. NaN under MIN_INTERVAL_BLOCKS blocks: too few to
    estimate a spread at all.
    """
    h, length = max(int(horizon), 1), max(int(block_length), 1)
    blocks = float(n_dates) / length
    if not blocks >= MIN_INTERVAL_BLOCKS:
        return math.nan
    kept = 1.0 - h / (3.0 * length) if length >= h else (length / h) * (1.0 - length / (3.0 * h))
    level = 1.0 - alpha / 2.0
    return (math.sqrt(1.0 / kept) * math.sqrt(blocks / (blocks - 1.0))
            * _t_quantile(level, blocks - 1.0) / NormalDist().inv_cdf(level))


def _chronological_codes(dates) -> np.ndarray:
    """0..n_dates-1 per row, numbered in date order. Integers are taken as day numbers."""
    values = dates.to_numpy() if isinstance(dates, (pd.Series, pd.Index)) else np.asarray(dates)
    if values.ndim == 0:
        values = values.reshape(1)
    if np.issubdtype(values.dtype, np.integer):
        values = values.astype(np.int64).ravel()
    else:
        values = _day_numbers(dates)
    _, codes = np.unique(values, return_inverse=True)
    return np.asarray(codes, dtype=np.int64).ravel()


def _circular_block_weights(rng: np.random.Generator, n_dates: int, block_length: int,
                            n_boot: int) -> np.ndarray:
    """
    (n_boot, n_dates) date multiplicities from a circular block bootstrap.

    Each resample strings together ceil(n / L) blocks of L consecutive dates,
    each starting at a uniformly drawn date and wrapping from the last date to
    the first, and keeps the first n dates. Every date is equally likely to be
    drawn, which the non-circular moving-block scheme cannot say of the ends.
    """
    length = int(min(max(int(block_length), 1), n_dates))
    n_blocks = -(-n_dates // length)
    starts = rng.integers(0, n_dates, size=(int(n_boot), n_blocks))
    picked = (starts[:, :, None] + np.arange(length)).reshape(int(n_boot), -1)[:, :n_dates] % n_dates
    flat = (picked + (np.arange(int(n_boot), dtype=np.int64) * n_dates)[:, None]).ravel()
    return np.bincount(flat, minlength=int(n_boot) * n_dates).reshape(int(n_boot), n_dates).astype(float)


def block_bootstrap_auc(y, p, dates, n_boot: int = 200, seed: int = 0, alpha: float = 0.10,
                        *, groups=None, block_length: int = 1, joint: bool = True,
                        same_date: bool = True, horizon: int | None = None) -> tuple[float, float]:
    """
    Percentile interval of AUC under a circular block bootstrap over dates.

    ``same_date`` (the default, and what every metrics block reports) scores
    only pairs of rows on the same date (``same_date_auc``); False ranks all
    rows across dates as one set (``auc_pooled``, kept for diagnostics).

    ``horizon`` (the label length in sessions): when given, both ends of the
    percentile interval are pushed away from the point estimate by
    ``interval_inflation(horizon, block_length, n_dates, alpha)``, the width a
    block bootstrap misses on overlapping labels; every metrics block passes
    it. Without it the raw percentile interval comes back.

    Rows of one date are not independent (every ticker shares the market's
    move), so whole dates are resampled and all of a date's rows come along.
    Neighbouring dates are not independent either: an h-session label shares
    h - 1 sessions with the next date's, and the forecasts behind them barely
    move, so a draw of single dates treats near-duplicates as fresh evidence
    and the interval stops widening with the horizon. Dates are therefore
    drawn in runs of ``block_length`` consecutive sessions (pass
    ``block_length_for(h)``; 1 is the plain date bootstrap).

    ``groups`` (e.g. fold ids): the statistic is the mean of the per-group
    AUCs, i.e. the interval of a fold-averaged AUC such as ``auc_mean``. A
    group without a single (up, down) pair contributes 0.5, as it does in
    ``metrics_block``. With ``joint`` the blocks are drawn once over every
    date in the call, so a block may straddle two adjacent folds (whose
    boundary labels overlap); otherwise each group draws its own.

    NaN, NaN when the interval cannot be trusted: no row set with both
    classes, a two-class group spanning fewer than two blocks (its spread
    would be a guess), fewer than MIN_INTERVAL_BLOCKS blocks in all when
    ``horizon`` is given, or fewer than half the resamples usable (a resample
    that leaves a group without a pair is skipped).
    """
    y = np.asarray(y, dtype=float).ravel()
    p = np.asarray(p, dtype=float).ravel()
    if not (y.size == p.size == np.asarray(dates).size):
        raise ValueError("block_bootstrap_auc: y, p and dates differ in length")
    if y.size == 0:
        return math.nan, math.nan
    codes = _chronological_codes(dates)
    statistic = _SameDateAUC if same_date else _DateWeightedAUC
    if groups is None:
        blocks = [statistic(y, p, codes)]
    else:
        g = _date_codes(groups)
        if g.size != y.size:
            raise ValueError("block_bootstrap_auc: groups differ in length from y")
        blocks = [statistic(y[g == k], p[g == k], codes[g == k]) for k in np.unique(g)]
    if not any(b.two_class for b in blocks):
        return math.nan, math.nan
    length = max(1, int(block_length))
    if any(b.two_class and b.n_dates < 2 * length for b in blocks):
        log.debug("model_eval.bootstrap_groups_too_short", block_length=length,
                  group_dates=[b.n_dates for b in blocks])
        return math.nan, math.nan

    n_dates = int(codes.max()) + 1
    inflation = 1.0
    if horizon is not None:
        inflation = interval_inflation(horizon, length, n_dates, alpha)
        if not math.isfinite(inflation):
            log.debug("model_eval.bootstrap_too_few_blocks", block_length=length, n_dates=n_dates)
            return math.nan, math.nan

    rng = np.random.default_rng(seed)
    n_boot = int(n_boot)
    if joint:
        drawn = _circular_block_weights(rng, n_dates, length, n_boot)
        weights = [drawn[:, b.date_codes] for b in blocks]
    else:
        weights = [_circular_block_weights(rng, b.n_dates, length, n_boot) if b.two_class else None
                   for b in blocks]
    stats = []
    for r in range(n_boot):
        values = [b.auc(w[r]) if b.two_class else 0.5 for b, w in zip(blocks, weights)]
        if all(math.isfinite(v) for v in values):
            stats.append(float(np.mean(values)))
    if len(stats) < 0.5 * n_boot:
        return math.nan, math.nan
    lo, hi = np.percentile(stats, [50.0 * alpha, 100.0 - 50.0 * alpha])
    if inflation != 1.0:
        point = float(np.mean([b.auc() if b.two_class else 0.5 for b in blocks]))
        lo = max(0.0, point - inflation * max(point - lo, 0.0))
        hi = min(1.0, point + inflation * max(hi - point, 0.0))
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _reliability(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Fixed-width probability bins: mean forecast vs observed up-rate. Empty bins omitted."""
    if p.size == 0:
        return []
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        m = bins == b
        n = int(m.sum())
        if n:
            out.append({"bin": b, "lo": b / n_bins, "hi": (b + 1) / n_bins,
                        "p_mean": float(p[m].mean()), "y_rate": float(y[m].mean()), "n": n})
    return out


def _same_date_spread(rank: np.ndarray, fwd: np.ndarray, codes: np.ndarray) -> float:
    """
    Mean over dates of (mean fwd of that date's top-decile rows by ``rank`` minus
    its bottom-decile rows). Deciles are each date's own quantile cut-offs, so
    tied scores fall on both sides and a constant forecast spreads exactly 0.
    Dates with fewer than two rows, or no finite return on a side, are skipped;
    NaN when no date is left.
    """
    local = pd.factorize(codes)[0]
    frame = pd.DataFrame({"date": local, "rank": rank})
    by_date = frame.groupby("date")["rank"]
    top = rank >= by_date.quantile(0.9).to_numpy()[local]
    bottom = rank <= by_date.quantile(0.1).to_numpy()[local]
    k = int(local.max()) + 1
    finite = np.isfinite(fwd)
    value = np.where(finite, fwd, 0.0)

    def side_mean(mask: np.ndarray) -> np.ndarray:
        count = np.bincount(local, weights=(mask & finite).astype(float), minlength=k)
        total = np.bincount(local, weights=np.where(mask, value, 0.0), minlength=k)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(count > 0, total / np.maximum(count, 1.0), np.nan)

    spread = side_mean(top) - side_mean(bottom)
    usable = (np.bincount(local, minlength=k) >= 2) & np.isfinite(spread)
    return float(spread[usable].mean()) if usable.any() else math.nan


def _empty_metrics(n_dates: int, horizon: int) -> dict:
    nan = math.nan
    out = {"auc": nan, "auc_pooled": nan, "log_loss": nan, "brier": nan, "brier_prior": nan,
           "brier_skill": nan, "acc": nan, "acc_majority": nan, "acc_skill": nan}
    for _, key in HI_CONF_LEVELS:
        out[key] = {"acc": nan, "n": 0}
    out.update({"decile_spread": nan, "reliability": [], "n": 0, "n_dates": n_dates,
                "n_eff": n_dates / horizon if horizon else nan})
    return out


def metrics_block(y, p, fwd_ret, prior_rate, horizon: int, dates, *, p_rank=None) -> dict:
    """
    Skill metrics for one set of forecasts.

    ``p`` is P(up); ``prior_rate`` is the base rate the forecasts must beat (the
    training up-rate), a scalar or one value per row. ``p_rank`` (default ``p``)
    is used for the ranking metrics only - ``auc``, ``auc_pooled`` and
    ``decile_spread`` - so a caller can rank on raw scores and judge
    probabilities after calibration.

    Keys: auc (``same_date_auc``: pairs of rows on one date only; 0.5 without
    such a pair), auc_pooled (every row ranked as one set, across dates;
    biased for forecasts built from the price path, reported, never decided
    on), log_loss, brier, brier_prior, brier_skill (1 - brier / brier_prior),
    acc (p >= 0.5 means up), acc_majority (always calling the class the prior
    favours), acc_skill (acc - acc_majority), hi_conf_10 / hi_conf_15 ({acc, n}
    over |p - 0.5| >= 0.10 / 0.15), decile_spread (per date, mean fwd_ret of
    the top decile of p_rank minus the bottom decile, averaged over dates; NaN
    under 20 rows), reliability (10 fixed-width bins of p), n, n_dates, n_eff
    (= n_dates / horizon: overlapping labels make rows on neighbouring dates
    near-duplicates).
    """
    y = np.asarray(y, dtype=float).ravel()
    p = np.clip(np.asarray(p, dtype=float).ravel(), 0.0, 1.0)
    rank = p if p_rank is None else np.asarray(p_rank, dtype=float).ravel()
    n = y.size
    codes = _date_codes(dates) if n else np.empty(0, dtype=np.int64)
    if not (p.size == rank.size == codes.size == n):
        raise ValueError("metrics_block: y, p, p_rank and dates differ in length")
    n_dates = int(np.unique(codes).size)
    if n == 0:
        return _empty_metrics(n_dates, horizon)

    prior = np.clip(np.broadcast_to(np.asarray(prior_rate, dtype=float), (n,)), 0.0, 1.0)
    up = y == 1.0
    pc = np.clip(p, EPS, 1.0 - EPS)
    brier = float(np.mean((p - y) ** 2))
    brier_prior = float(np.mean((prior - y) ** 2))
    acc = float(np.mean((p >= 0.5) == up))
    acc_majority = float(np.mean((prior >= 0.5) == up))

    out = {
        "auc": same_date_auc(y, rank, codes),
        "auc_pooled": _auc(y, rank),
        "log_loss": float(-np.mean(y * np.log(pc) + (1.0 - y) * np.log1p(-pc))),
        "brier": brier,
        "brier_prior": brier_prior,
        "brier_skill": 1.0 - brier / brier_prior if brier_prior > 0 else math.nan,
        "acc": acc,
        "acc_majority": acc_majority,
        "acc_skill": acc - acc_majority,
    }
    for threshold, key in HI_CONF_LEVELS:
        m = np.abs(p - 0.5) >= threshold - _HI_CONF_TOL
        k = int(m.sum())
        out[key] = {"acc": float(np.mean((p[m] >= 0.5) == up[m])) if k else math.nan, "n": k}

    spread = math.nan
    if fwd_ret is not None and n >= MIN_DECILE_ROWS:
        fwd = np.asarray(fwd_ret, dtype=float).ravel()
        if fwd.size != n:
            raise ValueError("metrics_block: fwd_ret differs in length from y")
        spread = _same_date_spread(rank, fwd, codes)
    out["decile_spread"] = spread
    out["reliability"] = _reliability(p, y)
    out["n"] = int(n)
    out["n_dates"] = n_dates
    out["n_eff"] = n_dates / horizon if horizon else math.nan
    return out


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def _logit(p: np.ndarray) -> np.ndarray:
    pc = np.clip(p, EPS, 1.0 - EPS)
    return np.log(pc) - np.log1p(-pc)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, dtype=float)))


class Calibrator:
    """
    Monotone map from a raw model probability to a calibrated one. Picklable.

    Every fitted method is anchored on ``base_rate``, the training up-rate of
    the outcome it answers for, and reads the model's score as its deviation
    from ``score_base_rate``, the rate the model's raw probabilities are
    centred on, ``d = logit(p) - logit(score_base_rate)``. The two differ only
    for a model fitted on another target than the outcome it is judged on (a
    cross-sectional target's up-rate is about 0.5 whatever the market did);
    ``score_base_rate`` None means ``base_rate``.

    * "platt": ``sigmoid(logit(base_rate) + slope * d)``, a temperature on the
      deviation. ``slope`` 0 answers the base rate, 1 passes the score through.
    * "isotonic": ``base_rate + weight * iso(d)``, ``iso`` fitted to the excess
      up-rate over each row's own base rate and damped by ``weight``.
    * "base_rate": the base rate for every row (no usable evidence).
    * "identity": the raw probability, for callers that asked for no calibration.
    """

    def __init__(self, method: str = "identity", model=None, *, base_rate: float = 0.5,
                 slope: float = 0.0, weight: float = 0.0, score_base_rate: float | None = None):
        self.method = method
        self.model = model
        self.base_rate = float(base_rate)
        self.slope = float(slope)
        self.weight = float(weight)
        self.score_base_rate = None if score_base_rate is None else float(score_base_rate)

    def transform(self, p) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=float).ravel(), 0.0, 1.0)
        if self.method == "identity":
            return p
        anchor = float(_logit(np.array([self.base_rate]))[0])
        # getattr: a calibrator pickled before score_base_rate existed has none.
        score_rate = getattr(self, "score_base_rate", None)
        centre = anchor if score_rate is None else float(_logit(np.array([score_rate]))[0])
        if self.method == "platt":
            return _sigmoid(anchor + self.slope * (_logit(p) - centre))
        if self.method == "isotonic":
            excess = self.model.predict(_logit(p) - centre)
            return np.clip(self.base_rate + self.weight * excess, 0.0, 1.0)
        return np.full(p.size, self.base_rate)

    def __repr__(self) -> str:
        score_rate = getattr(self, "score_base_rate", None)
        scores = "" if score_rate is None else f", score_base_rate={score_rate:.4f}"
        return (f"Calibrator(method={self.method!r}, base_rate={self.base_rate:.4f}{scores}, "
                f"slope={self.slope:.4f}, weight={self.weight:.4f})")


def _shrunk_slope(deviation: np.ndarray, y: np.ndarray, offset: np.ndarray, n_eff: float) -> float:
    """
    Slope of ``y ~ sigmoid(offset + slope * deviation)``: a ridge-penalised,
    evidence-weighted logistic fit with no free intercept.

    The rows' log-likelihood is scaled to count ``n_eff`` observations, and a
    Gaussian prior on the standardised slope is worth CALIBRATION_PRIOR_N_EFF
    observations of no relationship. Thin evidence therefore lands near 0 (the
    base rate) and abundant evidence near the maximum-likelihood slope.
    """
    scale = math.sqrt(float(np.mean(deviation ** 2)))
    if not math.isfinite(scale) or scale < 1e-9 or n_eff <= 0:
        return 0.0
    u = deviation / scale
    w = float(n_eff) / u.size
    rate = float(np.mean(_sigmoid(offset)))
    ridge = CALIBRATION_PRIOR_N_EFF * rate * (1.0 - rate)
    b = 0.0
    for _ in range(50):
        q = _sigmoid(offset + b * u)
        grad = w * float(np.dot(y - q, u)) - ridge * b
        hess = w * float(np.dot(q * (1.0 - q), u * u)) + ridge
        step = max(-1.0, min(1.0, grad / hess))
        b += step
        if abs(step) < 1e-10:
            break
    return b / scale


def fit_calibrator(p_oof, y_oof, method: str = "platt", *, base_rate: float | None = None,
                   oof_base_rate=None, n_eff: float | None = None,
                   score_base_rate: float | None = None, oof_score_rate=None) -> Calibrator:
    """
    Fit a calibrator on out-of-fold predictions.

    ``base_rate`` is the anchor the returned calibrator transforms around: the
    training up-rate of the model whose scores it will see (default: the mean
    of ``y_oof``). ``oof_base_rate`` (scalar or per row, default
    ``base_rate``) is the up-rate each out-of-fold score was anchored on, i.e.
    the training up-rate of the fold model that produced it, so scores from
    models trained on different windows are read on one scale.

    ``score_base_rate`` and ``oof_score_rate`` (scalar or per row) are for
    models fitted on another target than ``y_oof``: the rate the raw scores are
    centred on, i.e. the up-rate of that training target, for the model the
    calibrator will see and for each fold model behind the out-of-fold scores.
    Deviations are then read against the score rates while outcomes are still
    anchored on the base rates. Omitted, they are the base rates, which is
    exact whenever the model was trained on the outcome itself; pass both or
    neither.

    ``n_eff`` is how many independent observations the rows hold: distinct
    dates / horizon for overlapping labels (default: the row count). The fitted
    slope (Platt) or weight (isotonic, ``n_eff / (n_eff + prior)``) shrinks
    towards the base rate with CALIBRATION_PRIOR_N_EFF observations of prior,
    so a calibrator fitted on one short regime cannot turn a no-skill model's
    noise into confident forecasts. Under 50 rows, or with one class, the
    calibrator answers the base rate; with no rows and no ``base_rate`` it is
    the identity.
    """
    if method not in CALIBRATION_METHODS:
        raise ValueError(f"calibration method must be one of {CALIBRATION_METHODS}, got {method!r}")
    p = np.asarray(p_oof, dtype=float).ravel()
    y = np.asarray(y_oof, dtype=float).ravel()
    if p.size != y.size:
        raise ValueError("fit_calibrator: p_oof and y_oof differ in length")
    row_rate = (np.full(p.size, math.nan) if oof_base_rate is None
                else np.broadcast_to(np.asarray(oof_base_rate, dtype=float), p.shape).astype(float))
    score_rows = (None if oof_score_rate is None
                  else np.broadcast_to(np.asarray(oof_score_rate, dtype=float), p.shape).astype(float))
    ok = np.isfinite(p) & np.isfinite(y)
    p, y, row_rate = p[ok], y[ok], row_rate[ok]
    if score_rows is not None:
        score_rows = score_rows[ok]
    if method == "identity":
        return Calibrator("identity")
    if base_rate is None:
        if p.size == 0:
            return Calibrator("identity")
        base_rate = float(y.mean())
    anchor = float(np.clip(base_rate, EPS, 1.0 - EPS))
    if p.size < MIN_CALIBRATION_ROWS or np.unique(y).size < 2:
        return Calibrator("base_rate", base_rate=anchor)
    row_rate = np.where(np.isfinite(row_rate), np.clip(row_rate, EPS, 1.0 - EPS), anchor)
    score_anchor = None if score_base_rate is None else float(np.clip(score_base_rate, EPS, 1.0 - EPS))
    evidence = float(p.size if n_eff is None else max(float(n_eff), 0.0))
    offset = _logit(row_rate)
    if score_rows is None:
        deviation = _logit(p) - offset
    else:
        # A row without a score rate is read against its base rate, as by default.
        centre = np.where(np.isfinite(score_rows), np.clip(score_rows, EPS, 1.0 - EPS), row_rate)
        deviation = _logit(p) - _logit(centre)
    if method == "platt":
        slope = _shrunk_slope(deviation, y, offset, evidence)
        if CALIBRATION_NONNEGATIVE_SLOPE:
            slope = max(slope, 0.0)
        return Calibrator("platt", base_rate=anchor, slope=slope, score_base_rate=score_anchor)
    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(deviation, y - row_rate)
    return Calibrator("isotonic", model, base_rate=anchor,
                      weight=evidence / (evidence + CALIBRATION_PRIOR_N_EFF),
                      score_base_rate=score_anchor)


# ---------------------------------------------------------------------------
# Estimators
#
# Factories are functools.partial objects over module-level builders, so they
# pickle and a fitted PerTickerClassifier (which keeps its factory) does too.
# ---------------------------------------------------------------------------

_HGB_ACCEPTED_PARAMS = frozenset(
    inspect.signature(HistGradientBoostingClassifier.__init__).parameters) - {"self"}
# Parameters that exist only in newer scikit-learn; dropped rather than failing
# on an older install (max_features arrived in 1.4). Anything else unknown is a
# typo and still raises in the constructor.
_HGB_VERSION_DEPENDENT = ("max_features",)
_IMPUTER_KW = ({"keep_empty_features": True}
               if "keep_empty_features" in inspect.signature(SimpleImputer.__init__).parameters
               else {})


def _hgb_kwargs(params: Optional[dict], categorical_idx) -> dict:
    kwargs = dict(params or {})
    for name in _HGB_VERSION_DEPENDENT:
        if name in kwargs and name not in _HGB_ACCEPTED_PARAMS:
            kwargs.pop(name)
            log.debug("model_eval.hgb_param_unsupported", param=name,
                      sklearn_version=sklearn.__version__)
    if categorical_idx is not None or "categorical_features" not in kwargs:
        # An empty list becomes None: "no categorical columns" on every version.
        kwargs["categorical_features"] = [int(i) for i in categorical_idx] if categorical_idx else None
    return kwargs


def _make_hgb(params: dict, categorical_idx) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(**_hgb_kwargs(params, categorical_idx))


def hgb_factory(params: dict, categorical_idx: list[int] | None = None) -> Callable[[], HistGradientBoostingClassifier]:
    """
    Factory for ``HistGradientBoostingClassifier(**params, categorical_features=...)``.

    NaN-safe as is. ``categorical_idx`` are column indices holding small
    non-negative integer codes (sector_id, dow). ``max_features`` is dropped on
    a scikit-learn that does not accept it.
    """
    cats = None if categorical_idx is None else tuple(int(i) for i in categorical_idx)
    return functools.partial(_make_hgb, dict(params or {}), cats)


def _make_logreg(C: float) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", **_IMPUTER_KW)),
        ("scale", StandardScaler()),
        ("logreg", LogisticRegression(C=C, max_iter=500)),
    ])


def baseline_logreg_factory(C: float = 0.1) -> Callable[[], Pipeline]:
    """Factory for median-impute -> standardise -> L2 logistic regression."""
    return functools.partial(_make_logreg, float(C))


def _make_gbm(params: dict) -> Pipeline:
    # GradientBoostingClassifier rejects NaN, so it gets an imputer in front.
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", **_IMPUTER_KW)),
        ("gbm", GradientBoostingClassifier(**params)),
    ])


def gbm_factory(params: dict) -> Callable[[], Pipeline]:
    """Factory for median-impute -> GradientBoostingClassifier(**params) (the v3 model)."""
    return functools.partial(_make_gbm, dict(params or {}))


def _fit_estimator(estimator, X, y, sample_weight):
    """Fit, passing weights only when there are any (and to a Pipeline's last step)."""
    if sample_weight is None:
        return estimator.fit(X, y)
    if isinstance(estimator, Pipeline):
        return estimator.fit(X, y, **{f"{estimator.steps[-1][0]}__sample_weight": sample_weight})
    return estimator.fit(X, y, sample_weight=sample_weight)


def _proba_up(estimator, X) -> np.ndarray:
    """P(y == 1) from predict_proba, whatever column order the estimator's classes_ have."""
    proba = estimator.predict_proba(X)
    classes = list(getattr(estimator, "classes_", [0, 1]))
    if 1 in classes:
        return np.asarray(proba[:, classes.index(1)], dtype=float)
    return np.zeros(len(X), dtype=float)


class PerTickerClassifier(ClassifierMixin, BaseEstimator):
    """
    One sub-model per ticker, routed by an integer ticker code held in column
    ``ticker_col`` of X (that column is removed before the sub-model sees X).

    Exists to evaluate the old per-ticker approach (config ``run0``) with the
    same folds and metrics as the pooled model. A ticker with fewer than
    ``min_rows`` training rows, or a single class, predicts its own up-rate; a
    ticker unseen in training predicts the pooled up-rate.
    """

    def __init__(self, base_factory=None, ticker_col: int = -1, min_rows: int = 60):
        self.base_factory = base_factory
        self.ticker_col = ticker_col
        self.min_rows = min_rows

    def _split(self, X):
        X = np.asarray(X, dtype=float)
        col = self.ticker_col % X.shape[1]
        return X[:, col], np.delete(X, col, axis=1)

    def fit(self, X, y, sample_weight=None):
        codes, features = self._split(X)
        y = np.asarray(y).astype(int)
        w = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = int(np.asarray(X).shape[1])
        self.prior_ = float(np.average(y, weights=w)) if y.size else 0.5
        self.models_ = {}
        for code in np.unique(codes[np.isfinite(codes)]):
            m = codes == code
            y_k = y[m]
            if int(m.sum()) < self.min_rows or np.unique(y_k).size < 2:
                self.models_[float(code)] = float(y_k.mean())
                continue
            model = self.base_factory()
            _fit_estimator(model, features[m], y_k, None if w is None else w[m])
            self.models_[float(code)] = model
        return self

    def predict_proba(self, X):
        codes, features = self._split(X)
        p = np.full(codes.size, self.prior_, dtype=float)
        for code in np.unique(codes[np.isfinite(codes)]):
            model = self.models_.get(float(code))
            if model is None:
                continue
            m = codes == code
            p[m] = model if isinstance(model, float) else _proba_up(model, features[m])
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def per_ticker_factory(base_factory: Callable, ticker_col: int, *, min_rows: int = 60) -> Callable[[], PerTickerClassifier]:
    """
    Factory for ``PerTickerClassifier``. The caller appends the ticker code
    column, e.g. ``X_run0 = np.column_stack([X, pd.factorize(tickers)[0]])`` and
    ``ticker_col = X.shape[1]``; permutation importance then reports that column
    too (name it "ticker" when mapping names).
    """
    return functools.partial(PerTickerClassifier, base_factory=base_factory,
                             ticker_col=int(ticker_col), min_rows=int(min_rows))


# ---------------------------------------------------------------------------
# Aggregation and results
# ---------------------------------------------------------------------------

# Scalar per-fold metrics aggregated across folds as <name>_mean / <name>_std.
AGGREGATED_METRICS = (
    "auc", "auc_pooled", "log_loss", "brier", "brier_prior", "brier_skill", "acc", "acc_majority",
    "acc_skill", "hi_conf_10_acc", "hi_conf_10_n", "hi_conf_15_acc", "hi_conf_15_n",
    "decile_spread", "n", "n_dates", "n_eff", "prior_rate",
)


def _fold_scalar(row: dict, key: str) -> float:
    for _, hi_conf in HI_CONF_LEVELS:
        if key.startswith(hi_conf + "_"):
            return (row.get(hi_conf) or {}).get(key[len(hi_conf) + 1:], math.nan)
    return row.get(key, math.nan)


def _mean_std(values) -> tuple[float, float]:
    v = np.asarray([math.nan if x is None else x for x in values], dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return math.nan, math.nan
    return float(v.mean()), (float(v.std(ddof=1)) if v.size > 1 else math.nan)


def _aggregate(rows: list[dict], y, p_raw, p_cal, day, fold_pos, horizon: int, *,
               seed: int, n_boot: int) -> dict:
    """
    Mean/std of the fold metrics plus pooled figures over the same folds' OOF rows.

    ``auc_ci_low``/``auc_ci_high`` bracket the fold-averaged same-date AUC
    (``auc_mean``): blocks of dates are resampled and the per-fold AUCs
    averaged, so fold models whose raw probabilities sit at different levels
    never meet in one ranking, and the percentile interval is widened by
    ``ci_inflation`` (``interval_inflation``) for what blocks miss.
    """
    if not rows:
        return {}
    out: dict = {"n_folds": len(rows), "folds": [r["fold"] for r in rows]}
    for key in AGGREGATED_METRICS:
        out[f"{key}_mean"], out[f"{key}_std"] = _mean_std([_fold_scalar(r, key) for r in rows])
    for _, hi_conf in HI_CONF_LEVELS:
        n_total = int(sum((r.get(hi_conf) or {}).get("n", 0) for r in rows))
        hits = sum(r[hi_conf]["acc"] * r[hi_conf]["n"] for r in rows
                   if r.get(hi_conf) and r[hi_conf]["n"])
        out[f"{hi_conf}_n_total"] = n_total
        out[f"{hi_conf}_acc_pooled"] = hits / n_total if n_total else math.nan
    n_dates = int(np.unique(day).size)
    out["horizon"] = int(horizon)
    out["n_total"] = int(y.size)
    out["n_dates_total"] = n_dates
    out["n_eff_total"] = n_dates / horizon
    out["block_length"] = block_length_for(horizon)
    out["ci_inflation"] = interval_inflation(horizon, out["block_length"], n_dates)
    out["auc_ci_low"], out["auc_ci_high"] = block_bootstrap_auc(
        y, p_raw, day, n_boot=n_boot, seed=seed, groups=fold_pos,
        block_length=out["block_length"], joint=BOOTSTRAP_JOINT, horizon=horizon)
    out["reliability"] = _reliability(np.clip(p_cal, 0.0, 1.0), y)
    return out


def _jsonable(obj):
    """numpy -> python scalars, NaN/inf -> None, timestamps -> ISO strings, recursively."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(obj) else None
    if isinstance(obj, np.datetime64):
        return None if np.isnat(obj) else str(obj)
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return obj


@dataclass
class EvalResult:
    """
    Output of ``evaluate``.

    ``folds``: one metrics dict per fold (see ``metrics_block``) plus fold
    bookkeeping. ``overall`` / ``confirm`` / ``selection``: aggregates over all
    folds / the last ``confirm_folds`` folds (what ``ship_decision`` reads) /
    the folds before those (what hyperparameter selection may read).
    ``oof``: numpy arrays over every test row - ``idx`` (row position), ``p_raw``,
    ``p_cal`` (leakage-free calibration), ``y``, ``fold`` (position in the fold
    list), ``prior`` (that fold's training up-rate), ``score_rate`` (the up-rate
    of the fold model's training target, which its raw probabilities are
    centred on: ``prior`` unless ``evaluate`` was given a different ``y_fit``),
    ``label_day`` (datetime64[D]; NaT when ``evaluate`` was not given
    label_dates). ``importance``: permutation importance on the last fold, one
    entry per column in column order.
    """

    horizon: int
    folds: list[dict]
    overall: dict
    confirm: dict
    per_ticker: list[dict]
    oof: dict
    importance: list[dict]
    calibrator_method: str
    n_rows: int
    n_dates: int
    n_tickers: int
    selection: dict = field(default_factory=dict)
    confirm_folds: int = 0

    def to_dict(self, include_oof: bool = False) -> dict:
        """JSON-ready (``json.dumps(..., allow_nan=False)`` works). OOF arrays only on request."""
        out = {
            "horizon": self.horizon,
            "calibrator_method": self.calibrator_method,
            "n_rows": self.n_rows,
            "n_dates": self.n_dates,
            "n_tickers": self.n_tickers,
            "n_folds": len(self.folds),
            "confirm_folds": self.confirm_folds,
            "folds": self.folds,
            "overall": self.overall,
            "confirm": self.confirm,
            "selection": self.selection,
            "per_ticker": self.per_ticker,
            "importance": self.importance,
        }
        if include_oof:
            out["oof"] = self.oof
        return _jsonable(out)


def _as_vector(values, n: int, name: str) -> np.ndarray:
    if isinstance(values, (pd.Series, pd.Index)):
        arr = values.to_numpy(dtype=float, na_value=np.nan)
    else:
        arr = np.asarray(values, dtype=float)
    arr = arr.ravel()
    if arr.size != n:
        raise ValueError(f"{name} has {arr.size} rows, expected {n}")
    return arr


def _confirm_count(confirm_folds: int, n_folds: int) -> int:
    """Confirm folds, clamped so at least one fold is left for selection (when there are two)."""
    if confirm_folds < 1:
        raise ValueError(f"confirm_folds must be >= 1, got {confirm_folds}")
    allowed = max(1, n_folds - 1)
    if confirm_folds > allowed:
        log.info("model_eval.confirm_folds_clamped", requested=confirm_folds,
                 n_folds=n_folds, confirm_folds=allowed)
        return allowed
    return confirm_folds


def _fold_rows_union(folds: list[Fold]) -> np.ndarray:
    return np.unique(np.concatenate(
        [np.asarray(f.train_idx, dtype=np.int64) for f in folds]
        + [np.asarray(f.test_idx, dtype=np.int64) for f in folds]))


def _per_ticker(tickers, y, p_raw, p_cal, prior) -> list[dict]:
    """
    One ticker's out-of-fold rows at a time. Its ``auc`` ranks that ticker's
    dates against each other, which is the across-dates ranking the ship rule
    avoids (see the module docstring): a diagnostic, never a decision.
    """
    codes, names = pd.factorize(tickers)
    out = []
    for code, name in enumerate(names):
        m = codes == code
        y_k = y[m]
        brier = float(np.mean((p_cal[m] - y_k) ** 2))
        brier_prior = float(np.mean((prior[m] - y_k) ** 2))
        out.append({
            "ticker": str(name),
            "n": int(m.sum()),
            "up_rate": float(y_k.mean()),
            "auc": _auc(y_k, p_raw[m]),
            "acc": float(np.mean((p_cal[m] >= 0.5) == (y_k == 1))),
            "brier_skill": 1.0 - brier / brier_prior if brier_prior > 0 else math.nan,
        })
    out.sort(key=lambda r: r["ticker"])
    return out


def _permutation_importance(model, X_test, y_test, dates_test, random_state: int,
                            n_threads: int) -> list[dict]:
    """Same-date AUC lost when each column is shuffled, on the last fold's test rows."""
    if model is None or np.unique(y_test).size < 2:
        return []
    codes = _date_codes(dates_test)

    def scorer(estimator, X, y):
        # permutation_importance shuffles columns, never rows, so row i keeps date codes[i]
        return same_date_auc(y, _proba_up(estimator, X), codes)

    try:
        with threadpool_limits(limits=n_threads):
            result = permutation_importance(model, X_test, y_test, scoring=scorer,
                                            n_repeats=3, random_state=random_state)
    except ValueError as exc:
        log.debug("model_eval.importance_failed", error=str(exc))
        return []
    return [{"feature": j, "importance": float(result.importances_mean[j]),
             "std": float(result.importances_std[j])} for j in range(X_test.shape[1])]


def _aggregate_positions(rows, oof, oof_day, positions, horizon, seed, n_boot) -> dict:
    positions = list(positions)
    if not positions:
        return {}
    m = np.isin(oof["fold"], positions)
    return _aggregate([rows[i] for i in positions], oof["y"][m].astype(float),
                      oof["p_raw"][m], oof["p_cal"][m], oof_day[m], oof["fold"][m],
                      horizon, seed=seed, n_boot=n_boot)


def evaluate(model_factory, X, y, fwd_ret, dates, tickers, folds: list[Fold], *,
             sample_weight=None, calibration: str = "platt", confirm_folds: int = 2,
             importance: bool = True, random_state: int = 0, train_mask=None,
             label_dates=None, horizon: int | None = None,
             n_threads: int = DEFAULT_THREADS, n_boot: int = 200, y_fit=None) -> EvalResult:
    """
    Fit ``model_factory()`` on each fold's training rows and score its test rows.

    ``X`` may hold NaN (HistGradientBoosting handles it; other factories must
    impute). ``y`` is 0/1 (NaN allowed only on rows no fold uses).

    ``y_fit`` (optional, default ``y``) is the training target when it is not
    the outcome being judged, e.g. "above the date's median return" for a
    model whose forecasts are read as P(close higher). It is used only to fit:
    every metric, ``prior_rate``, the calibration evidence and permutation
    importance read ``y``. Its NaN rows must be dropped by ``train_mask``. A fold
    model's raw probabilities are centred on its target's up-rate, the mean of
    ``y_fit`` over the fold's training rows (``oof["score_rate"]``), so the
    calibrator reads them against that rate and answers on ``y``'s scale. A
    ``y_fit`` equal to ``y`` is ``y``. Per fold:

    * AUC and decile spread use the raw probabilities and compare rows on the
      same date only (``same_date_auc``; the across-dates AUC is kept as
      ``auc_pooled``); log loss, Brier,
      accuracy, hi-conf accuracy and reliability use probabilities calibrated
      by a calibrator fitted on the OOF predictions of strictly earlier folds
      and anchored on this fold's ``prior_rate`` (``fit_calibrator``). The
      first fold has no earlier evidence and answers its base rate.
    * ``label_dates`` (pass the same array the folds were built from) purges
      that calibration set with the training rule: only earlier-fold rows whose
      ``label_date < test_start - embargo`` are used. The last sessions of the
      previous test block are labelled inside this block, so their outcomes are
      not yet known at ``test_start``. Without ``label_dates`` every earlier row
      is used (logged as ``model_eval.calibration_unpurged``). ``n_calib`` in the
      fold row is the size of that set and ``calib_n_eff`` its distinct dates /
      horizon, the evidence the calibrator's shrinkage is weighed against; below
      50 rows the calibrator answers the base rate.
    * ``prior_rate`` = mean(y) over the fold's training rows.
    * ``sample_weight`` (optional) is rescaled to mean 1 within each fold's
      training rows, so regularisation means the same under every scheme.
    * ``train_mask`` (optional, True = keep) drops rows from training only
      (the dead-band option); test rows and ``prior_rate`` ignore it.
    * A training set with a single class predicts its up-rate instead of fitting.

    ``horizon`` defaults to ``folds[0].horizon``. ``confirm_folds`` is clamped to
    ``len(folds) - 1`` (logged). Fits run under a ``n_threads`` thread cap.
    """
    if not folds:
        raise ValueError("evaluate: no folds")
    if calibration not in CALIBRATION_METHODS:
        raise ValueError(f"calibration must be one of {CALIBRATION_METHODS}, got {calibration!r}")
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"evaluate: X must be 2-D, got shape {X.shape}")
    n = X.shape[0]
    y = _as_vector(y, n, "y")
    fit_labels = None if y_fit is None else _as_vector(y_fit, n, "y_fit")
    if fit_labels is not None and np.array_equal(fit_labels, y, equal_nan=True):
        fit_labels = None
    fwd = None if fwd_ret is None else _as_vector(fwd_ret, n, "fwd_ret")
    day = _day_numbers(dates)
    if day.size != n:
        raise ValueError(f"dates has {day.size} rows, expected {n}")
    tick = None
    if tickers is not None:
        tick = tickers.to_numpy() if isinstance(tickers, (pd.Series, pd.Index)) else np.asarray(tickers)
        if tick.size != n:
            raise ValueError(f"tickers has {tick.size} rows, expected {n}")
    weights = None if sample_weight is None else _as_vector(sample_weight, n, "sample_weight")
    keep = None
    if train_mask is not None:
        keep = np.asarray(train_mask, dtype=bool).ravel()
        if keep.size != n:
            raise ValueError(f"train_mask has {keep.size} rows, expected {n}")
    label_day = None
    if label_dates is not None:
        label_day = _day_numbers(label_dates)
        if label_day.size != n:
            raise ValueError(f"label_dates has {label_day.size} rows, expected {n}")
    elif calibration != "identity" and len(folds) > 1:
        log.debug("model_eval.calibration_unpurged", n_folds=len(folds))
    h = int(horizon if horizon is not None else folds[0].horizon)
    if h < 1:
        raise ValueError("evaluate: horizon unknown; pass horizon= or folds from walkforward_folds")
    k_confirm = _confirm_count(int(confirm_folds), len(folds))

    rows: list[dict] = []
    parts: dict[str, list] = {k: [] for k in ("idx", "p_raw", "p_cal", "y", "fold", "prior",
                                              "score_rate")}
    last_model = None
    for pos, fold in enumerate(folds):
        train_all = np.asarray(fold.train_idx, dtype=np.int64)
        test = np.asarray(fold.test_idx, dtype=np.int64)
        if train_all.size == 0 or test.size == 0:
            raise ValueError(f"evaluate: fold {fold.index} has no training or no test rows")
        if not (np.isfinite(y[train_all]).all() and np.isfinite(y[test]).all()):
            raise ValueError(f"evaluate: fold {fold.index} includes unlabeled rows (NaN y)")
        prior = float(y[train_all].mean())
        train = train_all if keep is None else train_all[keep[train_all]]
        if fit_labels is None:
            score_rate = prior
            y_train = y[train].astype(int)
        else:
            if not np.isfinite(fit_labels[train]).all():
                raise ValueError(f"evaluate: fold {fold.index} trains on rows without a y_fit "
                                 "label (NaN); drop them with train_mask")
            target = fit_labels[train_all]
            target = target[np.isfinite(target)]
            score_rate = float(target.mean()) if target.size else prior
            y_train = fit_labels[train].astype(int)

        started = time.perf_counter()
        model = None
        if train.size == 0 or np.unique(y_train).size < 2:
            p_raw = np.full(test.size, float(y_train.mean()) if train.size else score_rate)
        else:
            w = None
            if weights is not None:
                w = weights[train]
                if not np.isfinite(w).all() or w.min() < 0 or w.sum() <= 0:
                    raise ValueError(f"evaluate: fold {fold.index} has invalid sample weights")
                w = w / w.mean()
            model = model_factory()
            # Thread cap: the phone's SoC is shared with the API and worker.
            with threadpool_limits(limits=n_threads):
                _fit_estimator(model, X[train], y_train, w)
                p_raw = _proba_up(model, X[test])
        fit_seconds = time.perf_counter() - started

        n_calib, calib_n_eff = 0, 0.0
        if calibration == "identity":
            calibrator = Calibrator("identity")
        elif pos == 0:
            calibrator = Calibrator("base_rate", base_rate=prior)
        else:
            prev_idx = np.concatenate(parts["idx"])
            p_prev = np.concatenate(parts["p_raw"])
            y_prev = np.concatenate(parts["y"])
            prior_prev = np.concatenate(parts["prior"])
            score_prev = np.concatenate(parts["score_rate"])
            if label_day is not None:
                # The training purge, applied to the calibration set: an earlier
                # fold's row may calibrate this fold only if its outcome was
                # known before test_start - embargo.
                cutoff = int(_day_numbers([fold.test_start])[0]) - int(fold.embargo_days)
                prev_label_day = label_day[prev_idx]
                known = (prev_label_day != _NAT_DAY) & (prev_label_day < cutoff)
                prev_idx, p_prev, y_prev, prior_prev, score_prev = (
                    prev_idx[known], p_prev[known], y_prev[known], prior_prev[known],
                    score_prev[known])
            n_calib = int(p_prev.size)
            calib_n_eff = int(np.unique(day[prev_idx]).size) / h
            scores = ({} if fit_labels is None
                      else {"score_base_rate": score_rate, "oof_score_rate": score_prev})
            calibrator = fit_calibrator(p_prev, y_prev, calibration, base_rate=prior,
                                        oof_base_rate=prior_prev, n_eff=calib_n_eff, **scores)
        p_cal = calibrator.transform(p_raw)
        metrics = metrics_block(y[test], p_cal, None if fwd is None else fwd[test], prior, h,
                                day[test], p_rank=p_raw)
        rows.append({
            "fold": int(fold.index),
            "test_start": str(pd.Timestamp(fold.test_start).date()),
            "test_end": str(pd.Timestamp(fold.test_end).date()),
            "train_start": _iso(day[train_all].min()),
            "train_end": _iso(day[train_all].max()),
            "embargo_days": int(fold.embargo_days),
            "n_train": int(train.size),
            "n_test": int(test.size),
            "prior_rate": prior,
            "calibrator": calibrator.method,
            "n_calib": n_calib,
            "calib_n_eff": calib_n_eff,
            "calib_slope": calibrator.slope if calibrator.method == "platt" else None,
            "fit_seconds": round(fit_seconds, 3),
            **metrics,
        })
        parts["idx"].append(test)
        parts["p_raw"].append(np.asarray(p_raw, dtype=float))
        parts["p_cal"].append(p_cal)
        parts["y"].append(y[test])
        parts["fold"].append(np.full(test.size, pos))
        parts["prior"].append(np.full(test.size, prior))
        parts["score_rate"].append(np.full(test.size, score_rate))
        if pos == len(folds) - 1:
            last_model = model
        log.debug("model_eval.fold", horizon=h, fold=int(fold.index), n_train=int(train.size),
                  n_test=int(test.size), auc=round(metrics["auc"], 4),
                  brier_skill=round(metrics["brier_skill"], 4), fit_seconds=round(fit_seconds, 2))

    oof = {
        "idx": np.concatenate(parts["idx"]).astype(np.int64),
        "p_raw": np.concatenate(parts["p_raw"]),
        "p_cal": np.concatenate(parts["p_cal"]),
        "y": np.concatenate(parts["y"]).astype(np.int8),
        "fold": np.concatenate(parts["fold"]).astype(np.int64),
        "prior": np.concatenate(parts["prior"]),
        "score_rate": np.concatenate(parts["score_rate"]),
    }
    oof["label_day"] = (label_day[oof["idx"]].astype("datetime64[D]") if label_day is not None
                        else np.full(oof["idx"].size, np.datetime64("NaT"), dtype="datetime64[D]"))
    oof_day = day[oof["idx"]]
    n_folds = len(folds)
    overall = _aggregate_positions(rows, oof, oof_day, range(n_folds), h, random_state, n_boot)
    confirm = _aggregate_positions(rows, oof, oof_day, range(n_folds - k_confirm, n_folds), h,
                                   random_state, n_boot)
    selection = _aggregate_positions(rows, oof, oof_day, range(n_folds - k_confirm), h,
                                     random_state, n_boot)
    per_ticker = ([] if tick is None else
                  _per_ticker(tick[oof["idx"]], oof["y"].astype(float), oof["p_raw"],
                              oof["p_cal"], oof["prior"]))
    last_test = np.asarray(folds[-1].test_idx, dtype=np.int64)
    imp = (_permutation_importance(last_model, X[last_test], y[last_test].astype(int),
                                   day[last_test], random_state, n_threads) if importance else [])

    used = _fold_rows_union(folds)
    result = EvalResult(
        horizon=h, folds=rows, overall=overall, confirm=confirm, per_ticker=per_ticker,
        oof=oof, importance=imp, calibrator_method=calibration, n_rows=int(used.size),
        n_dates=int(np.unique(day[used]).size),
        # factorize skips missing tickers, as _per_ticker does
        n_tickers=0 if tick is None else int(len(pd.factorize(tick[used])[1])),
        selection=selection, confirm_folds=k_confirm,
    )
    log.debug("model_eval.evaluated", horizon=h, n_folds=n_folds, n_rows=result.n_rows,
              auc_mean=round(overall["auc_mean"], 4), confirm_auc=round(confirm["auc_mean"], 4))
    return result


# ---------------------------------------------------------------------------
# Baselines and controls
# ---------------------------------------------------------------------------

def _baseline(name: str, predict_fold: Callable, y, fwd_ret, dates, folds: list[Fold],
              horizon: int, *, confirm_folds: int, seed: int, n_boot: int) -> dict:
    """Score a fit-free forecast rule on the same folds with the same metrics."""
    if not folds:
        raise ValueError(f"baseline_{name}: no folds")
    y = _as_vector(y, len(y), "y")
    n = y.size
    fwd = None if fwd_ret is None else _as_vector(fwd_ret, n, "fwd_ret")
    day = _day_numbers(dates)
    if day.size != n:
        raise ValueError(f"dates has {day.size} rows, expected {n}")
    k_confirm = _confirm_count(int(confirm_folds), len(folds))
    rows: list[dict] = []
    parts: dict[str, list] = {k: [] for k in ("idx", "p", "y", "fold")}
    for pos, fold in enumerate(folds):
        train = np.asarray(fold.train_idx, dtype=np.int64)
        test = np.asarray(fold.test_idx, dtype=np.int64)
        prior = float(y[train].mean())
        p = np.asarray(predict_fold(test, prior), dtype=float)
        metrics = metrics_block(y[test], p, None if fwd is None else fwd[test], prior, horizon,
                                day[test])
        rows.append({"fold": int(fold.index),
                     "test_start": str(pd.Timestamp(fold.test_start).date()),
                     "test_end": str(pd.Timestamp(fold.test_end).date()),
                     "n_train": int(train.size), "n_test": int(test.size),
                     "prior_rate": prior, **metrics})
        parts["idx"].append(test)
        parts["p"].append(p)
        parts["y"].append(y[test])
        parts["fold"].append(np.full(test.size, pos))
    idx = np.concatenate(parts["idx"])
    oof = {"p_raw": np.concatenate(parts["p"]), "y": np.concatenate(parts["y"]),
           "fold": np.concatenate(parts["fold"]).astype(np.int64)}
    oof["p_cal"] = oof["p_raw"]
    n_folds = len(folds)
    blocks = {
        label: _aggregate_positions(rows, oof, day[idx], positions, horizon, seed, n_boot)
        for label, positions in (("overall", range(n_folds)),
                                 ("confirm", range(n_folds - k_confirm, n_folds)),
                                 ("selection", range(n_folds - k_confirm)))
    }
    return _jsonable({"name": name, "horizon": horizon, "n_folds": n_folds,
                      "confirm_folds": k_confirm, "folds": rows, **blocks})


def baseline_prior(y, fwd_ret, dates, folds: list[Fold], horizon: int, *,
                   confirm_folds: int = 2, seed: int = 0, n_boot: int = 200) -> dict:
    """
    Forecast the training up-rate for every test row.

    Returns a JSON-ready dict ``{name, horizon, n_folds, confirm_folds, folds,
    overall, confirm, selection}`` whose blocks carry the same keys as
    ``EvalResult``'s. By construction brier_skill = acc_skill = 0 and AUC = 0.5
    per fold: the bar every model must clear.
    """
    return _baseline("prior", lambda test, prior: np.full(test.size, prior), y, fwd_ret, dates,
                     folds, horizon, confirm_folds=confirm_folds, seed=seed, n_boot=n_boot)


def baseline_momentum(X, y, fwd_ret, dates, folds: list[Fold], horizon: int, feature_idx: int, *,
                      confirm_folds: int = 2, seed: int = 0, n_boot: int = 200) -> dict:
    """
    P(up) = 0.55 when ``X[:, feature_idx] > 0``, 0.45 when <= 0, 0.5 when NaN.

    Same return shape as ``baseline_prior``. A trend rule with no fitting: a
    model that cannot beat it has learned nothing a sign test would not.
    """
    column = np.asarray(X, dtype=float)[:, int(feature_idx)]

    def predict(test, prior):
        v = column[test]
        return np.where(np.isnan(v), 0.5, np.where(v > 0, 0.55, 0.45))

    return _baseline("momentum", predict, y, fwd_ret, dates, folds, horizon,
                     confirm_folds=confirm_folds, seed=seed, n_boot=n_boot)


def _as_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def ship_reasons(confirm: dict, *, min_auc: float = MIN_SHIP_AUC,
                 min_n_eff: float = MIN_CONFIRM_N_EFF) -> list[str]:
    """
    Every ship condition the confirm block fails, in plain words; empty means ship.

    The conditions, in the order listed: n_eff_total >= ``min_n_eff`` (enough
    independent label windows to measure anything), auc_mean >= ``min_auc``,
    auc_ci_low > 0.50, brier_skill_mean > 0, decile_spread_mean > 0. The AUC
    and the spread are the same-date figures ``metrics_block`` reports. A
    value that is missing, None or NaN fails its condition.
    """
    confirm = confirm or {}
    reasons = []
    n_eff = _as_float(confirm.get("n_eff_total"))
    horizon = confirm.get("horizon")
    windows = f"{int(horizon)}-session" if isinstance(horizon, (int, np.integer)) else "label"
    if not math.isfinite(n_eff):
        reasons.append("n_eff not reported: the evidence on the confirm folds cannot be counted")
    elif n_eff < min_n_eff:
        reasons.append(f"n_eff {n_eff:.1f} < {min_n_eff:g}: too few independent {windows} "
                       "windows to measure")
    auc = _as_float(confirm.get("auc_mean"))
    if not math.isfinite(auc):
        reasons.append("AUC not available")
    elif auc < min_auc:
        reasons.append(f"AUC {auc:.3f} < {min_auc:.2f}")
    ci_low = _as_float(confirm.get("auc_ci_low"))
    if not math.isfinite(ci_low):
        reasons.append("AUC CI not available (the confirm folds hold too few bootstrap blocks)")
    elif ci_low <= 0.50:
        reasons.append(f"AUC CI low {ci_low:.3f} <= 0.50")
    brier_skill = _as_float(confirm.get("brier_skill_mean"))
    if not math.isfinite(brier_skill):
        reasons.append("Brier skill not available")
    elif brier_skill <= 0:
        reasons.append(f"Brier skill {brier_skill:+.4f} <= 0")
    spread = _as_float(confirm.get("decile_spread_mean"))
    if not math.isfinite(spread):
        reasons.append("decile spread not available")
    elif spread <= 0:
        reasons.append(f"decile spread {spread:+.4f} <= 0")
    return reasons


def is_measurable(confirm: dict, *, min_n_eff: float = MIN_CONFIRM_N_EFF) -> bool:
    """
    Whether the confirm folds hold enough evidence to judge a model at all:
    n_eff_total >= ``min_n_eff`` and an AUC interval could be drawn. When not,
    their AUC is not a measurement and a report should say "not measurable"
    rather than print it; ``ship_decision`` answers "prior" either way.
    """
    confirm = confirm or {}
    n_eff = _as_float(confirm.get("n_eff_total"))
    return (math.isfinite(n_eff) and n_eff >= min_n_eff
            and math.isfinite(_as_float(confirm.get("auc_ci_low"))))


def ship_decision(confirm: dict, *, min_auc: float = MIN_SHIP_AUC) -> Literal["model", "prior"]:
    """
    "model" iff, on the confirm folds, n_eff_total >= MIN_CONFIRM_N_EFF,
    auc_mean >= min_auc, auc_ci_low > 0.50 (the same-date AUC and its
    interval), brier_skill_mean > 0 and decile_spread_mean > 0. Anything
    missing, None or NaN means "prior": no measurable edge, serve the base
    rate. ``ship_reasons`` says which failed.
    """
    return "prior" if ship_reasons(confirm, min_auc=min_auc) else "model"


def shuffle_control(model_factory, X, y, fwd_ret, dates, tickers, folds: list[Fold],
                    seed: int = 0, *, horizon: int | None = None,
                    n_threads: int = DEFAULT_THREADS) -> float:
    """
    Overall AUC after permuting labels (and forward returns, with the same
    permutation) across the labeled rows the folds use. A harness that leaks
    nothing scores ~0.50 here; anything well above means the folds or the
    features see the future.
    """
    if not folds:
        raise ValueError("shuffle_control: no folds")
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    rows = _fold_rows_union(folds)
    perm = rows[np.random.default_rng(seed).permutation(rows.size)]
    y_arr = _as_vector(y, n, "y")
    y_shuffled = y_arr.copy()
    y_shuffled[rows] = y_arr[perm]
    fwd_shuffled = None
    if fwd_ret is not None:
        fwd = _as_vector(fwd_ret, n, "fwd_ret")
        fwd_shuffled = fwd.copy()
        fwd_shuffled[rows] = fwd[perm]
    result = evaluate(model_factory, X, y_shuffled, fwd_shuffled, dates, tickers, folds,
                      calibration="identity", importance=False, random_state=seed,
                      horizon=horizon, n_threads=n_threads)
    auc = float(result.overall["auc_mean"])
    log.debug("model_eval.shuffle_control", horizon=result.horizon, auc_mean=round(auc, 4))
    return auc


# ---------------------------------------------------------------------------
# Training-row weights and dead-band
# ---------------------------------------------------------------------------

WEIGHT_SCHEMES = ("none", "abs_ret_scaled")


def weights_for(scheme: str, fwd_ret, vol_scaled_ret, dates, *,
                half_life_years: float | None = None) -> np.ndarray | None:
    """
    Per-row training weights, or None for unweighted.

    * ``none``: no weights (None) unless a half-life is given.
    * ``abs_ret_scaled``: clip(|vol_scaled_ret|, 0.1, 3.0) - big moves count
      more, capped so one crash day cannot dominate; NaN -> 1.0.
    * ``half_life_years``: multiply by 0.5 ** (age / half_life), age in years
      back from the latest date; rows with a NaT date get age 0.

    ``vol_scaled_ret`` = fwd_ret / (vol_21 * sqrt(h)), supplied by the caller.
    ``evaluate`` rescales weights to mean 1 per fold, so only ratios matter.
    """
    if scheme not in WEIGHT_SCHEMES:
        raise ValueError(f"weight scheme must be one of {WEIGHT_SCHEMES}, got {scheme!r}")
    n = np.asarray(fwd_ret).size
    weights = None
    if scheme == "abs_ret_scaled":
        if vol_scaled_ret is None:
            raise ValueError("abs_ret_scaled needs vol_scaled_ret")
        v = np.abs(_as_vector(vol_scaled_ret, n, "vol_scaled_ret"))
        weights = np.where(np.isfinite(v), np.clip(v, 0.1, 3.0), 1.0)
    if half_life_years is not None:
        if half_life_years <= 0:
            raise ValueError(f"half_life_years must be > 0, got {half_life_years}")
        day = _day_numbers(dates)
        if day.size != n:
            raise ValueError(f"dates has {day.size} rows, expected {n}")
        valid = day != _NAT_DAY
        age_years = np.zeros(n, dtype=float)
        if valid.any():
            age_years[valid] = (day[valid].max() - day[valid]) / 365.25
        decay = 0.5 ** (age_years / float(half_life_years))
        weights = decay if weights is None else weights * decay
    return weights


def deadband_mask(vol_scaled_ret, threshold: float) -> np.ndarray:
    """
    True = keep for training. Drops rows whose |vol_scaled_ret| < threshold
    (moves too small to call a direction); NaN rows are kept, and a threshold
    <= 0 keeps everything. Pass as ``evaluate(train_mask=...)``: test rows are
    never dropped.
    """
    v = np.abs(np.asarray(vol_scaled_ret, dtype=float).ravel())
    if threshold <= 0:
        return np.ones(v.shape, dtype=bool)
    return ~(np.isfinite(v) & (v < threshold))


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fmt(value, digits: int = 3) -> str:
    v = _as_float(value)
    return f"{v:.{digits}f}" if math.isfinite(v) else "n/a"


def _fmt_mean_std(block: dict, key: str, digits: int = 3) -> str:
    if not block:
        return "n/a"
    text = _fmt(block.get(f"{key}_mean"), digits)
    std = _as_float(block.get(f"{key}_std"))
    return f"{text} +/- {std:.{digits}f}" if math.isfinite(std) and text != "n/a" else text


def _auc_cell(block: dict) -> str:
    if not block:
        return "n/a"
    return (f"{_fmt_mean_std(block, 'auc')} (90% CI {_fmt(block.get('auc_ci_low'))}"
            f" to {_fmt(block.get('auc_ci_high'))})")


def _hi_conf_cell(block: dict, key: str) -> str:
    if not block:
        return "n/a"
    return f"{_fmt(block.get(f'{key}_acc_pooled'))} (n={int(block.get(f'{key}_n_total') or 0)})"


_REPORT_ROWS = (
    ("AUC across dates (timing included; biased over short folds, not in the ship rule)",
     "auc_pooled", 3),
    ("Log loss", "log_loss", 4),
    ("Brier", "brier", 4),
    ("Brier of the base rate", "brier_prior", 4),
    ("Brier skill (> 0 beats the base rate)", "brier_skill", 4),
    ("Accuracy", "acc", 3),
    ("Always-majority accuracy", "acc_majority", 3),
    ("Accuracy skill (acc - majority)", "acc_skill", 3),
    ("Decile spread (same-date top - bottom decile fwd log return)", "decile_spread", 4),
    ("Base rate (training up-rate)", "prior_rate", 3),
    ("Effective observations per fold", "n_eff", 1),
)


def summarize_for_report(result: EvalResult, feature_names: list[str]) -> str:
    """Compact markdown: overall vs confirm, per fold, top-10 importances, reliability."""
    overall, confirm = result.overall, result.confirm
    reasons = ship_reasons(confirm)
    lines = [
        f"### {result.horizon}d horizon: ship decision `{ship_decision(confirm)}`",
        "",
        f"{len(result.folds)} walk-forward folds (confirm = last {result.confirm_folds}), "
        f"{result.n_rows:,} rows, {result.n_dates:,} dates, {result.n_tickers} tickers, "
        f"calibration `{result.calibrator_method}`, bootstrap blocks of "
        f"{confirm.get('block_length', block_length_for(result.horizon))} sessions, confirm "
        f"interval widened x{_fmt(confirm.get('ci_inflation'), 2)} for what blocks miss.",
        "",
    ]
    if reasons:
        lines += ["Failed ship conditions: " + "; ".join(reasons) + ".", ""]
    lines += [
        f"| Metric | Overall (mean +/- sd over {overall.get('n_folds', 0)} folds) "
        f"| Confirm ({confirm.get('n_folds', 0)} folds) |",
        "|---|---|---|",
        f"| AUC, tickers ranked against others on the same date (0.50 = no ranking skill) "
        f"| {_auc_cell(overall)} | {_auc_cell(confirm)} |",
    ]
    for label, key, digits in _REPORT_ROWS:
        lines.append(f"| {label} | {_fmt_mean_std(overall, key, digits)} "
                     f"| {_fmt_mean_std(confirm, key, digits)} |")
    for threshold, key in HI_CONF_LEVELS:
        lines.append(f"| Accuracy when abs(p - 0.5) >= {threshold:.2f} | "
                     f"{_hi_conf_cell(overall, key)} | {_hi_conf_cell(confirm, key)} |")

    lines += ["", "| Fold | Test window | Train rows | Test rows | AUC | Brier skill | Accuracy "
                  "| Decile spread | Calibrator |", "|---|---|---|---|---|---|---|---|---|"]
    for row in result.folds:
        slope = _as_float(row.get("calib_slope"))
        calibration = (f"{row['calibrator']} ({row.get('n_calib', 0):,} rows, n_eff "
                       f"{_fmt(row.get('calib_n_eff'), 1)}"
                       + (f", slope {slope:.3f}" if math.isfinite(slope) else "") + ")")
        lines.append(f"| {row['fold']} | {row['test_start']} to {row['test_end']} | "
                     f"{row['n_train']:,} | {row['n_test']:,} | {_fmt(row['auc'])} | "
                     f"{_fmt(row['brier_skill'], 4)} | {_fmt(row['acc'])} | "
                     f"{_fmt(row['decile_spread'], 4)} | {calibration} |")

    if result.importance:
        ranked = sorted(result.importance, reverse=True,
                        key=lambda r: r["importance"] if math.isfinite(r["importance"]) else -math.inf)
        lines += ["", "Permutation importance on the last fold (same-date AUC lost when the column "
                      "is shuffled):",
                  "", "| # | Feature | AUC drop |", "|---|---|---|"]
        for rank, row in enumerate(ranked[:10], start=1):
            i = int(row["feature"])
            name = feature_names[i] if 0 <= i < len(feature_names) else f"col_{i}"
            lines.append(f"| {rank} | {name} | {_fmt(row['importance'], 4)} |")

    if overall.get("reliability"):
        lines += ["", "Reliability, all folds pooled (calibrated forecasts):", "",
                  "| Forecast bin | Mean forecast | Observed up-rate | Rows |", "|---|---|---|---|"]
        for b in overall["reliability"]:
            lines.append(f"| {b['lo']:.1f}-{b['hi']:.1f} | {b['p_mean']:.3f} | "
                         f"{b['y_rate']:.3f} | {b['n']:,} |")
    return "\n".join(lines) + "\n"
