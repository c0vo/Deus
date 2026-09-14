"""
The saved form of one pooled direction model: what training produced and what
the live predictor needs to score a row.

Kept import-light (numpy and the standard library) on purpose. The predictor
loads one of these per horizon on every prediction path, and the class itself
must unpickle without dragging the training stack in with it — the fitted
estimator and calibrator inside bring their own imports when they are loaded.

An artifact is either a model or the prior. A horizon whose walk-forward
evaluation failed the ship rule (pipeline.model_eval.ship_decision), or could
not be measured at all, is saved with status "prior": no estimator, and
`predict_proba_up` answers the training up-rate. The metrics that explain the
decision are kept either way.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, MISSING
from typing import Any, Optional

import numpy as np

# A calibrated probability is never served closer to certainty than this. The
# calibrator is fitted on a few thousand out-of-fold rows, and a 99% reading
# from a model whose walk-forward AUC is in the 0.5s would be a claim the
# evaluation never supported.
PROBABILITY_FLOOR = 0.02
PROBABILITY_CEILING = 0.98

STATUS_MODEL = "model"
STATUS_PRIOR = "prior"


def _finite_or_none(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


@dataclass
class PooledArtifact:
    """One horizon's pooled model (or its no-edge fallback), as joblib saves it.

    feature_names    the columns the estimator was fitted on, in fit order
    feature_index    their positions in pipeline.features.FEATURE_NAMES
    categorical_idx  positions of the categorical columns within feature_names
    status           "model" or "prior"
    prior_up_rate    share of labeled training rows that closed higher
    trained_at       UTC ISO timestamp of the final fit
    train_end        last session date among the rows the final model saw
    metrics          confirm-fold summary: auc_mean, auc_ci_low, auc_ci_high,
                     brier_skill_mean, hi_conf_10_acc_pooled,
                     hi_conf_10_n_total, decile_spread_mean, prior_rate
    top_features     up to 8 names by permutation importance (may be empty)
    """

    model: Any = None
    calibrator: Any = None
    feature_names: list[str] = field(default_factory=list)
    feature_index: list[int] = field(default_factory=list)
    categorical_idx: list[int] = field(default_factory=list)
    horizon: int = 0
    schema_version: int = 0
    status: str = STATUS_PRIOR
    prior_up_rate: float = 0.5
    trained_at: str = ""
    train_end: str = ""
    config_name: str = ""
    universe: str = ""
    n_tickers: int = 0
    n_rows: int = 0
    metrics: dict = field(default_factory=dict)
    top_features: list[str] = field(default_factory=list)

    def __setstate__(self, state: dict) -> None:
        # An artifact pickled before a field existed still loads: the missing
        # field takes its default instead of raising AttributeError on first use.
        for f in fields(self):
            if f.name in state:
                continue
            if f.default is not MISSING:
                state[f.name] = f.default
            elif f.default_factory is not MISSING:  # type: ignore[misc]
                state[f.name] = f.default_factory()  # type: ignore[misc]
        self.__dict__.update(state)

    @property
    def is_model(self) -> bool:
        return self.status == STATUS_MODEL and self.model is not None

    def predict_proba_up(self, full_row_values: Any) -> float:
        """Calibrated P(close higher over the horizon) for one row.

        Accepts either a full row in FEATURE_NAMES order (it is narrowed with
        `feature_index`) or a row already narrowed to `feature_names`. A prior
        artifact, or a model that cannot score the row, answers `prior_up_rate`.
        """
        prior = float(self.prior_up_rate)
        if not self.is_model:
            return prior

        values = np.asarray(full_row_values, dtype=float).ravel()
        width = len(self.feature_names)
        if values.size != width:
            if self.feature_index and values.size > max(self.feature_index):
                values = values[np.asarray(self.feature_index, dtype=int)]
            else:
                raise ValueError(
                    f"row has {values.size} values; the {self.horizon}d model expects "
                    f"{width} selected or a full row covering index {max(self.feature_index or [0])}"
                )

        proba = self.model.predict_proba(values.reshape(1, -1))
        classes = list(getattr(self.model, "classes_", [0, 1]))
        p_raw = float(proba[0, classes.index(1)]) if 1 in classes else 0.0
        p = p_raw
        if self.calibrator is not None:
            p = float(np.asarray(self.calibrator.transform([p_raw]), dtype=float).ravel()[0])
        if not math.isfinite(p):
            return prior
        return float(min(max(p, PROBABILITY_FLOOR), PROBABILITY_CEILING))

    def meta(self) -> dict:
        """The model_meta block every prediction carries. NaN becomes None."""
        return {
            "auc": _finite_or_none(self.metrics.get("auc_mean")),
            "auc_ci_low": _finite_or_none(self.metrics.get("auc_ci_low")),
            "auc_ci_high": _finite_or_none(self.metrics.get("auc_ci_high")),
            "brier_skill": _finite_or_none(self.metrics.get("brier_skill_mean")),
            "base_rate": _finite_or_none(self.prior_up_rate),
            "trained_at": self.trained_at,
            "config": self.config_name,
            "universe": self.universe,
            "n_tickers": int(self.n_tickers),
        }
