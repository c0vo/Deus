"""
Desktop experiment runner for the pooled direction model.

Evaluates named configs (pipeline/model_configs.py) on purged walk-forward folds
through pipeline.model_training — the same code path the weekly retrain uses —
and writes what it measured to storage/experiments/, one directory per command:

    <UTC yyyymmdd-HHMMSS>-<config>[-tag]/
        config.json      the config, database, universe, tickers, git commit
        metrics.json     per horizon: EvalResult.to_dict(), baselines, timing, status
        per_ticker.csv   out-of-fold AUC / accuracy / Brier skill per ticker
        reliability.csv  calibrated forecast bins vs observed up-rate
        report.md        decision, confirm vs selection vs baselines, importances, folds

and regenerates storage/experiments/LEADERBOARD.md from every metrics.json.

Subcommands:

    run --config NAME       evaluate one config (--horizons, --universe, --no-importance, --tag)
    sweep --base NAME       the small hyperparameter grid on a base config; the winner is
                            picked on the selection folds, its confirm folds are reported
    control --config NAME   label-shuffle control: AUC must sit near 0.50 (flags |AUC-0.5| > 0.03)
    baselines               prior, momentum and logreg (run2's columns)
    coverage                share of missing values per feature group by calendar year
    leaderboard             rebuild LEADERBOARD.md

The database is opened in place with Database(path) and never initialize()d, so
a phone snapshot keeps its schema. It is read, with one exception:
Database.get_ticker_sector looks a sector up on yfinance and caches it in
ticker_info when a ticker has none stored. Sectors are warmed before the panel
cache key is taken, so that write does not invalidate the cache it precedes.

Panels are cached as storage/experiments/cache/panel_<sha1>.pkl, keyed on the
database path and mtime, the ticker list and the feature schema version.

Usage:
    python scripts/manual/train_experiments.py run --config run1 --db storage/phone_snapshot_2026-09-13.db
    python scripts/manual/train_experiments.py run --config run3 --horizons 5,21
    python scripts/manual/train_experiments.py sweep --base run2 --grid small --horizons 5,21
    python scripts/manual/train_experiments.py control --config run1
    python scripts/manual/train_experiments.py baselines
    python scripts/manual/train_experiments.py coverage
    python scripts/manual/train_experiments.py leaderboard
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(REPO_ROOT))

import numpy as np
import pandas as pd

from config.logging_config import get_logger
from config.settings import settings
from data.database import Database
from data.watchlist import ETF_TICKERS
from pipeline import model_training
from pipeline.features import FEATURE_GROUPS, FEATURE_SCHEMA_VERSION, HORIZONS
from pipeline.model_configs import (
    CONFIGS,
    SWEEP_GRID_SMALL,
    UNIVERSES,
    ModelConfig,
    fold_plan,
    get_config,
    sweep_configs,
    train_stride_for,
)
from pipeline.model_eval import (
    BLOCK_LENGTH_MULT,
    MIN_CONFIRM_N_EFF,
    MIN_SHIP_AUC,
    EvalResult,
    is_measurable,
    ship_reasons,
    shuffle_control,
    summarize_for_report,
)
from pipeline.model_training import SERVABLE_MODELS

log = get_logger(__name__)

EXPERIMENTS_DIR = REPO_ROOT / "storage" / "experiments"
CACHE_DIR = EXPERIMENTS_DIR / "cache"
LEADERBOARD_PATH = EXPERIMENTS_DIR / "LEADERBOARD.md"

GRIDS = {"small": SWEEP_GRID_SMALL}
BASELINE_CONFIGS = ("prior", "momentum", "logreg")
# Configs the leaderboard lists but never picks: the baselines, and the linear
# baseline of the run6 family (evaluated with the family, not by `baselines`).
NEVER_PICKED = BASELINE_CONFIGS + ("logreg_cs",)
CONTROL_TOLERANCE = 0.03
# Stamped into every entry. Entries written before the AUC counted same-date
# pairs only carry none: their AUCs rank rows across dates, which is biased, so
# the leaderboard lists them but never picks one.
AUC_STATISTIC = "same_date"

METRIC_GUIDE = (
    ("AUC", "ranking skill on the test folds, counting only pairs of tickers on the same date: "
            "0.50 is none, 0.53-0.57 is a real but small edge. Ranking a fold's rows across dates "
            "(`AUC across dates`) is biased for anything built from the price path - a late forecast "
            "already holds the returns behind the fold's earlier labels - so no decision reads it."),
    ("90% CI", f"circular block bootstrap over test dates, in runs of {BLOCK_LENGTH_MULT:g} x horizon "
               "consecutive sessions drawn across the folds of the block (overlapping labels make "
               "neighbouring dates near-duplicates); the ship rule needs its lower bound above 0.50."),
    ("Brier skill", "1 - Brier / Brier of always answering the base rate; above 0 means the "
                    "probabilities beat the base rate. Calibration is shrunk towards the base rate "
                    "in proportion to how little independent evidence earlier folds hold."),
    ("Accuracy vs majority", "raw hit rate against always calling the more common direction."),
    ("Hi-conf accuracy (n)", "hit rate when |p - 0.5| >= 0.10, over n such calls: what a badge "
                             "percentage should mean."),
    ("Decile spread", "per date, mean forward log return of that date's top decile of forecasts "
                      "minus its bottom decile, averaged over dates; above 0 means the ranking is "
                      "economically meaningful."),
    ("n_eff", "distinct test dates / horizon: overlapping labels make neighbouring dates "
              "near-duplicates, so this is how much independent evidence a fold holds. The fold "
              "plan sizes test blocks per horizon so the confirm folds can hold "
              f"{MIN_CONFIRM_N_EFF:g}; the 1y horizon cannot, and serves the base rate."),
    ("Ship rule", f"model iff the confirm folds hold n_eff >= {MIN_CONFIRM_N_EFF:g}, AUC >= "
                  f"{MIN_SHIP_AUC:.2f}, CI low > 0.50, Brier skill > 0 and decile spread > 0; "
                  "otherwise the horizon serves the base rate (prior), and the report lists every "
                  "condition it failed."),
    ("Selection vs confirm", "configs and hyperparameters are chosen on the selection folds only "
                             "(the leaderboard picks by selection AUC, among configs whose selection "
                             f"folds hold n_eff >= {MIN_CONFIRM_N_EFF:g}: a selection AUC from a few "
                             "weeks of folds is noise to choose on); the confirm folds (the "
                             "last two) are the pseudo-holdout the ship rule reads, and choosing "
                             "on them would spend it."),
    ("Training target", "a config's label chooses only what its model is fitted on (abs: close "
                        "higher; excess_spy: beat SPY; cs_median: beat the same date's median). "
                        "Every metric, the calibration and the decision read close-higher labels, "
                        "so every probability means P(close higher)."),
)


# ── Small helpers ────────────────────────────────────────────────────────────

def _clean(obj: Any) -> Any:
    """JSON-ready: numpy scalars unwrapped, NaN/inf as None, timestamps as ISO strings."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_clean(v) for v in obj.tolist()]
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


def _num(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _fmt(value: Any, digits: int = 3, *, signed: bool = False) -> str:
    number = _num(value)
    if number is None:
        return "n/a"
    return f"{number:+.{digits}f}" if signed else f"{number:.{digits}f}"


def _auc_text(auc: Any, low: Any, high: Any) -> str:
    if _num(auc) is None:
        return "n/a"
    if _num(low) is None or _num(high) is None:
        return _fmt(auc)
    return f"{_fmt(auc)} [{_fmt(low, 2)}-{_fmt(high, 2)}]"


def _hi_conf_text(acc: Any, n: Any) -> str:
    count = int(_num(n) or 0)
    return f"{_fmt(acc)} ({count:,})" if _num(acc) is not None else f"n/a ({count:,})"


def parse_horizons(text: Optional[str], default: tuple[int, ...] = HORIZONS) -> list[int]:
    if not text:
        return list(default)
    horizons = [int(part) for part in str(text).split(",") if part.strip()]
    unknown = [h for h in horizons if h not in HORIZONS]
    if unknown:
        raise SystemExit(f"unknown horizons {unknown}; the panel has {list(HORIZONS)}")
    return horizons


def git_commit() -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                             text=True, timeout=10)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def new_run_dir(name: str, tag: Optional[str]) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
    suffix = f"-{''.join(c if c.isalnum() or c in '._-' else '_' for c in tag)}" if tag else ""
    path = EXPERIMENTS_DIR / f"{stamp}-{safe}{suffix}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_clean(payload), indent=2, allow_nan=False), encoding="utf-8")


# ── Database and panel ───────────────────────────────────────────────────────

def open_db(path_text: Optional[str]) -> tuple[Database, Path]:
    """Database(path) on an existing file. Never initialize()d."""
    raw = Path(path_text or settings.db_path).expanduser()
    candidates = [raw] if raw.is_absolute() else [Path.cwd() / raw, REPO_ROOT / raw]
    for candidate in candidates:
        if candidate.exists():
            resolved = candidate.resolve()
            return Database(str(resolved)), resolved
    raise SystemExit(f"database not found: {raw} (looked in {', '.join(str(c) for c in candidates)})")


def warm_sectors(db: Database, symbols: list[str]) -> None:
    """Fill ticker_info for the symbols the feature loader will ask about (see module docstring)."""
    for symbol in symbols:
        if symbol in ETF_TICKERS or symbol.startswith("^") or symbol.endswith("-USD"):
            continue
        try:
            db.get_ticker_sector(symbol)
        except Exception as e:
            log.warning("experiments.sector_lookup_failed", ticker=symbol, error=str(e))


def panel_cache_path(db_path: Path, tickers: list[str]) -> Path:
    stat = db_path.stat()
    key = "|".join([str(db_path), str(stat.st_mtime_ns), ",".join(sorted(tickers)),
                    f"schema{FEATURE_SCHEMA_VERSION}", ",".join(str(h) for h in HORIZONS)])
    return CACHE_DIR / f"panel_{hashlib.sha1(key.encode('utf-8')).hexdigest()}.pkl"


def load_bundle(db: Database, db_path: Path, universe: str) -> dict:
    """{tickers, inputs, panel, info} for a universe, from the panel cache when it is current."""
    if universe not in UNIVERSES:
        raise SystemExit(f"unknown universe {universe!r}; known: {UNIVERSES}")
    tickers = model_training.training_tickers(db, universe)
    if not tickers:
        raise SystemExit(f"universe {universe!r} has no tickers in {db_path}")
    warm_sectors(db, tickers)
    cache = panel_cache_path(db_path, tickers)
    started = time.perf_counter()
    if cache.exists():
        inputs, panel = pd.read_pickle(cache)
        hit = True
    else:
        inputs, panel = model_training.load_panel(db, tickers)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        pd.to_pickle((inputs, panel), cache)
        hit = False
    info = {
        "cache": str(cache), "cache_hit": hit, "seconds": round(time.perf_counter() - started, 2),
        "rows": int(len(panel)), "tickers_with_rows": int(panel["ticker"].nunique()) if len(panel) else 0,
        "market_symbol": inputs.market_symbol,
        "coverage_start": (inputs.coverage_start.strftime("%Y-%m-%d")
                           if inputs.coverage_start is not None else None),
    }
    print(f"panel: {info['rows']:,} rows, {info['tickers_with_rows']}/{len(tickers)} tickers, "
          f"{'cache hit' if hit else 'built'} in {info['seconds']}s")
    return {"tickers": tickers, "inputs": inputs, "panel": panel, "info": info}


# ── Entries ──────────────────────────────────────────────────────────────────

def _block(result: Any, name: str) -> dict:
    if isinstance(result, EvalResult):
        return getattr(result, name) or {}
    if isinstance(result, dict):
        return result.get(name) or {}
    return {}


def summary_of(ev: dict) -> dict:
    result = ev.get("result")
    confirm, selection = _block(result, "confirm"), _block(result, "selection")
    baselines = ev.get("baselines") or {}
    design = ev.get("design")
    return _clean({
        "confirm_auc": confirm.get("auc_mean"),
        "confirm_auc_ci_low": confirm.get("auc_ci_low"),
        "confirm_auc_ci_high": confirm.get("auc_ci_high"),
        "confirm_brier_skill": confirm.get("brier_skill_mean"),
        "confirm_acc": confirm.get("acc_mean"),
        "confirm_acc_majority": confirm.get("acc_majority_mean"),
        "confirm_hi_conf_acc": confirm.get("hi_conf_10_acc_pooled"),
        "confirm_hi_conf_n": confirm.get("hi_conf_10_n_total"),
        "confirm_decile_spread": confirm.get("decile_spread_mean"),
        "confirm_prior_rate": confirm.get("prior_rate_mean"),
        "confirm_n_eff": confirm.get("n_eff_total"),
        "confirm_auc_pooled": confirm.get("auc_pooled_mean"),
        "selection_auc": selection.get("auc_mean"),
        "selection_auc_ci_low": selection.get("auc_ci_low"),
        "selection_auc_ci_high": selection.get("auc_ci_high"),
        "selection_n_eff": selection.get("n_eff_total"),
        "prior_confirm_auc": _block(baselines.get("prior"), "confirm").get("auc_mean"),
        "momentum_confirm_auc": _block(baselines.get("momentum"), "confirm").get("auc_mean"),
        "n_rows": design.n_rows if design is not None else 0,
        "n_tickers": int(pd.Series(design.tickers).nunique()) if design is not None and design.n_rows else 0,
    })


def entry_of(ev: dict, universe: str) -> dict:
    """What metrics.json keeps of one evaluate_config result."""
    cfg: ModelConfig = ev["config"]
    result = ev.get("result")
    design = ev.get("design")
    importance_named = []
    if isinstance(result, EvalResult) and design is not None:
        importance_named = model_training.ranked_importance(
            result.importance, model_training.design_column_names(cfg, design))
    plan = ev.get("plan") or fold_plan(cfg, ev["horizon"])
    return _clean({
        "config": cfg.name,
        "model": cfg.model,
        "horizon": ev["horizon"],
        "universe": universe,
        "status": ev["status"],
        "decision": ev.get("decision"),
        "reasons": ev.get("reasons"),
        "auc_statistic": AUC_STATISTIC,
        "label": cfg.label,
        "train_stride": train_stride_for(cfg, ev["horizon"]),
        "test_window": cfg.test_window,
        "fold_plan": dict(plan._asdict()),
        "error": ev.get("error"),
        "timing": {k: round(v, 3) for k, v in (ev.get("timing") or {}).items()},
        "columns": list(design.columns) if design is not None else [],
        "summary": summary_of(ev),
        "importance_named": importance_named,
        "result": result.to_dict() if isinstance(result, EvalResult) else result,
        "baselines": ev.get("baselines") or {},
    })


# ── Writers ──────────────────────────────────────────────────────────────────

def write_tables(run_dir: Path, evals: list[dict]) -> None:
    with (run_dir / "per_ticker.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["config", "horizon", "ticker", "n", "up_rate", "auc", "acc", "brier_skill"])
        for ev in evals:
            result = ev.get("result")
            if not isinstance(result, EvalResult):
                continue
            for r in result.per_ticker:
                writer.writerow([ev["config"].name, ev["horizon"], r["ticker"], r["n"],
                                 _fmt(r["up_rate"], 4), _fmt(r["auc"], 4), _fmt(r["acc"], 4),
                                 _fmt(r["brier_skill"], 4)])

    with (run_dir / "reliability.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["config", "horizon", "block", "bin", "lo", "hi", "p_mean", "y_rate", "n"])
        for ev in evals:
            for block_name in ("overall", "confirm"):
                for b in _block(ev.get("result"), block_name).get("reliability") or []:
                    writer.writerow([ev["config"].name, ev["horizon"], block_name, b["bin"], b["lo"],
                                     b["hi"], _fmt(b["p_mean"], 4), _fmt(b["y_rate"], 4), b["n"]])


def _comparison_table(ev: dict) -> list[str]:
    result = ev.get("result")
    baselines = ev.get("baselines") or {}
    columns = [("Selection", _block(result, "selection")), ("Confirm", _block(result, "confirm")),
               ("Momentum (selection)", _block(baselines.get("momentum"), "selection")),
               ("Momentum (confirm)", _block(baselines.get("momentum"), "confirm")),
               ("Prior (confirm)", _block(baselines.get("prior"), "confirm"))]
    rows = [
        ("AUC [90% CI]", lambda b: _auc_text(b.get("auc_mean"), b.get("auc_ci_low"), b.get("auc_ci_high"))),
        ("AUC across dates (not decided on)", lambda b: _fmt(b.get("auc_pooled_mean"))),
        ("Brier skill", lambda b: _fmt(b.get("brier_skill_mean"), 4, signed=True)),
        ("Accuracy", lambda b: _fmt(b.get("acc_mean"))),
        ("Majority accuracy", lambda b: _fmt(b.get("acc_majority_mean"))),
        ("Hi-conf accuracy (n)", lambda b: _hi_conf_text(b.get("hi_conf_10_acc_pooled"),
                                                          b.get("hi_conf_10_n_total"))),
        ("Decile spread", lambda b: _fmt(b.get("decile_spread_mean"), 4, signed=True)),
        ("Base rate", lambda b: _fmt(b.get("prior_rate_mean"))),
        ("n_eff per fold", lambda b: _fmt(b.get("n_eff_mean"), 1)),
        ("n_eff, all folds", lambda b: _fmt(b.get("n_eff_total"), 1)),
        ("Folds", lambda b: str(b.get("n_folds", 0))),
    ]
    lines = ["| Metric | " + " | ".join(name for name, _ in columns) + " |",
             "|---|" + "---|" * len(columns)]
    for label, render in rows:
        lines.append(f"| {label} | " + " | ".join(render(block) if block else "n/a"
                                                  for _, block in columns) + " |")
    return lines


def _importance_lines(ev: dict) -> list[str]:
    result = ev.get("result")
    design = ev.get("design")
    if not isinstance(result, EvalResult) or design is None or not result.importance:
        return ["Permutation importance: not computed for this run."]
    cfg = ev["config"]
    names = model_training.design_column_names(cfg, design)
    ranked = model_training.ranked_importance(result.importance, names)
    group_of = {c: g for g, cols in FEATURE_GROUPS.items() for c in cols}
    lines = ["Top 15 features by permutation importance on the last fold (AUC lost when shuffled):", "",
             "| # | Feature | Group | AUC drop | sd |", "|---|---|---|---|---|"]
    for rank, row in enumerate(ranked[:15], start=1):
        lines.append(f"| {rank} | {row['feature']} | {group_of.get(row['feature'], '-')} | "
                     f"{_fmt(row['importance'], 4, signed=True)} | {_fmt(row['std'], 4)} |")
    sums: dict[str, float] = {}
    for row in ranked:
        value = _num(row["importance"])
        group = group_of.get(row["feature"])
        if value is not None and group:
            sums[group] = sums.get(group, 0.0) + value
    if sums:
        lines += ["", "Group sums: " + ", ".join(f"{g} {v:+.4f}" for g, v in
                                                  sorted(sums.items(), key=lambda kv: -kv[1]))]
    return lines


def write_report(run_dir: Path, title: str, header: list[str], evals: list[dict],
                 extra: Optional[list[str]] = None) -> None:
    lines = [f"# {title}", ""] + header + ["", "## How to read this", ""]
    lines += [f"- **{name}**: {text}" for name, text in METRIC_GUIDE]
    if extra:
        lines += [""] + extra
    for ev in evals:
        cfg: ModelConfig = ev["config"]
        h = ev["horizon"]
        plan = ev.get("plan") or fold_plan(cfg, h)
        lines += ["", f"## {cfg.name}, {h}d horizon: decision `{ev.get('decision') or 'n/a'}`", ""]
        if cfg.description:
            lines += [cfg.description, ""]
        lines += [f"Fold plan: {plan.n_folds} folds of {plan.test_days} calendar days, the last "
                  f"{plan.confirm_folds} confirm.", ""]
        stride = train_stride_for(cfg, h)
        if cfg.label != "abs" or stride > 1:
            rows = "every session" if stride == 1 else f"every {stride}th session, whole dates"
            lines += [f"Training: fitted on label `{cfg.label}` over {rows}; test rows, calibration "
                      "and every figure below read close-higher labels on every session.", ""]
        if ev["status"] != "evaluated":
            lines += [f"Not measurable: {ev.get('error')}"]
            continue
        if cfg.model in ("prior", "momentum"):
            lines += ["Baseline rule: its decision is reported for comparison only; it is not "
                      "a servable model.", ""]
        reasons = ev.get("reasons")
        if reasons:
            lines += ["Why it serves the base rate (confirm folds):", ""]
            lines += [f"- {reason}" for reason in reasons] + [""]
        lines += ["Compare configs on the Selection column; Confirm is the holdout the decision "
                  "reads.", ""]
        lines += _comparison_table(ev) + [""] + _importance_lines(ev)
        result = ev.get("result")
        if isinstance(result, EvalResult):
            names = model_training.design_column_names(cfg, ev["design"])
            lines += ["", summarize_for_report(result, names)]
        timing = ev.get("timing") or {}
        lines += ["", "Timing: " + ", ".join(f"{k} {v:.1f}s" for k, v in timing.items())]
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_metrics_files() -> list[tuple[str, dict]]:
    """(run directory, metrics.json) per run, entries annotated with their test window.

    Entries written before they recorded `test_window` take it from the run's
    config.json (a sweep point from its base config), else from CONFIGS by name.
    """
    out = []
    for path in sorted(EXPERIMENTS_DIR.glob("*/metrics.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            log.warning("experiments.metrics_unreadable", path=str(path), error=str(e))
            continue
        run_configs: list = []
        config_path = path.parent / "config.json"
        if config_path.exists():
            try:
                run_configs = json.loads(config_path.read_text(encoding="utf-8")).get("configs") or []
            except (OSError, ValueError, AttributeError) as e:
                log.warning("experiments.config_unreadable", path=str(config_path), error=str(e))
        for entry in payload.get("entries") or []:
            if isinstance(entry, dict) and "test_window" not in entry:
                entry["test_window"] = legacy_test_window(entry, run_configs)
        out.append((path.parent.name, payload))
    return out


def legacy_test_window(entry: dict, run_configs: list) -> Optional[str]:
    """The test window of an entry written before entries recorded it, or None when unknown."""
    by_name = {c.get("name"): c for c in run_configs if isinstance(c, dict)}
    config = by_name.get(entry.get("config"))
    if config is None and len(by_name) == 1:        # a sweep point: the grid's base config
        config = next(iter(by_name.values()))
    if config is None and entry.get("config") in CONFIGS:
        config = CONFIGS[entry["config"]].to_dict()
    return (config or {}).get("test_window")


def entry_selection_n_eff(entry: dict) -> Optional[float]:
    """Independent label windows on the entry's selection folds (distinct test dates / horizon).

    The run summary's selection_n_eff; for runs written before it was recorded,
    the stored result's selection block, else the per-fold test dates of the
    folds before the confirm folds. None when none of them is stored.
    """
    summary = entry.get("summary") or {}
    value = _num(summary.get("selection_n_eff"))
    if value is not None:
        return value
    result = entry.get("result") if isinstance(entry.get("result"), dict) else {}
    value = _num((result.get("selection") or {}).get("n_eff_total"))
    if value is not None:
        return value
    folds = [f for f in result.get("folds") or [] if isinstance(f, dict)]
    horizon = _num(entry.get("horizon"))
    confirm = _num(result.get("confirm_folds") or (entry.get("fold_plan") or {}).get("confirm_folds"))
    if not folds or not horizon or confirm is None or len(folds) <= int(confirm):
        return None
    dates = [_num(f.get("n_dates")) for f in folds[:len(folds) - int(confirm)]]
    if any(d is None for d in dates):
        return None
    return float(sum(dates)) / horizon


def _basic_pick_exclusion(entry: dict) -> Optional[str]:
    if entry.get("status") != "evaluated":
        return "not evaluated"
    if entry.get("model") not in SERVABLE_MODELS or entry.get("config") in NEVER_PICKED:
        return "baseline"
    if entry.get("auc_statistic") != AUC_STATISTIC:
        return "across-dates AUC: re-run"
    if _num((entry.get("summary") or {}).get("selection_auc")) is None:
        return "no selection AUC"
    return None


def selection_evidence_exclusion(entry: dict) -> Optional[str]:
    """Why the entry's selection folds are too thin to choose on, or None when they hold enough.

    A selection AUC over fewer than MIN_CONFIRM_N_EFF independent label windows
    (run3's few weeks of news coverage) is noise, and ranking on it would pick
    whichever thin run got lucky. A coverage-era run written before the
    summary recorded selection_n_eff is excluded outright; any other old run
    is recomputed from its stored folds, and excluded when it cannot be.
    """
    summary = entry.get("summary") or {}
    recorded = _num(summary.get("selection_n_eff"))
    if recorded is None and entry.get("test_window") == "coverage_era":
        return "coverage-era run written before selection n_eff was recorded: re-run"
    n_eff = recorded if recorded is not None else entry_selection_n_eff(entry)
    if n_eff is None:
        return "selection n_eff not recorded and not recoverable from the stored folds: re-run"
    if n_eff < MIN_CONFIRM_N_EFF:
        return (f"selection n_eff {n_eff:.1f} < {MIN_CONFIRM_N_EFF:g}: too few independent "
                "windows to choose on")
    return None


def pick_exclusion(entry: dict) -> Optional[str]:
    """Why the leaderboard may not pick this entry for PRODUCTION, or None when it may."""
    return _basic_pick_exclusion(entry) or selection_evidence_exclusion(entry)


def is_pick_candidate(entry: dict) -> bool:
    """Whether the leaderboard may pick this entry for PRODUCTION: evaluated, servable, not a
    baseline, scored with the same-date AUC (AUC_STATISTIC), and chosen on selection folds that
    hold at least MIN_CONFIRM_N_EFF independent label windows."""
    return pick_exclusion(entry) is None


def entry_reasons(entry: dict) -> list[str]:
    """The entry's ship_reasons, recomputed from its summary for runs written before they were stored."""
    if entry.get("reasons") is not None:
        return list(entry["reasons"])
    s = entry.get("summary") or {}
    return ship_reasons({"auc_mean": s.get("confirm_auc"), "auc_ci_low": s.get("confirm_auc_ci_low"),
                         "brier_skill_mean": s.get("confirm_brier_skill"),
                         "decile_spread_mean": s.get("confirm_decile_spread"),
                         "n_eff_total": s.get("confirm_n_eff"), "horizon": entry.get("horizon")})


def entry_measurable(entry: dict) -> bool:
    """model_eval.is_measurable on the entry's confirm summary: False for too few independent windows."""
    s = entry.get("summary") or {}
    return entry.get("status") == "evaluated" and is_measurable(
        {"n_eff_total": s.get("confirm_n_eff"), "auc_ci_low": s.get("confirm_auc_ci_low")})


def _status_text(entry: dict) -> str:
    status = str(entry.get("decision") or "n/a") if entry_measurable(entry) else "not measurable"
    if entry.get("status") == "evaluated" and entry.get("auc_statistic") != AUC_STATISTIC:
        status += " (across-dates AUC: re-run)"
    if entry.get("config") in NEVER_PICKED or entry.get("model") not in SERVABLE_MODELS:
        return f"{status} (baseline)"
    return status


def write_leaderboard() -> str:
    """Rebuild LEADERBOARD.md from every metrics.json and return its text.

    Rows are ranked by selection AUC, and the pick per horizon is the best
    selection AUC among is_pick_candidate entries. Ranking on the confirm folds
    would choose PRODUCTION on the holdout the ship rule is meant to read
    blind, so confirm figures are shown beside the pick, never used to find it.
    Entries that would be candidates but for thin selection folds are listed
    under the pick with the reason.
    """
    runs = _load_metrics_files()
    rows, controls = [], []
    for run_name, payload in runs:
        kind = payload.get("kind")
        for entry in payload.get("entries") or []:
            if kind == "control":
                controls.append((run_name, entry))
            elif kind in ("run", "sweep"):
                rows.append((run_name, entry))

    lines = ["# Direction model leaderboard", "",
             f"Rebuilt {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC from {len(runs)} experiment "
             "directories. Ranked by selection AUC (the walk-forward folds before the last two); "
             "`*` marks the pick per horizon, the best selection AUC among evaluated, servable, "
             "non-baseline configs. Confirm = the last two folds, the holdout the ship rule reads: "
             "it is shown, never ranked on, because picking on it would spend it. A sweep winner's "
             "selection AUC is the best of its grid, so it is optimistic next to a single run's. "
             f"A config is picked only when its selection folds hold n_eff >= {MIN_CONFIRM_N_EFF:g} "
             "independent label windows. "
             "AUCs rank tickers on the same date; runs written before that are marked "
             "`across-dates AUC: re-run` and never picked.", ""]
    for h in sorted({entry["horizon"] for _, entry in rows}):
        at_h = [r for r in rows if r[1]["horizon"] == h]
        candidates = [r for r in at_h if is_pick_candidate(r[1])]
        thin = [(run_name, entry, selection_evidence_exclusion(entry)) for run_name, entry in at_h
                if _basic_pick_exclusion(entry) is None and selection_evidence_exclusion(entry)]
        pick = max(candidates, key=lambda r: _num(r[1]["summary"]["selection_auc"])) if candidates else None
        lines += [f"## {h}d", ""]
        if pick is None:
            lines += ["No pick: no evaluated, servable, non-baseline config at this horizon with "
                      f"selection folds holding n_eff >= {MIN_CONFIRM_N_EFF:g}.", ""]
        else:
            run_name, entry = pick
            s = entry.get("summary") or {}
            reasons = entry_reasons(entry)
            lines += [f"**Pick: `{entry.get('config')}`** ({run_name}, {entry.get('universe', '')}): "
                      f"selection AUC {_fmt(s.get('selection_auc'))}; confirm AUC "
                      f"{_auc_text(s.get('confirm_auc'), s.get('confirm_auc_ci_low'), s.get('confirm_auc_ci_high'))}, "
                      f"confirm n_eff {_fmt(s.get('confirm_n_eff'), 1)}, decision "
                      f"`{entry.get('decision') or 'n/a'}`.", ""]
            if reasons:
                lines += ["Why the pick serves the base rate:", ""] + [f"- {r}" for r in reasons] + [""]
        if thin:
            lines += ["Not pickable, selection evidence too thin to choose on:", ""]
            lines += [f"- `{entry.get('config')}` ({run_name}): {reason}" for run_name, entry, reason in thin]
            lines += [""]
        lines += ["| Run | Config | Universe | Status | Selection AUC [90% CI] | Selection n_eff | "
                  "Confirm AUC [90% CI] | Confirm n_eff | Brier skill | Hi-conf acc (n) | Decile spread | "
                  "Prior AUC | Momentum AUC |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        ranked = sorted(at_h, key=lambda r: -(_num((r[1].get("summary") or {}).get("selection_auc")) or -1.0))
        for run_name, entry in ranked:
            s = entry.get("summary") or {}
            mark = " *" if pick is not None and entry is pick[1] else ""
            lines.append(
                f"| {run_name} | {entry.get('config')}{mark} | {entry.get('universe', '')} | "
                f"{_status_text(entry)} | "
                f"{_auc_text(s.get('selection_auc'), s.get('selection_auc_ci_low'), s.get('selection_auc_ci_high'))} | "
                f"{_fmt(entry_selection_n_eff(entry), 1)} | "
                f"{_auc_text(s.get('confirm_auc'), s.get('confirm_auc_ci_low'), s.get('confirm_auc_ci_high'))} | "
                f"{_fmt(s.get('confirm_n_eff'), 1)} | "
                f"{_fmt(s.get('confirm_brier_skill'), 4, signed=True)} | "
                f"{_hi_conf_text(s.get('confirm_hi_conf_acc'), s.get('confirm_hi_conf_n'))} | "
                f"{_fmt(s.get('confirm_decile_spread'), 4, signed=True)} | "
                f"{_fmt(s.get('prior_confirm_auc'))} | {_fmt(s.get('momentum_confirm_auc'))} |")
        lines.append("")
    if controls:
        lines += ["## Label-shuffle controls", "",
                  "| Run | Config | Horizon | Shuffled AUC | Within +/-0.03 |", "|---|---|---|---|---|"]
        for run_name, entry in controls:
            auc = _num(entry.get("shuffled_auc"))
            ok = "n/a" if auc is None else ("yes" if abs(auc - 0.5) <= CONTROL_TOLERANCE else "NO")
            lines.append(f"| {run_name} | {entry.get('config')} | {entry.get('horizon')}d | "
                         f"{_fmt(auc)} | {ok} |")
        lines.append("")
    text = "\n".join(lines) + "\n"
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    LEADERBOARD_PATH.write_text(text, encoding="utf-8")
    return text


def print_console_table(entries: list[dict]) -> None:
    headers = ("config", "h", "status", "decision", "selection AUC", "sel n_eff", "confirm AUC [CI]",
               "n_eff", "Brier skill", "hi-conf acc (n)", "spread", "prior AUC", "mom AUC", "sec")
    body = []
    for e in entries:
        s = e.get("summary") or {}
        body.append((
            str(e.get("config")), f"{e.get('horizon')}d", str(e.get("status")),
            str(e.get("decision") or "-"), _fmt(s.get("selection_auc")),
            _fmt(s.get("selection_n_eff"), 1),
            _auc_text(s.get("confirm_auc"), s.get("confirm_auc_ci_low"), s.get("confirm_auc_ci_high")),
            _fmt(s.get("confirm_n_eff"), 1),
            _fmt(s.get("confirm_brier_skill"), 4, signed=True),
            _hi_conf_text(s.get("confirm_hi_conf_acc"), s.get("confirm_hi_conf_n")),
            _fmt(s.get("confirm_decile_spread"), 4, signed=True),
            _fmt(s.get("prior_confirm_auc")), _fmt(s.get("momentum_confirm_auc")),
            f"{(e.get('timing') or {}).get('total', 0.0):.1f}",
        ))
    widths = [max(len(headers[i]), *(len(row[i]) for row in body)) if body else len(headers[i])
              for i in range(len(headers))]
    print()
    print(" | ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("-+-".join("-" * w for w in widths))
    for row in body:
        print(" | ".join(c.ljust(w) for c, w in zip(row, widths)))
    for e in entries:
        if e.get("status") != "evaluated":
            print(f"  {e.get('config')} {e.get('horizon')}d not measurable: {e.get('error')}")
        elif e.get("reasons"):
            print(f"  {e.get('config')} {e.get('horizon')}d prior: {'; '.join(e['reasons'])}")


# ── Commands ─────────────────────────────────────────────────────────────────

def _threads(args) -> int:
    return int(args.threads or settings.predictor_threads)


def _run_header(bundle: dict, db_path: Path, universe: str, threads: int) -> list[str]:
    info = bundle["info"]
    return [f"- Database: `{db_path}`",
            f"- Universe: {universe} ({len(bundle['tickers'])} tickers, "
            f"{info['tickers_with_rows']} with rows): {', '.join(bundle['tickers'])}",
            f"- Panel: {info['rows']:,} rows, market series {info['market_symbol']}, "
            f"news coverage from {info['coverage_start']}",
            f"- Feature schema v{FEATURE_SCHEMA_VERSION}, threads {threads}, commit {git_commit() or 'unknown'}",
            f"- Created {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC"]


def _config_payload(configs: list[ModelConfig], bundle: dict, db_path: Path, universe: str,
                    horizons: list[int], threads: int, tag: Optional[str], command: str) -> dict:
    return {
        "command": command,
        "configs": [c.to_dict() for c in configs],
        "db": str(db_path),
        "universe": universe,
        "horizons": horizons,
        "threads": threads,
        "tag": tag,
        "tickers": bundle["tickers"],
        "schema_version": FEATURE_SCHEMA_VERSION,
        "panel": bundle["info"],
        "git_commit": git_commit(),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _evaluate_configs(configs: list[ModelConfig], bundle: dict, horizons: list[int], threads: int,
                      importance: bool) -> list[dict]:
    evals = []
    for cfg in configs:
        for h in horizons:
            print(f"evaluating {cfg.name} at {h}d ...", flush=True)
            ev = model_training.evaluate_config(bundle["panel"], bundle["inputs"], cfg, h,
                                                n_threads=threads, importance=importance)
            evals.append(ev)
    return evals


def _finish_run(run_dir: Path, title: str, configs: list[ModelConfig], evals: list[dict], bundle: dict,
                db_path: Path, universe: str, horizons: list[int], threads: int, tag: Optional[str],
                command: str, started: float, kind: str = "run", extra_payload: Optional[dict] = None,
                extra_report: Optional[list[str]] = None) -> list[dict]:
    entries = [entry_of(ev, universe) for ev in evals]
    write_json(run_dir / "config.json",
               _config_payload(configs, bundle, db_path, universe, horizons, threads, tag, command))
    payload = {"kind": kind, "run": run_dir.name, "universe": universe,
               "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "seconds": round(time.perf_counter() - started, 1), "entries": entries}
    if extra_payload:
        payload.update(extra_payload)
    write_json(run_dir / "metrics.json", payload)
    write_tables(run_dir, evals)
    write_report(run_dir, title, _run_header(bundle, db_path, universe, threads), evals, extra_report)
    write_leaderboard()
    return entries


def cmd_run(args) -> None:
    started = time.perf_counter()
    cfg = get_config(args.config)
    universe = args.universe or cfg.universe
    horizons = parse_horizons(args.horizons)
    threads = _threads(args)
    db, db_path = open_db(args.db)
    bundle = load_bundle(db, db_path, universe)
    run_dir = new_run_dir(cfg.name, args.tag)
    evals = _evaluate_configs([cfg], bundle, horizons, threads, importance=not args.no_importance)
    entries = _finish_run(run_dir, f"{cfg.name} ({universe})", [cfg], evals, bundle, db_path, universe,
                          horizons, threads, args.tag, "run", started)
    print_console_table(entries)
    print(f"\nwrote {run_dir} in {time.perf_counter() - started:.1f}s")


def cmd_baselines(args) -> None:
    started = time.perf_counter()
    configs = [get_config(name) for name in BASELINE_CONFIGS]
    universe = args.universe or get_config("run2").universe
    horizons = parse_horizons(args.horizons)
    threads = _threads(args)
    db, db_path = open_db(args.db)
    bundle = load_bundle(db, db_path, universe)
    run_dir = new_run_dir("baselines", args.tag)
    evals = _evaluate_configs(configs, bundle, horizons, threads, importance=False)
    entries = _finish_run(run_dir, f"Baselines ({universe})", configs, evals, bundle, db_path, universe,
                          horizons, threads, args.tag, "baselines", started)
    print_console_table(entries)
    print(f"\nwrote {run_dir} in {time.perf_counter() - started:.1f}s")


def cmd_sweep(args) -> None:
    started = time.perf_counter()
    base = get_config(args.base)
    if base.model != "hgb":
        raise SystemExit(f"sweep grids tune HistGradientBoosting; {base.name} is {base.model}")
    configs = sweep_configs(base, GRIDS[args.grid])
    universe = args.universe or base.universe
    horizons = parse_horizons(args.horizons, default=(5, 21))
    threads = _threads(args)
    db, db_path = open_db(args.db)
    bundle = load_bundle(db, db_path, universe)
    run_dir = new_run_dir(f"sweep-{base.name}", args.tag)

    winners, points_by_h = [], {}
    for h in horizons:
        points, best_ev, best_auc = [], None, -math.inf
        for i, cfg in enumerate(configs, start=1):
            ev = model_training.evaluate_config(bundle["panel"], bundle["inputs"], cfg, h,
                                                n_threads=threads, importance=False)
            s = summary_of(ev)
            points.append({"config": cfg.name, "params": cfg.params, "status": ev["status"],
                           "decision": ev.get("decision"), "reasons": ev.get("reasons"),
                           "error": ev.get("error"),
                           "selection_auc": s["selection_auc"], "confirm_auc": s["confirm_auc"],
                           "confirm_brier_skill": s["confirm_brier_skill"],
                           "confirm_n_eff": s["confirm_n_eff"],
                           "seconds": round(ev["timing"].get("total", 0.0), 1)})
            print(f"[{h}d {i}/{len(configs)}] {cfg.name}: selection AUC {_fmt(s['selection_auc'])}, "
                  f"confirm AUC {_fmt(s['confirm_auc'])}", flush=True)
            selection_auc = _num(s["selection_auc"])
            if ev["status"] == "evaluated" and selection_auc is not None and selection_auc > best_auc:
                if best_ev is not None:
                    best_ev.clear()   # the design matrix of a superseded leader
                best_ev, best_auc = ev, selection_auc
            else:
                ev.clear()   # release the design matrix of a point that did not lead
        points_by_h[str(h)] = points
        if best_ev is None:
            print(f"{h}d: no sweep point was measurable")
            continue
        winner_cfg = best_ev["config"]
        if not args.no_importance:
            best_ev = model_training.evaluate_config(bundle["panel"], bundle["inputs"], winner_cfg, h,
                                                     n_threads=threads, importance=True)
        winners.append(best_ev)
        print(f"{h}d winner on selection folds: {winner_cfg.name} (selection AUC {best_auc:.3f}); "
              f"confirm decision {best_ev.get('decision')}")

    ranking = ["## Sweep points (ranked by selection AUC)", "",
               "The winner is the top row; its confirm figures are the only ones the decision reads. "
               "Confirm AUCs of the other points are listed for completeness, not for choosing.", ""]
    for h, points in points_by_h.items():
        ranking += [f"### {h}d", "",
                    "| Config | Selection AUC | Confirm AUC | Confirm n_eff | Confirm Brier skill | Decision |",
                    "|---|---|---|---|---|---|"]
        for p in sorted(points, key=lambda p: -(_num(p["selection_auc"]) or -1.0)):
            ranking.append(f"| {p['config']} | {_fmt(p['selection_auc'])} | {_fmt(p['confirm_auc'])} | "
                           f"{_fmt(p['confirm_n_eff'], 1)} | "
                           f"{_fmt(p['confirm_brier_skill'], 4, signed=True)} | {p['decision'] or '-'} |")
        ranking.append("")
    entries = _finish_run(run_dir, f"Sweep on {base.name} ({args.grid} grid, {universe})", [base],
                          winners, bundle, db_path, universe, horizons, threads, args.tag, "sweep", started,
                          kind="sweep", extra_payload={"grid": GRIDS[args.grid], "points": points_by_h},
                          extra_report=ranking)
    print_console_table(entries)
    print(f"\nwrote {run_dir} in {time.perf_counter() - started:.1f}s")


def cmd_control(args) -> None:
    started = time.perf_counter()
    cfg = get_config(args.config)
    if cfg.model in ("prior", "momentum"):
        raise SystemExit(f"{cfg.name} fits nothing; a shuffle control needs a model config")
    universe = args.universe or cfg.universe
    horizons = parse_horizons(args.horizons)
    threads = _threads(args)
    db, db_path = open_db(args.db)
    bundle = load_bundle(db, db_path, universe)
    run_dir = new_run_dir(f"control-{cfg.name}", args.tag)

    entries = []
    for h in horizons:
        t0 = time.perf_counter()
        entry = {"config": cfg.name, "horizon": h, "universe": universe, "status": "evaluated",
                 "shuffled_auc": None, "flag": None, "error": None}
        design = model_training.build_design(bundle["panel"], cfg, h, inputs=bundle["inputs"])
        try:
            folds = model_training.make_folds(design, cfg, coverage_start=bundle["inputs"].coverage_start)
        except ValueError as e:
            folds, entry["error"] = [], str(e)
        if not folds:
            entry.update(status="not_measurable", error=entry["error"] or "no coverage-era folds")
        else:
            auc = shuffle_control(model_training.factory_for(cfg, design),
                                  model_training.design_matrix(cfg, design), design.y, design.fwd,
                                  design.dates, design.tickers, folds, seed=0, horizon=h, n_threads=threads)
            entry.update(shuffled_auc=auc, flag=bool(abs(auc - 0.5) > CONTROL_TOLERANCE))
        entry["seconds"] = round(time.perf_counter() - t0, 1)
        entries.append(_clean(entry))

    write_json(run_dir / "config.json",
               _config_payload([cfg], bundle, db_path, universe, horizons, threads, args.tag, "control"))
    write_json(run_dir / "metrics.json", {"kind": "control", "run": run_dir.name, "universe": universe,
                                          "seconds": round(time.perf_counter() - started, 1),
                                          "entries": entries})
    report = [f"# Label-shuffle control: {cfg.name} ({universe})", ""]
    report += _run_header(bundle, db_path, universe, threads)
    report += ["", "Labels and forward returns are permuted across the rows the folds use, then the "
               "config is evaluated as usual. A harness that leaks nothing scores about 0.50; "
               f"anything outside 0.50 +/- {CONTROL_TOLERANCE} is flagged.", "",
               "| Horizon | Shuffled AUC | Flag | Seconds |", "|---|---|---|---|"]
    for e in entries:
        report.append(f"| {e['horizon']}d | {_fmt(e['shuffled_auc'])} | "
                      f"{'LEAK?' if e['flag'] else ('ok' if e['flag'] is False else e['error'])} | {e['seconds']} |")
    (run_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_leaderboard()

    print()
    for e in entries:
        verdict = "not measurable: " + str(e["error"]) if e["status"] != "evaluated" else (
            "FLAGGED: outside 0.50 +/- 0.03" if e["flag"] else "ok")
        print(f"{cfg.name} {e['horizon']}d shuffled AUC {_fmt(e['shuffled_auc'])}  {verdict}")
    print(f"\nwrote {run_dir} in {time.perf_counter() - started:.1f}s")


def cmd_coverage(args) -> None:
    started = time.perf_counter()
    universe = args.universe or "core"
    db, db_path = open_db(args.db)
    bundle = load_bundle(db, db_path, universe)
    panel = bundle["panel"]
    run_dir = new_run_dir("coverage", args.tag)

    years = pd.DatetimeIndex(panel["date"]).year
    table = pd.DataFrame({group: panel[cols].isna().groupby(years).mean().mean(axis=1)
                          for group, cols in FEATURE_GROUPS.items()})
    table.insert(0, "rows", panel.groupby(years).size())
    table.insert(1, "tickers", panel.groupby(years)["ticker"].nunique())
    table.index.name = "year"
    table.to_csv(run_dir / "coverage.csv", float_format="%.4f")

    groups = list(FEATURE_GROUPS)
    lines = [f"# Feature coverage ({universe})", ""] + _run_header(bundle, db_path, universe, _threads(args))
    lines += ["", "Share of missing values per feature group, averaged over the group's columns, by "
              "calendar year of the row. Sentiment is missing by construction before news coverage "
              f"started ({bundle['info']['coverage_start']}).", "",
              "| Year | Rows | Tickers | " + " | ".join(groups) + " |",
              "|---|---|---|" + "---|" * len(groups)]
    for year, row in table.iterrows():
        lines.append(f"| {year} | {int(row['rows']):,} | {int(row['tickers'])} | "
                     + " | ".join(f"{row[g]:.0%}" for g in groups) + " |")
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pd.option_context("display.max_rows", 200, "display.width", 200):
        printable = table.copy()
        for g in groups:
            printable[g] = printable[g].map(lambda v: f"{v:.0%}")
        print(printable.to_string())
    print(f"\nsentiment coverage start: {bundle['info']['coverage_start']}")
    print(f"wrote {run_dir} in {time.perf_counter() - started:.1f}s")


def cmd_leaderboard(args) -> None:
    text = write_leaderboard()
    print(text)
    print(f"wrote {LEADERBOARD_PATH}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p, *, horizons: bool = True, importance: bool = False):
        p.add_argument("--db", default=None, help=f"database path (default {settings.db_path})")
        p.add_argument("--universe", choices=UNIVERSES, default=None,
                       help="training universe (default: the config's)")
        p.add_argument("--threads", type=int, default=None,
                       help=f"thread cap per fit (default PREDICTOR_THREADS={settings.predictor_threads})")
        p.add_argument("--tag", default=None, help="suffix for the output directory")
        if horizons:
            p.add_argument("--horizons", default=None, help="comma-separated, e.g. 5,21")
        if importance:
            p.add_argument("--no-importance", action="store_true",
                           help="skip permutation importance (faster)")

    run = sub.add_parser("run", help="evaluate one config")
    run.add_argument("--config", required=True)
    common(run, importance=True)
    run.set_defaults(func=cmd_run)

    sweep = sub.add_parser("sweep", help="hyperparameter grid on a base config")
    sweep.add_argument("--base", required=True)
    sweep.add_argument("--grid", choices=sorted(GRIDS), default="small")
    common(sweep, importance=True)
    sweep.set_defaults(func=cmd_sweep)

    control = sub.add_parser("control", help="label-shuffle control")
    control.add_argument("--config", required=True)
    common(control)
    control.set_defaults(func=cmd_control)

    baselines = sub.add_parser("baselines", help="prior, momentum and logreg")
    common(baselines)
    baselines.set_defaults(func=cmd_baselines)

    coverage = sub.add_parser("coverage", help="missing-value share per feature group by year")
    common(coverage, horizons=False)
    coverage.set_defaults(func=cmd_coverage)

    leaderboard = sub.add_parser("leaderboard", help="rebuild LEADERBOARD.md")
    leaderboard.set_defaults(func=cmd_leaderboard)
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
