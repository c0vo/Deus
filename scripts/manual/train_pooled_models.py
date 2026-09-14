"""
One-shot production training of the pooled direction model, every horizon.

What the Sunday retrain (orchestrator.scheduler.retrain_models) does, run by
hand: after the first deployment of feature schema v4, after any schema bump,
or whenever the artifacts should be rebuilt before the weekly job would. It
builds the panel once, trains each horizon through StockPredictor.train_pooled
— which evaluates on purged walk-forward folds, saves
storage/models/universal_model_{h}d_v4.joblib (a model, or the prior when the
horizon shows no measurable edge) and inserts a model_metrics row — and prints
the skill table the Telegram report sends.

Heavy: minutes of CPU per horizon on the phone. PREDICTOR_THREADS caps the
threads per fit; run it while the worker is idle, or stop the worker first.
Uses Database(), so it honours DB_PATH, and runs initialize() so an older
database gets the columns the predictions write.

Usage:
    python scripts/manual/train_pooled_models.py
    python scripts/manual/train_pooled_models.py --horizons 5,21
    python scripts/manual/train_pooled_models.py --universe backbone
"""

import argparse
import asyncio
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.logging_config import get_logger
from config.settings import settings
from data.database import Database
from pipeline.features import HORIZONS
from pipeline.model_configs import UNIVERSES
from pipeline.model_report import render_metrics_table_text
from pipeline.model_training import load_panel, training_tickers
from pipeline.predictor import HORIZON_LABELS, StockPredictor

log = get_logger(__name__)


def parse_horizons(text: str) -> list[int]:
    horizons = [int(part) for part in text.split(",") if part.strip()]
    unknown = [h for h in horizons if h not in HORIZONS]
    if unknown:
        raise SystemExit(f"unknown horizons {unknown}; choose from {list(HORIZONS)}")
    return horizons


async def train(horizons: list[int], universe: str) -> int:
    db = Database()
    db.initialize()
    predictor = StockPredictor(db)
    run_id = f"manual-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    started = time.perf_counter()

    tickers = training_tickers(db, universe)
    print(f"Universe {universe}: {len(tickers)} tickers ({', '.join(tickers)})")
    bundle = await asyncio.to_thread(load_panel, db, tickers)
    print(f"Panel: {len(bundle[1]):,} rows in {time.perf_counter() - started:.0f}s")

    failures: list[tuple[str, str]] = []
    for horizon_days in horizons:
        label = HORIZON_LABELS.get(horizon_days, f"{horizon_days}d")
        t0 = time.perf_counter()
        try:
            path, row = await predictor.train_pooled(
                horizon_days, universe=universe, panel_bundle=bundle, run_id=run_id)
            print(f"{label}: {row['status']} -> {path} ({time.perf_counter() - t0:.0f}s)")
        except Exception as e:
            log.error("train_pooled_models.horizon_failed", horizon_days=horizon_days,
                      error=str(e) or repr(e))
            failures.append((label, str(e) or repr(e)))
            print(f"{label}: FAILED after {time.perf_counter() - t0:.0f}s: {e}")

    print()
    print(render_metrics_table_text(db.get_latest_model_metrics()))
    minutes, seconds = divmod(int(time.perf_counter() - started), 60)
    print(f"\nTotal {minutes}m {seconds:02d}s, run_id {run_id}")
    for label, error in failures:
        print(f"Failed {label}: {error}")
    return 1 if failures else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the pooled direction model for every horizon.")
    parser.add_argument("--horizons", default=",".join(str(h) for h in HORIZONS),
                        help="comma-separated horizons in sessions (default: all)")
    parser.add_argument("--universe", choices=UNIVERSES, default=None,
                        help=f"training universe (default PREDICTOR_UNIVERSE={settings.predictor_universe})")
    args = parser.parse_args()
    universe = args.universe or settings.predictor_universe
    sys.exit(asyncio.run(train(parse_horizons(args.horizons), universe)))


if __name__ == "__main__":
    main()
