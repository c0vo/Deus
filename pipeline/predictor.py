"""
Deus — direction predictions (feature schema v4).

One pooled model per horizon serves every ticker. It is trained weekly by the
worker (orchestrator/scheduler.py -> StockPredictor.train_pooled ->
pipeline.model_training) on purged walk-forward folds and saved as
storage/models/universal_model_{h}d_v4.joblib. A horizon that shows no
measurable skill on its confirm folds is saved as the prior instead: it answers
the historical base rate and says so.

What a prediction is made of, whatever the tier:

  1. the model's reading — the calibrated P(up) of the pooled model, or the
     base rate for a no-edge horizon or a ticker with too little history;
  2. fresh context — the in-house classified news for the ticker merged with a
     live web search;
  3. an LLM narrative that interprets (1) against (2).

All three are part of the product. The web search and the narrative run for
every stored prediction — model rows, prior rows and short-history rows — and
are never skipped, templated or reused to save tokens: the model's statistics
are one input the LLM reads alongside the news, not a replacement for it.

model_type on the stored row:
  universal     the pooled model scored this ticker
  prior         no measurable edge (horizon status prior, or < 63 sessions)
  llm_only      no loadable v4 artifact; the LLM makes the call from the
                features and news, as before
  multi_agent   written by predict_with_agents after the Bull/Bear debate

This module never trains inline. Training belongs to the scheduler; an API or
bot path that finds no artifact gets the llm_only call or, with
fast_fallback, a non-persisted TRAINING placeholder.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from config.llm import complete, is_llm_configured, parse_structured
from config.logging_config import get_logger
from config.settings import settings
from config.usage import track_llm
from data.database import Database
# The sector -> ETF map lives in data.watchlist, where the price feed and the
# feature builder read it; re-exported here for older importers.
from data.watchlist import SECTOR_ETF_MAP  # noqa: F401
from pipeline import features, model_training
from pipeline.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION, HORIZONS  # noqa: F401
from pipeline.model_artifact import PooledArtifact
from pipeline.web_search import (
    search_ticker_news,
    summarize_search_results,
    build_ticker_search_query,
    TavilySearchProvider,
)

log = get_logger(__name__)

# The prediction horizons the dashboard renders, in sessions. Defined here rather
# than in api/server.py because the worker's jobs need them too, and importing the
# API module into the worker process just to read a constant pulled the whole
# FastAPI router in with it. Must match features.HORIZONS.
HORIZON_LABELS = {
    5: "5d",
    21: "1m",
    63: "3m",
    252: "1y",
}

# Resolved against this file, not the working directory: the worker, the API and
# the manual scripts are started from different places.
MODELS_DIR = Path(__file__).resolve().parent.parent / "storage" / "models"

# Below this many sessions the pooled model is not asked. Its 63-session windows
# (ret_63d, vol_63, beta_63, skew_63, the rolling highs) are still empty, and a
# row made mostly of missing values is scored off the training rows' averages.
MIN_MODEL_BARS = 63

# A live row whose newest bar is older than this, in calendar days, means the
# price feed has stopped updating; the prediction still runs, and says so in the log.
STALE_FEATURE_DAYS = 5

# Feature readings shown to the narrative when the artifact has no permutation
# importances to rank by, and for every no-edge row.
NARRATIVE_FEATURES = (
    "ret_21d", "ret_63d", "dist_sma50", "rsi_14", "vol_21", "rel_spy_21d", "vix_z63", "sent_7d",
)

# path -> (mtime_ns, validated artifact or None). Module-level because the API
# builds a StockPredictor per request; a per-instance memo would reload the
# joblib file on every call.
_ARTIFACT_MEMO: dict[str, tuple[int, Optional[PooledArtifact]]] = {}


class LlmPrediction(BaseModel):
    """Response schema for the LLM-only predictor (used when no ML model exists)."""

    direction: Literal["UP", "DOWN"]
    confidence: float = Field(ge=0.0, le=1.0)
    narrative: str = Field(
        description="2-3 sentence explanation of the key factors and main risk."
    )


# ── Helpers ──────────────────────────────────────────────────────────────────

def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _snapshot_values(row: pd.Series, names: list[str]) -> dict[str, Optional[float]]:
    """Feature values by name, NaN and missing columns as None."""
    out: dict[str, Optional[float]] = {}
    for name in names:
        value = _finite(row[name]) if name in row.index else None
        out[name] = round(value, 6) if value is not None else None
    return out


def feature_snapshot_json(row: Optional[pd.Series], names: list[str], asof: Optional[str]) -> str:
    """The stored feature_snapshot: plain JSON, NaN as null, plus the as-of session."""
    values = _snapshot_values(row, names) if row is not None else {}
    return json.dumps({**values, "_asof": asof}, allow_nan=False)


def resolve_after_date(asof: Optional[str], horizon_days: int) -> str:
    """`horizon_days` US weekdays after the as-of session (after today, UTC, if unknown).

    Weekdays, not exchange sessions: on a holiday week this lands a day early,
    and grading waits for the h-th stored session anyway (see
    orchestrator.scheduler.grade_prediction), so an early date only means the
    row is checked once before it can be graded.
    """
    if asof:
        start = np.datetime64(str(asof)[:10], "D")
    else:
        start = np.datetime64(datetime.now(timezone.utc).date(), "D")
    return str(np.busday_offset(start, int(horizon_days), roll="forward"))


def parse_timestamp_utc(value: Any) -> Optional[datetime]:
    """A stored timestamp (SQLite CURRENT_TIMESTAMP or ISO 8601) as an aware UTC datetime."""
    if not value:
        return None
    if isinstance(value, datetime):
        stamp = value
    else:
        try:
            stamp = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc)


def made_with_artifact(created_at: Any, trained_at: Any) -> bool:
    """Whether a prediction row was made no earlier than the artifact's training.

    Both stamps have one-second resolution (SQLite CURRENT_TIMESTAMP and an ISO
    trained_at), so a row stamped in the training's own second counts as made
    with it; only a strictly earlier row predates the model. The scheduler's
    refresh rule reads the same way.
    """
    created, trained = parse_timestamp_utc(created_at), parse_timestamp_utc(trained_at)
    return created is not None and trained is not None and created >= trained


class StockPredictor:
    def __init__(self, db: Database):
        self.db = db
        self.models_dir = MODELS_DIR
        self._web_search_cache: dict[str, tuple[float, str]] = {}
        self._web_search_cache_ttl: int = 3600  # 1 hour

    # ── Models ───────────────────────────────────────────────────────────

    def _get_model_path(self, name: str, horizon_days: int = 5) -> Path:
        # The feature schema version is part of the filename so that changing the
        # feature set makes old models simply *unfindable* rather than deleted.
        # Reverting the code reverts the version and the previous models are live
        # again; without this, a rollback would leave nothing to roll back to.
        return Path(self.models_dir) / f"{name}_model_{int(horizon_days)}d_v{FEATURE_SCHEMA_VERSION}.joblib"

    def _load_model(self, ticker: Optional[str], horizon_days: int = 5) -> tuple[Optional[PooledArtifact], str]:
        """The horizon's pooled artifact and the model_type its predictions carry.

        One artifact serves every ticker; `ticker` selects nothing and is kept
        for the callers. Returns (artifact, "universal") for a model horizon,
        (artifact, "prior") for a no-edge horizon, and (None, "llm_only") when
        there is no loadable v4 artifact. Memoized on the file's mtime, so a
        retrain is picked up on the next call. A stale or mismatched file is
        logged and left on disk, never deleted.
        """
        path = self._get_model_path("universal", horizon_days)
        try:
            mtime = path.stat().st_mtime_ns
        except OSError:
            return None, "llm_only"

        key = str(path)
        memo = _ARTIFACT_MEMO.get(key)
        if memo is not None and memo[0] == mtime:
            artifact = memo[1]
        else:
            artifact = self._read_artifact(path, int(horizon_days))
            _ARTIFACT_MEMO[key] = (mtime, artifact)

        if artifact is None:
            return None, "llm_only"
        return artifact, ("universal" if artifact.is_model else "prior")

    @staticmethod
    def _read_artifact(path: Path, horizon_days: int) -> Optional[PooledArtifact]:
        try:
            artifact = joblib.load(path)
        except Exception as e:
            log.warning("predictor.model_load_failed", path=str(path), error=str(e) or repr(e))
            return None

        names = list(getattr(artifact, "feature_names", None) or [])
        unknown = [n for n in names if n not in FEATURE_NAMES]
        version = getattr(artifact, "schema_version", None)
        horizon = getattr(artifact, "horizon", None)
        if (version != FEATURE_SCHEMA_VERSION or unknown or horizon != horizon_days
                or not callable(getattr(artifact, "predict_proba_up", None))):
            # Not deleted: the filename is version-stamped, so a mismatch is a
            # stale artifact rather than a corrupt one, and deleting it would make
            # a revert impossible while /api/markets keeps polling.
            log.warning("predictor.model_schema_mismatch", path=str(path),
                        schema_version=version, expected=FEATURE_SCHEMA_VERSION,
                        horizon=horizon, expected_horizon=horizon_days,
                        unknown_features=unknown[:10])
            return None
        return artifact

    async def train_pooled(self, horizon_days: int, *, config_name: Optional[str] = None,
                           universe: Optional[str] = None, panel_bundle=None,
                           run_id: Optional[str] = None) -> tuple[str, dict]:
        """Train, save and record one horizon's pooled artifact. Returns (path, metrics_row).

        The whole evaluation and fit runs in a worker thread. The artifact is
        written to `<path>.tmp` and moved into place, so a crash mid-write
        leaves the previous artifact untouched. `panel_bundle` (inputs, panel)
        lets the weekly retrain build the panel once for every horizon.
        """
        run_id = run_id or f"weekly-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
        artifact, row, _eval = await asyncio.to_thread(
            model_training.train_horizon, self.db, int(horizon_days),
            config_name=config_name, universe=universe, panel_bundle=panel_bundle,
            n_threads=settings.predictor_threads, run_id=run_id,
        )
        del _eval  # the design matrix inside is the panel's size; nothing reads it here
        path = self._get_model_path("universal", horizon_days)
        await asyncio.to_thread(self._save_artifact, artifact, path)
        await asyncio.to_thread(self.db.insert_model_metrics, row)
        _ARTIFACT_MEMO.pop(str(path), None)
        log.info("predictor.model_saved", horizon_days=int(horizon_days), path=str(path),
                 status=artifact.status, config=artifact.config_name, universe=artifact.universe,
                 rows=artifact.n_rows, tickers=artifact.n_tickers, run_id=run_id)
        return str(path), row

    @staticmethod
    def _save_artifact(artifact: PooledArtifact, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        joblib.dump(artifact, tmp)
        os.replace(tmp, path)

    # ── The model's reading ──────────────────────────────────────────────

    @staticmethod
    def _ml_baseline(artifact: PooledArtifact, live: Optional[tuple[pd.Series, dict]]) -> tuple[str, float]:
        """(status, probability_up) for one ticker — the rule predict and the debate share.

        The pooled model scores the row only when the horizon shipped a model
        and the ticker has MIN_MODEL_BARS sessions; otherwise the answer is the
        base rate, labelled prior.
        """
        bars = int(live[1]["bars"]) if live is not None else 0
        if not artifact.is_model or live is None or bars < MIN_MODEL_BARS:
            return "prior", float(artifact.prior_up_rate)
        values = live[0].reindex(artifact.feature_names).to_numpy(dtype=float)
        return "model", float(artifact.predict_proba_up(values))

    @staticmethod
    def _key_features(row: pd.Series, artifact: PooledArtifact, status: str) -> dict[str, Optional[float]]:
        """The readings the narrative is shown: the model's top features, or a fixed set."""
        names = [n for n in artifact.top_features if n in row.index] if status == "model" else []
        if not names:
            names = [n for n in NARRATIVE_FEATURES if n in row.index]
        readings: dict[str, Optional[float]] = {}
        for name in names:
            value = _finite(row[name])
            readings[name] = round(value, 4) if value is not None else None
        return readings

    @staticmethod
    def _with_contract_fields(cached: dict, artifact: PooledArtifact) -> dict:
        """A cached v4 row in the shape predict() returns fresh ones."""
        out = dict(cached)
        p = _finite(out.get("probability_up"))
        out["edge"] = p - 0.5 if p is not None else None
        out["status"] = "model" if out.get("model_type") == "universal" else "prior"
        out["model_meta"] = artifact.meta()
        if isinstance(out.get("feature_snapshot"), dict):
            out["feature_snapshot"] = json.dumps(out["feature_snapshot"], allow_nan=False)
        return out

    async def _news_context(self, ticker: str, research_callback=None) -> str:
        recent_summaries = await asyncio.to_thread(
            self.db.get_recent_summaries_for_ticker, ticker, 72)
        db_context = "\n".join(f"- {s}" for s in recent_summaries) if recent_summaries else ""
        return await self._enrich_with_web_search(ticker, db_context, research_callback=research_callback)

    # ── Predictions ──────────────────────────────────────────────────────

    async def predict(self, ticker: str, horizon_days: int = 5, fast_fallback: bool = False) -> dict:
        ticker = ticker.upper()
        horizon_days = int(horizon_days)
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        artifact, _label = await asyncio.to_thread(self._load_model, ticker, horizon_days)

        if artifact is None:
            cached = await asyncio.to_thread(
                self.db.get_existing_prediction, ticker, horizon_days, today_str)
            if cached:
                log.info("predictor.cached_prediction", ticker=ticker, horizon_days=horizon_days,
                         model_type=cached.get("model_type"))
                return cached
            if fast_fallback:
                # Nothing is stored: a placeholder must never be graded, or read
                # back by a later caller as a real call.
                return {
                    "ticker": ticker,
                    "horizon_days": horizon_days,
                    "predicted_direction": "TRAINING",
                    "confidence": 0.0,
                    "model_type": "untrained",
                }
            return await self._predict_llm_only(ticker, horizon_days)

        live = await asyncio.to_thread(features.build_live_row, self.db, ticker)
        if live is None:
            return {"error": f"Insufficient price history to build features for {ticker}"}
        row, meta = live

        status, p = await asyncio.to_thread(self._ml_baseline, artifact, live)
        model_type = "universal" if status == "model" else "prior"

        # Today's row is reused only if it is this artifact's kind of call and
        # was made after this artifact was trained. An llm_only row from before
        # the first training, or a row from last week's model, is not an answer
        # this model gave.
        cached = await asyncio.to_thread(
            self.db.get_existing_prediction, ticker, horizon_days, today_str)
        if (cached and cached.get("model_type") == model_type
                and made_with_artifact(cached.get("created_at"), artifact.trained_at)):
            log.info("predictor.cached_prediction", ticker=ticker, horizon_days=horizon_days,
                     model_type=model_type)
            return self._with_contract_fields(cached, artifact)

        if meta["stale_days"] > STALE_FEATURE_DAYS:
            log.warning("predictor.stale_features", ticker=ticker, feature_asof=meta["asof_date"],
                        stale_days=meta["stale_days"])

        short_history = meta["bars"] < MIN_MODEL_BARS
        direction = "UP" if p >= 0.5 else "DOWN"
        confidence = max(p, 1.0 - p)
        model_meta = artifact.meta()

        # Fresh context and an LLM reading for every row type — see the module
        # docstring. The model's numbers go into the prompt, not instead of it.
        news_context = await self._news_context(ticker)
        narrative = await self._generate_narrative(
            ticker, direction, confidence, self._key_features(row, artifact, status),
            horizon_days, news_context,
            status=status, model_meta=model_meta,
            history_bars=meta["bars"] if short_history else None,
        )

        prediction_data = {
            "ticker": ticker,
            "horizon_days": horizon_days,
            "predicted_direction": direction,
            "confidence": confidence,
            "probability_up": p,
            "edge": p - 0.5,
            "model_type": model_type,
            "status": status,
            "feature_asof": meta["asof_date"],
            "feature_snapshot": feature_snapshot_json(
                row, artifact.feature_names or FEATURE_NAMES, meta["asof_date"]),
            "llm_narrative": narrative,
            "resolve_after": resolve_after_date(meta["asof_date"], horizon_days),
            "model_meta": model_meta,
            "actual_direction": None,
            "actual_change_pct": None,
            "is_correct": None,
        }
        prediction_data["id"] = await asyncio.to_thread(self.db.insert_prediction, prediction_data)
        log.info("predictor.prediction_made", ticker=ticker, horizon_days=horizon_days,
                 model_type=model_type, probability_up=round(p, 4),
                 feature_asof=meta["asof_date"], bars=meta["bars"])
        return prediction_data

    async def _predict_llm_only(self, ticker: str, horizon_days: int) -> dict:
        """No v4 artifact for the horizon: the LLM makes the call from features and news."""
        live = await asyncio.to_thread(features.build_live_row, self.db, ticker)
        if live is None:
            return {"error": f"Insufficient price history to build features for {ticker}"}
        row, meta = live

        readings = {k: v for k, v in _snapshot_values(row, FEATURE_NAMES).items() if v is not None}
        news_context = await self._news_context(ticker)
        llm_result = await self._generate_narrative_with_confidence(
            ticker, readings, horizon_days, news_context)
        direction = llm_result.get("direction", "UP")
        confidence = llm_result.get("confidence", 0.5)

        prediction_data = {
            "ticker": ticker,
            "horizon_days": horizon_days,
            "predicted_direction": direction,
            "confidence": confidence,
            "probability_up": None,
            "edge": None,
            "model_type": "llm_only",
            "status": None,
            "feature_asof": meta["asof_date"],
            "feature_snapshot": feature_snapshot_json(row, FEATURE_NAMES, meta["asof_date"]),
            "llm_narrative": llm_result.get("narrative", f"LLM predicts {direction} for {ticker}."),
            "resolve_after": resolve_after_date(meta["asof_date"], horizon_days),
            "model_meta": None,
            "actual_direction": None,
            "actual_change_pct": None,
            "is_correct": None,
        }
        prediction_data["id"] = await asyncio.to_thread(self.db.insert_prediction, prediction_data)
        log.info("predictor.prediction_made", ticker=ticker, horizon_days=horizon_days,
                 model_type="llm_only", feature_asof=meta["asof_date"])
        return prediction_data

    def _ml_prediction(self, artifact: Optional[PooledArtifact],
                       live: Optional[tuple[pd.Series, dict]]) -> dict:
        """The quantitative baseline handed to the debate."""
        row = live[0] if live is not None else None
        asof = live[1]["asof_date"] if live is not None else None
        unknown = {
            "predicted_direction": "UNKNOWN",
            "confidence": 0.0,
            "probability_up": None,
            "edge": None,
            # The arena says "no model trained" off this rather than printing
            # UNKNOWN at 0% — every horizon reads llm_only after a schema bump
            # until the scheduler has retrained it.
            "model_type": "llm_only",
            "status": None,
            "feature_asof": asof,
            "feature_snapshot": feature_snapshot_json(row, FEATURE_NAMES, asof) if row is not None else "{}",
            "model_meta": None,
        }
        if artifact is None:
            return unknown
        try:
            status, p = self._ml_baseline(artifact, live)
        except Exception as e:
            log.warning("predictor.model_score_failed", error=str(e) or repr(e),
                        horizon_days=artifact.horizon)
            return unknown
        names = artifact.feature_names or FEATURE_NAMES
        return {
            "predicted_direction": "UP" if p >= 0.5 else "DOWN",
            "confidence": max(p, 1.0 - p),
            "probability_up": p,
            "edge": p - 0.5,
            "model_type": "universal" if status == "model" else "prior",
            "status": status,
            "feature_asof": asof,
            "feature_snapshot": feature_snapshot_json(row, names, asof) if row is not None else "{}",
            "model_meta": artifact.meta(),
        }

    async def predict_with_agents(self, ticker: str, horizon_days: int = 5, progress_callback=None, debate_chunk_callback=None, research_callback=None) -> dict:
        """Generate a prediction with the full LangGraph multi-agent debate.

        The pooled model's reading for the horizon is the debate's quantitative
        baseline; the debate itself is the interpretation, so no separate
        narrative is written here. Never trains: a horizon without an artifact
        debates without a baseline until the scheduler has trained one.
        """
        ticker = ticker.upper()
        horizon_days = int(horizon_days)

        # 1. Quantitative baseline (ML)
        artifact, _label = await asyncio.to_thread(self._load_model, ticker, horizon_days)
        live = await asyncio.to_thread(features.build_live_row, self.db, ticker)
        ml_prediction = await asyncio.to_thread(self._ml_prediction, artifact, live)

        # 2. Past lessons and recent news
        past_lessons = {"ticker_lessons": [], "sector_lessons": [], "market_lessons": []}
        if hasattr(self.db, "get_relevant_reflections"):
            past_lessons = await asyncio.to_thread(
                lambda: self.db.get_relevant_reflections(ticker, limit=5))
        elif hasattr(self.db, "get_recent_reflections"):
            # Fallback for backward compatibility
            reflections = await asyncio.to_thread(
                lambda: self.db.get_recent_reflections(ticker, limit=3))
            if reflections:
                past_lessons["ticker_lessons"] = [
                    {"lesson_learned": r, "was_successful": True, "date": ""}
                    for r in reflections
                ]

        news_context = await self._news_context(ticker, research_callback=research_callback)

        # 3. Run AdvisoryGraph. Imported here: pipeline.agents reads this
        # module's constants, and a module-level import would make that a cycle.
        from pipeline.agents import AdvisoryGraph
        graph = AdvisoryGraph(self.db, progress_callback=progress_callback, debate_chunk_callback=debate_chunk_callback)
        final_state = await graph.run(ticker, ml_prediction, past_lessons, news_context)

        # 4. Save to DB and cache
        asof = ml_prediction["feature_asof"]
        prediction_data = {
            "ticker": ticker,
            "predicted_direction": ml_prediction["predicted_direction"],
            "confidence": ml_prediction["confidence"],
            "probability_up": ml_prediction["probability_up"],
            "horizon_days": horizon_days,
            "model_type": "multi_agent",
            "feature_asof": asof,
            "feature_snapshot": ml_prediction["feature_snapshot"],
            "llm_narrative": final_state.get("final_advisory", "Error generating advisory."),
            "resolve_after": resolve_after_date(asof, horizon_days),
            "actual_direction": None,
            "actual_change_pct": None,
            "is_correct": None
        }
        await asyncio.to_thread(self.db.insert_prediction, prediction_data)

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if hasattr(self.db, "set_cached_advisory"):
            await asyncio.to_thread(self.db.set_cached_advisory, ticker, today, json.dumps(final_state))

        return final_state

    # ── Context ──────────────────────────────────────────────────────────

    async def _enrich_with_web_search(self, ticker: str, db_news_context: str, research_callback=None) -> str:
        """Enrich DB news context with live web search results for a ticker.

        Merges in-house classified news with real-time web results so the
        debate agents have the freshest possible information.  Falls back to
        DB-only gracefully if the search provider is unconfigured, fails, or
        returns nothing.

        If *research_callback* is provided, it receives structured events
        during the search so the frontend can animate the research process:
          - ("research_start", {"ticker": ..., "query": ...})
          - ("research_source", {"title": ..., "url": ..., "domain": ..., "index": ..., "total": ...})
          - ("research_summarizing", {"message": ...})
          - ("research_complete", {"sources_found": ...})
        """
        async def _emit(event_type: str, data: dict):
            if research_callback:
                try:
                    await research_callback(event_type, data)
                except Exception:
                    pass  # never let callback failures break the pipeline

        query = build_ticker_search_query(ticker)

        # Check in-memory cache first (keyed by ticker + hour)
        cache_key = f"{ticker}:{datetime.now(timezone.utc).strftime('%Y%m%d%H')}"
        cached = self._web_search_cache.get(cache_key)
        cached_age = datetime.now(timezone.utc).timestamp() - cached[0] if cached else self._web_search_cache_ttl + 1
        if cached and cached_age < self._web_search_cache_ttl:
            log.info("web_search.cache_hit", ticker=ticker)
            web_context = cached[1]
            # Replay minimal research event for visual feedback even from cache
            await _emit("research_start", {"ticker": ticker, "query": query})
            await _emit("research_complete", {"sources_found": -1, "cached": True})
        else:
            await _emit("research_start", {"ticker": ticker, "query": query})

            # Fetch fresh results
            try:
                raw_results = await search_ticker_news(ticker, max_results=settings.web_search_max_results)
            except Exception as exc:
                log.warning("web_search.failed", ticker=ticker, error=str(exc))
                raw_results = []

            total = len(raw_results)

            # Stream each source as it's discovered
            for i, r in enumerate(raw_results):
                await _emit("research_source", {
                    "title": r.title,
                    "url": r.url,
                    "domain": r.source or TavilySearchProvider._extract_domain(r.url),
                    "index": i,
                    "total": total,
                })

            if raw_results:
                await _emit("research_summarizing", {
                    "message": f"DeepSeek is analyzing {total} source{'s' if total != 1 else ''}..."
                })
                web_context = await summarize_search_results(ticker, raw_results, db=self.db)
                if web_context:
                    self._web_search_cache[cache_key] = (datetime.now(timezone.utc).timestamp(), web_context)
            else:
                web_context = ""

            await _emit("research_complete", {"sources_found": total})

        return self._merge_db_and_web(db_news_context, web_context)

    @staticmethod
    def _merge_db_and_web(db_news: str, web_context: str) -> str:
        """Merge DB news and web results into a single labelled context block."""
        parts: list[str] = []

        if db_news:
            parts.append(
                "========================================\n"
                "IN-HOUSE NEWS (classified & ranked)\n"
                "========================================\n"
                f"{db_news}"
            )

        if web_context:
            parts.append(
                "========================================\n"
                "LIVE WEB SEARCH RESULTS\n"
                "========================================\n"
                f"{web_context}"
            )

        if not parts:
            return "No recent news context available."

        return "\n\n".join(parts)

    # ── Narratives ───────────────────────────────────────────────────────

    @staticmethod
    def _model_stats_sentence(ticker: str, p: float, horizon_days: int, meta: dict) -> str:
        auc, skill, base = meta.get("auc"), meta.get("brier_skill"), meta.get("base_rate")
        skill_parts = []
        if auc is not None:
            skill_parts.append(f"walk-forward AUC {auc:.2f}")
        if skill is not None:
            skill_parts.append(f"Brier skill {skill:+.3f}")
        detail = "calibrated"
        if skill_parts:
            detail += "; " + ", ".join(skill_parts)
        if base is not None:
            detail += f"; base rate {base:.0%}"
        return (f"The pooled walk-forward model gives {ticker} a {p:.0%} probability of closing "
                f"higher over the next {horizon_days} trading days ({detail}).")

    @staticmethod
    def _no_edge_sentence(ticker: str, horizon_days: int, meta: dict,
                          history_bars: Optional[int]) -> str:
        label = HORIZON_LABELS.get(horizon_days, f"{horizon_days}d")
        auc, low, high, base = (meta.get("auc"), meta.get("auc_ci_low"),
                                meta.get("auc_ci_high"), meta.get("base_rate"))
        if history_bars is not None:
            evidence = f"only {history_bars} sessions of price history"
        elif auc is not None and low is not None and high is not None:
            evidence = f"walk-forward AUC {auc:.2f}, 90% CI {low:.2f}-{high:.2f}"
        elif auc is not None:
            evidence = f"walk-forward AUC {auc:.2f}"
        else:
            evidence = "its walk-forward skill could not be measured yet"
        base_text = (f"the historical base rate is {base:.0%} of {horizon_days}-session windows closing higher"
                     if base is not None else "no historical base rate is available")
        return (f"The pooled walk-forward model shows no measurable edge for {ticker} at the {label} "
                f"horizon ({evidence}); {base_text}. Interpret the recent news and the feature readings "
                f"for this horizon, say plainly that the model has no edge here, and do not present the "
                f"base rate as a model call.")

    async def _generate_narrative(self, ticker: str, direction: str, confidence: float, features: dict,
                                  horizon_days: int, news_context: str = "", *,
                                  status: Optional[str] = None, model_meta: Optional[dict] = None,
                                  history_bars: Optional[int] = None) -> str:
        """Plain-English reading of the model's call against the news.

        `status` "model" frames the calibrated probability and its walk-forward
        skill; "prior" says the horizon has no measurable edge (or, with
        `history_bars`, that the ticker is too new to model) and asks for an
        interpretation of the news without dressing the base rate up as a call.
        No status keeps the original prompt.
        """
        model_name = settings.model_predictor_narrative
        meta = model_meta or {}
        p = confidence if direction == "UP" else 1.0 - confidence
        base = meta.get("base_rate")
        if status == "prior":
            fallback = ("No measurable model edge at this horizon"
                        + (f"; the historical base rate is {base:.0%} up." if base is not None else "."))
        else:
            fallback = f"Model predicts {direction} based on current technical and sentiment features."
        try:
            if not is_llm_configured() or not model_name:
                return fallback

            rules = "Do NOT use emojis. Do NOT use Markdown or HTML. Plain professional English only."
            if status == "model":
                prompt = (
                    f"You are a senior financial analyst. "
                    f"{self._model_stats_sentence(ticker, p, horizon_days, meta)}\n\n"
                    f"Key features behind this reading:\n{json.dumps(features, indent=2)}\n\n"
                    f"Recent News Context:\n{news_context}\n\n"
                    f"Write a concise 2-3 sentence narrative that:\n"
                    f"1. Names the 1-2 MOST impactful features or news items behind the {direction} lean\n"
                    f"2. Mentions any contradictory signals (e.g., bullish news but bearish technicals)\n"
                    f"3. States the key risk to the prediction\n"
                    f"{rules}"
                )
            elif status == "prior":
                prompt = (
                    f"You are a senior financial analyst. "
                    f"{self._no_edge_sentence(ticker, horizon_days, meta, history_bars)}\n\n"
                    f"Feature readings for {ticker}:\n{json.dumps(features, indent=2)}\n\n"
                    f"Recent News Context:\n{news_context}\n\n"
                    f"Write a concise 2-3 sentence narrative that:\n"
                    f"1. Names the 1-2 news items or feature readings that matter most over this horizon\n"
                    f"2. States plainly that the model has no measurable edge here\n"
                    f"3. States the key risk or catalyst to watch\n"
                    f"{rules}"
                )
            else:
                prompt = (
                    f"You are a senior financial analyst. The ML model predicts {ticker} will go {direction} "
                    f"over the next {horizon_days} day(s) with {confidence*100:.1f}% confidence.\n\n"
                    f"Key features driving this prediction:\n{json.dumps(features, indent=2)}\n\n"
                    f"Recent News Context:\n{news_context}\n\n"
                    f"Write a concise 2-3 sentence narrative that:\n"
                    f"1. Names the 1-2 MOST impactful features or news items driving the {direction} call\n"
                    f"2. Mentions any contradictory signals (e.g., bullish news but bearish technicals)\n"
                    f"3. States the key risk to the prediction\n"
                    f"{rules}"
                )
            with track_llm(self.db, model_name, "ml_narrative") as u:
                u.response = resp = await complete(model=model_name, prompt=prompt)
            return resp.text.strip()
        except Exception as e:
            log.error("predictor.narrative_failed", ticker=ticker, error=str(e) or repr(e))
            return fallback

    async def _generate_narrative_with_confidence(self, ticker: str, features: dict, horizon_days: int, news_context: str = "") -> dict:
        """When no ML model exists, ask the LLM to act as the full predictor."""
        model_name = settings.model_predictor_narrative
        try:
            if not is_llm_configured() or not model_name:
                return {"direction": "UP", "confidence": 0.5, "narrative": "Insufficient data for analysis."}

            prompt = (
                f"You are a senior quantitative analyst specializing in equity prediction. "
                f"Predict whether {ticker} will go UP or DOWN over the next {horizon_days} day(s) "
                f"based on the technical features, sentiment signals, and news context below.\n\n"
                f"=== FEATURES ===\n{json.dumps(features, indent=2)}\n\n"
                f"=== RECENT NEWS ===\n{news_context}\n\n"
                f"CONFIDENCE CALIBRATION:\n"
                f"- 0.50-0.55: Very uncertain (mixed/weak signals, or insufficient data)\n"
                f"- 0.55-0.65: Moderate lean (one or two signals point clearly, others are neutral)\n"
                f"- 0.65-0.75: Reasonably confident (multiple signals align, news supports)\n"
                f"- 0.75-0.85: High confidence (strong alignment across sentiment, technicals, and news)\n"
                f"- 0.85+: Only if there is overwhelming, unambiguous evidence (rare)\n\n"
                f"Fill in the fields of the required response schema."
            )

            # The substitute predictor when no ML model exists — deliberately
            # left on the stronger model, but now visible in the cost log.
            with track_llm(self.db, model_name, "llm_only_prediction") as u:
                u.response = resp = await complete(
                    model=model_name,
                    prompt=prompt,
                    schema=LlmPrediction,
                )
            parsed = resp.parsed if isinstance(resp.parsed, LlmPrediction) else parse_structured(resp.text, LlmPrediction)
            return {
                "direction": parsed.direction,
                # Clamp confidence to reasonable range
                "confidence": max(0.5, min(0.85, parsed.confidence)),
                "narrative": parsed.narrative,
            }
        except Exception as e:
            log.error("predictor.llm_prediction_failed", ticker=ticker, error=str(e) or repr(e))
            return {"direction": "UP", "confidence": 0.5, "narrative": f"Unable to generate LLM analysis for {ticker}."}
