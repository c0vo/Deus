"""
Tests for the pooled direction predictor (feature schema v4).

Everything runs against a temporary SQLite file filled with synthetic daily
bars: no network, no live LLM. `complete` and `is_llm_configured` are patched as
pipeline.predictor imports them, and `_enrich_with_web_search` on the class.

The load-bearing assertions are about the product contract, not only the
numbers: every stored prediction — a model row, a no-edge prior row, a
short-history row — still runs the web search and an LLM narrative, with the
model's statistics in the prompt.

Training uses a tiny HistGradientBoosting config registered in place of the
production ones, so a full train takes a couple of seconds.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import joblib
import numpy as np
import pandas as pd
import pytest

from config.settings import settings
from data.database import Database
from data.watchlist import SECTOR_ETFS, TRAINING_BACKBONE
from orchestrator.scheduler import (PREDICTION_REFRESH_DAYS, PipelineOrchestrator, grade_prediction,
                                    has_price_history)
from pipeline import features, model_configs, model_training
from pipeline import predictor as predictor_module
from pipeline.model_artifact import PooledArtifact
from pipeline.predictor import FEATURE_SCHEMA_VERSION, StockPredictor, resolve_after_date
from tests.conftest import make_llm_response

N_SESSIONS = 700
START = "2023-06-01"
TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE", "SIG"]
SHORT = "NEWCO"          # 40 sessions: below the pooled model's 63-session minimum
NARRATOR = "test/narrator"

TINY = model_configs.ModelConfig(
    name="test_tiny",
    feature_groups=("price", "market", "calendar", "context"),
    params=dict(learning_rate=0.1, max_iter=40, max_leaf_nodes=7, min_samples_leaf=40,
                early_stopping=False, random_state=0),
    n_folds=3, test_days=60, min_train_days=250, universe="tracked",
)

CONTRACT_KEYS = {
    "ticker", "horizon_days", "predicted_direction", "confidence", "probability_up", "edge",
    "model_type", "status", "feature_asof", "feature_snapshot", "llm_narrative",
    "resolve_after", "model_meta",
}
MODEL_META_KEYS = {"auc", "auc_ci_low", "auc_ci_high", "brier_skill", "base_rate",
                   "trained_at", "config", "universe", "n_tickers"}


# ── Synthetic data ───────────────────────────────────────────────────────────

def _bar_rows(n: int, seed: int, *, price: float = 100.0, vol: float = 0.02,
              signal: bool = False) -> list[dict]:
    """A business-day random walk as price_history rows.

    With `signal`, each session drifts in the direction of the trailing
    21-session return, so ret_21d predicts the next five sessions' direction.
    """
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(START, periods=n)
    noise = rng.normal(0.0003, vol, n)
    log_close = np.empty(n)
    level = np.log(price)
    for i in range(n):
        drift = 0.004 * np.sign(log_close[i - 1] - log_close[i - 22]) if signal and i >= 22 else 0.0
        level += noise[i] + drift
        log_close[i] = level
    close = np.exp(log_close)
    open_ = close * np.exp(rng.normal(0.0, vol / 3, n))
    volume = rng.integers(1_000_000, 5_000_000, n)
    return [{"date": d.strftime("%Y-%m-%d"), "open": float(o), "high": float(max(o, c) * 1.004),
             "low": float(min(o, c) * 0.996), "close": float(c), "volume": int(v)}
            for d, o, c, v in zip(days, open_, close, volume)]


@pytest.fixture(scope="module")
def bar_rows() -> dict[str, list[dict]]:
    rows = {t: _bar_rows(N_SESSIONS, i, signal=(t == "SIG")) for i, t in enumerate(TICKERS)}
    rows[SHORT] = _bar_rows(40, 77)
    rows["^GSPC"] = _bar_rows(N_SESSIONS, 100, price=4000.0, vol=0.010)
    rows["^VIX"] = _bar_rows(N_SESSIONS, 101, price=18.0, vol=0.050)
    rows["^TNX"] = _bar_rows(N_SESSIONS, 102, price=4.0, vol=0.020)
    return rows


def _make_db(path: Path, bar_rows: dict[str, list[dict]]) -> Database:
    database = Database(db_path=str(path))
    database.initialize()
    for symbol, rows in bar_rows.items():
        database.upsert_price_history(symbol, rows)
    with database.connection() as conn:
        # A stored sector keeps get_ticker_sector off yfinance.
        for symbol in TICKERS + [SHORT]:
            conn.execute("INSERT OR REPLACE INTO ticker_info (ticker, sector) VALUES (?, ?)",
                         (symbol, "Technology"))
    database.set_config("tracked_tickers", json.dumps(TICKERS + [SHORT]))
    return database


@contextmanager
def _tiny_production_config():
    with patch.dict(model_configs.CONFIGS, {TINY.name: TINY}), \
            patch.dict(model_configs.PRODUCTION, {h: TINY.name for h in features.HORIZONS}), \
            patch.object(settings, "predictor_universe", "tracked"), \
            patch.object(settings, "predictor_threads", 1):
        yield


@pytest.fixture(autouse=True)
def tiny_config():
    with _tiny_production_config():
        yield


@pytest.fixture
def db(tmp_path, bar_rows) -> Database:
    return _make_db(tmp_path / "predictor.db", bar_rows)


@pytest.fixture
def predictor(db, tmp_path) -> StockPredictor:
    p = StockPredictor(db)
    p.models_dir = tmp_path / "models"
    return p


@pytest.fixture(scope="module")
def trained(tmp_path_factory, bar_rows) -> dict:
    """One 5d training run on the synthetic panel, shared by the tests that only read it."""
    root = tmp_path_factory.mktemp("trained")
    database = _make_db(root / "train.db", bar_rows)
    with _tiny_production_config():
        artifact, row, eval_dict = model_training.train_horizon(
            database, 5, universe="tracked", n_threads=1, run_id="test-run")
    return {"artifact": artifact, "row": row, "eval": eval_dict, "db": database}


@pytest.fixture(scope="module")
def noise_trained(tmp_path_factory, bar_rows) -> dict:
    """The same training on a panel with no planted signal anywhere."""
    root = tmp_path_factory.mktemp("noise")
    noise_rows = {s: r for s, r in bar_rows.items() if s != "SIG"}
    database = _make_db(root / "noise.db", noise_rows)
    database.set_config("tracked_tickers", json.dumps([t for t in TICKERS if t != "SIG"]))
    with _tiny_production_config():
        artifact, row, _ = model_training.train_horizon(
            database, 5, universe="tracked", n_threads=1, run_id="noise-run")
    return {"artifact": artifact, "row": row}


def _model_artifact(database: Database, horizon: int = 5) -> PooledArtifact:
    """A "model" artifact fitted directly, so the model path does not hinge on the ship rule."""
    tickers = model_training.training_tickers(database, "tracked")
    inputs, panel = model_training.load_panel(database, tickers)
    design = model_training.build_design(panel, TINY, horizon, inputs=inputs)
    model = model_training.factory_for(TINY, design)()
    model.fit(design.X, design.y.astype(int))
    return PooledArtifact(
        model=model, calibrator=None, feature_names=list(design.columns),
        feature_index=list(design.feature_index), categorical_idx=list(design.cat_idx),
        horizon=horizon, schema_version=FEATURE_SCHEMA_VERSION, status="model",
        prior_up_rate=float(design.y.mean()), trained_at="2020-01-01T00:00:00+00:00",
        train_end="2026-01-01", config_name=TINY.name, universe="tracked",
        n_tickers=len(tickers), n_rows=design.n_rows,
        metrics={"auc_mean": 0.56, "auc_ci_low": 0.53, "auc_ci_high": 0.59,
                 "brier_skill_mean": 0.004, "decile_spread_mean": 0.01},
        top_features=["ret_21d", "vol_21", "rsi_14"],
    )


def _save(predictor: StockPredictor, artifact: PooledArtifact, horizon: int = 5) -> Path:
    path = predictor._get_model_path("universal", horizon)
    predictor._save_artifact(artifact, path)
    return path


@contextmanager
def _llm_and_web():
    """Patch the narrative LLM and the web search; yields (complete_mock, web_mock)."""
    complete = AsyncMock(return_value=make_llm_response("A plain-English reading."))
    web = AsyncMock(return_value="IN-HOUSE NEWS\n- something happened")
    with patch.object(predictor_module, "complete", complete), \
            patch.object(predictor_module, "is_llm_configured", return_value=True), \
            patch.object(predictor_module.settings, "model_predictor_narrative", NARRATOR), \
            patch.object(StockPredictor, "_enrich_with_web_search", web):
        yield complete, web


def _no_nan_json(text: str) -> dict:
    def reject(token):
        raise ValueError(f"non-finite JSON token {token}")
    return json.loads(text, parse_constant=reject)


# ── Universe ─────────────────────────────────────────────────────────────────

def test_training_tickers_exclude_crypto_indices_and_market_inputs():
    database = MagicMock()
    database.get_tracked_tickers.return_value = ["nvda", "BTC-USD", "^VIX", "SPY", "^GSPC", "QQQM", "NVDA"]

    assert model_training.training_tickers(database, "tracked") == ["NVDA", "QQQM"]

    core = model_training.training_tickers(database, "core")
    assert core == sorted(set(core))
    assert set(SECTOR_ETFS) <= set(core)
    assert {"NVDA", "QQQM", "AAPL"} <= set(core)
    assert not any(s.endswith("-USD") or s.startswith("^") for s in core)
    assert "SPY" not in core

    backbone = model_training.training_tickers(database, "backbone")
    assert set(TRAINING_BACKBONE) <= set(backbone) and set(core) <= set(backbone)

    with pytest.raises(ValueError):
        model_training.training_tickers(database, "everything")


# ── Training ─────────────────────────────────────────────────────────────────

def test_train_horizon_returns_a_v4_artifact_and_a_metrics_row(trained):
    artifact, row = trained["artifact"], trained["row"]

    assert artifact.schema_version == FEATURE_SCHEMA_VERSION == 4
    assert artifact.feature_names and set(artifact.feature_names) <= set(features.FEATURE_NAMES)
    assert artifact.feature_index == [features.FEATURE_NAMES.index(n) for n in artifact.feature_names]
    assert artifact.status in ("model", "prior")
    assert 0.0 < artifact.prior_up_rate < 1.0
    assert artifact.horizon == 5 and artifact.config_name == TINY.name
    assert artifact.n_tickers == len(TICKERS) + 1 and artifact.n_rows > 0

    columns = set(Database._MODEL_METRICS_COLUMNS) - {"created_at"}
    assert columns <= set(row)
    assert row["status"] == artifact.status and row["horizon_days"] == 5
    assert row["run_id"] == "test-run" and row["schema_version"] == 4
    assert row["auc_mean"] is not None and row["folds_json"]

    # The row is insertable as-is, and reads back as the latest for its horizon.
    trained["db"].insert_model_metrics(row)
    latest = trained["db"].get_latest_model_metrics()
    assert [r["horizon_days"] for r in latest] == [5]
    assert latest[0]["config_json"]["name"] == TINY.name


def test_pure_noise_panel_ships_the_prior(noise_trained):
    artifact = noise_trained["artifact"]
    assert artifact.status == "prior"
    assert artifact.model is None
    # A prior artifact answers its base rate whatever the row says.
    assert artifact.predict_proba_up(np.zeros(len(features.FEATURE_NAMES))) == artifact.prior_up_rate
    # The evaluation that explains the decision is kept.
    assert noise_trained["row"]["status"] == "prior"
    assert noise_trained["row"]["auc_mean"] is not None


def test_horizon_without_enough_history_is_saved_as_prior_with_a_note(db):
    artifact, row, eval_dict = model_training.train_horizon(db, 252, universe="tracked", n_threads=1)
    assert eval_dict["status"] == "not_measurable"
    assert artifact.status == "prior" and artifact.model is None
    assert row["status"] == "prior" and row["auc_mean"] is None
    assert row["config_json"]["note"]


def test_hgb_ignores_columns_with_no_observed_training_value():
    # scikit-learn 1.9 raises on a numeric column that is NaN on every training row;
    # every early walk-forward fold has such columns (dark pool, regime, sentiment).
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 4))
    y = (X[:, 0] > 0).astype(int)
    X[:, 2] = np.nan
    X[:, 3] = rng.integers(0, 3, 400)
    model = model_training.ObservedColumnsHGB(params={"max_iter": 20}, categorical_idx=(2, 3))
    model.fit(X, y)
    assert list(model.columns_) == [0, 1, 3]
    proba = model.predict_proba(X[:5])
    assert proba.shape == (5, 2) and np.all((proba >= 0) & (proba <= 1))


async def test_train_pooled_saves_the_artifact_and_load_model_round_trips_it(predictor, db):
    path, row = await predictor.train_pooled(5, universe="tracked")

    assert Path(path).name == "universal_model_5d_v4.joblib"
    assert Path(path).exists() and not Path(path + ".tmp").exists()
    artifact, label = predictor._load_model("AAA", 5)
    assert isinstance(artifact, PooledArtifact)
    assert label == ("universal" if artifact.status == "model" else "prior")
    assert artifact.status == row["status"]
    assert [r["horizon_days"] for r in db.get_latest_model_metrics()] == [5]


def test_load_model_without_an_artifact_is_llm_only(predictor):
    assert predictor._load_model("AAA", 5) == (None, "llm_only")


def test_load_model_rejects_an_artifact_from_another_schema_without_deleting_it(predictor, trained):
    stale = PooledArtifact(**{**trained["artifact"].__dict__, "schema_version": 3})
    path = _save(predictor, stale)
    assert predictor._load_model("AAA", 5) == (None, "llm_only")
    assert path.exists()


# ── Predictions ──────────────────────────────────────────────────────────────

async def test_prior_prediction_is_web_grounded_and_narrated(predictor, db, noise_trained):
    artifact = noise_trained["artifact"]
    _save(predictor, artifact)

    with _llm_and_web() as (complete, web):
        pred = await predictor.predict("AAA", 5)

    assert CONTRACT_KEYS <= set(pred)
    assert pred["model_type"] == "prior" and pred["status"] == "prior"
    assert pred["probability_up"] == artifact.prior_up_rate
    assert pred["confidence"] == max(pred["probability_up"], 1 - pred["probability_up"])
    assert pred["edge"] == pred["probability_up"] - 0.5
    assert set(pred["model_meta"]) == MODEL_META_KEYS

    snapshot = _no_nan_json(pred["feature_snapshot"])
    assert snapshot["_asof"] == pred["feature_asof"]
    assert set(snapshot) - {"_asof"} == set(artifact.feature_names)

    # The product contract: web search and the LLM run for a no-edge row too.
    web.assert_awaited_once()
    complete.assert_awaited_once()
    prompt = complete.await_args.kwargs["prompt"]
    assert "no measurable edge" in prompt
    assert "IN-HOUSE NEWS" in prompt
    assert pred["llm_narrative"] == "A plain-English reading."

    stored = db.get_recent_predictions("AAA", 1)[0]
    assert stored["model_type"] == "prior"
    assert stored["probability_up"] == pytest.approx(artifact.prior_up_rate)
    assert stored["feature_asof"] == pred["feature_asof"]


async def test_model_prediction_is_web_grounded_and_narrated(predictor, db):
    artifact = _model_artifact(db)
    _save(predictor, artifact)

    with _llm_and_web() as (complete, web):
        pred = await predictor.predict("SIG", 5)

    assert CONTRACT_KEYS <= set(pred)
    assert pred["model_type"] == "universal" and pred["status"] == "model"
    assert 0.02 <= pred["probability_up"] <= 0.98
    assert pred["confidence"] >= 0.5
    assert pred["edge"] == pred["probability_up"] - 0.5
    assert pred["predicted_direction"] == ("UP" if pred["probability_up"] >= 0.5 else "DOWN")
    _no_nan_json(pred["feature_snapshot"])

    web.assert_awaited_once()
    complete.assert_awaited_once()
    prompt = complete.await_args.kwargs["prompt"]
    assert "probability of closing higher" in prompt
    assert "walk-forward AUC 0.56" in prompt
    assert "ret_21d" in prompt


async def test_short_history_ticker_gets_a_narrated_prior_under_a_model_artifact(predictor, db):
    _save(predictor, _model_artifact(db))

    with _llm_and_web() as (complete, web):
        pred = await predictor.predict(SHORT, 5)

    assert pred["model_type"] == "prior" and pred["status"] == "prior"
    web.assert_awaited_once()
    complete.assert_awaited_once()
    assert "only 40 sessions of price history" in complete.await_args.kwargs["prompt"]


async def test_todays_row_is_reused_only_when_this_artifact_made_it(predictor, db, noise_trained):
    artifact = noise_trained["artifact"]
    _save(predictor, artifact)
    # An llm_only row from earlier today is not an answer this model gave. Stamped
    # at the start of the day so it cannot share a second with the fresh row.
    old_id = db.insert_prediction({"ticker": "AAA", "predicted_direction": "UP", "confidence": 0.7,
                                   "horizon_days": 5, "model_type": "llm_only",
                                   "feature_snapshot": "{}", "llm_narrative": "old",
                                   "resolve_after": "2099-01-01"})
    with db.connection() as conn:
        conn.execute("UPDATE predictions SET created_at = datetime('now', 'start of day') WHERE id = ?",
                     (old_id,))

    with _llm_and_web() as (complete, _web):
        first = await predictor.predict("AAA", 5)
        second = await predictor.predict("AAA", 5)

    assert first["model_type"] == "prior"
    assert complete.await_count == 1
    assert second["id"] == first["id"]
    assert second["status"] == "prior" and second["model_meta"] == artifact.meta()
    assert isinstance(second["feature_snapshot"], str)


async def test_fast_fallback_without_an_artifact_is_an_unstored_placeholder(predictor, db):
    with _llm_and_web() as (complete, web):
        pred = await predictor.predict("AAA", 21, fast_fallback=True)

    assert pred == {"ticker": "AAA", "horizon_days": 21, "predicted_direction": "TRAINING",
                    "confidence": 0.0, "model_type": "untrained"}
    assert db.get_recent_predictions("AAA", 5) == []
    complete.assert_not_awaited()
    web.assert_not_awaited()


async def test_predict_with_agents_hands_the_debate_the_ml_contract(predictor, db, noise_trained):
    _save(predictor, noise_trained["artifact"])
    seen = {}

    class FakeGraph:
        def __init__(self, *args, **kwargs):
            pass

        async def run(self, ticker, ml_prediction, past_lessons, news_context):
            seen["ml"] = ml_prediction
            return {"final_advisory": "Hold steady."}

    with _llm_and_web() as (_complete, web), patch("pipeline.agents.AdvisoryGraph", FakeGraph):
        state = await predictor.predict_with_agents("AAA", 5)

    assert state == {"final_advisory": "Hold steady."}
    web.assert_awaited_once()
    ml = seen["ml"]
    assert set(ml) == {"predicted_direction", "confidence", "probability_up", "edge", "model_type",
                       "status", "feature_asof", "feature_snapshot", "model_meta"}
    assert ml["model_type"] == "prior" and ml["status"] == "prior"
    assert isinstance(ml["feature_snapshot"], str)
    stored = db.get_recent_predictions("AAA", 1)[0]
    assert stored["model_type"] == "multi_agent"
    assert stored["probability_up"] == pytest.approx(ml["probability_up"])
    assert stored["feature_asof"] == ml["feature_asof"]


# ── Dates and grading ────────────────────────────────────────────────────────

def test_resolve_after_counts_weekdays_from_the_asof_session():
    assert resolve_after_date("2026-09-11", 5) == "2026-09-18"      # Friday -> next Friday
    assert resolve_after_date("2026-09-14", 1) == "2026-09-15"
    for asof in ("2026-01-02", "2026-03-31", "2026-12-24"):
        for h in features.HORIZONS:
            resolved = date.fromisoformat(resolve_after_date(asof, h))
            assert resolved.weekday() < 5 and resolved >= date.fromisoformat(asof)


def test_grade_prediction_uses_the_session_the_features_described(db, bar_rows):
    history = bar_rows["AAA"]
    base, outcome = history[-10], history[-5]
    grade = grade_prediction({"ticker": "AAA", "horizon_days": 5, "feature_asof": base["date"],
                              "predicted_direction": "UP"}, db)
    assert grade["base_date"] == base["date"] and grade["resolved_date"] == outcome["date"]
    assert grade["actual_change_pct"] == pytest.approx((outcome["close"] / base["close"] - 1) * 100)
    assert grade["actual_direction"] == ("UP" if outcome["close"] > base["close"] else "DOWN")
    assert grade["is_correct"] == (grade["actual_direction"] == "UP")


def test_grade_prediction_anchors_created_at_on_the_new_york_close(db, bar_rows):
    history = bar_rows["AAA"]
    session, previous = history[-30]["date"], history[-31]["date"]
    before_close = f"{session} 14:00:00"      # 09:00-10:00 in New York
    after_close = f"{session} 22:00:00"       # 17:00-18:00 in New York

    early = grade_prediction({"ticker": "AAA", "horizon_days": 5, "created_at": before_close,
                              "predicted_direction": "UP"}, db)
    late = grade_prediction({"ticker": "AAA", "horizon_days": 5, "created_at": after_close,
                             "predicted_direction": "UP"}, db)
    assert early["base_date"] == previous
    assert late["base_date"] == session


def test_grade_prediction_waits_until_the_resolving_session_is_stored(db, bar_rows):
    history = bar_rows["AAA"]
    pending = {"ticker": "AAA", "horizon_days": 5, "feature_asof": history[-3]["date"],
               "predicted_direction": "UP"}
    assert grade_prediction(pending, db) is None
    assert has_price_history(db, "AAA") and not has_price_history(db, "ZZZZ")


# ── Refresh selection ────────────────────────────────────────────────────────
#
# train_missing_models over a stubbed database and predictor. The row that
# decides whether a horizon is refreshed is the one latest_by_horizon picks, so
# the debate's newer multi_agent row does not buy a predict() on its own.

def _live_row(ticker: str, horizon: int, model_type: str, hours_ago: float) -> dict:
    created = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
    return {"ticker": ticker, "horizon_days": horizon, "model_type": model_type,
            "created_at": created.strftime("%Y-%m-%d %H:%M:%S")}


def _current_model_rows(ticker: str, *, except_horizon: int | None = None) -> list[dict]:
    """A universal row from three hours ago for every horizon but `except_horizon`."""
    return [_live_row(ticker, h, "universal", 3) for h in features.HORIZONS if h != except_horizon]


async def _run_refresh(live: dict[str, list[dict]]):
    """One train_missing_models run, with an artifact trained 30 days ago for every horizon.

    `live` maps each tracked ticker to its active prediction rows. Returns the
    predict mock and the refreshes the run logged, as (ticker, horizon, reason).
    """
    trained_at = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    database = MagicMock()
    database.get_tracked_tickers.return_value = list(live)
    database.get_recent_predictions.side_effect = lambda ticker, limit, active_only: sorted(
        live[ticker], key=lambda row: row["created_at"], reverse=True)   # newest first
    stub = MagicMock()
    stub._load_model.side_effect = lambda _ticker, horizon: (
        PooledArtifact(status="model", horizon=horizon, trained_at=trained_at), "universal")
    stub.predict = AsyncMock(return_value={"model_type": "universal"})

    orchestrator = PipelineOrchestrator(db=database)
    with patch("orchestrator.scheduler.StockPredictor", return_value=stub), \
            patch("orchestrator.scheduler.log") as log:
        await orchestrator.train_missing_models()

    # The job swallows its own exceptions; one would pass a no-refresh case vacuously.
    log.error.assert_not_called()
    refreshes = [(c.kwargs["ticker"], c.kwargs["horizon_days"], c.kwargs["reason"])
                 for c in log.info.call_args_list if c.args == ("orchestrator.prediction_refresh",)]
    return stub.predict, refreshes


async def test_a_debate_row_newer_than_a_current_model_row_does_not_refresh_it():
    live = {"AAA": _current_model_rows("AAA") + [_live_row("AAA", 5, "multi_agent", 1)],
            "BBB": []}

    predict, refreshes = await _run_refresh(live)

    # Every AAA horizon is still current, so the run's one unit goes to BBB.
    assert refreshes == [("BBB", 5, "missing")]
    predict.assert_awaited_once_with("BBB", horizon_days=5, fast_fallback=False)


async def test_a_horizon_with_only_a_debate_row_is_refreshed_as_not_pooled_model():
    live = {"AAA": _current_model_rows("AAA", except_horizon=5)
            + [_live_row("AAA", 5, "multi_agent", 1)]}

    predict, refreshes = await _run_refresh(live)

    assert refreshes == [("AAA", 5, "not_pooled_model")]
    predict.assert_awaited_once_with("AAA", horizon_days=5, fast_fallback=False)


async def test_a_stale_model_row_behind_a_newer_debate_row_is_refreshed_as_stale():
    stale_hours = (PREDICTION_REFRESH_DAYS[5] + 1) * 24
    live = {"AAA": _current_model_rows("AAA", except_horizon=5)
            + [_live_row("AAA", 5, "universal", stale_hours), _live_row("AAA", 5, "multi_agent", 1)]}

    predict, refreshes = await _run_refresh(live)

    assert refreshes == [("AAA", 5, "stale")]
    predict.assert_awaited_once_with("AAA", horizon_days=5, fast_fallback=False)
