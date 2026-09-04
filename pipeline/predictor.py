import json
import asyncio
import bisect
import logging
import math
from pathlib import Path
from datetime import datetime, timedelta, timezone, date
from typing import Optional, Tuple, Dict, Any, List, Literal
import numpy as np
import joblib
from pydantic import BaseModel, Field
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

from config.llm import complete, is_llm_configured, parse_structured
from config.settings import settings
from config.usage import track_llm
from config.logging_config import get_logger
from data.database import Database
from data.tickers import KR, US, classify_market, to_krx_code
from pipeline.web_search import (
    search_ticker_news,
    summarize_search_results,
    build_ticker_search_query,
    TavilySearchProvider,
)

log = get_logger(__name__)

MARKET_REGIME_TICKERS = {
    "vix": "^VIX",
    "sp500": "^GSPC",
    "treasury_10y": "^TNX",
}

# Keys are normalized via _normalize_sector() so that both yfinance's own strings
# ("Financial Services", "Consumer Cyclical") and the older underscored spellings
# resolve to the same ETF. Before normalization only "Technology" matched anything
# actually tracked, so sector_etf_return_1d was silently 0.0 for most tickers.
SECTOR_ETF_MAP = {
    "technology": "XLK",
    "financial services": "XLF",
    "finance": "XLF",
    "financial": "XLF",
    "energy": "XLE",
    "healthcare": "XLV",
    "health care": "XLV",
    "consumer cyclical": "XLY",
    "consumer defensive": "XLP",
    "communication services": "XLC",
    "industrials": "XLI",
    "defense": "XLI",
    "utilities": "XLU",
    "real estate": "XLRE",
    "basic materials": "XLB",
    "index": "SPY",
}


def _normalize_sector(sector: Optional[str]) -> str:
    """Fold sector spellings to a single lookup key ('Consumer_Cyclical' -> 'consumer cyclical')."""
    if not sector:
        return ""
    return str(sector).replace("_", " ").replace("-", " ").strip().lower()


def _sector_etf(sector: Optional[str]) -> Optional[str]:
    """Resolve a sector name to its tracking ETF, or None if unmapped."""
    return SECTOR_ETF_MAP.get(_normalize_sector(sector))


# Bump whenever the feature set changes. It is part of the model filename, so a
# bump makes previous models unfindable rather than invalid — old artifacts stay
# on disk and a code revert brings them straight back.
#   v1: the original 23 features (price, sentiment, market regime)
#   v2: adds insider, >5%-stake and Korean investor-flow features
#   v3: adds FINRA off-exchange (dark pool) volume and the market-wide
#       DIX / GEX / OCC put-call regime series
FEATURE_SCHEMA_VERSION = 3

# The prediction horizons the dashboard renders, in days. Defined here rather
# than in api/server.py because the worker's training jobs need them too, and
# importing the API module into the worker process just to read a constant
# pulled the whole FastAPI router in with it.
HORIZON_LABELS = {
    5: "5d",
    21: "1m",
    63: "3m",
    252: "1y",
}


def _fit_and_score_fold(X_train, y_train, w_train, X_val, y_val):
    """
    Fit and score a single cross-validation fold.

    Deliberately synchronous and module-level so train_model can hand it to a
    worker thread — sklearn's fit yields nothing back to the event loop while
    it runs.

    Returns (accuracy, brier, auc):
      * accuracy — fraction of correct UP/DOWN calls.
      * brier    — mean squared error between predicted probability and
                   outcome on the positive (UP) class. Lower is better, 0.0 is
                   perfect; measures calibration and discrimination together.
      * auc      — ability to rank UP days above DOWN days. 0.5 is random, and
                   is also what we report for a single-class fold, where AUC
                   is undefined.
    """
    if len(np.unique(y_train)) >= 2:
        fold_model = GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            subsample=0.9, random_state=42
        )
    else:
        fold_model = DummyClassifier(strategy="prior")
    fold_model.fit(X_train, y_train, sample_weight=w_train)

    y_pred = fold_model.predict(X_val)
    y_proba = fold_model.predict_proba(X_val)

    acc = accuracy_score(y_val, y_pred)
    if y_proba.shape[1] == 2:
        brier = brier_score_loss(y_val, y_proba[:, 1])
    else:
        brier = brier_score_loss(y_val, y_proba[:, 0])

    try:
        if len(np.unique(y_val)) > 1 and y_proba.shape[1] == 2:
            auc = roc_auc_score(y_val, y_proba[:, 1])
        else:
            auc = 0.5
    except ValueError:
        auc = 0.5

    return float(acc), float(brier), float(auc)


class LlmPrediction(BaseModel):
    """Response schema for the LLM-only predictor (used when no ML model exists)."""

    direction: Literal["UP", "DOWN"]
    confidence: float = Field(ge=0.0, le=1.0)
    narrative: str = Field(
        description="2-3 sentence explanation of the key factors and main risk."
    )

# Every feature and the value to use when its source has nothing to say.
#
# Vectors are built by merging *into* this dict, which guarantees a fixed width:
# train_model does np.array(X_list) over ~1200 rows with no length check, so one
# short dict on one day would otherwise surface as an inhomogeneous-shape error
# a thousand iterations deep with no clue which key went missing.
FEATURE_DEFAULTS: Dict[str, float] = {
    # ── News sentiment (from ticker_mentions + articles) ──
    "sentiment_avg_1d": 0.0,
    "sentiment_avg_3d": 0.0,
    "sentiment_avg_7d": 0.0,
    "sentiment_momentum": 0.0,
    "news_velocity": 1.0,
    "avg_importance": 0.0,
    "bullish_ratio": 0.5,
    "max_urgency_24h": 0.0,
    # ── Price technicals (yfinance) ──
    "return_1d": 0.0,
    "return_5d": 0.0,
    "sma_crossover": 0.0,
    "rsi_14": 50.0,
    "volatility": 0.0,
    "volume_anomaly": 1.0,
    "macd_hist": 0.0,
    "bb_width": 0.0,
    "atr_14": 0.0,
    # ── Market regime ──
    "vix_level": 20.0,
    "vix_change_1d": 0.0,
    "market_return_1d": 0.0,
    "treasury_yield_change": 0.0,
    # ── Cross-asset / meta ──
    "sector_etf_return_1d": 0.0,
    "llm_historical_accuracy": 0.5,
    # ── Insider disclosure (SEC Form 4, US only) ──
    # Neutral default is 0.5: "no disclosed trading" is genuinely balanced, not
    # bearish, and 0.0 would read as unanimous selling.
    "insider_buy_ratio_90d": 0.5,
    "insider_net_value_30d_norm": 0.0,
    "insider_cluster_buy_30d": 0.0,
    "days_since_insider_buy": 180.0,
    # ── >5% stake disclosure (SEC 13D/13G, US only) ──
    "activist_stake_90d": 0.0,
    "stake_filings_180d": 0.0,
    # ── Korean investor flows (KRX/Naver, KR only) ──
    "kr_inst_net_5d_norm": 0.0,
    "kr_foreign_net_5d_norm": 0.0,
    "kr_flow_momentum": 0.0,
    # ── Off-exchange (dark pool) volume — FINRA CNMS, US only ──
    # Detrended, never raw. Roughly 40% of the off-exchange tape is market-maker
    # and retail-wholesaler hedging, so the level of each ratio is close to a
    # per-ticker constant that carries no signal, and comparing it across
    # tickers carries less than none. Every feature here is a deviation from the
    # ticker's OWN trailing distribution, which makes the neutral default
    # unambiguous: 0.0 means "sitting exactly on its own average".
    "offexch_short_ratio_z20": 0.0,
    "offexch_volume_share_z20": 0.0,
    "offexch_short_ratio_mom": 0.0,
    # ── Market-wide regime (global, like vix_level above) ──
    # Z-scored rather than raw. GEX is ~8e9, six orders of magnitude above every
    # other feature — gradient boosting is scale-invariant per split so it would
    # not break training, but it would make every feature_snapshot JSON and
    # feature-importance log unreadable. The put/call ratio gets the same
    # treatment because no stable neutral level exists to default to.
    "dix_z60": 0.0,
    "gex_z60": 0.0,
    "occ_put_call_ratio_z60": 0.0,
}

# Insider windows, in days.
_INSIDER_RATIO_WINDOW = 90
_INSIDER_NET_WINDOW = 30
_STAKE_ACTIVIST_WINDOW = 90
_STAKE_ANY_WINDOW = 180
_DAYS_SINCE_BUY_CAP = 180.0

# Off-exchange windows, in sessions.
_OFFEXCH_Z_WINDOW = 20
_OFFEXCH_MOM_WINDOW = 5
# Below this many visible sessions there is no distribution to z-score against.
# Must stay above the z-window: _z_score requires a full window and returns
# None otherwise, so a lower gate here would be dead code and the feature would
# quietly never fire.
_OFFEXCH_MIN_SESSIONS = _OFFEXCH_Z_WINDOW + 1
# A z-score computed against a window that ended weeks ago is worse than the
# neutral default, because it presents as a live reading. If the newest visible
# session is staler than this, fall back to the defaults.
_OFFEXCH_MAX_STALE_DAYS = 7

# Market-regime z-score window, in sessions. Same invariant as above — the gate
# has to clear the z-window or the feature never fires. DIX/GEX carry 15 years
# of history and OCC backfills a session per request, so 60 is cheap to satisfy.
_REGIME_Z_WINDOW = 60
_REGIME_MIN_SESSIONS = _REGIME_Z_WINDOW + 1
_REGIME_MAX_STALE_DAYS = 10
# The regime series is market-wide, so it is cached under one sentinel key
# rather than per ticker: loading it once per symbol would be N identical
# full-table scans of the same rows.
_REGIME_CACHE_KEY = "__market__"


def _signed_log_scale(value: float, cap: float = 10.0) -> float:
    """Compress a signed dollar amount onto roughly [-1, 1].

    Insider trade sizes span six orders of magnitude, from a $20k director
    purchase to a $500m block sale, so the raw figure would let a handful of
    mega-trades dominate every split. A signed log keeps the ordering and the
    direction while flattening the scale, and needs no market-cap or float
    lookup — which matters because this is evaluated once per training day.
    """
    if not value:
        return 0.0
    sign = 1.0 if value > 0 else -1.0
    return sign * min(math.log10(1.0 + abs(value)) / cap, 1.0)


def _z_score(values: List[float], window: int) -> Optional[float]:
    """Standard score of the newest value against its own trailing window.

    Returns None on a short or flat window rather than 0.0. Both cases mean "no
    deviation can be measured", which is not the same claim as "no deviation" —
    and letting the caller fall through to the neutral default keeps that
    distinction in one place instead of two.
    """
    if len(values) < window or window <= 0:
        return None
    tail = values[-window:]
    mean = sum(tail) / window
    variance = sum((v - mean) ** 2 for v in tail) / window
    if variance <= 0:
        return None
    return (tail[-1] - mean) / math.sqrt(variance)



class StockPredictor:
    def __init__(self, db: Database):
        self.db = db
        self.models_dir = Path("storage/models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._price_cache: Dict[str, List[dict]] = {}
        # ticker -> full resolved (created_at, is_correct) series, sliced by as-of date
        self._llm_accuracy_cache: Dict[str, List[tuple]] = {}
        self._sector_cache: Dict[str, str] = {}
        # ticker -> (rows, sorted disclosure-key list); see _cached_series
        self._insider_cache: Dict[str, tuple] = {}
        self._stakes_cache: Dict[str, tuple] = {}
        self._kr_flow_cache: Dict[str, tuple] = {}
        self._darkpool_cache: Dict[str, tuple] = {}
        # Keyed by _REGIME_CACHE_KEY, not by ticker — see the constant.
        self._regime_cache: Dict[str, tuple] = {}
        self._web_search_cache: Dict[str, tuple[float, str]] = {}
        self._web_search_cache_ttl: int = 3600  # 1 hour

    async def predict(self, ticker: str, horizon_days: int = 1, fast_fallback: bool = False) -> dict:
        ticker = ticker.upper()
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        
        # 1. Check cache
        cached = self.db.get_existing_prediction(ticker, horizon_days, today_str)
        if cached:
            log.info(f"Returning cached prediction for {ticker} ({horizon_days}d)")
            return cached

        # 2. Build feature vector
        features = await self.build_feature_vector(ticker)
        if not features:
            return {"error": f"Insufficient data to build features for {ticker}"}
        # 3. Load model and predict
        model, model_type = self._load_model(ticker, horizon_days)
        
        if not model:
            if fast_fallback:
                # Fast heuristic based on recent price trend
                try:
                    prices = await self._fetch_and_cache_prices(ticker, range="1mo")
                    if len(prices) >= 2:
                        current_price = prices[-1]["close"]
                        past_price = prices[-2]["close"] if len(prices) < horizon_days else prices[-min(horizon_days, len(prices))]["close"]
                        direction = "UP" if current_price >= past_price else "DOWN"
                        
                        # Calculate a dynamic confidence score based on the trend magnitude
                        change_pct = abs(current_price - past_price) / past_price if past_price else 0.0
                        # Map change_pct to confidence: base is 52% (0.52), capped at 85% (0.85)
                        confidence = min(0.52 + (change_pct * 2.0), 0.85)
                    else:
                        direction = "UP"
                        confidence = 0.50
                except Exception:
                    direction = "UP"
                    confidence = 0.50
                
                resolve_date = (datetime.now(timezone.utc) + timedelta(days=horizon_days)).strftime("%Y-%m-%d")
                prediction_data = {
                    "ticker": ticker,
                    "horizon_days": horizon_days,
                    "date": today_str,
                    "predicted_direction": direction,
                    "confidence": confidence,
                    "model_type": "fast_heuristic",
                    "feature_snapshot": {},
                    "llm_narrative": f"Fast trend heuristic prediction for {ticker} over {horizon_days}d.",
                    "resolve_after": resolve_date
                }
                self.db.insert_prediction(prediction_data)
                return prediction_data

            log.info(f"No model found for {ticker} ({horizon_days}d). Initiating cold-start training.")
            try:
                _path, _metrics = await self.train_model(ticker, scope="per_ticker", horizon_days=horizon_days)
                model, model_type = self._load_model(ticker, horizon_days)
            except Exception as e:
                log.warning(f"Cold-start training failed for {ticker}: {e}")
                model = None # ensure it falls back to llm_only
                model_type = "llm_only"
            
        confidence = 0.5
        direction = "UP"
        
        if model:
            # Prepare feature array in consistent order
            feature_names = sorted(list(features.keys()))
            X = np.array([[features[k] for k in feature_names]])
            
            # Predict — the loaded model is a CalibratedClassifierCV wrapper,
            # so predict_proba() returns Platt-scaled calibrated probabilities
            pred = model.predict(X)[0]
            proba = model.predict_proba(X)[0]
            
            direction = "UP" if pred == 1 else "DOWN"
            confidence = float(max(proba))
        else:
            model_type = "llm_only"

        recent_summaries = self.db.get_recent_summaries_for_ticker(ticker, hours=72)
        db_context = "\n".join(f"- {s}" for s in recent_summaries) if recent_summaries else ""
        news_context = await self._enrich_with_web_search(ticker, db_context)

        # 4. If LLM-only, ask the LLM to also provide a confidence estimate
        if model_type == "llm_only":
            llm_result = await self._generate_narrative_with_confidence(ticker, features, horizon_days, news_context)
            direction = llm_result.get("direction", "UP")
            confidence = llm_result.get("confidence", 0.5)
            narrative = llm_result.get("narrative", f"LLM predicts {direction} for {ticker}.")
        else:
            # 4. Generate Narrative (LLM)
            narrative = await self._generate_narrative(ticker, direction, confidence, features, horizon_days, news_context)

        resolve_date = (datetime.now(timezone.utc) + timedelta(days=horizon_days)).strftime("%Y-%m-%d")

        prediction_data = {
            "ticker": ticker,
            "predicted_direction": direction,
            "confidence": confidence,
            "horizon_days": horizon_days,
            "model_type": model_type,
            "feature_snapshot": json.dumps(features),
            "llm_narrative": narrative,
            "resolve_after": resolve_date,
            "actual_direction": None,
            "actual_change_pct": None,
            "is_correct": None
        }

        # 5. Store in DB
        pred_id = self.db.insert_prediction(prediction_data)
        prediction_data["id"] = pred_id
        
        return prediction_data

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

    async def predict_with_agents(self, ticker: str, horizon_days: int = 5, progress_callback=None, debate_chunk_callback=None, research_callback=None) -> dict:
        """New method to generate a prediction using the full LangGraph multi-agent flow."""
        # 1. Get Quantitative Baseline (ML)
        features = await self.build_feature_vector(ticker)
        model, model_type = self._load_model(ticker, horizon_days)
        should_train_after = False
        if not model:
            log.info(f"No model found for {ticker} ({horizon_days}d). Will train in background after debate.")
            should_train_after = True
            
        direction = "UNKNOWN"
        confidence = 0.0

        if model and features:
            feature_names = sorted(list(features.keys()))
            X = np.array([[features[k] for k in feature_names]])
            pred = model.predict(X)[0]
            proba = model.predict_proba(X)[0]
            direction = "UP" if pred == 1 else "DOWN"
            confidence = float(max(proba))
        
        ml_prediction = {
            "predicted_direction": direction,
            "confidence": confidence,
            # Which tier answered, so the arena can say "no model trained"
            # instead of printing the literal string UNKNOWN at 0% confidence.
            # `_load_model` reports "llm_only" when nothing was found at any
            # tier — which is every ticker for a horizon whose artifacts were
            # orphaned by a FEATURE_SCHEMA_VERSION bump.
            "model_type": model_type,
            "feature_snapshot": json.dumps(features or {})
        }
        
        # 2. Get past lessons and recent news
        past_lessons = {"ticker_lessons": [], "sector_lessons": [], "market_lessons": []}
        if hasattr(self.db, "get_relevant_reflections"):
            past_lessons = self.db.get_relevant_reflections(ticker, limit=5)
        elif hasattr(self.db, "get_recent_reflections"):
            # Fallback for backward compatibility
            reflections = self.db.get_recent_reflections(ticker, limit=3)
            if reflections:
                past_lessons["ticker_lessons"] = [
                    {"lesson_learned": r, "was_successful": True, "date": ""}
                    for r in reflections
                ]

        recent_summaries = self.db.get_recent_summaries_for_ticker(ticker, hours=72)
        db_context = "\n".join(f"- {s}" for s in recent_summaries) if recent_summaries else ""
        news_context = await self._enrich_with_web_search(ticker, db_context, research_callback=research_callback)

        # 3. Run AdvisoryGraph
        from pipeline.agents import AdvisoryGraph
        graph = AdvisoryGraph(self.db, progress_callback=progress_callback, debate_chunk_callback=debate_chunk_callback)
        final_state = await graph.run(ticker, ml_prediction, past_lessons, news_context)

        # 4. Save to DB and cache
        resolve_date = (datetime.now(timezone.utc) + timedelta(days=horizon_days)).strftime("%Y-%m-%d")
        prediction_data = {
            "ticker": ticker,
            "predicted_direction": direction,
            "confidence": confidence,
            "horizon_days": horizon_days,
            "model_type": "multi_agent",
            "feature_snapshot": ml_prediction["feature_snapshot"],
            "llm_narrative": final_state.get("final_advisory", "Error generating advisory."),
            "resolve_after": resolve_date,
            "actual_direction": None,
            "actual_change_pct": None,
            "is_correct": None
        }
        
        pred_id = self.db.insert_prediction(prediction_data)
        
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if hasattr(self.db, "set_cached_advisory"):
            self.db.set_cached_advisory(ticker, today, json.dumps(final_state))
            
        if should_train_after:
            async def train_in_bg():
                log.info(f"Initiating background cold-start training for {ticker} ({horizon_days}d)")
                try:
                    await self.train_model(ticker, scope="per_ticker", horizon_days=horizon_days)
                except Exception as e:
                    log.warning(f"Background training failed for {ticker}: {e}")
            asyncio.create_task(train_in_bg())
            
        return final_state

    async def build_feature_vector(self, ticker: str, as_of_date: str = None) -> Optional[dict]:
        sentiment_features = self._get_sentiment_features(ticker, as_of_date)
        price_features = await self._get_price_features(ticker, as_of_date)
        
        if not price_features:
            return None
            
        market_features = await self._get_market_regime_features(as_of_date)
        smart_money_features = self._get_smart_money_features(ticker, as_of_date)

        # Merge INTO the defaults so the vector is always the same width and in
        # the same order, whatever any individual source had to say.
        features = {
            **FEATURE_DEFAULTS,
            **sentiment_features,
            **price_features,
            **market_features,
            **smart_money_features,
            "llm_historical_accuracy": self._get_llm_accuracy(ticker, as_of_date),
        }

        # Add sector ETF return if available
        sector = self._get_sector(ticker)
        etf = _sector_etf(sector)
        if etf:
            etf_features = await self._get_price_features(etf, as_of_date)
            if etf_features and "return_1d" in etf_features:
                features["sector_etf_return_1d"] = etf_features["return_1d"]

        # A source returning an unexpected key would silently widen the vector
        # and desync it from the trained model; fail loudly at the source.
        if len(features) != len(FEATURE_DEFAULTS):
            unexpected = set(features) - set(FEATURE_DEFAULTS)
            raise ValueError(
                f"Feature vector width {len(features)} != schema "
                f"{len(FEATURE_DEFAULTS)}; unexpected keys: {sorted(unexpected)}"
            )
        return features

    @staticmethod
    def _as_of_dt(as_of_date: str = None) -> Optional[datetime]:
        """Coerce a 'YYYY-MM-DD' training date into an end-of-day UTC datetime.

        Returns None for the live path so callers fall back to 'now'. The end-of-day
        edge matches how price features treat as_of_date: the bar for that session
        has closed, so news published during that session is legitimately visible.
        """
        if not as_of_date:
            return None
        try:
            d = datetime.strptime(str(as_of_date)[:10], "%Y-%m-%d")
            return d.replace(hour=23, minute=59, second=59, tzinfo=timezone.utc)
        except (ValueError, TypeError):
            return None

    def _get_llm_accuracy(self, ticker: str, as_of_date: str = None) -> float:
        """Rolling accuracy of the last 10 resolved multi-agent predictions as of a date.

        The whole resolved series is fetched once per ticker and sliced in memory —
        training walks ~1200 dates and a query per date would open ~1200 connections.
        """
        if ticker not in self._llm_accuracy_cache:
            series: List[tuple[str, int]] = []
            try:
                with self.db.connection() as conn:
                    rows = conn.execute(
                        """
                        SELECT created_at, is_correct FROM predictions
                        WHERE ticker = ? AND is_correct IS NOT NULL
                          AND model_type = 'multi_agent'
                        ORDER BY created_at ASC
                        """,
                        (ticker,),
                    ).fetchall()
                series = [(str(r[0]), int(r[1])) for r in rows]
            except Exception:
                series = []
            self._llm_accuracy_cache[ticker] = series

        series = self._llm_accuracy_cache[ticker]
        if not series:
            return 0.5

        as_of = self._as_of_dt(as_of_date)
        if as_of is None:
            window = series[-10:]
        else:
            # Only predictions already resolved by as_of are knowable at as_of.
            cutoff = as_of.isoformat()
            idx = bisect.bisect_right([s[0] for s in series], cutoff)
            window = series[max(0, idx - 10):idx]

        if not window:
            return 0.5
        return sum(c for _, c in window) / len(window)

    # ── Smart money: insider / stake / KR flow features ──────────────────
    #
    # Each series is fetched once per ticker and sliced in memory. Training
    # evaluates ~1200 as-of dates per ticker per horizon, and db.connection()
    # opens a fresh connection (and reloads the sqlite-vec extension) on every
    # call, so a query per date per feature would dominate training time.

    def _cached_series(self, cache: Dict[str, tuple], ticker: str,
                       loader, key_field: str) -> tuple[list, list]:
        """Load a ticker's series once, memoised as (rows, sorted key list)."""
        if ticker not in cache:
            try:
                rows = loader(ticker)
            except Exception as e:
                log.warning("predictor.series_load_failed", ticker=ticker,
                            field=key_field, error=str(e))
                rows = []
            cache[ticker] = (rows, [str(r[key_field]) for r in rows])
        return cache[ticker]

    @staticmethod
    def _visible(rows: list, keys: list, as_of: Optional[datetime]) -> list:
        """Rows already public as of a date.

        Slices on the DISCLOSURE key, not the event key. A Form 4 covers a trade
        made up to two business days before it was filed; filtering on the trade
        date would show the model transactions nobody could act on yet, which
        flatters the backtest and does nothing live. Measured mean lag on real
        NVDA/DIS data is 1.6-4.1 days.
        """
        if as_of is None:
            return rows
        return rows[: bisect.bisect_right(keys, as_of.isoformat())]

    def _get_smart_money_features(self, ticker: str, as_of_date: str = None) -> dict:
        """Insider, >5%-stake and Korean-flow features for one as-of date."""
        market = classify_market(ticker)
        as_of = self._as_of_dt(as_of_date)
        now = as_of or datetime.now(timezone.utc)
        out: dict[str, float] = {}

        if market == US:
            out.update(self._insider_features(ticker, as_of, now))
            out.update(self._stake_features(ticker, as_of, now))
            # FINRA's TRFs cover US equities and ETFs, so VOO and SPY belong
            # here too — classify_market already returns US for them.
            out.update(self._darkpool_features(ticker, as_of, now))
        elif market == KR:
            out.update(self._kr_flow_features(ticker, as_of))
        # Anything else (indices, crypto) keeps the defaults, and short-circuits
        # before touching SQLite — a US ticker never queries the KR table.
        return out

    def _insider_features(self, ticker: str, as_of: Optional[datetime],
                          now: datetime) -> dict:
        rows, keys = self._cached_series(
            self._insider_cache, ticker, self.db.get_insider_series, "filed_at")
        visible = self._visible(rows, keys, as_of)
        if not visible:
            return {}

        ratio_cut = (now - timedelta(days=_INSIDER_RATIO_WINDOW)).isoformat()
        net_cut = (now - timedelta(days=_INSIDER_NET_WINDOW)).isoformat()

        buy_value = sell_value = net_value = 0.0
        buyers: set[str] = set()
        last_buy: Optional[str] = None

        for r in visible:
            # Only open-market buys and sales carry a view. Grants, option
            # exercises and tax withholding are compensation mechanics.
            if not r["is_discretionary"]:
                continue
            value = abs(r["value_usd"] or 0.0)
            is_buy = r["transaction_code"] == "P"

            if r["filed_at"] >= ratio_cut:
                if is_buy:
                    buy_value += value
                else:
                    sell_value += value
            if r["filed_at"] >= net_cut:
                net_value += value if is_buy else -value
                if is_buy and r["insider_name"]:
                    buyers.add(r["insider_name"])
            if is_buy:
                last_buy = r["transaction_date"]

        features: dict[str, float] = {}
        denom = buy_value + sell_value
        if denom > 0:
            features["insider_buy_ratio_90d"] = buy_value / denom
        features["insider_net_value_30d_norm"] = _signed_log_scale(net_value)
        features["insider_cluster_buy_30d"] = float(len(buyers))

        if last_buy:
            try:
                delta = (now - datetime.strptime(last_buy[:10], "%Y-%m-%d")
                         .replace(tzinfo=timezone.utc)).days
                features["days_since_insider_buy"] = float(
                    min(max(delta, 0), _DAYS_SINCE_BUY_CAP))
            except ValueError:
                pass
        return features

    def _stake_features(self, ticker: str, as_of: Optional[datetime],
                        now: datetime) -> dict:
        rows, keys = self._cached_series(
            self._stakes_cache, ticker, self.db.get_stakes_series, "filed_at")
        visible = self._visible(rows, keys, as_of)
        if not visible:
            return {}

        activist_cut = (now - timedelta(days=_STAKE_ACTIVIST_WINDOW)).isoformat()
        any_cut = (now - timedelta(days=_STAKE_ANY_WINDOW)).isoformat()
        return {
            # 13D means the holder intends to influence the company; 13G is a
            # passive holding and a much weaker signal, so they stay separate.
            "activist_stake_90d": float(sum(
                1 for r in visible if r["is_activist"] and r["filed_at"] >= activist_cut)),
            "stake_filings_180d": float(sum(
                1 for r in visible if r["filed_at"] >= any_cut)),
        }

    def _kr_flow_features(self, ticker: str, as_of: Optional[datetime]) -> dict:
        code = to_krx_code(ticker)
        if not code:
            return {}
        rows, keys = self._cached_series(
            self._kr_flow_cache, code, self.db.get_kr_flow_series, "trade_date")
        # Sessions are same-day published, so the as-of edge is the date itself.
        cutoff = as_of.strftime("%Y-%m-%d") if as_of else None
        visible = rows[: bisect.bisect_right(keys, cutoff)] if cutoff else rows
        if len(visible) < 5:
            return {}

        def net(window: int, field: str) -> float:
            return sum(r[field] or 0.0 for r in visible[-window:])

        volume_20d = sum(r["total_value"] or 0.0 for r in visible[-20:])
        if volume_20d <= 0:
            return {}

        inst_5d = net(5, "inst_net")
        foreign_5d = net(5, "foreign_net")
        # Normalising by traded volume keeps this comparable across tickers and
        # unit-agnostic: Naver reports shares, the KRX API reports KRW, and both
        # divide out against a total_value carried in the same unit.
        smart_5d = (inst_5d + foreign_5d) / volume_20d
        smart_20d = (net(20, "inst_net") + net(20, "foreign_net")) / volume_20d

        return {
            "kr_inst_net_5d_norm": inst_5d / volume_20d,
            "kr_foreign_net_5d_norm": foreign_5d / volume_20d,
            # Is the last week's flow accelerating relative to the month?
            "kr_flow_momentum": smart_5d - (smart_20d / 4.0),
        }

    def _darkpool_features(self, ticker: str, as_of: Optional[datetime],
                           now: datetime) -> dict:
        """Off-exchange volume features for one as-of date.

        Everything here is a z-score against the ticker's own trailing window,
        never a level — see the FEATURE_DEFAULTS comment and the
        pipeline.darkpool docstring for why the raw ratios carry no signal.
        """
        rows, keys = self._cached_series(
            self._darkpool_cache, ticker,
            self.db.get_offexchange_series, "published_at")
        visible = self._visible(rows, keys, as_of)
        if len(visible) < _OFFEXCH_MIN_SESSIONS:
            return {}

        # A gap means the sync has been failing. Scoring today against a window
        # that ended a fortnight ago produces a confident-looking number built
        # from stale inputs, which is strictly worse than admitting no data.
        try:
            newest = datetime.strptime(
                str(visible[-1]["session_date"])[:10], "%Y-%m-%d"
            ).replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            return {}
        if (now - newest).days > _OFFEXCH_MAX_STALE_DAYS:
            return {}

        short_ratios: list[float] = []
        shares: list[float] = []
        for r in visible:
            total = r["total_volume"] or 0.0
            if total <= 0:
                continue
            if r["short_volume"] is not None:
                short_ratios.append(r["short_volume"] / total)
            consolidated = r["consolidated_volume"] or 0.0
            if consolidated > 0:
                shares.append(total / consolidated)

        features: dict[str, float] = {}

        short_z = _z_score(short_ratios, _OFFEXCH_Z_WINDOW)
        if short_z is not None:
            features["offexch_short_ratio_z20"] = short_z

        # Kept separate from the z-score rather than folded into it: the point
        # reading says "unusual today", this says "trending", and a ratio can
        # sit inside its band for a week while walking steadily across it.
        if len(short_ratios) >= _OFFEXCH_Z_WINDOW:
            fast = short_ratios[-_OFFEXCH_MOM_WINDOW:]
            slow = short_ratios[-_OFFEXCH_Z_WINDOW:]
            features["offexch_short_ratio_mom"] = (
                sum(fast) / len(fast) - sum(slow) / len(slow)
            )

        # price_history is filled on demand, so this leg is often absent while
        # the short-ratio leg is present. Defaulting it independently keeps the
        # available half of the signal instead of discarding both.
        share_z = _z_score(shares, _OFFEXCH_Z_WINDOW)
        if share_z is not None:
            features["offexch_volume_share_z20"] = share_z

        return features

    def _regime_series_features(self, as_of: Optional[datetime],
                                now: datetime) -> dict:
        """DIX / GEX / OCC put-call z-scores. Market-wide, so ticker-independent."""
        rows, keys = self._cached_series(
            self._regime_cache, _REGIME_CACHE_KEY,
            lambda _key: self.db.get_market_regime_series(), "published_at")
        visible = self._visible(rows, keys, as_of)
        if not visible:
            return {}

        by_metric: dict[str, list[float]] = {}
        latest: dict[str, str] = {}
        for r in visible:
            if r["value"] is None:
                continue
            by_metric.setdefault(r["metric"], []).append(float(r["value"]))
            latest[r["metric"]] = str(r["session_date"])[:10]

        out: dict[str, float] = {}
        for metric, feature in (("dix", "dix_z60"), ("gex", "gex_z60"),
                                ("occ_put_call_ratio", "occ_put_call_ratio_z60")):
            values = by_metric.get(metric, [])
            if len(values) < _REGIME_MIN_SESSIONS:
                continue
            try:
                newest = datetime.strptime(
                    latest[metric], "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except (ValueError, TypeError, KeyError):
                continue
            if (now - newest).days > _REGIME_MAX_STALE_DAYS:
                continue
            score = _z_score(values, _REGIME_Z_WINDOW)
            if score is not None:
                out[feature] = score
        return out

    def _get_sentiment_features(self, ticker: str, as_of_date: str = None) -> dict:
        features = self.db.get_ticker_sentiment_features(
            ticker, lookback_days=7, as_of=self._as_of_dt(as_of_date)
        )
        if not features:
            # Defaults must match the EXACT keys returned by get_ticker_sentiment_features
            return {
                "sentiment_avg_1d": 0.0,
                "sentiment_avg_3d": 0.0,
                "sentiment_avg_7d": 0.0,
                "sentiment_momentum": 0.0,
                "news_velocity": 1.0,
                "avg_importance": 0.0,
                "bullish_ratio": 0.5,
                "max_urgency_24h": 0.0,
            }
        return features

    async def _get_price_features(self, ticker: str, as_of_date: str = None) -> Optional[dict]:
        prices = await self._fetch_and_cache_prices(ticker, range="6mo")
        if not prices or len(prices) < 20:
            return None
            
        if as_of_date:
            # Filter prices up to as_of_date
            prices = [p for p in prices if p["date"] <= as_of_date]
            
        if len(prices) < 20:
            return None
            
        import pandas as pd
        df = pd.DataFrame(prices)
        closes = df['close'].values
        volumes = df['volume'].values
        
        current_close = closes[-1]
        prev_1d_close = closes[-2]
        prev_5d_close = closes[-6] if len(closes) >= 6 else closes[0]
        
        ret_1d = (current_close - prev_1d_close) / prev_1d_close
        ret_5d = (current_close - prev_5d_close) / prev_5d_close
        
        sma5 = self._compute_sma(closes, 5)
        sma20 = self._compute_sma(closes, 20)
        sma_crossover = (sma5 - sma20) / sma20 if sma20 != 0 else 0
        
        rsi_14 = self._compute_rsi(closes, 14)
        
        returns = np.diff(closes[-21:]) / closes[-21:-1]
        volatility = float(np.std(returns))
        
        avg_vol_20 = np.mean(volumes[-20:])
        volume_anomaly = volumes[-1] / avg_vol_20 if avg_vol_20 > 0 else 1.0

        # MACD (12, 26, 9)
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = macd.iloc[-1] - macd_signal.iloc[-1]
        
        # Bollinger Bands (20, 2)
        sma20_series = df['close'].rolling(window=20).mean()
        std20_series = df['close'].rolling(window=20).std()
        upper_bb = sma20_series + (std20_series * 2)
        lower_bb = sma20_series - (std20_series * 2)
        bb_width = ((upper_bb - lower_bb) / sma20_series).iloc[-1]
        if pd.isna(bb_width): bb_width = 0.0
        
        # ATR (14)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr14 = tr.rolling(window=14).mean().iloc[-1]
        if pd.isna(atr14): atr14 = 0.0
        
        return {
            "return_1d": float(ret_1d),
            "return_5d": float(ret_5d),
            "sma_crossover": float(sma_crossover),
            "rsi_14": float(rsi_14),
            "volatility": float(volatility),
            "volume_anomaly": float(volume_anomaly),
            "macd_hist": float(macd_hist),
            "bb_width": float(bb_width),
            "atr_14": float(atr14),
        }

    async def _get_market_regime_features(self, as_of_date: str = None) -> dict:
        vix_prices = await self._fetch_and_cache_prices(MARKET_REGIME_TICKERS["vix"], "1mo")
        sp500_prices = await self._fetch_and_cache_prices(MARKET_REGIME_TICKERS["sp500"], "1mo")
        tnx_prices = await self._fetch_and_cache_prices(MARKET_REGIME_TICKERS["treasury_10y"], "1mo")
        
        def get_recent(prices):
            if as_of_date:
                prices = [p for p in prices if p["date"] <= as_of_date]
            return prices[-2:] if len(prices) >= 2 else None
            
        features = {
            "vix_level": 20.0,
            "vix_change_1d": 0.0,
            "market_return_1d": 0.0,
            "treasury_yield_change": 0.0,
        }
        
        vix_recent = get_recent(vix_prices)
        if vix_recent:
            features["vix_level"] = vix_recent[-1]["close"]
            features["vix_change_1d"] = (vix_recent[-1]["close"] - vix_recent[-2]["close"]) / vix_recent[-2]["close"]
            
        sp500_recent = get_recent(sp500_prices)
        if sp500_recent:
            features["market_return_1d"] = (sp500_recent[-1]["close"] - sp500_recent[-2]["close"]) / sp500_recent[-2]["close"]
            
        tnx_recent = get_recent(tnx_prices)
        if tnx_recent:
            features["treasury_yield_change"] = tnx_recent[-1]["close"] - tnx_recent[-2]["close"]

        # Dark-pool index, gamma exposure and the market-wide put/call ratio.
        # Market-wide like the three above, but read from market_regime_daily
        # rather than yfinance, so they slot in here rather than in the
        # per-ticker smart-money branch.
        as_of = self._as_of_dt(as_of_date)
        features.update(self._regime_series_features(
            as_of, as_of or datetime.now(timezone.utc)))

        return features

    async def _fetch_and_cache_prices(self, ticker: str, range: str = "6mo") -> list[dict]:
        import yfinance as yf
        
        # In-memory cache to avoid thousands of identical network calls during loops
        if ticker in self._price_cache:
            cached = self._price_cache[ticker]
            # Since train_model pre-fetches "5y", the cache will have sufficient data.
            if range == "5y" and len(cached) < 1000:
                pass # fetch again if we somehow don't have enough
            else:
                return cached
        
        try:
            yticker = yf.Ticker(ticker)
            df = yticker.history(period=range)
            if df.empty:
                return []
                
            # Drop rows with NaN in Close to avoid DB NOT NULL constraint failures
            import pandas as pd
            df = df.dropna(subset=['Close'])
                
            rows = []
            for date, row in df.iterrows():
                rows.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "open": row["Open"],
                    "high": row["High"],
                    "low": row["Low"],
                    "close": row["Close"],
                    "volume": int(row["Volume"] if not pd.isna(row["Volume"]) else 0)
                })
            
            # Guarantee chronological order
            rows.sort(key=lambda r: r["date"])
            
            # Upsert DB cache
            self.db.upsert_price_history(ticker, rows)
            
            # Update in-memory cache
            self._price_cache[ticker] = rows
            return rows
        except Exception as e:
            log.error(f"Error fetching prices for {ticker}: {e}")
            return []

    def _compute_rsi(self, closes: np.ndarray, period: int = 14) -> float:
        if len(closes) < period + 1:
            return 50.0
        diffs = np.diff(closes)
        gains = np.where(diffs > 0, diffs, 0)
        losses = np.where(diffs < 0, -diffs, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _compute_sma(self, closes: np.ndarray, period: int) -> float:
        if len(closes) < period:
            return float(np.mean(closes)) if len(closes) > 0 else 0.0
        return float(np.mean(closes[-period:]))

    def _load_model(self, ticker: str, horizon_days: int = 1) -> tuple[Optional[Any], str]:
        expected_features = len(FEATURE_DEFAULTS)

        def load_and_validate(path):
            if not path.exists():
                return None
            try:
                model = joblib.load(path)
                n_features = getattr(model, "n_features_in_", None)
                if n_features is None and hasattr(model, "calibrated_classifiers_") and len(model.calibrated_classifiers_) > 0:
                    cc = model.calibrated_classifiers_[0]
                    if hasattr(cc, "estimator"):
                        n_features = getattr(cc.estimator, "n_features_in_", None)
                    elif hasattr(cc, "base_estimator"):
                        n_features = getattr(cc.base_estimator, "n_features_in_", None)
                
                if n_features is not None and n_features != expected_features:
                    # Do NOT delete. The filename is version-stamped, so a
                    # mismatch here means a stale artifact rather than a
                    # corrupt one, and deleting it would make the change
                    # irreversible while /api/markets is polling every 10s.
                    log.warning(
                        "predictor.model_feature_mismatch",
                        path=str(path), found=n_features, expected=expected_features,
                    )
                    return None
                return model
            except Exception as e:
                log.warning(f"Error loading model at {path}: {e}")
                return None

        # 1. per-ticker
        path = self._get_model_path(ticker, horizon_days)
        model = load_and_validate(path)
        if model is not None:
            return model, "per_ticker"
            
        # 2. sector
        sector = self._get_sector(ticker)
        path = self._get_model_path(sector, horizon_days)
        model = load_and_validate(path)
        if model is not None:
            return model, "sector"
            
        # 3. universal
        path = self._get_model_path("universal", horizon_days)
        model = load_and_validate(path)
        if model is not None:
            return model, "universal"
            
        # 4. fallback
        return None, "llm_only"

    def _get_sector(self, ticker: str) -> str:
        # Memoized: build_feature_vector calls this on every training day, and
        # db.get_ticker_sector opens a connection (plus a yfinance lookup on miss).
        if ticker not in self._sector_cache:
            try:
                self._sector_cache[ticker] = self.db.get_ticker_sector(ticker)
            except Exception:
                return "Unknown"
        return self._sector_cache[ticker]

    async def train_model(self, ticker: str, scope: str = "per_ticker", horizon_days: int = 1) -> Tuple[str, dict]:
        """Train a calibrated GradientBoosting model with time-series cross-validation.

        Uses 5-fold walk-forward TimeSeriesSplit to evaluate out-of-sample
        performance, then trains a final model on the full dataset and wraps
        it in a Platt Scaling calibrator fitted on the last CV fold's
        validation set.

        Returns:
            (model_path, cv_metrics) where cv_metrics contains per-fold
            averages for accuracy, Brier score, and ROC AUC.
        """
        log.info(f"Training model for {ticker} (scope: {scope}, horizon: {horizon_days}d)")
        
        # Increase range to 5y to give model robust data for the 252d (1y) horizon
        prices = await self._fetch_and_cache_prices(ticker, range="5y")
        
        # Pre-fetch market regime tickers
        for m_ticker in MARKET_REGIME_TICKERS.values():
            await self._fetch_and_cache_prices(m_ticker, range="5y")
            
        # Pre-fetch sector ETF
        sector = self._get_sector(ticker)
        etf = _sector_etf(sector)
        if etf:
            await self._fetch_and_cache_prices(etf, range="5y")
        
        
        if not prices or len(prices) < 30 + horizon_days:
            raise ValueError(f"Insufficient historical data ({len(prices) if prices else 0} days) to train model.")
            
        X_list = []
        y_list = []
        y_returns_list = []
        
        # Start from index 20 to allow rolling features (SMA20, etc.) to warm up
        # End at len(prices) - 1 - horizon_days because we need prices[i+horizon_days] for the true label
        for i in range(20, len(prices) - horizon_days):
            current_date = prices[i]["date"]
            future_close = prices[i + horizon_days]["close"]
            current_close = prices[i]["close"]
            
            # The label: 1 if the future day closed higher than current day
            label = 1 if future_close > current_close else 0
            ret = (future_close - current_close) / current_close
            
            features = await self.build_feature_vector(ticker, as_of_date=current_date)
            if features:
                feature_names = sorted(list(features.keys()))
                feature_values = [features[k] for k in feature_names]
                X_list.append(feature_values)
                y_list.append(label)
                y_returns_list.append(ret)
            await asyncio.sleep(0)
                
        if len(X_list) < 10:
            raise ValueError(f"Failed to build enough feature vectors ({len(X_list)}) for {ticker}.")
            
        X = np.array(X_list)
        y = np.array(y_list)
        y_ret = np.array(y_returns_list)
        
        # Calculate exponential sample weights to prioritize recent data
        # Using decay_rate = 0.003 for ~11 month half-life to emphasize past year
        decay_rate = 0.003
        n_samples = len(X)
        time_weights = np.exp(-decay_rate * np.arange(n_samples - 1, -1, -1))
        
        # Multiply by absolute return magnitude to prioritize volatile days
        # Add small constant to avoid zero weights for completely flat days
        abs_returns = np.abs(y_ret)
        weights = time_weights * (abs_returns + 0.001)
        
        # ── Time-Series Cross-Validation (5-fold walk-forward) ──
        # TimeSeriesSplit ensures no future data leaks into training.
        # Each fold expands the training window and slides the validation
        # window forward, simulating real deployment conditions:
        #   Fold 1: train[0:N1]  → val[N1:N2]
        #   Fold 2: train[0:N2]  → val[N2:N3]
        #   ...and so on.
        tscv = TimeSeriesSplit(n_splits=5)
        
        fold_accuracies = []
        fold_briers = []
        fold_aucs = []
        last_val_X = None
        last_val_y = None
        
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            w_train = weights[train_idx]
            
            # Offloaded: fitting a GradientBoostingClassifier is seconds of
            # uninterruptible CPU, and the `await asyncio.sleep(0)` below only
            # yields *between* folds. Left on the loop it stalls every other
            # job in this process — most visibly the price refresh that keeps
            # the dashboard's quotes current.
            acc, brier, auc = await asyncio.to_thread(
                _fit_and_score_fold, X_train, y_train, w_train, X_val, y_val
            )
            fold_accuracies.append(acc)
            fold_briers.append(brier)
            fold_aucs.append(auc)
            
            log.info(
                f"  Fold {fold_idx + 1}/5: acc={acc:.3f} brier={brier:.3f} auc={auc:.3f} "
                f"(train={len(train_idx)}, val={len(val_idx)})"
            )
            
            # Keep the last fold's validation set for Platt Scaling calibration
            last_val_X = X_val
            last_val_y = y_val
            await asyncio.sleep(0)
        
        cv_metrics = {
            "accuracy_mean": float(np.mean(fold_accuracies)),
            "accuracy_std": float(np.std(fold_accuracies)),
            "brier_mean": float(np.mean(fold_briers)),
            "brier_std": float(np.std(fold_briers)),
            "auc_mean": float(np.mean(fold_aucs)),
            "auc_std": float(np.std(fold_aucs)),
            "n_samples": n_samples,
            "n_folds": 5,
        }
        
        log.info(
            f"CV results for {ticker}: "
            f"acc={cv_metrics['accuracy_mean']:.3f}±{cv_metrics['accuracy_std']:.3f} "
            f"brier={cv_metrics['brier_mean']:.3f}±{cv_metrics['brier_std']:.3f} "
            f"auc={cv_metrics['auc_mean']:.3f}±{cv_metrics['auc_std']:.3f}"
        )
        
        # ── Train final production model on full dataset ──
        if len(np.unique(y)) >= 2:
            from sklearn.model_selection import RandomizedSearchCV
            tscv_search = TimeSeriesSplit(n_splits=3)
            
            valid_folds = True
            for train_idx, _ in tscv_search.split(X):
                if len(np.unique(y[train_idx])) < 2:
                    valid_folds = False
                    break

            if valid_folds:
                param_dist = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'subsample': [0.8, 1.0]
                }
                from sklearn.metrics import make_scorer
                
                def safe_roc_auc(y_true, y_pred):
                    if len(np.unique(y_true)) == 1:
                        return 0.5
                    return roc_auc_score(y_true, y_pred)
                    
                safe_auc_scorer = make_scorer(safe_roc_auc, response_method='predict_proba')
    
                base_model = GradientBoostingClassifier(random_state=42)
                search = RandomizedSearchCV(
                    base_model, param_dist, n_iter=5, cv=tscv_search, 
                    scoring=safe_auc_scorer, n_jobs=2, random_state=42
                )
                # n_iter=5 over cv=3 is up to 15 more fits — by far the
                # heaviest step in training, and equally unfit for the loop.
                await asyncio.to_thread(search.fit, X, y, sample_weight=weights)
                final_model = search.best_estimator_
                log.info(f"Best hyperparameters for {ticker}: {search.best_params_}")
            else:
                log.warning(f"Skipping hyperparameter search for {ticker} due to single-class training folds.")
                final_model = GradientBoostingClassifier(random_state=42)
                await asyncio.to_thread(final_model.fit, X, y, sample_weight=weights)
            
            # Log feature importances. Names come from the schema, not from a
            # fresh build_feature_vector(ticker) call — that would rebuild the
            # vector for *today* to label a matrix built from historical dates,
            # and returns None when today's data is thin, taking the whole
            # training run down with an AttributeError.
            feature_names = sorted(FEATURE_DEFAULTS.keys())
            importances = final_model.feature_importances_
            feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
            log.info(f"Feature Importances for {ticker}: {feat_imp[:5]}")
        else:
            final_model = DummyClassifier(strategy="prior")
            final_model.fit(X, y, sample_weight=weights)
        
        # ── Platt Scaling Calibration ──
        # Attempt to calibrate probabilities. If the dataset is too small
        # or imbalanced (causing fold class imbalance errors), we safely
        # catch the exception and fall back to the uncalibrated model.
        calibrated_model = final_model
        try:
            if len(np.unique(y)) == 2 and min(np.bincount(y)) >= 5:
                # Use TimeSeriesSplit for calibration to avoid future-leakage
                tscv_calib = TimeSeriesSplit(n_splits=min(5, len(X)//20))
                calib = CalibratedClassifierCV(
                    estimator=final_model, method="sigmoid", cv=tscv_calib
                )
                calib.fit(X, y)
                calibrated_model = calib
        except Exception as e:
            log.warning(f"Calibration failed for {ticker}: {e}. Using uncalibrated model.")
        
        name = ticker if scope == "per_ticker" else (
            self._get_sector(ticker) if scope == "sector" else "universal"
        )
        path = self._get_model_path(name, horizon_days)
        joblib.dump(calibrated_model, path)
        
        log.info(f"Saved calibrated model to {path}")
        return str(path), cv_metrics

    async def _generate_narrative(self, ticker: str, direction: str, confidence: float, features: dict, horizon_days: int, news_context: str = "") -> str:
        model_name = settings.model_predictor_narrative
        try:
            if not is_llm_configured() or not model_name:
                return f"Model predicts {direction} based on current technical and sentiment features."

            # Select the most impactful features for the narrative (top 6 by importance)
            key_features = {}
            priority_keys = ["sentiment_avg_1d", "sentiment_momentum", "return_1d", "rsi_14",
                           "avg_importance", "volatility", "bullish_ratio", "news_velocity",
                           "sma_crossover", "macd_hist", "volume_anomaly", "vix_level"]
            for k in priority_keys:
                if k in features:
                    key_features[k] = features[k]
            # Add any remaining features not in priority list
            for k, v in features.items():
                if k not in key_features:
                    key_features[k] = v

            prompt = (
                f"You are a senior financial analyst. The ML model predicts {ticker} will go {direction} "
                f"over the next {horizon_days} day(s) with {confidence*100:.1f}% confidence.\n\n"
                f"Key features driving this prediction:\n{json.dumps(key_features, indent=2)}\n\n"
                f"Recent News Context:\n{news_context}\n\n"
                f"Write a concise 2-3 sentence narrative that:\n"
                f"1. Names the 1-2 MOST impactful features or news items driving the {direction} call\n"
                f"2. Mentions any contradictory signals (e.g., bullish news but bearish technicals)\n"
                f"3. States the key risk to the prediction\n"
                f"Do NOT use emojis. Do NOT use Markdown or HTML. Plain professional English only."
            )
            with track_llm(self.db, model_name, "ml_narrative") as u:
                u.response = resp = await complete(model=model_name, prompt=prompt)
            return resp.text.strip()
        except Exception as e:
            log.error(f"Error generating narrative: {e}")
            return f"Model predicts {direction} based on current technical and sentiment features."

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
            log.error(f"Error in LLM prediction for {ticker}: {e}")
            return {"direction": "UP", "confidence": 0.5, "narrative": f"Unable to generate LLM analysis for {ticker}."}

    def _get_model_path(self, name: str, horizon_days: int = 1) -> Path:
        # The feature schema version is part of the filename so that changing the
        # feature set makes old models simply *unfindable* rather than deleted.
        # Reverting the code reverts the version and the previous models are live
        # again; without this, a rollback would leave nothing to roll back to.
        return self.models_dir / f"{name}_model_{horizon_days}d_v{FEATURE_SCHEMA_VERSION}.joblib"
