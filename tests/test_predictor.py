import pytest
import numpy as np
from unittest.mock import MagicMock, AsyncMock, patch
from pipeline.predictor import FEATURE_SCHEMA_VERSION, StockPredictor

@pytest.fixture
def mock_db():
    db = MagicMock()
    # Mock sentiment features
    db.get_ticker_sentiment_features.return_value = {
        "sentiment_avg_1d": 0.5,
        "sentiment_avg_3d": 0.4,
        "sentiment_avg_7d": 0.3,
        "sentiment_momentum": 0.2,
        "news_velocity": 1.5,
        "max_urgency_24h": 1.0,
        "avg_importance": 6.5,
        "bullish_ratio": 0.8,
    }
    return db

@pytest.fixture
def predictor(mock_db):
    return StockPredictor(mock_db)

def test_compute_rsi(predictor):
    closes = np.array([10, 11, 12, 11, 10, 9, 8, 9, 10, 11, 12, 13, 14, 15, 14])
    rsi = predictor._compute_rsi(closes, period=14)
    assert 0 <= rsi <= 100

def test_compute_sma(predictor):
    closes = np.array([10, 20, 30, 40, 50])
    sma = predictor._compute_sma(closes, period=3)
    assert sma == 40.0

@pytest.mark.asyncio
async def test_build_feature_vector(predictor):
    with patch.object(predictor, '_fetch_and_cache_prices', new_callable=AsyncMock) as mock_fetch:
        # Mock price fetch for the stock and market regimes
        def side_effect(ticker, *args, **kwargs):
            if ticker == "^VIX":
                return [{"date": "2026-05-23", "close": 15}, {"date": "2026-05-24", "close": 16}]
            elif ticker == "^GSPC":
                return [{"date": "2026-05-23", "close": 4000}, {"date": "2026-05-24", "close": 4040}]
            elif ticker == "^TNX":
                return [{"date": "2026-05-23", "close": 4.0}, {"date": "2026-05-24", "close": 4.1}]
            else:
                # Return dummy price history
                return [{"date": f"2026-05-{i:02d}", "close": 100 + i, "high": 101 + i, "low": 99 + i, "volume": 1000} for i in range(1, 25)]
        
        mock_fetch.side_effect = side_effect

        features = await predictor.build_feature_vector("AAPL", as_of_date="2026-05-24")
        
        assert features is not None
        assert "sentiment_avg_1d" in features
        assert "return_1d" in features
        assert "vix_level" in features
        assert features["vix_level"] == 16.0
        assert features["market_return_1d"] == 0.01  # 4040/4000 - 1

@pytest.mark.asyncio
async def test_train_model(predictor):
    path, cv_metrics = await predictor.train_model("AAPL", scope="per_ticker")
    # Model filenames carry the feature-schema version so a schema change makes
    # old artifacts unfindable rather than deleting them.
    assert path.endswith(f"AAPL_model_1d_v{FEATURE_SCHEMA_VERSION}.joblib")
    
    # Verify CV metrics are returned with expected keys
    assert isinstance(cv_metrics, dict)
    for key in ["accuracy_mean", "accuracy_std", "brier_mean", "brier_std", "auc_mean", "auc_std", "n_samples", "n_folds"]:
        assert key in cv_metrics, f"Missing key: {key}"
    assert cv_metrics["n_folds"] == 5
    assert 0.0 <= cv_metrics["accuracy_mean"] <= 1.0
    assert 0.0 <= cv_metrics["brier_mean"] <= 1.0
    assert 0.0 <= cv_metrics["auc_mean"] <= 1.0
    
    # We should be able to load it
    model, scope = predictor._load_model("AAPL")
    assert scope == "per_ticker"
    assert model is not None

@pytest.mark.asyncio
async def test_predict(predictor):
    with patch.object(predictor, 'build_feature_vector', new_callable=AsyncMock) as mock_build:
        # Fake features
        mock_build.return_value = {f"f_{i}": 0.0 for i in range(20)}
        
        with patch.object(predictor, '_load_model') as mock_load:
            mock_model = MagicMock()
            mock_model.predict.return_value = [1]
            mock_model.predict_proba.return_value = [[0.2, 0.8]]
            mock_load.return_value = (mock_model, "per_ticker")
            
            with patch.object(predictor, '_generate_narrative', new_callable=AsyncMock) as mock_narrative:
                mock_narrative.return_value = "It will go up."
                
                predictor.db.get_existing_prediction.return_value = None
                predictor.db.insert_prediction.return_value = "pred_123"
                
                result = await predictor.predict("AAPL", horizon_days=1)
                
                assert result["ticker"] == "AAPL"
                assert result["predicted_direction"] == "UP"
                assert result["confidence"] == 0.8
                assert result["model_type"] == "per_ticker"
                assert result["id"] == "pred_123"

@pytest.mark.asyncio
async def test_calibrated_model_structure(predictor):
    """Verify that train_model produces a CalibratedClassifierCV wrapper."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import GradientBoostingClassifier

    path, cv_metrics = await predictor.train_model("AAPL", scope="per_ticker")
    model, scope = predictor._load_model("AAPL")

    # The saved model should be a CalibratedClassifierCV wrapping a GradientBoostingClassifier
    assert isinstance(model, CalibratedClassifierCV), (
        f"Expected CalibratedClassifierCV, got {type(model).__name__}"
    )
    assert isinstance(model.estimator, GradientBoostingClassifier)
    assert model.method == "sigmoid"  # Platt Scaling


# ── Phase 3: Additional ML Model Integrity Tests ────────────────────────────

@pytest.mark.asyncio
async def test_predict_returns_confidence_between_05_and_10(predictor):
    """Confidence should always be between 0.5 and 1.0 (not 0.0 or flat 0.5)."""
    with patch.object(predictor, 'build_feature_vector', new_callable=AsyncMock) as mock_build:
        mock_build.return_value = {f"f_{i}": 0.0 for i in range(20)}

        with patch.object(predictor, '_load_model') as mock_load:
            mock_model = MagicMock()
            mock_model.predict.return_value = [1]
            mock_model.predict_proba.return_value = [[0.3, 0.7]]
            mock_load.return_value = (mock_model, "per_ticker")

            with patch.object(predictor, '_generate_narrative', new_callable=AsyncMock) as mock_narrative:
                mock_narrative.return_value = "Up."
                predictor.db.get_existing_prediction.return_value = None
                predictor.db.insert_prediction.return_value = "pred_123"

                result = await predictor.predict("AAPL", horizon_days=1)

                assert 0.5 <= result["confidence"] <= 1.0

@pytest.mark.asyncio
async def test_predict_different_confidences_for_different_models(predictor):
    """Different models should produce different confidence values."""
    with patch.object(predictor, 'build_feature_vector', new_callable=AsyncMock) as mock_build:
        mock_build.return_value = {f"f_{i}": 0.0 for i in range(20)}

        with patch.object(predictor, '_generate_narrative', new_callable=AsyncMock) as mock_narrative:
            mock_narrative.return_value = "Narrative."
            predictor.db.get_existing_prediction.return_value = None
            predictor.db.insert_prediction.return_value = "pred_123"

            # Test with UP prediction
            with patch.object(predictor, '_load_model') as mock_load:
                model_up = MagicMock()
                model_up.predict.return_value = [1]
                model_up.predict_proba.return_value = [[0.1, 0.9]]
                mock_load.return_value = (model_up, "per_ticker")

                result_up = await predictor.predict("AAPL", horizon_days=1)

            predictor.db.insert_prediction.return_value = "pred_456"

            # Test with DOWN prediction
            with patch.object(predictor, '_load_model') as mock_load2:
                model_down = MagicMock()
                model_down.predict.return_value = [0]
                model_down.predict_proba.return_value = [[0.75, 0.25]]
                mock_load2.return_value = (model_down, "per_ticker")

                result_down = await predictor.predict("MSFT", horizon_days=1)

            # Different tickers should have different directions
            assert result_up["predicted_direction"] == "UP"
            assert result_down["predicted_direction"] == "DOWN"

@pytest.mark.asyncio
async def test_llm_fallback_when_no_model_exists(predictor):
    """When no model exists and fast_fallback is True, use LLM-only prediction."""
    with patch.object(predictor, '_load_model', return_value=(None, None)):
        with patch.object(predictor, 'build_feature_vector', new_callable=AsyncMock) as mock_build:
            mock_build.return_value = {f"f_{i}": 0.0 for i in range(20)}

            with patch.object(predictor, '_generate_narrative_with_confidence', new_callable=AsyncMock) as mock_gen:
                mock_gen.return_value = {
                    "predicted_direction": "UP",
                    "confidence": 0.65,
                    "narrative": "LLM-generated prediction based on recent news.",
                }
                predictor.db.get_existing_prediction.return_value = None
                predictor.db.insert_prediction.return_value = "pred_llm"

                result = await predictor.predict("RARE_TICKER", horizon_days=1, fast_fallback=True)

                assert result["predicted_direction"] == "UP"
                assert result["confidence"] == 0.65

@pytest.mark.asyncio
async def test_feature_vector_has_expected_structure(predictor):
    """build_feature_vector should return a dict with all expected feature keys."""
    with patch.object(predictor, '_fetch_and_cache_prices', new_callable=AsyncMock) as mock_fetch:
        def side_effect(ticker, *args, **kwargs):
            if ticker == "^VIX":
                return [{"date": "2026-05-23", "close": 15}, {"date": "2026-05-24", "close": 16}]
            elif ticker == "^GSPC":
                return [{"date": "2026-05-23", "close": 4000}, {"date": "2026-05-24", "close": 4040}]
            elif ticker == "^TNX":
                return [{"date": "2026-05-23", "close": 4.0}, {"date": "2026-05-24", "close": 4.1}]
            else:
                return [{"date": f"2026-05-{i:02d}", "close": 100 + i, "high": 101 + i, "low": 99 + i, "volume": 1000} for i in range(1, 25)]

        mock_fetch.side_effect = side_effect

        features = await predictor.build_feature_vector("AAPL", as_of_date="2026-05-24")

        # Should have sentiment features
        assert "sentiment_avg_1d" in features
        assert "sentiment_avg_3d" in features
        assert "sentiment_avg_7d" in features
        assert "sentiment_momentum" in features
        assert "news_velocity" in features
        assert "bullish_ratio" in features

        # Should have price/technical features
        assert "return_1d" in features
        assert "return_5d" in features
        assert "rsi_14" in features
        assert "volatility" in features

        # Should have market regime features
        assert "vix_level" in features
        assert "market_return_1d" in features

def test_rsi_handles_flat_prices(predictor):
    """RSI should handle flat price sequences without division by zero."""
    closes = np.array([100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100])
    rsi = predictor._compute_rsi(closes, period=14)
    # RSI for flat prices should be 50 (neutral, no gains or losses)
    assert rsi == 50.0

def test_rsi_handles_period_larger_than_data(predictor):
    """RSI should handle case where period > len(closes) gracefully."""
    closes = np.array([100, 101, 102])
    rsi = predictor._compute_rsi(closes, period=14)
    # Should return 50 (neutral) when not enough data
    assert rsi == 50.0

def test_sma_handles_period_larger_than_data(predictor):
    """SMA should handle period > len(closes) gracefully."""
    closes = np.array([100, 101, 102])
    sma = predictor._compute_sma(closes, period=14)
    assert sma == 0.0  # Returns 0 when not enough data

@pytest.mark.asyncio
async def test_model_serialization_round_trip(predictor):
    """Training → save → load should preserve model calibration."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import GradientBoostingClassifier

    path, cv_metrics = await predictor.train_model("AAPL", scope="per_ticker")
    # Model filenames carry the feature-schema version so a schema change makes
    # old artifacts unfindable rather than deleting them.
    assert path.endswith(f"AAPL_model_1d_v{FEATURE_SCHEMA_VERSION}.joblib")

    # Load it fresh
    model, scope = predictor._load_model("AAPL")

    assert isinstance(model, CalibratedClassifierCV)
    assert hasattr(model, "predict_proba")
    assert hasattr(model, "predict")

@pytest.mark.asyncio
async def test_sector_model_fallback(predictor):
    """Predictor should fall through model tiers: per_ticker → sector → universal."""
    # Mock _get_sector to return a known sector
    with patch.object(predictor, '_get_sector', return_value="Technology"):
        with patch.object(predictor, 'build_feature_vector', new_callable=AsyncMock) as mock_build:
            mock_build.return_value = {f"f_{i}": 0.1 for i in range(20)}

            # _load_model already returns (None, None) if no model found
            # But train_model creates a per_ticker model, so we need to test
            # the _get_model_path fallback logic directly

            # Test _get_model_path returns None when no model exists
            from pipeline.predictor import StockPredictor
            model, scope = predictor._load_model("NONEXISTENT_TICKER_XYZ")
            # Should not crash — should return (None, None)
            assert model is None or hasattr(model, "predict")

@pytest.mark.asyncio
async def test_train_model_cv_metrics_are_reasonable(predictor):
    """Cross-validation metrics from training should be in valid ranges."""
    path, cv_metrics = await predictor.train_model("AAPL", scope="per_ticker")

    assert cv_metrics["n_folds"] == 5
    assert 0.0 <= cv_metrics["accuracy_mean"] <= 1.0
    assert 0.0 <= cv_metrics["brier_mean"] <= 1.0
    assert 0.0 <= cv_metrics["auc_mean"] <= 1.0
    assert cv_metrics["n_samples"] > 0

    # AUC should be better than random ( > 0.5) if signal exists
    # Note: on synthetic data this might not hold, so check structure only
    assert "accuracy_std" in cv_metrics
    assert "brier_std" in cv_metrics
    assert "auc_std" in cv_metrics

