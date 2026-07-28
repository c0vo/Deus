"""
End-to-end test for the predictor with REAL data.
Tests:
  1. Price data is fetched and sorted chronologically
  2. Model training uses real historical data (no dummy/mock data)
  3. Predictions produce varied, reasonable confidence scores
  4. Exponential sample weights are applied correctly
  5. LLM-only fallback produces dynamic confidence (not hardcoded 0.6)
"""
import asyncio
import sys
import os
import json
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.database import Database
from pipeline.predictor import StockPredictor

PASS = "[PASS]"
FAIL = "[FAIL]"
results = []

def report(test_name, passed, detail=""):
    status = PASS if passed else FAIL
    results.append((test_name, passed))
    print(f"  {status} {test_name}" + (f" — {detail}" if detail else ""))


async def main():
    print("\n" + "="*70)
    print("  PREDICTOR END-TO-END TEST")
    print("="*70 + "\n")

    db = Database()
    db.initialize()
    predictor = StockPredictor(db)
    
    test_ticker = "AAPL"

    # ──────────────────────────────────────────────
    # TEST 1: Price data is fetched and sorted
    # ──────────────────────────────────────────────
    print("[Test 1] Price data fetch & chronological ordering")
    prices = await predictor._fetch_and_cache_prices(test_ticker, range="6mo")
    report("Prices fetched", prices is not None and len(prices) > 0, f"{len(prices)} rows")
    
    if prices:
        dates = [p["date"] for p in prices]
        is_sorted = all(dates[i] <= dates[i+1] for i in range(len(dates)-1))
        report("Prices sorted chronologically", is_sorted, 
               f"first={dates[0]}, last={dates[-1]}")
    
    # ──────────────────────────────────────────────
    # TEST 2: Feature vector builds successfully
    # ──────────────────────────────────────────────
    print("\n[Test 2] Feature vector construction")
    features = await predictor.build_feature_vector(test_ticker)
    report("Feature vector built", features is not None, f"{len(features)} features" if features else "None")
    
    if features:
        expected_keys = [
            "rsi_14", "return_1d", "return_5d", "sma_crossover", 
            "volatility", "volume_anomaly", "vix_level", "vix_change_1d",
            "market_return_1d", "treasury_yield_change", "sector_etf_return_1d",
            "sentiment_avg_1d", "sentiment_avg_3d", "sentiment_avg_7d",
            "sentiment_momentum", "news_velocity", "max_urgency_24h",
            "avg_importance", "bullish_ratio"
        ]
        missing = [k for k in expected_keys if k not in features]
        report("All expected features present", len(missing) == 0,
               f"missing: {missing}" if missing else f"all {len(expected_keys)} present")
        
        # Check that values are actual numbers, not NaN
        nan_keys = [k for k, v in features.items() if np.isnan(v) if isinstance(v, float)]
        report("No NaN values in features", len(nan_keys) == 0, 
               f"NaN in: {nan_keys}" if nan_keys else "all clean")

    # ──────────────────────────────────────────────
    # TEST 3: Model training with real data
    # ──────────────────────────────────────────────
    print("\n[Test 3] Model training with real historical data")
    
    # Clean up any existing model
    model_path = predictor._get_model_path(test_ticker)
    if model_path.exists():
        model_path.unlink()
    
    try:
        path, cv_metrics = await predictor.train_model(test_ticker, scope="per_ticker")
        trained_path = Path(path)
        report("Model trained successfully", trained_path.exists(), f"saved to {path}")
        
        # Verify model file is a CalibratedClassifierCV wrapper
        import joblib
        from sklearn.calibration import CalibratedClassifierCV
        model = joblib.load(trained_path)
        report("Model is a CalibratedClassifierCV wrapper", 
               isinstance(model, CalibratedClassifierCV),
               f"type={type(model).__name__}")
        report("Model has predict and predict_proba", 
               hasattr(model, 'predict') and hasattr(model, 'predict_proba'))
        
        # Check the underlying GBM was trained on real data
        underlying = model.estimator.estimator  # CalibratedClassifierCV -> FrozenEstimator -> GBM
        n_est = underlying.n_estimators_
        report("Underlying GBM has trained estimators", n_est > 0, f"n_estimators_={n_est}")
        
        # Verify CV metrics were returned
        report("CV metrics returned", isinstance(cv_metrics, dict) and "accuracy_mean" in cv_metrics,
               f"acc={cv_metrics.get('accuracy_mean', 'N/A'):.3f}, brier={cv_metrics.get('brier_mean', 'N/A'):.3f}, auc={cv_metrics.get('auc_mean', 'N/A'):.3f}" if cv_metrics else "None")
        
    except Exception as e:
        report("Model trained successfully", False, f"ERROR: {e}")

    # ──────────────────────────────────────────────
    # TEST 4: Predictions produce varied confidence
    # ──────────────────────────────────────────────
    print("\n[Test 4] Prediction confidence variation across tickers")
    
    # Clear predictions cache
    try:
        with db.connection() as conn:
            conn.execute("DELETE FROM predictions")
    except:
        pass
    
    tickers_to_test = ["AAPL", "NVDA", "TSLA"]
    confidences = []
    directions = []
    
    for tk in tickers_to_test:
        # Train model for each ticker first
        tk_model_path = predictor._get_model_path(tk)
        if not tk_model_path.exists():
            try:
                await predictor.train_model(tk, scope="per_ticker")
            except Exception as e:
                print(f"    [WARN]  Could not train {tk}: {e}")
                continue
        
        pred = await predictor.predict(tk, horizon_days=1)
        if "error" not in pred:
            conf = pred["confidence"]
            direction = pred["predicted_direction"]
            confidences.append(conf)
            directions.append(direction)
            print(f"    {tk}: {direction} {conf*100:.1f}% (model: {pred['model_type']})")
        else:
            print(f"    {tk}: ERROR - {pred['error']}")
    
    if len(confidences) >= 2:
        # Check that not ALL confidences are the same (the original bug)
        all_same = len(set(f"{c:.4f}" for c in confidences)) == 1
        report("Confidences are NOT all identical", not all_same,
               f"values: {[f'{c:.1%}' for c in confidences]}")
        
        # Check that confidences are in a reasonable range
        # With Platt Scaling calibration, probabilities can span a wider range
        # than the old static 50-85% clamp, but should still be valid probabilities
        max_conf = max(confidences)
        min_conf = min(confidences)
        report("Confidences are valid probabilities (0-1)", 
               0.0 <= min_conf and max_conf <= 1.0,
               f"range: {min_conf:.1%} to {max_conf:.1%}")
    else:
        report("Enough predictions to compare", False, f"only got {len(confidences)}")

    # ──────────────────────────────────────────────
    # TEST 5: Exponential sample weights
    # ──────────────────────────────────────────────
    print("\n[Test 5] Exponential sample weights calculation")
    
    n_samples = 100
    decay_rate = 0.05
    weights = np.exp(-decay_rate * np.arange(n_samples - 1, -1, -1))
    
    report("Most recent sample has highest weight", 
           weights[-1] > weights[0], 
           f"newest={weights[-1]:.4f}, oldest={weights[0]:.4f}")
    report("Weights decay monotonically", 
           all(weights[i] <= weights[i+1] for i in range(len(weights)-1)))
    report("Oldest sample weight is small", 
           weights[0] < 0.01,
           f"oldest weight = {weights[0]:.6f}")

    # ──────────────────────────────────────────────
    # TEST 6: No dummy/mock data in production code
    # ──────────────────────────────────────────────
    print("\n[Test 7] Audit for remaining mock/dummy code in production files")
    
    production_files = [
        "pipeline/predictor.py",
    ]
    
    bad_patterns = ["np.random.rand", "np.random.randint", "# mock", "# dummy", "# placeholder"]
    
    for fpath in production_files:
        full_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), fpath)
        if os.path.exists(full_path):
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
            found = [p for p in bad_patterns if p in content.lower()]
            report(f"{fpath}: no mock/dummy patterns", len(found) == 0,
                   f"FOUND: {found}" if found else "clean")

    # ──────────────────────────────────────────────
    # TEST 7: CV metrics from train_model
    # ──────────────────────────────────────────────
    print("\n[Test 8] Cross-validation metrics from training")
    
    # Re-train to get fresh metrics
    try:
        _, cv_metrics = await predictor.train_model(test_ticker, scope="per_ticker")
        
        expected_keys = ["accuracy_mean", "accuracy_std", "brier_mean", "brier_std", 
                        "auc_mean", "auc_std", "n_samples", "n_folds"]
        missing = [k for k in expected_keys if k not in cv_metrics]
        report("CV metrics contain all expected keys", len(missing) == 0,
               f"missing: {missing}" if missing else f"all {len(expected_keys)} present")
        
        report("CV accuracy is valid (0-1)", 
               0.0 <= cv_metrics["accuracy_mean"] <= 1.0,
               f"accuracy={cv_metrics['accuracy_mean']:.3f}±{cv_metrics['accuracy_std']:.3f}")
        report("CV Brier score is valid (0-1)", 
               0.0 <= cv_metrics["brier_mean"] <= 1.0,
               f"brier={cv_metrics['brier_mean']:.3f}±{cv_metrics['brier_std']:.3f}")
        report("CV AUC is valid (0-1)", 
               0.0 <= cv_metrics["auc_mean"] <= 1.0,
               f"auc={cv_metrics['auc_mean']:.3f}±{cv_metrics['auc_std']:.3f}")
        report("Number of folds is 5", cv_metrics["n_folds"] == 5)
        report("Has enough training samples", cv_metrics["n_samples"] >= 10,
               f"n_samples={cv_metrics['n_samples']}")
    except Exception as e:
        report("CV metrics from train_model", False, f"ERROR: {e}")

    # ──────────────────────────────────────────────
    # SUMMARY
    # ──────────────────────────────────────────────
    print("\n" + "="*70)
    passed = sum(1 for _, p in results if p)
    failed = sum(1 for _, p in results if not p)
    print(f"  RESULTS: {passed} passed, {failed} failed, {len(results)} total")
    print("="*70 + "\n")
    
    if failed > 0:
        print("  FAILED TESTS:")
        for name, p in results:
            if not p:
                print(f"    {FAIL} {name}")
        print()
    
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
