import sys
import asyncio
from pathlib import Path

# Add root directory to sys.path to allow imports when run directly
sys.path.append(str(Path(__file__).parent.parent))

from data.database import Database
from config.settings import settings
from pipeline.predictor import StockPredictor

async def main():
    print("Testing ML Prediction Engine Orchestration...")

    # Initialize DB
    db = Database(settings.db_path)
    db.initialize()

    # 1. Predict
    print("\n--- Running Prediction for AAPL ---")
    predictor = StockPredictor(db)

    # Train dummy model just to have one available for test
    await predictor.train_model("AAPL", "per_ticker")

    # Predict 1 day
    pred_result = await predictor.predict("AAPL", 1)
    print(f"Prediction Result: {pred_result['predicted_direction']} (Confidence: {pred_result['confidence']:.2f})")
    print(f"Narrative: {pred_result['llm_narrative']}")

    print("\nOrchestration test completed successfully.")

if __name__ == "__main__":
    asyncio.run(main())

