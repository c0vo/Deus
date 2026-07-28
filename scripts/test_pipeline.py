import asyncio
import os
import sys
import logging

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Fix Windows console encoding for emojis
import io
if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding='utf-8')

from data.database import Database
from pipeline.predictor import StockPredictor
from config.logging_config import get_logger

# Optional: configure root logger to see internal logs
logging.basicConfig(level=logging.INFO)

async def main():
    db = Database()
    db.initialize()
    predictor = StockPredictor(db)
    
    async def progress_callback(msg: str):
        print(f"[{msg}]")
        
    tickers = ["AAPL", "NVDA"]
    for t in tickers:
        print(f"\n==============================")
        print(f"Testing Pipeline for {t}")
        print(f"==============================")
        
        result = await predictor.predict_with_agents(t, progress_callback=progress_callback)
        
        print("\n--- DEBATE HISTORY ---")
        history = result.get("debate_history", [])
        for round_text in history:
            print(f"\n{round_text}")
            print("-" * 40)
            
        print("\n--- FINAL ADVISORY ---")
        print(result.get("final_advisory", "No advisory generated."))
        print("\n==============================\n")

if __name__ == "__main__":
    asyncio.run(main())
