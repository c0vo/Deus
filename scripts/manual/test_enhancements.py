import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import asyncio
import json
import uuid
import datetime
from zoneinfo import ZoneInfo

from data.database import Database
from orchestrator.scheduler import run_reflection_job

class MockBot:
    async def send_message(self, chat_id, text, parse_mode=None, **kwargs):
        print("\n--- MOCK TELEGRAM ALERT SENT ---")
        print(f"Chat ID: {chat_id}")
        print(f"Parse Mode: {parse_mode}")
        print(f"Text:\n{text}")
        print("--------------------------------\n")

class MockAlertManager:
    def __init__(self):
        self.chat_id = "test_chat_123"
        self.bot = MockBot()

async def run_tests():
    print("Initializing test database (in-memory if possible or local test.db)...")
    db_path = "test_enhancements.db"
    if os.path.exists(db_path):
        os.remove(db_path)
        
    db = Database(db_path=db_path)
    db.initialize()
    
    # 1. Test 5-day Cache Logic
    print("\n--- Testing 5-Day Cache Logic ---")
    ticker = "TEST_TICKER"
    # Insert 3 days ago
    date_3_days_ago = (datetime.datetime.now() - datetime.timedelta(days=3)).strftime("%Y-%m-%d")
    old_advisory = {"final_advisory": "This is a 3-day old advisory"}
    db.set_cached_advisory(ticker, date_3_days_ago, json.dumps(old_advisory))
    
    cached_val = db.get_cached_advisory(ticker, days=5)
    if cached_val and cached_val.get("final_advisory") == "This is a 3-day old advisory":
        print("[SUCCESS] get_cached_advisory correctly fetched 3-day old cache (within 5 days).")
    else:
        print("[ERROR] get_cached_advisory failed to fetch 3-day old cache.")
        
    cached_val_strict = db.get_cached_advisory(ticker, days=1)
    if cached_val_strict is None:
        print("[SUCCESS] get_cached_advisory correctly ignored 3-day old cache when looking for 1 day.")
    else:
        print("[ERROR] get_cached_advisory fetched 3-day old cache when it shouldn't have.")

    # 2. Test DeepSeek Token Cost Logging
    print("\n--- Testing DeepSeek LLM Usage Logging ---")
    try:
        db.log_llm_usage(
            model_name="deepseek-v4-flash",
            operation="test_op",
            prompt_tokens=1_000_000,
            candidate_tokens=1_000_000
        )
        with db.connection() as conn:
            row = conn.execute("SELECT model_name, cost_usd FROM llm_usage_log ORDER BY id DESC LIMIT 1").fetchone()
            print(f"[SUCCESS] Logged DeepSeek usage. Model: {row['model_name']}, Cost: ${row['cost_usd']}")
    except Exception as e:
        print(f"[ERROR] Failed to log DeepSeek usage: {e}")

    # 3. Test run_reflection_job triggering re-prediction
    print("\n--- Testing run_reflection_job (Incorrect Prediction Triggering Re-Prediction) ---")
    # Insert a mocked unresolved prediction
    pred_id = str(uuid.uuid4())
    db.insert_prediction({
        "id": pred_id,
        "ticker": "AAPL",
        "predicted_direction": "UP",
        "confidence": 0.85,
        "horizon_days": 1,
        "model_type": "multi_agent",
        "llm_narrative": "AAPL looks very strong right now due to positive earnings.",
        "resolve_after": (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    })
    
    # Resolve it as incorrect (actual direction DOWN)
    db.resolve_prediction(pred_id, actual_direction="DOWN", actual_change_pct=-2.5, is_correct=False)
    
    # Run reflection job
    alert_mgr = MockAlertManager()
    
    # Note: run_reflection_job makes real API calls to DeepSeek and Gemini.
    # To avoid real API calls taking too long or failing if no key, we'll run it as is,
    # but the test might fail if keys aren't set in environment. Let's see if it runs.
    print("Running run_reflection_job... (This may make real API calls to DeepSeek/Gemini)")
    try:
        await run_reflection_job(db, alert_manager=alert_mgr)
        
        # Verify reflection was inserted
        with db.connection() as conn:
            row = conn.execute("SELECT lesson_learned FROM reflection_log WHERE prediction_id = ?", (pred_id,)).fetchone()
            if row:
                print(f"[SUCCESS] Reflection successfully inserted: '{row['lesson_learned']}'")
            else:
                print("[ERROR] Reflection was NOT inserted into DB.")
                
        # Verify cache was updated for AAPL today
        today_str = datetime.datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d")
        cache_row = db.get_cached_advisory("AAPL", days=1)
        if cache_row:
            print("[SUCCESS] 5-Day Cache was successfully overwritten with new advisory after re-prediction.")
            print(f"New Advisory: {cache_row.get('final_advisory')[:100]}...")
        else:
            print("[ERROR] Cache was NOT updated after reflection job.")
            
    except Exception as e:
        print(f"[ERROR] run_reflection_job failed with error: {e}")
        
    print("\nTests complete!")
    if os.path.exists(db_path):
        os.remove(db_path)

if __name__ == "__main__":
    asyncio.run(run_tests())
