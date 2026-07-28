"""
Integration test script to verify all REST and SSE endpoints for Milestone 1.
Spins up the server in a subprocess, runs the test suite, and shuts down the server.
"""

import os
import sys
import time
import json
import asyncio
import subprocess
import httpx

# Ensure project root is in python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings

host = "127.0.0.1" if settings.api_host == "0.0.0.0" else settings.api_host
BASE_URL = f"http://{host}:{settings.api_port}"

async def run_tests():
    print(f"=== Starting API Integration Tests on {BASE_URL} ===")
    
    async with httpx.AsyncClient(timeout=90.0) as client:
        # 1. Status Check
        print("Testing GET /api/status...")
        r = await client.get(f"{BASE_URL}/api/status")
        assert r.status_code == 200, f"Status failed: {r.status_code}"
        stats = r.json()
        assert "total_articles" in stats, "Invalid stats payload"
        print("[OK] GET /api/status passed.")

        # 2. Usage Check
        print("Testing GET /api/usage...")
        r = await client.get(f"{BASE_URL}/api/usage")
        assert r.status_code == 200, f"Usage failed: {r.status_code}"
        usage = r.json()
        assert "total_tokens" in usage, "Invalid usage payload"
        assert "total_cost_usd" in usage, "Invalid usage payload"
        print("[OK] GET /api/usage passed.")

        # 3. Watchlist CRUD
        print("Testing Watchlist CRUD...")
        TEST_TICKER = "TESTXYZ"
        # Clean up test ticker just in case it exists from a prior failed run
        await client.delete(f"{BASE_URL}/api/watchlist/{TEST_TICKER}")
        await asyncio.sleep(0.3)  # Allow SQLite WAL write to propagate
        # Add test ticker
        r = await client.post(f"{BASE_URL}/api/watchlist", json={"ticker": TEST_TICKER})
        assert r.status_code == 200, f"Add watchlist failed: {r.status_code}"
        resp = r.json()
        assert resp["success"] is True, f"Failed to add ticker: {resp}"

        # Verify test ticker is in watchlist
        r = await client.get(f"{BASE_URL}/api/watchlist")
        assert r.status_code == 200
        assert TEST_TICKER in r.json()["data"], f"{TEST_TICKER} not found in watchlist"

        # Remove test ticker
        r = await client.delete(f"{BASE_URL}/api/watchlist/{TEST_TICKER}")
        assert r.status_code == 200
        assert r.json()["success"] is True, "Failed to remove ticker"
        print("[OK] Watchlist CRUD passed.")

        # 4. Markets Grid
        print("Testing GET /api/markets...")
        r = await client.get(f"{BASE_URL}/api/markets")
        assert r.status_code == 200, f"Markets failed: {r.status_code}"
        data = r.json()["data"]
        assert isinstance(data, list), "Markets data should be a list"
        if data:
            assert "ticker" in data[0], "Missing ticker in market grid"
            assert "price" in data[0], "Missing price in market grid"
        print("[OK] GET /api/markets passed.")

        # 5. News Briefings
        print("Testing GET /api/briefing...")
        r = await client.get(f"{BASE_URL}/api/briefing")
        assert r.status_code == 200, f"Briefing failed: {r.status_code}"
        assert "data" in r.json(), "Missing data in briefings"
        print("[OK] GET /api/briefing passed.")

        # 5b. Trending Tickers
        print("Testing GET /api/trending...")
        r = await client.get(f"{BASE_URL}/api/trending?hours=24")
        assert r.status_code == 200, f"Trending failed: {r.status_code}"
        assert "data" in r.json(), "Missing data in trending response"
        print("[OK] GET /api/trending passed.")

        # 6. Technical Indicators & Chart Coordinates
        print("Testing GET /api/charts/AAPL...")
        r = await client.get(f"{BASE_URL}/api/charts/AAPL?days=30")
        assert r.status_code == 200, f"Charts failed: {r.status_code}"
        chart_data = r.json()["data"]
        assert isinstance(chart_data, list), "Charts data should be a list"
        if chart_data:
            assert "time" in chart_data[0]
            assert "sma20" in chart_data[0]
            assert "macd" in chart_data[0]
            assert "rsi14" in chart_data[0]
        print("[OK] GET /api/charts/AAPL passed.")

        # 7. Accuracy
        print("Testing GET /api/accuracy...")
        r = await client.get(f"{BASE_URL}/api/accuracy?ticker=AAPL")
        assert r.status_code == 200, f"Accuracy failed: {r.status_code}"
        acc = r.json()
        assert "accuracy" in acc
        assert "recent" in acc
        print("[OK] GET /api/accuracy passed.")

        # 9. Reflections (POST & GET)
        print("Testing Reflections GET and POST...")
        # POST
        r = await client.post(f"{BASE_URL}/api/reflections/AAPL", json={
            "lesson_learned": "Integration test lesson",
            "was_successful": True
        })
        assert r.status_code == 200, f"Reflections POST failed: {r.status_code}"
        assert r.json()["success"] is True

        # GET
        r = await client.get(f"{BASE_URL}/api/reflections/AAPL")
        assert r.status_code == 200, f"Reflections GET failed: {r.status_code}"
        reflections = r.json()["data"]
        assert isinstance(reflections, list)
        assert any(ref["lesson_learned"] == "Integration test lesson" for ref in reflections)
        print("[OK] Reflections GET and POST passed.")

        # 10. Predict SSE Stream (GET /api/predict/AAPL/stream)
        print("Testing Predict SSE Stream...")
        async with client.stream("GET", f"{BASE_URL}/api/predict/AAPL/stream") as response:
            assert response.status_code == 200, f"Predict SSE failed: {response.status_code}"
            assert response.headers["content-type"].startswith("text/event-stream")
            events = []
            async for line in response.aiter_lines():
                if line.startswith("event:"):
                    events.append(line.split(":", 1)[1].strip())
            print(f"Predict stream events: {events}")
            assert "agent_update" in events or "verdict" in events
            assert "done" in events
        print("[OK] Predict SSE Stream passed.")

        # 11. Chat SSE Stream (POST /api/chat/stream)
        print("Testing Chat SSE Stream...")
        async with client.stream("POST", f"{BASE_URL}/api/chat/stream", json={"message": "Is AAPL a buy?"}) as response:
            assert response.status_code == 200, f"Chat SSE failed: {response.status_code}"
            assert response.headers["content-type"].startswith("text/event-stream")
            events = []
            async for line in response.aiter_lines():
                if line.startswith("event:"):
                    events.append(line.split(":", 1)[1].strip())
            print(f"Chat stream events: {events}")
            assert "step" in events
            assert "token" in events or "error" in events
            assert "done" in events
        print("[OK] Chat SSE Stream passed.")

    print("\n=== All Integration Tests Completed Successfully! ===")

def main():
    # Spin up the server in a subprocess
    print("Starting FastAPI server in subprocess...")
    proc = subprocess.Popen(
        [sys.executable, "-u", "main.py"],
        stdout=sys.stdout,
        stderr=sys.stderr
    )

    try:
        # Poll server until active (up to 15 seconds)
        print("Waiting for server to become healthy...")
        start_time = time.time()
        server_ready = False
        while time.time() - start_time < 20:
            try:
                r = httpx.get(f"{BASE_URL}/api/status")
                if r.status_code == 200:
                    server_ready = True
                    break
            except Exception:
                pass
            time.sleep(0.5)

        if not server_ready:
            print("[FAIL] Server failed to start or become healthy within timeout.")
            sys.exit(1)

        # Run async test suite
        asyncio.run(run_tests())
        exit_code = 0
    except Exception as e:
        import traceback
        print(f"\n[FAIL] Integration Test Execution Failed: {e}", file=sys.stderr)
        traceback.print_exc()
        exit_code = 1
    finally:
        print("Shutting down FastAPI server...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("Force-killing FastAPI server...")
            proc.kill()
            proc.wait()
        print("Server stopped.")
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
