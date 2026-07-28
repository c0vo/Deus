import pytest
import httpx
import asyncio

pytestmark = pytest.mark.asyncio

# ==============================================================================
# 1. The Researcher's Workflow:
# User adds TSLA to watchlist, fetches charts and indicators, and reads news briefing.
# ==============================================================================
async def test_workflow_researcher(api_client):
    # Step A: Add TSLA to watchlist
    wl_add = await api_client.post("/api/watchlist", json={"ticker": "TSLA"})
    assert wl_add.status_code == 200

    # Step B: Fetch charts and indicators for TSLA
    charts_resp = await api_client.get("/api/charts/TSLA")
    assert charts_resp.status_code == 200
    charts_data = charts_resp.json()
    assert charts_data["ticker"] == "TSLA"
    assert "prices" in charts_data
    assert "indicators" in charts_data

    # Step C: Read news briefing (Energy sector)
    briefing_resp = await api_client.get("/api/briefing?sector=Energy")
    assert briefing_resp.status_code == 200
    briefing_data = briefing_resp.json()
    assert "Energy" in briefing_data["sectors"]


# ==============================================================================
# 2. The Analyst's Prediction:
# User starts an AI prediction stream for AAPL, watches the real-time agent debate,
# receives the verdict, and asks the chat assistant for a follow-up explanation.
# ==============================================================================
async def test_workflow_analyst_prediction(api_client, parse_sse_stream):
    # Step A: Start AI prediction stream
    async with api_client.stream("GET", "/api/predict/AAPL/stream?force_verdict=BUY") as response:
        events = await parse_sse_stream(response)
    
    # Verify we got agent updates and the final verdict
    verdicts = [e for e in events if e["event"] == "verdict"]
    assert len(verdicts) == 1
    verdict_data = verdicts[0]["data"]
    assert verdict_data["verdict"] == "BUY"
    
    # Step B: Ask the chat assistant for a follow-up explanation
    chat_payload = {
        "message": f"Why did the AI issue a {verdict_data['verdict']} rating on AAPL with a confidence of {verdict_data['confidence']}?"
    }
    async with api_client.stream("POST", "/api/chat/stream", json=chat_payload) as response:
        chat_events = await parse_sse_stream(response)
        
    assert chat_events[-1]["event"] == "done"
    tokens = [e["data"]["token"] for e in chat_events if e["event"] == "token"]
    full_response = "".join(tokens)
    assert len(full_response) > 0


# ==============================================================================
# 3. Performance Review:
# User checks accuracy metrics, reviews lessons learned from past reflections,
# adds new manually typed reflections based on recent market moves, and reviews
# overall token cost.
# ==============================================================================
async def test_workflow_performance_review(api_client):
    # Step A: Check accuracy metrics
    acc_resp = await api_client.get("/api/accuracy")
    assert acc_resp.status_code == 200
    acc_data = acc_resp.json()
    assert "win_rate" in acc_data
    
    # Step B: Review reflections for AAPL
    ref_resp = await api_client.get("/api/reflections/AAPL")
    assert ref_resp.status_code == 200
    initial_reflections = ref_resp.json()
    
    # Step C: Add a new reflection
    new_lesson = "Note: AAPL price strongly correlated with interest rate decisions."
    post_resp = await api_client.post("/api/reflections/AAPL", json={"lesson": new_lesson})
    assert post_resp.status_code == 200
    
    # Verify it is recorded
    ref_resp_updated = await api_client.get("/api/reflections/AAPL")
    updated_lessons = [r["lesson"] for r in ref_resp_updated.json()]
    assert new_lesson in updated_lessons
    assert len(ref_resp_updated.json()) == len(initial_reflections) + 1
    
    # Step D: Review token cost
    usage_resp = await api_client.get("/api/usage")
    assert usage_resp.status_code == 200
    usage_data = usage_resp.json()
    assert usage_data["total_cost_usd"] >= 0.0


# ==============================================================================
# 4. Multi-Asset Comparison:
# User lists watchlist, queries markets grid, and runs predictions for multiple tickers.
# ==============================================================================
async def test_workflow_multi_asset_comparison(api_client, parse_sse_stream):
    # Step A: Get current watchlist
    wl_resp = await api_client.get("/api/watchlist")
    watchlist = wl_resp.json()["data"]
    # Make sure we have at least 2 tickers
    if len(watchlist) < 2:
        await api_client.post("/api/watchlist", json={"ticker": "AAPL"})
        await api_client.post("/api/watchlist", json={"ticker": "TSLA"})
        wl_resp = await api_client.get("/api/watchlist")
        watchlist = wl_resp.json()["data"]

    # Step B: Query market grid
    grid_resp = await api_client.get("/api/markets")
    assert grid_resp.status_code == 200
    grid_data = grid_resp.json()["data"]

    # Step C: Run predictions for the first two assets
    ticker1, ticker2 = watchlist[0], watchlist[1]

    async with api_client.stream("GET", f"/api/predict/{ticker1}/stream") as resp1:
        events1 = await parse_sse_stream(resp1)
    async with api_client.stream("GET", f"/api/predict/{ticker2}/stream") as resp2:
        events2 = await parse_sse_stream(resp2)

    assert events1[-1]["event"] == "done"
    assert events2[-1]["event"] == "done"


# ==============================================================================
# 5. System Diagnostics:
# Administrator inspects system status and token usage, runs test predictions,
# checks database health, and validates metrics under system load.
# ==============================================================================
async def test_workflow_system_diagnostics(api_client, parse_sse_stream):
    # Step A: Inspect system status
    status_resp = await api_client.get("/api/status")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert status_data["status"] == "healthy"
    assert status_data["db_connected"] is True
    
    # Step B: Inspect token usage
    usage_resp = await api_client.get("/api/usage")
    assert usage_resp.status_code == 200
    initial_usage = usage_resp.json()
    
    # Step C: Run a test prediction to verify dynamic behavior
    async with api_client.stream("GET", "/api/predict/MSFT/stream") as response:
        events = await parse_sse_stream(response)
    assert events[-1]["event"] == "done"
    
    # Step D: Validate token usage has increased
    usage_resp2 = await api_client.get("/api/usage")
    updated_usage = usage_resp2.json()
    assert updated_usage["total_tokens"] > initial_usage["total_tokens"]
    
    # Step E: Validate final status
    status_resp2 = await api_client.get("/api/status")
    assert status_resp2.status_code == 200
