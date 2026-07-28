import pytest
import httpx
import asyncio

pytestmark = pytest.mark.asyncio

# 1. Add ticker to watchlist -> verify it immediately appears in the Live Market Grid.
async def test_cross_watchlist_market_sync(api_client):
    # Ensure ticker not in watchlist
    await api_client.delete("/api/watchlist/INTC")
    
    # Add to watchlist
    add_resp = await api_client.post("/api/watchlist", json={"ticker": "INTC"})
    assert add_resp.status_code == 200
    
    # Verify in market grid
    grid_resp = await api_client.get("/api/markets")
    assert grid_resp.status_code == 200
    grid_tickers = [item["ticker"] for item in grid_resp.json()["data"]]
    assert "INTC" in grid_tickers
    
    # Clean up
    await api_client.delete("/api/watchlist/INTC")


# 2. Run Prediction on a ticker -> verify prediction gets cached and is returned in subsequent Live Market Grid requests.
async def test_cross_predict_caching(api_client, parse_sse_stream):
    # Add AMZN to watchlist
    await api_client.post("/api/watchlist", json={"ticker": "AMZN"})
    
    # Run prediction stream
    async with api_client.stream("GET", "/api/predict/AMZN/stream?force_verdict=BUY") as response:
        events = await parse_sse_stream(response)
    assert events[-1]["event"] == "done"
    
    # Check market grid to verify it's cached
    grid_resp = await api_client.get("/api/markets")
    grid_data = grid_resp.json()["data"]
    amzn_item = next((item for item in grid_data if item["ticker"] == "AMZN"), None)
    
    assert amzn_item is not None
    assert amzn_item["cached_prediction"] is not None
    assert amzn_item["cached_prediction"]["direction"] == "BUY"
    assert amzn_item["cached_debate"] is not None
    assert amzn_item["cached_debate"]["verdict"] == "BUY"
    
    # Clean up
    await api_client.delete("/api/watchlist/AMZN")


# 3. Start Predict Stream for ticker -> trigger Chat Stream about the ticker -> verify both streams run concurrently.
async def test_cross_concurrent_streams(api_client, parse_sse_stream):
    # We can fetch both streams concurrently using asyncio.gather
    async def run_predict():
        async with api_client.stream("GET", "/api/predict/MSFT/stream") as response:
            return await parse_sse_stream(response)
            
    async def run_chat():
        payload = {"message": "Tell me about MSFT"}
        async with api_client.stream("POST", "/api/chat/stream", json=payload) as response:
            return await parse_sse_stream(response)
            
    predict_events, chat_events = await asyncio.gather(run_predict(), run_chat())
    assert predict_events[-1]["event"] == "done"
    assert chat_events[-1]["event"] == "done"


# 4. Run Predictions -> check if Accuracy endpoint reflects the new predictions.
async def test_cross_predict_accuracy_sync(api_client, parse_sse_stream):
    # Get initial accuracy
    acc_initial_resp = await api_client.get("/api/accuracy")
    initial_total = acc_initial_resp.json()["total_predictions"]
    
    # Run prediction
    async with api_client.stream("GET", "/api/predict/AAPL/stream") as response:
        events = await parse_sse_stream(response)
    assert events[-1]["event"] == "done"
    
    # Get updated accuracy
    acc_updated_resp = await api_client.get("/api/accuracy")
    updated_total = acc_updated_resp.json()["total_predictions"]
    
    assert updated_total == initial_total + 1


# 5. Use Chat Stream -> verify LLM token usage cost in `/api/usage` increases.
async def test_cross_chat_token_usage_increase(api_client, parse_sse_stream):
    # Get initial usage
    usage_init_resp = await api_client.get("/api/usage")
    init_cost = usage_init_resp.json()["total_cost_usd"]
    init_tokens = usage_init_resp.json()["total_tokens"]
    
    # Run chat
    payload = {"message": "Is TSLA a buy?"}
    async with api_client.stream("POST", "/api/chat/stream", json=payload) as response:
        events = await parse_sse_stream(response)
    assert events[-1]["event"] == "done"
    
    # Get updated usage
    usage_up_resp = await api_client.get("/api/usage")
    up_cost = usage_up_resp.json()["total_cost_usd"]
    up_tokens = usage_up_resp.json()["total_tokens"]
    
    assert up_tokens > init_tokens
    assert up_cost > init_cost


# 6. Run prediction stream -> verify LLM token usage cost in `/api/usage` increases.
async def test_cross_predict_token_usage_increase(api_client, parse_sse_stream):
    # Get initial usage
    usage_init_resp = await api_client.get("/api/usage")
    init_cost = usage_init_resp.json()["total_cost_usd"]
    init_tokens = usage_init_resp.json()["total_tokens"]
    
    # Run prediction
    async with api_client.stream("GET", "/api/predict/TSLA/stream") as response:
        events = await parse_sse_stream(response)
    assert events[-1]["event"] == "done"
    
    # Get updated usage
    usage_up_resp = await api_client.get("/api/usage")
    up_cost = usage_up_resp.json()["total_cost_usd"]
    up_tokens = usage_up_resp.json()["total_tokens"]
    
    assert up_tokens > init_tokens
    assert up_cost > init_cost


# 7. Add ticker -> delete ticker -> check watchlist -> check market grid -> verify deleted ticker is removed from both.
async def test_cross_add_delete_sync(api_client):
    # Add
    await api_client.post("/api/watchlist", json={"ticker": "NFLX"})
    
    # Verify present
    wl_resp = await api_client.get("/api/watchlist")
    assert "NFLX" in wl_resp.json()["data"]
    
    grid_resp = await api_client.get("/api/markets")
    grid_tickers = [item["ticker"] for item in grid_resp.json()["data"]]
    assert "NFLX" in grid_tickers
    
    # Delete
    await api_client.delete("/api/watchlist/NFLX")
    
    # Verify removed
    wl_resp2 = await api_client.get("/api/watchlist")
    assert "NFLX" not in wl_resp2.json()["data"]
    
    grid_resp2 = await api_client.get("/api/markets")
    grid_tickers2 = [item["ticker"] for item in grid_resp2.json()["data"]]
    assert "NFLX" not in grid_tickers2
