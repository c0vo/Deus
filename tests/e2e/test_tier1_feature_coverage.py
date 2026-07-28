import pytest
import httpx

pytestmark = pytest.mark.asyncio

# ==========================================
# FEATURE 1: WATCHLIST CRUD (5 tests)
# ==========================================

async def test_watchlist_get_initial(api_client):
    response = await api_client.get("/api/watchlist")
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    watchlist = data["data"]
    assert isinstance(watchlist, list)
    assert "AAPL" in watchlist
    assert "TSLA" in watchlist

async def test_watchlist_add_valid(api_client):
    payload = {"ticker": "NVDA"}
    response = await api_client.post("/api/watchlist", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["ticker"] == "NVDA"

async def test_watchlist_get_updated(api_client):
    response = await api_client.get("/api/watchlist")
    assert response.status_code == 200
    data = response.json()
    watchlist = data["data"]
    assert "NVDA" in watchlist

async def test_watchlist_add_duplicate(api_client):
    payload = {"ticker": "NVDA"}
    response = await api_client.post("/api/watchlist", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True

async def test_watchlist_delete(api_client):
    response = await api_client.delete("/api/watchlist/NVDA")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["ticker"] == "NVDA"

    # Verify deleted
    get_res = await api_client.get("/api/watchlist")
    assert "NVDA" not in get_res.json()["data"]


# ==========================================
# FEATURE 2: LIVE MARKET GRID (5 tests)
# ==========================================

async def test_markets_grid_status(api_client):
    response = await api_client.get("/api/markets")
    assert response.status_code == 200

async def test_markets_grid_structure(api_client):
    response = await api_client.get("/api/markets")
    data = response.json()
    assert "data" in data
    assert isinstance(data["data"], list)

async def test_markets_grid_item_fields(api_client):
    response = await api_client.get("/api/markets")
    grid_data = response.json()["data"]
    for item in grid_data:
        assert "ticker" in item
        assert "price" in item
        assert "cached_prediction" in item
        assert "cached_debate" in item

async def test_markets_grid_prices_numeric(api_client):
    response = await api_client.get("/api/markets")
    grid_data = response.json()["data"]
    for item in grid_data:
        assert isinstance(item["price"], (int, float))
        assert item["price"] > 0

async def test_markets_grid_watchlist_alignment(api_client):
    w_resp = await api_client.get("/api/watchlist")
    watchlist = w_resp.json()["data"]
    
    m_resp = await api_client.get("/api/markets")
    grid_tickers = [item["ticker"] for item in m_resp.json()["data"]]
    
    assert set(watchlist).issubset(set(grid_tickers))


# ==========================================
# FEATURE 3: SSE PREDICT STREAM (5 tests)
# ==========================================

async def test_predict_stream_status_headers(api_client):
    async with api_client.stream("GET", "/api/predict/AAPL/stream") as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")

async def test_predict_stream_agent_update(api_client, parse_sse_stream):
    async with api_client.stream("GET", "/api/predict/AAPL/stream") as response:
        events = await parse_sse_stream(response)
    updates = [e for e in events if e["event"] == "agent_update"]
    assert len(updates) >= 1
    assert "message" in updates[0]["data"]

async def test_predict_stream_verdict(api_client, parse_sse_stream):
    async with api_client.stream("GET", "/api/predict/AAPL/stream") as response:
        events = await parse_sse_stream(response)
    verdicts = [e for e in events if e["event"] == "verdict"]
    assert len(verdicts) == 1
    assert "verdict" in verdicts[0]["data"]
    assert "confidence" in verdicts[0]["data"]

async def test_predict_stream_done(api_client, parse_sse_stream):
    async with api_client.stream("GET", "/api/predict/AAPL/stream") as response:
        events = await parse_sse_stream(response)
    dones = [e for e in events if e["event"] == "done"]
    assert len(dones) == 1

async def test_predict_stream_event_sequence(api_client, parse_sse_stream):
    async with api_client.stream("GET", "/api/predict/AAPL/stream") as response:
        events = await parse_sse_stream(response)
    assert len(events) >= 3
    assert events[0]["event"] == "agent_update"
    assert events[-2]["event"] == "verdict"
    assert events[-1]["event"] == "done"


# ==========================================
# FEATURE 4: SSE CHAT STREAM (5 tests)
# ==========================================

async def test_chat_stream_headers(api_client):
    payload = {"message": "Should I buy AAPL?"}
    async with api_client.stream("POST", "/api/chat/stream", json=payload) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")

async def test_chat_stream_steps(api_client, parse_sse_stream):
    payload = {"message": "Should I buy AAPL?"}
    async with api_client.stream("POST", "/api/chat/stream", json=payload) as response:
        events = await parse_sse_stream(response)
    steps = [e for e in events if e["event"] == "step"]
    assert len(steps) >= 1
    assert "step" in steps[0]["data"]

async def test_chat_stream_tokens(api_client, parse_sse_stream):
    payload = {"message": "Should I buy AAPL?"}
    async with api_client.stream("POST", "/api/chat/stream", json=payload) as response:
        events = await parse_sse_stream(response)
    tokens = [e for e in events if e["event"] == "token"]
    assert len(tokens) >= 1
    assert "token" in tokens[0]["data"]

async def test_chat_stream_done(api_client, parse_sse_stream):
    payload = {"message": "Should I buy AAPL?"}
    async with api_client.stream("POST", "/api/chat/stream", json=payload) as response:
        events = await parse_sse_stream(response)
    dones = [e for e in events if e["event"] == "done"]
    assert len(dones) == 1

async def test_chat_stream_sequence(api_client, parse_sse_stream):
    payload = {"message": "Should I buy AAPL?"}
    async with api_client.stream("POST", "/api/chat/stream", json=payload) as response:
        events = await parse_sse_stream(response)
    assert len(events) >= 3
    assert events[0]["event"] == "step"
    assert events[-1]["event"] == "done"


# ==========================================
# FEATURE 5: NEWS BRIEFINGS (5 tests)
# ==========================================

async def test_briefing_status(api_client):
    response = await api_client.get("/api/briefing")
    assert response.status_code == 200

async def test_briefing_keys(api_client):
    response = await api_client.get("/api/briefing")
    data = response.json()
    assert "sectors" in data
    assert isinstance(data["sectors"], dict)

async def test_briefing_sectors_contain_lists(api_client):
    response = await api_client.get("/api/briefing")
    sectors = response.json()["sectors"]
    for sector_name, articles in sectors.items():
        assert isinstance(articles, list)

async def test_briefing_article_properties(api_client):
    response = await api_client.get("/api/briefing")
    sectors = response.json()["sectors"]
    for articles in sectors.values():
        for article in articles:
            assert "title" in article
            assert "sentiment" in article
            assert "importance" in article
            assert "timestamp" in article

async def test_briefing_limit_filtering(api_client):
    response = await api_client.get("/api/briefing?limit=1")
    sectors = response.json()["sectors"]
    for articles in sectors.values():
        assert len(articles) <= 1


# ==========================================
# FEATURE 6: CHARTS/INDICATORS (5 tests)
# ==========================================

async def test_charts_status(api_client):
    response = await api_client.get("/api/charts/AAPL")
    assert response.status_code == 200

async def test_charts_basic_fields(api_client):
    response = await api_client.get("/api/charts/AAPL")
    data = response.json()
    assert data["ticker"] == "AAPL"
    assert isinstance(data["prices"], list)
    assert len(data["prices"]) > 0

async def test_charts_sma_indicator(api_client):
    response = await api_client.get("/api/charts/AAPL")
    indicators = response.json()["indicators"]
    assert "sma" in indicators
    assert isinstance(indicators["sma"], list)
    assert len(indicators["sma"]) == len(response.json()["prices"])

async def test_charts_macd_indicator(api_client):
    response = await api_client.get("/api/charts/AAPL")
    macd = response.json()["indicators"]["macd"]
    assert "macd_line" in macd
    assert "signal_line" in macd
    assert "histogram" in macd
    assert len(macd["macd_line"]) == len(response.json()["prices"])

async def test_charts_bollinger_bands(api_client):
    response = await api_client.get("/api/charts/AAPL")
    bb = response.json()["indicators"]["bollinger_bands"]
    assert "upper" in bb
    assert "middle" in bb
    assert "lower" in bb
    assert len(bb["upper"]) == len(response.json()["prices"])


# ==========================================
# ==========================================
# FEATURE 7: ACCURACY & REFLECTIONS (5 tests)
# ==========================================

async def test_accuracy_endpoint(api_client):
    response = await api_client.get("/api/accuracy")
    assert response.status_code == 200
    data = response.json()
    assert "win_rate" in data
    assert "total_predictions" in data

async def test_reflections_get_initial(api_client):
    response = await api_client.get("/api/reflections/AAPL")
    assert response.status_code == 200
    reflections = response.json()
    assert isinstance(reflections, list)
    assert len(reflections) >= 1
    assert reflections[0]["ticker"] == "AAPL"

async def test_reflections_post(api_client):
    payload = {"lesson": "Never bet against the Fed."}
    response = await api_client.post("/api/reflections/AAPL", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["lesson"] == "Never bet against the Fed."

async def test_reflections_get_updated_list(api_client):
    response = await api_client.get("/api/reflections/AAPL")
    reflections = response.json()
    lessons = [r["lesson"] for r in reflections]
    assert "Never bet against the Fed." in lessons

async def test_reflections_non_existent_ticker(api_client):
    # Retrieve reflections for a ticker that has none (returns empty list)
    response = await api_client.get("/api/reflections/GOOG")
    assert response.status_code == 200
    assert response.json() == []


# ==========================================
# FEATURE 9: TOKEN & STATUS (5 tests)
# ==========================================

async def test_status_endpoint_healthy(api_client):
    response = await api_client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["db_connected"] is True

async def test_status_uptime_positive(api_client):
    response = await api_client.get("/api/status")
    data = response.json()
    assert data["uptime_seconds"] > 0

async def test_usage_endpoint_fields(api_client):
    response = await api_client.get("/api/usage")
    assert response.status_code == 200
    data = response.json()
    assert "total_tokens" in data
    assert "total_cost_usd" in data
    assert "calls_count" in data

async def test_usage_cost_calculation(api_client):
    response = await api_client.get("/api/usage")
    data = response.json()
    assert data["total_tokens"] > 0
    assert data["total_cost_usd"] > 0

async def test_usage_ratio_check(api_client):
    response = await api_client.get("/api/usage")
    data = response.json()
    # verify sum of components matches total
    assert data["prompt_tokens"] + data["completion_tokens"] == data["total_tokens"]
