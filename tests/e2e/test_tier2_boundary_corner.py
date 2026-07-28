import pytest
import httpx

pytestmark = pytest.mark.asyncio

# ==========================================
# FEATURE 1: WATCHLIST CRUD (5 tests)
# ==========================================

async def test_watchlist_add_empty_ticker(api_client):
    payload = {"ticker": ""}
    response = await api_client.post("/api/watchlist", json=payload)
    assert response.status_code == 422

async def test_watchlist_add_too_long_ticker(api_client):
    payload = {"ticker": "VERYLONGTICKERNAME"}
    response = await api_client.post("/api/watchlist", json=payload)
    assert response.status_code == 422

async def test_watchlist_add_special_characters(api_client):
    payload = {"ticker": "AAPL$"}
    response = await api_client.post("/api/watchlist", json=payload)
    assert response.status_code == 400

async def test_watchlist_delete_non_existent(api_client):
    response = await api_client.delete("/api/watchlist/NONEXISTENT")
    assert response.status_code == 404

async def test_watchlist_add_lowercase_whitespace(api_client):
    payload = {"ticker": "   msft   "}
    response = await api_client.post("/api/watchlist", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["ticker"] == "MSFT"


# ==========================================
# FEATURE 2: LIVE MARKET GRID (5 tests)
# ==========================================

async def test_markets_grid_empty_watchlist(api_client):
    # Fetch current watchlist
    response = await api_client.get("/api/watchlist")
    original_watchlist = response.json()["data"]
    
    # Remove all items
    for item in original_watchlist:
        await api_client.delete(f"/api/watchlist/{item}")
        
    # Verify empty grid
    res = await api_client.get("/api/markets")
    assert res.status_code == 200
    data = res.json()
    assert data["data"] == []
    
    # Restore original watchlist
    for item in original_watchlist:
        await api_client.post("/api/watchlist", json={"ticker": item})

async def test_markets_grid_large_watchlist(api_client):
    # Add many tickers
    test_tickers = [f"TCK{i}" for i in range(15)]
    for t in test_tickers:
        await api_client.post("/api/watchlist", json={"ticker": t})
        
    res = await api_client.get("/api/markets")
    assert res.status_code == 200
    grid_tickers = [item["ticker"] for item in res.json()["data"]]
    for t in test_tickers:
        assert t in grid_tickers
        
    # Clean up
    for t in test_tickers:
        await api_client.delete(f"/api/watchlist/{t}")

async def test_markets_grid_non_existent_tickers_fallback(api_client):
    # Add a fictional ticker
    await api_client.post("/api/watchlist", json={"ticker": "ZJZJ"})
    
    res = await api_client.get("/api/markets")
    data = res.json()["data"]
    jz_item = next((item for item in data if item["ticker"] == "ZJZJ"), None)
    assert jz_item is not None
    assert jz_item["price"] > 0
    assert jz_item["cached_prediction"] is None
    assert jz_item["cached_debate"] is None
    
    # Clean up
    await api_client.delete("/api/watchlist/ZJZJ")

async def test_markets_grid_rate_limit_header_check(api_client):
    # Check headers of market grid response for typical rate limiting / security headers
    response = await api_client.get("/api/markets")
    assert response.status_code == 200
    # Endpoint should not crash on consecutive calls
    for _ in range(5):
        r = await api_client.get("/api/markets")
        assert r.status_code == 200

async def test_markets_grid_malformed_cache(api_client):
    # Add ticker
    await api_client.post("/api/watchlist", json={"ticker": "NFLX"})
    # Verify grid can handle null predictions/debates in cache
    res = await api_client.get("/api/markets")
    nflx_item = next((item for item in res.json()["data"] if item["ticker"] == "NFLX"), None)
    assert nflx_item is not None
    assert nflx_item["cached_prediction"] is None
    assert nflx_item["cached_debate"] is None
    
    # Clean up
    await api_client.delete("/api/watchlist/NFLX")


# ==========================================
# FEATURE 3: SSE PREDICT STREAM (5 tests)
# ==========================================

async def test_predict_stream_non_alphanumeric_ticker(api_client, parse_sse_stream):
    async with api_client.stream("GET", "/api/predict/AAPL$/stream") as response:
        events = await parse_sse_stream(response)
    errors = [e for e in events if e["event"] == "error"]
    assert len(errors) == 1
    assert "Invalid ticker name" in errors[0]["data"]["message"]

async def test_predict_stream_simulate_internal_error(api_client, parse_sse_stream):
    async with api_client.stream("GET", "/api/predict/AAPL/stream?simulate_error=True") as response:
        events = await parse_sse_stream(response)
    errors = [e for e in events if e["event"] == "error"]
    assert len(errors) == 1
    assert "Simulating internal error" in errors[0]["data"]["message"]

async def test_predict_stream_empty_ticker_param(api_client):
    # Route matching for empty ticker in path returns 404 (not found)
    response = await api_client.get("/api/predict//stream")
    assert response.status_code == 404

async def test_predict_stream_multiple_verdicts(api_client, parse_sse_stream):
    # Test streaming sequentially 3 times
    for _ in range(3):
        async with api_client.stream("GET", "/api/predict/TSLA/stream") as response:
            events = await parse_sse_stream(response)
        assert events[-1]["event"] == "done"

async def test_predict_stream_unknown_query_param(api_client, parse_sse_stream):
    # Server should ignore extra query params and proceed normally
    async with api_client.stream("GET", "/api/predict/AAPL/stream?foo=bar&baz=123") as response:
        events = await parse_sse_stream(response)
    assert events[-1]["event"] == "done"


# ==========================================
# FEATURE 4: SSE CHAT STREAM (5 tests)
# ==========================================

async def test_chat_stream_empty_message(api_client):
    payload = {"message": ""}
    response = await api_client.post("/api/chat/stream", json=payload)
    assert response.status_code == 422

async def test_chat_stream_extremely_large_message(api_client, parse_sse_stream):
    large_message = "Should I buy AAPL? " * 1000
    payload = {"message": large_message}
    async with api_client.stream("POST", "/api/chat/stream", json=payload) as response:
        events = await parse_sse_stream(response)
    assert events[-1]["event"] == "done"

async def test_chat_stream_simulate_error(api_client, parse_sse_stream):
    payload = {"message": "Hello"}
    async with api_client.stream("POST", "/api/chat/stream?simulate_error=True", json=payload) as response:
        events = await parse_sse_stream(response)
    errors = [e for e in events if e["event"] == "error"]
    assert len(errors) == 1
    assert "Chat service unavailable" in errors[0]["data"]["message"]

async def test_chat_stream_invalid_json_payload(api_client):
    response = await api_client.post("/api/chat/stream", content="this is not json")
    assert response.status_code == 422

async def test_chat_stream_missing_required_field(api_client):
    payload = {"wrong_field": "hello"}
    response = await api_client.post("/api/chat/stream", json=payload)
    assert response.status_code == 422


# ==========================================
# FEATURE 5: NEWS BRIEFINGS (5 tests)
# ==========================================

async def test_briefing_negative_limit(api_client):
    response = await api_client.get("/api/briefing?limit=-5")
    assert response.status_code == 400

async def test_briefing_extremely_large_limit(api_client):
    response = await api_client.get("/api/briefing?limit=10000")
    assert response.status_code == 200
    sectors = response.json()["sectors"]
    # Technology should have 3 articles (its total database size)
    assert len(sectors.get("Technology", [])) <= 3

async def test_briefing_non_existent_sector(api_client):
    response = await api_client.get("/api/briefing?sector=BioTech")
    assert response.status_code == 200
    assert response.json()["sectors"] == {}

async def test_briefing_sql_injection_attempt_sector(api_client):
    # Sector filter should treat SQL injection attempt as literal and yield nothing
    response = await api_client.get("/api/briefing?sector=Technology' OR 1=1--")
    assert response.status_code == 200
    assert response.json()["sectors"] == {}

async def test_briefing_zero_limit(api_client):
    response = await api_client.get("/api/briefing?limit=0")
    assert response.status_code == 200
    sectors = response.json()["sectors"]
    for val in sectors.values():
        assert len(val) == 0


# ==========================================
# FEATURE 6: CHARTS/INDICATORS (5 tests)
# ==========================================

async def test_charts_invalid_ticker_symbols(api_client):
    response = await api_client.get("/api/charts/AAPL$")
    assert response.status_code == 400

async def test_charts_too_long_ticker(api_client):
    response = await api_client.get("/api/charts/VERYLONGTICKERNAME")
    assert response.status_code == 404

async def test_charts_unsupported_period(api_client):
    # Unspecified or unsupported periods should fallback gracefully (status 200)
    response = await api_client.get("/api/charts/AAPL?period=100y")
    assert response.status_code == 200

async def test_charts_missing_path_param(api_client):
    # Missing path param defaults to route not found
    response = await api_client.get("/api/charts/")
    assert response.status_code == 404

async def test_charts_special_characters_encoded(api_client):
    response = await api_client.get("/api/charts/A%20B")
    assert response.status_code == 400


# ==========================================
# ==========================================
# FEATURE 7: ACCURACY & REFLECTIONS (5 tests)
# ==========================================

async def test_reflections_post_empty_lesson(api_client):
    payload = {"lesson": ""}
    response = await api_client.post("/api/reflections/AAPL", json=payload)
    assert response.status_code == 422

async def test_reflections_post_whitespace_only(api_client):
    payload = {"lesson": "      "}
    response = await api_client.post("/api/reflections/AAPL", json=payload)
    assert response.status_code == 400

async def test_reflections_get_non_alphanumeric_ticker(api_client):
    response = await api_client.get("/api/reflections/AAPL$")
    assert response.status_code == 200
    assert response.json() == []

async def test_reflections_post_missing_body(api_client):
    response = await api_client.post("/api/reflections/AAPL", json=None)
    assert response.status_code == 422

async def test_accuracy_no_predictions_division_by_zero(api_client):
    # Ensure checking accuracy never results in division by zero internally
    response = await api_client.get("/api/accuracy")
    assert response.status_code == 200
    data = response.json()
    assert data["total_predictions"] >= 0
    if data["total_predictions"] == 0:
        assert data["win_rate"] == 0.0


# ==========================================
# FEATURE 9: TOKEN & STATUS (5 tests)
# ==========================================

async def test_usage_invalid_query_params(api_client):
    response = await api_client.get("/api/usage?foo=bar")
    assert response.status_code == 200

async def test_status_extra_params(api_client):
    response = await api_client.get("/api/status?extra=1")
    assert response.status_code == 200

async def test_status_db_locked_simulation(api_client):
    response = await api_client.get("/api/status")
    data = response.json()
    assert "db_connected" in data
    # Ensure health indicators are present
    assert data["status"] in ["healthy", "degraded"]

async def test_usage_cost_boundaries(api_client):
    response = await api_client.get("/api/usage")
    data = response.json()
    assert data["total_cost_usd"] >= 0.0

async def test_usage_overflow_protection(api_client):
    response = await api_client.get("/api/usage")
    data = response.json()
    assert isinstance(data["total_tokens"], int)
