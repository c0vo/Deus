from fastapi import FastAPI, HTTPException, Query, Body, Path, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import json
import asyncio
import random
from typing import List, Dict, Any, Optional

app = FastAPI(title="Deus Mock Backend")

# --- In-Memory State ---
watchlist = ["AAPL", "TSLA", "MSFT"]

reflections = {
    "AAPL": [
        {"ticker": "AAPL", "lesson": "Earnings beats tend to run for 3 days.", "timestamp": "2026-06-28T10:00:00Z"}
    ],
    "TSLA": [
        {"ticker": "TSLA", "lesson": "Highly volatile, watch sentiment closely.", "timestamp": "2026-06-28T11:00:00Z"}
    ]
}

predictions_cache = {
    "AAPL": {"direction": "BUY", "confidence": 0.85, "timestamp": "2026-06-28T20:25:50Z"},
    "TSLA": {"direction": "SELL", "confidence": 0.65, "timestamp": "2026-06-28T20:25:50Z"}
}

debate_cache = {
    "AAPL": {"verdict": "BUY", "bull_argument": "Strong earnings growth", "bear_argument": "High valuation"},
    "TSLA": {"verdict": "SELL", "bull_argument": "Gigafactory expansion", "bear_argument": "Regulatory headwinds"}
}

token_usage = {
    "total_tokens": 150000,
    "prompt_tokens": 100000,
    "completion_tokens": 50000,
    "total_cost_usd": 3.00,
    "calls_count": 120
}

accuracy_metrics = {
    "win_rate": 0.68,
    "total_predictions": 50,
    "correct_predictions": 34,
    "incorrect_predictions": 16,
    "average_confidence": 0.81
}

# --- Pydantic Models ---
class WatchlistAdd(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=10)

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)

class ReflectionAdd(BaseModel):
    lesson: str = Field(..., min_length=1)


# --- Helper Function for SSE ---
def format_sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# --- 1. Watchlist CRUD ---
@app.get("/api/watchlist")
async def get_watchlist():
    return {"data": watchlist}

@app.post("/api/watchlist")
async def add_watchlist(payload: WatchlistAdd):
    ticker = payload.ticker.upper().strip()
    if not ticker.isalnum():
        raise HTTPException(status_code=400, detail="Ticker must be alphanumeric")
    if ticker in watchlist:
        # Duplicate is fine, just return success
        return {"success": True, "ticker": ticker, "message": "Already exists"}
    watchlist.append(ticker)
    return {"success": True, "ticker": ticker}

@app.delete("/api/watchlist/{ticker}")
async def delete_watchlist(ticker: str):
    ticker_upper = ticker.upper().strip()
    if ticker_upper not in watchlist:
        raise HTTPException(status_code=404, detail=f"Ticker {ticker_upper} not found in watchlist")
    watchlist.remove(ticker_upper)
    return {"success": True, "ticker": ticker_upper}


# --- 2. Live Market Grid ---
@app.get("/api/markets")
async def get_markets():
    data = []
    for t in watchlist:
        # Return mock values
        price = 150.0 + random.random() * 100.0 if t not in ["AAPL", "TSLA", "MSFT"] else (175.50 if t == "AAPL" else (185.20 if t == "TSLA" else 420.50))
        data.append({
            "ticker": t,
            "price": round(price, 2),
            "cached_prediction": predictions_cache.get(t),
            "cached_debate": debate_cache.get(t)
        })
    return {"data": data}


# --- 3. SSE Predict Stream ---
@app.get("/api/predict/{ticker}/stream")
async def predict_stream(ticker: str, simulate_error: bool = False, force_verdict: Optional[str] = None):
    ticker_upper = ticker.upper().strip()
    
    # Validation
    if not ticker_upper.isalnum():
        async def error_generator_invalid():
            yield format_sse("error", {"message": "Invalid ticker name"})
            yield format_sse("done", {})
        return StreamingResponse(error_generator_invalid(), media_type="text/event-stream")

    if simulate_error:
        async def error_generator():
            yield format_sse("agent_update", {"message": "Initializing model..."})
            await asyncio.sleep(0.01)
            yield format_sse("error", {"message": "Failed to run prediction: Simulating internal error"})
            yield format_sse("done", {})
        return StreamingResponse(error_generator(), media_type="text/event-stream")

    async def event_generator():
        # Update token usage
        token_usage["total_tokens"] += 2500
        token_usage["prompt_tokens"] += 2000
        token_usage["completion_tokens"] += 500
        token_usage["total_cost_usd"] += 0.05
        token_usage["calls_count"] += 1

        # Simulate agent updates
        yield format_sse("agent_update", {"message": f"Agent Bull analyzing technical indicators for {ticker_upper}..."})
        await asyncio.sleep(0.01)
        yield format_sse("agent_update", {"message": f"Agent Bear analyzing regulatory risk for {ticker_upper}..."})
        await asyncio.sleep(0.01)
        
        # Verdict calculation
        verdict_val = force_verdict if force_verdict else random.choice(["BUY", "SELL", "HOLD"])
        confidence_val = round(0.5 + random.random() * 0.45, 2)
        
        # Save to cache
        predictions_cache[ticker_upper] = {
            "direction": verdict_val,
            "confidence": confidence_val,
            "timestamp": "2026-06-28T20:25:50Z"
        }
        debate_cache[ticker_upper] = {
            "verdict": verdict_val,
            "bull_argument": "Strong technical momentum and support levels",
            "bear_argument": "Short-term valuation overstretched"
        }

        # Update accuracy metrics
        accuracy_metrics["total_predictions"] += 1
        is_correct = random.choice([True, False])
        if is_correct:
            accuracy_metrics["correct_predictions"] += 1
        else:
            accuracy_metrics["incorrect_predictions"] += 1
        accuracy_metrics["win_rate"] = round(accuracy_metrics["correct_predictions"] / accuracy_metrics["total_predictions"], 2)
        # Recalculate average confidence (mock progression)
        accuracy_metrics["average_confidence"] = round((accuracy_metrics["average_confidence"] * (accuracy_metrics["total_predictions"] - 1) + confidence_val) / accuracy_metrics["total_predictions"], 2)

        yield format_sse("verdict", {
            "verdict": verdict_val,
            "confidence": confidence_val,
            "bull_argument": "Strong technical momentum and support levels",
            "bear_argument": "Short-term valuation overstretched"
        })
        await asyncio.sleep(0.01)
        yield format_sse("done", {})

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# --- 4. SSE Chat Stream ---
@app.post("/api/chat/stream")
async def chat_stream(payload: ChatRequest, simulate_error: bool = False):
    if simulate_error:
        async def error_generator():
            yield format_sse("error", {"message": "Chat service unavailable"})
            yield format_sse("done", {})
        return StreamingResponse(error_generator(), media_type="text/event-stream")

    async def event_generator():
        # Update token usage
        token_usage["total_tokens"] += 1000
        token_usage["prompt_tokens"] += 800
        token_usage["completion_tokens"] += 200
        token_usage["total_cost_usd"] += 0.02
        token_usage["calls_count"] += 1

        # Yield steps
        yield format_sse("step", {"step": "Searching knowledge base..."})
        await asyncio.sleep(0.01)
        yield format_sse("step", {"step": "Retrieving market data..."})
        await asyncio.sleep(0.01)

        # Yield token chunks
        words = ["Based ", "on ", "my ", "analysis, ", "market ", "sentiment ", "is ", "neutral-bullish. "]
        for word in words:
            yield format_sse("token", {"token": word})
            await asyncio.sleep(0.005)
            
        yield format_sse("done", {})

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# --- 5. News Briefings ---
# Predefined news database
news_db = {
    "Technology": [
        {"title": "Tech stocks rally on AI sector breakthrough", "sentiment": "bullish", "importance": 0.95, "timestamp": "2026-06-28T12:00:00Z"},
        {"title": "Regulatory scrutiny rises for big tech acquisitions", "sentiment": "bearish", "importance": 0.75, "timestamp": "2026-06-28T11:00:00Z"},
        {"title": "Chip supply chain bottlenecks ease slightly", "sentiment": "bullish", "importance": 0.65, "timestamp": "2026-06-28T09:00:00Z"}
    ],
    "Energy": [
        {"title": "OPEC+ extends voluntary production cuts", "sentiment": "bullish", "importance": 0.85, "timestamp": "2026-06-28T10:00:00Z"},
        {"title": "Renewable energy adoption reaches record high", "sentiment": "neutral", "importance": 0.70, "timestamp": "2026-06-28T08:00:00Z"}
    ],
    "Finance": [
        {"title": "Fed hints at potential rate cuts later this year", "sentiment": "bullish", "importance": 0.90, "timestamp": "2026-06-28T07:00:00Z"},
        {"title": "Regional banks face pressure from commercial real estate", "sentiment": "bearish", "importance": 0.80, "timestamp": "2026-06-28T06:00:00Z"}
    ]
}

@app.get("/api/briefing")
async def get_briefing(limit: int = Query(5), sector: Optional[str] = Query(None)):
    if limit < 0:
        raise HTTPException(status_code=400, detail="Limit must be a non-negative integer")
    
    # Filter sectors
    result = {}
    sectors_to_process = [sector] if sector else list(news_db.keys())
    
    for sec in sectors_to_process:
        if sec not in news_db:
            if sector: # Specifying non-existent sector should return empty or error. Let's return empty.
                continue
            continue
        articles = news_db[sec]
        # Sort by importance descending
        sorted_articles = sorted(articles, key=lambda x: x["importance"], reverse=True)
        result[sec] = sorted_articles[:limit]
        
    return {"sectors": result}


# --- 6. Charts/Indicators ---
@app.get("/api/charts/{ticker}")
async def get_charts(ticker: str, period: str = Query("1mo")):
    ticker_upper = ticker.upper().strip()
    if not ticker_upper.isalnum():
        raise HTTPException(status_code=400, detail="Invalid ticker name")
        
    if ticker_upper not in ["AAPL", "TSLA", "MSFT"] and len(ticker_upper) > 5:
        raise HTTPException(status_code=404, detail="Ticker not found")

    # Generate deterministic mock prices based on ticker string
    seed_val = sum(ord(c) for c in ticker_upper)
    random.seed(seed_val)
    
    base_price = 100.0 + (seed_val % 200)
    prices = []
    curr_price = base_price
    for _ in range(30):
        curr_price += round(random.uniform(-5.0, 5.5), 2)
        prices.append(round(curr_price, 2))
        
    # Calculate indicators (mocked formulas based on price)
    sma = []
    for i in range(len(prices)):
        window = prices[max(0, i-19):i+1]
        sma.append(round(sum(window) / len(window), 2))
        
    macd_line = []
    signal_line = []
    histogram = []
    for i in range(len(prices)):
        val = round(math_macd_sim(i, seed_val), 2)
        macd_line.append(val)
        sig = round(val * 0.9, 2)
        signal_line.append(sig)
        histogram.append(round(val - sig, 2))
        
    upper_band = []
    middle_band = []
    lower_band = []
    for i in range(len(prices)):
        mid = sma[i]
        std = 5.0 + (i % 3)
        upper_band.append(round(mid + 2 * std, 2))
        middle_band.append(mid)
        lower_band.append(round(mid - 2 * std, 2))
        
    rsi = []
    for i in range(len(prices)):
        rsi.append(round(40.0 + 30.0 * (i % 2) + random.uniform(-5, 5), 2))
        
    return {
        "ticker": ticker_upper,
        "prices": prices,
        "indicators": {
            "sma": sma,
            "macd": {
                "macd_line": macd_line,
                "signal_line": signal_line,
                "histogram": histogram
            },
            "bollinger_bands": {
                "upper": upper_band,
                "middle": middle_band,
                "lower": lower_band
            },
            "rsi": rsi
        }
    }

def math_macd_sim(index: int, seed: int) -> float:
    # Just a deterministic wave
    import math
    return 2.0 * math.sin(index / 3.0) + (seed % 5) / 5.0


# --- 7. Accuracy & Reflections ---
@app.get("/api/accuracy")
async def get_accuracy():
    return accuracy_metrics

@app.get("/api/reflections/{ticker}")
async def get_reflections(ticker: str):
    ticker_upper = ticker.upper().strip()
    if ticker_upper not in reflections:
        return []
    return reflections[ticker_upper]

@app.post("/api/reflections/{ticker}")
async def add_reflection(ticker: str, payload: ReflectionAdd):
    ticker_upper = ticker.upper().strip()
    lesson = payload.lesson.strip()
    if not lesson:
        raise HTTPException(status_code=400, detail="Lesson content cannot be empty")
        
    entry = {
        "ticker": ticker_upper,
        "lesson": lesson,
        "timestamp": "2026-06-28T20:25:50Z"
    }
    if ticker_upper not in reflections:
        reflections[ticker_upper] = []
    reflections[ticker_upper].append(entry)
    return {"success": True, "ticker": ticker_upper, "lesson": lesson}


# --- 9. Token & Status ---
@app.get("/api/status")
async def get_status():
    return {
        "status": "healthy",
        "db_connected": True,
        "db_size_bytes": 102400,
        "active_tasks": 2,
        "uptime_seconds": 3600
    }

@app.get("/api/usage")
async def get_usage():
    return token_usage
