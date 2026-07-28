import io
import pytest
from bot.visualizations import get_price_chart, get_sentiment_chart

@pytest.mark.asyncio
async def test_get_price_chart():
    ticker = "AAPL"
    prices = [150.0, 155.0, 152.0, 160.0]
    dates = ["2023-10-01", "2023-10-02", "2023-10-03", "2023-10-04"]
    
    buf = await get_price_chart(ticker, prices, dates)
    
    assert isinstance(buf, io.BytesIO)
    content = buf.getvalue()
    assert len(content) > 0
    # The output should be a PNG image
    assert content.startswith(b"\x89PNG")

@pytest.mark.asyncio
async def test_get_sentiment_chart():
    ticker = "AAPL"
    sentiment_scores = [0.1, 0.5, -0.2, 0.8]
    dates = ["2023-10-01", "2023-10-02", "2023-10-03", "2023-10-04"]
    
    buf = await get_sentiment_chart(ticker, sentiment_scores, dates)
    
    assert isinstance(buf, io.BytesIO)
    content = buf.getvalue()
    assert len(content) > 0
    assert content.startswith(b"\x89PNG")
