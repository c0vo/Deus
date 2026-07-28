import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from pipeline.market_scanner import MarketScanner
from data.database import Database
from data.models import NewsArticle
from datetime import datetime, timezone

@pytest.fixture
def db(tmp_path):
    db_path = str(tmp_path / "test.db")
    database = Database(db_path=db_path)
    database.initialize()
    return database

@pytest.mark.asyncio
async def test_market_scanner_exact_ticker_match(db):
    """Test that the market scanner uses exact ticker matching when querying DB context."""
    # Insert an article with ticker 'MSFT'
    article_msft = NewsArticle(
        id="test_msft",
        headline="MSFT Earnings",
        source_name="test",
        source_type="api",
        content_hash="hash1",
        url="https://example.com/msft",
        published_at=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
        classification_summary="Microsoft beats earnings"
    )
    db.insert_article(article_msft)
    db.insert_ticker_mentions(article_msft.id, ["MSFT"], 0.8, "high")

    # Insert an article with ticker 'MS'
    article_ms = NewsArticle(
        id="test_ms",
        headline="MS Earnings",
        source_name="test",
        source_type="api",
        content_hash="hash2",
        url="https://example.com/ms",
        published_at=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
        classification_summary="Morgan Stanley beats earnings"
    )
    db.insert_article(article_ms)
    db.insert_ticker_mentions(article_ms.id, ["MS"], 0.7, "high")

    mock_alert_manager = MagicMock()
    mock_alert_manager.bot.send_message = AsyncMock()

    scanner = MarketScanner(db=db, alert_manager=mock_alert_manager)
    scanner.client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "This is a reason."
    mock_response.usage_metadata = None
    scanner.client.models.generate_content.return_value = mock_response

    # Call _generate_and_send_alert for 'MS' (Morgan Stanley)
    await scanner._generate_and_send_alert("MS", 100.0, 5.5)

    # Check that LLM prompt included the Morgan Stanley summary, NOT MSFT
    prompt_used = scanner.client.models.generate_content.call_args[1]["contents"]
    assert "Morgan Stanley beats earnings" in prompt_used
    assert "Microsoft beats earnings" not in prompt_used
    assert "Microsoft beats earnings" not in prompt_used
