"""Shared test fixtures for the Project Scrooge V2 test suite."""

import pytest
import numpy as np
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timezone
from data.models import NewsArticle


@pytest.fixture
def sample_article():
    """A fully populated NewsArticle with classification fields."""
    return NewsArticle(
        id="test_article_001",
        headline="Apple beats Q3 estimates, announces $90B buyback",
        summary="Apple reported Q3 revenue of $85.8B vs $84.4B expected.",
        source_name="reuters",
        source_type="rss",
        url="https://example.com/apple-earnings",
        published_at=datetime.now(timezone.utc),
        content_hash="abc123",
        event_type="earnings",
        sentiment_score=0.55,
        urgency="high",
        suggested_direction="bullish",
        affected_sectors=["Technology", "Consumer Electronics"],
        affected_tickers=["AAPL"],
        classification_summary="Apple beat estimates and announced buyback.",
        importance_score=7.5,
    )


@pytest.fixture
def sample_noise_article():
    """An article that should be filtered out as noise."""
    return NewsArticle(
        id="test_noise_001",
        headline="Best mechanical keyboards under $100",
        summary="Looking for suggestions for a new keyboard.",
        source_name="reddit_investing",
        source_type="social",
        url="https://reddit.com/r/investing/test_noise_001",
        published_at=datetime.now(timezone.utc),
        raw_data={"comments": [{"author": "user", "body": "I like MX Browns."}]},
    )


@pytest.fixture
def sample_reddit_article():
    """A Reddit article with WSB-style content."""
    return NewsArticle(
        id="test_reddit_001",
        headline="GME YOLO update — $50k to $500k, still not selling 💎🙌",
        summary="Position up 10x, diamond hands.",
        source_name="reddit_wallstreetbets",
        source_type="social",
        url="https://reddit.com/r/wallstreetbets/test_gme",
        published_at=datetime.now(timezone.utc),
        raw_data={
            "comments": [
                {"author": "deepvalue", "body": "Not selling till $1M", "score": 500},
                {"author": "wsb_god", "body": "This is the way", "score": 300},
            ]
        },
    )


@pytest.fixture
def mock_db():
    """A fully mocked Database instance."""
    db = MagicMock()
    db.has_sqlite_vec = False

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = []
    mock_cursor.fetchone.return_value = None
    mock_conn.execute.return_value = mock_cursor
    db.connection.return_value.__enter__.return_value = mock_conn

    db.get_ticker_sentiment_features.return_value = {
        "sentiment_avg_1d": 0.5,
        "sentiment_avg_3d": 0.4,
        "sentiment_avg_7d": 0.3,
        "sentiment_momentum": 0.2,
        "news_velocity": 1.5,
        "max_urgency_24h": 1.0,
        "avg_importance": 6.5,
        "bullish_ratio": 0.8,
    }
    db.get_unclassified_articles.return_value = []
    db.insert_article.return_value = True
    db.url_exists.return_value = False
    db.log_llm_usage.return_value = None
    db.get_existing_prediction.return_value = None
    db.insert_prediction.return_value = "pred_test_123"

    return db


@pytest.fixture
def mock_gemini_client():
    """Patch config.llm.get_client with a mock Gemini client."""
    with patch("config.llm.get_client") as mock_get:
        client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = '{"decision": "shallow"}'
        mock_resp.usage_metadata = None
        client.aio.models.generate_content.return_value = mock_resp
        client.models.generate_content.return_value = mock_resp
        mock_get.return_value = client
        yield mock_get


@pytest.fixture
def mock_deepseek_client():
    """Patch config.llm.get_deepseek_client with a mock DeepSeek client."""
    with patch("config.llm.get_deepseek_client") as mock_get:
        client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = (
            '{"event_type": "earnings", "sentiment_score": 0.5, '
            '"urgency": "high", "suggested_direction": "bullish", '
            '"affected_sectors": ["Technology"], "affected_tickers": ["AAPL"], '
            '"classification_summary": "Test summary"}'
        )
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock(prompt_tokens=100, completion_tokens=50)
        client.chat.completions.create.return_value = mock_response
        mock_get.return_value = client
        yield mock_get


@pytest.fixture
def anyio_backend():
    return "asyncio"
