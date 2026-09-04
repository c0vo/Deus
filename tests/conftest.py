"""Shared test fixtures for the Deus test suite."""

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


def make_llm_response(text: str = "", parsed=None, *, prompt_tokens: int = 100,
                      completion_tokens: int = 50, cost: float = 0.0001,
                      finish_reason: str = "stop"):
    """
    Build an `LLMResponse` the way `config.llm.complete` would.

    Tests assert against this rather than against a provider SDK's response
    object, which is the point of the facade: a model or provider swap must not
    be able to break a test.
    """
    from config.llm import LLMResponse

    usage = MagicMock(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                      cost=cost)
    return LLMResponse(
        text=text, parsed=parsed, usage=usage,
        finish_reason=finish_reason, cost=cost,
    )


def make_stream(chunks, *, prompt_tokens: int = 100, completion_tokens: int = 50,
                cost: float = 0.0001, finish_reason: str = "stop"):
    """
    An async iterator of `StreamChunk`s ending in a usage-only chunk, matching
    what `config.llm.stream_complete` yields.

    `chunks` may be plain strings (content deltas) or (text, reasoning) pairs.
    """
    from config.llm import StreamChunk

    async def _gen():
        last = len(chunks) - 1
        for i, c in enumerate(chunks):
            text, reasoning = (c, "") if isinstance(c, str) else c
            yield StreamChunk(
                text=text, reasoning=reasoning,
                finish_reason=finish_reason if i == last else None,
            )
        yield StreamChunk(usage=MagicMock(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, cost=cost,
        ))

    return _gen()


@pytest.fixture
def anyio_backend():
    return "asyncio"
