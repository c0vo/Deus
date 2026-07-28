"""Tests for ArticleClassifier — classification logic and LLM response parsing."""

import json
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timezone

from data.models import NewsArticle


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def classifier():
    """Create ArticleClassifier with mocked clients (no real API calls)."""
    from pipeline.classifier import ArticleClassifier
    clf = ArticleClassifier(
        client=MagicMock(),
        deepseek_client=MagicMock(),
        db=MagicMock(),
    )
    clf.model_name = "test-gemini-model"
    return clf


@pytest.fixture
def earnings_article():
    return NewsArticle(
        id="test_earnings_001",
        headline="Apple beats Q3 estimates, announces $90B buyback",
        summary="Apple reported revenue of $85.8B vs $84.4B expected. EPS of $1.40 vs $1.35 expected.",
        source_name="reuters",
        source_type="rss",
        url="https://example.com/apple-earnings",
        published_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def noise_article():
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
def reddit_article():
    return NewsArticle(
        id="test_reddit_001",
        headline="GME YOLO update — $50k to $500k, still not selling 💎🙌",
        summary="Position up 10x. Diamond hands.",
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


def make_deepseek_response(json_str: str):
    """Build a mock DeepSeek response object."""
    choice = MagicMock()
    choice.message.content = json_str
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = MagicMock(prompt_tokens=100, completion_tokens=50)
    return resp


def make_gemini_response(json_str: str):
    """Build a mock Gemini response object."""
    resp = MagicMock()
    resp.text = json_str
    resp.usage_metadata = MagicMock(prompt_token_count=100, candidates_token_count=50)
    return resp


# ── should_classify tests ───────────────────────────────────────────────────

class TestShouldClassify:
    """Pre-filter logic — should this article be sent to the LLM?"""

    def test_cashtag_triggers_classification(self, classifier):
        article = NewsArticle(
            id="t1", headline="$AAPL is mooning", summary="",
            source_name="test", source_type="rss",
            url="https://example.com/t1", published_at=datetime.now(timezone.utc),
        )
        assert classifier.should_classify(article) is True

    def test_ticker_pattern_triggers_classification(self, classifier):
        article = NewsArticle(
            id="t2", headline="AAPL earnings preview", summary="",
            source_name="test", source_type="rss",
            url="https://example.com/t2", published_at=datetime.now(timezone.utc),
        )
        assert classifier.should_classify(article) is True

    def test_financial_keyword_triggers_classification(self, classifier):
        article = NewsArticle(
            id="t3", headline="Fed holds interest rates steady", summary="Central bank decision.",
            source_name="test", source_type="rss",
            url="https://example.com/t3", published_at=datetime.now(timezone.utc),
        )
        assert classifier.should_classify(article) is True

    def test_reddit_keyword_triggers_classification(self, classifier):
        article = NewsArticle(
            id="t4", headline="My calls are printing! 🚀", summary="",
            source_name="reddit_wallstreetbets", source_type="social",
            url="https://reddit.com/r/wsb/t4", published_at=datetime.now(timezone.utc),
        )
        assert classifier.should_classify(article) is True

    def test_non_financial_text_returns_false(self, classifier, noise_article):
        assert classifier.should_classify(noise_article) is False

    def test_empty_text_returns_false(self, classifier):
        article = NewsArticle(
            id="t5", headline="", summary="",
            source_name="test", source_type="rss",
            url="https://example.com/t5", published_at=datetime.now(timezone.utc),
        )
        assert classifier.should_classify(article) is False


# ── classify tests ──────────────────────────────────────────────────────────

class TestClassify:
    """Full classification flow."""

    @pytest.mark.asyncio
    async def test_deepseek_success_path(self, classifier, earnings_article):
        """DeepSeek returns valid JSON → article fields are populated."""
        valid_json = json.dumps({
            "event_type": "earnings",
            "sentiment_score": 0.55,
            "urgency": "high",
            "suggested_direction": "bullish",
            "affected_sectors": ["Technology", "Consumer Electronics"],
            "affected_tickers": ["AAPL"],
            "classification_summary": "Apple beat Q3 estimates on both revenue and EPS.",
        })
        classifier.deepseek_client.chat.completions.create.return_value = make_deepseek_response(valid_json)

        result = await classifier.classify(earnings_article)

        assert result.event_type == "earnings"
        assert result.sentiment_score == 0.55
        assert result.urgency == "high"
        assert result.suggested_direction == "bullish"
        assert "AAPL" in result.affected_tickers
        assert "Technology" in result.affected_sectors
        assert classifier.deepseek_client.chat.completions.create.called
        assert classifier.db.log_llm_usage.called

    @pytest.mark.asyncio
    async def test_gemini_fallback_on_deepseek_failure(self, classifier, earnings_article):
        """DeepSeek fails → Gemini fallback is used."""
        classifier.deepseek_client.chat.completions.create.side_effect = Exception("DeepSeek API error")

        gemini_json = json.dumps({
            "event_type": "earnings",
            "sentiment_score": 0.50,
            "urgency": "high",
            "suggested_direction": "bullish",
            "affected_sectors": ["Technology"],
            "affected_tickers": ["AAPL"],
            "classification_summary": "Fallback classification.",
        })
        classifier.client.aio.models.generate_content.return_value = make_gemini_response(gemini_json)

        result = await classifier.classify(earnings_article)

        assert result.event_type == "earnings"
        assert result.sentiment_score == 0.50
        assert classifier.client.aio.models.generate_content.called

    @pytest.mark.asyncio
    async def test_noise_article_skips_llm(self, classifier):
        """Article with no financial content gets 'noise' event_type, no LLM calls."""
        article = NewsArticle(
            id="noise_1", headline="Best mechanical keyboards under $100", summary="",
            source_name="reddit_investing", source_type="social",
            url="https://example.com/noise", published_at=datetime.now(timezone.utc),
            raw_data={"comments": []},
        )

        result = await classifier.classify(article)

        assert result.event_type == "noise"
        assert result.sentiment_score == 0.0
        assert result.urgency == "low"
        assert not classifier.deepseek_client.chat.completions.create.called
        assert not classifier.client.aio.models.generate_content.called

    @pytest.mark.asyncio
    async def test_reddit_article_uses_reddit_prompt(self, classifier, reddit_article):
        """Reddit social articles use the REDDIT_CLASSIFICATION_PROMPT."""
        valid_json = json.dumps({
            "event_type": "meme_stock",
            "sentiment_score": 0.55,
            "urgency": "high",
            "suggested_direction": "bullish",
            "affected_sectors": ["Retail"],
            "affected_tickers": ["GME"],
            "classification_summary": "Highly bullish WSB sentiment.",
        })
        classifier.deepseek_client.chat.completions.create.return_value = make_deepseek_response(valid_json)

        result = await classifier.classify(reddit_article)

        assert result.event_type == "meme_stock"
        assert "GME" in result.affected_tickers
        # Check that the prompt sent to DeepSeek contains WSB-specific content
        call_args = classifier.deepseek_client.chat.completions.create.call_args
        messages = call_args[1].get("messages", [])
        user_prompt = messages[0]["content"] if messages else ""
        assert "WSB" in user_prompt or "meme_stock" in user_prompt

    @pytest.mark.asyncio
    async def test_invalid_json_from_llm(self, classifier, earnings_article):
        """LLM returns invalid JSON → article is returned unchanged (graceful failure)."""
        classifier.deepseek_client.chat.completions.create.return_value = make_deepseek_response("not valid json at all {{{")

        result = await classifier.classify(earnings_article)

        # Article fields should NOT have been updated by the LLM
        assert result.event_type is None
        # The function didn't crash — article returned with original id intact
        assert result.id == earnings_article.id

    @pytest.mark.asyncio
    async def test_no_llm_configured(self, classifier, earnings_article):
        """When no LLM is configured, article is returned unchanged."""
        with patch("pipeline.classifier.is_configured", return_value=False), \
             patch("pipeline.classifier.is_deepseek_configured", return_value=False):

            result = await classifier.classify(earnings_article)

            assert not classifier.deepseek_client.chat.completions.create.called
            assert not classifier.client.aio.models.generate_content.called

    @pytest.mark.asyncio
    async def test_classify_set_article_tickers_includes_existing(self, classifier):
        """Existing article tickers are merged with LLM-extracted tickers."""
        article = NewsArticle(
            id="merge_test", headline="AAPL and MSFT both reporting",
            summary="Earnings season continues.",
            source_name="alpha_vantage", source_type="api",
            url="https://example.com/merge", published_at=datetime.now(timezone.utc),
            affected_tickers=["AAPL"],  # Pre-populated by source
        )
        valid_json = json.dumps({
            "event_type": "earnings",
            "sentiment_score": 0.3,
            "urgency": "high",
            "suggested_direction": "bullish",
            "affected_sectors": ["Technology"],
            "affected_tickers": ["MSFT"],
            "classification_summary": "Earnings season.",
        })
        classifier.deepseek_client.chat.completions.create.return_value = make_deepseek_response(valid_json)

        result = await classifier.classify(article)

        assert "AAPL" in result.affected_tickers  # Original ticker preserved
        assert "MSFT" in result.affected_tickers   # New ticker added

    @pytest.mark.asyncio
    async def test_deepseek_logs_usage_on_success(self, classifier, earnings_article):
        """LLM usage is logged after a successful DeepSeek classification."""
        valid_json = json.dumps({"event_type": "earnings", "sentiment_score": 0.5, "urgency": "high",
                                 "suggested_direction": "bullish", "classification_summary": "OK"})
        classifier.deepseek_client.chat.completions.create.return_value = make_deepseek_response(valid_json)

        await classifier.classify(earnings_article)

        classifier.db.log_llm_usage.assert_called_once()
        call_kwargs = classifier.db.log_llm_usage.call_args[1]
        assert call_kwargs["operation"] == "classify"
        assert call_kwargs["model_name"] == "deepseek-v4-flash"
