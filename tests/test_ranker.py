"""Tests for ArticleRanker — batch importance scoring logic."""

import json
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from data.models import NewsArticle


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def ranker():
    """Create ArticleRanker with mocked Gemini client (no real API calls)."""
    from pipeline.ranker import ArticleRanker
    r = ArticleRanker(client=MagicMock(), db=MagicMock())
    r.model_name = "test-ranker-model"
    return r


@pytest.fixture
def articles():
    """Create a batch of test articles with classification fields."""
    base = dict(source_name="test", source_type="rss", published_at=datetime.now(timezone.utc))
    return [
        NewsArticle(id="a1", headline="AAPL beats earnings", summary="Strong quarter", event_type="earnings",
                     sentiment_score=0.6, urgency="high", classification_summary="Good", url="http://a.com/1", **base),
        NewsArticle(id="a2", headline="Fed raises rates by 25bps", summary="Hike expected", event_type="macro",
                     sentiment_score=-0.2, urgency="critical", classification_summary="Hawkish", url="http://a.com/2", **base),
        NewsArticle(id="a3", headline="TSLA delivery numbers miss", summary="Below expectations", event_type="earnings",
                     sentiment_score=-0.4, urgency="high", classification_summary="Miss", url="http://a.com/3", **base),
        NewsArticle(id="a4", headline="New tech IPO priced at $40", summary="Cloud company goes public", event_type="ipo",
                     sentiment_score=0.3, urgency="medium", classification_summary="IPO pricing", url="http://a.com/4", **base),
        NewsArticle(id="a5", headline="Oil prices drop 5%", summary="Supply concerns ease", event_type="macro",
                     sentiment_score=-0.3, urgency="medium", classification_summary="Oil down", url="http://a.com/5", **base),
    ]


def make_gemini_response(json_str: str):
    """Build a mock Gemini response object."""
    resp = MagicMock()
    resp.text = json_str
    resp.usage_metadata = MagicMock(prompt_token_count=200, candidates_token_count=50)
    return resp


# ── Tests ───────────────────────────────────────────────────────────────────

class TestRankBatch:
    """ArticleRanker.rank_batch() tests."""

    @pytest.mark.asyncio
    async def test_rank_batch_sets_importance_scores(self, ranker, articles):
        """Valid Gemini response should set importance_score on each article."""
        scores = [
            {"id": "a1", "importance_score": 8.5},
            {"id": "a2", "importance_score": 9.0},
            {"id": "a3", "importance_score": 6.2},
            {"id": "a4", "importance_score": 4.1},
            {"id": "a5", "importance_score": 7.0},
        ]
        ranker.client.aio.models.generate_content.return_value = make_gemini_response(json.dumps(scores))

        result = await ranker.rank_batch(articles)

        article_map = {a.id: a for a in result}
        assert article_map["a1"].importance_score == 8.5
        assert article_map["a2"].importance_score == 9.0
        assert article_map["a3"].importance_score == 6.2
        assert article_map["a4"].importance_score == 4.1
        assert article_map["a5"].importance_score == 7.0

    @pytest.mark.asyncio
    async def test_rank_batch_empty_list(self, ranker):
        """Empty article list returns empty list, no API calls."""
        result = await ranker.rank_batch([])
        assert result == []
        assert not ranker.client.aio.models.generate_content.called

    @pytest.mark.asyncio
    async def test_rank_batch_no_llm_configured(self, ranker, articles):
        """When LLM is not configured, articles returned unchanged."""
        with patch("pipeline.ranker.is_configured", return_value=False):
            result = await ranker.rank_batch(articles)

            assert len(result) == len(articles)
            for a in result:
                assert a.importance_score is None
            assert not ranker.client.aio.models.generate_content.called

    @pytest.mark.asyncio
    async def test_rank_batch_invalid_json_response(self, ranker, articles):
        """Invalid JSON from Gemini should be handled gracefully (articles unchanged)."""
        ranker.client.aio.models.generate_content.return_value = make_gemini_response("not valid json {{{")

        result = await ranker.rank_batch(articles)

        assert len(result) == len(articles)
        # Importance scores should remain None since parsing failed
        for a in result:
            assert a.importance_score is None

    @pytest.mark.asyncio
    async def test_rank_batch_partial_mapping(self, ranker, articles):
        """If Gemini returns scores for only a subset of articles, only those get scores."""
        scores = [
            {"id": "a1", "importance_score": 8.0},
            # a2 is missing
            {"id": "a3", "importance_score": 5.0},
        ]
        ranker.client.aio.models.generate_content.return_value = make_gemini_response(json.dumps(scores))

        result = await ranker.rank_batch(articles)
        article_map = {a.id: a for a in result}

        assert article_map["a1"].importance_score == 8.0
        assert article_map["a3"].importance_score == 5.0
        assert article_map["a2"].importance_score is None  # Not in response
        assert article_map["a4"].importance_score is None
        assert article_map["a5"].importance_score is None

    @pytest.mark.asyncio
    async def test_rank_batch_scores_in_range(self, ranker, articles):
        """Importance scores must be in 0.0–10.0 range."""
        scores = [
            {"id": "a1", "importance_score": 10.0},
            {"id": "a2", "importance_score": 0.0},
            {"id": "a3", "importance_score": 5.5},
        ]
        ranker.client.aio.models.generate_content.return_value = make_gemini_response(json.dumps(scores))

        result = await ranker.rank_batch(articles[:3])

        for a in result:
            assert 0.0 <= a.importance_score <= 10.0

    @pytest.mark.asyncio
    async def test_rank_batch_logs_usage(self, ranker, articles):
        """LLM usage should be logged after a successful ranking."""
        scores = [{"id": a.id, "importance_score": 5.0} for a in articles]
        ranker.client.aio.models.generate_content.return_value = make_gemini_response(json.dumps(scores))

        await ranker.rank_batch(articles)

        ranker.db.log_llm_usage.assert_called_once()

    @pytest.mark.asyncio
    async def test_rank_batch_prompt_includes_article_data(self, ranker, articles):
        """The prompt sent to Gemini should include article metadata."""
        scores = [{"id": a.id, "importance_score": 5.0} for a in articles]
        ranker.client.aio.models.generate_content.return_value = make_gemini_response(json.dumps(scores))

        await ranker.rank_batch(articles)

        call_args = ranker.client.aio.models.generate_content.call_args
        contents = call_args[1].get("contents", "")
        # The prompt should contain article identifiers
        assert "a1" in contents
        assert articles[0].headline in contents

    @pytest.mark.asyncio
    async def test_rank_batch_single_article(self, ranker):
        """Ranking a single article should work."""
        article = NewsArticle(
            id="single", headline="Breaking news", summary="Big news",
            source_name="test", source_type="rss", url="http://a.com/single",
            published_at=datetime.now(timezone.utc),
            event_type="macro", sentiment_score=0.0, urgency="high",
        )
        scores = [{"id": "single", "importance_score": 9.5}]
        ranker.client.aio.models.generate_content.return_value = make_gemini_response(json.dumps(scores))

        result = await ranker.rank_batch([article])

        assert result[0].importance_score == 9.5
