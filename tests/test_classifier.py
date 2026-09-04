"""Tests for ArticleClassifier — classification logic and LLM response parsing."""

import json
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timezone

from data.models import NewsArticle


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def classifier(monkeypatch):
    """
    ArticleClassifier wired to a stubbed `complete`, so no real API calls.

    Tests patch the facade rather than a provider SDK — that is the point of
    having one: swapping the model behind MODEL_CLASSIFIER must not be able to
    break a test.
    """
    from pipeline.classifier import ArticleClassifier
    clf = ArticleClassifier(db=MagicMock())
    clf.model_name = "test/primary-model"
    monkeypatch.setattr("pipeline.classifier.settings.model_classifier", "test/primary-model")
    monkeypatch.setattr("pipeline.classifier.settings.model_classifier_fallback", "test/fallback-model")
    monkeypatch.setattr("pipeline.classifier.settings.model_reddit_sentiment", "test/primary-model")
    monkeypatch.setattr("pipeline.classifier.settings.model_reddit_sentiment_fallback", "")
    monkeypatch.setattr("pipeline.classifier.is_llm_configured", lambda: True)
    clf.complete = AsyncMock()
    monkeypatch.setattr("pipeline.classifier.complete", clf.complete)
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


def make_response(json_str: str):
    """Build what `config.llm.complete` returns, for any model."""
    from tests.conftest import make_llm_response
    return make_llm_response(text=json_str)


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
        classifier.complete.return_value = make_response(valid_json)

        result = await classifier.classify(earnings_article)

        assert result.event_type == "earnings"
        assert result.sentiment_score == 0.55
        assert result.urgency == "high"
        assert result.suggested_direction == "bullish"
        assert "AAPL" in result.affected_tickers
        assert "Technology" in result.affected_sectors
        assert classifier.complete.called
        assert classifier.db.log_llm_usage.called

    @pytest.mark.asyncio
    async def test_fallback_model_on_primary_failure(self, classifier, earnings_article):
        """Primary model fails → the configured fallback slug is tried next."""
        fallback_json = json.dumps({
            "event_type": "earnings",
            "sentiment_score": 0.50,
            "urgency": "high",
            "suggested_direction": "bullish",
            "affected_sectors": ["Technology"],
            "affected_tickers": ["AAPL"],
            "classification_summary": "Fallback classification.",
        })
        classifier.complete.side_effect = [
            Exception("primary model error"),
            make_response(fallback_json),
        ]

        result = await classifier.classify(earnings_article)

        assert result.event_type == "earnings"
        assert result.sentiment_score == 0.50
        assert classifier.complete.await_count == 2
        # Second attempt must go to the fallback slug, not retry the primary.
        assert classifier.complete.await_args_list[1].kwargs["model"] == "test/fallback-model"

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
        assert not classifier.complete.called
        
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
        classifier.complete.return_value = make_response(valid_json)

        result = await classifier.classify(reddit_article)

        assert result.event_type == "meme_stock"
        assert "GME" in result.affected_tickers
        # Check that the prompt sent to DeepSeek contains WSB-specific content
        user_prompt = classifier.complete.await_args.kwargs["prompt"]
        assert "WSB" in user_prompt or "meme_stock" in user_prompt

    @pytest.mark.asyncio
    async def test_invalid_json_from_llm(self, classifier, earnings_article):
        """LLM returns invalid JSON → article is returned unchanged (graceful failure)."""
        classifier.complete.return_value = make_response("not valid json at all {{{")

        result = await classifier.classify(earnings_article)

        # Article fields should NOT have been updated by the LLM
        assert result.event_type is None
        # The function didn't crash — article returned with original id intact
        assert result.id == earnings_article.id

    @pytest.mark.asyncio
    async def test_no_llm_configured(self, classifier, earnings_article):
        """When no LLM is configured, article is returned unchanged."""
        with patch("pipeline.classifier.is_llm_configured", return_value=False):
            result = await classifier.classify(earnings_article)

            assert not classifier.complete.called
            
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
        classifier.complete.return_value = make_response(valid_json)

        result = await classifier.classify(article)

        assert "AAPL" in result.affected_tickers  # Original ticker preserved
        assert "MSFT" in result.affected_tickers   # New ticker added

    @pytest.mark.asyncio
    async def test_logs_usage_on_success(self, classifier, earnings_article):
        """LLM usage is logged after a successful classification."""
        valid_json = json.dumps({"event_type": "earnings", "sentiment_score": 0.5, "urgency": "high",
                                 "suggested_direction": "bullish", "classification_summary": "OK"})
        classifier.complete.return_value = make_response(valid_json)

        await classifier.classify(earnings_article)

        classifier.db.log_llm_usage.assert_called_once()
        call_kwargs = classifier.db.log_llm_usage.call_args[1]
        assert call_kwargs["operation"] == "classify"
        assert call_kwargs["model_name"] == "test/primary-model"
        # Real reported cost, not an estimate from a price table.
        assert call_kwargs["cost_usd"] == pytest.approx(0.0001)


class TestClassifyBatch:
    """
    Batch classification. The failure mode that matters here is silently
    attaching one article's classification to a different article, so these
    focus on the id mapping rather than on happy-path parsing.
    """

    @staticmethod
    def _articles():
        return [
            NewsArticle(
                id=f"batch_{i}",
                headline=f"Company {i} beats earnings estimates",
                summary=f"Revenue and EPS both ahead of consensus for company {i}.",
                source_name="reuters",
                source_type="rss",
                url=f"https://example.com/{i}",
                published_at=datetime.now(timezone.utc),
            )
            for i in range(3)
        ]

    @staticmethod
    def _result(article_id, sentiment):
        return {
            "id": article_id,
            "event_type": "earnings",
            "sentiment_score": sentiment,
            "urgency": "high",
            "suggested_direction": "bullish",
            "affected_sectors": ["Technology"],
            "affected_tickers": [f"TCK{article_id[-1]}"],
            "countries": ["US"],
            "classification_summary": f"Summary for {article_id}",
        }

    @pytest.mark.asyncio
    async def test_results_map_by_label_not_position(self, classifier):
        """A reordered response must still land on the right articles.

        The batch goes out under short labels a1..aN rather than the article
        ids, because models degenerate on echoing a 30-character hash. The
        guarantee is unchanged: results are matched by the label that came
        back, so a reordered response still lands correctly.
        """
        articles = self._articles()
        # a1/a2/a3 correspond to batch_0/batch_1/batch_2, deliberately
        # reversed relative to the input order.
        payload = [
            self._result("a3", 0.2),
            self._result("a1", 0.9),
            self._result("a2", -0.5),
        ]
        with patch.object(
            classifier, "_call_with_fallback",
            AsyncMock(return_value=json.dumps(payload)),
        ):
            result = await classifier.classify_batch(articles)

        by_id = {a.id: a for a in result}
        assert by_id["batch_0"].sentiment_score == 0.9
        assert by_id["batch_1"].sentiment_score == -0.5
        assert by_id["batch_2"].sentiment_score == 0.2
        assert by_id["batch_0"].classification_summary == "Summary for a1"

    @pytest.mark.asyncio
    async def test_one_malformed_item_does_not_lose_the_others(self, classifier):
        articles = self._articles()
        payload = [
            self._result("a1", 0.4),
            {"id": "a2", "sentiment_score": "not-a-number", "urgency": "nope"},
            self._result("a3", -0.3),
        ]
        with patch.object(
            classifier, "_call_with_fallback",
            AsyncMock(return_value=json.dumps(payload)),
        ):
            result = await classifier.classify_batch(articles)

        by_id = {a.id: a for a in result}
        assert by_id["batch_0"].event_type == "earnings"
        assert by_id["batch_2"].event_type == "earnings"
        # The bad item is left untouched for the caller to mark as failed.
        assert by_id["batch_1"].event_type is None

    @pytest.mark.asyncio
    async def test_missing_result_leaves_article_untouched(self, classifier):
        articles = self._articles()
        payload = [self._result("a1", 0.4)]
        with patch.object(
            classifier, "_call_with_fallback",
            AsyncMock(return_value=json.dumps(payload)),
        ):
            result = await classifier.classify_batch(articles)

        by_id = {a.id: a for a in result}
        assert by_id["batch_0"].event_type == "earnings"
        assert by_id["batch_1"].event_type is None
        assert by_id["batch_2"].event_type is None

    @pytest.mark.asyncio
    async def test_batch_is_sent_under_short_labels_not_article_ids(self, classifier):
        """The prompt must not ask the model to echo 30-character hash ids.

        Doing so cost whole batches: a model that started repeating a run of
        characters from an id ran the response into its token cap and
        truncated the JSON.
        """
        articles = self._articles()
        mock_call = AsyncMock(return_value="[]")
        with patch.object(classifier, "_call_with_fallback", mock_call):
            await classifier.classify_batch(articles)

        prompt = mock_call.await_args.args[0]
        assert '"id": "a1"' in prompt
        assert '"id": "a3"' in prompt
        for article in articles:
            assert article.id not in prompt

    @pytest.mark.asyncio
    async def test_prefiltered_noise_is_never_sent_to_the_llm(self, classifier, noise_article):
        mock_call = AsyncMock(return_value="[]")
        with patch.object(classifier, "_call_with_fallback", mock_call):
            result = await classifier.classify_batch([noise_article])

        assert result[0].event_type == "noise"
        mock_call.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unparseable_response_leaves_batch_untouched(self, classifier):
        articles = self._articles()
        with patch.object(
            classifier, "_call_with_fallback",
            AsyncMock(return_value="not json at all"),
        ):
            result = await classifier.classify_batch(articles)

        assert all(a.event_type is None for a in result)

    @pytest.mark.asyncio
    async def test_reddit_and_news_are_sent_as_separate_calls(self, classifier, reddit_article):
        articles = self._articles() + [reddit_article]
        mock_call = AsyncMock(return_value="[]")
        with patch.object(classifier, "_call_with_fallback", mock_call):
            await classifier.classify_batch(articles)

        assert mock_call.await_count == 2
        # (prompt, models, operation, max_output_tokens)
        operations = {c[0][2] for c in mock_call.await_args_list}
        assert operations == {"classify_batch", "classify_batch_reddit"}
