"""Tests that all prompt templates are well-formed and contain required placeholders."""

import pytest
from pipeline.classifier import CLASSIFICATION_PROMPT, REDDIT_CLASSIFICATION_PROMPT
from pipeline.ranker import RANKING_PROMPT
from pipeline.agents import (
    _BULL_SYSTEM_MESSAGE,
    _BEAR_SYSTEM_MESSAGE,
    _TRADER_SYSTEM_MESSAGE,
    _COMMON_RULES,
)
from pipeline.chat_orchestrator import build_chat_prompt


class TestClassifierPrompts:
    """Verify classification prompts are well-formed."""

    def test_classification_prompt_is_non_empty(self):
        assert CLASSIFICATION_PROMPT and len(CLASSIFICATION_PROMPT) > 100

    def test_classification_prompt_has_required_placeholders(self):
        assert "{headline}" in CLASSIFICATION_PROMPT
        assert "{summary}" in CLASSIFICATION_PROMPT

    def test_classification_prompt_has_json_schema(self):
        assert "event_type" in CLASSIFICATION_PROMPT
        assert "sentiment_score" in CLASSIFICATION_PROMPT
        assert "urgency" in CLASSIFICATION_PROMPT
        assert "suggested_direction" in CLASSIFICATION_PROMPT

    def test_classification_prompt_has_sentiment_calibration(self):
        assert "SENTIMENT CALIBRATION" in CLASSIFICATION_PROMPT

    def test_classification_prompt_has_event_type_taxonomy(self):
        assert "EVENT TYPE TAXONOMY" in CLASSIFICATION_PROMPT
        assert "earnings" in CLASSIFICATION_PROMPT
        assert "macro" in CLASSIFICATION_PROMPT
        assert "geopolitical" in CLASSIFICATION_PROMPT

    def test_classification_prompt_has_urgency_definitions(self):
        assert "URGENCY" in CLASSIFICATION_PROMPT
        assert "critical" in CLASSIFICATION_PROMPT

    def test_classification_prompt_has_few_shot_examples(self):
        assert "Example 1:" in CLASSIFICATION_PROMPT
        assert "Example 2:" in CLASSIFICATION_PROMPT

    def test_reddit_prompt_is_non_empty(self):
        assert REDDIT_CLASSIFICATION_PROMPT and len(REDDIT_CLASSIFICATION_PROMPT) > 100

    def test_reddit_prompt_has_required_placeholders(self):
        assert "{headline}" in REDDIT_CLASSIFICATION_PROMPT
        assert "{summary}" in REDDIT_CLASSIFICATION_PROMPT
        assert "{comments}" in REDDIT_CLASSIFICATION_PROMPT

    def test_reddit_prompt_has_wsb_lingo_guide(self):
        assert "WSB LINGO GUIDE" in REDDIT_CLASSIFICATION_PROMPT
        assert "tendies" in REDDIT_CLASSIFICATION_PROMPT

    def test_reddit_prompt_has_meme_stock_type(self):
        assert "meme_stock" in REDDIT_CLASSIFICATION_PROMPT

    def test_reddit_and_news_prompts_are_distinct(self):
        """These should be different prompts, not copies of each other."""
        assert CLASSIFICATION_PROMPT != REDDIT_CLASSIFICATION_PROMPT


class TestRankerPrompts:
    """Verify ranking prompt is well-formed."""

    def test_ranker_prompt_is_non_empty(self):
        assert RANKING_PROMPT and len(RANKING_PROMPT) > 100

    def test_ranker_prompt_has_articles_placeholder(self):
        assert "{articles_json}" in RANKING_PROMPT

    def test_ranker_prompt_has_score_calibration(self):
        assert "SCORE CALIBRATION" in RANKING_PROMPT

    def test_ranker_prompt_has_scoring_factors(self):
        assert "SCORING FACTORS" in RANKING_PROMPT

    def test_ranker_prompt_has_json_schema(self):
        assert "importance_score" in RANKING_PROMPT

    def test_ranker_prompt_does_not_penalize_sectors(self):
        assert "Do NOT automatically penalize any sector" in RANKING_PROMPT

    def test_ranker_prompt_requires_json_array(self):
        assert "JSON array" in RANKING_PROMPT


class TestDebateSystemMessages:
    """Verify debate agent system messages are distinct and well-formed."""

    def test_bull_system_message_is_non_empty(self):
        assert _BULL_SYSTEM_MESSAGE and len(_BULL_SYSTEM_MESSAGE) > 50

    def test_bear_system_message_is_non_empty(self):
        assert _BEAR_SYSTEM_MESSAGE and len(_BEAR_SYSTEM_MESSAGE) > 50

    def test_trader_system_message_is_non_empty(self):
        assert _TRADER_SYSTEM_MESSAGE and len(_TRADER_SYSTEM_MESSAGE) > 50

    def test_common_rules_is_non_empty(self):
        assert _COMMON_RULES and len(_COMMON_RULES) > 50

    def test_bull_and_bear_messages_are_distinct(self):
        """Bull and Bear should have different perspectives, not copy-pasted."""
        assert _BULL_SYSTEM_MESSAGE != _BEAR_SYSTEM_MESSAGE

    def test_bull_message_mentions_bullish_traits(self):
        assert "bullish" in _BULL_SYSTEM_MESSAGE.lower()
        assert "long thesis" in _BULL_SYSTEM_MESSAGE.lower()

    def test_bear_message_mentions_bearish_traits(self):
        assert "skeptical" in _BEAR_SYSTEM_MESSAGE.lower()
        assert "risk" in _BEAR_SYSTEM_MESSAGE.lower()

    def test_trader_message_mentions_synthesis(self):
        assert "synthesize" in _TRADER_SYSTEM_MESSAGE.lower()
        assert "conviction" in _TRADER_SYSTEM_MESSAGE.lower()

    def test_common_rules_mentions_grounding(self):
        assert "news" in _COMMON_RULES.lower()
        assert "CRITICAL RULES" in _COMMON_RULES


class TestChatPrompts:
    """Verify chat/orchestrator prompts are well-formed."""

    def test_build_chat_prompt_accepts_query(self):
        """build_chat_prompt should be callable with a query."""
        result = build_chat_prompt("What about AAPL?")
        assert result and len(result) > 10

    def test_build_chat_prompt_includes_context_when_provided(self):
        result = build_chat_prompt("What about AAPL?", context="AAPL is a tech company.")
        assert result and len(result) > 10

    def test_build_chat_prompt_returns_different_for_different_queries(self):
        r1 = build_chat_prompt("What about AAPL?")
        r2 = build_chat_prompt("What about TSLA?")
        assert r1 != r2

    def test_build_chat_prompt_without_context(self):
        """Should work with only the query, no context argument."""
        result = build_chat_prompt("Analyze the market")
        assert "Analyze the market" in result
