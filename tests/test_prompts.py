"""Tests that all prompt templates are well-formed and contain required placeholders."""

from pathlib import Path

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
from pipeline.grounded_answer import (
    GRADER_PROMPT,
    HONESTY_SENTENCE,
    MOVE_EXPLANATION_PROMPT,
)
from pipeline.daily_stance import (
    DAILY_STANCE_PROMPT,
    FIELD_MAX_WORDS,
    THESIS_MAX_WORDS,
)
from pipeline.weekly_tip import WEEKLY_TIP_PROMPT


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

    def test_ranker_prompt_requires_one_result_per_article(self):
        """The root shape now travels as a json_schema, not as prose. What the
        prompt still has to carry is the per-article contract: asking for a
        bare array while response_format demanded an object is what got a whole
        batch answered with one flat result."""
        assert "one result per input article" in RANKING_PROMPT
        assert "matched by id, not by position" in RANKING_PROMPT
        assert "JSON array" not in RANKING_PROMPT


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

    def test_no_edge_baseline_is_not_directional_evidence(self):
        """A prior baseline is the base rate; argued as a call it becomes invented conviction."""
        assert "no measurable edge" in _COMMON_RULES
        assert "no measurable edge" in _TRADER_SYSTEM_MESSAGE


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

    def test_honesty_rule_is_absent_by_default(self):
        assert HONESTY_SENTENCE not in build_chat_prompt("Why is NVDA down?", "ctx")

    def test_honesty_rule_appears_when_required(self):
        prompt = build_chat_prompt(
            "Why is NVDA down?", "some weakly related context",
            honesty_required=True,
        )
        assert HONESTY_SENTENCE in prompt
        assert "Do NOT invent a cause" in prompt

    def test_honesty_rule_appears_without_any_context(self):
        prompt = build_chat_prompt("Why is NVDA down?", "", honesty_required=True)
        assert HONESTY_SENTENCE in prompt

    def test_honesty_rule_is_suppressed_when_web_results_exist(self):
        """The rule would be a lie: a web block is grounding by definition."""
        context = (
            "IN-HOUSE NEWS\nnothing much\n\n"
            "LIVE WEB SEARCH RESULTS\n1. [reuters] Guidance cut"
        )
        prompt = build_chat_prompt(
            "Why is NVDA down?", context, honesty_required=True
        )
        assert HONESTY_SENTENCE not in prompt


class TestGroundedAnswerPrompts:
    """
    The two prompts that stop an alert from being generic. The no-speculation
    rule is the whole mechanism: without it the model answers "profit taking"
    for every ticker on every red day, which is what these replace.
    """

    def test_move_prompt_is_non_empty(self):
        assert MOVE_EXPLANATION_PROMPT and len(MOVE_EXPLANATION_PROMPT) > 200

    def test_move_prompt_has_required_placeholders(self):
        for token in ("{ticker}", "{direction}", "{abs_pct", "{price",
                      "{session_line}", "{macro_line}", "{evidence}"):
            assert token in MOVE_EXPLANATION_PROMPT

    def test_move_prompt_forbids_speculation(self):
        assert "Do NOT speculate" in MOVE_EXPLANATION_PROMPT
        assert "general knowledge" in MOVE_EXPLANATION_PROMPT

    def test_move_prompt_names_the_generic_answers_it_rejects(self):
        """Naming them is what makes the rule enforceable by the model."""
        assert "Profit taking" in MOVE_EXPLANATION_PROMPT
        assert "risk-off sentiment" in MOVE_EXPLANATION_PROMPT

    def test_move_prompt_requires_citation_indices(self):
        assert "source_indices" in MOVE_EXPLANATION_PROMPT
        assert "catalyst_found=false" in MOVE_EXPLANATION_PROMPT

    def test_move_prompt_says_an_unexplained_move_is_an_answer(self):
        assert "unexplained move is a legitimate" in MOVE_EXPLANATION_PROMPT

    def test_grader_prompt_has_required_placeholders(self):
        for token in ("{purpose}", "{query}", "{context}"):
            assert token in GRADER_PROMPT

    def test_grader_prompt_errs_toward_false(self):
        assert "Err toward false" in GRADER_PROMPT

    def test_grader_prompt_does_not_ask_for_an_answer(self):
        assert "not answering the question" in GRADER_PROMPT


class TestDailyStancePrompt:
    """The morning stance rules, and the absence of the rule they replaced."""

    def test_prompt_is_non_empty(self):
        assert DAILY_STANCE_PROMPT and len(DAILY_STANCE_PROMPT) > 500

    def test_hold_is_not_a_default(self):
        """The load-bearing rule: without it the model reverts to HOLD for all."""
        assert "HOLD is not a default" in DAILY_STANCE_PROMPT

    def test_all_four_actions_are_offered(self):
        for action in ("BUY/ADD", "HOLD", "TRIM", "SELL"):
            assert action in DAILY_STANCE_PROMPT

    def test_no_news_still_requires_quoted_indicators(self):
        lowered = DAILY_STANCE_PROMPT.lower()
        assert "no news" in lowered
        assert "technicals" in lowered and "positioning" in lowered

    def test_absent_data_is_framed_as_information(self):
        assert "no analyst coverage" in DAILY_STANCE_PROMPT
        assert "not a gap to fill" in DAILY_STANCE_PROMPT

    def test_no_edge_means_no_model_signal(self):
        """The fact block writes a prior row as "no edge"; the rules say what that means."""
        assert '"no edge" means the model has no signal' in DAILY_STANCE_PROMPT

    def test_model_is_not_asked_whether_it_changed_its_mind(self):
        """changed_since_yesterday is computed from the record, never asked."""
        assert "Do NOT mention yesterday's stance" in DAILY_STANCE_PROMPT

    def test_falsifier_must_be_checkable(self):
        assert "what_would_change_my_mind" in DAILY_STANCE_PROMPT
        assert "never a sentiment" in DAILY_STANCE_PROMPT

    def test_free_text_fields_are_word_capped(self):
        """One batched response per morning: an unbounded field is paid per ticker."""
        assert f"thesis: at most {THESIS_MAX_WORDS} words" in DAILY_STANCE_PROMPT
        assert f"key_risk: at most {FIELD_MAX_WORDS} words" in DAILY_STANCE_PROMPT
        assert (f"what_would_change_my_mind: at most {FIELD_MAX_WORDS} words"
                in DAILY_STANCE_PROMPT)

    def test_scheduler_no_longer_defaults_to_hold(self):
        """
        The old advisor prompt lived inline in send_daily_advisor and told the
        model to fall back to HOLD. Nothing in the scheduler may say that again:
        it is the instruction that made every morning note identical, and the
        two hardcoded strings below did the same thing without an LLM at all.
        """
        source = (Path(__file__).resolve().parents[1]
                  / "orchestrator" / "scheduler.py").read_text(encoding="utf-8")
        assert "Default to HOLD" not in source
        assert "HOLD - No significant news today." not in source
        assert "HOLD - Unable to generate advice." not in source


class TestWeeklyTipPrompt:
    def test_prompt_limits_the_model_to_the_fact_block(self):
        assert "ONLY" in WEEKLY_TIP_PROMPT
        assert "FACTS:" in WEEKLY_TIP_PROMPT
