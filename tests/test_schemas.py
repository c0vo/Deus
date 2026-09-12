"""Tests that all Pydantic models / JSON schemas used for LLM response parsing are valid."""

import json
import pytest
from pathlib import Path
from pydantic import ValidationError
from pipeline.classifier import ClassifierResult
from config.llm import parse_structured, salvage_json_field
from data.models import TickerNote, notes_to_dict
from pipeline.agents import TraderAdvisory
from pipeline.daily_stance import NO_CALL, UNAVAILABLE, Stance, StanceRow
from pipeline.weekly_tip import Tip
from pipeline.predictor import LlmPrediction
from pipeline.grounded_answer import GradeVerdict, MoveExplanation

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestClassifierSchema:
    """Verify ClassifierResult Pydantic model validates correctly."""

    def test_valid_classifier_result(self):
        data = {
            "event_type": "earnings",
            "sentiment_score": 0.5,
            "urgency": "high",
            "suggested_direction": "bullish",
            "affected_sectors": ["Technology"],
            "affected_tickers": ["AAPL"],
            "classification_summary": "Strong earnings beat.",
        }
        result = ClassifierResult.model_validate(data)
        assert result.event_type == "earnings"
        assert result.sentiment_score == 0.5

    def test_validates_from_deepseek_fixture(self):
        path = FIXTURES_DIR / "classifier_response_deepseek.json"
        if not path.exists():
            pytest.skip("Fixture file not found")
        data = json.loads(path.read_text())
        result = ClassifierResult.model_validate(data)
        assert result.event_type == "earnings"

    def test_validates_from_gemini_fixture(self):
        path = FIXTURES_DIR / "classifier_response_gemini.json"
        if not path.exists():
            pytest.skip("Fixture file not found")
        data = json.loads(path.read_text())
        result = ClassifierResult.model_validate(data)
        assert result.event_type == "macro"

    def test_rejects_sentiment_too_high(self):
        with pytest.raises(ValidationError):
            ClassifierResult.model_validate({
                "event_type": "earnings",
                "sentiment_score": 99.0,
                "urgency": "low",
                "suggested_direction": "neutral",
                "classification_summary": "",
            })

    def test_rejects_sentiment_too_low(self):
        with pytest.raises(ValidationError):
            ClassifierResult.model_validate({
                "event_type": "earnings",
                "sentiment_score": -99.0,
                "urgency": "low",
                "suggested_direction": "neutral",
                "classification_summary": "",
            })

    def test_rejects_invalid_urgency(self):
        with pytest.raises(ValidationError):
            ClassifierResult.model_validate({
                "event_type": "earnings",
                "sentiment_score": 0.0,
                "urgency": "super_critical",
                "suggested_direction": "neutral",
                "classification_summary": "",
            })

    def test_rejects_invalid_direction(self):
        with pytest.raises(ValidationError):
            ClassifierResult.model_validate({
                "event_type": "earnings",
                "sentiment_score": 0.0,
                "urgency": "low",
                "suggested_direction": "super_bullish",
                "classification_summary": "",
            })

    def test_accepts_minimal_valid(self):
        """Test the minimum valid schema — all fields with defaults."""
        result = ClassifierResult()
        assert result.event_type == "unknown"
        assert result.sentiment_score == 0.0
        assert result.urgency == "low"
        assert result.suggested_direction == "neutral"
        assert result.affected_sectors == []
        assert result.affected_tickers == []
        assert result.classification_summary == ""

    def test_parses_json_from_fixture_string(self):
        """Test that model_validate_json works on raw JSON strings (like LLM output)."""
        path = FIXTURES_DIR / "classifier_response_deepseek.json"
        if not path.exists():
            pytest.skip("Fixture file not found")
        raw_json = path.read_text()
        result = ClassifierResult.model_validate_json(raw_json)
        assert result.sentiment_score == 0.55

    def test_rejects_sentiment_as_string(self):
        """Sentiment must be a number, not a string."""
        with pytest.raises(ValidationError):
            ClassifierResult.model_validate({
                "event_type": "earnings",
                "sentiment_score": "high",
                "urgency": "low",
                "suggested_direction": "neutral",
                "classification_summary": "",
            })


class TestRankerResponseSchema:
    """Verify the ranking response (JSON array of {id, importance_score}) is handled correctly."""

    def test_ranking_response_structure(self):
        path = FIXTURES_DIR / "ranker_response.json"
        if not path.exists():
            pytest.skip("Fixture file not found")
        data = json.loads(path.read_text())
        assert isinstance(data, list)
        for item in data:
            assert "id" in item
            assert "importance_score" in item
            assert isinstance(item["importance_score"], (int, float))
            assert 0.0 <= item["importance_score"] <= 10.0


class TestDebateResponseSchema:
    """Verify debate fixture structures."""

    def test_bull_fixture_structure(self):
        path = FIXTURES_DIR / "debate_bull_response.json"
        if not path.exists():
            pytest.skip("Fixture file not found")
        data = json.loads(path.read_text())
        assert "speaker" in data
        assert "round" in data
        assert "content" in data
        assert data["speaker"] == "bull"

    def test_bear_fixture_structure(self):
        path = FIXTURES_DIR / "debate_bear_response.json"
        if not path.exists():
            pytest.skip("Fixture file not found")
        data = json.loads(path.read_text())
        assert "speaker" in data
        assert "round" in data
        assert "content" in data
        assert data["speaker"] == "bear"

    def test_trader_fixture_structure(self):
        path = FIXTURES_DIR / "trader_synthesis.json"
        if not path.exists():
            pytest.skip("Fixture file not found")
        data = json.loads(path.read_text())
        assert "direction" in data
        assert "conviction" in data
        assert "executive_summary" in data
        assert "full_advisory" in data
        assert len(data["executive_summary"]) > 0
        assert len(data["full_advisory"]) > 0

    def test_trader_has_buy_sell_hold_recommendation(self):
        path = FIXTURES_DIR / "trader_synthesis.json"
        if not path.exists():
            pytest.skip("Fixture file not found")
        data = json.loads(path.read_text())
        summary = data["executive_summary"].upper()
        assert any(word in summary for word in ["BUY", "SELL", "HOLD"])


class TestStructuredOutputParsing:
    """
    Regression cover for the malformed-advisory bug.

    A Trader response with real newlines inside `full_advisory` used to raise
    `Invalid control character`, and the except branch handed `response.text`
    — the raw JSON blob — to the frontend, which rendered it verbatim.
    """

    def test_parses_unescaped_newlines_in_string_field(self):
        raw = (
            '{"direction": "BUY", "conviction": "High",'
            ' "executive_summary": "TLDR: BUY - asymmetric setup.",'
            ' "full_advisory": "\nTrade Action\nAction: BUY\n"}'
        )
        with pytest.raises(json.JSONDecodeError):
            json.loads(raw)  # the failure this guards against

        result = parse_structured(raw, TraderAdvisory)
        assert result.executive_summary.startswith("TLDR: BUY")
        assert "Trade Action" in result.full_advisory

    def test_parses_fenced_json(self):
        raw = ('```json\n{"direction": "HOLD", "conviction": "Low",'
               ' "executive_summary": "TLDR: HOLD.", "full_advisory": "### Call\nHold."}\n```')
        result = parse_structured(raw, TraderAdvisory)
        assert result.executive_summary == "TLDR: HOLD."

    def test_missing_required_field_is_rejected(self):
        with pytest.raises(ValidationError):
            parse_structured('{"executive_summary": "TLDR: BUY."}', TraderAdvisory)

    def test_salvages_prose_from_unparseable_blob(self):
        # Truncated mid-object with a doubled tail — beyond any JSON decoder.
        broken = (
            '{"executive_summary": "TLDR: BUY - dense catalysts.",'
            ' "full_advisory": "\nTrade Action\n'
        )
        assert salvage_json_field(broken, "executive_summary") == "TLDR: BUY - dense catalysts."

    def test_salvage_returns_none_for_absent_field(self):
        assert salvage_json_field('{"other": "x"}', "full_advisory") is None

    def test_llm_prediction_rejects_out_of_range_confidence(self):
        with pytest.raises(ValidationError):
            parse_structured(
                '{"direction": "UP", "confidence": 1.4, "narrative": "x"}', LlmPrediction
            )

    def test_llm_prediction_rejects_unknown_direction(self):
        with pytest.raises(ValidationError):
            parse_structured(
                '{"direction": "SIDEWAYS", "confidence": 0.6, "narrative": "x"}', LlmPrediction
            )

    def test_ticker_notes_collapse_to_dict(self):
        raw = '[{"ticker": "skhy", "summary": "HBM demand."}, {"ticker": "NVDA", "summary": "AI capex."}]'
        assert notes_to_dict(parse_structured(raw, list[TickerNote])) == {
            "SKHY": "HBM demand.",
            "NVDA": "AI capex.",
        }

    def test_notes_to_dict_accepts_raw_dicts(self):
        # `response.parsed` is typed loosely enough that a list schema is not
        # guaranteed to come back as model instances.
        assert notes_to_dict([{"ticker": "aapl", "summary": "x"}]) == {"AAPL": "x"}


class TestGroundedAnswerSchemas:
    """
    GradeVerdict and MoveExplanation decide whether an alert says something
    specific or says nothing. Both are parsed with `strict` off on the
    json_schema, so the fallback path matters as much as the happy one.
    """

    def test_grade_verdict_parses_fixture(self):
        data = (FIXTURES_DIR / "grader_response.json").read_text()
        verdict = parse_structured(data, GradeVerdict)
        assert verdict.sufficient is True
        assert verdict.specificity == "specific"
        assert "2026-09-11" in verdict.reason

    def test_grade_verdict_rejects_unknown_specificity(self):
        with pytest.raises(ValidationError):
            parse_structured(
                '{"sufficient": false, "specificity": "vague", "reason": "x"}',
                GradeVerdict,
            )

    def test_grade_verdict_reason_is_optional(self):
        verdict = parse_structured(
            '{"sufficient": false, "specificity": "none"}', GradeVerdict
        )
        assert verdict.reason == ""

    def test_move_explanation_parses_fixture(self):
        data = (FIXTURES_DIR / "move_explanation.json").read_text()
        explanation = parse_structured(data, MoveExplanation)
        assert explanation.catalyst_found is True
        assert explanation.catalyst_kind == "company"
        assert explanation.source_indices == [2, 1]

    def test_move_explanation_defaults_to_uncited_unknown(self):
        """A bare catalyst_found=false must validate — it is the honest answer."""
        explanation = parse_structured('{"catalyst_found": false}', MoveExplanation)
        assert explanation.catalyst_kind == "unknown"
        assert explanation.source_indices == []
        assert explanation.cause == ""

    def test_move_explanation_rejects_unknown_catalyst_kind(self):
        with pytest.raises(ValidationError):
            parse_structured(
                '{"catalyst_found": true, "catalyst_kind": "astrology"}',
                MoveExplanation,
            )


class TestStanceSchema:
    """
    The morning stance schema.

    `Stance.action` carries a four-value Literal because that Literal IS the
    JSON schema the model is handed. NO CALL and UNAVAILABLE are statements
    about the pipeline, not about the position, so they live on `StanceRow` and
    are unspellable here — which is what keeps "the model did not answer" from
    ever being rendered as a considered HOLD.
    """

    VALID = {
        "ticker": "NVDA",
        "action": "TRIM",
        "conviction": "Medium",
        "thesis": "Daily rating flipped to Sell with RSI14 at 71.",
        "key_risk": "Analyst target still implies +14% upside.",
        "evidence_used": ["rsi", "technical_rating", "analyst_target"],
        "what_would_change_my_mind": "A daily close back above $185.",
    }

    def test_valid_stance(self):
        stance = Stance.model_validate(self.VALID)
        assert stance.action == "TRIM"
        assert stance.evidence_used == ["rsi", "technical_rating", "analyst_target"]

    @pytest.mark.parametrize("action", ["BUY/ADD", "HOLD", "TRIM", "SELL"])
    def test_every_action_is_available(self, action):
        """All four, not just HOLD and SELL — that menu was the original bug."""
        assert Stance.model_validate({**self.VALID, "action": action}).action == action

    @pytest.mark.parametrize("action", ["MOON", "BUY", "buy/add", NO_CALL, UNAVAILABLE])
    def test_rejects_actions_outside_the_literal(self, action):
        with pytest.raises(ValidationError):
            Stance.model_validate({**self.VALID, "action": action})

    def test_rejects_conviction_outside_the_literal(self):
        with pytest.raises(ValidationError):
            Stance.model_validate({**self.VALID, "conviction": "Very High"})

    def test_evidence_defaults_to_empty(self):
        payload = {k: v for k, v in self.VALID.items() if k != "evidence_used"}
        assert Stance.model_validate(payload).evidence_used == []

    def test_parses_from_a_bare_array(self):
        """What the `parse_structured` fallback in `compose` has to swallow."""
        raw = json.dumps([self.VALID, {**self.VALID, "ticker": "AMD",
                                       "action": "BUY/ADD"}])
        rows = parse_structured(raw, list[Stance])
        assert [r.action for r in rows] == ["TRIM", "BUY/ADD"]

    def test_stance_row_carries_the_codes_the_schema_cannot(self):
        for action in (NO_CALL, UNAVAILABLE):
            row = StanceRow(ticker="QQQ", action=action)
            assert row.action == action

    def test_stance_row_from_stance_upper_cases_the_ticker(self):
        row = StanceRow.from_stance(Stance.model_validate({**self.VALID,
                                                          "ticker": "nvda"}))
        assert row.ticker == "NVDA"
        assert row.thesis == self.VALID["thesis"]


class TestWeeklyTipSchema:
    def test_valid_tip_and_severity(self):
        tip = Tip.model_validate({
            "title": "FOMC week",
            "precedent": "Median -1.5% across 34 years.",
            "evidence": "SPY September statistics.",
            "action": "Watch the release.",
            "severity": "warning",
        })
        assert tip.severity == "warning"

    def test_invalid_severity_is_rejected(self):
        with pytest.raises(ValidationError):
            Tip.model_validate({
                "title": "x", "precedent": "x", "evidence": "x",
                "action": "x", "severity": "urgent",
            })
