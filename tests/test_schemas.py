"""Tests that all Pydantic models / JSON schemas used for LLM response parsing are valid."""

import json
import pytest
from pathlib import Path
from pydantic import ValidationError
from pipeline.classifier import ClassifierResult
from config.llm import parse_structured, salvage_json_field
from data.models import TickerNote, notes_to_dict
from pipeline.agents import TraderAdvisory
from pipeline.predictor import LlmPrediction

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
            '{"executive_summary": "TLDR: BUY - asymmetric setup.",'
            ' "full_advisory": "\nTrade Action\nAction: BUY\n"}'
        )
        with pytest.raises(json.JSONDecodeError):
            json.loads(raw)  # the failure this guards against

        result = parse_structured(raw, TraderAdvisory)
        assert result.executive_summary.startswith("TLDR: BUY")
        assert "Trade Action" in result.full_advisory

    def test_parses_fenced_json(self):
        raw = '```json\n{"executive_summary": "TLDR: HOLD.", "full_advisory": "### Call\nHold."}\n```'
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
