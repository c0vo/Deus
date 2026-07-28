"""Tests that all Pydantic models / JSON schemas used for LLM response parsing are valid."""

import json
import pytest
from pathlib import Path
from pydantic import ValidationError
from pipeline.classifier import ClassifierResult

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
