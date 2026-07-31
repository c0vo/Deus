"""Tests for AdvisoryGraph — Bull/Bear debate and Trader synthesis."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timezone


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_db():
    db = MagicMock()
    db.log_llm_usage.return_value = None
    return db


@pytest.fixture
def advisory_graph(mock_db):
    """Create AdvisoryGraph with mocked db (no real API/database calls)."""
    from pipeline.agents import AdvisoryGraph
    return AdvisoryGraph(db=mock_db)


@pytest.fixture
def minimal_state():
    """A minimal AdvisoryState dictionary."""
    return {
        "ticker": "AAPL",
        "ml_prediction": {"predicted_direction": "UP", "confidence": 0.75},
        "past_lessons": {
            "ticker_lessons": [],
            "sector_lessons": [],
            "market_lessons": [],
        },
        "news_context": "AAPL reported strong earnings. Revenue up 8% YoY.",
        "fundamentals_report": "",
        "technical_report": "",
        "debate_history": [],
        "debate_round_count": 0,
        "final_advisory": "",
    }


# ── Graph Structure Tests ───────────────────────────────────────────────────

class TestGraphStructure:
    """AdvisoryGraph.build_graph() and node structure."""

    def test_build_graph_compiles(self, advisory_graph):
        """build_graph() should return a compiled LangGraph state graph."""
        graph = advisory_graph.build_graph()
        # It should have a .ainvoke method (CompiledGraph)
        assert hasattr(graph, "ainvoke")
        assert hasattr(graph, "invoke")

    def test_graph_has_all_nodes(self, advisory_graph):
        """Graph should have all expected node names."""
        graph = advisory_graph.build_graph()
        # Use the graph's internal node registry
        nodes = graph.nodes
        assert "aggregate_data" in nodes
        assert "bull_researcher" in nodes
        assert "bear_researcher" in nodes
        assert "trader_risk_manager" in nodes


# ── Aggregrate Data Node Tests ──────────────────────────────────────────────

class TestAggregateDataNode:
    """aggregate_data_node() populates fundamentals/technicals."""

    @pytest.mark.asyncio
    async def test_populates_reports(self, advisory_graph, minimal_state):
        """With valid ticker, should populate both reports."""
        with patch("pipeline.agents.yf.Ticker") as mock_ticker:
            mock_info = MagicMock()
            mock_info.get.side_effect = lambda key, default="N/A": {
                "trailingPE": 28.5,
                "forwardPE": 25.0,
                "revenueGrowth": 0.08,
                "profitMargins": 0.25,
            }.get(key, default)
            mock_ticker.return_value.info = mock_info

            result = await advisory_graph.aggregate_data_node(minimal_state)

        assert "fundamentals_report" in result
        assert "technical_report" in result
        assert "P/E=28.5" in result["fundamentals_report"]
        assert "ML Predicts UP" in result["technical_report"]
        assert result["debate_history"] == []
        assert result["debate_round_count"] == 0

    @pytest.mark.asyncio
    async def test_handles_yfinance_error(self, advisory_graph, minimal_state):
        """If yfinance fails, error is captured in the report."""
        with patch("pipeline.agents.yf.Ticker") as mock_ticker:
            mock_ticker.return_value.info = {}
            mock_ticker.return_value.info.get.side_effect = Exception("API error")

            result = await advisory_graph.aggregate_data_node(minimal_state)

        assert "Error fetching fundamentals" in result["fundamentals_report"]

    @pytest.mark.asyncio
    async def test_with_unknown_prediction(self, advisory_graph):
        """When ML prediction is UNKNOWN, use the no-model template."""
        state = {
            "ticker": "TSLA",
            "ml_prediction": {"predicted_direction": "UNKNOWN", "confidence": 0.0},
            "past_lessons": {},
            "news_context": "Some news.",
            "fundamentals_report": "",
            "technical_report": "",
            "debate_history": [],
            "debate_round_count": 0,
            "final_advisory": "",
        }
        with patch("pipeline.agents.yf.Ticker") as mock_ticker:
            mock_ticker.return_value.info = {"trailingPE": 50.0}
            result = await advisory_graph.aggregate_data_node(state)

        assert "No trained ML model exists" in result["technical_report"]


# ── Consensus Detection Tests ───────────────────────────────────────────────

class TestConsensusDetection:
    """_debate_has_consensus() heuristic logic."""

    def test_no_consensus_when_bull_bullish_and_bear_bearish(self, advisory_graph):
        """Genuine disagreement — Bull says bullish things, Bear says bearish things."""
        state = {
            "debate_history": [
                "Bull: AAPL has strong growth and upside catalysts. The bullish momentum is clear.",
                "Bear: AAPL faces downside risk and headwinds. The concerns are bearish.",
            ],
            "debate_round_count": 1,
        }
        assert advisory_graph._debate_has_consensus(state) is False

    def test_consensus_when_both_bullish(self, advisory_graph):
        """Both lean bullish → skip to trader."""
        state = {
            "debate_history": [
                "Bull: AAPL has strong growth. Bullish outlook with upside.",
                "Bear: I concede the bull case. The growth and momentum are strong.",
            ],
            "debate_round_count": 1,
        }
        assert advisory_graph._debate_has_consensus(state) is True

    def test_consensus_when_both_bearish(self, advisory_graph):
        """Both lean bearish → skip to trader."""
        state = {
            "debate_history": [
                "Bull: The risks are concerning. Downside is likely.",
                "Bear: Strong headwinds ahead. Bearish outlook with downside risk.",
            ],
            "debate_round_count": 1,
        }
        assert advisory_graph._debate_has_consensus(state) is True

    def test_no_consensus_with_short_history(self, advisory_graph):
        """Less than 2 entries → cannot determine consensus."""
        state = {
            "debate_history": ["Bull: Strong growth!"],
            "debate_round_count": 0,
        }
        assert advisory_graph._debate_has_consensus(state) is False

    def test_no_consensus_with_empty_history(self, advisory_graph):
        state = {"debate_history": [], "debate_round_count": 0}
        assert advisory_graph._debate_has_consensus(state) is False


# ── Should Continue Debate Tests ────────────────────────────────────────────

class TestShouldContinueDebate:
    """Conditional routing logic."""

    def test_route_to_trader_after_two_rounds(self, advisory_graph):
        state = {"debate_round_count": 2, "debate_history": ["Bull: X", "Bear: Y"]}
        assert advisory_graph.should_continue_debate(state) == "trader_risk_manager"

    def test_route_to_trader_with_consensus(self, advisory_graph):
        state = {
            "debate_round_count": 1,
            "debate_history": [
                "Bull: Strong growth and catalysts.",
                "Bear: I agree, the growth is undeniable.",
            ],
        }
        assert advisory_graph.should_continue_debate(state) == "trader_risk_manager"

    def test_route_to_bull_without_consensus(self, advisory_graph):
        state = {
            "debate_round_count": 1,
            "debate_history": [
                "Bull: Strong growth and bullish upside!",
                "Bear: Serious downside risk and bearish headwinds!",
            ],
        }
        assert advisory_graph.should_continue_debate(state) == "bull_researcher"


# ── Researcher Node Tests ───────────────────────────────────────────────────

class TestResearcherNodes:
    """Bull and Bear researcher nodes."""

    @pytest.mark.asyncio
    async def test_bull_researcher_appends_to_history(self, advisory_graph, minimal_state):
        """Bull node appends to debate_history with 'Bull:' prefix."""
        # Mock _call_deepseek to return a canned response
        advisory_graph._call_deepseek = AsyncMock(return_value="AAPL has strong catalysts for growth.")

        result = await advisory_graph.bull_researcher_node(minimal_state)

        assert len(result["debate_history"]) == 1
        assert result["debate_history"][0].startswith("Bull:")
        assert "catalysts" in result["debate_history"][0]

    @pytest.mark.asyncio
    async def test_bear_researcher_appends_and_increments_round(self, advisory_graph, minimal_state):
        """Bear node appends and increments debate_round_count."""
        # Set up initial state with some debate history
        state = dict(minimal_state)
        state["debate_history"] = ["Bull: Initial bull case."]
        state["debate_round_count"] = 0
        advisory_graph._call_deepseek = AsyncMock(return_value="The bull ignores key risks.")

        result = await advisory_graph.bear_researcher_node(state)

        assert len(result["debate_history"]) == 2
        assert result["debate_history"][1].startswith("Bear:")
        assert result["debate_round_count"] == 1

    @pytest.mark.asyncio
    async def test_bull_uses_correct_system_message(self, advisory_graph, minimal_state):
        """Bull node should pass the composed bull system message."""
        from pipeline.agents import _BULL_SYSTEM, _BULL_SYSTEM_MESSAGE, _COMMON_RULES
        advisory_graph._call_deepseek = AsyncMock(return_value="Bull case.")

        await advisory_graph.bull_researcher_node(minimal_state)

        # Verify _call_deepseek was called with the bull system message
        call_kwargs = advisory_graph._call_deepseek.call_args
        assert call_kwargs is not None, "_call_deepseek was not called"
        assert call_kwargs[1].get("system_message") == _BULL_SYSTEM
        assert _BULL_SYSTEM_MESSAGE in _BULL_SYSTEM

    def test_shared_rules_live_in_the_system_message_not_the_user_prompt(self):
        """
        The rules are invariant, so they belong in the cacheable prefix. If they
        drift back into the user prompt they land after the debate history,
        which changes every round and defeats prefix caching.
        """
        from pipeline.agents import _BULL_SYSTEM, _BEAR_SYSTEM, _COMMON_RULES
        assert _COMMON_RULES in _BULL_SYSTEM
        assert _COMMON_RULES in _BEAR_SYSTEM

    @pytest.mark.asyncio
    async def test_bull_prompt_does_not_repeat_the_shared_rules(self, advisory_graph, minimal_state):
        from pipeline.agents import _COMMON_RULES
        advisory_graph._call_deepseek = AsyncMock(return_value="Bull case.")

        await advisory_graph.bull_researcher_node(minimal_state)

        user_prompt = advisory_graph._call_deepseek.call_args[0][0]
        assert _COMMON_RULES not in user_prompt

    @pytest.mark.asyncio
    async def test_bull_rebuttal_uses_debate_history(self, advisory_graph):
        """On second round, Bull should reference the Bear's arguments."""
        state = {
            "ticker": "AAPL",
            "ml_prediction": {},
            "past_lessons": {},
            "news_context": "News.",
            "fundamentals_report": "Fundamentals.",
            "technical_report": "Technicals.",
            "debate_history": ["Bull: Initial case.", "Bear: The risks are high."],
            "debate_round_count": 1,
            "final_advisory": "",
        }
        advisory_graph._call_deepseek = AsyncMock(return_value="Rebuttal to bear.")

        await advisory_graph.bull_researcher_node(state)

        # Verify the prompt passed to DeepSeek includes the debate history
        call_text = advisory_graph._call_deepseek.call_args[0][0]
        assert "Bear" in call_text or "Debate History" in call_text


# ── Format Lessons Tests ────────────────────────────────────────────────────

class TestFormatLessons:
    """_format_lessons() formatting."""

    def test_empty_lessons(self, advisory_graph):
        result = advisory_graph._format_lessons({})
        assert result == "No relevant past lessons available."

    def test_empty_all_values(self, advisory_graph):
        result = advisory_graph._format_lessons({
            "ticker_lessons": [],
            "sector_lessons": [],
            "market_lessons": [],
        })
        assert "No relevant past lessons" in result

    def test_ticker_lessons_formatted(self, advisory_graph):
        lessons = {
            "ticker_lessons": [
                {"lesson_learned": "Don't fight the trend", "was_successful": True},
                {"lesson_learned": "Cut losses early", "was_successful": False},
            ],
            "sector_lessons": [],
            "market_lessons": [],
        }
        result = advisory_graph._format_lessons(lessons)
        assert "Ticker-Specific Lessons" in result
        assert "[SUCCESS]" in result
        assert "[FAILURE]" in result
        assert "Don't fight the trend" in result
        assert "Cut losses early" in result

    def test_sector_lessons_include_sector_name(self, advisory_graph):
        lessons = {
            "ticker_lessons": [],
            "sector_lessons": [
                {"lesson_learned": "Tech is cyclical", "was_successful": True, "sector": "Technology"},
            ],
            "market_lessons": [],
        }
        result = advisory_graph._format_lessons(lessons)
        assert "Sector Lessons" in result
        assert "Technology" in result

    def test_market_lessons_formatted(self, advisory_graph):
        lessons = {
            "ticker_lessons": [],
            "sector_lessons": [],
            "market_lessons": [
                {"lesson_learned": "Bear markets kill momentum plays", "was_successful": True},
            ],
        }
        result = advisory_graph._format_lessons(lessons)
        assert "Market-Wide Lessons" in result
        assert "Bear markets" in result


# ── _call_deepseek Tests ────────────────────────────────────────────────────

class TestCallDeepSeek:
    """_call_deepseek internal method."""

    @pytest.mark.asyncio
    async def test_no_client_returns_error_message(self, advisory_graph):
        """If no DeepSeek client, return an error string, don't crash."""
        with patch("pipeline.agents.get_deepseek_client", return_value=None):
            result = await advisory_graph._call_deepseek("test prompt")
        assert "not configured" in result.lower()

    @pytest.mark.asyncio
    async def test_with_streaming_callback(self, advisory_graph):
        """When debate_chunk_callback is set, use streaming."""
        callback = AsyncMock()
        advisory_graph.debate_chunk_callback = callback

        mock_client = MagicMock()
        # Create a mock streaming response (async iterable)
        async def mock_stream():
            chunk = MagicMock()
            delta = MagicMock()
            delta.content = "test "
            chunk.choices = [MagicMock(delta=delta)]
            yield chunk
            chunk2 = MagicMock()
            delta2 = MagicMock()
            delta2.content = "response"
            chunk2.choices = [MagicMock(delta=delta2)]
            yield chunk2

        mock_client.chat.completions.create.return_value = mock_stream()

        with patch("pipeline.agents.get_deepseek_client", return_value=mock_client):
            result = await advisory_graph._call_deepseek("prompt", speaker="bull", round_num=1, system_message="You are bullish.")

        assert "test response" in result
        assert callback.called
