import pytest
import asyncio
import numpy as np
from unittest.mock import MagicMock, patch, AsyncMock
from pipeline.chat_orchestrator import ChatOrchestrator, ChatState, build_chat_prompt
from pipeline.grounded_answer import GradeVerdict, HONESTY_SENTENCE
from tests.conftest import make_llm_response, make_stream


@pytest.fixture(autouse=True)
def _models_configured(monkeypatch):
    """Every test here assumes the chat lane has models assigned."""
    monkeypatch.setattr("pipeline.chat_orchestrator.is_llm_configured", lambda: True)
    monkeypatch.setattr("pipeline.chat_orchestrator.settings.model_router", "test/router")
    monkeypatch.setattr("pipeline.chat_orchestrator.settings.model_chat_shallow", "test/shallow")
    monkeypatch.setattr("pipeline.chat_orchestrator.settings.model_chat_complex", "test/complex")

@pytest.fixture
def mock_db():
    db = MagicMock()
    db.has_sqlite_vec = False

    # Mock embedding return
    db.get_all_embeddings.return_value = [
        ("id1", np.array([0.9, 0.1])),
        ("id2", np.array([0.8, 0.2])),
        ("id3", np.array([0.1, 0.9]))
    ]

    # Mock articles return for RAG
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = [
        {"id": "id1", "headline": "H1", "summary": "S1", "importance_score": 10.0},
        {"id": "id2", "headline": "H2", "summary": "S2", "importance_score": 50.0}, # Higher importance
        {"id": "id3", "headline": "H3", "summary": "S3", "importance_score": 5.0}
    ]
    mock_conn.execute.return_value = mock_cursor
    db.connection.return_value.__enter__.return_value = mock_conn
    return db


def _orchestrator(db):
    """An orchestrator whose embedder is stubbed, so RAG never needs a model."""
    embedder = AsyncMock()
    embedder.get_embedding.return_value = np.array([0.9, 0.1])
    embedder._initialized = True
    return ChatOrchestrator(db, embedder=embedder)


def _state(**overrides) -> ChatState:
    state: ChatState = {
        "query": "Why is NVDA down?", "context": "", "routing_decision": "shallow",
        "final_answer": "", "top_articles": [], "grade": None,
        "web_sources": [], "grounded_by": "none",
    }
    state.update(overrides)
    return state


def _verdict(sufficient: bool) -> GradeVerdict:
    return GradeVerdict(
        sufficient=sufficient,
        specificity="specific" if sufficient else "generic",
        reason="fixture",
    )


@pytest.mark.asyncio
async def test_router_node_shallow():
    db = MagicMock()
    orchestrator = ChatOrchestrator(db)

    state = _state(query="What is AAPL?", routing_decision="")

    with patch("pipeline.chat_orchestrator.complete",
               AsyncMock(return_value=make_llm_response('{"decision": "shallow"}'))):
        result = await orchestrator.router_node(state)
        assert result["routing_decision"] == "shallow"

@pytest.mark.asyncio
async def test_router_node_complex():
    db = MagicMock()
    orchestrator = ChatOrchestrator(db)

    state = _state(query="Deep fundamental analysis of TSLA", routing_decision="")

    with patch("pipeline.chat_orchestrator.complete",
               AsyncMock(return_value=make_llm_response('{"decision": "complex"}'))):
        result = await orchestrator.router_node(state)
        assert result["routing_decision"] == "complex"

@pytest.mark.asyncio
async def test_rag_node_reranking(mock_db):
    orchestrator = _orchestrator(mock_db)

    result = await orchestrator.rag_node(_state(query="test query"))

    # id2 should be first because it has importance_score 50.0
    assert "Title: H2" in result["context"]
    assert "Title: H1" in result["context"]
    assert result["context"].index("Title: H2") < result["context"].index("Title: H1")
    # Context existing is provisional grounding; the grader decides if it holds.
    assert result["grounded_by"] == "db"


@pytest.mark.asyncio
async def test_rag_node_reports_no_grounding_when_nothing_matches(mock_db):
    orchestrator = _orchestrator(mock_db)
    mock_db.get_all_embeddings.return_value = []
    result = await orchestrator.rag_node(_state())
    assert result["context"] == ""
    assert result["grounded_by"] == "none"


@pytest.mark.asyncio
async def test_agent_nodes(mock_db):
    orchestrator = ChatOrchestrator(mock_db)

    state = _state(query="Test", context="Context", grounded_by="db")

    mock_complete = AsyncMock(return_value=make_llm_response("Shallow Answer"))
    with patch("pipeline.chat_orchestrator.complete", mock_complete):
        # Test shallow agent
        result = await orchestrator.shallow_agent_node(state)
        assert result["final_answer"] == "Shallow Answer"

        # The shallow lane must not ask for reasoning — that is the whole
        # point of routing to it.
        assert mock_complete.await_args.kwargs["reasoning"] is None
        assert mock_complete.await_args.kwargs["model"] == "test/shallow"

        # Test complex agent
        mock_complete.return_value = make_llm_response("Complex Answer")
        result = await orchestrator.complex_agent_node(state)
        assert result["final_answer"] == "Complex Answer"

        assert mock_complete.await_args.kwargs["reasoning"] == "medium"
        assert mock_complete.await_args.kwargs["model"] == "test/complex"


@pytest.mark.asyncio
async def test_complex_agent_no_longer_searches_inline(mock_db):
    """The search moved to its own node; leaving a copy here double-searches."""
    orchestrator = ChatOrchestrator(mock_db)
    enrich = AsyncMock(return_value=("enriched", [{"url": "u"}]))
    with patch("pipeline.chat_orchestrator.enrich_chat_context", enrich), \
         patch("pipeline.chat_orchestrator.complete",
               AsyncMock(return_value=make_llm_response("A"))):
        await orchestrator.complex_agent_node(_state(context="ctx"))
    assert enrich.await_count == 0


# ── Grading ─────────────────────────────────────────────────────────────────


class TestGradeNode:

    @pytest.mark.asyncio
    async def test_complex_skips_the_grader_entirely(self, mock_db):
        """That lane always searches, so a verdict nothing acts on is wasted."""
        orchestrator = ChatOrchestrator(mock_db)
        grade_context = AsyncMock()
        with patch("pipeline.chat_orchestrator.grade_context", grade_context):
            patch_out = await orchestrator.grade_node(
                _state(routing_decision="complex", context="lots of context")
            )
        assert grade_context.await_count == 0
        assert patch_out["grade"].sufficient is False
        assert patch_out["grade"].reason == "complex always searches"

    @pytest.mark.asyncio
    async def test_insufficient_grade_demotes_the_grounding(self, mock_db):
        orchestrator = ChatOrchestrator(mock_db)
        with patch("pipeline.chat_orchestrator.grade_context",
                   AsyncMock(return_value=_verdict(False))):
            patch_out = await orchestrator.grade_node(
                _state(context="weakly related", grounded_by="db")
            )
        assert patch_out["grounded_by"] == "none"

    @pytest.mark.asyncio
    async def test_sufficient_grade_leaves_the_grounding_alone(self, mock_db):
        orchestrator = ChatOrchestrator(mock_db)
        with patch("pipeline.chat_orchestrator.grade_context",
                   AsyncMock(return_value=_verdict(True))):
            patch_out = await orchestrator.grade_node(
                _state(context="dated catalyst", grounded_by="db")
            )
        assert "grounded_by" not in patch_out


class TestDecideAfterGrade:
    """The truth table the whole routing change rests on."""

    def setup_method(self):
        self.orchestrator = ChatOrchestrator(MagicMock())

    def test_shallow_and_sufficient_answers_directly(self):
        state = _state(routing_decision="shallow", grade=_verdict(True))
        assert self.orchestrator.decide_after_grade(state) == "answer"

    def test_shallow_and_insufficient_searches(self):
        state = _state(routing_decision="shallow", grade=_verdict(False))
        assert self.orchestrator.decide_after_grade(state) == "web_search"

    def test_complex_always_searches_even_when_sufficient(self):
        state = _state(routing_decision="complex", grade=_verdict(True))
        assert self.orchestrator.decide_after_grade(state) == "web_search"

    def test_complex_and_insufficient_searches(self):
        state = _state(routing_decision="complex", grade=_verdict(False))
        assert self.orchestrator.decide_after_grade(state) == "web_search"

    def test_missing_grade_searches(self):
        """Absent evidence about the evidence still means go and look."""
        state = _state(routing_decision="shallow", grade=None)
        assert self.orchestrator.decide_after_grade(state) == "web_search"

    def test_route_after_grade_maps_answer_to_the_routed_agent(self):
        state = _state(routing_decision="complex", grade=_verdict(True))
        assert self.orchestrator.route_after_grade(state) == "web_search"
        state = _state(routing_decision="shallow", grade=_verdict(True))
        assert self.orchestrator.route_after_grade(state) == "shallow"


class TestGradeVerdictParsing:

    def test_fixture_parses(self):
        verdict = GradeVerdict.model_validate_json(
            '{"sufficient": true, "specificity": "specific", "reason": "ok"}'
        )
        assert verdict.sufficient is True

    @pytest.mark.asyncio
    async def test_empty_context_is_insufficient_without_an_llm_call(self):
        from pipeline.grounded_answer import grade_context
        complete = AsyncMock()
        with patch("pipeline.grounded_answer.complete", complete):
            verdict = await grade_context(MagicMock(), "why?", "   ")
        assert complete.await_count == 0
        assert verdict.sufficient is False
        assert verdict.specificity == "none"

    @pytest.mark.asyncio
    async def test_parse_failure_fails_toward_searching(self, monkeypatch):
        from pipeline import grounded_answer
        monkeypatch.setattr(grounded_answer.settings, "model_grader", "test/grader")
        monkeypatch.setattr(grounded_answer, "is_llm_configured", lambda: True)
        with patch("pipeline.grounded_answer.complete",
                   AsyncMock(return_value=make_llm_response("not json at all"))):
            verdict = await grounded_answer.grade_context(MagicMock(), "why?", "ctx")
        assert verdict.sufficient is False
        assert "insufficient" in verdict.reason.lower()

    @pytest.mark.asyncio
    async def test_unset_model_fails_toward_searching(self, monkeypatch):
        from pipeline import grounded_answer
        monkeypatch.setattr(grounded_answer.settings, "model_grader", "")
        monkeypatch.setattr(grounded_answer.settings, "model_router", "")
        complete = AsyncMock()
        with patch("pipeline.grounded_answer.complete", complete):
            verdict = await grounded_answer.grade_context(MagicMock(), "why?", "ctx")
        assert complete.await_count == 0
        assert verdict.sufficient is False

    @pytest.mark.asyncio
    async def test_grader_falls_back_to_the_router_model(self, monkeypatch):
        from pipeline import grounded_answer
        monkeypatch.setattr(grounded_answer.settings, "model_grader", "")
        monkeypatch.setattr(grounded_answer.settings, "model_router", "test/router")
        monkeypatch.setattr(grounded_answer, "is_llm_configured", lambda: True)
        complete = AsyncMock(return_value=make_llm_response(
            '{"sufficient": true, "specificity": "specific", "reason": "r"}'
        ))
        with patch("pipeline.grounded_answer.complete", complete):
            verdict = await grounded_answer.grade_context(MagicMock(), "why?", "ctx")
        assert complete.await_args.kwargs["model"] == "test/router"
        assert verdict.sufficient is True


# ── Web search routing ──────────────────────────────────────────────────────


class TestWebSearchRouting:

    @pytest.mark.asyncio
    async def test_shallow_and_insufficient_calls_enrich_chat_context(self, mock_db):
        orchestrator = _orchestrator(mock_db)
        enrich = AsyncMock(return_value=("db ctx + web", [{"url": "https://x"}]))

        with patch("pipeline.chat_orchestrator.enrich_chat_context", enrich), \
             patch("pipeline.chat_orchestrator.complete",
                   AsyncMock(return_value=make_llm_response('{"decision": "shallow"}'))), \
             patch("pipeline.chat_orchestrator.grade_context",
                   AsyncMock(return_value=_verdict(False))), \
             patch("pipeline.chat_orchestrator.stream_complete",
                   MagicMock(return_value=make_stream(["ok"]))):
            events = [e async for e in orchestrator.iter_events("Why is NVDA down?")]

        assert enrich.await_count == 1
        steps = [d["step"] for e, d in events if e == "step"]
        assert "web_search" in steps

    @pytest.mark.asyncio
    async def test_shallow_and_sufficient_does_not_search(self, mock_db):
        orchestrator = _orchestrator(mock_db)
        enrich = AsyncMock(return_value=("unused", []))

        with patch("pipeline.chat_orchestrator.enrich_chat_context", enrich), \
             patch("pipeline.chat_orchestrator.complete",
                   AsyncMock(return_value=make_llm_response('{"decision": "shallow"}'))), \
             patch("pipeline.chat_orchestrator.grade_context",
                   AsyncMock(return_value=_verdict(True))), \
             patch("pipeline.chat_orchestrator.stream_complete",
                   MagicMock(return_value=make_stream(["ok"]))):
            events = [e async for e in orchestrator.iter_events("Summarize TSLA news")]

        assert enrich.await_count == 0
        steps = [d["step"] for e, d in events if e == "step"]
        assert "web_search" not in steps


# ── iter_events ─────────────────────────────────────────────────────────────


class TestIterEvents:

    async def _run(self, mock_db, *, decision="shallow", sufficient=True,
                   web_sources=None, chunks=("Hello ", "world")):
        orchestrator = _orchestrator(mock_db)
        enrich = AsyncMock(
            return_value=("db ctx + web", list(web_sources or []))
        )
        with patch("pipeline.chat_orchestrator.enrich_chat_context", enrich), \
             patch("pipeline.chat_orchestrator.complete",
                   AsyncMock(return_value=make_llm_response(
                       '{"decision": "%s"}' % decision))), \
             patch("pipeline.chat_orchestrator.grade_context",
                   AsyncMock(return_value=_verdict(sufficient))), \
             patch("pipeline.chat_orchestrator.stream_complete",
                   MagicMock(return_value=make_stream(list(chunks)))) as stream:
            events = [e async for e in orchestrator.iter_events("Why is NVDA down?")]
        return events, stream

    @pytest.mark.asyncio
    async def test_yields_grading_then_sources(self, mock_db):
        events, _ = await self._run(mock_db)
        names = [e for e, _ in events]
        grading_at = next(
            i for i, (e, d) in enumerate(events)
            if e == "step" and d["step"] == "grading"
        )
        sources_at = names.index("sources")
        assert grading_at < sources_at

    @pytest.mark.asyncio
    async def test_grading_step_carries_the_verdict(self, mock_db):
        events, _ = await self._run(mock_db, sufficient=True)
        grading = next(d for e, d in events if e == "step" and d["step"] == "grading")
        assert grading["sufficient"] is True
        assert grading["specificity"] == "specific"
        assert grading["reason"] == "fixture"

    @pytest.mark.asyncio
    async def test_retrieval_step_carries_the_citation_count(self, mock_db):
        """The REST handler offsets the research counters by this."""
        events, _ = await self._run(mock_db)
        retrieval = next(
            d for e, d in events if e == "step" and d["step"] == "retrieval"
        )
        assert retrieval["count"] == 3

    @pytest.mark.asyncio
    async def test_classification_step_comes_first(self, mock_db):
        events, _ = await self._run(mock_db)
        assert events[0][0] == "step"
        assert events[0][1]["step"] == "classification"

    @pytest.mark.asyncio
    async def test_tokens_then_done(self, mock_db):
        events, _ = await self._run(mock_db, chunks=("Hel", "lo"))
        tokens = [d["text"] for e, d in events if e == "token"]
        assert tokens == ["Hel", "lo"]
        assert events[-1][0] == "done"
        assert events[-1][1]["answer"] == "Hello"

    @pytest.mark.asyncio
    async def test_sources_merge_db_and_web(self, mock_db):
        events, _ = await self._run(
            mock_db, sufficient=False,
            web_sources=[{"title": "W", "url": "https://w", "source_type": "web"}],
        )
        articles = next(d for e, d in events if e == "sources")["articles"]
        assert len(articles) == 4          # 3 from RAG + 1 from the web
        assert articles[-1]["url"] == "https://w"

    @pytest.mark.asyncio
    async def test_honesty_prefix_when_nothing_was_found(self, mock_db):
        """grounded_by == "none" has to reach the prompt, not just the log."""
        events, stream = await self._run(mock_db, sufficient=False, web_sources=[])
        prompt = stream.call_args.kwargs["prompt"]
        assert HONESTY_SENTENCE in prompt
        assert events[-1][1]["grounded_by"] == "none"

    @pytest.mark.asyncio
    async def test_no_honesty_prefix_once_the_web_lands_something(self, mock_db):
        events, stream = await self._run(
            mock_db, sufficient=False,
            web_sources=[{"title": "W", "url": "https://w"}],
        )
        prompt = stream.call_args.kwargs["prompt"]
        assert HONESTY_SENTENCE not in prompt
        assert events[-1][1]["grounded_by"] == "web"

    @pytest.mark.asyncio
    async def test_no_honesty_prefix_when_the_grade_holds(self, mock_db):
        events, stream = await self._run(mock_db, sufficient=True)
        assert HONESTY_SENTENCE not in stream.call_args.kwargs["prompt"]
        assert events[-1][1]["grounded_by"] == "db"

    @pytest.mark.asyncio
    async def test_shallow_streams_without_reasoning(self, mock_db):
        _, stream = await self._run(mock_db, decision="shallow")
        assert stream.call_args.kwargs["reasoning"] is None
        assert stream.call_args.kwargs["model"] == "test/shallow"

    @pytest.mark.asyncio
    async def test_complex_streams_with_medium_reasoning(self, mock_db):
        _, stream = await self._run(mock_db, decision="complex")
        assert stream.call_args.kwargs["reasoning"] == "medium"
        assert stream.call_args.kwargs["model"] == "test/complex"

    @pytest.mark.asyncio
    async def test_unconfigured_model_yields_an_error_string(self, mock_db, monkeypatch):
        monkeypatch.setattr(
            "pipeline.chat_orchestrator.settings.model_chat_shallow", ""
        )
        orchestrator = _orchestrator(mock_db)
        with patch("pipeline.chat_orchestrator.complete",
                   AsyncMock(return_value=make_llm_response('{"decision": "shallow"}'))), \
             patch("pipeline.chat_orchestrator.grade_context",
                   AsyncMock(return_value=_verdict(True))):
            events = [e async for e in orchestrator.iter_events("hi")]
        assert events[-1][0] == "error"
        assert isinstance(events[-1][1], str)

    @pytest.mark.asyncio
    async def test_stream_failure_yields_an_error_not_an_exception(self, mock_db):
        orchestrator = _orchestrator(mock_db)

        def boom(**kwargs):
            raise RuntimeError("upstream 502")

        with patch("pipeline.chat_orchestrator.complete",
                   AsyncMock(return_value=make_llm_response('{"decision": "shallow"}'))), \
             patch("pipeline.chat_orchestrator.grade_context",
                   AsyncMock(return_value=_verdict(True))), \
             patch("pipeline.chat_orchestrator.stream_complete", boom):
            events = [e async for e in orchestrator.iter_events("hi")]
        assert events[-1][0] == "error"
        assert "502" in events[-1][1]

    @pytest.mark.asyncio
    async def test_research_callback_is_passed_through_to_the_search(self, mock_db):
        orchestrator = _orchestrator(mock_db)
        seen = []

        async def cb(event_type, data):
            seen.append(event_type)

        async def fake_enrich(*, query, db_context, max_results, research_callback=None):
            await research_callback("research_start", {"query": query})
            return db_context, []

        with patch("pipeline.chat_orchestrator.enrich_chat_context", fake_enrich), \
             patch("pipeline.chat_orchestrator.complete",
                   AsyncMock(return_value=make_llm_response('{"decision": "complex"}'))), \
             patch("pipeline.chat_orchestrator.stream_complete",
                   MagicMock(return_value=make_stream(["x"]))):
            [e async for e in orchestrator.iter_events("q", research_callback=cb)]

        assert seen == ["research_start"]


# ── Graph structure and run() ───────────────────────────────────────────────


def test_graph_compiles_with_the_new_nodes(mock_db):
    """A missing node or an unmapped conditional edge raises at compile time."""
    graph = ChatOrchestrator(mock_db).build_graph()
    nodes = set(graph.get_graph().nodes)
    assert {"router", "rag", "grade", "web_search", "shallow", "complex"} <= nodes


@pytest.mark.asyncio
async def test_end_to_end_run_joins_the_tokens(mock_db):
    orchestrator = _orchestrator(mock_db)

    async def fake_complete(**kwargs):
        return make_llm_response('{"decision": "complex"}')

    with patch("pipeline.chat_orchestrator.complete", side_effect=fake_complete), \
         patch("pipeline.chat_orchestrator.enrich_chat_context",
               AsyncMock(return_value=("ctx", []))), \
         patch("pipeline.chat_orchestrator.stream_complete",
               MagicMock(return_value=make_stream(["Final ", "Complex ", "Answer"]))):
        answer = await orchestrator.run("Test full graph execution")

    assert answer == "Final Complex Answer"


@pytest.mark.asyncio
async def test_run_surfaces_the_error_when_nothing_streamed(mock_db, monkeypatch):
    monkeypatch.setattr("pipeline.chat_orchestrator.settings.model_chat_shallow", "")
    orchestrator = _orchestrator(mock_db)
    with patch("pipeline.chat_orchestrator.complete",
               AsyncMock(return_value=make_llm_response('{"decision": "shallow"}'))), \
         patch("pipeline.chat_orchestrator.grade_context",
               AsyncMock(return_value=_verdict(True))):
        answer = await orchestrator.run("hi")
    assert "not configured" in answer


def test_build_chat_prompt_is_unchanged_without_the_flag():
    """Existing callers that never pass honesty_required must be unaffected."""
    assert "HONESTY REQUIREMENT" not in build_chat_prompt("q", "ctx")
