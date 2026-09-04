import pytest
import asyncio
import numpy as np
from unittest.mock import MagicMock, patch, AsyncMock
from pipeline.chat_orchestrator import ChatOrchestrator, ChatState
from tests.conftest import make_llm_response


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

@pytest.mark.asyncio
async def test_router_node_shallow():
    db = MagicMock()
    orchestrator = ChatOrchestrator(db)
    
    state: ChatState = {"query": "What is AAPL?", "context": "", "routing_decision": "", "final_answer": ""}
    
    with patch("pipeline.chat_orchestrator.complete",
               AsyncMock(return_value=make_llm_response('{"decision": "shallow"}'))):
        result = await orchestrator.router_node(state)
        assert result["routing_decision"] == "shallow"

@pytest.mark.asyncio
async def test_router_node_complex():
    db = MagicMock()
    orchestrator = ChatOrchestrator(db)
    
    state: ChatState = {"query": "Deep fundamental analysis of TSLA", "context": "", "routing_decision": "", "final_answer": ""}
    
    with patch("pipeline.chat_orchestrator.complete",
               AsyncMock(return_value=make_llm_response('{"decision": "complex"}'))):
        result = await orchestrator.router_node(state)
        assert result["routing_decision"] == "complex"

@pytest.mark.asyncio
async def test_rag_node_reranking(mock_db):
    mock_embedder = AsyncMock()
    mock_embedder.get_embedding.return_value = np.array([0.9, 0.1])
    mock_embedder._initialized = True
    orchestrator = ChatOrchestrator(mock_db, embedder=mock_embedder)
    
    state: ChatState = {"query": "test query", "context": "", "routing_decision": "", "final_answer": ""}
    
    result = await orchestrator.rag_node(state)
    
    # id2 should be first because it has importance_score 50.0
    assert "Title: H2" in result["context"]
    assert "Title: H1" in result["context"]
    assert result["context"].index("Title: H2") < result["context"].index("Title: H1")
    
@pytest.mark.asyncio
async def test_agent_nodes(mock_db):
    orchestrator = ChatOrchestrator(mock_db)
    
    state: ChatState = {"query": "Test", "context": "Context", "routing_decision": "shallow", "final_answer": ""}
    
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
async def test_end_to_end_graph(mock_db):
    mock_embedder = AsyncMock()
    mock_embedder.get_embedding.return_value = np.array([0.9, 0.1])
    mock_embedder._initialized = True
    orchestrator = ChatOrchestrator(mock_db, embedder=mock_embedder)
    
    async def fake_complete(**kwargs):
        prompt = kwargs.get("prompt") or ""
        if "decision" in prompt:
            return make_llm_response('{"decision": "complex"}')
        return make_llm_response("Final Complex Answer")

    with patch("pipeline.chat_orchestrator.complete", side_effect=fake_complete):
        answer = await orchestrator.run("Test full graph execution")
        assert answer == "Final Complex Answer"
