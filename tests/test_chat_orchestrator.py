import pytest
import asyncio
import numpy as np
from unittest.mock import MagicMock, patch, AsyncMock
from pipeline.chat_orchestrator import ChatOrchestrator, ChatState
from google.genai import types

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
    
    with patch("pipeline.chat_orchestrator.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = '{"decision": "shallow"}'
        mock_resp.usage_metadata = None
        mock_client.models.generate_content.return_value = mock_resp
        mock_get_client.return_value = mock_client
        
        result = await orchestrator.router_node(state)
        assert result["routing_decision"] == "shallow"

@pytest.mark.asyncio
async def test_router_node_complex():
    db = MagicMock()
    orchestrator = ChatOrchestrator(db)
    
    state: ChatState = {"query": "Deep fundamental analysis of TSLA", "context": "", "routing_decision": "", "final_answer": ""}
    
    with patch("pipeline.chat_orchestrator.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = '{"decision": "complex"}'
        mock_resp.usage_metadata = None
        mock_client.models.generate_content.return_value = mock_resp
        mock_get_client.return_value = mock_client
        
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
    
    with patch("pipeline.chat_orchestrator.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = 'Shallow Answer'
        mock_resp.usage_metadata = None
        mock_client.models.generate_content.return_value = mock_resp
        mock_get_client.return_value = mock_client
        
        # Test shallow agent
        result = await orchestrator.shallow_agent_node(state)
        assert result["final_answer"] == "Shallow Answer"
        
        # Ensure thinking_config is NOT passed for shallow
        call_kwargs = mock_client.models.generate_content.call_args[1]
        assert 'thinking_config' not in call_kwargs['config']
        
        # Test complex agent
        mock_resp.text = 'Complex Answer'
        result = await orchestrator.complex_agent_node(state)
        assert result["final_answer"] == "Complex Answer"
        
        # Ensure thinking_config IS passed for complex
        call_kwargs = mock_client.models.generate_content.call_args[1]
        assert 'thinking_config' in call_kwargs['config']
        assert call_kwargs['config']['thinking_config'].thinking_level == types.ThinkingLevel.MEDIUM

@pytest.mark.asyncio
async def test_end_to_end_graph(mock_db):
    mock_embedder = AsyncMock()
    mock_embedder.get_embedding.return_value = np.array([0.9, 0.1])
    mock_embedder._initialized = True
    orchestrator = ChatOrchestrator(mock_db, embedder=mock_embedder)
    
    with patch("pipeline.chat_orchestrator.get_client") as mock_get_client:
        mock_client = MagicMock()
        
        def mock_generate_content(*args, **kwargs):
            mock_resp = MagicMock()
            mock_resp.usage_metadata = None
            prompt_content = kwargs.get('contents', '')
            if isinstance(prompt_content, str) and 'decision' in prompt_content:
                mock_resp.text = '{"decision": "complex"}'
            else:
                mock_resp.text = "Final Complex Answer"
            return mock_resp
            
        mock_client.models.generate_content.side_effect = mock_generate_content
        mock_get_client.return_value = mock_client
        
        answer = await orchestrator.run("Test full graph execution")
        assert answer == "Final Complex Answer"
