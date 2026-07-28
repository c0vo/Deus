"""Tests for GeminiEmbedder — embedding generation."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from data.models import NewsArticle


@pytest.fixture
def embedder():
    """Create GeminiEmbedder without initialization (no API calls)."""
    from pipeline.embedder import GeminiEmbedder
    emb = GeminiEmbedder(model_name="test-embedding-model")
    emb._initialized = True
    emb.client = MagicMock()
    return emb


@pytest.fixture
def sample_article():
    return NewsArticle(
        id="test_001",
        headline="Apple beats earnings",
        summary="Apple reported strong Q3 results with revenue up 8%.",
        source_name="reuters",
        source_type="rss",
        url="https://example.com/aaple",
        published_at=datetime.now(timezone.utc),
    )


class TestInitialization:
    """Embedder init and setup."""

    def test_default_model_name(self):
        from pipeline.embedder import GeminiEmbedder
        emb = GeminiEmbedder()
        assert emb.model_name == "gemini-embedding-001"

    def test_custom_model_name(self):
        from pipeline.embedder import GeminiEmbedder
        emb = GeminiEmbedder(model_name="custom-model")
        assert emb.model_name == "custom-model"

    def test_initialized_flag_starts_false(self):
        from pipeline.embedder import GeminiEmbedder
        emb = GeminiEmbedder()
        assert emb._initialized is False

    @pytest.mark.asyncio
    async def test_initialize_sets_client(self):
        """initialize() should set client and _initialized flag."""
        from pipeline.embedder import GeminiEmbedder
        with patch("pipeline.embedder.get_client") as mock_get:
            mock_get.return_value = MagicMock()
            emb = GeminiEmbedder()
            await emb.initialize()
            assert emb._initialized is True
            assert emb.client is not None

    @pytest.mark.asyncio
    async def test_initialize_with_no_client(self):
        """initialize() should handle missing client gracefully."""
        from pipeline.embedder import GeminiEmbedder
        with patch("pipeline.embedder.get_client") as mock_get:
            mock_get.return_value = None
            emb = GeminiEmbedder()
            await emb.initialize()
            assert emb._initialized is False


class TestGetEmbedding:
    """get_embedding() method."""

    @pytest.mark.asyncio
    async def test_returns_numpy_array(self, embedder):
        """Valid embedding should return a numpy float32 array."""
        mock_values = [0.1, 0.2, 0.3, 0.4]
        mock_response = MagicMock()
        mock_embedding = MagicMock()
        mock_embedding.values = mock_values
        mock_response.embeddings = [mock_embedding]
        embedder.client.models.embed_content.return_value = mock_response

        result = await embedder.get_embedding("Test text")

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert np.array_equal(result, np.array(mock_values, dtype=np.float32))

    @pytest.mark.asyncio
    async def test_returns_none_for_empty_text(self, embedder):
        result = await embedder.get_embedding("")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_for_whitespace_text(self, embedder):
        result = await embedder.get_embedding("   \n  ")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_not_initialized(self):
        from pipeline.embedder import GeminiEmbedder
        emb = GeminiEmbedder()
        result = await emb.get_embedding("test")
        assert result is None

    @pytest.mark.asyncio
    async def test_handles_api_failure(self, embedder):
        """API exception should return None without crashing."""
        embedder.client.models.embed_content.side_effect = Exception("API error")
        result = await embedder.get_embedding("test")
        assert result is None

    @pytest.mark.asyncio
    async def test_deterministic_output(self, embedder):
        """Same text should produce same embedding with the same mock."""
        mock_values = [0.5, 0.5, 0.5]
        mock_response = MagicMock()
        mock_embedding = MagicMock()
        mock_embedding.values = mock_values
        mock_response.embeddings = [mock_embedding]
        embedder.client.models.embed_content.return_value = mock_response

        r1 = await embedder.get_embedding("same text")
        r2 = await embedder.get_embedding("same text")
        assert np.array_equal(r1, r2)


class TestEmbedArticle:
    """embed_article() method."""

    @pytest.mark.asyncio
    async def test_embed_article_combines_headline_and_summary(self, embedder, sample_article):
        """embed_article should concatenate headline + summary before embedding."""
        mock_values = [0.1, 0.2, 0.3]
        mock_response = MagicMock()
        mock_embedding = MagicMock()
        mock_embedding.values = mock_values
        mock_response.embeddings = [mock_embedding]
        embedder.client.models.embed_content.return_value = mock_response

        result = await embedder.embed_article(sample_article)

        assert isinstance(result, np.ndarray)
        call_text = embedder.client.models.embed_content.call_args[1]["contents"]
        assert sample_article.headline in call_text
        assert sample_article.summary in call_text

    @pytest.mark.asyncio
    async def test_embed_article_returns_none_when_not_initialized(self, sample_article):
        from pipeline.embedder import GeminiEmbedder
        emb = GeminiEmbedder()
        result = await emb.embed_article(sample_article)
        assert result is None
