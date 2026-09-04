"""Tests for Embedder — embedding generation through the LLM facade."""

import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from data.models import NewsArticle
from pipeline.embedder import EMBEDDING_DIM


def make_result(*vectors, cost: float = 1.2e-06, prompt_tokens: int = 8):
    """Build what `config.llm.embed` returns."""
    from config.llm import EmbeddingResult

    return EmbeddingResult(
        vectors=list(vectors),
        usage=MagicMock(prompt_tokens=prompt_tokens, cost=cost),
        cost=cost,
    )


def vec(fill: float = 0.1, dim: int = EMBEDDING_DIM) -> list[float]:
    """A correctly-sized vector. Width is load-bearing — see the guard tests."""
    return [fill] * dim


@pytest.fixture
def embedder(monkeypatch):
    """An initialized Embedder wired to a stubbed `embed`."""
    from pipeline.embedder import Embedder

    emb = Embedder(model_name="test/embedding-model", db=MagicMock())
    emb._initialized = True
    emb.embed = AsyncMock()
    monkeypatch.setattr("pipeline.embedder.embed", emb.embed)
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

    def test_default_model_comes_from_settings(self):
        from config.settings import settings
        from pipeline.embedder import Embedder

        assert Embedder().model_name == settings.model_embedding

    def test_custom_model_name(self):
        from pipeline.embedder import Embedder
        assert Embedder(model_name="custom/model").model_name == "custom/model"

    def test_initialized_flag_starts_false(self):
        from pipeline.embedder import Embedder
        assert Embedder()._initialized is False

    @pytest.mark.asyncio
    async def test_initialize_sets_flag(self):
        from pipeline.embedder import Embedder
        with patch("pipeline.embedder.is_llm_configured", return_value=True):
            emb = Embedder(model_name="test/embedding-model")
            await emb.initialize()
            assert emb._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_without_api_key(self):
        from pipeline.embedder import Embedder
        with patch("pipeline.embedder.is_llm_configured", return_value=False):
            emb = Embedder(model_name="test/embedding-model")
            await emb.initialize()
            assert emb._initialized is False

    @pytest.mark.asyncio
    async def test_initialize_without_model_configured(self, monkeypatch):
        """An unset MODEL_EMBEDDING must not look like a working embedder."""
        from pipeline.embedder import Embedder
        monkeypatch.setattr("pipeline.embedder.settings.model_embedding", "")
        with patch("pipeline.embedder.is_llm_configured", return_value=True):
            emb = Embedder()
            await emb.initialize()
            assert emb._initialized is False


class TestGetEmbedding:
    """get_embedding() method."""

    @pytest.mark.asyncio
    async def test_returns_numpy_array(self, embedder):
        embedder.embed.return_value = make_result(vec(0.25))

        result = await embedder.get_embedding("Test text")

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == (EMBEDDING_DIM,)

    @pytest.mark.asyncio
    async def test_returns_none_for_empty_text(self, embedder):
        assert await embedder.get_embedding("") is None

    @pytest.mark.asyncio
    async def test_returns_none_for_whitespace_text(self, embedder):
        assert await embedder.get_embedding("   \n  ") is None

    @pytest.mark.asyncio
    async def test_returns_none_when_not_initialized(self):
        from pipeline.embedder import Embedder
        assert await Embedder().get_embedding("test") is None

    @pytest.mark.asyncio
    async def test_handles_api_failure(self, embedder):
        embedder.embed.side_effect = Exception("API error")
        assert await embedder.get_embedding("test") is None

    @pytest.mark.asyncio
    async def test_logs_real_token_count_and_cost(self, embedder):
        """Embeddings are metered in tokens now, not estimated from characters."""
        embedder.embed.return_value = make_result(vec(), cost=3.4e-06, prompt_tokens=42)

        await embedder.get_embedding("Test text")

        kwargs = embedder.db.log_llm_usage.call_args[1]
        assert kwargs["operation"] == "embed"
        assert kwargs["prompt_tokens"] == 42
        assert kwargs["cost_usd"] == pytest.approx(3.4e-06)


class TestDimensionGuard:
    """
    The guard between a mis-set MODEL_EMBEDDING and a corrupted corpus.

    A wrong-width vector fails nothing downstream — it just quietly poisons
    dedup, semantic search and thesis grounding for every article it touches.
    """

    @pytest.mark.asyncio
    async def test_wrong_dimension_is_discarded(self, embedder):
        embedder.embed.return_value = make_result([0.1, 0.2, 0.3])
        assert await embedder.get_embedding("test") is None

    @pytest.mark.asyncio
    async def test_wrong_dimension_in_batch_is_discarded(self, embedder):
        """One bad vector must not take the whole batch with it."""
        embedder.embed.return_value = make_result(vec(0.1), [0.1, 0.2], vec(0.3))

        results = await embedder.get_embeddings(["a", "b", "c"])

        assert results[0] is not None
        assert results[1] is None
        assert results[2] is not None

    @pytest.mark.asyncio
    async def test_correct_dimension_is_kept(self, embedder):
        embedder.embed.return_value = make_result(vec())
        result = await embedder.get_embedding("test")
        assert result is not None and result.shape == (EMBEDDING_DIM,)


class TestEmbedArticle:
    """embed_article() method."""

    @pytest.mark.asyncio
    async def test_combines_headline_and_summary(self, embedder, sample_article):
        embedder.embed.return_value = make_result(vec())

        result = await embedder.embed_article(sample_article)

        assert isinstance(result, np.ndarray)
        sent = embedder.embed.await_args[0][0][0]
        assert sample_article.headline in sent
        assert sample_article.summary in sent

    @pytest.mark.asyncio
    async def test_returns_none_when_not_initialized(self, sample_article):
        from pipeline.embedder import Embedder
        assert await Embedder().embed_article(sample_article) is None


class TestGetEmbeddings:
    """Batch path — results must stay aligned with the input list."""

    @pytest.mark.asyncio
    async def test_results_align_with_input(self, embedder):
        embedder.embed.return_value = make_result(vec(0.1), vec(0.2), vec(0.3))

        results = await embedder.get_embeddings(["a", "b", "c"])

        assert len(results) == 3
        assert all(r is not None for r in results)
        assert results[1][0] == pytest.approx(0.2)

    @pytest.mark.asyncio
    async def test_empty_texts_are_skipped_not_sent(self, embedder):
        """The API rejects empty strings, and one would fail the whole chunk."""
        embedder.embed.return_value = make_result(vec(0.1), vec(0.3))

        results = await embedder.get_embeddings(["a", "   ", "c"])

        assert embedder.embed.await_args[0][0] == ["a", "c"]
        assert results[0] is not None
        assert results[1] is None
        assert results[2] is not None

    @pytest.mark.asyncio
    async def test_batch_failure_returns_all_none(self, embedder):
        embedder.embed.side_effect = Exception("API error")
        assert await embedder.get_embeddings(["a", "b"]) == [None, None]

    @pytest.mark.asyncio
    async def test_length_mismatch_is_rejected(self, embedder):
        """Fewer vectors than texts means the mapping is unknowable."""
        embedder.embed.return_value = make_result(vec())
        assert await embedder.get_embeddings(["a", "b"]) == [None, None]

    @pytest.mark.asyncio
    async def test_empty_input(self, embedder):
        assert await embedder.get_embeddings([]) == []
