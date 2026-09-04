"""
Embedding Pipeline Component

Generates embeddings through OpenRouter rather than a local ONNX model, which
keeps RAM free and removes heavy C++ dependencies.

The model is pinned by `settings.model_embedding` and is the one model in this
project that is not freely swappable: every stored vector is compared directly
against new ones, so a model of a different width silently breaks dedup, RAG
and thesis grounding. `_validate` below refuses off-width vectors rather than
storing them.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from config.logging_config import get_logger
from config.llm import embed, is_llm_configured
from config.settings import settings
from data.models import NewsArticle

log = get_logger(__name__)

# Dimensionality of gemini-embedding-001 output. Hoisted out of the scheduler's
# inline np.zeros(3072) so the fallback vector and the dedup shape guard cannot
# drift apart from the model actually in use.
EMBEDDING_DIM = 3072


def _article_text(article: NewsArticle) -> str:
    """The text an article is embedded from. Single source of truth."""
    parts = [article.headline]
    if article.summary:
        parts.append(article.summary)
    return "\n\n".join(parts)


class Embedder:
    """Generates embeddings via OpenRouter."""

    def __init__(self, model_name: Optional[str] = None, db=None):
        self.model_name = model_name or settings.model_embedding
        self._initialized = False
        # Duck-typed rather than typed as Database, which would import the data
        # layer into the pipeline layer. Optional so existing callers that
        # construct a bare embedder keep working — they simply log nothing.
        self.db = db

    def _log_embedding_usage(self, result, count: int) -> None:
        """
        Records an embedding call.

        Unlike the previous provider, OpenRouter meters embeddings in tokens
        and reports the cost outright, so this is a measured figure rather than
        the characters/4 estimate it used to be — hence the operation name is
        now plain `embed`.
        """
        if not self.db:
            return

        usage = getattr(result, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None

        try:
            self.db.log_llm_usage(
                model_name=self.model_name,
                operation="embed",
                prompt_tokens=int(prompt_tokens or 0),
                candidate_tokens=0,
                cost_usd=result.cost,
            )
        except Exception as e:
            log.warning("embedder.usage_log_failed", error=str(e), count=count)

    def _validate(self, values) -> Optional[np.ndarray]:
        """
        The guard between a mis-set MODEL_EMBEDDING and a corrupted corpus.

        Stored vectors are compared by cosine distance against whatever is
        stored next to them, so a batch at the wrong width does not fail
        loudly — it quietly poisons dedup and semantic search for every article
        it touches. Dropping the vector costs one article's embedding; storing
        it costs the corpus.
        """
        if not values:
            return None

        if len(values) != EMBEDDING_DIM:
            log.error(
                "embedder.dimension_mismatch",
                expected=EMBEDDING_DIM,
                received=len(values),
                model=self.model_name,
                action="vector discarded",
                hint="MODEL_EMBEDDING must be a 3072-dim model, or the corpus needs re-embedding",
            )
            return None

        return np.array(values, dtype=np.float32)

    async def initialize(self):
        """Initializes the embedder."""
        if not is_llm_configured():
            log.warning("embedder.initialization_failed", reason="No API key found")
            return

        if not self.model_name:
            log.warning("embedder.initialization_failed", reason="MODEL_EMBEDDING is not set")
            return

        self._initialized = True
        log.info("embedder.initialized", model=self.model_name)

    async def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Returns the embedding for a single string as a numpy array."""
        if not self._initialized:
            log.warning("embedder.not_initialized")
            return None

        if not text or not text.strip():
            return None

        try:
            result = await embed([text], model=self.model_name)
        except Exception as e:
            log.error("embedder.generation_failed", error=str(e))
            return None

        self._log_embedding_usage(result, 1)
        return self._validate(result.vectors[0] if result.vectors else None)

    async def embed_article(self, article: NewsArticle) -> Optional[np.ndarray]:
        """Generates embedding for a single article and returns it as a numpy array."""
        if not self._initialized:
            return None

        return await self.get_embedding(_article_text(article))

    async def get_embeddings(
        self, texts: list[str], batch_size: Optional[int] = None
    ) -> list[Optional[np.ndarray]]:
        """
        Embeds many texts per request. Returns a list aligned 1:1 with `texts`,
        with None wherever a text was empty or its chunk failed.

        The endpoint accepts a list, so this collapses one HTTP round-trip per
        article into one per batch. It does not reduce token spend — embeddings
        bill per input token either way — but it removes the per-article retry
        amplification that sat on top of those round-trips.
        """
        if not self._initialized:
            log.warning("embedder.not_initialized")
            return [None] * len(texts)

        if not texts:
            return []

        size = batch_size or settings.embed_batch_size
        results: list[Optional[np.ndarray]] = [None] * len(texts)

        # Empty strings are dropped rather than sent — the API rejects them, and
        # one bad item would otherwise fail the whole chunk.
        indexed = [(i, t) for i, t in enumerate(texts) if t and t.strip()]

        for start in range(0, len(indexed), size):
            chunk = indexed[start:start + size]

            try:
                result = await embed([t for _, t in chunk], model=self.model_name)
            except Exception as e:
                # Chunk-local failure: the remaining chunks still go out.
                log.error("embedder.batch_failed", error=str(e), count=len(chunk))
                continue

            self._log_embedding_usage(result, len(chunk))

            if len(result.vectors) != len(chunk):
                log.error(
                    "embedder.batch_length_mismatch",
                    expected=len(chunk), received=len(result.vectors),
                )
                continue

            # `embed` places results by the response's own index, so this zip
            # is a mapping back onto the caller's original positions.
            for (idx, _), values in zip(chunk, result.vectors):
                results[idx] = self._validate(values)

        return results

    async def embed_articles(self, articles: list[NewsArticle]) -> list[Optional[np.ndarray]]:
        """Batch counterpart to embed_article. Result is aligned 1:1 with input."""
        if not self._initialized:
            return [None] * len(articles)

        return await self.get_embeddings([_article_text(a) for a in articles])
