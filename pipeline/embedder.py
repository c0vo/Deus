"""
Gemini Embedding Pipeline Component

Uses Gemini's API to generate embeddings instead of local ONNX models.
This saves local RAM and removes heavy C++ dependencies.
"""

from __future__ import annotations

import asyncio
from typing import Optional

import numpy as np

from config.logging_config import get_logger
from config.llm import get_client
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


class GeminiEmbedder:
    """Generates embeddings using Gemini API."""

    def __init__(self, model_name: str = "gemini-embedding-001", db=None):
        self.model_name = model_name
        self.client = None
        self._initialized = False
        # Duck-typed rather than typed as Database, which would import the data
        # layer into the pipeline layer. Optional so existing callers that
        # construct a bare embedder keep working — they simply log nothing.
        self.db = db

    def _log_embedding_usage(self, response, count: int) -> None:
        """
        Records an embedding call. Embeddings were previously invisible in
        llm_usage_log entirely, despite running on every ingested article.

        The API reports billable CHARACTERS, not tokens, so the token figure
        here is a ~4-chars-per-token estimate and the operation is named to say
        so. It is a volume signal, not a metered count.
        """
        if not self.db:
            return

        metadata = getattr(response, "metadata", None)
        characters = getattr(metadata, "billable_character_count", None) if metadata else None

        try:
            self.db.log_llm_usage(
                model_name=self.model_name,
                operation="embed_estimated",
                prompt_tokens=int(characters / 4) if characters else 0,
                candidate_tokens=0,
            )
        except Exception as e:
            log.warning("embedder.usage_log_failed", error=str(e), count=count)

    async def initialize(self):
        """Initializes the embedder."""
        self.client = get_client()
        if not self.client:
            log.warning("embedder.initialization_failed", reason="No API key found")
            return
            
        self._initialized = True
        log.info("embedder.initialized", model=self.model_name)

    async def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Returns the embedding for a single string as a numpy array."""
        if not self._initialized or not self.client:
            log.warning("embedder.not_initialized")
            return None

        if not text or not text.strip():
            return None
            
        try:
            # Use the newer google.genai SDK
            # Since get_embedding might block the event loop if the SDK is sync, 
            # we should run it in the executor.
            loop = asyncio.get_running_loop()
            
            def call_api():
                return self.client.models.embed_content(
                    model=self.model_name,
                    contents=text
                )
                
            response = await loop.run_in_executor(None, call_api)
            
            if response.embeddings and len(response.embeddings) > 0:
                values = response.embeddings[0].values
                return np.array(values, dtype=np.float32)
            return None
            
        except Exception as e:
            log.error("embedder.generation_failed", error=str(e))
            return None

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

        embed_content accepts a list for `contents`, so the previous one-string-
        per-request pattern was paying a full HTTP round-trip per article. This
        does not reduce token spend (embeddings bill per input token either way)
        but it collapses ~50 requests per batch into one and removes the
        per-article retry amplification that sat on top of them.
        """
        if not self._initialized or not self.client:
            log.warning("embedder.not_initialized")
            return [None] * len(texts)

        if not texts:
            return []

        size = batch_size or settings.embed_batch_size
        loop = asyncio.get_running_loop()
        results: list[Optional[np.ndarray]] = [None] * len(texts)

        # Empty strings are dropped rather than sent — the API rejects them, and
        # one bad item would otherwise fail the whole chunk.
        indexed = [(i, t) for i, t in enumerate(texts) if t and t.strip()]

        for start in range(0, len(indexed), size):
            chunk = indexed[start:start + size]

            def call_api(payload=[t for _, t in chunk]):
                return self.client.models.embed_content(
                    model=self.model_name,
                    contents=payload,
                )

            try:
                response = await loop.run_in_executor(None, call_api)
            except Exception as e:
                # Chunk-local failure: the remaining chunks still go out.
                log.error("embedder.batch_failed", error=str(e), count=len(chunk))
                continue

            self._log_embedding_usage(response, len(chunk))

            embeddings = getattr(response, "embeddings", None) or []
            if len(embeddings) != len(chunk):
                log.error(
                    "embedder.batch_length_mismatch",
                    expected=len(chunk), received=len(embeddings),
                )
                continue

            # Responses come back in request order, so zip is the mapping.
            for (idx, _), emb in zip(chunk, embeddings):
                values = getattr(emb, "values", None)
                if values:
                    results[idx] = np.array(values, dtype=np.float32)

        return results

    async def embed_articles(self, articles: list[NewsArticle]) -> list[Optional[np.ndarray]]:
        """Batch counterpart to embed_article. Result is aligned 1:1 with input."""
        if not self._initialized:
            return [None] * len(articles)

        return await self.get_embeddings([_article_text(a) for a in articles])
