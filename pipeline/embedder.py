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
from data.models import NewsArticle

log = get_logger(__name__)

class GeminiEmbedder:
    """Generates embeddings using Gemini API."""

    def __init__(self, model_name: str = "gemini-embedding-001"):
        self.model_name = model_name
        self.client = None
        self._initialized = False

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

        parts = [article.headline]
        if article.summary:
            parts.append(article.summary)
            
        content = "\n\n".join(parts)
        
        return await self.get_embedding(content)
