"""
Batch Importance Ranker

Uses Gemini to evaluate a list of classified articles and rank them
by their importance/market impact score from 0.0 to 10.0.
"""

from __future__ import annotations

import json
from typing import Optional

from google import genai
from google.genai import types

from config.logging_config import get_logger
from config.llm import get_client, is_configured, is_transient, DEFAULT_SAFETY_SETTINGS
from config.settings import settings
from data.models import NewsArticle
from data.database import Database

log = get_logger(__name__)

RANKING_PROMPT = """
You are a senior financial analyst evaluating news for an active retail stock investor. Below is a list of classified news articles. Assign an importance score (0.0–10.0) to each based on its potential market impact.

─── SCORE CALIBRATION (use these anchors) ───
- 0.0–2.0: Noise, clickbait, or purely technical articles with zero tradable impact
- 2.1–4.0: Minor company-specific news affecting one small/mid-cap (new product feature, minor partnership, single-analyst note)
- 4.1–6.0: Notable event affecting a sector or a single large-cap (earnings from a major company, sector rotation signal, regulatory development)
- 6.1–8.0: Major event affecting multiple sectors or mega-caps (Fed rate decision, mega-cap earnings surprise, geopolitical flare-up, key economic data miss)
- 8.1–10.0: Market-moving emergency requiring immediate attention (black swan, surprise policy change, systemic risk event, major geopolitical crisis)

─── SCORING FACTORS (weigh these in your assessment) ───
+ Market cap affected: How large is the total market value impacted? (global > national > sector > single large-cap > small-cap)
+ Immediacy: Is the impact now/today, this week, or months away? Sooner = higher score.
+ Breadth: How many sectors/tickers are touched? More = higher score.
+ Actionability: Can an investor actually trade on this? Clear catalyst > vague trend piece.
+ Novelty: Is this new information or already priced in? Surprise > expected.

─── IMPORTANT ───
- Do NOT automatically penalize any sector (including real estate/property — a housing crash is highly market-relevant).
- Score based on market impact, not personal interest.
- A score of 9.0+ means: "If I could only read ONE article today, this would be it."

Articles:
{articles_json}

Respond ONLY with a JSON array of objects in the exact order and IDs provided. No markdown, no backticks.
Schema:
[
  {{
    "id": "string",
    "importance_score": 0.0
  }}
]
"""

class ArticleRanker:
    """Ranks a batch of NewsArticles by importance."""

    def __init__(self, client: Optional[genai.Client] = None, db: Optional[Database] = None):
        self.client = client or get_client()
        self.db = db or Database()
        self.model_name = settings.gemini_model_ranker

    async def rank_batch(self, articles: list[NewsArticle]) -> list[NewsArticle]:
        """
        Evaluates a batch of articles and assigns importance_score.
        Modifies the articles in place and returns them.
        """
        if not articles:
            return []

        if not self.client or not is_configured():
            log.warning("ranker.skipped", reason="LLM not configured", count=len(articles))
            return articles

        # Prepare payload
        payload = []
        for a in articles:
            payload.append({
                "id": a.id,
                "headline": a.headline,
                "event_type": a.event_type,
                "urgency": a.urgency,
                "sentiment_score": a.sentiment_score,
                "classification_summary": a.classification_summary
            })

        prompt = RANKING_PROMPT.format(articles_json=json.dumps(payload, indent=2))

        try:
            import time
            start_time = time.time()
            is_error = False
            error_msg = None
            response_text = None
            try:
                response = await self.client.aio.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        safety_settings=DEFAULT_SAFETY_SETTINGS,
                        response_mime_type="application/json",
                        # {"id","importance_score"} per article is ~25 tokens;
                        # the headroom is because JSON mode truncates into
                        # unparseable output rather than degrading.
                        max_output_tokens=settings.rank_max_output_tokens_per_article * len(articles),
                    )
                )
                response_text = response.text
            except Exception as e:
                is_error = True
                error_msg = str(e)
                raise
            finally:
                latency_ms = int((time.time() - start_time) * 1000)
                if not is_error and response and response.usage_metadata:
                    self.db.log_llm_usage(
                        model_name=self.model_name,
                        operation="rank_batch",
                        prompt_tokens=response.usage_metadata.prompt_token_count,
                        candidate_tokens=response.usage_metadata.candidates_token_count,
                        latency_ms=latency_ms,
                        is_error=False,
                        error_message=None,
                        prompt_text=prompt,
                        response_text=response_text
                    )
                elif is_error:
                    self.db.log_llm_usage(
                        model_name=self.model_name,
                        operation="rank_batch",
                        prompt_tokens=0,
                        candidate_tokens=0,
                        latency_ms=latency_ms,
                        is_error=True,
                        error_message=error_msg,
                        prompt_text=prompt,
                        response_text=None
                    )
            
            text = response.text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
                
            results = json.loads(text.strip())
            
            # Map results back to articles
            result_map = {item["id"]: item for item in results if "id" in item}
            
            for article in articles:
                if article.id in result_map:
                    res = result_map[article.id]
                    article.importance_score = float(res.get("importance_score", 0.0))
            
            log.info("ranker.success", count=len(articles))
            
        except Exception as e:
            log.error("ranker.failed", error=str(e), count=len(articles))
            # Surface infrastructure faults so the caller can retry them. A
            # parse failure is swallowed as before — at temperature 0.1 a retry
            # buys substantially the same response for the same price.
            if is_transient(e):
                raise

        return articles
