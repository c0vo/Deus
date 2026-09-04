"""
Batch Importance Ranker

Scores a list of classified articles by their importance / market impact,
0.0 to 10.0. The model is whatever MODEL_RANKER points at.
"""

from __future__ import annotations

import json
from typing import Optional

from pydantic import BaseModel, Field

from config.logging_config import get_logger
from config.llm import complete, is_llm_configured, is_transient, parse_json_list
from config.settings import settings
from config.usage import track_llm
from data.models import NewsArticle
from data.database import Database

log = get_logger(__name__)


class RankedArticle(BaseModel):
    """One scored article. Sent as `list[RankedArticle]` so the required shape
    travels with the request instead of as a sentence in the prompt."""
    id: str
    importance_score: float = Field(default=0.0, ge=0.0, le=10.0)


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

Return exactly one result per input article, echoing each article's "id" back
verbatim — results are matched by id, not by position. No markdown, no backticks.

Reply with a single object holding one "items" array, and one entry in it per
input article — not one object per line, and not one result for the batch:
{{
  "items": [
    {{ "id": "a1", "importance_score": 0.0 }}
  ]
}}
"""

class ArticleRanker:
    """Ranks a batch of NewsArticles by importance."""

    def __init__(self, db: Optional[Database] = None):
        self.db = db or Database()
        self.model_name = settings.model_ranker

    async def rank_batch(self, articles: list[NewsArticle]) -> list[NewsArticle]:
        """
        Evaluates a batch of articles and assigns importance_score.
        Modifies the articles in place and returns them.
        """
        if not articles:
            return []

        if not is_llm_configured() or not self.model_name:
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
            with track_llm(self.db, self.model_name, "rank_batch",
                           prompt_text=prompt, store_text=True) as u:
                u.response = response = await complete(
                    model=self.model_name,
                    prompt=prompt,
                    temperature=0.1,
                    # A schema rather than json_mode. json_mode only constrains
                    # the root to *an object*, which contradicts a prompt asking
                    # for an array — and a model resolving that conflict answers
                    # with one flat result for the whole batch. `list[...]`
                    # travels as a json_schema and comes back unwrapped.
                    schema=list[RankedArticle],
                    # Scoring against fixed calibration anchors is recall, not
                    # deliberation. Left at the model default, reasoning tokens
                    # draw down the same max_tokens budget as the answer and a
                    # batch comes back with an empty content field.
                    reasoning="none",
                    # {"id","importance_score"} per article is ~25 tokens;
                    # the headroom is because a constrained response truncates
                    # into unparseable output rather than degrading.
                    max_tokens=settings.rank_max_output_tokens_per_article * len(articles),
                )

            results = response.parsed
            if results is None:
                # The response missed the schema. The tolerant text parse still
                # salvages the shapes a model reaches for unprompted, and one
                # malformed entry must not cost the rest of the batch.
                results = []
                for item in parse_json_list(response.text):
                    if not isinstance(item, dict):
                        continue
                    try:
                        results.append(RankedArticle.model_validate(item))
                    except Exception:
                        continue

            result_map = {r.id: r.importance_score for r in results}
            
            applied = 0
            for article in articles:
                if article.id in result_map:
                    article.importance_score = float(result_map[article.id])
                    applied += 1

            # Report what was scored, not what was sent. These were the same
            # number until a response shape the parser did not expect scored
            # none of them and still logged a full batch.
            log.info("ranker.success", count=len(articles), applied=applied)
            if applied < len(articles):
                log.warning("ranker.partial", count=len(articles), applied=applied)
            
        except Exception as e:
            log.error("ranker.failed", error=str(e), count=len(articles))
            # Surface infrastructure faults so the caller can retry them. A
            # parse failure is swallowed as before — at temperature 0.1 a retry
            # buys substantially the same response for the same price.
            if is_transient(e):
                raise

        return articles
