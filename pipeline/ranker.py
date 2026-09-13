"""
Batch Importance Ranker

Scores a list of classified articles by their importance / market impact,
0.0 to 10.0. The model is whatever MODEL_RANKER points at.
"""

from __future__ import annotations

import json
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, create_model

from config.logging_config import get_logger
from config.llm import complete, is_llm_configured, is_transient, parse_json_list
from config.settings import settings
from config.usage import track_llm
from data.models import NewsArticle
from data.database import Database
from pipeline.classifier import batch_label

log = get_logger(__name__)


class RankedArticle(BaseModel):
    """One scored article, as parsed back out of a response.

    `importance_score` has no default, and that is the point of it. It used to
    default to 0.0, which did damage twice over. Pydantic leaves a defaulted
    field out of the JSON schema's `required` list, so the request told the
    model the score was optional — the shape that got the batch classifier
    answered with near-empty objects. And a result that duly left it out
    validated as a real 0.0, a verdict that sinks a story below the brief and
    alert floors for good. Required, a result with no usable score fails
    validation on its own: its article stays unscored, and the rest of the
    batch still lands.
    """
    id: str
    importance_score: float = Field(ge=0.0, le=10.0)


def _rank_schema(labels: list[str]) -> Any:
    """
    `list[RankedArticle]` with `id` narrowed to exactly this batch's labels.

    Built per call because the allowed set is the batch, as `_batch_schema` is
    for the classifier: the labels reach the request as an `enum` on the id
    property, and both fields as `required`. The array's length is the one
    thing no Python type can carry, so it is `complete(exact_items=...)`.
    """
    model = create_model(
        "RankedArticleLabelled",
        __base__=RankedArticle,
        id=(Literal[tuple(labels)], ...),  # type: ignore[valid-type]
    )
    return list[model]  # type: ignore[valid-type]


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

There are {count} articles above, labelled {first_label} through {last_label}.
Return {count} results, exactly one result per input article, no more and no fewer.
Echo each article's "id" back verbatim: results are matched by id, not by position.
Every result needs an importance_score; a result without one leaves its article
unscored. No markdown, no backticks.

Reply with a single object holding one "items" array, one entry in it per input
article — not one object per line, and not one result for the batch:
{{"items": [{{"id": "item_1", "importance_score": 0.0}}]}}
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

        An article the response gives no valid score keeps `importance_score`
        None rather than a 0.0 standing in for "no answer"; as with the
        classifier, the caller decides what an unanswered article means.
        """
        if not articles:
            return []

        if not is_llm_configured() or not self.model_name:
            log.warning("ranker.skipped", reason="LLM not configured", count=len(articles))
            return articles

        # Batch-local labels rather than article ids, as the classifier sends
        # (see `batch_label`). A 30-plus character id echoed once per result is
        # output paid for nothing, and a model repeating one runs to the cap and
        # truncates the whole batch into unparseable JSON.
        labels = [batch_label(i) for i in range(1, len(articles) + 1)]
        by_label = dict(zip(labels, articles))

        payload = [
            {
                "id": label,
                "headline": a.headline,
                "event_type": a.event_type,
                "urgency": a.urgency,
                "sentiment_score": a.sentiment_score,
                "classification_summary": a.classification_summary,
            }
            for label, a in by_label.items()
        ]

        prompt = RANKING_PROMPT.format(
            # Compact. indent=2 spent a line and its indentation on every field
            # of every article, and ASCII escaping turned each curly quote or
            # dash in a headline into a six-character \u sequence.
            articles_json=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            count=len(articles),
            first_label=labels[0],
            last_label=labels[-1],
        )

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
                    schema=_rank_schema(labels),
                    # The length the schema type cannot state. Without it the
                    # items array has no minimum, a single object satisfies it,
                    # and a model answers a json_schema with the minimum it
                    # permits — how the batch classifier got `sent=10 matched=1`.
                    exact_items=len(articles),
                    # Scoring against fixed calibration anchors is recall, not
                    # deliberation. Left at the model default, reasoning tokens
                    # draw down the same max_tokens budget as the answer and a
                    # batch comes back with an empty content field.
                    reasoning="none",
                    # {"id": "item_N", "importance_score": x} is ~20 tokens per
                    # article, so 80 leaves ample room; the headroom is because a
                    # constrained response truncates into unparseable output
                    # rather than degrading.
                    max_tokens=settings.rank_max_output_tokens_per_article * len(articles),
                )

            results = response.parsed
            if results is None:
                # The response missed the schema. The tolerant text parse still
                # salvages the shapes a model reaches for unprompted, and one
                # malformed entry must not cost the rest of the batch — an item
                # with no score, or one outside 0–10, is skipped here and its
                # article stays unscored.
                results = []
                for item in parse_json_list(response.text):
                    if not isinstance(item, dict):
                        continue
                    try:
                        results.append(RankedArticle.model_validate(item))
                    except Exception:
                        continue

            scores = {r.id: r.importance_score for r in results}

            applied = 0
            for label, article in by_label.items():
                if label in scores:
                    article.importance_score = float(scores[label])
                    applied += 1

            # Report what was scored, not what was sent. These were the same
            # number until a response shape the parser did not expect scored
            # none of them and still logged a full batch.
            log.info("ranker.success", count=len(articles), applied=applied)
            if applied < len(articles):
                log.warning(
                    "ranker.partial", count=len(articles), applied=applied,
                    unscored=len(articles) - applied,
                )

        except Exception as e:
            log.error("ranker.failed", error=str(e), count=len(articles))
            # Surface infrastructure faults so the caller can retry them. A
            # parse failure is swallowed as before — at temperature 0.1 a retry
            # buys substantially the same response for the same price.
            if is_transient(e):
                raise

        return articles
