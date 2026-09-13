"""
Event Classifier Pipeline Component

Uses Gemini to classify incoming news articles into event types,
extract sentiment, urgency, and affected sectors/tickers.
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, create_model
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from config.logging_config import get_logger
from config.llm import (
    complete,
    is_llm_configured,
    is_transient,
    parse_json_list,
    salvage_json_array,
)
from config.settings import settings
from config.usage import track_llm
from data.models import NewsArticle
from data.database import Database
from data.filters import CASHTAG_PATTERN, FINANCIAL_KEYWORDS, REDDIT_KEYWORDS

log = get_logger(__name__)


class ClassifierNotConfigured(RuntimeError):
    """
    No model slug is set for the lane that was asked to classify.

    Raised rather than returning None, because None is indistinguishable from
    "the model answered with nothing" and the caller treats that as a parse
    failure: `_persist_classifications` stamps every row in the batch
    `event_type='error'`. On the phone that turned an unset MODEL_CLASSIFIER
    into ~20 permanently poisoned rows per cycle, with nothing in the logs
    beyond a generic `classifier.batch_no_response`.
    """


class ClassifierResult(BaseModel):
    event_type: str = "unknown"
    sentiment_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    urgency: str = Field(default="low", pattern="^(low|medium|high|critical)$")
    suggested_direction: str = Field(default="neutral", pattern="^(bullish|bearish|neutral)$")
    affected_sectors: list[str] = Field(default_factory=list)
    affected_tickers: list[str] = Field(default_factory=list)
    countries: list[str] = Field(default_factory=list)
    classification_summary: str = ""


class BatchClassifierResult(ClassifierResult):
    """`ClassifierResult` plus the id it belongs to: what one batch item is.

    The declaration `_batch_schema` narrows per call. Sent as a list so that
    "one result per article" is carried by the request schema; asked in prose
    alone, against a json_mode root that must be an object, models answer a
    whole batch with a single flat classification.

    `id` has no default on purpose. A default puts it outside the generated JSON
    schema's `required` list, and a model that is told a field is optional omits
    it — results then match nothing, every article comes back unclassified, and
    the whole batch is written off as a parse failure. Required is the only
    version of this field that does its job.

    The inherited fields keep their defaults, because this class is also read as
    the *parsing* contract: a thin or salvaged response must still validate per
    item. `_batch_schema` strips those defaults for the request only, having
    learnt the other half of the same lesson — a field a model is told is
    optional is a field it omits, id or not.
    """
    id: str


# How batch items are labelled in the request, and the only ids a response may
# echo. `a1`..`aN` was the previous scheme and it had a specific, expensive
# failure: a model would fall into repeating the label — `"id": "a1a1a1a1a1…` —
# until it hit the output cap, truncating the JSON mid-string and costing every
# article in the batch. 305 such responses in one day on the phone, alongside
# 1,328 unmatched items. A longer, structured label is a far less likely
# repetition basin, and `_batch_schema` then pins the allowed values in the
# request's JSON schema so a conforming decoder cannot emit anything else.
def batch_label(index: int) -> str:
    """The label for the 1-based `index`-th article of a batch."""
    return f"item_{index}"


def _as_required(info: FieldInfo) -> FieldInfo:
    """The same field, minus its default, so it lands in `required`."""
    clone = copy.deepcopy(info)
    clone.default = PydanticUndefined
    clone.default_factory = None
    return clone


def _batch_schema(labels: list[str]) -> Any:
    """
    `list[BatchClassifierResult]` with `id` narrowed to exactly these labels and
    every other field demanded rather than merely offered.

    Built per call because the allowed set is the batch. The labels reach the
    request as an `enum` on the id property, which is the difference between
    "please echo the id" as prose and as a constraint.

    The `required` list is the other half, and the expensive half. Pydantic only
    lists default-less fields there, so a schema built from `ClassifierResult`
    as written marks eight of its nine properties optional — and a model reading
    a json_schema `response_format` answers with the *minimum* the schema
    permits. Measured against deepseek-v4-flash with `required: ["id"]`, ten
    articles came back as a single `{"id": "item_1", "urgency": "medium"}`: 41
    output tokens, nine rows unmatched, three attempts burnt in fifteen minutes.
    The fallback slug obliged with all ten ids and nothing else in them — the
    shape that logs `applied: 10` while classifying nothing. Re-declaring the
    fields without defaults takes output back to ~1,400 tokens of real
    classifications, 7/7 runs across both slugs.

    `ClassifierResult` itself keeps its defaults: it is the *parsing* model, and
    a thin or truncated response still has to validate per item so one bad
    object costs one article rather than the batch.

    Array *length* cannot be said in a Python type, so "exactly N of these" is
    `complete(exact_items=...)` instead — see `_classify_group`.
    """
    fields: dict[str, Any] = {
        "id": (Literal[tuple(labels)], ...),  # type: ignore[valid-type]
    }
    for name, info in BatchClassifierResult.model_fields.items():
        if name != "id":
            fields[name] = (info.annotation, _as_required(info))
    model = create_model("BatchClassifierResultLabelled", **fields)
    return list[model]  # type: ignore[valid-type]

# Taxonomy, calibration anchors, few-shot examples and country rules are
# identical whether one article or twenty are being classified, so they live in
# one constant and both prompts compose from it.
#
# This block is ~4.4k characters — roughly 1.1k tokens, and the bulk of a
# single-article classification prompt. Sent once per article it dominates the
# input bill; sent once per batch of N it amortises away. That is the whole
# argument for classify_batch.
_CLASSIFICATION_GUIDANCE = """─── EVENT TYPE TAXONOMY ───
Choose the most specific category that fits:
- "earnings": Quarterly/annual reports, guidance updates, revenue warnings, profit warnings
- "macro": Central bank decisions, inflation/CPI/PPI, employment reports, GDP, PMI, interest rates
- "geopolitical": Trade wars, sanctions, military conflicts, diplomatic events, supply chain disruptions
- "merger": M&A announcements, acquisition rumors, takeover bids, spin-offs, divestitures
- "product_launch": New products, FDA approvals, key partnerships, major contract wins
- "regulatory": Government investigations, antitrust, new legislation, compliance, fines
- "ipo": Public offerings, direct listings, SPAC mergers, IPO pricing updates
- "personnel": CEO/CFO changes, board shakeups, major layoffs, activist investor moves
- "general": Market-relevant news that doesn't fit above (use sparingly — prefer a specific category)

─── SENTIMENT CALIBRATION ───
Use these anchors to assign sentiment_score (-1.0 to +1.0):
- -0.9 to -0.7: Catastrophic (fraud, bankruptcy, major lawsuit loss, CEO criminal charges)
- -0.6 to -0.4: Significantly negative (earnings miss >10%, regulatory crackdown, product recall)
- -0.3 to -0.1: Mildly negative (minor guidance cut, sector headwinds, single-analyst downgrade)
- -0.1 to +0.1: Neutral or mixed (routine filings, balanced reporting, no clear bias)
- +0.1 to +0.3: Mildly positive (beat low expectations, new partnership, analyst upgrade)
- +0.4 to +0.6: Significantly positive (strong earnings beat, major contract, FDA approval)
- +0.7 to +0.9: Exceptional (blockbuster drug results, takeover offer at large premium)

─── URGENCY ───
- "low": Background info, no time pressure
- "medium": Notable event worth acting on this week
- "high": Time-sensitive, action needed within 24h (earnings surprise, major announcement)
- "critical": Breaking news requiring immediate attention (black swan, flash crash, crisis)

─── FEW-SHOT EXAMPLES ───

Example 1:
Headline: "Apple beats Q3 estimates, announces $90B buyback"
Summary: Apple reported Q3 revenue of $85.8B vs $84.4B expected. EPS of $1.40 vs $1.35 expected. The company also announced a $90 billion share buyback program and raised its dividend by 4%.
→ {{"event_type": "earnings", "sentiment_score": 0.55, "urgency": "high", "suggested_direction": "bullish", "affected_sectors": ["Technology", "Consumer Electronics"], "affected_tickers": ["AAPL"], "classification_summary": "Apple beat Q3 estimates on both revenue and EPS. The $90B buyback and dividend increase signal strong management confidence. Stock likely to see positive reaction in near term."}}

Example 2:
Headline: "Fed raises rates by 25bps, signals potential pause"
Summary: The Federal Reserve raised its benchmark interest rate by 25 basis points to 5.25-5.50%, but indicated it may pause further hikes to assess economic data. Markets rallied on the dovish tone.
→ {{"event_type": "macro", "sentiment_score": 0.35, "urgency": "high", "suggested_direction": "bullish", "affected_sectors": ["Financials", "Real Estate", "Technology"], "affected_tickers": ["SPY", "QQQ", "XLF"], "classification_summary": "Fed raised rates as expected but signaled a potential pause, which markets interpreted dovishly. Rate-sensitive sectors like Tech and Real Estate likely to benefit if the pause materializes."}}

─── COUNTRIES ───
List the countries this story is materially about, as ISO 3166-1 alpha-2 codes
(e.g. ["US", "CN"]). Judge by where the economic impact lands, not where the
article was published: a Reuters story about Chinese export controls is ["CN"],
or ["CN", "US"] if it turns on the trade relationship. Use at most three, and
return [] for a story with no clear geography.
"""

CLASSIFICATION_PROMPT = """
You are a professional financial analyst AI. Analyze the following news article and extract structured information.
Respond ONLY with a valid JSON object matching the requested schema. No markdown formatting, no backticks.

Article Headline: {headline}
Article Summary: {summary}

""" + _CLASSIFICATION_GUIDANCE + """
Now classify the actual article using the same JSON schema:
{{
  "event_type": "string",
  "sentiment_score": 0.0,
  "urgency": "string",
  "suggested_direction": "string",
  "affected_sectors": ["string"],
  "affected_tickers": ["string"],
  "countries": ["string"],
  "classification_summary": "string"
}}
"""

BATCH_CLASSIFICATION_PROMPT = """
You are a professional financial analyst AI. Analyze EVERY news article in the list below and extract structured information for each one independently.
Return one result per input article — never a single result standing for the
whole batch. No markdown formatting, no backticks.

""" + _CLASSIFICATION_GUIDANCE + """
Articles to classify:
{articles_json}

There are {count} articles above, labelled {first_label} through {last_label}.
Return {count} objects — one per article, no more and no fewer. Echo each
article's "id" back verbatim; results are matched by id, not by position, and an
object with a missing or invented id is discarded. Every field below is
mandatory on every object: a classification with no event_type or an empty
classification_summary is the same as no classification at all. Judge each
article on its own; do not let one article's sentiment influence another's.

"urgency" must be one of low, medium, high, critical, and
"suggested_direction" one of bullish, bearish, neutral — a value outside those
vocabularies discards that article's result.

Reply with a single object holding one "items" array, and one entry in it per
input article — not one object per line, and not one result for the batch:
{{
  "items": [
    {{
      "id": "item_1",
      "event_type": "string",
      "sentiment_score": 0.0,
      "urgency": "string",
      "suggested_direction": "string",
      "affected_sectors": ["string"],
      "affected_tickers": ["string"],
      "countries": ["string"],
      "classification_summary": "string"
    }}
  ]
}}
"""

REDDIT_CLASSIFICATION_PROMPT = """
You are a financial sentiment analyst specializing in Reddit's r/WallStreetBets (WSB) and investing communities.
Analyze the following Reddit post and its top comments, then extract structured information.

─── WSB LINGO GUIDE ───
Bullish signals: "tendies", "diamond hands", "rocket/moon", "calls", "YOLO", "to the moon", "loading up", "buying the dip", "DD" (due diligence posted), "🚀"
Bearish signals: "puts", "bag holder", "drill", "rug pull", "paper hands", "inverse", "dump it", "shorting", "GUH", "📉"
Neutral/mixed: "wheel strategy", "theta gang", "selling covered calls", "wheel", "iron condor"

─── SENTIMENT CALIBRATION (WSB-adjusted) ───
WSB sentiment is often amplified/extreme. Adjust accordingly:
- -0.9 to -0.7: Mass panic, widespread loss porn, coordinated short attack narrative
- -0.6 to -0.4: Strongly negative consensus across comments, credible bearish DD
- -0.3 to -0.1: Mildly skeptical, mixed but leaning negative
- -0.1 to +0.1: Genuinely divided community, or purely meme/noise content
- +0.1 to +0.3: Mildly optimistic, some bullish DD but not viral
- +0.4 to +0.6: Strong bullish consensus, front-page DD with awards, gamma squeeze talk
- +0.7 to +0.9: Viral YOLO, "MOASS" narrative, massive FOMO, coordinated buying

─── FEW-SHOT EXAMPLE ───
Post Title: "GME YOLO update — $50k to $500k, still not selling 💎🙌"
Summary: User posts screenshot of GameStop position up 10x. Comments overwhelmingly bullish with "not selling till $1M" and "shorts haven't covered" narratives.
Top Comments:
- u/deepfuckingvalue: "The squeeze hasn't even started yet. Holding."
- u/wsb_god: "This is the way. Diamond hands pay."
- u/skeptic: "Take profits you idiot, this is a pump and dump"
→ {{"event_type": "meme_stock", "sentiment_score": 0.55, "urgency": "high", "suggested_direction": "bullish", "affected_sectors": ["Retail", "Consumer Discretionary"], "affected_tickers": ["GME"], "classification_summary": "Highly bullish WSB sentiment around GME with viral YOLO update. Diamond hands narrative dominates comments but one skeptical voice warns of pump and dump. Momentum-driven retail interest remains elevated."}}

Post Title: {headline}
Post Summary: {summary}

Top Comments:
{comments}

Determine the following fields based on the overall sentiment (post + comments combined, weighting highly-upvoted comments more):
- event_type: Choose from the event type taxonomy (add "meme_stock" for WSB-specific hype).
- sentiment_score: Float between -1.0 and +1.0 using the WSB-adjusted calibration above.
- urgency: "low", "medium", "high", or "critical".
- suggested_direction: "bullish", "bearish", or "neutral".
- affected_sectors: List of market sectors affected.
- affected_tickers: List of relevant tickers. Extract from '$TICKER' mentions or contextual bare tickers. Be thorough.
- countries: Countries the discussion materially concerns, as ISO 3166-1 alpha-2 codes (max 3). Return [] if there is no clear geography.
- classification_summary: A short paragraph summarizing the community sentiment and market impact. Mention specific catalysts, dates, and key figures discussed.

Respond ONLY with a valid JSON object. No markdown formatting, no backticks.

JSON Schema:
{{
  "event_type": "string",
  "sentiment_score": 0.0,
  "urgency": "string",
  "suggested_direction": "string",
  "affected_sectors": ["string"],
  "affected_tickers": ["string"],
  "countries": ["string"],
  "classification_summary": "string"
}}
"""

class ArticleClassifier:
    """Classifies NewsArticles with MODEL_CLASSIFIER, falling back to its pair."""

    def __init__(self, db: Optional[Database] = None):
        self.db = db or Database()
        self.model_name = settings.model_classifier

    def is_configured(self) -> bool:
        """
        Whether a news classification call can be made at all.

        Checked before the backlog job spends a tick on candidate selection and
        a stale-marking UPDATE, so an unconfigured deployment reports that fact
        on the dashboard instead of looping over work it cannot do. Reads
        `settings` live rather than the `self.model_name` captured at __init__,
        which is also what lets a test patch the slug after construction.

        Reddit slugs are deliberately not part of this: Reddit is a minority of
        the intake and its own lane, so an unset MODEL_REDDIT_SENTIMENT should
        not stop news from being classified. `is_reddit_configured` answers for
        that lane.
        """
        return bool(settings.model_classifier or settings.model_classifier_fallback)

    def is_reddit_configured(self) -> bool:
        """
        Whether a Reddit post can be classified at all.

        Leaving both Reddit slugs unset is a supported configuration, not an
        outage: news is classified as normal and Reddit rows wait, unclassified,
        for a slug. `classify_batch` checks it so an unconfigured lane skips its
        call instead of raising, and the backlog job checks it to keep those rows
        out of its candidate slots. Read live from `settings`, like
        `is_configured`.
        """
        return bool(
            settings.model_reddit_sentiment or settings.model_reddit_sentiment_fallback
        )

    def should_classify(self, article: NewsArticle) -> bool:
        """Determines if the article has enough financial relevance to warrant classification."""
        # 1. Extract potential tickers & text context
        # For Reddit, include comments. For regular news, comments will be empty.
        comments_data = article.raw_data.get("comments", [])
        comments_text = " ".join(
            str(c.get("body", "")) for c in comments_data if isinstance(c, dict)
        )
        combined_text = f"{article.headline} {article.summary} {comments_text}"

        # 1. Look for explicit ticker symbols (e.g. $AAPL)
        if CASHTAG_PATTERN.search(combined_text):
            return True

        # 2. Look for financial keywords and specific terms
        combined_text_lower = combined_text.lower()
        
        # Fast lookup against set (for exact word boundaries, use regex)
        all_keywords = FINANCIAL_KEYWORDS | REDDIT_KEYWORDS
        for keyword in all_keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', combined_text_lower):
                return True

        # If no tickers, no financial words, and no options slang are present, skip it.
        return False

    # ── Shared helpers (used by both the single and batch paths) ─────────

    NOISE_SUMMARY = "Filtered out by pre-classification noise heuristics."

    def _mark_noise(self, article: NewsArticle) -> NewsArticle:
        """Applies the terminal 'noise' verdict locally, without an LLM call."""
        log.info("classifier.pre_filtered", article_id=article.id, headline=article.headline)
        article.event_type = "noise"
        article.sentiment_score = 0.0
        article.urgency = "low"
        article.suggested_direction = "neutral"
        article.classification_summary = self.NOISE_SUMMARY
        return article

    @staticmethod
    def _is_reddit(article: NewsArticle) -> bool:
        return article.source_type == "social" and article.source_name.startswith("reddit")

    @staticmethod
    def _reddit_comments_text(article: NewsArticle) -> str:
        comments_data = article.raw_data.get("comments", [])
        comments = [
            f"- {c.get('author', '[unknown]')}: {c.get('body', '')}"
            for c in comments_data
            if isinstance(c, dict) and c.get("body")
        ]
        return "\n".join(comments) if comments else "(No comments)"

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        """Models occasionally wrap JSON in markdown despite being told not to."""
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    @staticmethod
    def _apply_result(article: NewsArticle, parsed: ClassifierResult) -> None:
        """Writes a parsed result onto an article. Shared by single and batch."""
        article.event_type = parsed.event_type
        article.sentiment_score = parsed.sentiment_score
        article.urgency = parsed.urgency
        article.suggested_direction = parsed.suggested_direction
        article.affected_sectors = parsed.affected_sectors
        # Normalise to upper-case ISO codes; models sometimes return "us".
        article.countries = [
            c.strip().upper() for c in parsed.countries if c and len(c.strip()) == 2
        ]
        # Combine with existing tickers if alpha vantage provided them
        article.affected_tickers = list(
            set(article.affected_tickers) | set(parsed.affected_tickers)
        )
        article.classification_summary = parsed.classification_summary

    # ── Batch classification ────────────────────────────────────────────

    async def classify_batch(self, articles: list[NewsArticle]) -> list[NewsArticle]:
        """
        Classifies many articles in as few calls as possible.

        The ~1.1k-token guidance preamble is the bulk of a classification
        prompt, and per-article calls re-sent it every single time. Batching
        amortises it across the group.

        Reddit posts use both a different prompt and a different model, so a
        mixed input goes out as at most two calls. Articles rejected by the free
        should_classify() heuristic are marked noise locally and never sent.

        Returns the same list, modified in place. Articles the model did not
        answer for are returned untouched — the caller decides what that means.
        So are Reddit posts while the Reddit lane has no slug, and those are not
        a failure of any kind: nothing was asked about them.
        """
        if not articles:
            return []

        if not is_llm_configured():
            log.warning("classifier.skipped", reason="No LLM configured", count=len(articles))
            return articles

        candidates = articles
        if not self.is_reddit_configured():
            # Held back before the noise heuristic, not after: with no Reddit slug
            # a Reddit post gets no verdict of any kind until one is set. The call
            # used to be made regardless — the news group below went first, the
            # Reddit group then raised `ClassifierNotConfigured`, and the news
            # results that had just been paid for were discarded with the
            # exception, never persisted and never counted, so the same rows were
            # sent and billed again on every run. The backlog job reports the
            # unconfigured lane once per run, so this stays at debug.
            candidates = [a for a in articles if not self._is_reddit(a)]
            if len(candidates) < len(articles):
                log.debug(
                    "classifier.reddit_lane_unconfigured_held_back",
                    count=len(articles) - len(candidates),
                )

        to_send: list[NewsArticle] = []
        for article in candidates:
            if self.should_classify(article):
                to_send.append(article)
            else:
                self._mark_noise(article)

        reddit_group = [a for a in to_send if self._is_reddit(a)]
        news_group = [a for a in to_send if not self._is_reddit(a)]

        if news_group:
            await self._classify_group(news_group, is_reddit=False)
        if reddit_group:
            await self._classify_group(reddit_group, is_reddit=True)

        return articles

    async def _classify_group(self, articles: list[NewsArticle], is_reddit: bool) -> None:
        """One LLM call for a homogeneous group. Modifies articles in place."""
        # Batch-local labels rather than article ids. An article id is a
        # 30-plus character hash the model has to echo verbatim for every
        # result, and models degenerate on them: an id ending c34c came back
        # as "...c34cc34cc34cc34" repeated until the batch hit its token cap
        # and truncated into unparseable JSON, costing every article in it.
        # Short labels stay explicit — results are still matched by the label
        # the model echoes, never by position — and `batch_label` explains why
        # they are no longer as short as `a1`.
        keys = [batch_label(i) for i in range(1, len(articles) + 1)]
        by_key_article = dict(zip(keys, articles))

        payload = []
        for key, a in zip(keys, articles):
            item = {"id": key, "headline": a.headline, "summary": a.summary or ""}
            if is_reddit:
                item["top_comments"] = self._reddit_comments_text(a)
            payload.append(item)

        base = REDDIT_CLASSIFICATION_PROMPT if is_reddit else BATCH_CLASSIFICATION_PROMPT
        if is_reddit:
            # The Reddit template is single-article; wrap it for batch use by
            # reusing its lingo guide and calibration, then swapping the tail.
            prompt = (
                base.split("Post Title:")[0]
                + "\nClassify EVERY post in the list below independently.\n"
                  "Return one result per input post — never one result for "
                  "the whole batch.\n\n"
                  f"Posts:\n{json.dumps(payload, ensure_ascii=False)}\n\n"
                  f"There are {len(articles)} posts above, labelled {keys[0]} "
                  f"through {keys[-1]}. Return {len(articles)} objects — one per "
                  "post, no more and no fewer. Every field is mandatory on every "
                  "object.\n"
                  'Echo each "id" back verbatim. Reply with a single object '
                  'holding one "items" array, one entry per input post — not '
                  'one object per line:\n'
                  '{"items": [{"id": "item_1", "event_type": "string", '
                  '"sentiment_score": 0.0, "urgency": "string", '
                  '"suggested_direction": "string", "affected_sectors": ["string"], '
                  '"affected_tickers": ["string"], "countries": ["string"], '
                  '"classification_summary": "string"}]}'
            )
            models = [settings.model_reddit_sentiment,
                      settings.model_reddit_sentiment_fallback]
        else:
            # No indent= — pretty-printing a batch payload is pure token waste.
            prompt = base.format(
                articles_json=json.dumps(payload, ensure_ascii=False),
                count=len(articles),
                first_label=keys[0],
                last_label=keys[-1],
            )
            models = [self.model_name, settings.model_classifier_fallback]

        operation = "classify_batch_reddit" if is_reddit else "classify_batch"
        # The cap has to scale with the batch: with response_format=json_object,
        # running out of budget yields truncated, unparseable JSON rather than a
        # short answer, which would cost the whole batch.
        max_output = settings.classify_max_output_tokens_per_article * len(articles)
        text = await self._call_with_fallback(
            prompt, models, operation, max_output,
            # The enum of exactly this batch's labels, so the id the model has to
            # echo is a constrained choice rather than free text — and every
            # field required, so the model cannot satisfy the schema with an
            # object holding only that id.
            schema=_batch_schema(keys),
            # The length the schema cannot state in a Python type. Without it the
            # items array has no minimum, one object satisfies it, and that is
            # exactly what came back: `sent=10 matched=1`.
            exact_items=len(articles),
        )
        if not text:
            log.error("classifier.batch_no_response", count=len(articles), operation=operation)
            return

        # Whether the response parsed cleanly or was recovered from a truncated
        # one. Only a clean parse is allowed to fall back to positional mapping
        # below: a salvaged response is by definition missing its tail, so "as
        # many results as articles" can no longer mean they line up.
        parsed_cleanly = True
        try:
            results = parse_json_list(text)
        except Exception as e:
            # A truncated array still holds whole objects in front of the break.
            # Recover those rather than discarding a paid-for response; the
            # articles that go unmatched stay NULL and are retried, bounded by
            # classification_attempts.
            results = salvage_json_array(text, key="items")
            parsed_cleanly = False
            if not results:
                # The response head is the whole diagnosis for this failure — the
                # shape a model returns under json_mode is the thing that varies,
                # and without it a parse failure is indistinguishable from an
                # outage in the logs.
                log.error(
                    "classifier.batch_parse_failed",
                    error=str(e), count=len(articles), operation=operation,
                    response_head=text[:400],
                )
                return
            log.warning(
                "classifier.batch_salvaged_partial",
                error=str(e), sent=len(articles), recovered=len(results),
                operation=operation, response_head=text[:400],
            )

        # Match by label, never by position — a model that drops or reorders an
        # item would otherwise silently attach the wrong classification to the
        # wrong article, which is far worse than not classifying it at all.
        by_key = {
            str(item["id"]): item
            for item in results
            if isinstance(item, dict) and item.get("id")
        }

        if parsed_cleanly and not by_key and len(results) == len(articles) and all(
            isinstance(item, dict) for item in results
        ):
            # The one case where position is safe: the model echoed no ids at
            # all, and returned exactly as many objects as were sent. Nothing
            # can have been reordered relative to a mapping that does not exist,
            # and the alternative is discarding a complete, paid-for response.
            #
            # Strictly zero ids. A *partial* set of ids is the dangerous shape —
            # it means the model was tracking identity and dropped or merged
            # some items, so its ordering is exactly what cannot be trusted.
            by_key = dict(zip(by_key_article.keys(), results))
            log.warning(
                "classifier.batch_positional_fallback",
                operation=operation, count=len(articles),
                reason="no ids in response; mapped by position",
            )
        elif by_key and len(by_key) < len(articles):
            log.warning(
                "classifier.batch_unmatched_ids",
                operation=operation, sent=len(articles),
                matched=len(by_key.keys() & by_key_article.keys()),
                returned_ids=sorted(by_key)[:20],
            )

        applied = 0
        for key, article in by_key_article.items():
            item = by_key.get(key)
            if item is None:
                log.warning("classifier.batch_missing_result", article_id=article.id)
                continue
            try:
                # Per-item validation: one malformed object must not cost the
                # whole batch.
                parsed = ClassifierResult.model_validate(
                    {k: v for k, v in item.items() if k != "id"}
                )
                self._apply_result(article, parsed)
                applied += 1
            except Exception as e:
                log.error(
                    "classifier.batch_item_invalid",
                    article_id=article.id, error=str(e),
                )

        if applied == 0:
            # A parseable response that classified nothing is the shape that
            # hides worst: the batch looks like it ran, and upstream writes every
            # row off as 'error'. The head of the response is the only thing that
            # says which shape actually came back.
            log.error(
                "classifier.batch_applied_none",
                operation=operation, sent=len(articles),
                returned=len(results), response_head=text[:400],
            )
        log.info(
            "classifier.batch_complete",
            operation=operation, sent=len(articles), applied=applied,
        )

    async def _call_one(
        self, model: str, prompt: str, operation: str, max_output_tokens: int,
        schema: Any = None, exact_items: Optional[int] = None,
    ) -> str:
        """One attempt at one model. Logs it, returns the text, or raises.

        `schema` constrains the response shape on the request itself, and
        `exact_items` its length. The text is still what comes back: the batch
        path validates per item, so one model drifting outside the urgency
        vocabulary costs that article rather than the whole batch, which is what
        `parsed` would do.
        """
        with track_llm(self.db, model, operation,
                       prompt_text=prompt, store_text=True) as u:
            u.response = response = await complete(
                model=model,
                prompt=prompt,
                temperature=0.0,
                schema=schema,
                json_mode=schema is None,
                max_tokens=max_output_tokens,
                exact_items=exact_items,
                # Classification needs fast JSON, not deep reasoning. This was
                # the provider-specific thinking:{"type":"disabled"}.
                reasoning="none",
            )
        return response.text.strip()

    async def _call_with_fallback(
        self, prompt: str, models: list[str], operation: str,
        max_output_tokens: int, schema: Any = None,
        exact_items: Optional[int] = None,
    ) -> Optional[str]:
        """
        Try each configured model in turn. Returns raw response text.

        This used to be a *provider* fallback — DeepSeek, then Gemini — written
        out as two near-identical blocks. With one gateway it is a *model*
        fallback, and still worth having: OpenRouter surfaces a failing
        upstream as an error on that slug, and the second slug is a different
        upstream. What it can no longer survive is OpenRouter itself being down.

        Raises if every attempt failed transiently, so the caller can leave the
        batch unclassified for a later pass instead of writing a permanent
        'error' verdict over an outage. Raises `ClassifierNotConfigured` if
        there is no slug to try at all.
        """
        last_transient: Optional[BaseException] = None

        configured = [m for m in models if m]
        if not configured:
            # This loop used to run zero times and fall through to `return None`,
            # which upstream reads as "the model answered with garbage". An unset
            # MODEL_CLASSIFIER therefore presented as a permanent parse failure
            # and stamped every row 'error' — unrecoverable without a manual
            # UPDATE, and invisible in the logs.
            log.error(
                "classifier.not_configured",
                operation=operation,
                settings_checked=["MODEL_CLASSIFIER", "MODEL_CLASSIFIER_FALLBACK"]
                if operation.startswith("classify_batch") and "reddit" not in operation
                else ["MODEL_REDDIT_SENTIMENT", "MODEL_REDDIT_SENTIMENT_FALLBACK"],
                impact="ingest is embedding-only; no article will be classified",
                hint="set a slug from https://openrouter.ai/models and restart",
            )
            raise ClassifierNotConfigured(
                f"No model configured for {operation}; set MODEL_CLASSIFIER"
            )

        for model in configured:
            try:
                return await self._call_one(
                    model, prompt, operation, max_output_tokens, schema,
                    exact_items,
                )
            except Exception as e:
                # track_llm has already written the error row and re-raised.
                transient = is_transient(e)
                log.warning(
                    "classifier.model_failed",
                    model=model, operation=operation,
                    error=str(e), transient=transient,
                )
                if transient:
                    last_transient = e

        if last_transient is not None:
            raise last_transient

        return None

    async def classify(self, article: NewsArticle) -> NewsArticle:
        """
        Classifies the given article and populates its classification fields.
        Returns the modified article.

        Prefer `classify_batch` — the ~1.1k-token guidance preamble is sent
        once per call either way, so a per-article call pays it in full.
        """
        if not is_llm_configured():
            log.warning("classifier.skipped", reason="No LLM configured", article_id=article.id)
            return article

        # Apply pre-filter
        if not self.should_classify(article):
            return self._mark_noise(article)

        if self._is_reddit(article):
            prompt = REDDIT_CLASSIFICATION_PROMPT.format(
                headline=article.headline,
                summary=article.summary,
                comments=self._reddit_comments_text(article),
            )
            models = [settings.model_reddit_sentiment,
                      settings.model_reddit_sentiment_fallback]
        else:
            prompt = CLASSIFICATION_PROMPT.format(
                headline=article.headline,
                summary=article.summary,
            )
            models = [self.model_name, settings.model_classifier_fallback]

        try:
            text = await self._call_with_fallback(
                prompt, models, "classify",
                settings.classify_max_output_tokens_per_article,
            )
        except Exception as e:
            # Every model failed transiently. Leave the article unclassified so
            # a later pass retries it, rather than stamping a verdict.
            log.warning("classifier.failed", article_id=article.id, error=str(e))
            return article

        if not text:
            return article

        try:
            parsed = ClassifierResult.model_validate_json(self._strip_code_fence(text))
            self._apply_result(article, parsed)

            log.debug("classifier.success", article_id=article.id, event_type=article.event_type)

        except Exception as e:
            log.error("classifier.parse_failed", article_id=article.id, error=str(e), text=text)
            
        return article
