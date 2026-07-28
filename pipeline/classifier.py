"""
Event Classifier Pipeline Component

Uses Gemini to classify incoming news articles into event types,
extract sentiment, urgency, and affected sectors/tickers.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from config.logging_config import get_logger
from config.llm import get_client, is_configured, DEFAULT_SAFETY_SETTINGS, get_deepseek_client, is_deepseek_configured
from config.settings import settings
from data.models import NewsArticle
from data.database import Database
from data.filters import FINANCIAL_KEYWORDS, REDDIT_KEYWORDS

log = get_logger(__name__)

class ClassifierResult(BaseModel):
    event_type: str = "unknown"
    sentiment_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    urgency: str = Field(default="low", pattern="^(low|medium|high|critical)$")
    suggested_direction: str = Field(default="neutral", pattern="^(bullish|bearish|neutral)$")
    affected_sectors: list[str] = Field(default_factory=list)
    affected_tickers: list[str] = Field(default_factory=list)
    countries: list[str] = Field(default_factory=list)
    classification_summary: str = ""

CLASSIFICATION_PROMPT = """
You are a professional financial analyst AI. Analyze the following news article and extract structured information.
Respond ONLY with a valid JSON object matching the requested schema. No markdown formatting, no backticks.

Article Headline: {headline}
Article Summary: {summary}

─── EVENT TYPE TAXONOMY ───
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
    """Classifies NewsArticles using Gemini."""

    def __init__(self, client: Optional[genai.Client] = None, deepseek_client=None, db: Optional[Database] = None):
        self.client = client or get_client()
        self.deepseek_client = deepseek_client or get_deepseek_client()
        self.db = db or Database()
        self.model_name = settings.gemini_model_classifier

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
        if re.search(r'\$[A-Z]{1,5}\b', combined_text):
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

    async def classify(self, article: NewsArticle) -> NewsArticle:
        """
        Classifies the given article and populates its classification fields.
        Returns the modified article.
        """
        if not is_configured() and not is_deepseek_configured():
            log.warning("classifier.skipped", reason="No LLM configured", article_id=article.id)
            return article

        # Apply pre-filter
        if not self.should_classify(article):
            log.info("classifier.pre_filtered", article_id=article.id, headline=article.headline)
            article.event_type = "noise"
            article.sentiment_score = 0.0
            article.urgency = "low"
            article.suggested_direction = "neutral"
            article.classification_summary = "Filtered out by pre-classification noise heuristics."
            return article

        is_reddit = article.source_type == "social" and article.source_name.startswith("reddit")
        
        if is_reddit:
            comments_data = article.raw_data.get("comments", [])
            comments = [
                f"- {c.get('author', '[unknown]')}: {c.get('body', '')}"
                for c in comments_data
                if isinstance(c, dict) and c.get("body")
            ]
            comments_text = "\n".join(comments) if comments else "(No comments)"
            
            prompt = REDDIT_CLASSIFICATION_PROMPT.format(
                headline=article.headline,
                summary=article.summary,
                comments=comments_text
            )
            ds_model_name = settings.deepseek_model_reddit_sentiment
            gemini_model_name = settings.gemini_model_reddit_sentiment
        else:
            prompt = CLASSIFICATION_PROMPT.format(
                headline=article.headline,
                summary=article.summary
            )
            ds_model_name = settings.deepseek_model_classifier
            gemini_model_name = self.model_name

        text = None
        import time

        # DeepSeek attempt
        if is_deepseek_configured() and self.deepseek_client:
            start_time = time.time()
            is_error = False
            error_msg = None
            try:
                # Explicitly disable reasoning/thinking for classification tasks
                # to save tokens — classification needs fast JSON, not deep reasoning
                response = await self.deepseek_client.chat.completions.create(
                    model=ds_model_name,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    extra_body={"thinking": {"type": "disabled"}}
                )
                text = response.choices[0].message.content.strip()
                
                latency_ms = int((time.time() - start_time) * 1000)
                if response.usage:
                    self.db.log_llm_usage(
                        model_name=ds_model_name,
                        operation="classify",
                        prompt_tokens=response.usage.prompt_tokens,
                        candidate_tokens=response.usage.completion_tokens,
                        latency_ms=latency_ms,
                        is_error=False,
                        error_message=None,
                        prompt_text=prompt,
                        response_text=text
                    )
            except Exception as e:
                is_error = True
                error_msg = str(e)
                latency_ms = int((time.time() - start_time) * 1000)
                self.db.log_llm_usage(
                    model_name=ds_model_name,
                    operation="classify",
                    prompt_tokens=0,
                    candidate_tokens=0,
                    latency_ms=latency_ms,
                    is_error=True,
                    error_message=error_msg,
                    prompt_text=prompt,
                    response_text=None
                )
                log.warning("classifier.deepseek_failed", article_id=article.id, error=error_msg, fallback="gemini")
                text = None

        # Gemini fallback attempt
        if text is None and is_configured() and self.client:
            start_time = time.time()
            is_error = False
            error_msg = None
            try:
                response = await self.client.aio.models.generate_content(
                    model=gemini_model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        safety_settings=DEFAULT_SAFETY_SETTINGS,
                        response_mime_type="application/json"
                    )
                )
                text = response.text.strip()
                
                latency_ms = int((time.time() - start_time) * 1000)
                if response.usage_metadata:
                    self.db.log_llm_usage(
                        model_name=gemini_model_name,
                        operation="classify",
                        prompt_tokens=response.usage_metadata.prompt_token_count,
                        candidate_tokens=response.usage_metadata.candidates_token_count,
                        latency_ms=latency_ms,
                        is_error=False,
                        error_message=None,
                        prompt_text=prompt,
                        response_text=text
                    )
            except Exception as e:
                is_error = True
                error_msg = str(e)
                latency_ms = int((time.time() - start_time) * 1000)
                self.db.log_llm_usage(
                    model_name=gemini_model_name,
                    operation="classify",
                    prompt_tokens=0,
                    candidate_tokens=0,
                    latency_ms=latency_ms,
                    is_error=True,
                    error_message=error_msg,
                    prompt_text=prompt,
                    response_text=None
                )
                log.error("classifier.gemini_failed", article_id=article.id, error=error_msg)
                text = None

        if not text:
            return article

        try:
            # Clean up potential markdown formatting just in case
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
                
            parsed = ClassifierResult.model_validate_json(text.strip())
            
            # Update article fields
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
            existing_tickers = set(article.affected_tickers)
            new_tickers = set(parsed.affected_tickers)
            article.affected_tickers = list(existing_tickers | new_tickers)
            
            article.classification_summary = parsed.classification_summary
            
            log.debug("classifier.success", article_id=article.id, event_type=article.event_type)
            
        except Exception as e:
            log.error("classifier.parse_failed", article_id=article.id, error=str(e), text=text)
            
        return article
