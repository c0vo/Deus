"""
Deus — News Article Data Models

Pydantic models representing news articles at various pipeline stages.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator
import hashlib


class NewsArticle(BaseModel):
    """
    A single news article as it flows through the pipeline.

    Fields are populated progressively:
    - Source adapter fills: id, headline, summary, source_name, source_type, url, published_at
    - Classifier fills: event_type, sentiment_score, urgency, etc.
    - Ranker fills: importance_score
    - Embedder fills: embedding
    """

    # ── Core fields (populated by source adapters) ───────────────────────
    id: str = Field(..., description="Unique ID, e.g. 'rss_reuters_abc123'")
    headline: str = Field(..., description="Article title/headline")
    summary: str = Field(default="", description="Article body or summary text")
    content_hash: str = Field(default="", description="SHA256 hash of headline + summary for strict deduplication if needed")
    source_name: str = Field(..., description="Origin source, e.g. 'reuters', 'finnhub'")
    source_type: str = Field(
        ..., description="Source category: 'rss', 'api', 'social', 'scrape'"
    )
    url: str = Field(..., description="Article URL (used for deduplication)")
    published_at: datetime = Field(..., description="Publication timestamp")
    fetched_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When we fetched this article",
    )

    # ── Classification fields (populated by Gemini classifier) ───────────
    event_type: Optional[str] = Field(
        default=None,
        description="Event category: earnings, geopolitical, macro, regulatory, meme, etc.",
    )
    sentiment_score: Optional[float] = Field(
        default=None, description="Sentiment from -1.0 (bearish) to +1.0 (bullish)"
    )
    urgency: Optional[str] = Field(
        default=None, description="Urgency level: low, medium, high, critical"
    )
    suggested_direction: Optional[str] = Field(
        default=None, description="Market direction: bullish, bearish, neutral"
    )
    affected_sectors: list[str] = Field(
        default_factory=list, description="Affected market sectors"
    )
    affected_tickers: list[str] = Field(
        default_factory=list, description="Affected ticker symbols"
    )
    countries: list[str] = Field(
        default_factory=list,
        description="ISO 3166-1 alpha-2 codes for the countries the story concerns",
    )
    classification_summary: Optional[str] = Field(
        default=None, description="LLM one-line summary of market impact"
    )

    # ── Ranking fields (populated by batch ranker) ───────────────────────
    importance_score: Optional[float] = Field(
        default=None, description="Importance score 0.0-10.0 from LLM ranking"
    )

    # ── Raw data (for debugging) ─────────────────────────────────────────
    raw_data: dict = Field(
        default_factory=dict, description="Original raw data from source API/feed"
    )

    # Allow arbitrary types for potential future numpy array fields
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    @model_validator(mode="after")
    def compute_content_hash(self) -> 'NewsArticle':
        if not self.content_hash:
            content = f"{self.headline}{self.summary}".encode("utf-8")
            self.content_hash = hashlib.sha256(content).hexdigest()
        return self


# The element type for every batch "summarise these tickers" call — trending
# summaries and the daily advisor.
#
# Those prompts used to ask for a JSON object keyed by ticker, which cannot be
# expressed as a response schema: there is no way to say "an object whose keys
# are arbitrary strings" that Gemini's structured output accepts. A list of
# these is expressible, so the call can be schema-constrained; the caller
# rebuilds the dict with `notes_to_dict`.
#
# Docstring and field descriptions are sent to the model as the schema
# `description` on every call, so both are kept to one line.
class TickerNote(BaseModel):
    """One ticker plus one line of commentary."""

    ticker: str = Field(description="The ticker symbol, exactly as given in the prompt.")
    summary: str = Field(description="Plain-text commentary. No markdown, no HTML, no emojis.")


def notes_to_dict(notes: list) -> dict[str, str]:
    """
    Collapse a TickerNote list back into the {ticker: summary} shape callers use.

    Accepts raw dicts as well as models: `response.parsed` is only typed as
    `BaseModel | dict | Enum | None`, so a list schema is not guaranteed to come
    back as model instances on every SDK version.
    """
    out: dict[str, str] = {}
    for n in notes:
        ticker = (n.get("ticker") if isinstance(n, dict) else getattr(n, "ticker", None)) or ""
        summary = (n.get("summary") if isinstance(n, dict) else getattr(n, "summary", None)) or ""
        if ticker.strip():
            out[ticker.strip().upper()] = summary
    return out
