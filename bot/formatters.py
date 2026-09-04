"""
Telegram Message Formatters

Utility functions to format NewsArticle objects into visually appealing
HTML messages for Telegram, avoiding markdown parsing issues.
"""

from __future__ import annotations

from html import escape

from data.models import NewsArticle

# Sent when nothing clears the briefing importance floor. A quiet day is signal,
# so the brief says so rather than going silent — silence is indistinguishable
# from the job having failed.
EMPTY_BRIEFING_TEXT = "📰 No high-impact stories in the last 24h."

# Per-article summary budget. Six articles at full classification_summary
# length plus long google_news URLs runs to ~5000 characters, over Telegram's
# 4096 limit; truncating here keeps the common case to a single message.
BRIEFING_SUMMARY_MAX = 220

# Telegram's hard limit is 4096. The margin absorbs the HTML markup that
# escaping can add after a chunk boundary has already been chosen.
TELEGRAM_CHUNK_LIMIT = 3800

def escape_html(text: str) -> str:
    """Escape HTML special characters for Telegram HTML parse mode."""
    return escape(str(text or ""), quote=True)

def chunk_html(text: str, limit: int = TELEGRAM_CHUNK_LIMIT) -> list[str]:
    """
    Split a message on paragraph boundaries so each part fits Telegram's limit.

    Splitting on blank lines keeps HTML tags intact, since the renderers here
    never open a tag in one paragraph and close it in the next.
    """
    chunks: list[str] = []
    chunk = ""

    for paragraph in text.split("\n\n"):
        # A single paragraph over the limit cannot be placed by the normal path,
        # so hard-split it. Without this the paragraph goes out whole and
        # Telegram rejects the send.
        if len(paragraph) > limit:
            if chunk.strip():
                chunks.append(chunk.strip())
                chunk = ""
            for i in range(0, len(paragraph), limit):
                chunks.append(paragraph[i : i + limit])
            continue

        if len(chunk) + len(paragraph) > limit and chunk.strip():
            chunks.append(chunk.strip())
            chunk = ""
        chunk += paragraph + "\n\n"

    if chunk.strip():
        chunks.append(chunk.strip())

    return chunks

def render_briefing(
    lanes: list[tuple[str, list[dict]]], title: str = "📰 Daily Market Briefing"
) -> str:
    """
    Render lane-grouped articles as a Telegram HTML brief.

    Takes the output of data.taxonomy.select_briefing_lanes. Shared by the
    scheduled 05:00 push and the /briefing command so the two cannot drift.
    """
    text = f"<b>{escape_html(title)}</b>\n\n"

    for label, articles in lanes:
        text += f"<b>--- {escape_html(label)} ---</b>\n"
        for row in articles:
            score = row.get("importance_score") or 0.0
            # classification_summary is null for articles ranked before they
            # were classified; falling through to summary avoids rendering the
            # literal string "None".
            summary = row.get("classification_summary") or row.get("summary") or ""
            # Truncate before escaping: escape_html can sextuple a character,
            # so trimming afterwards would not bound the result.
            if len(summary) > BRIEFING_SUMMARY_MAX:
                summary = summary[:BRIEFING_SUMMARY_MAX].rstrip() + "…"

            text += f"🔹 <b>{escape_html(row.get('headline'))}</b> ({score:.1f})\n"
            if summary:
                text += f"<i>{escape_html(summary)}</i>\n"
            text += f"<a href='{escape_html(row.get('url'))}'>Read more</a>\n\n"

    return text.rstrip() + "\n"

def format_article_alert(article: NewsArticle) -> str:
    """Format a single critical article for an immediate alert."""
    
    # Determine emoji based on event type and urgency
    urgency_emoji = {
        "critical": "🚨",
        "high": "⚠️",
        "medium": "🔔",
        "low": "📰"
    }.get(article.urgency, "📰")
    
    direction_emoji = {
        "bullish": "📈",
        "bearish": "📉",
        "neutral": "⚖️"
    }.get(article.suggested_direction, "")
    
    headline = escape_html(article.headline)
    summary = escape_html(article.classification_summary or article.summary[:200])
    
    text = f"<b>{urgency_emoji} BREAKING NEWS: {headline}</b>\n\n"
    text += f"<i>{summary}</i>\n\n"
    
    # Optional fields if they exist
    if article.sentiment_score is not None:
        text += f"<b>Sentiment:</b> {article.sentiment_score:.2f} {direction_emoji}\n"
    
    if article.affected_tickers:
        tickers = ", ".join(f"${escape_html(t)}" for t in article.affected_tickers)
        text += f"<b>Affected Tickers:</b> {tickers}\n"
        
    text += f"\n<a href='{escape_html(article.url)}'>Read Full Article</a>"
    
    return text

