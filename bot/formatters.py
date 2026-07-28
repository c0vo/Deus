"""
Telegram Message Formatters

Utility functions to format NewsArticle objects into visually appealing
HTML messages for Telegram, avoiding markdown parsing issues.
"""

from __future__ import annotations

from html import escape

from data.models import NewsArticle

def escape_html(text: str) -> str:
    """Escape HTML special characters for Telegram HTML parse mode."""
    return escape(str(text or ""), quote=True)

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

