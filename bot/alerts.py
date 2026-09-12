"""
Telegram Critical Alerts

Logic to evaluate articles and send immediate alerts for critical news,
avoiding spam by utilizing the sent_alerts table.
"""

from __future__ import annotations

import asyncio

from telegram import Bot

from api.sse_manager import event_bus
from config.logging_config import get_logger
from config.settings import settings
from data.database import Database
from data.models import NewsArticle
from bot.formatters import chunk_html, format_article_alert

log = get_logger(__name__)

class AlertManager:
    """Handles sending push notifications for critical articles."""
    
    def __init__(self, db: Database, bot: Bot):
        self.db = db
        self.bot = bot
        self.chat_id = settings.telegram_chat_id
        
    async def process_for_alerts(self, articles: list[NewsArticle]):
        """
        Evaluate a list of recently classified/ranked articles and send
        alerts for any that meet the critical threshold.
        """
        if not self.bot or not self.chat_id:
            return
            
        for article in articles:
            # Criteria for a critical alert:
            # 1. LLM marked urgency as critical OR
            # 2. Importance score >= 9.0
            
            is_critical = (
                article.urgency in ["high", "critical"] and 
                article.importance_score is not None and 
                article.importance_score >= 9.0
            )
            
            if is_critical:
                # Check if we already alerted for this article
                if not self.db.was_alert_sent(article.id, "critical"):
                    await self._send_alert(article)
                    
    async def send_html(self, text: str, *, disable_preview: bool = True) -> None:
        """
        Send one HTML message, split across Telegram's 4096-character limit.

        Every sender of a long message needs this, and each one used to either
        hand-roll the chunk loop or skip it and have Telegram reject the send
        whole — which is how a busy news day silently produced no daily brief.
        Callers remain responsible for checking the bot is configured at all.
        """
        for chunk in chunk_html(text):
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=chunk,
                parse_mode="HTML",
                disable_web_page_preview=disable_preview,
            )

    async def _send_alert(self, article: NewsArticle):
        """Sends the actual alert to Telegram, records it, and publishes it."""
        try:
            text = format_article_alert(article)

            # Previews stay ON here, unlike every other sender: a breaking
            # alert is one headline and one link, and the preview is the point.
            await self.send_html(text, disable_preview=False)

            # Record it so we don't spam
            self.db.record_alert(article.id, "critical")
            log.info("alert.sent", article_id=article.id)

            await self._persist_and_publish(article, text)

        except Exception as e:
            log.error("alert.send_failed", article_id=article.id, error=str(e))

    async def _persist_and_publish(self, article: NewsArticle, text: str) -> None:
        """
        Store the breaking alert and push it onto the `alert` SSE topic.

        Breaking news used to exist only in Telegram history, which means an
        alert that fired while the phone was asleep was unrecoverable. It now
        lands in the same table and on the same topic as the price alerts, so
        one dashboard card covers everything that was ever pushed.

        Deliberately after the send and inside its own try: the message is the
        product, and a storage or bus failure must not make a delivered alert
        look like a failed one.
        """
        row = {
            "ticker": (article.affected_tickers or [None])[0],
            "kind": "breaking",
            "pct": None,
            "price": None,
            "severity": article.urgency,
            "title": article.headline,
            "summary": article.classification_summary or (article.summary or "")[:300],
            "body_html": text,
            "sources_json": [{
                "title": article.headline,
                "url": article.url,
                "source": article.source_name,
                "published_at": str(article.published_at)[:10],
                "kind": "db",
            }],
            "grounded_by": "db",
        }
        try:
            alert_id = await asyncio.to_thread(self.db.insert_alert, row)
            stored = await asyncio.to_thread(self.db.get_alert, alert_id)
        except Exception as e:
            log.warning("alert.persist_failed", article_id=article.id, error=str(e))
            return
        try:
            await event_bus.publish("alert", stored or row)
        except Exception as e:
            log.warning("alert.publish_failed", article_id=article.id, error=str(e))
