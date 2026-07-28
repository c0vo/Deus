"""
Telegram Critical Alerts

Logic to evaluate articles and send immediate alerts for critical news,
avoiding spam by utilizing the sent_alerts table.
"""

from __future__ import annotations

from telegram import Bot

from config.logging_config import get_logger
from config.settings import settings
from data.database import Database
from data.models import NewsArticle
from bot.formatters import format_article_alert

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
                    
    async def _send_alert(self, article: NewsArticle):
        """Sends the actual alert to Telegram and records it in DB."""
        try:
            text = format_article_alert(article)
            
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode="HTML",
                disable_web_page_preview=False
            )
            
            # Record it so we don't spam
            self.db.record_alert(article.id, "critical")
            log.info("alert.sent", article_id=article.id)
            
        except Exception as e:
            log.error("alert.send_failed", article_id=article.id, error=str(e))
