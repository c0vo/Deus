import asyncio
from datetime import datetime, timezone
from data.models import NewsArticle
from bot.alerts import AlertManager
from data.database import Database
from bot.telegram_bot import ScroogeBot

async def send_test_alert():
    db = Database()
    db.initialize()
    bot = ScroogeBot(db=db)
    bot.initialize()
    
    if not bot.application:
        print("Bot application not initialized.")
        return
        
    await bot.application.initialize()
    
    # Create a fake critical article
    article = NewsArticle(
        id="test_alert_001",
        headline="BREAKING: Central Bank Announces Emergency Rate Cut",
        summary="In a surprise move, the central bank has announced an immediate 100 basis point cut to interest rates to combat sudden market volatility.",
        source_name="Reuters",
        source_type="news",
        url="https://example.com/breaking-news",
        published_at=datetime.now(timezone.utc),
        event_type="macro",
        sentiment_score=0.85,
        urgency="critical",
        suggested_direction="bullish",
        affected_sectors=["Financials", "Broad Market"],
        affected_tickers=["SPY", "QQQ"],
        classification_summary="Massive unexpected liquidity injection likely to drive equities significantly higher in the short term.",
        importance_score=9.5
    )
    
    print("Sending critical alert...")
    await bot.alert_manager.process_for_alerts([article])
    print("Alert sent. Check your Telegram!")

if __name__ == "__main__":
    asyncio.run(send_test_alert())
