import asyncio
from telegram import Bot
from config.settings import settings
from config.logging_config import get_logger

log = get_logger(__name__)

async def test_telegram():
    if not settings.telegram_bot_token or not settings.telegram_chat_id:
        log.error("Missing telegram config")
        return

    bot = Bot(token=settings.telegram_bot_token)
    try:
        await bot.send_message(
            chat_id=settings.telegram_chat_id,
            text="👋 Hello from Deus!\n\nThis is a test message to verify the Telegram Bot configuration. If you received this, the setup was successful! ✅"
        )
        log.info("Test message sent successfully!")
    except Exception as e:
        log.error("Failed to send test message", error=str(e))

if __name__ == "__main__":
    asyncio.run(test_telegram())
