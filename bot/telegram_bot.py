"""
Project Scrooge V2 — Telegram Bot Setup

Initializes the python-telegram-bot Application, registers handlers,
and manages the bot lifecycle.
"""

from __future__ import annotations

from typing import Optional

from telegram.ext import Application, CommandHandler, MessageHandler, filters

from config.logging_config import get_logger
from config.settings import settings
from bot.commands import (
    start_command, help_command, status_command, trending_command, usage_command,
    markets_command, track_command, untrack_command, briefing_command, handle_query,
    predict_command, accuracy_command,
    sectors_command, ipos_command, events_command, themes_command, forecast_command,
)
from bot.alerts import AlertManager
from data.database import Database

log = get_logger(__name__)

class ScroogeBot:
    """Manages the Telegram Bot application and alerts."""
    
    def __init__(self, db: Database):
        self.db = db
        self.token = settings.telegram_bot_token
        self.application: Optional[Application] = None
        self.alert_manager: Optional[AlertManager] = None
        
    def initialize(self):
        """Builds the Application and registers handlers."""
        if not self.token:
            log.warning("telegram.missing_token", reason="No bot token in config")
            return
            
        # Build the application
        self.application = Application.builder().token(self.token).build()
        self.application.bot_data["db"] = self.db
        
        # Initialize AlertManager
        self.alert_manager = AlertManager(self.db, self.application.bot)
        
        # Register command handlers
        self.application.add_handler(CommandHandler("start", start_command))
        self.application.add_handler(CommandHandler("help", help_command))
        self.application.add_handler(CommandHandler("status", status_command))
        self.application.add_handler(CommandHandler("trending", trending_command))
        self.application.add_handler(CommandHandler("usage", usage_command))
        self.application.add_handler(CommandHandler("markets", markets_command))
        self.application.add_handler(CommandHandler("track", track_command))
        self.application.add_handler(CommandHandler("untrack", untrack_command))
        self.application.add_handler(CommandHandler("predict", predict_command))
        self.application.add_handler(CommandHandler("accuracy", accuracy_command))
        
        # Natural language queries (catch-all text messages)
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_query))
        
        # Add /briefing
        self.application.add_handler(CommandHandler("briefing", briefing_command))

        # New intelligence commands
        self.application.add_handler(CommandHandler("sectors", sectors_command))
        self.application.add_handler(CommandHandler("ipos", ipos_command))
        self.application.add_handler(CommandHandler("events", events_command))
        self.application.add_handler(CommandHandler("themes", themes_command))
        self.application.add_handler(CommandHandler("forecast", forecast_command))
        
        log.info("telegram.initialized")
        
    async def start(self):
        """Starts the bot in polling mode (non-blocking if managed externally)."""
        if not self.application:
            return
            
        await self.application.initialize()
        
        # Register commands with Telegram for autocomplete menu
        try:
            from telegram import BotCommand
            commands = [
                BotCommand("start", "Initialize your profile"),
                BotCommand("help", "Show help menu"),
                BotCommand("briefing", "Get an immediate daily market briefing"),
                BotCommand("trending", "See the most discussed tickers"),
                BotCommand("markets", "View live market performance & charts"),
                BotCommand("chart", "Generate a price/sentiment chart"),
                BotCommand("predict", "ML prediction (e.g. /predict AAPL)"),
                BotCommand("accuracy", "View prediction accuracy"),
                BotCommand("track", "Add to watchlist"),
                BotCommand("untrack", "Remove from watchlist"),
                BotCommand("status", "View system status"),
                BotCommand("usage", "View API token costs"),
                BotCommand("sectors", "Sector sentiment heatmap"),
                BotCommand("ipos", "IPO watchlist"),
                BotCommand("events", "Upcoming earnings & events"),
                BotCommand("themes", "Current macro themes"),
                BotCommand("forecast", "Sector outlook (e.g. /forecast Technology)"),
            ]
            await self.application.bot.set_my_commands(commands)
            log.info("telegram.commands_set")
        except Exception as e:
            log.warning("telegram.commands_set_failed", error=str(e))
            
        await self.application.start()
        await self.application.updater.start_polling()
        log.info("telegram.polling_started")
        
    async def stop(self):
        """Stops the bot polling."""
        if not self.application:
            return
            
        await self.application.updater.stop()
        await self.application.stop()
        await self.application.shutdown()
        log.info("telegram.polling_stopped")
