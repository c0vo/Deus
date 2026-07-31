"""
Market Scanner Component

Periodically checks for large price movements in monitored stocks.
If a >5% move is detected, it asks the LLM to explain the move based on recent context.
"""

from __future__ import annotations

import asyncio
import httpx
from typing import Optional

from google.genai import types
from config.logging_config import get_logger
from config.settings import settings
from config.llm import get_client, DEFAULT_SAFETY_SETTINGS
from data.database import Database
from bot.alerts import AlertManager
from bot.formatters import escape_html

log = get_logger(__name__)

DEFAULT_WATCHLIST = ["SPY", "QQQ", "AAPL", "MSFT", "TSLA", "NVDA", "BTC-USD"]

class MarketScanner:
    """Scans the market for significant price movements and generates intelligent alerts."""
    
    def __init__(self, db: Database, alert_manager: Optional[AlertManager] = None):
        self.db = db
        self.alert_manager = alert_manager
        self.client = get_client()

    async def run_scan(self) -> None:
        """Execute a single pass of the market scanner."""
        if not self.alert_manager:
            log.warning("scanner.no_alert_manager")
            return

        tracked = self.db.get_tracked_tickers()
        combined_watchlist = list(set(DEFAULT_WATCHLIST + tracked))
        log.info("scanner.starting", count=len(combined_watchlist))

        async with httpx.AsyncClient(timeout=10) as http_client:
            for ticker in combined_watchlist:
                try:
                    await self._check_ticker(http_client, ticker)
                except Exception as e:
                    log.error("scanner.ticker_failed", ticker=ticker, error=str(e))

    async def _check_ticker(self, http_client: httpx.AsyncClient, ticker: str) -> None:
        """Check a single ticker for a >= 5% move."""
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        params = {"range": "2d", "interval": "1d"}
        headers = {"User-Agent": "Mozilla/5.0"}
        
        resp = await http_client.get(url, params=params, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        
        chart = data.get("chart", {}).get("result", [])
        if not chart:
            return
            
        quote = chart[0].get("indicators", {}).get("quote", [{}])[0]
        closes = quote.get("close", [])
        volumes = quote.get("volume", [])
        
        closes = [c for c in closes if c is not None]
        volumes = [v for v in volumes if v is not None]
        
        if len(closes) < 2:
            return
            
        prev_close = closes[-2]
        current = closes[-1]
        if prev_close == 0:
            return
        diff = current - prev_close
        pct_diff = (diff / prev_close) * 100
        
        if abs(pct_diff) >= 5.0:
            log.info("scanner.trigger", ticker=ticker, pct=pct_diff)
            
            # Check if we already alerted today
            if self.db.was_price_alert_sent_today(ticker):
                log.debug("scanner.already_alerted", ticker=ticker)
                pass
            else:
                await self._generate_and_send_alert(ticker, current, pct_diff, alert_type="price")
                self.db.record_price_alert(ticker)
                
        # Check Anomalous Volume
        if len(volumes) >= 20: # need enough days for SMA
            sma_vol = sum(volumes[:-1]) / len(volumes[:-1])
            current_vol = volumes[-1]
            if sma_vol > 0 and current_vol > (3.0 * sma_vol):
                if not self.db.was_alert_sent(ticker, "anomalous_volume"):
                    await self._generate_and_send_alert(ticker, current, pct_diff, alert_type="volume", vol_multiple=current_vol/sma_vol)
                    self.db.record_alert(ticker, "anomalous_volume")

    async def _generate_and_send_alert(self, ticker: str, current_price: float, pct_diff: float, alert_type: str = "price", vol_multiple: float = 0.0) -> None:
        """Ask LLM for reasoning based on DB context and send the alert."""
        if not self.client:
            return

        # Get recent context for this ticker from DB
        context = ""
        with self.db.connection() as conn:
            rows = conn.execute(
                """
                SELECT a.headline, a.summary, a.classification_summary 
                FROM articles a
                JOIN ticker_mentions tm ON a.id = tm.article_id
                WHERE tm.ticker = ? AND a.published_at >= datetime('now', '-2 days')
                ORDER BY a.published_at DESC 
                LIMIT 10
                """,
                (ticker,)
            ).fetchall()
            
            if rows:
                context = "\n".join([f"- {r['headline']}: {r['classification_summary'] or r['summary']}" for r in rows])
            else:
                context = "No recent specific news found in the database."

        if alert_type == "price":
            emoji = "🚀" if pct_diff > 0 else "🩸"
            direction = "surged" if pct_diff > 0 else "plummeted"
            prompt = (
                f"You are a professional, precise Wall Street analyst.\n"
                f"The stock {ticker} has {direction} by {pct_diff:.2f}% today (current price: ${current_price:.2f}).\n\n"
                f"Recent news context for {ticker}:\n{context}\n\n"
                f"Provide exactly 2 sentences:\n"
                f"1. The single most likely cause of this move, citing a specific news item or event from the context if possible.\n"
                f"2. Whether this move looks sustainable or a short-term overreaction, with brief reasoning.\n"
                f"If the context is empty, say 'No specific news catalyst found — this may be a macro or technical move' and reason from general knowledge.\n"
                f"Use HTML for formatting: <b>bold</b> for key terms, <i>italic</i> for nuance. No emojis, no Markdown."
            )
        else:
            emoji = "🌊"
            prompt = (
                f"You are a professional, precise Wall Street analyst.\n"
                f"The stock {ticker} is trading at {vol_multiple:.1f}x its 30-day average volume today.\n\n"
                f"Recent news context for {ticker}:\n{context}\n\n"
                f"Provide exactly 2 sentences:\n"
                f"1. The most likely reason for this volume spike (earnings, news catalyst, options activity, sector rotation, or technical breakout).\n"
                f"2. Whether this volume suggests institutional accumulation/distribution or retail-driven noise.\n"
                f"If the context is empty, state that and reason from general market patterns.\n"
                f"Use HTML for formatting: <b>bold</b> for key terms, <i>italic</i> for nuance. No emojis, no Markdown."
            )
        try:
            # We use gemini_model_chat for conversational explanations
            loop = asyncio.get_running_loop()

            def ask_llm():
                import time
                start_time = time.time()
                is_error = False
                error_msg = None
                response_text = None
                try:
                    response = self.client.models.generate_content(
                        model=settings.gemini_model_chat,
                        contents=prompt,
                        config={
                            'safety_settings': DEFAULT_SAFETY_SETTINGS,
                            'thinking_config': types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW)
                        }
                    )
                    response_text = response.text
                except Exception as e:
                    is_error = True
                    error_msg = str(e)
                    raise
                finally:
                    latency_ms = int((time.time() - start_time) * 1000)
                    if not is_error and response and response.usage_metadata:
                        self.db.log_llm_usage(
                            model_name=settings.gemini_model_chat,
                            operation="market_scanner",
                            prompt_tokens=response.usage_metadata.prompt_token_count,
                            candidate_tokens=response.usage_metadata.candidates_token_count,
                            latency_ms=latency_ms,
                            is_error=False,
                            error_message=None,
                            prompt_text=prompt,
                            response_text=response_text
                        )
                    elif is_error:
                        self.db.log_llm_usage(
                            model_name=settings.gemini_model_chat,
                            operation="market_scanner",
                            prompt_tokens=0,
                            candidate_tokens=0,
                            latency_ms=latency_ms,
                            is_error=True,
                            error_message=error_msg,
                            prompt_text=prompt,
                            response_text=None
                        )
                return response

            response = await loop.run_in_executor(None, ask_llm)
            
            reasoning = escape_html(response.text.strip())
            reasoning = reasoning.replace("&lt;b&gt;", "<b>").replace("&lt;/b&gt;", "</b>")
            reasoning = reasoning.replace("&lt;i&gt;", "<i>").replace("&lt;/i&gt;", "</i>")
                
            safe_ticker = escape_html(ticker)
            if alert_type == "price":
                message = f"{emoji} <b>MARKET ALERT: {safe_ticker}</b> {emoji}\n\n"
                message += f"<b>{safe_ticker}</b> is {'up' if pct_diff > 0 else 'down'} <b>{abs(pct_diff):.2f}%</b> today to <b>${current_price:.2f}</b>.\n\n"
            else:
                message = f"{emoji} <b>VOLUME ALERT: {safe_ticker}</b> {emoji}\n\n"
                message += f"<b>{safe_ticker}</b> is trading at <b>{vol_multiple:.1f}x</b> its normal average volume.\n\n"
                
            message += f"<i>{reasoning}</i>"
            
            await self.alert_manager.bot.send_message(
                chat_id=self.alert_manager.chat_id,
                text=message,
                parse_mode="HTML"
            )
            log.info("scanner.alert_sent", ticker=ticker, alert_type=alert_type)
            
        except Exception as e:
            log.error("scanner.llm_failed", error=str(e))

    async def check_earnings(self) -> None:
        """Query Finnhub for upcoming earnings in the next 1 trading day."""
        from datetime import datetime, timedelta

        # settings reads .env through pydantic-settings, which does NOT export
        # into os.environ — os.getenv here returned None on every run, so this
        # alert has never fired.
        api_key = settings.finnhub_api_key
        if not api_key or not self.alert_manager:
            return
            
        tracked = self.db.get_tracked_tickers()
        combined_watchlist = list(set(DEFAULT_WATCHLIST + tracked))
        
        # We want earnings for tomorrow (or Monday if it's Friday)
        today = datetime.now()
        tomorrow = today + timedelta(days=1)
        if today.weekday() == 4: # Friday
            tomorrow = today + timedelta(days=3)
        
        date_str = tomorrow.strftime("%Y-%m-%d")
        
        async with httpx.AsyncClient(timeout=10) as http_client:
            for ticker in combined_watchlist:
                try:
                    url = f"https://finnhub.io/api/v1/calendar/earnings?from={date_str}&to={date_str}&symbol={ticker}&token={api_key}"
                    resp = await http_client.get(url)
                    resp.raise_for_status()
                    data = resp.json()
                    
                    earnings = data.get("earningsCalendar", [])
                    if earnings:
                        # Found an earnings event exactly tomorrow
                        if not self.db.was_alert_sent(f"{ticker}_{date_str}", "earnings_whisper"):
                            await self._send_earnings_whisper(ticker, date_str)
                            self.db.record_alert(f"{ticker}_{date_str}", "earnings_whisper")
                except Exception as e:
                    log.error("scanner.earnings_failed", ticker=ticker, error=str(e))

    async def _send_earnings_whisper(self, ticker: str, date_str: str) -> None:
        """Sends an earnings whisper alert generated by the LLM."""
        if not self.client:
            return

        context = ""
        with self.db.connection() as conn:
            rows = conn.execute(
                """
                SELECT a.headline, a.summary, a.classification_summary 
                FROM articles a
                JOIN ticker_mentions tm ON a.id = tm.article_id
                WHERE tm.ticker = ? AND a.published_at >= datetime('now', '-14 days')
                ORDER BY a.published_at DESC 
                LIMIT 15
                """,
                (ticker,)
            ).fetchall()
            
            if rows:
                context = "\n".join([f"- {r['headline']}: {r['classification_summary'] or r['summary']}" for r in rows])
            else:
                context = "No specific news found in the database over the last two weeks."

        prompt = (
            f"You are a professional, precise Wall Street analyst writing an earnings preview.\n"
            f"The stock {ticker} reports earnings tomorrow ({date_str}).\n\n"
            f"Recent news context (last 14 days) for {ticker}:\n{context}\n\n"
            f"Write a concise 'Whisper Alert' covering:\n"
            f"1. Overall market sentiment heading into this print (bullish/bearish/anxious/mixed) — with a verbatim quote from the context if available.\n"
            f"2. The 1-2 key metrics or themes the market will be watching most closely.\n"
            f"3. Whether expectations seem realistic or stretched based on the news flow.\n"
            f"Use HTML for formatting: <b>bold</b> for key terms, <i>italic</i> for nuance. No emojis, no Markdown."
        )
        try:
            loop = asyncio.get_running_loop()

            def ask_llm():
                import time
                start_time = time.time()
                is_error = False
                error_msg = None
                response_text = None
                try:
                    response = self.client.models.generate_content(
                        model=settings.gemini_model_chat,
                        contents=prompt,
                        config={
                            'safety_settings': DEFAULT_SAFETY_SETTINGS,
                            'thinking_config': types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW)
                        }
                    )
                    response_text = response.text
                except Exception as e:
                    is_error = True
                    error_msg = str(e)
                    raise
                finally:
                    latency_ms = int((time.time() - start_time) * 1000)
                    if not is_error and response and response.usage_metadata:
                        self.db.log_llm_usage(
                            model_name=settings.gemini_model_chat,
                            operation="earnings_whisper",
                            prompt_tokens=response.usage_metadata.prompt_token_count,
                            candidate_tokens=response.usage_metadata.candidates_token_count,
                            latency_ms=latency_ms,
                            is_error=False,
                            error_message=None,
                            prompt_text=prompt,
                            response_text=response_text
                        )
                    elif is_error:
                        self.db.log_llm_usage(
                            model_name=settings.gemini_model_chat,
                            operation="earnings_whisper",
                            prompt_tokens=0,
                            candidate_tokens=0,
                            latency_ms=latency_ms,
                            is_error=True,
                            error_message=error_msg,
                            prompt_text=prompt,
                            response_text=None
                        )
                return response

            response = await loop.run_in_executor(None, ask_llm)
            reasoning = escape_html(response.text.strip())
            reasoning = reasoning.replace("&lt;b&gt;", "<b>").replace("&lt;/b&gt;", "</b>")
            reasoning = reasoning.replace("&lt;i&gt;", "<i>").replace("&lt;/i&gt;", "</i>")
            
            safe_ticker = escape_html(ticker)
            message = f"🗣️ <b>EARNINGS WHISPER: {safe_ticker}</b> 🗣️\n\n"
            message += f"<b>{safe_ticker}</b> reports earnings tomorrow ({date_str}). Here is the read on the room:\n\n"
            message += f"<i>{reasoning}</i>"
            
            await self.alert_manager.bot.send_message(
                chat_id=self.alert_manager.chat_id,
                text=message,
                parse_mode="HTML"
            )
            log.info("scanner.earnings_whisper_sent", ticker=ticker)
            
        except Exception as e:
            log.error("scanner.whisper_failed", error=str(e))
