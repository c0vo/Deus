"""
Telegram Bot Commands

Handlers for commands like /start, /status, /trending.
"""

from __future__ import annotations

import asyncio
import datetime
import html
import json
import re
import time

import httpx
from telegram import Update
from telegram.ext import ContextTypes

from google.genai import types
from config.logging_config import get_logger
from config.settings import settings
from data.database import Database
from bot.formatters import escape_html

log = get_logger(__name__)
fallback_db = Database()


def get_db(context: ContextTypes.DEFAULT_TYPE) -> Database:
    """Return the bot's shared Database instance, falling back for direct tests."""
    if context and context.application:
        return context.application.bot_data.get("db", fallback_db)
    return fallback_db

async def auth_middleware(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Ensure the user is the authorized owner before processing."""
    if not update.effective_user:
        return False
        
    user_id = str(update.effective_user.id)
    if user_id != settings.telegram_chat_id:
        log.warning("telegram.unauthorized_access", user_id=user_id)
        await update.message.reply_text("⛔ Unauthorized. You are not the owner of this bot.")
        return False
    return True

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command by forwarding to /help."""
    await help_command(update, context)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command."""
    if not await auth_middleware(update, context):
        return

    help_text = (
        "<b>Help Menu</b>\n\n"
        "Here are the commands you can use to interact with me:\n"
        "• /start - Initialize your profile\n"
        "• /help - Show this menu\n"
        "• /briefing - Get an immediate daily market briefing (e.g. <code>/briefing</code>)\n"
        "• /trending [HOURS] - See the most discussed tickers (e.g. <code>/trending 24</code>)\n"
        "• /markets - View live market performance and charts for tracked tickers (e.g. <code>/markets</code>)\n"
        "• /chart &lt;TICKER&gt; - Generate a price/sentiment chart (e.g. <code>/chart AAPL</code>)\n"
        "• /predict &lt;TICKER&gt; [HORIZON] - ML prediction (e.g. <code>/predict NVDA 3d</code>)\n"
        "• /accuracy [TICKER] - View prediction accuracy (e.g. <code>/accuracy AAPL</code>)\n"
        "• /track &lt;TICKER&gt; - Add to watchlist (e.g. <code>/track MSFT</code>)\n"
        "• /untrack &lt;TICKER&gt; - Remove from watchlist (e.g. <code>/untrack MSFT</code>)\n"
        "• /status - View system usage statistics (e.g. <code>/status</code>)\n"
        "• /usage - View API token costs (e.g. <code>/usage</code>)\n\n"
        "<b>Proactive Features:</b>\n"
        "• <b>Breaking News Alerts</b>: I will automatically alert you to highly urgent news affecting your tickers.\n"
        "• <b>05:00 AM Daily Briefing</b>: You will receive an automated briefing every morning.\n"
        "• <b>Anomalous Volume Scanner</b>: I monitor your watchlist for unusual trading volume (>3x 30-day average).\n"
        "• <b>Earnings Whispers</b>: I send sentiment predictions 1 day before any of your tickers report earnings.\n"
        "• <b>Weekly Review</b>: Expect a comprehensive portfolio recap every Friday at 6:00 PM KST.\n\n"
        "Just ask me any natural language question about the market to get started!"
    )
    await update.message.reply_html(help_text)

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /status command."""
    if not await auth_middleware(update, context):
        return

    try:
        db = get_db(context)
        stats = db.get_stats()
        
        status_text = "<b>⚙️ System Status</b>\n\n"
        status_text += f"📦 DB Size: {stats['db_size_mb']} MB\n"
        status_text += f"📰 Total Articles: {stats['total_articles']}\n"
        status_text += f"🧠 Classified: {stats['classified_articles']}\n"
        status_text += f"🗑️ Noise (Skipped): {stats.get('noise_articles', 0)}\n"
        status_text += f"🔢 Embedded: {stats['embedded_articles']}\n\n"
        
        status_text += "<b>Sources Overview:</b>\n"
        for s in stats['sources']:
            status_text += f"- {s['source_name']}: {s['c']} articles\n"
            
        await update.message.reply_html(status_text)
        
    except Exception as e:
        log.error("telegram.status_failed", error=str(e))
        await update.message.reply_text("❌ Failed to fetch system status.")

async def usage_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /usage command."""
    if not await auth_middleware(update, context):
        return

    try:
        db = get_db(context)
        usage = db.get_usage_stats(days=7)

        text = "<b>💸 LLM API Usage & Costs</b>\n\n"
        text += f"<b>Last 7 Days:</b> {usage['total_tokens']:,} tokens / ${usage['total_cost_usd']:.4f}\n"
        text += f"<b>All Time:</b> {usage['all_time_tokens']:,} tokens / ${usage['all_time_cost_usd']:.4f}\n\n"

        if usage.get('details'):
            text += "<b>Last 7 Days Breakdown:</b>\n"
            for d in usage['details']:
                avg_lat = d.get('avg_latency_ms') or 0
                err_count = d.get('error_count') or 0
                reqs = d.get('requests_count') or 0
                err_str = f" | ⚠️ {err_count} errs" if err_count > 0 else ""
                
                text += f"• <b>{d['day']}</b> | <code>{d['model_name']}</code> (<i>{d['operation']}</i>)\n"
                text += f"  Cost: ${d['cost']:.4f} | Tokens: {d['tokens']:,}\n"
                text += f"  Reqs: {reqs} | Avg Latency: {int(avg_lat)}ms{err_str}\n\n"
        else:
            text += "<i>No usage data recorded in the last 7 days.</i>\n\n"
            
        text += "\n<i>Based on standard Gemini API pricing</i>"
            
        await update.message.reply_html(text)
        
    except Exception as e:
        log.error("telegram.usage_failed", error=str(e))
        await update.message.reply_text("❌ Failed to fetch usage stats.")

async def trending_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /trending command with AI summaries."""
    if not await auth_middleware(update, context):
        return

    try:
        db = get_db(context)
        
        # Parse hours argument if provided
        hours = 24
        if context.args:
            try:
                hours = int(context.args[0])
            except ValueError:
                pass
                
        trending_tickers = db.get_top_trending_tickers(hours=hours, limit=15)
        
        if not trending_tickers:
            await update.message.reply_text(f"No trending tickers found in the last {hours} hours.")
            return
            
        await update.message.reply_text(f"🔄 Fetching top trending tickers (last {hours}h) and generating AI summaries...")
            
        text = f"<b>📈 Top 15 Trending Tickers (Last {hours}h)</b>\n\n"
        
        from config.llm import get_client, parse_structured, DEFAULT_SAFETY_SETTINGS
        from data.models import TickerNote, notes_to_dict
        client = get_client()
        loop = asyncio.get_running_loop()
        
        # 1. Collect all context first
        all_ticker_contexts = {}
        for t in trending_tickers:
            ticker_name = t['ticker']
            summaries = db.get_recent_summaries_for_ticker(ticker_name, hours=hours)
            if summaries:
                all_ticker_contexts[ticker_name] = summaries
                    
        # 2. Make a single batch LLM call
        ai_summaries = {}
        if client and all_ticker_contexts:
            prompt = (
                f"You are a Professional, precise, and highly analytical Wall Street analyst.\n"
                f"Below is a list of trending tickers and their recent news summaries.\n"
                f"For EACH ticker, write a concise 1-sentence explanation of exactly why it is trending based ONLY on the context.\n"
                f"You MUST include an exact quote from the context if available. Do NOT use emojis.\n"
                f"Do NOT use markdown or HTML tags in your summaries. Just plain text.\n"
                f"Return one entry per ticker in the required response schema.\n\n"
            )
            for tk, sums in all_ticker_contexts.items():
                prompt += f"Ticker: {tk}\nContext:\n" + "\n".join(f"- {s}" for s in sums) + "\n\n"
                
            def ask_batch_llm():
                start_time = time.time()
                is_error = False
                error_msg = None
                response_text = None
                try:
                    response = client.models.generate_content(
                        model=settings.gemini_model_chat,
                        contents=prompt,
                        config={
                            'safety_settings': DEFAULT_SAFETY_SETTINGS,
                            'response_mime_type': 'application/json',
                            'response_schema': list[TickerNote],
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
                        db.log_llm_usage(
                            model_name=settings.gemini_model_chat,
                            operation="trending_summary_batch",
                            prompt_tokens=response.usage_metadata.prompt_token_count,
                            candidate_tokens=response.usage_metadata.candidates_token_count,
                            latency_ms=latency_ms,
                            is_error=False,
                            error_message=None,
                            prompt_text=prompt,
                            response_text=response_text
                        )
                    elif is_error:
                        db.log_llm_usage(
                            model_name=settings.gemini_model_chat,
                            operation="trending_summary_batch",
                            prompt_tokens=0,
                            candidate_tokens=0,
                            latency_ms=latency_ms,
                            is_error=True,
                            error_message=error_msg,
                            prompt_text=prompt,
                            response_text=None
                        )
                if isinstance(response.parsed, list):
                    return notes_to_dict(response.parsed)
                return notes_to_dict(parse_structured(response.text, list[TickerNote]))

            try:
                ai_summaries = await loop.run_in_executor(None, ask_batch_llm)
            except Exception as e:
                log.error("trending.batch_summary_failed", error=str(e))

        # 3. Construct the message
        all_tickers = []
        all_sentiments = []
        for i, t in enumerate(trending_tickers, 1):
            avg_sent = t['avg_sentiment'] or 0.0
            emoji = "🟢" if avg_sent > 0.2 else "🔴" if avg_sent < -0.2 else "⚪"
            ticker_name = t['ticker']
            text += f"<b>{i}. ${ticker_name}</b> - {t['mention_count']} mentions {emoji}\n"
            
            if ticker_name in all_ticker_contexts:
                summary = ai_summaries.get(ticker_name.upper())
                if summary:
                    text += f"   <i>Summary:</i> {escape_html(summary)}\n\n"
                else:
                    text += "   <i>(Summary failed)</i>\n\n"
            else:
                text += "   <i>(No news context)</i>\n\n"
                
            all_tickers.append(ticker_name)
            all_sentiments.append(avg_sent)
                
        # Send text first to avoid 1024 char caption limit
        await update.message.reply_html(text)
        
        # Generate composite sentiment chart
        if all_tickers:
            try:
                from bot.visualizations import get_sentiment_chart
                chart_buffer = await get_sentiment_chart("Market Sentiment", all_sentiments, all_tickers)
                await update.message.reply_photo(photo=chart_buffer)
            except Exception as e:
                log.error("trending.chart_failed", error=str(e))
        
    except Exception as e:
        log.error("telegram.trending_failed", error=str(e))
        await update.message.reply_text("❌ Failed to fetch trending tickers.")


async def markets_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /markets command."""
    if not await auth_middleware(update, context):
        return

    try:
        db = get_db(context)
        tickers = db.get_tracked_tickers()
        if not tickers:
            await update.message.reply_text("📉 No tickers tracked. Use /track <ticker> to add some.")
            return
            
        await update.message.reply_text("🔄 Fetching live market data...")
        
        sector_results = {}
        pct_diffs = []
        valid_tickers = []
        
        async with httpx.AsyncClient(timeout=10) as client:
            for t in tickers:
                sector = db.get_ticker_sector(t)
                if sector not in sector_results:
                    sector_results[sector] = []
                try:
                    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{t}"
                    params = {"range": "2d", "interval": "1d"}
                    headers = {"User-Agent": "Mozilla/5.0"}
                    resp = await client.get(url, params=params, headers=headers)
                    data = resp.json()
                    
                    chart = data["chart"]["result"][0]
                    closes = chart["indicators"]["quote"][0]["close"]
                    closes = [c for c in closes if c is not None]
                    
                    if len(closes) >= 2:
                        prev_close = closes[-2]
                        current = closes[-1]
                        diff = current - prev_close
                        pct_diff = (diff / prev_close) * 100
                        emoji = "📈" if diff >= 0 else "📉"
                        sign = "+" if diff >= 0 else ""
                        
                        import datetime as dt_module
                        today = dt_module.date.today().isoformat()
                        
                        from pipeline.predictor import StockPredictor
                        predictor = StockPredictor(db)
                        
                        pred_strs = []
                        for h in [5, 21, 63, 252]:
                            pred = db.get_existing_prediction(t, horizon_days=h, date=today)
                            if not pred:
                                try:
                                    pred = await predictor.predict(t, horizon_days=h)
                                except Exception as pe:
                                    log.error("markets.auto_predict_failed", ticker=t, horizon=h, error=str(pe))
                            
                            if pred:
                                p_dir = pred.get("predicted_direction", "UNK")
                                p_conf = int(pred.get("confidence", 0.0) * 100)
                                dir_emoji = "🟢" if p_dir == "UP" else "🔴"
                                h_label = {5: "5d", 21: "1m", 63: "3m", 252: "1y"}.get(h, f"{h}d")
                                pred_strs.append(f"{h_label}: {dir_emoji} {p_conf}%")
                                
                        pred_str = f"\n  " + " | ".join(pred_strs) if pred_strs else ""
                        
                        adv_str = ""
                        cached_adv = db.get_cached_advisory(t, days=5)
                        if cached_adv:
                            if isinstance(cached_adv, str):
                                try:
                                    cached_adv = json.loads(cached_adv)
                                except json.JSONDecodeError:
                                    pass
                            if isinstance(cached_adv, dict):
                                verdict = cached_adv.get("final_advisory", "")
                                exec_summary = cached_adv.get("executive_summary", "")
                                if not exec_summary:
                                    verdict = verdict.replace('*', '').replace('_', '').strip()
                                    exec_summary = verdict[:297] + "..." if len(verdict) > 300 else verdict
                                else:
                                    exec_summary = exec_summary.replace('*', '').replace('_', '').strip()
                                
                                cache_date = cached_adv.get("_cache_date", "Unknown")
                                adv_str = f"\n  Exec Summary: {exec_summary}\n  [Cached : {cache_date}]"
                        else:
                            adv_str = f"\n  Exec Summary: [Generating AI Advisory... check back later]"
                            asyncio.create_task(predictor.predict_with_agents(t))
                            
                        sector_results[sector].append(f"<b>{t}</b> • ${current:.2f} • {sign}{pct_diff:.2f}% {emoji}{pred_str}{adv_str}")
                        
                        pct_diffs.append(round(pct_diff, 2))
                        valid_tickers.append(t)
                        
                    elif len(closes) == 1:
                        current = closes[0]
                        sector_results[sector].append(f"<b>{t}</b> • ${current:.2f} 📊")
                    else:
                        sector_results[sector].append(f"<b>{t}</b> • No data ❓")
                except Exception as e:
                    sector_results[sector].append(f"<b>{t}</b> • Error fetching data ❌")
        
        await update.message.reply_html("<b>📊 Live Markets</b>")
        for sector, res_list in sorted(sector_results.items()):
            sector_text = f"<b>[ {sector} ]</b>\n"
            chunk = sector_text
            for res in res_list:
                if len(chunk) + len(res) > 3800:
                    await update.message.reply_html(chunk.strip())
                    chunk = f"<b>[ {sector} continued ]</b>\n"
                chunk += res + "\n\n"
            if chunk.strip() and chunk.strip() not in [f"<b>[ {sector} ]</b>", f"<b>[ {sector} continued ]</b>"]:
                await update.message.reply_html(chunk.strip())
        
        # Attach a bar chart for 1-day performance
        if valid_tickers:
            try:
                from bot.visualizations import get_sentiment_chart
                chart_buffer = await get_sentiment_chart("1-Day Performance (%)", pct_diffs, valid_tickers)
                await update.message.reply_photo(photo=chart_buffer)
            except Exception as e:
                log.error("markets.chart_failed", error=str(e))
        
    except Exception as e:
        log.error("telegram.markets_failed", error=str(e))
        await update.message.reply_text("❌ Failed to fetch market data.")

async def track_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /track command."""
    if not await auth_middleware(update, context):
        return
        
    if not context.args:
        await update.message.reply_text("Usage: /track <TICKER>")
        return
        
    ticker = context.args[0].upper()
    db = get_db(context)
    if db.add_tracked_ticker(ticker):
        await update.message.reply_text(f"✅ Now tracking <b>{ticker}</b>.", parse_mode="HTML")
    else:
        await update.message.reply_text(f"ℹ️ <b>{ticker}</b> is already being tracked.", parse_mode="HTML")

async def untrack_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /untrack command."""
    if not await auth_middleware(update, context):
        return
        
    if not context.args:
        await update.message.reply_text("Usage: /untrack <TICKER>")
        return
        
    ticker = context.args[0].upper()
    db = get_db(context)
    if db.remove_tracked_ticker(ticker):
        await update.message.reply_text(f"✅ Stopped tracking <b>{ticker}</b>.", parse_mode="HTML")
    else:
        await update.message.reply_text(f"ℹ️ <b>{ticker}</b> was not being tracked.", parse_mode="HTML")

async def briefing_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /briefing command."""
    if not await auth_middleware(update, context):
        return

    await update.message.reply_text("📰 Generating Market Briefing...")
    
    try:
        db = get_db(context)
        
        briefing = db.get_briefing_by_sector(hours=24, limit=10)
        if not briefing:
            await update.message.reply_text("No ranked articles found in the last 24 hours for a briefing.")
            return
            
        text = "<b>📰 Daily Market Briefing</b>\n\n"
        for sector, articles in briefing.items():
            text += f"<b>--- {escape_html(sector)} ---</b>\n"
            for r in articles:
                score = r["importance_score"]
                text += f"🔹 <b>{escape_html(r['headline'])}</b> (Score: {score})\n"
                text += f"<i>{escape_html(r['classification_summary'] or r['summary'])}</i>\n"
                text += f"<a href='{escape_html(r['url'])}'>Read more</a>\n\n"
            
        await update.message.reply_html(text, disable_web_page_preview=True)
    except Exception as e:
        log.error("telegram.briefing_failed", error=str(e))
        await update.message.reply_text("❌ Failed to generate briefing.")

async def chart_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /chart command."""
    if not await auth_middleware(update, context):
        return
        
    if not context.args:
        await update.message.reply_text("Usage: /chart <TICKER>")
        return
        
    ticker = context.args[0].upper()
    await update.message.reply_text(f"📊 Generating chart for {ticker}...")
    
    try:
        from bot.visualizations import get_price_chart
        
        # Fetch 30 day history
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        params = {"range": "1mo", "interval": "1d"}
        headers = {"User-Agent": "Mozilla/5.0"}
        
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, params=params, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            
        chart_data = data["chart"]["result"][0]
        closes = chart_data["indicators"]["quote"][0]["close"]
        timestamps = chart_data["timestamp"]
        
        # Clean data
        valid_closes = []
        valid_dates = []
        for i, c in enumerate(closes):
            if c is not None:
                valid_closes.append(round(c, 2))
                dt = datetime.datetime.fromtimestamp(timestamps[i])
                valid_dates.append(dt.strftime("%m-%d"))
                
        if not valid_closes:
            await update.message.reply_text("❌ Could not fetch data for chart.")
            return
            
        # Generate chart in memory
        chart_buffer = await get_price_chart(ticker, valid_closes, valid_dates)
        
        caption = f"📉 1 Month Price History: {ticker}"
        db = get_db(context)
        import datetime as dt_module
        today = dt_module.date.today().isoformat()
        prediction = db.get_existing_prediction(ticker, horizon_days=1, date=today)
        if prediction:
            direction = prediction.get("predicted_direction", "UNKNOWN")
            conf = int(prediction.get("confidence", 0.0) * 100)
            emoji = "📈" if direction == "UP" else "📉"
            caption += f"\n\n{emoji} Prediction: {direction} ({conf}% confident)"
        
        await update.message.reply_photo(photo=chart_buffer, caption=caption)
        
    except Exception as e:
        log.error("telegram.chart_failed", error=str(e))
        await update.message.reply_text("❌ Failed to generate chart.")

async def handle_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle natural language queries using the LangGraph orchestrator."""
    if not await auth_middleware(update, context):
        return
        
    query = update.message.text
    if not query or query.startswith("/"):
        return
        
    status_msg = await update.message.reply_text("🧠 Thinking...")
    
    try:
        db = get_db(context)
        from pipeline.chat_orchestrator import ChatOrchestrator
        
        async def progress_callback(msg: str):
            try:
                await context.bot.edit_message_text(
                    chat_id=update.effective_chat.id,
                    message_id=status_msg.message_id,
                    text=f"⏳ {msg}"
                )
            except Exception:
                pass
                
        orchestrator = ChatOrchestrator(db, progress_callback=progress_callback)
        answer = await orchestrator.run(query)
        
        # Send final answer
        await context.bot.edit_message_text(
            chat_id=update.effective_chat.id,
            message_id=status_msg.message_id,
            text=f"<b>Analyst Report</b>\n\n{answer}",
            parse_mode="HTML"
        )
        
    except Exception as e:
        log.error("telegram.query_failed", error=str(e))
        await context.bot.edit_message_text(
            chat_id=update.effective_chat.id,
            message_id=status_msg.message_id,
            text="❌ Failed to process query."
        )

async def predict_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /predict <TICKER> [force] command."""
    if not await auth_middleware(update, context):
        return
        
    if not context.args:
        await update.message.reply_text("Usage: /predict <TICKER> [force]")
        return
        
    ticker = context.args[0].upper()
    # Parse optional second argument: "force" to bypass cache, or a horizon label (5d, 1m, 3m, 1y)
    force = False
    horizon_days = None
    HORIZON_MAP = {"5d": 5, "1m": 21, "3m": 63, "1y": 252}
    if len(context.args) > 1:
        arg1 = context.args[1].lower()
        if arg1 == "force":
            force = True
        elif arg1 in HORIZON_MAP:
            horizon_days = HORIZON_MAP[arg1]
    
    try:
        from pipeline.predictor import StockPredictor
        db = get_db(context)
        
        def format_advisory_html(text: str) -> str:
            text = html.escape(text)
            text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text, flags=re.DOTALL)
            text = re.sub(r'(?m)^###\s+(.*)', r'<b>\1</b>', text)
            text = re.sub(r'(?m)^##\s+(.*)', r'<b>\1</b>', text)
            text = re.sub(r'(?m)^#\s+(.*)', r'<b>\1</b>', text)
            text = re.sub(r'(?m)^\*\s+', r'• ', text)
            text = re.sub(r'(?m)^-\s+', r'• ', text)
            return text

        def chunk_html(text: str, limit=3800) -> list[str]:
            chunks = []
            paragraphs = text.split('\n\n')
            chunk = ""
            for p in paragraphs:
                if len(chunk) + len(p) > limit:
                    chunks.append(chunk.strip())
                    chunk = ""
                chunk += p + "\n\n"
            if chunk.strip():
                chunks.append(chunk.strip())
            return chunks

        if not force:
            cached = db.get_cached_advisory(ticker, days=5)
            if cached:
                if isinstance(cached, str):
                    cached = json.loads(cached)
                advisory = cached.get('final_advisory', 'No plan found')
                cache_date = cached.get('_cache_date', 'Unknown')
                
                advisory_html = format_advisory_html(advisory)
                full_html = f"<b>🔮 Multi-Agent Advisory for {ticker}</b>\n<i>[Cached: {cache_date}]</i>\n\n<b>Final Advisory:</b>\n{advisory_html}"
                
                chunks = chunk_html(full_html)
                for chunk in chunks:
                    await update.message.reply_html(chunk)
                return

        status_msg = await update.message.reply_text(f"⏳ Starting Multi-Agent Analysis for {ticker}...")
        
        async def progress_callback(msg: str):
            try:
                await context.bot.edit_message_text(
                    chat_id=update.effective_chat.id,
                    message_id=status_msg.message_id,
                    text=f"⏳ {msg}"
                )
            except Exception:
                pass
                
        predictor = StockPredictor(db)
        kwargs = {"progress_callback": progress_callback}
        if horizon_days:
            kwargs["horizon_days"] = horizon_days
        result = await predictor.predict_with_agents(ticker, **kwargs)
        
        advisory = result.get('final_advisory', 'No plan found')
        advisory_html = format_advisory_html(advisory)
        full_html = f"<b>🔮 Multi-Agent Advisory for {ticker}</b>\n\n<b>Final Advisory:</b>\n{advisory_html}"
        
        chunks = chunk_html(full_html)
        if chunks:
            await context.bot.edit_message_text(
                chat_id=update.effective_chat.id,
                message_id=status_msg.message_id,
                text=chunks[0],
                parse_mode="HTML"
            )
            for chunk in chunks[1:]:
                await context.bot.send_message(chat_id=update.effective_chat.id, text=chunk, parse_mode="HTML")
        
    except Exception as e:
        log.error("telegram.predict_failed", error=str(e))
        await update.message.reply_text("❌ Failed to generate prediction.")

async def accuracy_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /accuracy [TICKER] command."""
    if not await auth_middleware(update, context):
        return
        
    ticker = context.args[0].upper() if context.args else None
    
    try:
        db = get_db(context)
        acc = db.get_prediction_accuracy(ticker)
        
        title = f"<b>🎯 Prediction Accuracy ({ticker})</b>\n\n" if ticker else "<b>🎯 Overall Prediction Accuracy</b>\n\n"
        
        if not acc or acc.get("total", 0) == 0:
            await update.message.reply_html(title + "No resolved predictions found.")
            return
            
        text = title
        text += f"<b>Total Predictions:</b> {acc.get('total', 0)}\n"
        text += f"<b>Correct:</b> {acc.get('correct', 0)}\n"
        text += f"<b>Incorrect:</b> {acc.get('incorrect', 0)}\n"
        text += f"<b>Accuracy:</b> {acc.get('accuracy_pct', 0):.1f}%\n\n"
        
        recent = db.get_recent_predictions(ticker, limit=5)
        if recent:
            text += "<b>Recent Predictions:</b>\n"
            for p in recent:
                icon = "✅" if p.get("is_correct") == 1 else ("❌" if p.get("is_correct") == 0 else "⏳")
                t = p.get("ticker", "UNK")
                dir = p.get("predicted_direction", "")
                actual = p.get('actual_change_pct')
                actual_str = f"{actual:.2f}%" if actual is not None else "N/A"
                text += f"{icon} {t}: {dir} ({int(p.get('confidence',0)*100)}%) -> {actual_str}\n"
                
        await update.message.reply_html(text)
        
    except Exception as e:
        log.error("telegram.accuracy_failed", error=str(e))
        await update.message.reply_text("❌ Failed to fetch accuracy.")


async def sectors_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /sectors command — show sector sentiment heatmap."""
    if not await auth_middleware(update, context):
        return
    try:
        db = get_db(context)
        from pipeline.sector_analyzer import SectorAnalyzer
        analyzer = SectorAnalyzer(db)
        data = analyzer._compute_sector_snapshot(hours=24)

        if not data:
            await update.message.reply_text("📊 No sector data available yet.")
            return

        text = "<b>📊 Sector Sentiment Heatmap</b>\n\n"
        for sd in data[:8]:
            emoji = "🟢" if sd["avg_sentiment"] > 0.15 else ("🔴" if sd["avg_sentiment"] < -0.15 else "🟡")
            direction = "▲" if sd["sentiment_momentum"] > 0 else "▼"
            text += (
                f"{emoji} <b>{sd['sector']}</b>\n"
                f"   Sentiment: {sd['avg_sentiment']:.2f} {direction}{abs(sd['sentiment_momentum']):.2f}\n"
                f"   Articles: {sd['article_count']} | "
                f"🟢{sd['bullish_count']} 🔴{sd['bearish_count']} ⚪{sd['neutral_count']}\n\n"
            )

        await update.message.reply_html(text)
    except Exception as e:
        log.error("telegram.sectors_failed", error=str(e))
        await update.message.reply_text("❌ Failed to fetch sector data.")


async def ipos_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /ipos command — show tracked IPO watchlist."""
    if not await auth_middleware(update, context):
        return
    try:
        db = get_db(context)
        from pipeline.ipo_detector import IPODetector
        detector = IPODetector(db)
        ipos = detector.get_ipo_watchlist(limit=10)

        if not ipos:
            await update.message.reply_text("📋 No IPOs detected in the news feed.")
            return

        text = "<b>📋 IPO Watchlist</b>\n\n"
        for ipo in ipos:
            status_emoji = {"upcoming": "⏳", "priced": "💰", "listed": "✅", "withdrawn": "❌", "rumored": "🔮"}
            emoji = status_emoji.get(ipo["status"], "📌")
            name = escape_html(ipo["company_name"])
            ticker = ipo["ticker"] or "TBA"
            sector = f" | {ipo['sector']}" if ipo.get("sector") else ""
            date_info = f"📅 {ipo['ipo_date']}" if ipo.get("ipo_date") else ""
            text += f"{emoji} <b>{name}</b> ({ticker}{sector})\n"
            if date_info:
                text += f"   {date_info}\n"
            text += f"   Status: <b>{ipo['status'].upper()}</b>\n\n"

        await update.message.reply_html(text)
    except Exception as e:
        log.error("telegram.ipos_failed", error=str(e))
        await update.message.reply_text("❌ Failed to fetch IPO data.")


async def events_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /events command — show upcoming earnings and big events."""
    if not await auth_middleware(update, context):
        return
    try:
        db = get_db(context)
        from pipeline.event_tracker import EventTracker
        tracker = EventTracker(db)

        ticker = context.args[0].upper() if context.args else None
        if ticker:
            events = tracker.get_ticker_events(ticker, days_ahead=30)
            title = f"📅 Upcoming Events for <b>{ticker}</b>"
        else:
            events = tracker.get_tracked_events_calendar(days_ahead=14)
            title = "📅 Upcoming Events (Tracked Tickers)"

        if not events:
            await update.message.reply_html(f"{title}\n\nNo upcoming events found.")
            return

        text = f"{title}\n\n"
        event_icons = {
            "earnings": "📅", "product_launch": "🚀", "investor_day": "🏛️",
            "fda_decision": "💊", "conference": "🎤", "dividend": "💰",
            "split": "🔀", "acquisition": "🤝",
        }
        for ev in events[:10]:
            icon = event_icons.get(ev["event_type"], "📌")
            ev_date = ev["event_date"]
            ticker_str = escape_html(ev["ticker"])
            event_title = escape_html(ev["event_title"] or ev["event_type"].replace("_", " ").title())
            confidence = ev.get("confidence", "estimated")
            badge = "✅" if confidence == "confirmed" else ("☑️" if confidence == "estimated" else "❓")
            text += f"{icon} <b>{ticker_str}</b> — {event_title}\n"
            text += f"   📅 {ev_date} {badge} {confidence}\n\n"

        await update.message.reply_html(text)
    except Exception as e:
        log.error("telegram.events_failed", error=str(e))
        await update.message.reply_text("❌ Failed to fetch events.")


async def themes_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /themes command — show current macro themes."""
    if not await auth_middleware(update, context):
        return
    try:
        db = get_db(context)
        from pipeline.trend_forecaster import TrendForecaster
        forecaster = TrendForecaster(db)

        # Read from in-memory cache first (refreshed every 4h by scheduler)
        themes = TrendForecaster.get_cached_macro_themes()
        if themes is None:
            themes = await forecaster.generate_and_cache_macro_themes()

        if not themes:
            await update.message.reply_text("🌍 No macro themes generated yet.")
            return

        text = "<b>🌍 Current Macro Themes</b>\n\n"
        for i, theme in enumerate(themes[:5], 1):
            title = escape_html(theme.get("title", "Unknown"))
            explanation = escape_html(theme.get("explanation", ""))
            confidence = theme.get("confidence", 0.5)
            bar = "█" * int(confidence * 10) + "░" * (10 - int(confidence * 10))
            text += (
                f"<b>{i}. {title}</b>\n"
                f"   {explanation}\n"
                f"   Confidence: {bar} {confidence:.0%}\n\n"
            )

        await update.message.reply_html(text)
    except Exception as e:
        log.error("telegram.themes_failed", error=str(e))
        await update.message.reply_text("❌ Failed to fetch macro themes.")


async def forecast_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /forecast <SECTOR> command — LLM sector outlook."""
    if not await auth_middleware(update, context):
        return
    if not context.args:
        await update.message.reply_text("Usage: /forecast <SECTOR>\nExample: /forecast Technology")
        return

    sector = " ".join(context.args).title()
    try:
        db = get_db(context)
        from pipeline.trend_forecaster import TrendForecaster
        forecaster = TrendForecaster(db)
        forecast = await forecaster._generate_sector_outlook(sector)

        if not forecast:
            await update.message.reply_text(f"📊 No forecast available for '{sector}'.")
            return

        narrative = forecast.get("narrative", "No analysis available.")
        drivers = forecast.get("key_drivers", [])
        confidence = forecast.get("confidence", 0.5)

        text = f"<b>📊 Sector Outlook: {escape_html(sector)}</b>\n\n"
        text += f"{escape_html(narrative)}\n\n"

        if drivers:
            text += "<b>Key Drivers:</b>\n"
            for d in drivers[:5]:
                text += f"• {escape_html(d)}\n"
            text += "\n"

        bar = "█" * int(confidence * 10) + "░" * (10 - int(confidence * 10))
        text += f"Analyst Confidence: {bar} {confidence:.0%}"

        await update.message.reply_html(text)
    except Exception as e:
        log.error("telegram.forecast_failed", error=str(e))
        await update.message.reply_text(f"❌ Failed to generate forecast for '{sector}'.")
