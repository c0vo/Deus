"""
Project Scrooge V2 — Pipeline Orchestrator

Manages the periodic execution of the entire data pipeline:
1. Fetching from sources
2. Classification
3. Ranking
4. Embedding
5. Alerting
"""

import datetime
import json
import time
import asyncio

import numpy as np
from zoneinfo import ZoneInfo
from typing import Optional
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from google.genai import types
from config.logging_config import get_logger
from config.settings import settings
from data.database import Database
from pipeline.aggregator import NewsAggregator
from pipeline.classifier import ArticleClassifier
from pipeline.ranker import ArticleRanker
from pipeline.embedder import GeminiEmbedder
from pipeline.market_scanner import MarketScanner
from pipeline.sector_analyzer import SectorAnalyzer
from pipeline.ipo_detector import IPODetector
from pipeline.geo_tagger import GeoTagger
from pipeline.event_tracker import EventTracker
from pipeline.trend_forecaster import TrendForecaster
from bot.alerts import AlertManager
from api.sse_manager import event_bus

log = get_logger(__name__)

class PipelineOrchestrator:
    """Orchestrates the periodic execution of the Scrooge pipeline."""

    def __init__(self, db: Database, alert_manager: Optional[AlertManager] = None):
        self.db = db
        self.alert_manager = alert_manager

        self.aggregator = NewsAggregator(db=self.db)
        self.classifier = ArticleClassifier(db=self.db)
        self.ranker = ArticleRanker(db=self.db)
        self.embedder = GeminiEmbedder()
        self.market_scanner = MarketScanner(db=self.db, alert_manager=self.alert_manager)
        self.sector_analyzer = SectorAnalyzer(db=self.db, alert_manager=self.alert_manager)
        self.ipo_detector = IPODetector(db=self.db)
        self.geo_tagger = GeoTagger(db=self.db)
        self.event_tracker = EventTracker(db=self.db, alert_manager=self.alert_manager)
        self.trend_forecaster = TrendForecaster(db=self.db)

        self.scheduler = AsyncIOScheduler()
        self.is_running = False

    async def run_pipeline_cycle(self) -> None:
        """Executes a single pass of the entire pipeline."""
        if self.is_running:
            log.warning("orchestrator.skipped", reason="Previous cycle still running")
            return

        self.is_running = True
        log.info("orchestrator.cycle_start")

        cycle_start = time.time()
        self._cycle_counts = {"fetched": 0, "inserted": 0, "classified": 0,
                              "ranked": 0, "embedded": 0, "alerts": 0, "errors": 0,
                              "llm_calls": 0, "llm_cost": 0.0}

        try:
            # Initialize embedder if not already initialized
            if not self.embedder._initialized:
                await self.embedder.initialize()

            # Start fetch in background
            fetch_task = asyncio.create_task(self.aggregator.fetch_all())

            # Continuously process while fetching is ongoing
            while not fetch_task.done():
                await self._process_batch()
                await asyncio.sleep(2)

            # Final pass to catch anything fetched right at the end
            await self._process_batch()

            # Capture fetch results
            fetch_result = await fetch_task
            self._cycle_counts["fetched"] = len(fetch_result)
            self._cycle_counts["inserted"] = len(fetch_result)

        except Exception as e:
            log.error("orchestrator.cycle_failed", error=str(e))
            self._cycle_counts["errors"] += 1
        finally:
            duration = time.time() - cycle_start
            self.is_running = False
            # Record pipeline metrics
            try:
                self.db.insert_pipeline_metrics(duration, self._cycle_counts)
            except Exception as e:
                log.error("orchestrator.metrics_failed", error=str(e))

            # Publish pipeline status to SSE event bus (fire-and-forget task)
            try:
                asyncio.create_task(
                    event_bus.publish("pipeline_status", {
                        "duration_seconds": round(duration, 2),
                        **self._cycle_counts,
                    })
                )
            except Exception as e:
                log.error("orchestrator.sse_publish_failed", error=str(e))

        # Scan for IPO mentions and upcoming events on freshly classified articles
        try:
            ipo_results = await self.ipo_detector.scan_for_ipos()
            if ipo_results:
                log.info("orchestrator.ipos_detected_in_cycle", count=len(ipo_results))
        except Exception as e:
            log.warning("orchestrator.ipo_scan_failed", error=str(e))

        try:
            event_results = await self.event_tracker.scan_upcoming_events()
            if event_results:
                log.info("orchestrator.events_detected_in_cycle", count=len(event_results))
        except Exception as e:
            log.warning("orchestrator.event_scan_failed", error=str(e))

        log.info("orchestrator.cycle_complete", duration_seconds=round(duration, 2))

    async def _process_batch(self) -> None:
        """Process a batch of articles (Embed -> Classify -> Rank -> Alert)."""
        try:

            # Step 2: Embed unembedded articles FIRST (for deduplication)
            with self.db.connection() as conn:
                unembedded = conn.execute(
                    "SELECT * FROM articles WHERE embedding IS NULL LIMIT 50"
                ).fetchall()

            if unembedded:
                log.info("orchestrator.embed", count=len(unembedded))
                self._cycle_counts["embedded"] += len(unembedded)
                articles_to_embed = [
                    self.db.row_to_article(row) for row in unembedded
                ]
                # Parallel embedding: process up to 5 articles at once via asyncio.gather
                BATCH_SIZE = 5
                for i in range(0, len(articles_to_embed), BATCH_SIZE):
                    batch = articles_to_embed[i:i+BATCH_SIZE]

                    async def embed_single(a):
                        embedding = None
                        for _ in range(3):
                            embedding = await self.embedder.embed_article(a)
                            if embedding is not None:
                                break
                            await asyncio.sleep(2)
                        return a.id, embedding

                    results = await asyncio.gather(*[embed_single(a) for a in batch])

                    for article_id, embedding in results:
                        if embedding is not None:
                            self.db.update_embedding(article_id, embedding)
                        else:
                            dummy_vector = np.zeros(3072, dtype=np.float32)
                            self.db.update_embedding(article_id, dummy_vector)

            # Step 3: Classify unclassified articles (with Semantic Deduplication)
            unclassified_data = self.db.get_unclassified_articles(limit=20)
            log.info("orchestrator.classify", count=len(unclassified_data))
            classified_count = 0

            for row_dict in unclassified_data:
                article = self.db.row_to_article(row_dict)

                # Fetch embedding
                article_emb = None
                with self.db.connection() as conn:
                    row = conn.execute("SELECT embedding FROM articles WHERE id = ?", (article.id,)).fetchone()
                    if row and row["embedding"]:
                        article_emb = np.frombuffer(row["embedding"], dtype=np.float32)

                is_duplicate = False

                if article_emb is not None:
                    # Windowed nearest-neighbour lookup. Bounding by publish date
                    # keeps an old story from suppressing a current one and stops
                    # every candidate query being a full scan of the corpus.
                    match = self.db.find_duplicate(
                        article_id=article.id,
                        embedding=article_emb,
                        published_at=article.published_at.isoformat()
                        if hasattr(article.published_at, "isoformat")
                        else str(article.published_at),
                        window_days=settings.dedup_window_days,
                        threshold=settings.dedup_similarity_threshold,
                    )

                    if match:
                        best_match_id, highest_sim = match
                        with self.db.connection() as conn:
                            source_row = conn.execute(
                                "SELECT * FROM articles WHERE id = ?", (best_match_id,)
                            ).fetchone()

                        if source_row and source_row["event_type"]:
                            is_duplicate = True
                            log.info(
                                "orchestrator.semantic_dedup.flagged",
                                article_id=article.id,
                                source_id=best_match_id,
                                sim=round(highest_sim, 3),
                            )
                            # Inherit the canonical article's classification so the
                            # row stays queryable, then flag it. Flagged rows are
                            # excluded from feeds, trending and coverage counts —
                            # and deliberately NOT written to ticker_mentions,
                            # which is what was inflating trending counts when the
                            # same story arrived from five syndicating outlets.
                            self.db.update_classification(
                                article_id=article.id,
                                event_type=source_row["event_type"],
                                sentiment_score=source_row["sentiment_score"],
                                urgency=source_row["urgency"],
                                suggested_direction=source_row["suggested_direction"],
                                affected_sectors=json.loads(source_row["affected_sectors"]) if source_row["affected_sectors"] else [],
                                affected_tickers=json.loads(source_row["affected_tickers"]) if source_row["affected_tickers"] else [],
                                classification_summary=source_row["classification_summary"],
                            )
                            self.db.mark_duplicate(article.id, best_match_id)
                        else:
                            # Nearest neighbour exists but isn't classified yet, so
                            # there is nothing to inherit. Leave it for a later pass.
                            log.debug(
                                "orchestrator.semantic_dedup.match_unclassified",
                                article_id=article.id,
                                sim=round(highest_sim, 3),
                            )
                    else:
                        self.db.mark_dedup_checked(article.id)

                if not is_duplicate:
                    classified_article = None
                    for _ in range(3):
                        classified_article = await self.classifier.classify(article)
                        if classified_article.event_type:
                            break
                        await asyncio.sleep(2)

                    if classified_article and classified_article.event_type:
                        self.db.update_classification(
                            article_id=classified_article.id,
                            event_type=classified_article.event_type,
                            sentiment_score=classified_article.sentiment_score,
                            urgency=classified_article.urgency,
                            suggested_direction=classified_article.suggested_direction,
                            affected_sectors=classified_article.affected_sectors,
                            affected_tickers=classified_article.affected_tickers,
                            classification_summary=classified_article.classification_summary,
                            countries=classified_article.countries or [],
                        )

                        self.db.insert_ticker_mentions(
                            article_id=classified_article.id,
                            tickers=classified_article.affected_tickers,
                            sentiment_score=classified_article.sentiment_score,
                            urgency=classified_article.urgency
                        )
                        asyncio.create_task(
                            event_bus.publish("new_articles", {"articles": [classified_article.model_dump()]})
                        )
                        classified_count += 1
                    else:
                        # Fallback: Mark as failed so it doesn't infinite loop
                        self.db.update_classification(
                            article_id=article.id,
                            event_type="error",
                            sentiment_score=0.0,
                            urgency="low",
                            suggested_direction="neutral",
                            affected_sectors=[],
                            affected_tickers=[],
                            classification_summary="Classification failed due to parsing error."
                        )
                        asyncio.create_task(
                            event_bus.publish("new_articles", {"articles": [article.model_dump()]})
                        )

            self._cycle_counts["classified"] += classified_count

            # Step 4: Rank unranked articles (skip low-signal articles to save LLM calls)
            unranked_data = self.db.get_unranked_articles(limit=20)
            log.info("orchestrator.rank", count=len(unranked_data))
            ranked_count = 0

            if unranked_data:
                # Filter: low-signal articles (low urgency + near-zero sentiment) skip LLM ranking
                to_rank = []
                for row_dict in unranked_data:
                    urgency = row_dict.get("urgency", "")
                    sentiment = row_dict.get("sentiment_score") or 0.0
                    if urgency == "low" and abs(sentiment) < 0.1:
                        self.db.update_ranking(
                            article_id=row_dict["id"],
                            importance_score=1.0
                        )
                    else:
                        to_rank.append(row_dict)

                if to_rank:
                    articles_to_rank = [
                        self.db.row_to_article(row_dict) for row_dict in to_rank
                    ]

                    ranked_articles = []
                    for _ in range(3):
                        ranked_articles = await self.ranker.rank_batch(articles_to_rank)
                        if ranked_articles and any(r.importance_score is not None for r in ranked_articles):
                            break
                        await asyncio.sleep(2)

                    for r in ranked_articles:
                        if r.importance_score is not None:
                            self.db.update_ranking(
                                article_id=r.id,
                                importance_score=r.importance_score
                            )
                            ranked_count += 1
                        else:
                            # Fallback: Mark as failed so it doesn't infinite loop
                            self.db.update_ranking(
                                article_id=r.id,
                                importance_score=0.0
                            )

                        # Step 5: Process Alerts
                        if self.alert_manager:
                            await self.alert_manager.process_for_alerts([r])

            self._cycle_counts["ranked"] += ranked_count

        except Exception as e:
            log.error("orchestrator.process_batch_failed", error=str(e))

    async def send_daily_briefing(self):
        """Sends the daily briefing at the scheduled time."""
        if not self.alert_manager:
            return
            
        try:
            from bot.formatters import escape_html
            
            briefing = self.db.get_briefing_by_sector(hours=24, limit=10)
            if not briefing:
                return
                
            text = "<b>📰 Daily Market Briefing</b>\n\n"
            for sector, articles in briefing.items():
                text += f"<b>--- {escape_html(sector)} ---</b>\n"
                for r in articles:
                    score = r["importance_score"]
                    text += f"🔹 <b>{escape_html(r['headline'])}</b> (Score: {score})\n"
                    text += f"<i>{escape_html(r['classification_summary'])}</i>\n"
                    text += f"<a href='{escape_html(r['url'])}'>Read more</a>\n\n"
                    
            await self.alert_manager.bot.send_message(
                chat_id=self.alert_manager.chat_id,
                text=text,
                parse_mode="HTML",
                disable_web_page_preview=True
            )
            log.info("orchestrator.daily_briefing_sent")
        except Exception as e:
            log.error("orchestrator.daily_briefing_failed", error=str(e))

    async def send_daily_advisor(self):
        """Sends daily hold/sell advice for tracked tickers."""
        if not self.alert_manager:
            return
            
        try:
            from bot.formatters import escape_html
            from config.llm import get_client, DEFAULT_SAFETY_SETTINGS

            tracked = self.db.get_tracked_tickers()
            if not tracked:
                return
                
            all_ticker_contexts = {}
            for t in tracked:
                summaries = self.db.get_recent_summaries_for_ticker(t, hours=24)
                if summaries:
                    all_ticker_contexts[t] = summaries
            
            ai_summaries = {}
            client = get_client()
            if client and all_ticker_contexts:
                prompt = (
                    "You are a professional Wall Street advisor reviewing your client's portfolio.\n"
                    "Below are the client's tracked tickers with their recent news from the past 24 hours.\n"
                    "For EACH ticker, recommend HOLD or SELL for tomorrow based ONLY on the news context.\n"
                    "Guidelines:\n"
                    "- HOLD: News is neutral-to-positive, or no significant negative catalyst. Default to HOLD unless there's a clear reason to sell.\n"
                    "- SELL: Specific negative catalyst in the news (earnings miss, downgrade, regulatory issue, macro headwind directly impacting the ticker).\n"
                    "Format: valid JSON dictionary. Keys = tickers. Values = 'HOLD - [1 sentence reason]' or 'SELL - [1 sentence reason]'.\n"
                    "Plain text only — no markdown, no HTML.\n\n"
                )
                for tk, sums in all_ticker_contexts.items():
                    prompt += f"Ticker: {tk}\nContext:\n" + "\n".join(f"- {s}" for s in sums) + "\n\n"

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
                            operation="daily_advisor_batch",
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
                            operation="daily_advisor_batch",
                            prompt_tokens=0,
                            candidate_tokens=0,
                            latency_ms=latency_ms,
                            is_error=True,
                            error_message=error_msg,
                            prompt_text=prompt,
                            response_text=None
                        )
                ai_summaries = json.loads(response.text.strip())
                
            text = "<b>🎯 Tracked Tickers Daily Advisor</b>\n\n"
            text += "Based on today's news flow, here is my outlook for your portfolio tomorrow:\n\n"
            
            for t in tracked:
                if t in all_ticker_contexts:
                    advice = ai_summaries.get(t, "HOLD - Unable to generate advice.")
                    emoji = "🛑" if advice.startswith("SELL") else "✋"
                    text += f"{emoji} <b>{t}</b>: {escape_html(advice)}\n\n"
                else:
                    text += f"✋ <b>{t}</b>: HOLD - No significant news today.\n\n"
                    
            await self.alert_manager.bot.send_message(
                chat_id=self.alert_manager.chat_id,
                text=text,
                parse_mode="HTML",
                disable_web_page_preview=True
            )
            log.info("orchestrator.daily_advisor_sent")
        except Exception as e:
            log.error("orchestrator.daily_advisor_failed", error=str(e))

    async def send_weekly_review(self):
        """Sends the weekly portfolio review every Friday."""
        if not self.alert_manager:
            return
            
        try:
            import httpx
            from bot.formatters import escape_html
            
            tracked = self.db.get_tracked_tickers()
            if not tracked:
                return
                
            text = "<b>📅 Weekly Portfolio Review</b>\n\n"
            text += "Here is how your tracked tickers performed this week:\n\n"
            
            async with httpx.AsyncClient(timeout=10) as client:
                for t in tracked:
                    try:
                        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{t}"
                        params = {"range": "5d", "interval": "1d"}
                        headers = {"User-Agent": "Mozilla/5.0"}
                        resp = await client.get(url, params=params, headers=headers)
                        data = resp.json()
                        
                        chart = data.get("chart", {}).get("result", [])
                        if chart:
                            closes = chart[0].get("indicators", {}).get("quote", [{}])[0].get("close", [])
                            closes = [c for c in closes if c is not None]
                            if len(closes) >= 2:
                                start_price = closes[0]
                                end_price = closes[-1]
                                diff = end_price - start_price
                                pct = (diff / start_price) * 100
                                emoji = "🟢" if diff >= 0 else "🔴"
                                text += f"{emoji} <b>{t}</b>: {pct:+.2f}% (Ended at ${end_price:.2f})\n"
                    except Exception:
                        pass
            
            text += "\n<b>Top News for Your Portfolio This Week:</b>\n"
            
            with self.db.connection() as conn:
                for t in tracked[:5]: # Limit to top 5 tracked to avoid huge messages
                    rows = conn.execute(
                        """
                        SELECT a.headline, a.url 
                        FROM articles a
                        JOIN ticker_mentions tm ON a.id = tm.article_id
                        WHERE tm.ticker = ? AND a.published_at >= datetime('now', '-7 days')
                        ORDER BY a.importance_score DESC NULLS LAST, a.published_at DESC
                        LIMIT 1
                        """,
                        (t,)
                    ).fetchall()
                    if rows:
                        text += f"• <b>{t}</b>: <a href='{escape_html(rows[0]['url'])}'>{escape_html(rows[0]['headline'])}</a>\n"
                        
            await self.alert_manager.bot.send_message(
                chat_id=self.alert_manager.chat_id,
                text=text,
                parse_mode="HTML",
                disable_web_page_preview=True
            )
            log.info("orchestrator.weekly_review_sent")
        except Exception as e:
            log.error("orchestrator.weekly_review_failed", error=str(e))

    async def check_api_usage_spikes(self):
        """Checks for API usage spikes and sends an alert if needed."""
        try:
            alert_msg = self.db.check_for_api_spikes()
            if alert_msg and self.alert_manager:
                await self.alert_manager.bot.send_message(
                    chat_id=self.alert_manager.chat_id,
                    text=alert_msg,
                    parse_mode="HTML"
                )
                log.info("orchestrator.api_spike_alert_sent")
        except Exception as e:
            log.error("orchestrator.check_api_usage_spikes_failed", error=str(e))

    async def backfill_duplicates(self) -> None:
        """
        Flag semantic duplicates among articles ingested before dedup existed.

        Runs a bounded batch per tick rather than one long pass, so a large
        backlog converges over time without blocking the pipeline or spiking
        memory on a phone. Converges because every article examined is marked
        ``dedup_checked``, duplicate or not.
        """
        try:
            backlog = self.db.get_dedup_backlog(limit=settings.dedup_backfill_batch)
            if not backlog:
                return

            flagged = 0
            for row in backlog:
                if not row.get("embedding"):
                    self.db.mark_dedup_checked(row["id"])
                    continue

                embedding = np.frombuffer(row["embedding"], dtype=np.float32)
                match = self.db.find_duplicate(
                    article_id=row["id"],
                    embedding=embedding,
                    published_at=str(row["published_at"]),
                    window_days=settings.dedup_window_days,
                    threshold=settings.dedup_similarity_threshold,
                )
                if match:
                    self.db.mark_duplicate(row["id"], match[0])
                    flagged += 1
                else:
                    self.db.mark_dedup_checked(row["id"])

            stats = self.db.get_dedup_stats()
            log.info(
                "orchestrator.dedup_backfill",
                examined=len(backlog),
                flagged=flagged,
                remaining=stats["unchecked"],
                duplicates_total=stats["duplicates"],
            )
        except Exception as e:
            log.error("orchestrator.dedup_backfill_failed", error=str(e))

    async def backfill_geo_tags(self) -> None:
        """
        Tag older articles with countries using the offline gazetteer.

        New articles get countries from the classifier; this covers everything
        ingested before that field existed. It makes no API calls, so the whole
        archive can be tagged for free.
        """
        try:
            tagged = self.geo_tagger.backfill(limit=settings.geo_backfill_batch)
            if tagged:
                remaining = self.db.get_geo_backlog_count()
                log.info(
                    "orchestrator.geo_backfill", tagged=tagged, remaining=remaining
                )
        except Exception as e:
            log.error("orchestrator.geo_backfill_failed", error=str(e))

    async def retire_stale_ipos(self) -> None:
        """Drop IPOs that have finished listing or were never real."""
        try:
            removed = self.ipo_detector.retire_stale(
                listed_retention_days=settings.ipo_listed_retention_days
            )
            if removed:
                log.info("orchestrator.ipo_retired", count=removed)
        except Exception as e:
            log.error("orchestrator.ipo_retire_failed", error=str(e))

    def start(self, interval_minutes: int = 5) -> None:
        """Starts the periodic scheduler."""
        # Run immediately on startup
        self.scheduler.add_job(
            self.run_pipeline_cycle,
            'date',
            run_date=datetime.datetime.now()
        )
        # Then run periodically
        self.scheduler.add_job(
            self.run_pipeline_cycle,
            'interval',
            minutes=interval_minutes,
            id='pipeline_cycle',
            replace_existing=True
        )
        self.scheduler.add_job(
            self.market_scanner.run_scan,
            'interval',
            minutes=10,
            id='market_scanner',
            replace_existing=True
        )
        self.scheduler.add_job(
            self.check_api_usage_spikes,
            'interval',
            minutes=10,
            id='api_spike_check',
            replace_existing=True
        )
        # Chips away at the pre-dedup backlog a batch at a time.
        self.scheduler.add_job(
            self.backfill_duplicates,
            'interval',
            minutes=7,
            id='dedup_backfill',
            replace_existing=True
        )
        # Same idea for country tags on pre-geo articles. No API cost.
        self.scheduler.add_job(
            self.backfill_geo_tags,
            'interval',
            minutes=4,
            id='geo_backfill',
            replace_existing=True
        )
        
        # 5:00 AM KST
        seoul_tz = ZoneInfo("Asia/Seoul")
        
        # 5:00 AM KST Daily Briefing
        self.scheduler.add_job(
            self.send_daily_briefing,
            CronTrigger(hour=5, minute=0, timezone=seoul_tz),
            id='daily_briefing',
            replace_existing=True
        )

        # 4:30 AM KST — retire IPOs that have already listed, before the briefing
        self.scheduler.add_job(
            self.retire_stale_ipos,
            CronTrigger(hour=4, minute=30, timezone=seoul_tz),
            id='ipo_retire',
            replace_existing=True
        )
        
        # 5:05 AM KST Daily Advisor
        self.scheduler.add_job(
            self.send_daily_advisor,
            CronTrigger(hour=5, minute=5, timezone=seoul_tz),
            id='daily_advisor',
            replace_existing=True
        )
        
        # 8:00 AM KST Daily Earnings Whisper Check
        self.scheduler.add_job(
            self.market_scanner.check_earnings,
            CronTrigger(hour=8, minute=0, timezone=seoul_tz),
            id='earnings_whisper',
            replace_existing=True
        )
        
        # 8:30 AM KST Daily Predictions
        self.scheduler.add_job(
            run_daily_predictions,
            CronTrigger(hour=8, minute=30, timezone=seoul_tz),
            args=[self.db, self.alert_manager],
            id='daily_predictions',
            replace_existing=True
        )
        
        # 18:00 KST (6:00 PM) Friday Weekly Review
        self.scheduler.add_job(
            self.send_weekly_review,
            CronTrigger(day_of_week='fri', hour=18, minute=0, timezone=seoul_tz),
            id='weekly_review',
            replace_existing=True
        )
        
        # 21:00 KST Daily Prediction Resolution
        self.scheduler.add_job(
            self.resolve_predictions,
            CronTrigger(hour=21, minute=0, timezone=seoul_tz),
            id='prediction_resolution',
            replace_existing=True
        )
        
        # 21:05 KST Daily Reflection Job
        self.scheduler.add_job(
            run_reflection_job,
            CronTrigger(hour=21, minute=5, timezone=seoul_tz),
            args=[self.db, self.alert_manager],
            id='reflection_job',
            replace_existing=True
        )
        
        # Sunday 22:00 KST Weekly Model Retraining
        self.scheduler.add_job(
            self.retrain_models,
            CronTrigger(day_of_week='sun', hour=22, minute=0, timezone=seoul_tz),
            id='model_retraining',
            replace_existing=True
        )

        # ── New Intelligence Jobs ────────────────────────────────────────

        # Every 15 min: Sector shift analysis
        self.scheduler.add_job(
            self.sector_analyzer.run_analysis,
            'interval', minutes=15,
            id='sector_analysis',
            replace_existing=True
        )

        # Every hour: IPO scan
        self.scheduler.add_job(
            self.ipo_detector.scan_for_ipos,
            'interval', minutes=60,
            id='ipo_scan',
            replace_existing=True
        )

        # Every 6 hours: Scan for upcoming events
        self.scheduler.add_job(
            self.event_tracker.scan_upcoming_events,
            'interval', hours=6,
            id='event_scan',
            replace_existing=True
        )

        # Every 4 hours: Trend forecasting + Macro themes
        async def run_trend_forecasting():
            await self.trend_forecaster.generate_forecasts()
            await self.trend_forecaster.generate_and_cache_macro_themes()

        self.scheduler.add_job(
            run_trend_forecasting,
            CronTrigger(hour='2,6,10,14,18,22', minute=0, timezone=seoul_tz),
            id='trend_forecasting',
            replace_existing=True
        )

        # Daily at 00:30 KST: Sector daily snapshot
        self.scheduler.add_job(
            self.sector_analyzer.capture_daily_snapshot,
            CronTrigger(hour=0, minute=30, timezone=seoul_tz),
            id='sector_snapshot',
            replace_existing=True
        )

        # Startup catch-up: run missed daily jobs once after a short delay
        self.scheduler.add_job(
            self._startup_catchup,
            'date',
            run_date=datetime.datetime.now() + datetime.timedelta(seconds=10),
            id='startup_catchup',
            replace_existing=True
        )

        self.scheduler.start()
        log.info("orchestrator.started", interval_minutes=interval_minutes)
        
    def stop(self) -> None:
        """Stops the scheduler."""
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
            log.info("orchestrator.stopped")

    async def _startup_catchup(self):
        """Check for and run any missed daily prediction generation or resolution jobs."""
        import datetime as dt
        seoul_tz = ZoneInfo("Asia/Seoul")
        today_str = dt.datetime.now(seoul_tz).strftime("%Y-%m-%d")

        try:
            tracked = self.db.get_tracked_tickers()
            if not tracked:
                log.info("orchestrator.startup_catchup.no_tickers")
                return

            # Check if daily predictions were generated today
            predictions_today = False
            with self.db.connection() as conn:
                count = conn.execute(
                    "SELECT COUNT(*) as c FROM predictions_cache WHERE date = ?",
                    (today_str,)
                ).fetchone()["c"]
                predictions_today = count > 0

            if not predictions_today:
                log.info("orchestrator.startup_catchup.missed_predictions",
                         tickers=len(tracked))
                try:
                    await run_daily_predictions(self.db, self.alert_manager)
                except Exception as e:
                    log.error("orchestrator.startup_catchup.predictions_failed", error=str(e))
            else:
                log.info("orchestrator.startup_catchup.predictions_ok")

            # Check if there are unresolved predictions past their resolve_after date
            unresolved = self.db.get_unresolved_predictions()
            if unresolved:
                log.info("orchestrator.startup_catchup.unresolved_found",
                         count=len(unresolved))
                try:
                    await self.resolve_predictions()
                except Exception as e:
                    log.error("orchestrator.startup_catchup.resolve_failed", error=str(e))
            else:
                log.info("orchestrator.startup_catchup.resolve_ok")

        except Exception as e:
            log.error("orchestrator.startup_catchup_error", error=str(e))

    @staticmethod
    def _horizon_to_yahoo_range(horizon_days: int) -> str:
        """Map prediction horizon_days to a Yahoo Finance chart range."""
        if horizon_days <= 5:
            return "5d"
        elif horizon_days <= 21:
            return "1mo"
        elif horizon_days <= 63:
            return "3mo"
        elif horizon_days <= 126:
            return "6mo"
        else:
            return "1y"

    async def resolve_predictions(self):
        """Fetches unresolved predictions, checks actual prices via Yahoo Finance, updates DB."""
        try:
            import httpx
            unresolved = self.db.get_unresolved_predictions()
            if not unresolved:
                return

            async with httpx.AsyncClient(timeout=10) as client:
                for p in unresolved:
                    try:
                        ticker = p["ticker"]
                        horizon_days = p.get("horizon_days", 5)
                        yahoo_range = self._horizon_to_yahoo_range(horizon_days)

                        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
                        params = {"range": yahoo_range, "interval": "1d"}
                        headers = {"User-Agent": "Mozilla/5.0"}
                        resp = await client.get(url, params=params, headers=headers)
                        data = resp.json()

                        chart = data.get("chart", {}).get("result", [])
                        if not chart:
                            continue

                        closes = chart[0].get("indicators", {}).get("quote", [{}])[0].get("close", [])
                        closes = [c for c in closes if c is not None]

                        if len(closes) >= 2:
                            # Compare first available close to last close over the horizon window
                            actual_direction = "UP" if closes[-1] > closes[0] else "DOWN"
                            actual_change_pct = ((closes[-1] - closes[0]) / closes[0]) * 100
                            is_correct = (actual_direction == p["predicted_direction"])
                            self.db.resolve_prediction(p["id"], actual_direction, actual_change_pct, is_correct)
                            log.info("orchestrator.prediction_resolved",
                                     ticker=ticker, horizon_days=horizon_days,
                                     predicted=p["predicted_direction"], actual=actual_direction,
                                     correct=is_correct)
                    except Exception as e:
                        log.error("orchestrator.resolve_prediction_failed", ticker=p.get("ticker", "?"), error=str(e))

            log.info("orchestrator.predictions_resolved")
        except Exception as e:
            log.error("orchestrator.resolve_predictions_error", error=str(e))

    async def retrain_models(self):
        """Retrains models for all tracked tickers and sends a CV metrics report via Telegram."""
        try:
            from bot.formatters import escape_html
            from pipeline.predictor import StockPredictor
            tracked = self.db.get_tracked_tickers()
            if not tracked:
                return
                
            predictor = StockPredictor(self.db)
            training_results = []  # (ticker, cv_metrics) tuples
            failed_tickers = []
            
            for t in tracked:
                try:
                    _path, cv_metrics = await predictor.train_model(t, scope="per_ticker")
                    training_results.append((t, cv_metrics))
                except Exception as e:
                    log.error("orchestrator.retrain_model_failed", ticker=t, error=str(e))
                    failed_tickers.append((t, str(e)))
                    
            log.info("orchestrator.models_retrained")
            
            # Send Telegram summary report
            if self.alert_manager and (training_results or failed_tickers):
                text = "<b>🧠 Weekly Model Retraining Complete</b>\n\n"
                
                if training_results:
                    text += "<pre>"
                    text += f"{'Ticker':<8}| {'CV Acc':>7} | {'Brier':>6} | {'AUC':>6}\n"
                    text += f"{'─'*8}|{'─'*9}|{'─'*8}|{'─'*8}\n"
                    for ticker, m in training_results:
                        acc_str = f"{m['accuracy_mean']*100:.1f}%"
                        brier_str = f"{m['brier_mean']:.3f}"
                        auc_str = f"{m['auc_mean']:.3f}"
                        text += f"{ticker:<8}| {acc_str:>7} | {brier_str:>6} | {auc_str:>6}\n"
                    text += "</pre>\n"
                    text += f"<i>Calibrated with Platt Scaling (5-fold TS-CV, {training_results[0][1]['n_samples']}+ samples)</i>\n"
                
                if failed_tickers:
                    text += "\n⚠️ <b>Failed:</b>\n"
                    for ticker, err in failed_tickers:
                        text += f"• {ticker}: {escape_html(err[:80])}\n"
                
                try:
                    await self.alert_manager.bot.send_message(
                        chat_id=self.alert_manager.chat_id,
                        text=text,
                        parse_mode="HTML"
                    )
                except Exception as e:
                    log.error("orchestrator.retrain_report_send_failed", error=str(e))
        except Exception as e:
            log.error("orchestrator.retrain_models_error", error=str(e))


async def run_daily_predictions(db, alert_manager=None):
    """Generates and caches multi-agent predictions for all tracked tickers daily."""
    from pipeline.predictor import StockPredictor
    log.info("Starting daily multi-agent predictions generation...")
    tracked = db.get_tracked_tickers()
    if not tracked:
        return
    predictor = StockPredictor(db)
    for t in tracked:
        try:
            # Check if there's already a valid cache within the 5-day window
            if db.get_cached_advisory(t, days=5):
                log.info(f"Valid 5-day cache exists for {t}. Skipping expensive LLM generation.")
                continue
                
            await predictor.predict_with_agents(t)
            log.info(f"Daily multi-agent prediction cached for {t}")
        except Exception as e:
            log.error(f"Daily multi-agent prediction failed for {t}: {e}")

async def run_reflection_job(db, alert_manager=None):
    """Analyzes newly resolved multi-agent predictions and extracts lessons learned."""
    from config.llm import get_deepseek_client
    log.info("Starting reflection job...")
    
    with db.connection() as conn:
        # Get predictions that are resolved, are multi-agent, and not yet in reflection_log
        predictions_to_reflect = conn.execute('''
            SELECT p.id, p.ticker, p.predicted_direction, p.confidence, p.llm_narrative, p.actual_direction, p.actual_change_pct, p.is_correct
            FROM predictions p
            LEFT JOIN reflection_log r ON p.id = r.prediction_id
            WHERE p.is_correct IS NOT NULL 
              AND p.model_type = 'multi_agent'
              AND r.id IS NULL
        ''').fetchall()
        
    if not predictions_to_reflect:
        return
        
    client = get_deepseek_client()
    if not client:
        log.warning("DeepSeek API key not configured for Reflection Job.")
        return
        
    today_str = datetime.datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d")
        
    for p in predictions_to_reflect:
        ticker = p["ticker"]
        pred_id = p["id"]
        pred_dir = p["predicted_direction"]
        actual_dir = p["actual_direction"]
        
        # safely handle None values in DB
        actual_change = p["actual_change_pct"] or 0.0
        narrative = p["llm_narrative"] or "No narrative"
        is_correct = bool(p["is_correct"])
        confidence = p["confidence"] or 0.5
        
        prompt = f"""
        You are a Reflection Agent reviewing a past stock prediction. Your job is to extract a concise, actionable lesson.

        Ticker: {ticker}
        Our Multi-Agent System predicted {pred_dir} with {int(confidence*100)}% confidence.
        The actual market direction was {actual_dir} ({actual_change:.2f}% change).
        The prediction was {'CORRECT' if is_correct else 'INCORRECT'}.

        Here is the original reasoning that led to our prediction:
        ---
        {narrative}
        ---

        Analyze WHY the reasoning succeeded or failed. Categorize the failure/success mode, then write a 1-2 sentence lesson.

        FAILURE MODES (use only if the prediction was wrong):
        - "overconfidence": Confidence was too high relative to evidence quality
        - "missed_catalyst": A key event (earnings, macro data, news) was missed or underweighted
        - "model_blindness": The ML model pointed one way but debate/trader ignored it
        - "black_swan": Unpredictable external event that no amount of analysis could foresee
        - "correct_process": The reasoning was sound but the market moved randomly (process win, outcome loss)

        SUCCESS MODES (use only if the prediction was right):
        - "catalyst_capture": Key catalysts were correctly identified and weighted
        - "risk_identification": Risks were properly flagged and the direction call was right
        - "contrarian_win": The system went against consensus and was right

        Respond with a valid JSON object (no markdown, no backticks):
        {{
            "lesson_learned": "1-2 sentence actionable lesson",
            "failure_mode": "one of the modes above, or 'none' if correct",
            "success_mode": "one of the modes above, or 'none' if incorrect",
            "actionable_fix": "What to change next time (1 sentence)",
            "should_adjust_strategy": true or false
        }}
        """

        try:
            # Use flash model for reflection — this is a simple summarization task,
            # not a reasoning-heavy analysis. Saves ~90% vs deepseek-v4-pro.
            model_name = settings.deepseek_model_classifier  # deepseek-v4-flash
            response = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You analyze past stock predictions and extract structured, actionable lessons. Output JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            prompt_tokens = response.usage.prompt_tokens if response.usage else 0
            completion_tokens = response.usage.completion_tokens if response.usage else 0
            db.log_llm_usage(model_name=model_name, operation="reflection", prompt_tokens=prompt_tokens, candidate_tokens=completion_tokens)

            raw = response.choices[0].message.content.strip()
            if raw.startswith("```json"): raw = raw[7:]
            if raw.startswith("```"): raw = raw[3:]
            if raw.endswith("```"): raw = raw[:-3]

            try:
                data = json.loads(raw.strip())
                lesson = data.get("lesson_learned", raw)
                failure_mode = data.get("failure_mode", "none")
                actionable_fix = data.get("actionable_fix", "")
                should_adjust = data.get("should_adjust_strategy", False)
            except (json.JSONDecodeError, AttributeError):
                lesson = raw
                failure_mode = "none"
                actionable_fix = ""
                should_adjust = False

            # Insert into reflection_log with structured metadata in tags
            if hasattr(db, "insert_reflection"):
                tags_json = json.dumps({
                    "failure_mode": failure_mode,
                    "actionable_fix": actionable_fix,
                    "should_adjust_strategy": should_adjust
                })
                db.insert_reflection(ticker, pred_id, today_str, lesson, is_correct, tags=tags_json)
                log.info(f"Reflection logged for {ticker} (Pred ID {pred_id}, mode={failure_mode})")
                
            if not is_correct:
                log.info(f"Prediction for {ticker} was incorrect. Triggering new prediction...")
                from pipeline.predictor import StockPredictor
                predictor = StockPredictor(db)
                new_advisory = await predictor.predict_with_agents(ticker)
                
                if alert_manager:
                    verdict = new_advisory.get('final_advisory', 'No plan found')
                    message = (
                        f"⚠️ *Correction for {ticker}*\n\n"
                        f"*Why we were wrong:*\n{lesson}\n\n"
                        f"*Updated Plan:*\n{verdict}"
                    )
                    await alert_manager.bot.send_message(
                        chat_id=alert_manager.chat_id,
                        text=message,
                        parse_mode="Markdown"
                    )
        except Exception as e:
            log.error(f"Failed to generate reflection for {ticker}: {e}")
