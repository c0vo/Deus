"""
Deus — Pipeline Orchestrator

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
from pydantic import BaseModel
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from config.logging_config import get_logger
from config.settings import settings
from config.llm import complete, is_llm_configured, is_transient, parse_structured
from config.usage import track_llm
from data.database import Database
from pipeline.aggregator import NewsAggregator
from pipeline.classifier import ArticleClassifier
from pipeline.ranker import ArticleRanker
from pipeline.embedder import Embedder
from pipeline.market_scanner import MarketScanner
from pipeline.predictor import HORIZON_LABELS, StockPredictor
from pipeline.price_feed import PriceFeed
from pipeline.sector_analyzer import SectorAnalyzer
from pipeline.ipo_detector import IPODetector
from pipeline.geo_tagger import GeoTagger
from pipeline.event_tracker import EventTracker
from pipeline.trend_forecaster import TrendForecaster
from pipeline.thesis_engine import ThesisEngine
from pipeline.insider_tracker import InsiderTracker
from pipeline.kr_flows import KrFlowTracker
from pipeline.darkpool import DarkPoolTracker
from pipeline.market_regime import MarketRegimeTracker
from pipeline.options_flow import OptionsSnapshotTracker
from pipeline.analyst_ratings import AnalystRatingsTracker
from pipeline.technical_rating import TechnicalRatingTracker
from bot.alerts import AlertManager
from api.sse_manager import event_bus

log = get_logger(__name__)

# Reflection job bounds. The job scans every resolved multi-agent prediction that
# has no lesson yet, so without these the whole historical backlog is processed in
# a single tick.
REFLECTION_LOOKBACK_DAYS = 14
REFLECTION_BATCH_LIMIT = 25

# How many wrong predictions get a fresh multi-agent debate per run. Reflection
# itself is one cheap call per prediction, but each correction is a full debate,
# so an uncapped batch could spend ~150 calls on the priciest models in one
# nightly job. Corrections are ranked by confidence — the misses we were most
# certain about are the ones worth re-running.
REFLECTION_REPREDICT_LIMIT = 3

# APScheduler's default misfire_grace_time is ONE SECOND: a cron job whose fire
# time passes while the event loop is busy is dropped for the day, not run late.
# On the Termux deployment that is the normal case, not an edge case — Android
# Doze suspends timers whenever the phone is idle, so a 06:30 job typically
# wakes minutes late and was silently skipped every morning. Daily thesis jobs
# are "sometime this morning" work, so a wide window costs nothing; coalesce
# (on by default) still collapses a backlog into a single run.
DAILY_MISFIRE_GRACE_SECONDS = 6 * 3600

# When the morning thesis is due, in Asia/Seoul. Shared by the cron trigger and
# the start-up catch-up, which must not fire before the scheduled time or a
# worker booted at 03:00 would generate one thesis then and a second at 06:30.
THESIS_GENERATION_HOUR = 6
THESIS_GENERATION_MINUTE = 30


class ReflectionLesson(BaseModel):
    """One extracted lesson from a resolved prediction."""

    lesson_learned: str
    failure_mode: str = "none"
    success_mode: str = "none"
    actionable_fix: str = ""
    should_adjust_strategy: bool = False


class PipelineOrchestrator:
    """Orchestrates the periodic execution of the Deus pipeline."""

    def __init__(self, db: Database, alert_manager: Optional[AlertManager] = None):
        self.db = db
        self.alert_manager = alert_manager

        self.aggregator = NewsAggregator(db=self.db)
        self.classifier = ArticleClassifier(db=self.db)
        self.ranker = ArticleRanker(db=self.db)
        self.embedder = Embedder(db=self.db)
        self.market_scanner = MarketScanner(db=self.db, alert_manager=self.alert_manager)
        self.price_feed = PriceFeed(db=self.db)
        self.sector_analyzer = SectorAnalyzer(db=self.db, alert_manager=self.alert_manager)
        self.ipo_detector = IPODetector(db=self.db)
        self.geo_tagger = GeoTagger(db=self.db)
        self.event_tracker = EventTracker(db=self.db, alert_manager=self.alert_manager)
        self.trend_forecaster = TrendForecaster(db=self.db)
        self.insider_tracker = InsiderTracker(db=self.db)
        self.kr_flow_tracker = KrFlowTracker(db=self.db)
        self.darkpool_tracker = DarkPoolTracker(db=self.db)
        self.market_regime_tracker = MarketRegimeTracker(db=self.db)
        self.options_tracker = OptionsSnapshotTracker(db=self.db)
        self.analyst_tracker = AnalystRatingsTracker(db=self.db)
        self.technical_rating_tracker = TechnicalRatingTracker(db=self.db)

        self.thesis_engine = ThesisEngine(db=self.db)

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

        # IPO and event scanning deliberately do NOT run here. Both are already
        # registered as their own jobs (ipo_scan hourly, event_scan every 6h);
        # running them inline as well meant they fired once per pipeline cycle,
        # which is over an order of magnitude more often than intended.
        log.info("orchestrator.cycle_complete", duration_seconds=round(duration, 2))

    async def _process_batch(self) -> None:
        """Process a batch of articles (Embed -> Classify -> Rank -> Alert)."""
        try:

            # Step 2: Embed unembedded articles FIRST (for deduplication).
            #
            # Rows already flagged noise by the aggregator's pre-insert filter
            # are excluded — they are never classified, ranked or searched, so
            # buying a vector for them is pure waste.
            with self.db.connection() as conn:
                unembedded = conn.execute(
                    """
                    SELECT * FROM articles
                    WHERE embedding IS NULL
                      AND (event_type IS NULL OR event_type != 'noise')
                      AND COALESCE(embed_attempts, 0) < 3
                    LIMIT 50
                    """
                ).fetchall()

            if unembedded:
                candidates = [self.db.row_to_article(row) for row in unembedded]

                # Apply the classifier's free keyword heuristic here rather than
                # one step later. It is the same verdict classify() would reach,
                # and reaching it now avoids paying to embed a row that is about
                # to be marked noise anyway.
                to_embed, prefiltered = [], []
                for a in candidates:
                    (to_embed if self.classifier.should_classify(a) else prefiltered).append(a)

                for a in prefiltered:
                    self.db.update_classification(
                        article_id=a.id,
                        event_type="noise",
                        sentiment_score=0.0,
                        urgency="low",
                        suggested_direction="neutral",
                        affected_sectors=[],
                        affected_tickers=[],
                        classification_summary=ArticleClassifier.NOISE_SUMMARY,
                    )

                if to_embed:
                    log.info(
                        "orchestrator.embed",
                        count=len(to_embed), prefiltered_noise=len(prefiltered),
                    )
                    self._cycle_counts["embedded"] += len(to_embed)

                    # One request per batch rather than one per article. The old
                    # gather-of-5 was concurrency, not batching — still 50 HTTP
                    # round-trips, each with its own 3x retry on top.
                    vectors = await self.embedder.embed_articles(to_embed)

                    for article, embedding in zip(to_embed, vectors):
                        if embedding is not None:
                            self.db.update_embedding(article.id, embedding)
                        else:
                            # Left NULL so a later pass retries it, bounded by
                            # embed_attempts. The previous zero-vector
                            # placeholder marked the row embedded forever and
                            # poisoned every cosine comparison it entered.
                            self.db.record_embed_failure(article.id)

            # Step 3: Classify unclassified articles (with Semantic Deduplication)
            #
            # Dedup runs as its own pass over the whole batch before any
            # classification, then the survivors go out in batched calls. Dedup
            # makes no LLM calls — it reads the article's own embedding and
            # older persisted rows — so hoisting it costs nothing and still
            # suppresses duplicates before the model sees them, as before.
            unclassified_data = self.db.get_unclassified_articles(limit=20)
            log.info("orchestrator.classify", count=len(unclassified_data))
            classified_count = 0

            articles = [self.db.row_to_article(r) for r in unclassified_data]
            needs_llm = [a for a in articles if not self._absorb_duplicate(a)]

            size = settings.classify_batch_size
            for i in range(0, len(needs_llm), size):
                chunk = needs_llm[i:i + size]
                if await self._classify_chunk_with_retry(chunk):
                    classified_count += self._persist_classifications(chunk)

            self._cycle_counts["classified"] += classified_count

            # Step 4: Rank unranked articles (skip low-signal articles to save LLM calls)
            await self._rank_pending()

        except Exception as e:
            log.error("orchestrator.batch_failed", error=str(e))
            self._cycle_counts["errors"] += 1

    async def _classify_chunk_with_retry(self, chunk: list) -> bool:
        """
        Classifies a chunk, retrying only infrastructure faults.

        Returns True if the results are safe to persist. A False means every
        attempt hit a transient fault, so the rows are left with event_type
        NULL for a later pass rather than being written off as 'error' — an
        outage should not permanently discard a batch of articles.
        """
        for attempt in range(settings.llm_max_retries):
            try:
                await self.classifier.classify_batch(chunk)
                return True
            except Exception as e:
                if not is_transient(e) or attempt == settings.llm_max_retries - 1:
                    log.error(
                        "orchestrator.classify_batch_failed",
                        error=str(e), count=len(chunk),
                        transient=is_transient(e), attempts=attempt + 1,
                    )
                    return not is_transient(e)
                # Exponential, not a flat 2s — a rate limit needs room to clear.
                await asyncio.sleep(2 ** attempt)
        return False

    def _absorb_duplicate(self, article) -> bool:
        """
        Semantic-dedup a single article. Returns True if it was absorbed into an
        already-classified near-duplicate, in which case it needs no LLM call.

        No LLM calls here — this reads the article's own embedding and compares
        against older persisted rows only.
        """
        with self.db.connection() as conn:
            row = conn.execute(
                "SELECT embedding FROM articles WHERE id = ?", (article.id,)
            ).fetchone()
        if not (row and row["embedding"]):
            return False

        article_emb = np.frombuffer(row["embedding"], dtype=np.float32)

        # Windowed nearest-neighbour lookup. Bounding by publish date keeps an
        # old story from suppressing a current one and stops every candidate
        # query being a full scan of the corpus.
        match = self.db.find_duplicate(
            article_id=article.id,
            embedding=article_emb,
            published_at=article.published_at.isoformat()
            if hasattr(article.published_at, "isoformat")
            else str(article.published_at),
            window_days=settings.dedup_window_days,
            threshold=settings.dedup_similarity_threshold,
        )

        if not match:
            self.db.mark_dedup_checked(article.id)
            return False

        best_match_id, highest_sim = match
        with self.db.connection() as conn:
            source_row = conn.execute(
                "SELECT * FROM articles WHERE id = ?", (best_match_id,)
            ).fetchone()

        if not (source_row and source_row["event_type"]):
            # Nearest neighbour exists but isn't classified yet, so there is
            # nothing to inherit. Leave it for a later pass.
            log.debug(
                "orchestrator.semantic_dedup.match_unclassified",
                article_id=article.id,
                sim=round(highest_sim, 3),
            )
            return False

        log.info(
            "orchestrator.semantic_dedup.flagged",
            article_id=article.id,
            source_id=best_match_id,
            sim=round(highest_sim, 3),
        )
        # Inherit the canonical article's classification so the row stays
        # queryable, then flag it. Flagged rows are excluded from feeds,
        # trending and coverage counts — and deliberately NOT written to
        # ticker_mentions, which is what was inflating trending counts when the
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
        return True

    def _persist_classifications(self, chunk: list) -> int:
        """
        Writes back a classified chunk. Returns how many succeeded.

        Articles the model never answered for come back with event_type still
        unset; those are marked 'error' so they do not requeue forever.
        """
        succeeded = 0
        for article in chunk:
            if article.event_type:
                self.db.update_classification(
                    article_id=article.id,
                    event_type=article.event_type,
                    sentiment_score=article.sentiment_score,
                    urgency=article.urgency,
                    suggested_direction=article.suggested_direction,
                    affected_sectors=article.affected_sectors,
                    affected_tickers=article.affected_tickers,
                    classification_summary=article.classification_summary,
                    countries=article.countries or [],
                )
                self.db.insert_ticker_mentions(
                    article_id=article.id,
                    tickers=article.affected_tickers,
                    sentiment_score=article.sentiment_score,
                    urgency=article.urgency,
                )
                succeeded += 1
            else:
                # Mark as failed so it doesn't infinite loop
                self.db.update_classification(
                    article_id=article.id,
                    event_type="error",
                    sentiment_score=0.0,
                    urgency="low",
                    suggested_direction="neutral",
                    affected_sectors=[],
                    affected_tickers=[],
                    classification_summary="Classification failed due to parsing error.",
                )

            asyncio.create_task(
                event_bus.publish("new_articles", {"articles": [article.model_dump()]})
            )

        return succeeded

    async def _rank_pending(self) -> None:
        """Batch-rank classified articles that have no importance score yet."""
        try:
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

                    # Retry infrastructure faults only. The old loop retried on
                    # any falsy result, so an unparseable response cost three
                    # identical calls at temperature 0.1.
                    ranked_articles = []
                    for attempt in range(settings.llm_max_retries):
                        try:
                            ranked_articles = await self.ranker.rank_batch(articles_to_rank)
                            break
                        except Exception as e:
                            if attempt == settings.llm_max_retries - 1:
                                log.error(
                                    "orchestrator.rank_failed",
                                    error=str(e), count=len(articles_to_rank),
                                )
                                ranked_articles = []
                                break
                            await asyncio.sleep(2 ** attempt)

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
            from bot.formatters import EMPTY_BRIEFING_TEXT, chunk_html, render_briefing
            from data.taxonomy import BRIEFING_MIN_IMPORTANCE, select_briefing_lanes

            rows = self.db.get_briefing_candidates(
                hours=24, min_importance=BRIEFING_MIN_IMPORTANCE, limit=40
            )
            lanes = select_briefing_lanes(rows)
            # Always send, even when nothing clears the floor. A quiet news day
            # and a job that silently died look identical from the chat.
            text = render_briefing(lanes) if lanes else EMPTY_BRIEFING_TEXT

            for chunk in chunk_html(text):
                await self.alert_manager.bot.send_message(
                    chat_id=self.alert_manager.chat_id,
                    text=chunk,
                    parse_mode="HTML",
                    disable_web_page_preview=True
                )
            log.info("orchestrator.daily_briefing_sent", articles=sum(len(a) for _, a in lanes))
        except Exception as e:
            log.error("orchestrator.daily_briefing_failed", error=str(e))

    async def send_daily_advisor(self):
        """Sends daily hold/sell advice for tracked tickers."""
        if not self.alert_manager:
            return
            
        try:
            from bot.formatters import escape_html
            from data.models import TickerNote, notes_to_dict

            tracked = self.db.get_tracked_tickers()
            if not tracked:
                return
                
            all_ticker_contexts = {}
            for t in tracked:
                summaries = self.db.get_recent_summaries_for_ticker(t, hours=24)
                if summaries:
                    all_ticker_contexts[t] = summaries
            
            ai_summaries = {}
            if is_llm_configured() and settings.model_daily_advisor and all_ticker_contexts:
                prompt = (
                    "You are a professional Wall Street advisor reviewing your client's portfolio.\n"
                    "Below are the client's tracked tickers with their recent news from the past 24 hours.\n"
                    "For EACH ticker, recommend HOLD or SELL for tomorrow based ONLY on the news context.\n"
                    "Guidelines:\n"
                    "- HOLD: News is neutral-to-positive, or no significant negative catalyst. Default to HOLD unless there's a clear reason to sell.\n"
                    "- SELL: Specific negative catalyst in the news (earnings miss, downgrade, regulatory issue, macro headwind directly impacting the ticker).\n"
                    "Return one entry per ticker in the required response schema. Each "
                    "summary must read 'HOLD - [1 sentence reason]' or 'SELL - [1 sentence reason]'.\n"
                    "Plain text only — no markdown, no HTML.\n\n"
                )
                for tk, sums in all_ticker_contexts.items():
                    prompt += f"Ticker: {tk}\nContext:\n" + "\n".join(f"- {s}" for s in sums) + "\n\n"

                with track_llm(self.db, settings.model_daily_advisor,
                               "daily_advisor_batch",
                               prompt_text=prompt, store_text=True) as u:
                    u.response = response = await complete(
                        model=settings.model_daily_advisor,
                        prompt=prompt,
                        schema=list[TickerNote],
                        reasoning="low",
                    )

                if isinstance(response.parsed, list):
                    ai_summaries = notes_to_dict(response.parsed)
                else:
                    ai_summaries = notes_to_dict(parse_structured(response.text, list[TickerNote]))

            text = "<b>🎯 Tracked Tickers Daily Advisor</b>\n\n"
            text += "Based on today's news flow, here is my outlook for your portfolio tomorrow:\n\n"
            
            for t in tracked:
                if t in all_ticker_contexts:
                    advice = ai_summaries.get(t.upper(), "HOLD - Unable to generate advice.")
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

    async def train_missing_models(self) -> None:
        """
        Fill in one missing (ticker, horizon) prediction per run.

        /api/markets renders a TRAINING badge for any horizon with no live
        prediction and, as of the process split, never trains anything itself.
        This is the other half of that contract — without it those badges would
        stay TRAINING forever.

        Deliberately one pair per run. A newly watchlisted ticker needs four
        models, and training them back to back would monopolise this process
        for as long as it took; spread out, the grid fills in over a few hours
        while everything else keeps running.
        """
        try:
            tracked = await asyncio.to_thread(self.db.get_tracked_tickers)
            if not tracked:
                return

            predictor = StockPredictor(self.db)
            for ticker in tracked:
                active = await asyncio.to_thread(
                    self.db.get_recent_predictions, ticker, 20, True
                )
                covered = {p.get("horizon_days") for p in active}

                for horizon_days in HORIZON_LABELS:
                    if horizon_days in covered:
                        continue

                    model, _scope = await asyncio.to_thread(
                        predictor._load_model, ticker, horizon_days
                    )
                    if model is None:
                        log.info("orchestrator.training_missing_model",
                                 ticker=ticker, horizon_days=horizon_days)
                        await predictor.train_model(
                            ticker, scope="per_ticker", horizon_days=horizon_days
                        )
                        model, _scope = await asyncio.to_thread(
                            predictor._load_model, ticker, horizon_days
                        )

                    if model is None:
                        # Training did not produce a loadable model — usually
                        # too little price history. Stop rather than fall
                        # through to predict(), whose llm_only path would spend
                        # a live LLM call on every cycle for a ticker that
                        # cannot be modelled.
                        log.warning("orchestrator.missing_model_unfilled",
                                    ticker=ticker, horizon_days=horizon_days)
                        return

                    await predictor.predict(
                        ticker, horizon_days=horizon_days, fast_fallback=False
                    )
                    log.info("orchestrator.missing_model_filled",
                             ticker=ticker, horizon_days=horizon_days)
                    return  # one pair per run, by design

        except Exception as e:
            log.error("orchestrator.train_missing_models_failed", error=str(e))

    async def refresh_prices(self) -> None:
        """Refresh last-known quotes so /api/markets never calls Yahoo inline."""
        try:
            await self.price_feed.refresh()
        except Exception as e:
            log.error("orchestrator.price_refresh_failed", error=str(e))

    async def sync_price_history(self) -> None:
        """Keep price_history current for the whole watchlist.

        Paired with darkpool_scan rather than with the quote refresh above.
        The dark-pool card divides FINRA off-exchange volume by the consolidated
        volume this writes, joined on the session date, so the two sides have to
        advance on the same cadence — otherwise the FINRA half keeps arriving
        for tickers whose price rows stopped, and the ratio silently goes NULL.
        """
        try:
            await self.price_feed.refresh_history()
        except Exception as e:
            log.error("orchestrator.price_history_sync_failed", error=str(e))

    async def trim_sse_outbox(self) -> None:
        """
        Drop delivered rows from the sse_events relay table.

        The API process tails this table roughly once a second, so anything
        older than a few minutes has already been read or was missed while no
        dashboard was open. Left alone it would grow unbounded.
        """
        try:
            removed = await asyncio.to_thread(self.db.trim_sse_events, 10)
            if removed:
                log.info("orchestrator.sse_outbox_trimmed", count=removed)
        except Exception as e:
            log.error("orchestrator.sse_outbox_trim_failed", error=str(e))

    def start(self, interval_minutes: int = 5) -> None:
        """Starts the periodic scheduler."""
        # First cycle is delayed rather than immediate. A full
        # fetch → classify → embed → rank cycle fired at t=0 competes with the
        # dashboard's very first page load, which on a phone is the difference
        # between a responsive app and one that appears not to load at all.
        self.scheduler.add_job(
            self.run_pipeline_cycle,
            'date',
            run_date=datetime.datetime.now() + datetime.timedelta(
                seconds=settings.pipeline_startup_delay_seconds
            )
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
        # Quotes for /api/markets. Cheap (one HTTP GET per ticker, no LLM), so
        # unlike the pipeline cycle this warms up almost immediately — the
        # dashboard needs prices to be useful at all.
        self.scheduler.add_job(
            self.refresh_prices,
            'date',
            run_date=datetime.datetime.now() + datetime.timedelta(seconds=5),
            id='price_feed_warmup',
            replace_existing=True
        )
        self.scheduler.add_job(
            self.refresh_prices,
            'interval',
            seconds=settings.price_refresh_seconds,
            id='price_feed',
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
        # The SSE outbox is a relay to the API process, not a log. Anything
        # older than a few minutes has already been delivered or missed.
        self.scheduler.add_job(
            self.trim_sse_outbox,
            'interval',
            minutes=10,
            id='sse_outbox_trim',
            replace_existing=True
        )
        # Closes the loop on the TRAINING badges /api/markets renders. One
        # (ticker, horizon) pair per run — see train_missing_models.
        self.scheduler.add_job(
            self.train_missing_models,
            'interval',
            minutes=20,
            id='train_missing_models',
            replace_existing=True
        )
        
        seoul_tz = ZoneInfo("Asia/Seoul")

        # Daily Briefing — 5:00 AM KST by default, configurable via .env
        self.scheduler.add_job(
            self.send_daily_briefing,
            CronTrigger(
                hour=settings.briefing_hour,
                minute=settings.briefing_minute,
                timezone=seoul_tz
            ),
            id='daily_briefing',
            replace_existing=True
        )

        # 4:30 AM KST — retire IPOs that have already listed. Placed ahead of the
        # default 5:00 briefing so the brief never cites a stale listing; moving
        # BRIEFING_HOUR earlier than 4:30 breaks that ordering.
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

        # Every hour: IPO scan — the LLM pass over recent news, then the Finnhub
        # calendar. Finnhub runs second so its confirmed dates and tickers land
        # on top of anything the news pass guessed in the same run.
        async def run_ipo_scan():
            await self.ipo_detector.scan_for_ipos()
            await self.ipo_detector.scan_finnhub_ipos()

        self.scheduler.add_job(
            run_ipo_scan,
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

        # 07:00 KST daily: SEC insider (Form 4) and >5% stake (13D/G) disclosures.
        # EDGAR accepts filings until 22:00 ET, which is ~11:00 KST the same
        # morning, so this picks up the previous US session's filings and lands
        # ahead of daily_predictions at 08:30.
        async def run_insider_scan():
            tickers = self.db.get_tracked_tickers()
            if tickers:
                await self.insider_tracker.sync_all(tickers)

        # Passes timezone= explicitly, like every other cron job here. The
        # scheduler itself is constructed without a default timezone, so the
        # bare 'cron' form this used to use fired at whatever the host's local
        # time was — correct on the Termux target only by coincidence.
        self.scheduler.add_job(
            run_insider_scan,
            CronTrigger(hour=7, minute=0, timezone=seoul_tz),
            id='insider_scan',
            replace_existing=True
        )

        # 18:00 KST daily: Korean investor flows, after the KRX close (15:30)
        # and after the day's figures are published.
        async def run_kr_flow_scan():
            tickers = self.db.get_tracked_tickers()
            if tickers:
                await self.kr_flow_tracker.sync_all(tickers, days=30)

        self.scheduler.add_job(
            run_kr_flow_scan,
            CronTrigger(hour=18, minute=0, timezone=seoul_tz),
            id='kr_flow_scan',
            replace_existing=True
        )

        # 07:15 KST daily: daily OHLCV bars for the whole watchlist.
        #
        # Fifteen minutes ahead of darkpool_scan, and largely for its benefit:
        # off-exchange share is the FINRA figure over the consolidated volume
        # this stores, joined on session_date, so the price side goes first and
        # the day's FINRA rows land on a price row that already exists. Also
        # puts fresh technicals under daily_predictions at 08:30, which until
        # now relied on the predictor fetching them inline per ticker.
        #
        # Quotes still refresh on their own few-minute timer; this is the daily
        # history table, which is a different row per session and only changes
        # once a day.
        self.scheduler.add_job(
            self.sync_price_history,
            CronTrigger(hour=7, minute=15, timezone=seoul_tz),
            id='price_history_sync',
            replace_existing=True
        )

        # 07:30 and 08:15 KST: FINRA off-exchange (dark pool) volume.
        #
        # FINRA posts the session's file at ~18:00 ET, which is 07:00 KST next
        # morning under EDT but 08:00 under EST — and daily_predictions runs at
        # 08:30. Rather than encode a DST rule in the trigger, this runs twice
        # and leans on the upsert being idempotent: in summer the second pass is
        # a no-op, in winter the first is, and neither races the 08:30 job.
        #
        # An ET-pinned trigger would be tidier and APScheduler handles DST
        # natively, but 18:15 ET lands at 08:15 KST in winter — 15 minutes of
        # margin against a job that has to finish first.
        async def run_darkpool_scan():
            tickers = self.db.get_tracked_tickers()
            if tickers:
                await self.darkpool_tracker.sync_recent(tickers)

        for hour, minute, suffix in ((7, 30, ''), (8, 15, '_retry')):
            self.scheduler.add_job(
                run_darkpool_scan,
                CronTrigger(hour=hour, minute=minute, timezone=seoul_tz),
                id=f'darkpool_scan{suffix}',
                replace_existing=True
            )

        # 07:45 KST daily: market-wide regime series (DIX/GEX, OCC put/call).
        # Both derive from the same US session close as the FINRA file, so they
        # follow it and still land before daily_predictions at 08:30.
        self.scheduler.add_job(
            self.market_regime_tracker.sync,
            CronTrigger(hour=7, minute=45, timezone=seoul_tz),
            id='market_regime_scan',
            replace_existing=True
        )

        # 06:00 KST daily: option-chain snapshot, ~2h after the US close
        # (16:00 ET = 05:00/06:00 KST) so the session's volume and open interest
        # have settled.
        #
        # Deliberately parked away from the 08:30 prediction run rather than
        # beside the other pre-prediction jobs. Nothing reads this table yet, so
        # it has no deadline — and yfinance blocks IPs on bursts, which would
        # take down the predictor's price fetching too. It gets its own quiet
        # window and is skipped entirely when the feature is disabled.
        if settings.options_snapshot_enabled:
            async def run_options_snapshot():
                tickers = self.db.get_tracked_tickers()
                if tickers:
                    await self.options_tracker.snapshot_all(tickers)

            self.scheduler.add_job(
                run_options_snapshot,
                CronTrigger(hour=6, minute=0, timezone=seoul_tz),
                id='options_snapshot',
                replace_existing=True
            )

        # 06:45 KST daily: analyst consensus and price targets.
        #
        # Slotted between the option snapshot at 06:00 and the price chain at
        # 07:15 on purpose. All three read Yahoo through yfinance, and the whole
        # point of the serial-with-delay design inside each collector is undone
        # if two of them overlap and present the burst pattern anyway.
        if settings.analyst_ratings_enabled:
            async def run_analyst_snapshot():
                tickers = self.db.get_tracked_tickers()
                if tickers:
                    await self.analyst_tracker.snapshot_all(tickers)

            self.scheduler.add_job(
                run_analyst_snapshot,
                CronTrigger(hour=6, minute=45, timezone=seoul_tz),
                id='analyst_snapshot',
                replace_existing=True
            )

        # 08:05 KST daily: technical ratings.
        #
        # Must run after price_history_sync at 07:15 — it reads that table and
        # nothing else, so running first would rate every ticker one session
        # stale. Before daily_predictions at 08:30 so the panel and the debate
        # context are current when predictions are generated. No network, so it
        # can sit inside the busy pre-prediction window that the yfinance jobs
        # above have to avoid.
        if settings.technical_rating_enabled:
            async def run_technical_rating():
                tickers = self.db.get_tracked_tickers()
                if tickers:
                    await self.technical_rating_tracker.compute_all(tickers)

            self.scheduler.add_job(
                run_technical_rating,
                CronTrigger(hour=8, minute=5, timezone=seoul_tz),
                id='technical_rating',
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

        if settings.thesis_enabled:
            # 06:30 KST: build one thesis from the most accelerated theme.
            # Ahead of daily_predictions at 08:30 so a name promoted this
            # morning is already tracked when the rest of the stack runs.
            async def run_thesis_generation():
                try:
                    ids = await self.thesis_engine.generate()
                    log.info("orchestrator.thesis_generated", count=len(ids))
                except Exception as e:
                    log.error("orchestrator.thesis_generation_failed", error=str(e))

            self.scheduler.add_job(
                run_thesis_generation,
                CronTrigger(hour=THESIS_GENERATION_HOUR,
                            minute=THESIS_GENERATION_MINUTE,
                            timezone=seoul_tz),
                id='thesis_generation',
                misfire_grace_time=DAILY_MISFIRE_GRACE_SECONDS,
                replace_existing=True
            )

            # 09:10 KST: re-score every live candidate. Costs no LLM calls —
            # prices and SQL only — and is what makes the EARLY -> CROWDED
            # transition visible, which is the actual sell signal. Runs after
            # price_history_sync (07:15), the dark-pool scans (07:30/08:15) and
            # daily_predictions (08:30) so it reads the day's data without
            # racing them for Yahoo.
            async def run_thesis_rescore():
                try:
                    await self.thesis_engine.rescore_all()
                except Exception as e:
                    log.error("orchestrator.thesis_rescore_failed", error=str(e))

            self.scheduler.add_job(
                run_thesis_rescore,
                CronTrigger(hour=9, minute=10, timezone=seoul_tz),
                id='thesis_rescore',
                misfire_grace_time=DAILY_MISFIRE_GRACE_SECONDS,
                replace_existing=True
            )

            # 09:40 KST: grade calls old enough to judge. Must run after the
            # re-score so the snapshot it reads as "latest" is today's rather
            # than yesterday's.
            async def run_thesis_reflection():
                try:
                    await run_thesis_reflection_job(self.db)
                except Exception as e:
                    log.error("orchestrator.thesis_reflection_failed", error=str(e))

            self.scheduler.add_job(
                run_thesis_reflection,
                CronTrigger(hour=9, minute=40, timezone=seoul_tz),
                id='thesis_reflection',
                misfire_grace_time=DAILY_MISFIRE_GRACE_SECONDS,
                replace_existing=True
            )

        # Startup catch-up: run missed daily jobs once, well after boot. This
        # can trigger multi-agent LLM debates and a Yahoo call per unresolved
        # prediction, so it must not land while the app is still starting up.
        self.scheduler.add_job(
            self._startup_catchup,
            'date',
            run_date=datetime.datetime.now() + datetime.timedelta(
                seconds=settings.startup_catchup_delay_seconds
            ),
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

        # Ahead of the watchlist check below: theme detection reads the article
        # corpus, not tracked tickers, so an empty watchlist must not skip it.
        # Wrapped separately so a thesis failure cannot cost the prediction and
        # resolution catch-ups that follow it.
        try:
            await self._catchup_thesis(seoul_tz)
        except Exception as e:
            log.error("orchestrator.startup_catchup.thesis_error", error=str(e))

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

    async def _catchup_thesis(self, seoul_tz) -> None:
        """Run the morning thesis if the worker was down when it was due.

        misfire_grace_time covers a process that was alive but late. It cannot
        cover one that was not running at 06:30 at all: on start-up APScheduler
        computes the next fire time from now, so a restart at 09:00 — a deploy,
        a reboot, Termux being killed — skips straight to tomorrow. That is the
        common case here, and it is why the morning thesis went missing on days
        the phone had been restarted.
        """
        if not settings.thesis_enabled:
            return

        now_seoul = datetime.datetime.now(seoul_tz)
        due_today = now_seoul.replace(
            hour=THESIS_GENERATION_HOUR, minute=THESIS_GENERATION_MINUTE,
            second=0, microsecond=0,
        )
        if now_seoul < due_today:
            # Booted before it was due; the cron job will handle it normally.
            log.info("orchestrator.startup_catchup.thesis_not_due_yet")
            return

        midnight_utc = (
            now_seoul.replace(hour=0, minute=0, second=0, microsecond=0)
            .astimezone(datetime.timezone.utc)
            # SQLite CURRENT_TIMESTAMP format, not isoformat() — see
            # Database.count_theses_since.
            .strftime("%Y-%m-%d %H:%M:%S")
        )
        try:
            generated_today = await asyncio.to_thread(
                self.db.count_theses_since, midnight_utc
            )
        except Exception as e:
            log.error("orchestrator.startup_catchup.thesis_query_failed", error=str(e))
            return

        if generated_today:
            log.info("orchestrator.startup_catchup.thesis_ok", count=generated_today)
            return

        log.info("orchestrator.startup_catchup.missed_thesis")
        try:
            ids = await self.thesis_engine.generate()
            log.info("orchestrator.startup_catchup.thesis_generated", count=len(ids))
        except Exception as e:
            log.error("orchestrator.startup_catchup.thesis_failed", error=str(e))

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
            training_results = []  # (label, cv_metrics) tuples
            failed_tickers = []

            # Retrain every horizon the UI actually reads. This used to call
            # train_model with its default horizon_days=1, producing *_1d models
            # that nothing loads while the 5/21/63/252d models the dashboard
            # needs were never refreshed here at all.

            for t in tracked:
                for horizon_days in HORIZON_LABELS:
                    label = f"{t} ({horizon_days}d)"
                    try:
                        _path, cv_metrics = await predictor.train_model(
                            t, scope="per_ticker", horizon_days=horizon_days)
                        training_results.append((label, cv_metrics))
                    except Exception as e:
                        log.error("orchestrator.retrain_model_failed",
                                  ticker=t, horizon_days=horizon_days, error=str(e))
                        failed_tickers.append((label, str(e)))
                    
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
    from pipeline.predictor import StockPredictor
    log.info("Starting reflection job...")
    
    with db.connection() as conn:
        # Get predictions that are resolved, are multi-agent, and not yet in reflection_log.
        # Bounded by recency and batch size: skipped rows never get a reflection_log row, so
        # without a cutoff they would be re-selected every night forever.
        predictions_to_reflect = conn.execute('''
            SELECT p.id, p.ticker, p.predicted_direction, p.confidence, p.llm_narrative, p.actual_direction, p.actual_change_pct, p.is_correct
            FROM predictions p
            LEFT JOIN reflection_log r ON p.id = r.prediction_id
            WHERE p.is_correct IS NOT NULL 
              AND p.model_type = 'multi_agent'
              AND r.id IS NULL
              AND COALESCE(p.resolved_at, p.resolve_after) >= date('now', 'localtime', ?)
            ORDER BY COALESCE(p.resolved_at, p.resolve_after) DESC
            LIMIT ?
        ''', (f'-{REFLECTION_LOOKBACK_DAYS} days', REFLECTION_BATCH_LIMIT)).fetchall()
        
    if not predictions_to_reflect:
        return

    # Only reflect on tickers the user actually follows. Ad-hoc debates from the Debate
    # Arena and /predict accept any symbol and write multi_agent rows, so without this
    # filter the job spends tokens on tickers nobody is tracking. Compare upper-cased:
    # the watchlist is normalized on write, predictions.ticker is not.
    tracked = {t.upper() for t in db.get_tracked_tickers()}
    skipped = [p for p in predictions_to_reflect if p["ticker"].upper() not in tracked]
    predictions_to_reflect = [p for p in predictions_to_reflect if p["ticker"].upper() in tracked]
    if skipped:
        log.info("reflection.skipped_untracked", count=len(skipped),
                 tickers=sorted({p["ticker"] for p in skipped}))
    if not predictions_to_reflect:
        return

    if not is_llm_configured() or not settings.model_reflection:
        log.warning("MODEL_REFLECTION is not configured; skipping Reflection Job.")
        return
        
    today_str = datetime.datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d")

    # Wrong predictions worth a fresh debate, ranked and capped after the loop.
    pending_corrections: list[dict] = []

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
            # Point MODEL_REFLECTION at a cheap model — this is a
            # summarization task, not reasoning-heavy analysis.
            model_name = settings.model_reflection
            with track_llm(db, model_name, "reflection") as u:
                u.response = response = await complete(
                    model=model_name,
                    system="You analyze past stock predictions and extract structured, actionable lessons. Output JSON only.",
                    prompt=prompt,
                    temperature=0.0,
                    json_mode=True,
                )

            raw = response.text

            try:
                parsed = parse_structured(raw, ReflectionLesson)
                lesson = parsed.lesson_learned
                failure_mode = parsed.failure_mode
                actionable_fix = parsed.actionable_fix
                should_adjust = parsed.should_adjust_strategy
            except Exception as parse_e:
                # Skip rather than store `raw`: these lessons are replayed into
                # future debate prompts, so a malformed blob written here would
                # keep poisoning every later advisory for this ticker.
                log.warning("reflection.parse_failed", ticker=ticker,
                            pred_id=pred_id, error=str(parse_e))
                continue

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
                # Same 5-day gate run_daily_predictions uses: the 08:30 job already
                # refreshed every tracked ticker, so re-debating here would pay for an
                # advisory we generated hours ago.
                if db.get_cached_advisory(ticker, days=5):
                    log.info("reflection.correction_skipped_cached", ticker=ticker)
                    continue

                # Collected rather than re-debated inline. Each correction is a
                # full multi-agent run, and the batch admits up to
                # REFLECTION_BATCH_LIMIT predictions — re-debating every miss
                # here made one nightly job worth ~150 calls on the priciest
                # models. The ranking and cap happen after the loop.
                pending_corrections.append({
                    "ticker": ticker,
                    "confidence": confidence if isinstance(confidence, (int, float)) else 0.0,
                    "lesson": lesson,
                })
        except Exception as e:
            log.error(f"Failed to generate reflection for {ticker}: {e}")

    # ── Corrections ──────────────────────────────────────────────────────
    # Deduped by ticker because two horizons on the same name produce two
    # misses but only one useful re-debate, then ranked by confidence: the
    # predictions we were most sure about are the ones worth re-running.
    by_ticker: dict[str, dict] = {}
    for item in pending_corrections:
        existing = by_ticker.get(item["ticker"])
        if existing is None or item["confidence"] > existing["confidence"]:
            by_ticker[item["ticker"]] = item

    ranked = sorted(by_ticker.values(), key=lambda c: c["confidence"], reverse=True)
    selected = ranked[:REFLECTION_REPREDICT_LIMIT]

    if len(ranked) > len(selected):
        log.info("reflection.corrections_capped",
                 eligible=len(ranked), running=len(selected),
                 skipped=[c["ticker"] for c in ranked[len(selected):]])

    for correction in selected:
        ticker = correction["ticker"]
        try:
            log.info(f"Prediction for {ticker} was incorrect. Triggering new prediction...")
            predictor = StockPredictor(db)
            new_advisory = await predictor.predict_with_agents(ticker)

            if alert_manager:
                verdict = new_advisory.get('final_advisory', 'No plan found')
                message = (
                    f"⚠️ *Correction for {ticker}*\n\n"
                    f"*Why we were wrong:*\n{correction['lesson']}\n\n"
                    f"*Updated Plan:*\n{verdict}"
                )
                await alert_manager.bot.send_message(
                    chat_id=alert_manager.chat_id,
                    text=message,
                    parse_mode="Markdown"
                )
        except Exception as e:
            log.error(f"Failed to generate correction for {ticker}: {e}")


async def run_thesis_reflection_job(db):
    """Grade matured EARLY thesis calls and write what they taught to reflection_log.

    Deterministic and LLM-free by design: the prices already say whether the
    call worked and the node's falsifier already says what would have broken
    it, so a model call here would restate what the row contains and bill for
    it. Lessons land at scope='ticker' so get_relevant_reflections feeds them
    straight back into that ticker's next Bull/Bear debate -- the loop that
    makes the engine sharpen instead of just accumulating calls.
    """
    log.info("Starting thesis reflection job...")
    try:
        rows = await asyncio.to_thread(
            db.get_thesis_calls_due_review,
            settings.thesis_review_min_age_days,
            settings.thesis_review_benchmark,
            settings.thesis_review_batch,
        )
    except Exception as e:
        log.error("thesis_reflection.query_failed", error=str(e))
        return

    if not rows:
        log.info("thesis_reflection.nothing_due")
        return

    today_str = datetime.datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d")
    written = 0

    for r in rows:
        ret = (r["latest_price"] - r["flagged_price"]) / r["flagged_price"]
        bench_start, bench_end = r.get("bench_start"), r.get("bench_end")

        if bench_start and bench_end and bench_start > 0:
            bench_ret = (bench_end - bench_start) / bench_start
            excess = ret - bench_ret
            was_successful = excess > 0
            verdict = (f"{ret:+.1%} against {settings.thesis_review_benchmark}'s "
                       f"{bench_ret:+.1%} ({excess:+.1%} excess)")
        else:
            # No benchmark bars cover this window. Grade on the absolute move
            # but say so -- silently calling a rising tide "outperformance" is
            # the one failure mode that would make these lessons misleading.
            excess = None
            was_successful = ret > 0
            verdict = f"{ret:+.1%} absolute (no benchmark coverage for this window)"

        lesson = (
            f"Thesis '{r['thesis_title']}' flagged {r['ticker']} EARLY on "
            f"{r['flagged_date']} at {r['flagged_price']:.2f} as "
            f"{r['role_in_chain'] or 'a chain participant'}. By {r['latest_date']} "
            f"it returned {verdict}; stage is now {r['latest_stage'] or 'UNKNOWN'}. "
            f"Chain claim: {r['claim'] or 'n/a'} "
            f"The link was said to fail if: {r['falsifier'] or 'n/a'}"
        )

        tags_json = json.dumps({
            "thesis_review": r["candidate_id"],
            "thesis_id": r["thesis_id"],
            "return": round(ret, 4),
            "excess_return": round(excess, 4) if excess is not None else None,
            "benchmark": settings.thesis_review_benchmark if excess is not None else None,
            "flagged_edge": r["flagged_edge"],
        })

        try:
            await asyncio.to_thread(
                db.insert_reflection, r["ticker"], None, today_str,
                lesson, was_successful, "ticker", None, tags_json,
            )
            written += 1
        except Exception as e:
            log.warning("thesis_reflection.insert_failed",
                        candidate=r["candidate_id"], error=str(e))

    log.info("thesis_reflection.completed", reviewed=len(rows), written=written)
