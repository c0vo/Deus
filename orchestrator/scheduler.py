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
# Imported as a module, not as `from config.usage import tally`: the cycle reads
# the counter at two different points in time, and binding the object into this
# namespace would hide a replacement (a test's, or a future reset) from both.
from config import usage as llm_usage
from data.database import Database
from pipeline.aggregator import NewsAggregator
from pipeline.classifier import ArticleClassifier, ClassifierNotConfigured
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
from pipeline.macro_calendar import MacroCalendar
from pipeline.seasonality import ensure_deep_history
from pipeline.weekly_tip import WeeklyTipComposer
from bot.alerts import AlertManager
from bot.formatters import render_weekly_tip
from api.sse_manager import event_bus

log = get_logger(__name__)

# Reflection job bounds. The job scans every resolved multi-agent prediction that
# has no lesson yet, so without these the whole historical backlog is processed in
# a single tick.
REFLECTION_LOOKBACK_DAYS = 14
REFLECTION_BATCH_LIMIT = 25

# How many tickers the startup catch-up will pull full history for. Each is the
# heaviest request we make of Yahoo, and the monthly seasonality job covers the
# rest — so a cold boot gets the benchmarks deep and then gets out of the way.
STARTUP_DEEP_HISTORY_TICKERS = 3

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
# wakes minutes late and was silently skipped every morning.
#
# Every job now inherits settings.job_misfire_grace_seconds as the scheduler-wide
# default (see the AsyncIOScheduler construction below), which is what covers the
# interval jobs. This wider window is the override applied to every daily and
# weekly job: a brief, a scan or a thesis is "sometime this morning" work, so
# hours of slack cost nothing, whereas a 15-minute interval job would rather be
# skipped than run four times back to back. coalesce (on by default, and now
# explicit) still collapses a backlog into a single run.
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

    # How long to leave a (ticker, horizon) pair alone after training failed for
    # want of price history. A day: the only thing that can change the answer is
    # more sessions being ingested, and that happens once per trading day.
    TRAINING_RETRY_BACKOFF_SECONDS = 24 * 3600

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
        self.macro_calendar = MacroCalendar(db=self.db)

        self.thesis_engine = ThesisEngine(db=self.db)

        # Every cron trigger below passes timezone= explicitly, but the
        # scheduler's own default was the host's local zone — and that is what
        # decided when the naive run_date jobs in start() actually fired.
        # job_defaults is what lifts APScheduler's one-second misfire window off
        # every job at once; the daily and weekly ones override it wider.
        self.tz = ZoneInfo(settings.timezone)
        self.scheduler = AsyncIOScheduler(
            job_defaults={
                "misfire_grace_time": settings.job_misfire_grace_seconds,
                "coalesce": True,
                "max_instances": 1,
            },
            timezone=self.tz,
        )
        self.is_running = False
        # Classification now has two triggers — the five-minute job and the tail
        # of every pipeline cycle — so it needs its own mutual exclusion.
        # APScheduler's max_instances=1 only stops a job overlapping *itself*.
        self._classify_lock = asyncio.Lock()
        # (ticker, horizon_days) → unix time before which not to retry training.
        # In memory on purpose: the only thing that changes the answer is more
        # price history arriving, and a worker restart retrying once is cheaper
        # than another user_config write path.
        self._training_backoff: dict[tuple[str, int], float] = {}
        # Seeded here, not only in run_pipeline_cycle. The classify job and the
        # embed pass both write into it, and the classify job can now fire from
        # its own interval before any cycle has run.
        self._cycle_counts = self._new_cycle_counts()

    @staticmethod
    def _new_cycle_counts() -> dict:
        return {"fetched": 0, "inserted": 0, "classified": 0,
                "ranked": 0, "embedded": 0, "alerts": 0, "errors": 0,
                "llm_calls": 0, "llm_cost": 0.0}

    async def run_pipeline_cycle(self) -> None:
        """Executes a single pass of the entire pipeline."""
        if self.is_running:
            log.warning("orchestrator.skipped", reason="Previous cycle still running")
            return

        self.is_running = True
        log.info("orchestrator.cycle_start")

        cycle_start = time.time()
        self._cycle_counts = self._new_cycle_counts()
        # llm_calls / llm_cost are a difference of two readings of a
        # process-wide counter, not a total accumulated here. Every call site
        # already goes through track_llm; asking each of them to also report
        # back up to the cycle is how these two fields stayed at zero for the
        # life of the pipeline_metrics table.
        usage_before = llm_usage.tally.snapshot()

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

            # Classify what this cycle just ingested rather than waiting out the
            # interval job: fresh news is the only news worth alerting on, and
            # the embeds it needs were written by the pass above.
            status = await self.run_classify_backlog(trigger="cycle")
            self._cycle_counts["classified"] += int(status.get("classified") or 0)

        except Exception as e:
            log.error("orchestrator.cycle_failed", error=str(e))
            self._cycle_counts["errors"] += 1
        finally:
            duration = time.time() - cycle_start
            self.is_running = False
            # Before the metrics row is written, not after — this is the only
            # place the two counters are ever filled in.
            usage_after = llm_usage.tally.snapshot()
            self._cycle_counts["llm_calls"] = usage_after.calls - usage_before.calls
            self._cycle_counts["llm_cost"] = round(
                usage_after.cost - usage_before.cost, 6
            )
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
        """
        Embed a batch of freshly-ingested articles, prefiltering obvious noise.

        Embedding only. Classification used to run here too, which tied its
        throughput to the fetch loop: ≤20 rows per pass, serial, no attempt
        counter, LIFO selection — capacity roughly equal to ingest with no
        margin, so a single persistently failing batch parked itself at the head
        of the queue and the backlog grew monotonically. It is now its own job;
        see `run_classify_backlog`.
        """
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

        except Exception as e:
            log.error("orchestrator.batch_failed", error=str(e))
            self._cycle_counts["errors"] += 1

    # Bumping the suffix re-runs the requeue on the next deploy. Only do that
    # alongside a fix that changes which rows end up 'error' — otherwise it just
    # re-sends the same unclassifiable articles to the same model.
    ERROR_REQUEUE_MARKER = "classify_error_requeue_v1"

    async def _requeue_error_articles_once(self) -> int:
        """
        One-time amnesty for rows the old failure semantics wrote off.

        `_persist_classifications` used to stamp `'error'` on the first failure,
        so a truncated batch, a parse failure and a genuinely unclassifiable
        article were all equally permanent — 5,662 rows on the phone, growing at
        100-400 a day. Those rows are real articles, and now that a failure
        counts attempts instead, they deserve the retries they never got.

        Guarded by a marker in `user_config` rather than by a schema version: it
        must run exactly once per deployment, and it has to be idempotent if the
        worker restarts mid-run.
        """
        marker = await asyncio.to_thread(
            self.db.get_config, self.ERROR_REQUEUE_MARKER, ""
        )
        if marker:
            return 0

        requeued = await asyncio.to_thread(
            self.db.requeue_error_articles, settings.classify_max_age_days
        )
        await asyncio.to_thread(
            self.db.set_config,
            self.ERROR_REQUEUE_MARKER,
            json.dumps({
                "requeued": requeued,
                "max_age_days": settings.classify_max_age_days,
                "at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }),
        )
        log.info(
            "orchestrator.classify_error_requeued",
            count=requeued, max_age_days=settings.classify_max_age_days,
        )
        return requeued

    # Same rule as above: bump the suffix only alongside a fix that changes why
    # rows exhaust their attempts. On its own it just re-parks them.
    ATTEMPTS_RESET_MARKER = "classify_attempts_reset_v1"

    # How far back to un-write-off 'error' rows in this repair. Short on purpose:
    # `classify_error_requeue_v1` already gave the whole 30-day window its
    # amnesty, so anything older than the bad deploy was judged under request
    # semantics this fix does not change, and re-sending it would be paying twice
    # for the same verdict.
    ATTEMPTS_RESET_ERROR_AGE_DAYS = 2

    async def _reset_parked_attempts_once(self) -> dict[str, int]:
        """
        One-time amnesty for rows the batch-schema regression parked.

        The request schema marked every field but `id` optional and put no
        minimum on the items array, so a batch of ten was answered with one
        near-empty object. Nine rows per batch logged `batch_missing_result`,
        and three cycles later — fifteen minutes — they had spent all three
        `classification_attempts` on requests that never asked about them. ~35
        of 60 candidates per run, every run, for as long as the deploy was up.

        Two repairs, because the regression produced two kinds of casualty:
        rows parked at max attempts, and rows stamped `'error'` when a whole
        batch came back unusable. The `'error'` sweep is deliberately narrow —
        `ATTEMPTS_RESET_ERROR_AGE_DAYS`, not the full classification window —
        so it catches what this bug wrote off and not what the earlier amnesty
        already reconsidered.

        Marker-guarded in `user_config` like `_requeue_error_articles_once`: it
        must run exactly once per deployment, and be idempotent if the worker
        dies mid-run.
        """
        marker = await asyncio.to_thread(
            self.db.get_config, self.ATTEMPTS_RESET_MARKER, ""
        )
        if marker:
            return {"unparked": 0, "error_requeued": 0}

        unparked = await asyncio.to_thread(
            self.db.reset_parked_classification_attempts,
            settings.classify_max_attempts,
        )
        error_requeued = await asyncio.to_thread(
            self.db.requeue_error_articles, self.ATTEMPTS_RESET_ERROR_AGE_DAYS
        )
        await asyncio.to_thread(
            self.db.set_config,
            self.ATTEMPTS_RESET_MARKER,
            json.dumps({
                "unparked": unparked,
                "error_requeued": error_requeued,
                "max_attempts": settings.classify_max_attempts,
                "error_max_age_days": self.ATTEMPTS_RESET_ERROR_AGE_DAYS,
                "at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }),
        )
        log.info(
            "orchestrator.classify_attempts_reset",
            unparked=unparked, error_requeued=error_requeued,
            max_attempts=settings.classify_max_attempts,
            error_max_age_days=self.ATTEMPTS_RESET_ERROR_AGE_DAYS,
        )
        return {"unparked": unparked, "error_requeued": error_requeued}

    async def run_classify_backlog(self, trigger: str = "interval") -> dict:
        """
        Classify the unclassified backlog, bounded on every axis.

        The job that replaces "classification as a step in the fetch loop". Four
        bounds, each answering one way the old arrangement went wrong:

        - **Budget** (`classify_per_run_limit`) and **concurrency**
          (`classify_concurrency`) make throughput independent of how long a
          fetch takes, so the backlog can be drained faster than it fills.
        - **Attempts** (`classify_max_attempts`) retire a poison batch instead
          of letting it re-occupy the head of a LIFO queue forever.
        - **Age** (`classify_max_age_days`) caps what is worth paying for at
          all; everything older is marked `'stale'` first.

        Returns the status dict it also persists, so the caller can fold
        `classified` into its own counters instead of reaching into private
        state.
        """
        if self._classify_lock.locked():
            log.info(
                "orchestrator.classify_backlog_skipped",
                reason="previous run still in progress", trigger=trigger,
            )
            return {"skipped": True, "trigger": trigger}

        async with self._classify_lock:
            started = time.time()
            status = {
                "last_run_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "trigger": trigger,
                "candidates": 0,
                "classified": 0,
                "failed": 0,
                "stale_marked": 0,
                "error_requeued": 0,
                "attempts_unparked": 0,
                "duration_s": 0.0,
                "configured": True,
            }

            try:
                # Cheapest check first. Without it an unconfigured deployment
                # burns a stale-marking UPDATE and a candidate scan every five
                # minutes to reach a call it cannot make.
                if not self.classifier.is_configured():
                    status["configured"] = False
                    log.error(
                        "orchestrator.classify_backlog_unconfigured",
                        impact="ingest is embedding-only; nothing will be classified",
                        hint="set MODEL_CLASSIFIER (or MODEL_CLASSIFIER_FALLBACK) "
                             "in .env and restart the worker",
                    )
                    return status

                # Give the rows the old write-off semantics killed one more
                # chance. Once, ever, guarded by a marker.
                status["error_requeued"] = await self._requeue_error_articles_once()

                # And the rows the batch-schema regression parked, whose attempts
                # were spent on requests that never asked about them. Also once,
                # also marker-guarded, and counted separately because the two
                # repairs answer different bugs.
                reset = await self._reset_parked_attempts_once()
                status["attempts_unparked"] = reset["unparked"]
                status["error_requeued"] += reset["error_requeued"]

                # Retire what is out of window before selecting candidates, so a
                # long tail of 2024 rows cannot keep the queue permanently
                # non-empty. Bounded per run; it converges over a few ticks.
                status["stale_marked"] = await asyncio.to_thread(
                    self.db.mark_stale_unclassified, settings.classify_max_age_days
                )

                rows = await asyncio.to_thread(
                    self.db.get_classification_candidates,
                    settings.classify_per_run_limit,
                    settings.classify_max_attempts,
                    settings.classify_max_age_days,
                )
                status["candidates"] = len(rows)
                if rows:
                    await self._classify_candidates(rows, status, trigger)
                else:
                    log.info(
                        "orchestrator.classify_backlog_empty",
                        trigger=trigger, stale_marked=status["stale_marked"],
                    )

                # Unconditionally, not only when this run classified something.
                # An article whose ranking call failed earlier is already
                # classified, so gating this on *new* candidates would leave it
                # unranked forever — and unranked means it reaches neither the
                # brief nor alerts. Costs one SELECT when there is nothing to do.
                await self._rank_pending()

            except ClassifierNotConfigured as e:
                # Raised from _call_with_fallback when every slug for the lane is
                # empty. Reported, never converted into an 'error' verdict.
                status["configured"] = False
                log.error("orchestrator.classify_backlog_unconfigured", error=str(e))
            except Exception as e:
                log.error("orchestrator.classify_backlog_failed", error=str(e))
            finally:
                status["duration_s"] = round(time.time() - started, 2)
                # Persisted as well as published: the SSE outbox is trimmed to
                # ten minutes, so a dashboard opened between runs has nothing to
                # read unless the last run is in user_config. It is also the one
                # artefact a `sqlite3` session on the phone can check.
                try:
                    await asyncio.to_thread(
                        self.db.set_config,
                        "classify_backlog_status", json.dumps(status),
                    )
                except Exception as e:
                    log.error("orchestrator.classify_status_write_failed", error=str(e))
                try:
                    await event_bus.publish("classification_status", status)
                except Exception as e:
                    log.error("orchestrator.classify_status_publish_failed", error=str(e))

            return status

    async def _classify_candidates(
        self, rows: list[dict], status: dict, trigger: str
    ) -> None:
        """
        Dedup, chunk and classify one run's candidates, updating `status`.

        Split out of `run_classify_backlog` so that the status write, the publish
        and the ranking pass in its `finally` apply whether or not there was
        anything to classify.
        """
        articles = [self.db.row_to_article(r) for r in rows]

        # Dedup first, as its own pass: it makes no LLM calls — it reads the
        # article's own embedding against older persisted rows — so an absorbed
        # duplicate inherits its canonical verdict and never reaches a model.
        # Each check is blocking SQLite plus numpy, and there are up to
        # `classify_per_run_limit` of them, which is far too long to hold the
        # event loop for on a phone.
        verdicts = await asyncio.gather(
            *(asyncio.to_thread(self._absorb_duplicate, a) for a in articles),
            return_exceptions=True,
        )
        needs_llm = []
        for article, verdict in zip(articles, verdicts):
            if isinstance(verdict, BaseException):
                # A dedup failure must not cost the classification. The row goes
                # to the model as if it had no near neighbour.
                log.warning(
                    "orchestrator.classify_backlog_dedup_failed",
                    article_id=article.id, error=str(verdict),
                )
                needs_llm.append(article)
            elif not verdict:
                needs_llm.append(article)

        size = max(1, settings.classify_batch_size)
        chunks = [needs_llm[i:i + size] for i in range(0, len(needs_llm), size)]
        # Bounds how many batches are in flight at once. The provider's rate
        # limit, not the phone, is what this protects.
        semaphore = asyncio.Semaphore(max(1, settings.classify_concurrency))

        async def process_chunk(chunk: list) -> tuple[int, int]:
            async with semaphore:
                persisted = await self._classify_chunk_with_retry(chunk)
            if persisted:
                return self._persist_classifications(chunk), 0
            # Transient exhaustion: every attempt hit an infrastructure fault.
            # Count the attempt and leave event_type NULL — a provider outage
            # must never be written down as a permanent 'error' verdict on the
            # article.
            await asyncio.to_thread(
                self._record_chunk_failure,
                [a.id for a in chunk],
                stamp_error_at_cap=False,
            )
            return 0, len(chunk)

        # return_exceptions so that every chunk is awaited. A bare gather
        # propagates the first failure and leaves the remaining tasks running
        # detached — they would then persist a chunk after the caller had already
        # written its status, or drop an unretrieved exception at collection time.
        results = await asyncio.gather(
            *(process_chunk(chunk) for chunk in chunks),
            return_exceptions=True,
        )
        unconfigured: Optional[ClassifierNotConfigured] = None
        for result in results:
            if isinstance(result, ClassifierNotConfigured):
                unconfigured = result
            elif isinstance(result, BaseException):
                # One chunk's write failing is not the whole run failing.
                log.error("orchestrator.classify_chunk_failed", error=str(result))
            else:
                classified, failed = result
                status["classified"] += classified
                status["failed"] += failed

        log.info(
            "orchestrator.classify_backlog",
            trigger=trigger,
            candidates=status["candidates"],
            classified=status["classified"],
            failed=status["failed"],
            stale_marked=status["stale_marked"],
            deduped=len(articles) - len(needs_llm),
        )

        if unconfigured is not None:
            # Raised for the caller's handler, which sets configured=False. The
            # one failure for which stamping anything at all would be wrong.
            raise unconfigured

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
            except ClassifierNotConfigured:
                # Re-raised, never retried and never counted. The generic handler
                # below would classify it as non-transient and so return True —
                # "safe to persist" — which is exactly how an unset
                # MODEL_CLASSIFIER came to stamp every row in the batch 'error'.
                raise
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

    def _record_chunk_failure(
        self, article_ids: list[str], *, stamp_error_at_cap: bool
    ) -> int:
        """
        Count one failed classification attempt per id. Returns how many rows
        have now used up their attempts.

        The row stays `event_type IS NULL` until the cap, so a later pass can
        retry it. That is the opposite of what this code used to do: any article
        the model did not answer for was stamped `'error'` on the spot "so it
        doesn't infinite loop", which on the phone turned one bad response into
        5,662 permanently dead rows — 9% of the corpus — with nothing
        distinguishing a truncated batch from an unclassifiable article.

        `stamp_error_at_cap` is False for a provider outage: the attempt counter
        already keeps those rows out of the candidate query (they surface as
        `exhausted`), and an outage is not a property of the article.
        """
        if not article_ids:
            return 0

        attempts = self.db.record_classification_failure(article_ids) or {}
        exhausted = [
            article_id for article_id in article_ids
            if attempts.get(article_id, 0) >= settings.classify_max_attempts
        ]
        if exhausted and stamp_error_at_cap:
            for article_id in exhausted:
                self.db.update_classification(
                    article_id=article_id,
                    event_type="error",
                    sentiment_score=0.0,
                    urgency="low",
                    suggested_direction="neutral",
                    affected_sectors=[],
                    affected_tickers=[],
                    classification_summary=(
                        f"Classification failed on "
                        f"{settings.classify_max_attempts} separate attempts."
                    ),
                )
        return len(exhausted)

    def _persist_classifications(self, chunk: list) -> int:
        """
        Writes back a classified chunk. Returns how many succeeded.

        Articles the model never answered for come back with event_type still
        unset. Those get an attempt counted and are left NULL for a later pass;
        only an article that has failed `classify_max_attempts` times is written
        off as `'error'`.
        """
        succeeded = 0
        failed_ids: list[str] = []
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
                failed_ids.append(article.id)

        if failed_ids:
            exhausted = self._record_chunk_failure(failed_ids, stamp_error_at_cap=True)
            log.warning(
                "orchestrator.classification_unanswered",
                count=len(failed_ids), written_off=exhausted,
            )

        # One event per chunk, not per article. A 10-article chunk used to
        # schedule ten publishes, each a separate INSERT into the sse_events
        # outbox and a separate SSE frame, for a feed the page prepends to in one
        # go anyway — the hook already reads `data.articles` as a list.
        if chunk:
            asyncio.create_task(
                event_bus.publish(
                    "new_articles",
                    {"articles": [article.model_dump() for article in chunk]},
                )
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
            from bot.formatters import EMPTY_BRIEFING_TEXT, render_briefing
            from data.taxonomy import BRIEFING_MIN_IMPORTANCE, select_briefing_lanes

            rows = self.db.get_briefing_candidates(
                hours=24, min_importance=BRIEFING_MIN_IMPORTANCE, limit=40
            )
            lanes = select_briefing_lanes(rows)
            # Always send, even when nothing clears the floor. A quiet news day
            # and a job that silently died look identical from the chat.
            text = render_briefing(lanes) if lanes else EMPTY_BRIEFING_TEXT

            await self.alert_manager.send_html(text)

            # Persisted as well as sent, so the brief survives the chat scrollback
            # and /api/digests can answer "what did it say on Tuesday?". Failing
            # to store it must not cost the send that already happened.
            try:
                today = datetime.datetime.now(self.tz).date().isoformat()
                await asyncio.to_thread(
                    self.db.insert_digest, "daily_briefing", text,
                    period_start=today, period_end=today,
                )
            except Exception as e:
                log.warning("orchestrator.daily_briefing_persist_failed", error=str(e))

            log.info("orchestrator.daily_briefing_sent", articles=sum(len(a) for _, a in lanes))
        except Exception as e:
            log.error("orchestrator.daily_briefing_failed", error=str(e))

    async def send_daily_advisor(self):
        """Sends the evidence-based morning stance for every tracked ticker.

        The note this replaced read the last 24h of classified headlines and
        offered the model a choice of only HOLD or SELL, instructing it to fall
        back to the former. A ticker with no news printed a hardcoded HOLD, and
        so did an unset MODEL_DAILY_ADVISOR, so every ETF and every
        misconfiguration rendered as a considered decision to hold.
        `pipeline/daily_stance.py` builds a full fact sheet per ticker instead,
        offers all four actions, and names a failure as NO CALL or UNAVAILABLE
        rather than HOLD.

        The Bull/Bear debate re-runs happen strictly AFTER the message is sent:
        four reasoning calls per ticker would otherwise delay the note by
        minutes, and the note is the product. Each debate writes today's
        `predictions_cache` row, which is the same cache the 08:30
        `run_daily_predictions` job checks — so a re-run here makes that job
        skip the ticker rather than paying for a second debate.
        """
        if not self.alert_manager:
            return

        from bot.formatters import render_stance_message
        from pipeline.daily_stance import DailyStanceEngine

        engine = DailyStanceEngine(self.db)
        try:
            tracked = await asyncio.to_thread(self.db.get_tracked_tickers)
            if not tracked:
                return

            batch = await engine.compose(tracked)
            await asyncio.to_thread(engine.persist, batch)

            text = render_stance_message(batch.rows, model_slug=batch.model,
                                         date=batch.date)
            # Stored before sending so the dashboard and a resend have the note
            # even if Telegram is the thing that is down.
            await asyncio.to_thread(lambda: self.db.insert_digest(
                "daily_advisor", text,
                facts_json=batch.facts,
                model=batch.model or None,
                period_start=batch.date,
                period_end=batch.date,
            ))

            await self.alert_manager.send_html(text)
            log.info("orchestrator.daily_advisor_sent", tickers=len(batch.rows),
                     model=batch.model or "unset")
        except Exception as e:
            log.error("orchestrator.daily_advisor_failed", error=str(e) or repr(e))
            return

        try:
            rerun = engine.material_changes(batch.rows, batch.facts)
        except Exception as e:
            log.error("orchestrator.advisor_rerun_select_failed",
                      error=str(e) or repr(e))
            return

        # Sequential, and each one isolated: the debates share the LLM budget
        # with everything else the worker does, and one ticker failing must not
        # cost the others their re-run.
        predictor = StockPredictor(self.db)
        for ticker in rerun:
            try:
                await predictor.predict_with_agents(ticker)
                log.info("orchestrator.advisor_debate_rerun", ticker=ticker)
            except Exception as e:
                log.error("orchestrator.advisor_debate_rerun_failed",
                          ticker=ticker, error=str(e) or repr(e))

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
                        
            await self.alert_manager.send_html(text)
            log.info("orchestrator.weekly_review_sent")
        except Exception as e:
            log.error("orchestrator.weekly_review_failed", error=str(e))

    async def check_api_usage_spikes(self):
        """Checks for API usage spikes and sends an alert if needed."""
        try:
            alert_msg = self.db.check_for_api_spikes()
            if alert_msg and self.alert_manager:
                await self.alert_manager.send_html(alert_msg)
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

        The whole body runs in a worker thread. Nothing in it is awaitable — it
        is a few hundred blocking SQLite round-trips plus a numpy cosine pass per
        tick — so on the loop it stalled every other job, the SSE outbox tail
        included, for its full duration.
        """
        def _run() -> None:
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

        try:
            await asyncio.to_thread(_run)
        except Exception as e:
            log.error("orchestrator.dedup_backfill_failed", error=str(e))

    async def backfill_geo_tags(self) -> None:
        """
        Tag older articles with countries using the offline gazetteer.

        New articles get countries from the classifier; this covers everything
        ingested before that field existed. It makes no API calls, so the whole
        archive can be tagged for free — and for the same reason it is purely
        blocking work, so it runs in a thread rather than on the loop.
        """
        def _run() -> None:
            tagged = self.geo_tagger.backfill(limit=settings.geo_backfill_batch)
            if tagged:
                remaining = self.db.get_geo_backlog_count()
                log.info(
                    "orchestrator.geo_backfill", tagged=tagged, remaining=remaining
                )

        try:
            await asyncio.to_thread(_run)
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

        A pair that cannot be trained *yet* — `train_model` raises on too little
        price history — is parked for `TRAINING_RETRY_BACKOFF_SECONDS` instead of
        being retried on the next tick. Without that, the first such ticker sits
        at the head of this loop and fails every 20 minutes forever (159 times in
        one log on the phone), and because the exception escaped to the outer
        handler it was also the *last* pair attempted each run: every other
        missing model behind it was never reached.
        """
        try:
            tracked = await asyncio.to_thread(self.db.get_tracked_tickers)
            if not tracked:
                return

            predictor = StockPredictor(self.db)
            now = time.time()
            for ticker in tracked:
                active = await asyncio.to_thread(
                    self.db.get_recent_predictions, ticker, 20, True
                )
                covered = {p.get("horizon_days") for p in active}

                for horizon_days in HORIZON_LABELS:
                    if horizon_days in covered:
                        continue

                    until = self._training_backoff.get((ticker, horizon_days), 0.0)
                    if until > now:
                        continue

                    model, _scope = await asyncio.to_thread(
                        predictor._load_model, ticker, horizon_days
                    )
                    if model is None:
                        log.info("orchestrator.training_missing_model",
                                 ticker=ticker, horizon_days=horizon_days)
                        try:
                            await predictor.train_model(
                                ticker, scope="per_ticker", horizon_days=horizon_days
                            )
                        except ValueError as e:
                            # Not a fault: the ticker simply has too short a
                            # price history to model yet. info, not error, and
                            # parked so the loop moves on to the next pair.
                            self._training_backoff[(ticker, horizon_days)] = (
                                now + self.TRAINING_RETRY_BACKOFF_SECONDS
                            )
                            log.info(
                                "orchestrator.training_deferred",
                                ticker=ticker, horizon_days=horizon_days,
                                reason=str(e),
                                retry_after_hours=round(
                                    self.TRAINING_RETRY_BACKOFF_SECONDS / 3600, 1
                                ),
                            )
                            continue
                        model, _scope = await asyncio.to_thread(
                            predictor._load_model, ticker, horizon_days
                        )

                    if model is None:
                        # Training did not produce a loadable model — usually
                        # too little price history. Stop rather than fall
                        # through to predict(), whose llm_only path would spend
                        # a live LLM call on every cycle for a ticker that
                        # cannot be modelled. Backed off for the same reason as
                        # above: otherwise this pair is retried every 20 minutes
                        # and every pair behind it is never reached.
                        self._training_backoff[(ticker, horizon_days)] = (
                            now + self.TRAINING_RETRY_BACKOFF_SECONDS
                        )
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

    async def refresh_macro_calendar(self) -> dict:
        """Monthly top-up of macro_events from the web. Never raises."""
        try:
            # refresh_from_web logs its own counts under macro_calendar.refreshed.
            return await self.macro_calendar.refresh_from_web()
        except Exception as e:
            # MacroCalendar.refresh_from_web already swallows its own failures;
            # this is the belt-and-braces layer so a bug there can never take the
            # cron job — and therefore the scheduler thread — down with it.
            log.error("orchestrator.macro_calendar_refresh_failed", error=str(e))
            return {}

    # ── Weekly tip ───────────────────────────────────────────────────────

    def _seasonality_universe(self) -> list[str]:
        """Benchmarks plus tracked tickers, the set seasonality is measured on."""
        universe: list[str] = [
            t.strip().upper()
            for t in (settings.seasonality_benchmarks or "").split(",")
            if t.strip()
        ]
        try:
            for ticker in self.db.get_tracked_tickers() or []:
                symbol = (ticker or "").strip().upper()
                if symbol and symbol not in universe:
                    universe.append(symbol)
        except Exception as e:
            log.warning("orchestrator.seasonality_universe_failed", error=str(e))
        return universe

    async def send_weekly_tip(self) -> None:
        """
        Compose, store and send the weekly tip. Never raises.

        Sends whatever the composer produced even when the model is unset or the
        call failed — the facts-only render is still the week's calendar and
        seasonal precedents, and a silent Sunday is indistinguishable from a dead
        worker.
        """
        try:
            composer = WeeklyTipComposer(self.db, self.price_feed)
            facts = await composer.gather_facts()
            tips = await composer.compose(facts)
            body_html = render_weekly_tip(facts, tips)
            await composer.persist_and_publish(facts, tips, body_html)

            if self.alert_manager:
                await self.alert_manager.send_html(body_html)
            log.info("orchestrator.weekly_tip_sent",
                     tips=len(tips), status=facts.get("tip_status"),
                     effects=len(facts.get("seasonality") or []))
        except Exception as e:
            log.error("orchestrator.weekly_tip_failed", error=str(e))

    async def refresh_seasonality_history(self, limit: Optional[int] = None) -> dict:
        """
        Deepen `price_history` for tickers with less than `seasonality_min_years`.

        Monthly, and normally a no-op: a ticker that already holds the history is
        skipped without a network call, so the job only costs anything when a new
        name joins the watchlist. `limit` caps how many get pulled in one pass,
        which is what the startup path uses to keep a cold boot short.
        """
        try:
            universe = self._seasonality_universe()
            if limit is not None:
                universe = universe[:limit]
            return await ensure_deep_history(
                self.db, self.price_feed, universe, settings.seasonality_min_years
            )
        except Exception as e:
            log.error("orchestrator.seasonality_refresh_failed", error=str(e))
            return {}

    def start(self, interval_minutes: int = 5) -> None:
        """Starts the periodic scheduler."""
        # First cycle is delayed rather than immediate. A full
        # fetch → classify → embed → rank cycle fired at t=0 competes with the
        # dashboard's very first page load, which on a phone is the difference
        # between a responsive app and one that appears not to load at all.
        self.scheduler.add_job(
            self.run_pipeline_cycle,
            'date',
            # Timezone-aware on purpose: APScheduler localizes a NAIVE run_date
            # to the scheduler's timezone, which is no longer the host's. Off a
            # machine running in settings.timezone, a naive "now + 90s" would be
            # read as 90 seconds from now *in Seoul* — hours away either way.
            run_date=datetime.datetime.now(self.tz) + datetime.timedelta(
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
        # Classification is its own job, decoupled from the fetch loop. First run
        # a minute after the first cycle, so the rows it ingested already have
        # embeddings for dedup; then on its own cadence, which is what lets
        # throughput exceed ingest instead of merely matching it.
        self.scheduler.add_job(
            self.run_classify_backlog,
            'date',
            run_date=datetime.datetime.now(self.tz) + datetime.timedelta(
                seconds=settings.pipeline_startup_delay_seconds + 60
            ),
            id='classify_backlog_warmup',
            replace_existing=True
        )
        self.scheduler.add_job(
            self.run_classify_backlog,
            'interval',
            minutes=settings.classify_backlog_interval_minutes,
            id='classify_backlog',
            # Narrower than the scheduler default: a classification run that was
            # due two minutes ago is still worth doing, one due ten minutes ago
            # has been superseded by the next tick.
            misfire_grace_time=120,
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
            run_date=datetime.datetime.now(self.tz) + datetime.timedelta(seconds=5),
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
        
        # Daily Briefing — 5:00 AM KST by default, configurable via .env
        self.scheduler.add_job(
            self.send_daily_briefing,
            CronTrigger(
                hour=settings.briefing_hour,
                minute=settings.briefing_minute,
                timezone=self.tz
            ),
            id='daily_briefing',
            misfire_grace_time=DAILY_MISFIRE_GRACE_SECONDS,
            replace_existing=True
        )

        # 4:30 AM KST — retire IPOs that have already listed. Placed ahead of the
        # default 5:00 briefing so the brief never cites a stale listing; moving
        # BRIEFING_HOUR earlier than 4:30 breaks that ordering.
        self.scheduler.add_job(
            self.retire_stale_ipos,
            CronTrigger(hour=4, minute=30, timezone=self.tz),
            id='ipo_retire',
            misfire_grace_time=DAILY_MISFIRE_GRACE_SECONDS,
            replace_existing=True
        )
        
        # 5:05 AM KST Daily Advisor
        self.scheduler.add_job(
            self.send_daily_advisor,
            CronTrigger(hour=5, minute=5, timezone=self.tz),
            id='daily_advisor',
            misfire_grace_time=DAILY_MISFIRE_GRACE_SECONDS,
            replace_existing=True
        )
        
        # 8:00 AM KST Daily Earnings Whisper Check
        self.scheduler.add_job(
            self.market_scanner.check_earnings,
            CronTrigger(hour=8, minute=0, timezone=self.tz),
            id='earnings_whisper',
            misfire_grace_time=DAILY_MISFIRE_GRACE_SECONDS,
            replace_existing=True
        )
        
        # 8:30 AM KST Daily Predictions
        self.scheduler.add_job(
            run_daily_predictions,
            CronTrigger(hour=8, minute=30, timezone=self.tz),
            args=[self.db, self.alert_manager],
            id='daily_predictions',
            misfire_grace_time=DAILY_MISFIRE_GRACE_SECONDS,
            replace_existing=True
        )
        
        # 18:00 KST (6:00 PM) Friday Weekly Review
        self.scheduler.add_job(
            self.send_weekly_review,
            CronTrigger(day_of_week='fri', hour=18, minute=0, timezone=self.tz),
            id='weekly_review',
            misfire_grace_time=DAILY_MISFIRE_GRACE_SECONDS,
            replace_existing=True
        )
        
        # 21:00 KST Daily Prediction Resolution
        self.scheduler.add_job(
            self.resolve_predictions,
            CronTrigger(hour=21, minute=0, timezone=self.tz),
            id='prediction_resolution',
            misfire_grace_time=DAILY_MISFIRE_GRACE_SECONDS,
            replace_existing=True
        )
        
        # 21:05 KST Daily Reflection Job
        self.scheduler.add_job(
            run_reflection_job,
            CronTrigger(hour=21, minute=5, timezone=self.tz),
            args=[self.db, self.alert_manager],
            id='reflection_job',
            misfire_grace_time=DAILY_MISFIRE_GRACE_SECONDS,
            replace_existing=True
        )
        
        # Sunday 22:00 KST Weekly Model Retraining
        self.scheduler.add_job(
            self.retrain_models,
            CronTrigger(day_of_week='sun', hour=22, minute=0, timezone=self.tz),
            id='model_retraining',
            misfire_grace_time=DAILY_MISFIRE_GRACE_SECONDS,
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
        # bare 'cron' form this used to use fired at whatever the scheduler's
        # default timezone was — back then the host's local time, correct on the
        # Termux target only by coincidence.
        self.scheduler.add_job(
            run_insider_scan,
            CronTrigger(hour=7, minute=0, timezone=self.tz),
            id='insider_scan',
            misfire_grace_time=DAILY_MISFIRE_GRACE_SECONDS,
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
            CronTrigger(hour=18, minute=0, timezone=self.tz),
            id='kr_flow_scan',
            misfire_grace_time=DAILY_MISFIRE_GRACE_SECONDS,
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
            CronTrigger(hour=7, minute=15, timezone=self.tz),
            id='price_history_sync',
            misfire_grace_time=DAILY_MISFIRE_GRACE_SECONDS,
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
                CronTrigger(hour=hour, minute=minute, timezone=self.tz),
                id=f'darkpool_scan{suffix}',
                misfire_grace_time=DAILY_MISFIRE_GRACE_SECONDS,
                replace_existing=True
            )

        # 07:45 KST daily: market-wide regime series (DIX/GEX, OCC put/call).
        # Both derive from the same US session close as the FINRA file, so they
        # follow it and still land before daily_predictions at 08:30.
        self.scheduler.add_job(
            self.market_regime_tracker.sync,
            CronTrigger(hour=7, minute=45, timezone=self.tz),
            id='market_regime_scan',
            misfire_grace_time=DAILY_MISFIRE_GRACE_SECONDS,
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
                CronTrigger(hour=6, minute=0, timezone=self.tz),
                id='options_snapshot',
                misfire_grace_time=DAILY_MISFIRE_GRACE_SECONDS,
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
                CronTrigger(hour=6, minute=45, timezone=self.tz),
                id='analyst_snapshot',
                misfire_grace_time=DAILY_MISFIRE_GRACE_SECONDS,
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
                CronTrigger(hour=8, minute=5, timezone=self.tz),
                id='technical_rating',
                misfire_grace_time=DAILY_MISFIRE_GRACE_SECONDS,
                replace_existing=True
            )

        # Every 4 hours: Trend forecasting + Macro themes
        async def run_trend_forecasting():
            await self.trend_forecaster.generate_forecasts()
            await self.trend_forecaster.generate_and_cache_macro_themes()

        self.scheduler.add_job(
            run_trend_forecasting,
            CronTrigger(hour='2,6,10,14,18,22', minute=0, timezone=self.tz),
            id='trend_forecasting',
            # Four-hourly, so the daily window is wrong in the other direction:
            # it would let a backlog fire a run the next schedule supersedes.
            misfire_grace_time=3600,
            replace_existing=True
        )

        # Daily at 00:30 KST: Sector daily snapshot
        self.scheduler.add_job(
            self.sector_analyzer.capture_daily_snapshot,
            CronTrigger(hour=0, minute=30, timezone=self.tz),
            id='sector_snapshot',
            misfire_grace_time=DAILY_MISFIRE_GRACE_SECONDS,
            replace_existing=True
        )

        # 03:00 KST on the 1st: top up the macro calendar from the web.
        #
        # Monthly rather than daily because the seeded schedule covers more than
        # a year and the agencies publish in annual batches — BLS, BEA and Census
        # post the following year in the autumn. Without this the calendar runs
        # dry in whichever January follows the seed's verification date.
        self.scheduler.add_job(
            self.refresh_macro_calendar,
            CronTrigger(day=1, hour=3, minute=0, timezone=self.tz),
            id='macro_calendar_refresh',
            misfire_grace_time=DAILY_MISFIRE_GRACE_SECONDS,
            replace_existing=True
        )

        # Weekly tip — DIGEST_DAY at DIGEST_HOUR, default Sunday 20:00 KST, which
        # is Sunday morning US Eastern: before the week it is written about opens,
        # which is the only time it is worth anything.
        self.scheduler.add_job(
            self.send_weekly_tip,
            CronTrigger(day_of_week=settings.digest_day, hour=settings.digest_hour,
                        minute=0, timezone=self.tz),
            id='weekly_tip',
            misfire_grace_time=DAILY_MISFIRE_GRACE_SECONDS,
            replace_existing=True
        )

        # 02:30 KST on the 1st: deepen price_history for seasonality.
        #
        # Monthly because the answer only changes when a ticker joins the
        # watchlist — everything already deep is skipped without a network call.
        # Ahead of the macro refresh at 03:00 so the two heavy monthly jobs do
        # not overlap on the phone.
        self.scheduler.add_job(
            self.refresh_seasonality_history,
            CronTrigger(day=1, hour=2, minute=30, timezone=self.tz),
            id='seasonality_refresh',
            misfire_grace_time=DAILY_MISFIRE_GRACE_SECONDS,
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
                            timezone=self.tz),
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
                CronTrigger(hour=9, minute=10, timezone=self.tz),
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
                CronTrigger(hour=9, minute=40, timezone=self.tz),
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
            run_date=datetime.datetime.now(self.tz) + datetime.timedelta(
                seconds=settings.startup_catchup_delay_seconds
            ),
            id='startup_catchup',
            replace_existing=True
        )

        # Before the scheduler, not as a job: /api/calendar, the dashboard card
        # and /macro all read macro_events directly, so on a fresh database the
        # calendar has to be populated by the time the API answers its first
        # request rather than whenever a cron next fires. Costs no network and no
        # LLM call — it is an upsert of compiled-in data.
        try:
            self.macro_calendar.seed()
        except Exception as e:
            log.error("orchestrator.macro_seed_failed", error=str(e))

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
        today_str = dt.datetime.now(self.tz).strftime("%Y-%m-%d")

        # Ahead of the watchlist check below: theme detection reads the article
        # corpus, not tracked tickers, so an empty watchlist must not skip it.
        # Wrapped separately so a thesis failure cannot cost the prediction and
        # resolution catch-ups that follow it.
        try:
            await self._catchup_thesis()
        except Exception as e:
            log.error("orchestrator.startup_catchup.thesis_error", error=str(e))

        # An empty next-60-days window means the seed has run dry — the compiled
        # schedule ended before today and the monthly cron has not fired since.
        # Waiting for the 1st would leave the calendar blank for up to a month,
        # so the refresh runs once here instead. Also ahead of the watchlist
        # check below: macro events belong to no ticker.
        try:
            if not await asyncio.to_thread(self.macro_calendar.upcoming, 60):
                log.info("orchestrator.startup_catchup.macro_calendar_empty")
                await self.refresh_macro_calendar()
            else:
                log.info("orchestrator.startup_catchup.macro_calendar_ok")
        except Exception as e:
            log.error("orchestrator.startup_catchup.macro_calendar_error", error=str(e))

        # Seasonality needs a decade of bars and refresh_history() stores three
        # months, so on a fresh database the first weekly tip would have nothing
        # to measure and would print no precedents at all. Capped at three
        # tickers: each one is a full-history Yahoo pull, and the monthly job
        # picks up the rest. Ahead of the watchlist check below because the
        # benchmarks are configured, not tracked.
        try:
            await self.refresh_seasonality_history(limit=STARTUP_DEEP_HISTORY_TICKERS)
        except Exception as e:
            log.error("orchestrator.startup_catchup.seasonality_error", error=str(e))

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

    async def _catchup_thesis(self) -> None:
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

        now_local = datetime.datetime.now(self.tz)
        due_today = now_local.replace(
            hour=THESIS_GENERATION_HOUR, minute=THESIS_GENERATION_MINUTE,
            second=0, microsecond=0,
        )
        if now_local < due_today:
            # Booted before it was due; the cron job will handle it normally.
            log.info("orchestrator.startup_catchup.thesis_not_due_yet")
            return

        midnight_utc = (
            now_local.replace(hour=0, minute=0, second=0, microsecond=0)
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
        
    today_str = datetime.datetime.now(ZoneInfo(settings.timezone)).strftime("%Y-%m-%d")

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

    today_str = datetime.datetime.now(ZoneInfo(settings.timezone)).strftime("%Y-%m-%d")
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
