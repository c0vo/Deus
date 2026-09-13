import asyncio
import json

import numpy as np
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from data.models import NewsArticle
from orchestrator.scheduler import PipelineOrchestrator


def _article_row(article_id: str, url: str) -> dict:
    now = datetime.now().isoformat()
    return {
        "id": article_id,
        "headline": article_id,
        "summary": "",
        "content_hash": article_id.lower(),
        "source_name": "t",
        "source_type": "api",
        "url": url,
        "published_at": now,
        "fetched_at": now,
        "raw_data": {},
    }


def _reddit_row(article_id: str, url: str) -> dict:
    """A row the classifier routes to its Reddit lane."""
    return {
        **_article_row(article_id, url),
        "source_name": "reddit_wallstreetbets",
        "source_type": "social",
    }


def _build_orchestrator(db_mock: MagicMock) -> PipelineOrchestrator:
    """Wire up an orchestrator whose external calls are all stubbed out."""
    orchestrator = PipelineOrchestrator(db=db_mock)

    # run_pipeline_cycle sets this up before calling _process_batch; the tests
    # drive the batch directly, so they have to seed it themselves.
    orchestrator._cycle_counts = {
        "fetched": 0, "inserted": 0, "classified": 0, "ranked": 0,
        "embedded": 0, "alerts": 0, "errors": 0, "llm_calls": 0, "llm_cost": 0.0,
    }

    orchestrator.aggregator = MagicMock()
    orchestrator.aggregator.fetch_all = AsyncMock(return_value=[])

    orchestrator.embedder = MagicMock()
    orchestrator.embedder.embed_articles = AsyncMock(return_value=[])

    classified = MagicMock()
    classified.id = "C"
    classified.event_type = "earnings"
    classified.sentiment_score = 0.5
    classified.urgency = "low"
    classified.suggested_direction = "bullish"
    classified.affected_sectors = []
    classified.affected_tickers = []
    classified.classification_summary = "test"

    orchestrator.classifier = MagicMock()
    orchestrator.classifier.classify = AsyncMock(return_value=classified)
    # The backlog job refuses to do anything when no model slug is set, so the
    # default stub says one is — for the Reddit lane too.
    orchestrator.classifier.is_configured = MagicMock(return_value=True)
    orchestrator.classifier.is_reddit_configured = MagicMock(return_value=True)

    # Classification is batched: the scheduler hands a whole chunk to
    # classify_batch, which labels the articles in place and returns them.
    async def _classify_batch(chunk):
        for article in chunk:
            article.event_type = "earnings"
            article.sentiment_score = 0.5
            article.urgency = "low"
            article.suggested_direction = "bullish"
            article.affected_sectors = []
            article.affected_tickers = []
            article.classification_summary = "test"
        return chunk

    orchestrator.classifier.classify_batch = AsyncMock(side_effect=_classify_batch)

    orchestrator.ranker = MagicMock()
    orchestrator.ranker.rank_batch = AsyncMock(return_value=[])
    orchestrator.market_scanner = MagicMock()
    orchestrator.market_scanner.run_scan = AsyncMock()
    orchestrator.market_scanner.check_earnings = AsyncMock()

    return orchestrator


def _make_db_mock(rows: list[dict] | None = None) -> MagicMock:
    db_mock = MagicMock()
    db_mock.has_sqlite_vec = False
    rows = rows if rows is not None else [
        _article_row("A", "1"),
        _article_row("B", "2"),
        _article_row("C", "3"),
    ]
    db_mock.get_unclassified_articles.return_value = list(rows)
    db_mock.get_classification_candidates.return_value = list(rows)
    db_mock.row_to_article.side_effect = lambda r: NewsArticle(**r)

    # Real return types matter here: these values are compared against ints and
    # serialised into the status JSON, so a bare MagicMock would blow up rather
    # than behave like a zero.
    db_mock.mark_stale_unclassified.return_value = 0
    db_mock.record_classification_failure.return_value = {}
    db_mock.requeue_error_articles.return_value = 0
    db_mock.reset_parked_classification_attempts.return_value = 0
    # Non-empty marker: both one-time repairs have already run, so the default
    # path under test is the steady state.
    db_mock.get_config.return_value = '{"requeued": 0}'
    db_mock.get_unranked_articles.return_value = []

    conn_mock = MagicMock()
    db_mock.connection.return_value.__enter__.return_value = conn_mock

    def mock_execute(query, params=None):
        cursor = MagicMock()
        if query.startswith("SELECT embedding"):
            cursor.fetchone.return_value = {
                "embedding": np.random.rand(10).astype(np.float32).tobytes()
            }
        elif query.startswith("SELECT * FROM articles WHERE id"):
            cursor.fetchone.return_value = {
                "event_type": "earnings",
                "sentiment_score": 0.5,
                "urgency": "low",
                "suggested_direction": "bullish",
                "affected_sectors": "[]",
                "affected_tickers": "[]",
                "classification_summary": "Summary",
            }
        else:
            cursor.fetchone.return_value = None
            cursor.fetchall.return_value = []
        return cursor

    conn_mock.execute.side_effect = mock_execute
    return db_mock


async def _run_backlog(
    orchestrator: PipelineOrchestrator, *, strict: bool = True
) -> dict:
    """
    Run one classification backlog pass with the event bus stubbed out.

    ``event_bus.publish`` writes to the real SSE outbox through its own
    ``Database`` instance, so it is always patched here — an unpatched publish
    would reach the actual ``storage/scrooge.db``.

    ``strict`` turns a swallowed ``log.error`` into a test failure, the way the
    old ``_run_batch`` helper did; tests that *expect* an error log pass False.
    """
    bus = MagicMock()
    bus.publish = AsyncMock()

    with patch("orchestrator.scheduler.event_bus", bus):
        if not strict:
            return await orchestrator.run_classify_backlog()
        with patch("orchestrator.scheduler.log.error") as mock_log:
            def raise_error(msg, **kwargs):
                raise AssertionError(f"{msg}: {kwargs}")
            mock_log.side_effect = raise_error
            return await orchestrator.run_classify_backlog()


def _error_stamps(db_mock: MagicMock) -> list:
    return [
        call for call in db_mock.update_classification.call_args_list
        if call.kwargs.get("event_type") == "error"
    ]


# ── Semantic dedup inside the backlog job ───────────────────────────────────
#
# These four used to drive _process_batch, which classified as a side effect of
# the fetch loop. Classification now lives in its own job, so they drive that.


@pytest.mark.asyncio
async def test_duplicate_inherits_classification_and_is_flagged():
    """A near-identical article copies the original's labels and is flagged."""
    db_mock = _make_db_mock()
    # The nearest-neighbour search lives in the database layer now, so the test
    # stubs that rather than patching numpy inside the scheduler.
    db_mock.find_duplicate.return_value = ("X", 0.90)

    orchestrator = _build_orchestrator(db_mock)
    await _run_backlog(orchestrator)

    inherited = [
        call for call in db_mock.update_classification.call_args_list
        if call.kwargs.get("event_type") == "earnings"
    ]
    assert inherited, "duplicate should inherit the source article's classification"

    assert db_mock.mark_duplicate.called, "duplicate should be flagged via duplicate_of"
    flagged_ids = {call.args[0] for call in db_mock.mark_duplicate.call_args_list}
    assert flagged_ids == {"A", "B", "C"}


@pytest.mark.asyncio
async def test_duplicates_do_not_inflate_ticker_mentions():
    """Flagged duplicates must not be counted again in trending."""
    db_mock = _make_db_mock()
    db_mock.find_duplicate.return_value = ("X", 0.90)

    orchestrator = _build_orchestrator(db_mock)
    await _run_backlog(orchestrator)

    assert not db_mock.insert_ticker_mentions.called, (
        "a duplicate wrote ticker mentions, which is what inflated trending counts"
    )


@pytest.mark.asyncio
async def test_unique_article_is_classified_not_flagged():
    """With no near neighbour, the article goes to the classifier."""
    db_mock = _make_db_mock()
    db_mock.find_duplicate.return_value = None

    orchestrator = _build_orchestrator(db_mock)
    await _run_backlog(orchestrator)

    # One batched call carrying all three, rather than three separate calls.
    assert orchestrator.classifier.classify_batch.await_count == 1
    sent = {a.id for a in orchestrator.classifier.classify_batch.call_args.args[0]}
    assert sent == {"A", "B", "C"}
    assert not db_mock.mark_duplicate.called
    assert db_mock.mark_dedup_checked.call_count == 3


@pytest.mark.asyncio
async def test_one_duplicate_does_not_abort_the_batch():
    """Regression: a `break` used to exit the loop, skipping the rest of the batch."""
    db_mock = _make_db_mock()

    # Only article A has a near neighbour. Keyed on id rather than a fixed
    # sequence so the test does not depend on how often the batch runs.
    def only_a_is_duplicate(article_id=None, **kwargs):
        return ("X", 0.90) if article_id == "A" else None

    db_mock.find_duplicate.side_effect = only_a_is_duplicate

    orchestrator = _build_orchestrator(db_mock)
    await _run_backlog(orchestrator)

    classified_ids = {
        article.id
        for call in orchestrator.classifier.classify_batch.call_args_list
        for article in call.args[0]
    }
    assert classified_ids == {"B", "C"}, (
        f"expected B and C to be classified after the duplicate, got {classified_ids}"
    )


# ── Failure handling ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_transient_exhaustion_counts_attempts_and_stamps_nothing():
    """A provider outage must not be written down as a verdict on the article.

    `_classify_chunk_with_retry` returning False means every attempt hit an
    infrastructure fault. The old code had no attempt counter at all, so those
    rows were re-selected by the LIFO query two seconds later, forever.
    """
    db_mock = _make_db_mock()
    db_mock.find_duplicate.return_value = None

    orchestrator = _build_orchestrator(db_mock)
    orchestrator._classify_chunk_with_retry = AsyncMock(return_value=False)

    status = await _run_backlog(orchestrator)

    assert db_mock.record_classification_failure.called
    recorded = set(db_mock.record_classification_failure.call_args.args[0])
    assert recorded == {"A", "B", "C"}
    assert not _error_stamps(db_mock), "an outage must never stamp 'error'"
    assert status["failed"] == 3
    assert status["classified"] == 0


@pytest.mark.asyncio
async def test_transient_exhaustion_stays_null_even_at_the_attempt_cap():
    """Attempts retire the row from the queue; they do not condemn it."""
    db_mock = _make_db_mock()
    db_mock.find_duplicate.return_value = None
    db_mock.record_classification_failure.return_value = {"A": 3, "B": 3, "C": 3}

    orchestrator = _build_orchestrator(db_mock)
    orchestrator._classify_chunk_with_retry = AsyncMock(return_value=False)

    await _run_backlog(orchestrator)

    assert not _error_stamps(db_mock)


@pytest.mark.asyncio
async def test_unanswered_article_is_retried_before_it_is_written_off():
    """A model that skips an article gets an attempt counted, not a verdict."""
    db_mock = _make_db_mock()
    db_mock.find_duplicate.return_value = None
    db_mock.record_classification_failure.return_value = {"A": 1, "B": 1, "C": 1}

    orchestrator = _build_orchestrator(db_mock)
    # classify_batch returns without labelling anything — the shape a truncated
    # or unmatched response produces.
    orchestrator.classifier.classify_batch = AsyncMock(side_effect=lambda chunk: chunk)

    status = await _run_backlog(orchestrator, strict=False)

    assert db_mock.record_classification_failure.called
    assert not _error_stamps(db_mock)
    assert status["classified"] == 0


@pytest.mark.asyncio
async def test_unanswered_article_is_stamped_error_once_attempts_run_out():
    db_mock = _make_db_mock()
    db_mock.find_duplicate.return_value = None
    db_mock.record_classification_failure.return_value = {"A": 3, "B": 3, "C": 3}

    orchestrator = _build_orchestrator(db_mock)
    orchestrator.classifier.classify_batch = AsyncMock(side_effect=lambda chunk: chunk)

    await _run_backlog(orchestrator, strict=False)

    stamped = {call.kwargs["article_id"] for call in _error_stamps(db_mock)}
    assert stamped == {"A", "B", "C"}


@pytest.mark.asyncio
async def test_unconfigured_classifier_aborts_before_touching_the_queue():
    db_mock = _make_db_mock()
    orchestrator = _build_orchestrator(db_mock)
    orchestrator.classifier.is_configured = MagicMock(return_value=False)

    status = await _run_backlog(orchestrator, strict=False)

    assert status["configured"] is False
    assert not db_mock.get_classification_candidates.called
    assert not db_mock.mark_stale_unclassified.called
    assert not _error_stamps(db_mock)


@pytest.mark.asyncio
async def test_classifier_not_configured_exception_aborts_without_stamping():
    """The mid-run version of the same failure.

    `_call_with_fallback` raises when the slug list is empty. It has to reach the
    job rather than be swallowed: the generic handler would read it as "the model
    answered with garbage" and stamp every row.
    """
    from pipeline.classifier import ClassifierNotConfigured

    db_mock = _make_db_mock()
    db_mock.find_duplicate.return_value = None

    orchestrator = _build_orchestrator(db_mock)
    orchestrator.classifier.classify_batch = AsyncMock(
        side_effect=ClassifierNotConfigured("no slug")
    )

    status = await _run_backlog(orchestrator, strict=False)

    assert status["configured"] is False
    assert not _error_stamps(db_mock)
    assert not db_mock.record_classification_failure.called


@pytest.mark.asyncio
async def test_reddit_rows_wait_uncounted_while_their_lane_is_unconfigured():
    """News is classified and persisted; Reddit rows are neither sent nor counted.

    The stub below behaves like the real classifier with no Reddit slug: it hands
    Reddit rows back untouched. `_persist_classifications` reads untouched as "the
    model skipped it", so a Reddit row that reached a chunk would spend an attempt
    every run and be stamped 'error' at the cap, for a call never made.
    """
    from pipeline.classifier import ArticleClassifier

    rows = [_article_row("A", "1"), _reddit_row("R", "2"), _article_row("B", "3")]
    db_mock = _make_db_mock(rows)
    db_mock.find_duplicate.return_value = None
    db_mock.record_classification_failure.side_effect = (
        lambda ids: {article_id: 3 for article_id in ids}
    )

    orchestrator = _build_orchestrator(db_mock)
    orchestrator.classifier.is_reddit_configured = MagicMock(return_value=False)

    async def _news_only(chunk):
        for article in chunk:
            if ArticleClassifier._is_reddit(article):
                continue
            article.event_type = "earnings"
            article.sentiment_score = 0.5
            article.urgency = "low"
            article.suggested_direction = "bullish"
            article.affected_sectors = []
            article.affected_tickers = []
            article.classification_summary = "test"
        return chunk

    orchestrator.classifier.classify_batch = AsyncMock(side_effect=_news_only)

    status = await _run_backlog(orchestrator)

    sent = {
        article.id
        for call in orchestrator.classifier.classify_batch.call_args_list
        for article in call.args[0]
    }
    assert sent == {"A", "B"}
    persisted = {
        call.kwargs["article_id"]
        for call in db_mock.update_classification.call_args_list
    }
    assert persisted == {"A", "B"}
    assert not db_mock.record_classification_failure.called
    assert not _error_stamps(db_mock)
    assert status["classified"] == 2
    assert status["failed"] == 0
    assert status["configured"] is True


@pytest.mark.asyncio
async def test_unconfigured_reddit_lane_is_reported_once_and_narrows_the_query(
    monkeypatch,
):
    from config.settings import settings

    # One article per chunk, so a report made per chunk would show up three times.
    monkeypatch.setattr(settings, "classify_batch_size", 1)

    db_mock = _make_db_mock()
    db_mock.find_duplicate.return_value = None

    orchestrator = _build_orchestrator(db_mock)
    orchestrator.classifier.is_reddit_configured = MagicMock(return_value=False)

    with patch("orchestrator.scheduler.log.warning") as warning:
        status = await _run_backlog(orchestrator)

    reports = [
        call for call in warning.call_args_list
        if call.args and call.args[0] == "orchestrator.classify_backlog_reddit_unconfigured"
    ]
    assert len(reports) == 1
    assert reports[0].kwargs["settings_checked"] == [
        "MODEL_REDDIT_SENTIMENT", "MODEL_REDDIT_SENTIMENT_FALLBACK",
    ]
    # Rows the run cannot act on must not take the slots news could use.
    assert db_mock.get_classification_candidates.call_args.kwargs == {
        "exclude_reddit": True,
    }
    # The news lane is configured and says so.
    assert status["configured"] is True
    assert status["classified"] == 3


# ── Bounds ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_candidates_are_requested_with_the_configured_bounds():
    from config.settings import settings

    db_mock = _make_db_mock()
    db_mock.find_duplicate.return_value = None

    orchestrator = _build_orchestrator(db_mock)
    await _run_backlog(orchestrator)

    assert db_mock.get_classification_candidates.call_args.args == (
        settings.classify_per_run_limit,
        settings.classify_max_attempts,
        settings.classify_max_age_days,
    )
    # Both lanes configured, so nothing is left out of the query.
    assert db_mock.get_classification_candidates.call_args.kwargs == {
        "exclude_reddit": False,
    }


@pytest.mark.asyncio
async def test_out_of_window_rows_are_retired_before_selection():
    from config.settings import settings

    db_mock = _make_db_mock()
    db_mock.find_duplicate.return_value = None
    db_mock.mark_stale_unclassified.return_value = 7

    orchestrator = _build_orchestrator(db_mock)
    status = await _run_backlog(orchestrator)

    db_mock.mark_stale_unclassified.assert_called_once_with(
        settings.classify_max_age_days
    )
    assert status["stale_marked"] == 7


@pytest.mark.asyncio
async def test_chunks_in_flight_never_exceed_the_concurrency_setting(monkeypatch):
    from config.settings import settings

    monkeypatch.setattr(settings, "classify_concurrency", 2)
    monkeypatch.setattr(settings, "classify_batch_size", 1)

    rows = [_article_row(f"A{i}", str(i)) for i in range(6)]
    db_mock = _make_db_mock(rows)
    db_mock.find_duplicate.return_value = None

    orchestrator = _build_orchestrator(db_mock)

    in_flight = 0
    peak = 0

    async def _tracked(chunk):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0)
        for article in chunk:
            article.event_type = "earnings"
            article.sentiment_score = 0.0
            article.urgency = "low"
            article.suggested_direction = "neutral"
            article.affected_sectors = []
            article.affected_tickers = []
            article.classification_summary = "t"
        in_flight -= 1
        return chunk

    orchestrator.classifier.classify_batch = AsyncMock(side_effect=_tracked)

    await _run_backlog(orchestrator)

    assert orchestrator.classifier.classify_batch.await_count == 6
    assert peak <= 2, f"{peak} chunks were in flight with concurrency 2"


@pytest.mark.asyncio
async def test_ranking_runs_even_with_an_empty_backlog():
    """Ranking must not be gated on there being new candidates.

    An article whose ranking call failed earlier is already classified, so it
    would never be retried — and unranked means it reaches neither the daily
    brief nor alerts. On the phone the backlog is currently *zero*, so gating
    would have disabled ranking outright.
    """
    db_mock = _make_db_mock(rows=[])
    orchestrator = _build_orchestrator(db_mock)
    orchestrator._rank_pending = AsyncMock()

    status = await _run_backlog(orchestrator)

    assert status["candidates"] == 0
    orchestrator._rank_pending.assert_awaited_once()
    assert not orchestrator.classifier.classify_batch.called


@pytest.mark.asyncio
async def test_a_second_run_is_skipped_while_one_is_in_progress():
    """Two triggers (the interval job and the tail of a cycle) need one lock.

    APScheduler's max_instances=1 only stops a job overlapping itself.
    """
    db_mock = _make_db_mock()
    db_mock.find_duplicate.return_value = None

    orchestrator = _build_orchestrator(db_mock)
    await orchestrator._classify_lock.acquire()
    try:
        status = await orchestrator.run_classify_backlog(trigger="cycle")
    finally:
        orchestrator._classify_lock.release()

    assert status == {"skipped": True, "trigger": "cycle"}
    assert not db_mock.get_classification_candidates.called


# ── Status reporting ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_status_is_persisted_and_published():
    """Both halves: user_config for a page load, SSE for a live update."""
    db_mock = _make_db_mock()
    db_mock.find_duplicate.return_value = None

    orchestrator = _build_orchestrator(db_mock)

    bus = MagicMock()
    bus.publish = AsyncMock()
    with patch("orchestrator.scheduler.event_bus", bus):
        status = await orchestrator.run_classify_backlog(trigger="interval")

    key, value = db_mock.set_config.call_args.args
    assert key == "classify_backlog_status"
    persisted = json.loads(value)
    assert persisted["trigger"] == "interval"
    assert persisted["candidates"] == 3
    assert persisted["classified"] == 3
    assert set(status) >= {
        "last_run_at", "trigger", "candidates", "classified", "failed",
        "stale_marked", "error_requeued", "attempts_unparked", "duration_s",
        "configured",
    }

    published = [c.args for c in bus.publish.call_args_list]
    assert ("classification_status", status) in published


def _marker_config(*absent: str):
    """`get_config` that reports only the named markers as never having run.

    There are two independent one-shot repairs on this path, both keyed in
    `user_config`. A blanket empty string makes both of them fire, which is how a
    test for one silently ends up asserting the sum of both.
    """
    def get_config(key, default=""):
        return "" if key in absent else '{"ran": true}'
    return get_config


@pytest.mark.asyncio
async def test_error_requeue_runs_once_and_only_once():
    db_mock = _make_db_mock()
    db_mock.find_duplicate.return_value = None
    # No marker yet: the amnesty has never run on this database.
    db_mock.get_config.side_effect = _marker_config(
        PipelineOrchestrator.ERROR_REQUEUE_MARKER
    )
    db_mock.requeue_error_articles.return_value = 42

    orchestrator = _build_orchestrator(db_mock)
    status = await _run_backlog(orchestrator)

    assert status["error_requeued"] == 42
    assert db_mock.set_config.call_args_list[0].args[0] == (
        PipelineOrchestrator.ERROR_REQUEUE_MARKER
    )

    # Second pass with the marker present writes nothing further.
    db_mock.get_config.side_effect = None
    db_mock.get_config.return_value = '{"requeued": 42}'
    db_mock.requeue_error_articles.reset_mock()
    status = await _run_backlog(orchestrator)

    assert status["error_requeued"] == 0
    assert not db_mock.requeue_error_articles.called


@pytest.mark.asyncio
async def test_parked_attempts_reset_runs_once_and_only_once():
    """
    The batch-schema regression's casualties get exactly one amnesty.

    Nine of every ten rows in a batch took a `batch_missing_result` against a
    request that asked for one object, and spent all three attempts inside
    fifteen minutes. Reset them once; run it every cycle instead and the attempt
    counter stops bounding anything.
    """
    from config.settings import settings

    db_mock = _make_db_mock()
    db_mock.find_duplicate.return_value = None
    db_mock.get_config.side_effect = _marker_config(
        PipelineOrchestrator.ATTEMPTS_RESET_MARKER
    )
    db_mock.reset_parked_classification_attempts.return_value = 137
    db_mock.requeue_error_articles.return_value = 8

    orchestrator = _build_orchestrator(db_mock)
    status = await _run_backlog(orchestrator)

    assert status["attempts_unparked"] == 137
    # The rows this regression wrote off as 'error' are requeued too, and folded
    # into the same counter the other amnesty reports.
    assert status["error_requeued"] == 8
    db_mock.reset_parked_classification_attempts.assert_called_once_with(
        settings.classify_max_attempts
    )
    db_mock.requeue_error_articles.assert_called_once_with(
        PipelineOrchestrator.ATTEMPTS_RESET_ERROR_AGE_DAYS
    )

    markers = [c.args[0] for c in db_mock.set_config.call_args_list]
    assert PipelineOrchestrator.ATTEMPTS_RESET_MARKER in markers

    # Second pass with the marker present touches neither table.
    db_mock.get_config.side_effect = None
    db_mock.get_config.return_value = '{"unparked": 137}'
    db_mock.reset_parked_classification_attempts.reset_mock()
    db_mock.requeue_error_articles.reset_mock()
    status = await _run_backlog(orchestrator)

    assert status["attempts_unparked"] == 0
    assert status["error_requeued"] == 0
    assert not db_mock.reset_parked_classification_attempts.called
    assert not db_mock.requeue_error_articles.called


@pytest.mark.asyncio
async def test_attempts_are_reset_before_candidates_are_selected():
    """An amnesty that lands after the candidate scan helps nobody this cycle."""
    db_mock = _make_db_mock()
    db_mock.find_duplicate.return_value = None
    db_mock.get_config.side_effect = _marker_config(
        PipelineOrchestrator.ATTEMPTS_RESET_MARKER
    )

    order: list[str] = []
    db_mock.reset_parked_classification_attempts.side_effect = (
        lambda *a: order.append("reset") or 0
    )
    db_mock.mark_stale_unclassified.side_effect = lambda *a: order.append("stale") or 0
    db_mock.get_classification_candidates.side_effect = (
        lambda *a, **k: order.append("candidates") or []
    )

    orchestrator = _build_orchestrator(db_mock)
    await _run_backlog(orchestrator)

    assert order == ["reset", "stale", "candidates"]


@pytest.mark.asyncio
async def test_new_articles_is_published_once_per_chunk_not_per_article():
    db_mock = _make_db_mock()
    db_mock.find_duplicate.return_value = None

    orchestrator = _build_orchestrator(db_mock)

    bus = MagicMock()
    bus.publish = AsyncMock()
    with patch("orchestrator.scheduler.event_bus", bus):
        await orchestrator.run_classify_backlog()
        # _persist_classifications schedules the publish as a task.
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    new_articles = [
        c.args[1] for c in bus.publish.call_args_list if c.args[0] == "new_articles"
    ]
    assert len(new_articles) == 1
    # The payload shape the dashboard's onNewArticles handler reads.
    assert len(new_articles[0]["articles"]) == 3


# ── Cycle-level LLM accounting ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cycle_records_real_llm_call_counts():
    """`pipeline_metrics.llm_calls_count` was initialised and never incremented.

    Every row in the table read 0 while `llm_usage_log` showed the calls, so the
    dashboard's LLM-calls cell was a permanent zero. The cycle now diffs
    `config.usage.tally` across itself.
    """
    from config.usage import track_llm

    db_mock = _make_db_mock()
    db_mock.find_duplicate.return_value = None

    orchestrator = _build_orchestrator(db_mock)

    async def _classify_and_bill(chunk):
        # Exercise the real instrumentation rather than the counter directly:
        # the thing under test is that track_llm and the cycle share one tally.
        with track_llm(MagicMock(), "test/model", "classify_batch") as usage:
            usage.cost = 0.0025
        for article in chunk:
            article.event_type = "earnings"
            article.sentiment_score = 0.0
            article.urgency = "low"
            article.suggested_direction = "neutral"
            article.affected_sectors = []
            article.affected_tickers = []
            article.classification_summary = "t"
        return chunk

    orchestrator.classifier.classify_batch = AsyncMock(side_effect=_classify_and_bill)

    bus = MagicMock()
    bus.publish = AsyncMock()

    # Shortened, not stubbed. The cycle's inner loop is
    # `while not fetch_task.done(): await _process_batch(); await sleep(2)`, so a
    # sleep that never suspends starves the fetch task and the loop spins forever.
    # The original is captured first: the patch target *is* asyncio.sleep.
    real_sleep = asyncio.sleep

    def _yield(*_args, **_kwargs):
        return real_sleep(0)

    with patch("orchestrator.scheduler.event_bus", bus), \
            patch("orchestrator.scheduler.asyncio.sleep", _yield):
        await orchestrator.run_pipeline_cycle()

    counts = db_mock.insert_pipeline_metrics.call_args.args[1]
    assert counts["llm_calls"] >= 1
    assert counts["llm_cost"] == pytest.approx(0.0025, abs=1e-6)
    # And the cycle folded the backlog run's result into its own tally.
    assert counts["classified"] == 3


@pytest.mark.asyncio
async def test_usage_tally_counts_failures_separately():
    import config.usage as usage_module

    before = usage_module.tally.snapshot()
    with pytest.raises(RuntimeError):
        with usage_module.track_llm(MagicMock(), "m", "op"):
            raise RuntimeError("boom")

    assert usage_module.tally.calls == before.calls + 1
    assert usage_module.tally.errors == before.errors + 1
    assert usage_module.tally.cost == pytest.approx(before.cost)


# ── Embed pass ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_process_batch_no_longer_classifies():
    """Classification moved out of the fetch loop entirely."""
    db_mock = _make_db_mock()
    db_mock.find_duplicate.return_value = None

    orchestrator = _build_orchestrator(db_mock)
    await orchestrator._process_batch()

    assert not orchestrator.classifier.classify_batch.called
    assert not db_mock.get_classification_candidates.called
