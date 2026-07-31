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
    orchestrator.aggregator.fetch_all = AsyncMock()

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


def _make_db_mock() -> MagicMock:
    db_mock = MagicMock()
    db_mock.has_sqlite_vec = False
    db_mock.get_unclassified_articles.return_value = [
        _article_row("A", "1"),
        _article_row("B", "2"),
        _article_row("C", "3"),
    ]
    db_mock.row_to_article.side_effect = lambda r: NewsArticle(**r)

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


async def _run_batch(orchestrator: PipelineOrchestrator) -> None:
    """
    Run a single batch, surfacing any logged error as a test failure.

    ``run_pipeline_cycle`` calls ``_process_batch`` repeatedly while the fetch
    task is in flight, so exercising the batch directly keeps call counts
    deterministic. ``_process_batch`` swallows exceptions into ``log.error``,
    hence the patch.
    """
    with patch("orchestrator.scheduler.log.error") as mock_log:
        def raise_error(msg, **kwargs):
            if "error" in kwargs:
                raise AssertionError(f"{msg}: {kwargs['error']}")
        mock_log.side_effect = raise_error
        await orchestrator._process_batch()


@pytest.mark.asyncio
async def test_duplicate_inherits_classification_and_is_flagged():
    """A near-identical article copies the original's labels and is flagged."""
    db_mock = _make_db_mock()
    # The nearest-neighbour search lives in the database layer now, so the test
    # stubs that rather than patching numpy inside the scheduler.
    db_mock.find_duplicate.return_value = ("X", 0.90)

    orchestrator = _build_orchestrator(db_mock)
    await _run_batch(orchestrator)

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
    await _run_batch(orchestrator)

    assert not db_mock.insert_ticker_mentions.called, (
        "a duplicate wrote ticker mentions, which is what inflated trending counts"
    )


@pytest.mark.asyncio
async def test_unique_article_is_classified_not_flagged():
    """With no near neighbour, the article goes to the classifier."""
    db_mock = _make_db_mock()
    db_mock.find_duplicate.return_value = None

    orchestrator = _build_orchestrator(db_mock)
    await _run_batch(orchestrator)

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
    await _run_batch(orchestrator)

    classified_ids = {
        article.id
        for call in orchestrator.classifier.classify_batch.call_args_list
        for article in call.args[0]
    }
    assert classified_ids == {"B", "C"}, (
        f"expected B and C to be classified after the duplicate, got {classified_ids}"
    )
