"""
Tests for the sector analyzer's 15-minute run.

The run is pure aggregation over our own tables. It used to end with a model
call that invented a one-sentence rationale per hot ticker from general
knowledge, which nothing read; these pin down that the hot tickers still reach
the table and the stream without one.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from pipeline.sector_analyzer import SectorAnalyzer


def discovered(ticker: str = "SMCI") -> dict:
    """The dict `_discover_hot_tickers` produces."""
    return {
        "ticker": ticker,
        "mention_count": 7,
        "avg_sentiment": 0.42,
        "sectors": ["Technology"],
    }


class TestRunAnalysis:
    def test_hot_tickers_are_upserted_and_published_without_a_rationale(self):
        db = MagicMock()
        analyzer = SectorAnalyzer(db)
        hot = discovered()

        with patch.object(analyzer, "_compute_sector_snapshot", return_value=[]), \
             patch.object(analyzer, "_detect_sector_rotations", return_value=[]), \
             patch.object(analyzer, "_discover_hot_tickers", return_value=[hot]), \
             patch("pipeline.sector_analyzer.event_bus.publish", new=AsyncMock()) as publish:
            results = asyncio.run(analyzer.run_analysis())

        assert results["hot_tickers_found"] == 1
        db.upsert_hot_ticker.assert_called_once_with(hot)
        # A payload without the key is what keeps upsert_hot_ticker from
        # touching the rationale a thesis promotion wrote.
        assert "rationale" not in db.upsert_hot_ticker.call_args.args[0]
        publish.assert_awaited_once_with("hot_tickers", [hot])

    def test_no_hot_tickers_means_no_upsert_and_no_publish(self):
        db = MagicMock()
        analyzer = SectorAnalyzer(db)

        with patch.object(analyzer, "_compute_sector_snapshot", return_value=[]), \
             patch.object(analyzer, "_detect_sector_rotations", return_value=[]), \
             patch.object(analyzer, "_discover_hot_tickers", return_value=[]), \
             patch("pipeline.sector_analyzer.event_bus.publish", new=AsyncMock()) as publish:
            results = asyncio.run(analyzer.run_analysis())

        assert results["hot_tickers_found"] == 0
        db.upsert_hot_ticker.assert_not_called()
        publish.assert_not_awaited()
