"""
Tests for the trend forecaster's scheduled job.

`complete` is patched as imported into `pipeline.trend_forecaster`, never
`config.llm`: the module bound the name at import time, so patching the source
has no effect.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from pipeline.trend_forecaster import TrendForecaster


def outlook(sector: str) -> dict:
    """The shape `_generate_sector_outlook` returns."""
    return {
        "ticker": None,
        "sector": sector,
        "forecast_type": "sector_outlook",
        "scenario_label": "Most likely: steady",
        "time_horizon": "1m",
        "confidence": 0.6,
        "narrative": "Base case:\nSteady demand.",
        "key_drivers": ["rates"],
        "supporting_evidence": "- Headline: summary",
        "scenarios": [],
    }


class TestGenerateForecasts:
    def test_generates_sector_outlooks_and_nothing_per_ticker(self):
        db = MagicMock()
        forecaster = TrendForecaster(db)
        tech = outlook("Technology")

        with patch.object(forecaster, "_get_top_sectors",
                          return_value=[{"sector": "Technology", "article_count": 12}]), \
             patch.object(forecaster, "get_sector_outlook",
                          new=AsyncMock(return_value=tech)) as get_outlook, \
             patch.object(forecaster, "_store_forecast") as store, \
             patch("pipeline.trend_forecaster.event_bus.publish", new=AsyncMock()) as publish, \
             patch("pipeline.trend_forecaster.complete", new=AsyncMock()) as complete:
            result = asyncio.run(forecaster.generate_forecasts())

        assert result == [tech]
        get_outlook.assert_awaited_once_with("Technology")
        store.assert_called_once_with(tech)
        publish.assert_awaited_once_with("trend_forecast", tech)
        # The per-ticker scenario sets were the reasoning-model call in this job
        # and nothing ever read them. With the sector path stubbed out, any model
        # call or trending-ticker read left here would be that branch returning.
        complete.assert_not_awaited()
        db.get_top_trending_tickers.assert_not_called()
        db.get_recent_summaries_for_ticker.assert_not_called()

    def test_a_sector_without_an_outlook_is_neither_stored_nor_published(self):
        forecaster = TrendForecaster(MagicMock())

        with patch.object(forecaster, "_get_top_sectors",
                          return_value=[{"sector": "Energy", "article_count": 4}]), \
             patch.object(forecaster, "get_sector_outlook", new=AsyncMock(return_value=None)), \
             patch.object(forecaster, "_store_forecast") as store, \
             patch("pipeline.trend_forecaster.event_bus.publish", new=AsyncMock()) as publish:
            result = asyncio.run(forecaster.generate_forecasts())

        assert result == []
        store.assert_not_called()
        publish.assert_not_awaited()
