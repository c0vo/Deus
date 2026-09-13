"""
Trend Forecaster Component

LLM-powered forward-looking analysis of current market trends.
Generates "If this trend continues..." scenario analysis for
sectors.

Runs every 4 hours as a background job.
"""

from __future__ import annotations

import json
import time
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Optional

from config.logging_config import get_logger
from config.llm import complete, is_llm_configured, strip_code_fence
from config.settings import settings
from config.usage import track_llm
from data.database import Database
from api.sse_manager import event_bus

log = get_logger(__name__)

# Module-level in-memory cache for macro themes (shared across all TrendForecaster instances)
_macro_themes_cache: dict = {"data": None, "timestamp": 0}
_MACRO_THEMES_TTL = 14400  # 4 hours (matches scheduler cadence)

# Per-sector outlook cache, keyed by normalised sector name. /forecast takes a
# free-text sector from the user, so the key space is unbounded — expired
# entries are evicted on write rather than left to accumulate.
_sector_outlook_cache: dict[str, tuple[Optional[dict], float]] = {}
_SECTOR_OUTLOOK_TTL = 14400  # 4 hours (matches scheduler cadence)


class TrendForecaster:
    """LLM-based forward-looking analysis of current market trends."""

    def __init__(self, db: Database):
        self.db = db

    async def generate_forecasts(self, max_sectors: int = 3) -> list[dict]:
        """
        Generate outlooks for the most-covered sectors.
        Called every 4 hours by the scheduler.

        Sectors only. The job used to open with a scenario set per trending
        ticker on the reasoning model at high effort, and nothing ever read
        those rows: every reader of trend_forecasts asks for a sector.
        """
        all_forecasts = []

        sector_data = self._get_top_sectors(hours=24, limit=max_sectors)
        for sd in sector_data:
            forecast = await self.get_sector_outlook(sd["sector"])
            if forecast:
                self._store_forecast(forecast)
                all_forecasts.append(forecast)

        if all_forecasts:
            log.info("trend_forecaster.generated", count=len(all_forecasts))
            # Publish to SSE event bus for real-time dashboard
            try:
                for forecast in all_forecasts:
                    await event_bus.publish("trend_forecast", forecast)
            except Exception as e:
                log.warning("trend_forecaster.sse_publish_failed", error=str(e))

        return all_forecasts

    def _get_top_sectors(self, hours: int = 24, limit: int = 5) -> list[dict]:
        """Get top sectors by article volume."""
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        sector_counts = {}

        with self.db.connection() as conn:
            rows = conn.execute(
                """
                SELECT affected_sectors FROM articles
                WHERE published_at >= ? AND affected_sectors IS NOT NULL
                  AND (event_type IS NULL OR event_type != 'noise')
                """,
                (cutoff,)
            ).fetchall()

        for row in rows:
            try:
                sectors = json.loads(row["affected_sectors"])
                for s in sectors:
                    sector_counts[s] = sector_counts.get(s, 0) + 1
            except (json.JSONDecodeError, TypeError):
                continue

        sorted_sectors = sorted(sector_counts.items(), key=lambda x: x[1], reverse=True)
        return [{"sector": s, "article_count": c} for s, c in sorted_sectors[:limit]]

    async def get_sector_outlook(self, sector: str) -> Optional[dict]:
        """
        Cached sector outlook — the entry point every caller should use.

        The scheduled 4-hourly forecast run and the /forecast command share this
        cache, so the scheduled job acts as the producer that warms the top
        sectors and the on-demand command usually costs nothing.
        """
        key = sector.strip().lower()
        now = time.time()

        cached = _sector_outlook_cache.get(key)
        if cached and (now - cached[1]) < _SECTOR_OUTLOOK_TTL:
            log.info("trend_forecaster.sector_outlook_cache_hit", sector=sector)
            return cached[0]

        outlook = await self._generate_sector_outlook(sector)

        # Drop expired keys before inserting, so a stream of one-off /forecast
        # arguments cannot grow this dict without bound.
        for stale in [k for k, (_, ts) in _sector_outlook_cache.items()
                      if (now - ts) >= _SECTOR_OUTLOOK_TTL]:
            del _sector_outlook_cache[stale]
        _sector_outlook_cache[key] = (outlook, now)

        return outlook

    async def _generate_sector_outlook(self, sector: str) -> Optional[dict]:
        """Generate a forward-looking outlook for a specific sector."""
        if not is_llm_configured() or not settings.model_trend_outlook:
            return None

        # Get recent high-importance articles for this sector
        with self.db.connection() as conn:
            rows = conn.execute(
                """
                SELECT headline, classification_summary
                FROM articles
                WHERE affected_sectors LIKE ?
                  AND published_at >= datetime('now', '-7 days')
                  AND importance_score >= 5.0
                  AND classification_summary IS NOT NULL
                  AND (event_type IS NULL OR event_type != 'noise')
                ORDER BY importance_score DESC
                LIMIT 10
                """,
                (f"%{sector}%",)
            ).fetchall()

        if not rows:
            return None

        context = "\n".join(f"- {r['headline']}: {r['classification_summary']}" for r in rows)

        prompt = (
            f"You are a Professional Wall Street analyst generating a sector outlook.\n\n"
            f"Sector: {sector}\n\n"
            f"Recent high-impact news for this sector:\n{context}\n\n"
            f"Generate a concise sector outlook with 3 scenarios:\n"
            f"1. Bull case: catalysts and growth drivers\n"
            f"2. Base case: most likely trajectory over the next 1-3 months\n"
            f"3. Bear case: risks and headwinds\n\n"
            f"Format as JSON:\n"
            "{\n"
            '  "scenarios": [\n'
            '    {"label": "...", "time_horizon": "1m|3m", '
            '"narrative": "...", "key_drivers": ["..."], "confidence": 0.0-1.0}\n'
            "  ]\n"
            "}\n\n"
            "Output ONLY valid JSON. No markdown, no backticks."
        )

        try:
            with track_llm(self.db, settings.model_trend_outlook, "sector_outlook") as u:
                u.response = response = await complete(
                    model=settings.model_trend_outlook,
                    prompt=prompt,
                    json_mode=True,
                    reasoning="low",
                )
            raw = strip_code_fence(response.text)

            parsed = json.loads(raw)
            scenarios = parsed.get("scenarios", [])
            if not scenarios:
                return None

            # Build a combined narrative with labeled scenarios for backward compat
            combined_narrative = "\n\n".join(
                f"{s.get('label', 'Scenario')}:\n{s.get('narrative', '')}"
                for s in scenarios
            )
            all_drivers = list(dict.fromkeys(
                d for s in scenarios for d in s.get("key_drivers", [])
            ))

            # Use base case (middle scenario) for summary metadata, fall back to first
            base_idx = min(1, len(scenarios) - 1)  # prefer index 1 (base case) if exists
            base_scenario = scenarios[base_idx]

            return {
                "ticker": None,
                "sector": sector,
                "forecast_type": "sector_outlook",
                "scenario_label": base_scenario.get("label", ""),
                "time_horizon": base_scenario.get("time_horizon", "1m"),
                "confidence": base_scenario.get("confidence", 0.5),
                "narrative": combined_narrative,
                "key_drivers": all_drivers,
                "supporting_evidence": context,
                "scenarios": scenarios,  # full scenario list for structured display
            }
        except Exception as e:
            log.warning("trend_forecaster.sector_outlook_failed", sector=sector, error=str(e))
            return None

    async def generate_macro_themes(self) -> list[dict]:
        """Identify 3-5 macro themes from recent high-importance news."""
        # Get top 15 most important articles from last 24h
        with self.db.connection() as conn:
            rows = conn.execute(
                """
                SELECT headline, classification_summary, event_type, affected_sectors
                FROM articles
                WHERE importance_score IS NOT NULL
                  AND published_at >= datetime('now', '-48 hours')
                  AND (event_type IS NULL OR event_type != 'noise')
                ORDER BY importance_score DESC
                LIMIT 15
                """,
            ).fetchall()

        if not rows:
            return []

        context = "\n".join(
            f"- [{r['event_type'] or 'general'}] {r['headline']}: {r['classification_summary'] or ''}"
            for r in rows
        )

        if not is_llm_configured() or not settings.model_trend_outlook:
            return []

        prompt = (
            "You are a Professional Wall Street analyst identifying macro themes.\n\n"
            f"Here are the most important financial news items from the last 48 hours:\n\n"
            f"{context}\n\n"
            "Identify 3-5 overarching macro themes connecting these events. "
            "For each theme:\n"
            "- A concise theme name\n"
            "- 2-3 sentence explanation\n"
            "- Which tickers/sectors are most impacted\n\n"
            "Format as JSON array:\n"
            '[\n'
            '  {\n'
            '    "title": "Theme name",\n'
            '    "explanation": "2-3 sentences",\n'
            '    "impacted_sectors": ["sector1", "sector2"],\n'
            '    "impacted_tickers": ["TICKER1", "TICKER2"],\n'
            '    "confidence": 0.0-1.0\n'
            '  }\n'
            ']\n\n'
            "Output ONLY valid JSON. No markdown."
        )

        try:
            with track_llm(self.db, settings.model_trend_outlook, "macro_themes") as u:
                u.response = response = await complete(
                    model=settings.model_trend_outlook,
                    prompt=prompt,
                    json_mode=True,
                    reasoning="low",
                )
            raw = strip_code_fence(response.text)

            themes = json.loads(raw)
            if isinstance(themes, list):
                return themes
        except Exception as e:
            log.warning("trend_forecaster.macro_themes_failed", error=str(e))

        return []

    def _store_forecast(self, forecast: dict) -> None:
        """Store a trend forecast in the database."""
        # Set expiry: 7 days for short-horizon, 30 days for longer
        horizon = forecast.get("time_horizon", "1m")
        expiry_days = 7 if horizon == "1w" else (14 if horizon == "1m" else 30)
        expires_at = (datetime.now(timezone.utc) + timedelta(days=expiry_days)).isoformat()

        with self.db.connection() as conn:
            # Deactivate old forecasts for the same ticker/sector
            if forecast.get("ticker"):
                conn.execute(
                    "UPDATE trend_forecasts SET is_active = 0 WHERE ticker = ?",
                    (forecast["ticker"],)
                )
            if forecast.get("sector"):
                conn.execute(
                    "UPDATE trend_forecasts SET is_active = 0 WHERE sector = ?",
                    (forecast["sector"],)
                )

            conn.execute(
                """
                INSERT INTO trend_forecasts
                    (ticker, sector, forecast_type, scenario_label, time_horizon,
                     confidence, narrative, key_drivers_json, supporting_evidence, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    forecast.get("ticker"),
                    forecast.get("sector"),
                    forecast.get("forecast_type", "trend_projection"),
                    forecast.get("scenario_label", ""),
                    forecast.get("time_horizon", "1m"),
                    forecast.get("confidence", 0.0),
                    forecast.get("narrative", ""),
                    json.dumps(forecast.get("key_drivers", [])),
                    forecast.get("supporting_evidence", ""),
                    expires_at,
                )
            )

    def get_active_forecasts(self, ticker: str = None, sector: str = None, limit: int = 20) -> list[dict]:
        """Get active trend forecasts, optionally filtered by ticker or sector."""
        with self.db.connection() as conn:
            if ticker:
                rows = conn.execute(
                    """
                    SELECT * FROM trend_forecasts
                    WHERE is_active = 1 AND ticker = ? AND (expires_at IS NULL OR expires_at >= datetime('now'))
                    ORDER BY generated_at DESC
                    """,
                    (ticker,)
                ).fetchall()
            elif sector:
                rows = conn.execute(
                    """
                    SELECT * FROM trend_forecasts
                    WHERE is_active = 1 AND sector = ? AND (expires_at IS NULL OR expires_at >= datetime('now'))
                    ORDER BY generated_at DESC
                    """,
                    (sector,)
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM trend_forecasts
                    WHERE is_active = 1 AND (expires_at IS NULL OR expires_at >= datetime('now'))
                    ORDER BY generated_at DESC LIMIT ?
                    """,
                    (limit,)
                ).fetchall()
            return [dict(row) for row in rows]

    async def generate_and_cache_macro_themes(self) -> list[dict]:
        """Generate macro themes and update the in-memory cache. Called by scheduler and API."""
        themes = await self.generate_macro_themes()
        if themes:
            _macro_themes_cache["data"] = themes
            _macro_themes_cache["timestamp"] = time.time()
            log.info("trend_forecaster.macro_themes_cached", count=len(themes))
            # Publish via SSE for real-time dashboard updates
            try:
                await event_bus.publish("macro_themes", themes)
            except Exception as e:
                log.warning("trend_forecaster.macro_themes_sse_failed", error=str(e))
        return themes

    @staticmethod
    def get_cached_macro_themes() -> list[dict] | None:
        """Return cached macro themes if within TTL. Returns None if cache is cold."""
        if _macro_themes_cache["data"] and (time.time() - _macro_themes_cache["timestamp"]) < _MACRO_THEMES_TTL:
            return _macro_themes_cache["data"]
        return None
