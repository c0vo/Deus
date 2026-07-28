"""
Trend Forecaster Component

LLM-powered forward-looking analysis of current market trends.
Generates "If this trend continues..." scenario analysis for
trending tickers and sectors.

Runs every 4 hours as a background job.
"""

from __future__ import annotations

import json
import time
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Optional

from config.logging_config import get_logger
from config.llm import get_client, get_deepseek_client, DEFAULT_SAFETY_SETTINGS
from config.settings import settings
from data.database import Database
from api.sse_manager import event_bus
from google.genai import types

log = get_logger(__name__)

# Module-level in-memory cache for macro themes (shared across all TrendForecaster instances)
_macro_themes_cache: dict = {"data": None, "timestamp": 0}
_MACRO_THEMES_TTL = 14400  # 4 hours (matches scheduler cadence)


class TrendForecaster:
    """LLM-based forward-looking analysis of current market trends."""

    def __init__(self, db: Database):
        self.db = db

    async def generate_forecasts(self, max_tickers: int = 5, max_sectors: int = 3) -> list[dict]:
        """
        Generate trend forecasts for top trending tickers and sectors.
        Called every 4 hours by the scheduler.
        """
        all_forecasts = []

        # 1. Get top trending tickers from last 24h
        trending = self.db.get_top_trending_tickers(hours=24, limit=max_tickers)
        ticker_contexts = {}
        for t in trending:
            ticker = t["ticker"]
            summaries = self.db.get_recent_summaries_for_ticker(ticker, hours=48)
            if summaries:
                ticker_contexts[ticker] = summaries

        if ticker_contexts:
            forecasts = await self._batch_generate_ticker_forecasts(ticker_contexts)
            for f in forecasts:
                self._store_forecast(f)
            all_forecasts.extend(forecasts)

        # 2. Generate sector-level outlooks
        sector_data = self._get_top_sectors(hours=24, limit=max_sectors)
        for sd in sector_data:
            forecast = await self._generate_sector_outlook(sd["sector"])
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

    async def _batch_generate_ticker_forecasts(self, ticker_contexts: dict) -> list[dict]:
        """
        Batch LLM call to generate forecasts for multiple tickers at once.
        Uses DeepSeek v4-pro for reasoning-heavy scenario generation.
        """
        client = get_deepseek_client()
        if not client:
            return []

        context_text = ""
        for ticker, summaries in ticker_contexts.items():
            context_text += f"\nTicker: {ticker}\n"
            context_text += "Recent context:\n" + "\n".join(f"- {s}" for s in summaries) + "\n"

        prompt = (
            "You are a professional Wall Street analyst generating forward-looking scenario analyses. "
            "Base every scenario on specific data points from the provided context.\n\n"
            f"{context_text}\n"
            "For EACH ticker, generate 3 scenarios with confidence estimates:\n"
            "1. Bull case (label: 'If [catalyst] materializes...'): The upside scenario with specific triggers\n"
            "2. Base case (label: 'Most likely: ...'): Your central estimate, weighted by evidence strength\n"
            "3. Bear case (label: 'If [risk] plays out...'): The downside scenario with specific triggers\n\n"
            "Confidence calibration: 0.3-0.5 = speculative, 0.5-0.7 = plausible with some evidence, "
            "0.7-0.85 = well-supported by context, 0.85+ = overwhelming evidence (rare).\n\n"
            "Format as valid JSON with ticker symbols as keys. Each value:\n"
            '{"scenarios": [{"label": "...", "time_horizon": "1w|1m|3m", '
            '"narrative": "2-3 sentence analysis citing specific context", '
            '"key_drivers": ["driver1", "driver2"], "confidence": 0.0-1.0}]}\n\n'
            "Do NOT use markdown. Output valid JSON only."
        )

        try:
            response = await client.chat.completions.create(
                model=settings.deepseek_model_reasoner,
                messages=[
                    {"role": "system", "content": "You generate forward-looking scenario analyses for stocks. Output valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"},
                reasoning_effort="high",
                extra_body={"thinking": {"type": "enabled"}},
            )

            raw = response.choices[0].message.content.strip()
            if raw.startswith("```json"): raw = raw[7:]
            if raw.startswith("```"): raw = raw[3:]
            if raw.endswith("```"): raw = raw[:-3]

            parsed = json.loads(raw.strip())
            forecasts = []
            for ticker, data in parsed.items():
                scenarios = data.get("scenarios", [])
                for sc in scenarios:
                    forecasts.append({
                        "ticker": ticker,
                        "sector": None,
                        "forecast_type": "trend_projection",
                        "scenario_label": sc.get("label", ""),
                        "time_horizon": sc.get("time_horizon", "1m"),
                        "confidence": sc.get("confidence", 0.5),
                        "narrative": sc.get("narrative", ""),
                        "key_drivers": sc.get("key_drivers", []),
                        "supporting_evidence": context_text[:500],
                    })
            return forecasts
        except Exception as e:
            log.warning("trend_forecaster.batch_failed", error=str(e))
            return []

    async def _generate_sector_outlook(self, sector: str) -> Optional[dict]:
        """Generate a forward-looking outlook for a specific sector."""
        client = get_client()
        if not client:
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
            loop = asyncio.get_running_loop()

            def ask_llm():
                response = client.models.generate_content(
                    model=settings.gemini_model_chat,
                    contents=prompt,
                    config={
                        "safety_settings": DEFAULT_SAFETY_SETTINGS,
                        "response_mime_type": "application/json",
                        "thinking_config": types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
                    }
                )
                return response.text.strip()

            raw = await loop.run_in_executor(None, ask_llm)
            if raw.startswith("```json"): raw = raw[7:]
            if raw.startswith("```"): raw = raw[3:]
            if raw.endswith("```"): raw = raw[:-3]

            parsed = json.loads(raw.strip())
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

        client = get_client()
        if not client:
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
            loop = asyncio.get_running_loop()

            def ask_llm():
                response = client.models.generate_content(
                    model=settings.gemini_model_chat,
                    contents=prompt,
                    config={
                        "safety_settings": DEFAULT_SAFETY_SETTINGS,
                        "response_mime_type": "application/json",
                        "thinking_config": types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
                    }
                )
                return response.text.strip()

            raw = await loop.run_in_executor(None, ask_llm)
            if raw.startswith("```json"): raw = raw[7:]
            if raw.startswith("```"): raw = raw[3:]
            if raw.endswith("```"): raw = raw[:-3]

            themes = json.loads(raw.strip())
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
