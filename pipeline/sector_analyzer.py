"""
Sector Analyzer Component

Aggregates ticker_mentions into sector-level sentiment snapshots,
detects sector rotations, and auto-discovers hot tickers not on the
user's watchlist.

Runs as a periodic background job (every 15 min).
"""

from __future__ import annotations

import json
import asyncio
import time
from datetime import datetime, timezone, timedelta

from config.logging_config import get_logger
from config.settings import settings
from config.llm import get_client, DEFAULT_SAFETY_SETTINGS
from data.database import Database
from api.sse_manager import event_bus
from google.genai import types

log = get_logger(__name__)


class SectorAnalyzer:
    """
    Aggregates sector-level sentiment from articles and ticker_mentions,
    detects rotations, and surfaces emerging tickers/sectors.
    """

    def __init__(self, db: Database, alert_manager=None):
        self.db = db
        self.alert_manager = alert_manager

    async def run_analysis(self) -> dict:
        """
        Main entry point: compute sector snapshots, detect shifts,
        and discover hot tickers. Called every 15 min by the scheduler.
        Returns a summary dict of what was found.
        """
        results = {}

        # 1. Compute current sector snapshot
        sector_data = self._compute_sector_snapshot(hours=6)
        results["sectors_analyzed"] = len(sector_data)
        if sector_data:
            self._store_sector_snapshot(sector_data)

        # 2. Detect sector rotations
        rotations = self._detect_sector_rotations(sector_data)
        results["rotations_detected"] = len(rotations)
        for rot in rotations:
            self.db.insert_rotation_signal(rot)

        # 3. Discover hot tickers not on user watchlist
        hot_tickers = self._discover_hot_tickers(hours=6)
        results["hot_tickers_found"] = len(hot_tickers)
        if hot_tickers:
            # Generate LLM rationales for why these tickers are surging
            hot_tickers = await self.generate_hot_ticker_rationales(hot_tickers)
        for ht in hot_tickers:
            # upsert_hot_ticker now stores rationale along with other fields
            self.db.upsert_hot_ticker(ht)

        if rotations or hot_tickers:
            log.info("sector_analyzer.results",
                     sectors=len(sector_data),
                     rotations=len(rotations),
                     hot_tickers=len(hot_tickers))

        # Publish results to SSE event bus for real-time dashboard
        try:
            if sector_data:
                await event_bus.publish("sector_heatmap", sector_data)
            for rot in rotations:
                await event_bus.publish("rotation_signal", rot)
            if hot_tickers:
                await event_bus.publish("hot_tickers", hot_tickers)
        except Exception as e:
            log.warning("sector_analyzer.sse_publish_failed", error=str(e))

        return results

    def _compute_sector_snapshot(self, hours: int = 6) -> list[dict]:
        """
        Compute current sector sentiment from ticker_mentions + articles.
        Returns list of {sector, avg_sentiment, article_count,
                         bullish_count, bearish_count, neutral_count,
                         avg_importance, top_tickers, sentiment_momentum}
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        sector_map = {}

        with self.db.connection() as conn:
            # Get articles within time window that have affected_sectors
            rows = conn.execute(
                """
                SELECT a.affected_sectors, a.sentiment_score, a.importance_score,
                       a.affected_tickers
                FROM articles a
                WHERE a.published_at >= ?
                  AND a.affected_sectors IS NOT NULL
                  AND a.sentiment_score IS NOT NULL
                  AND (a.event_type IS NULL OR a.event_type != 'noise')
                """,
                (cutoff,)
            ).fetchall()

        for row in rows:
            try:
                sectors = json.loads(row["affected_sectors"])
                tickers = json.loads(row["affected_tickers"]) if row["affected_tickers"] else []
                sentiment = row["sentiment_score"] or 0.0
                importance = row["importance_score"] or 0.0
            except (json.JSONDecodeError, TypeError):
                continue

            for sector in sectors:
                if sector not in sector_map:
                    sector_map[sector] = {
                        "sector": sector,
                        "sentiments": [],
                        "importance_sum": 0.0,
                        "article_count": 0,
                        "ticker_counts": {},
                    }
                s = sector_map[sector]
                s["sentiments"].append(sentiment)
                s["importance_sum"] += importance
                s["article_count"] += 1
                for t in tickers:
                    s["ticker_counts"][t] = s["ticker_counts"].get(t, 0) + 1

        results = []
        for sector, data in sector_map.items():
            total = len(data["sentiments"])
            if total == 0:
                continue
            avg_sent = sum(data["sentiments"]) / total
            bullish = sum(1 for s in data["sentiments"] if s > 0.15)
            bearish = sum(1 for s in data["sentiments"] if s < -0.15)
            neutral = total - bullish - bearish
            avg_imp = data["importance_sum"] / total

            # Top 5 tickers by mention count
            top_tickers = sorted(
                data["ticker_counts"].items(), key=lambda x: x[1], reverse=True
            )[:5]
            top_tickers_list = [{"ticker": t, "count": c} for t, c in top_tickers]

            results.append({
                "sector": sector,
                "avg_sentiment": round(avg_sent, 3),
                "article_count": total,
                "bullish_count": bullish,
                "bearish_count": bearish,
                "neutral_count": neutral,
                "avg_importance": round(avg_imp, 2),
                "top_tickers": top_tickers_list,
                "sentiment_momentum": 0.0,  # computed by _compute_momentum
            })

        results.sort(key=lambda x: x["article_count"], reverse=True)
        return results

    def _compute_momentum(self, sector: str, current_sentiment: float) -> float:
        """Compare current sector sentiment against the 48h rolling average."""
        try:
            with self.db.connection() as conn:
                row = conn.execute(
                    """
                    SELECT avg_sentiment FROM sector_sentiment_snapshots
                    WHERE sector = ?
                    ORDER BY snapshot_time DESC LIMIT 1 OFFSET 3
                    """,
                    (sector,)
                ).fetchone()
                if row and row["avg_sentiment"] is not None:
                    return current_sentiment - row["avg_sentiment"]
        except Exception:
            pass
        return 0.0

    def _store_sector_snapshot(self, sector_data: list[dict]) -> None:
        """Store computed sector snapshots to the database."""
        for sd in sector_data:
            momentum = self._compute_momentum(sd["sector"], sd["avg_sentiment"])
            sd["sentiment_momentum"] = round(momentum, 3)
            with self.db.connection() as conn:
                conn.execute(
                    """
                    INSERT INTO sector_sentiment_snapshots
                        (sector, avg_sentiment, article_count, bullish_count,
                         bearish_count, neutral_count, top_tickers_json, sentiment_momentum)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        sd["sector"],
                        sd["avg_sentiment"],
                        sd["article_count"],
                        sd["bullish_count"],
                        sd["bearish_count"],
                        sd["neutral_count"],
                        json.dumps(sd["top_tickers"]),
                        sd["sentiment_momentum"],
                    )
                )

    def capture_daily_snapshot(self) -> None:
        """Capture a daily sector snapshot for historical tracking (runs at midnight)."""
        sector_data = self._compute_sector_snapshot(hours=24)
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        for sd in sector_data:
            with self.db.connection() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO sector_daily_snapshot
                        (sector, date, mention_count, avg_sentiment,
                         avg_importance, bullish_ratio, top_tickers)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        sd["sector"],
                        today_str,
                        sd["article_count"],
                        sd["avg_sentiment"],
                        sd["avg_importance"],
                        sd["bullish_count"] / max(sd["article_count"], 1),
                        json.dumps([t["ticker"] for t in sd["top_tickers"]]),
                    )
                )

    def _detect_sector_rotations(self, current_data: list[dict]) -> list[dict]:
        """
        Compare current sector rankings vs 48h baseline.
        Returns rotation signals when sectors have >0.3 sentiment shift and
        >2x article volume vs baseline.
        """
        signals = []
        for sd in current_data:
            if abs(sd["sentiment_momentum"]) < 0.3:
                continue

            # Check if volume is significantly above baseline
            with self.db.connection() as conn:
                baseline = conn.execute(
                    """
                    SELECT AVG(article_count) as avg_count
                    FROM sector_sentiment_snapshots
                    WHERE sector = ?
                      AND snapshot_time >= datetime('now', '-48 hours')
                      AND snapshot_time < datetime('now', '-6 hours')
                    """,
                    (sd["sector"],)
                ).fetchone()

            baseline_count = baseline["avg_count"] if baseline and baseline["avg_count"] else 1
            volume_ratio = sd["article_count"] / max(baseline_count, 1)

            if volume_ratio < 1.5:
                continue

            direction = "bullish" if sd["sentiment_momentum"] > 0 else "bearish"
            signals.append({
                "from_sector": "Market_Neutral",  # placeholder — compared to overall market
                "to_sector": sd["sector"],
                "signal_strength": round(min(abs(sd["sentiment_momentum"]), 1.0), 2),
                "reasoning": f"{sd['sector']} sentiment shifted {direction} by "
                           f"{abs(sd['sentiment_momentum']):.2f} points with "
                           f"{sd['article_count']} articles ({volume_ratio:.1f}x baseline volume).",
                "triggered_by": "sentiment_shift",
                "is_active": 1,
            })

        return signals

    def _discover_hot_tickers(self, hours: int = 6) -> list[dict]:
        """
        Find tickers with surging mentions that are NOT on the user's watchlist.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        watchlist = set(self.db.get_tracked_tickers())

        with self.db.connection() as conn:
            # Find hot tickers by mention count (exclude noise articles)
            rows = conn.execute(
                """
                SELECT tm.ticker,
                       COUNT(*) as mention_count,
                       AVG(tm.sentiment_score) as avg_sentiment
                FROM ticker_mentions tm
                JOIN articles a ON a.id = tm.article_id
                WHERE tm.mentioned_at >= ?
                  AND (a.event_type IS NULL OR a.event_type != 'noise')
                GROUP BY tm.ticker
                HAVING mention_count >= 3
                ORDER BY mention_count DESC
                LIMIT 20
                """,
                (cutoff,)
            ).fetchall()

        results = []
        for row in rows:
            ticker = row["ticker"]
            if ticker in watchlist:
                continue

            # Fetch distinct sectors for this ticker from recent articles
            sectors = []
            with self.db.connection() as conn:
                sector_rows = conn.execute(
                    """
                    SELECT DISTINCT a.affected_sectors
                    FROM articles a
                    JOIN ticker_mentions tm ON a.id = tm.article_id
                    WHERE tm.ticker = ?
                      AND a.affected_sectors IS NOT NULL
                      AND tm.mentioned_at >= ?
                    LIMIT 10
                    """,
                    (ticker, cutoff)
                ).fetchall()
                for sr in sector_rows:
                    try:
                        parsed = json.loads(sr["affected_sectors"])
                        if isinstance(parsed, list):
                            for s in parsed:
                                if s not in sectors:
                                    sectors.append(s)
                    except (json.JSONDecodeError, TypeError):
                        continue

            results.append({
                "ticker": ticker,
                "mention_count": row["mention_count"],
                "avg_sentiment": round(row["avg_sentiment"] or 0.0, 3),
                "sectors": sectors[:3],
            })

        return results

    async def generate_hot_ticker_rationales(self, hot_tickers: list[dict]) -> list[dict]:
        """
        Batch LLM call to generate one-sentence "why this is moving" for hot tickers.
        """
        if not hot_tickers:
            return hot_tickers

        client = get_client()
        if not client:
            return hot_tickers

        ticker_list = ", ".join([h["ticker"] for h in hot_tickers])
        prompt = (
            f"You are a Professional Wall Street analyst. "
            f"The following tickers are surging in news mentions today:\n"
            f"{ticker_list}\n\n"
            f"For EACH ticker, write exactly ONE sentence explaining why it might be trending "
            f"based solely on recent market events. Base this on your general knowledge.\n"
            f"Format your response as a valid JSON object where keys are tickers and values "
            f"are the one-sentence explanations. Do NOT use markdown or HTML.\n"
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

            rationales = json.loads(raw.strip())
            for ht in hot_tickers:
                ht["rationale"] = rationales.get(ht["ticker"], "Surge in news mentions detected.")
        except Exception as e:
            log.warning("sector_analyzer.rationale_failed", error=str(e))
            for ht in hot_tickers:
                ht["rationale"] = "Surge in news mentions detected."

        return hot_tickers
