"""
Events Tracker Component

Tracks upcoming earnings dates, product launches, FDA decisions,
investor days, conferences, and other major events for tickers.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from typing import Optional

from config.logging_config import get_logger
from config.llm import get_deepseek_client
from config.settings import settings
from data.database import Database
from data.models import NewsArticle
from api.sse_manager import event_bus
from pipeline.date_utils import normalize_date

log = get_logger(__name__)

EVENT_KEYWORDS = [
    "earnings", "product launch", "investor day", "fda decision", "fda approval",
    "conference", "dividend", "stock split", "share buyback", "acquisition",
    "merger", "annual meeting", "keynote", "product event", "unveil", "release date",
    "clinical trial", "phase 3", "regulatory decision", "sec filing"
]


class EventTracker:
    """Tracks upcoming earnings dates, product launches, and major events for tickers."""

    def __init__(self, db: Database, alert_manager=None):
        self.db = db
        self.alert_manager = alert_manager

    async def scan_upcoming_events(self) -> list[dict]:
        """
        Main entry point: scan earnings calendar + recent news for upcoming events.
        Returns list of newly detected events.
        """
        detected = []

        # 1. Check Finnhub earnings calendar for tracked tickers (30-day forward look)
        earnings = await self._scan_earnings_calendar()
        detected.extend(earnings)

        # 2. Scan recent high-importance articles for event mentions
        news_events = await self._scan_news_for_events()
        detected.extend(news_events)

        if detected:
            log.info("event_tracker.found", count=len(detected))
            # Publish to SSE event bus for real-time dashboard
            try:
                await event_bus.publish("events_updated", {
                    "count": len(detected),
                    "events": detected[:10],
                })
            except Exception as e:
                log.warning("event_tracker.sse_publish_failed", error=str(e))

        return detected

    async def _scan_earnings_calendar(self) -> list[dict]:
        """Query Finnhub earnings calendar for tracked tickers (30 days ahead)."""
        import os
        api_key = os.getenv("FINNHUB_API_KEY")
        if not api_key:
            return []

        import httpx
        tracked = self.db.get_tracked_tickers()
        if not tracked:
            return []

        today = datetime.now()
        end_date = (today + timedelta(days=30)).strftime("%Y-%m-%d")
        detected = []

        async with httpx.AsyncClient(timeout=10) as client:
            for ticker in tracked:
                try:
                    url = f"https://finnhub.io/api/v1/calendar/earnings?from={today.strftime('%Y-%m-%d')}&to={end_date}&symbol={ticker}&token={api_key}"
                    resp = await client.get(url)
                    resp.raise_for_status()
                    data = resp.json()

                    earnings_list = data.get("earningsCalendar", [])
                    for e in earnings_list:
                        event_date = normalize_date(e.get("date") or "")
                        if not event_date:
                            continue

                        # Cross-source dedup: check ticker + date + event_type regardless of source
                        with self.db.connection() as conn:
                            existing = conn.execute(
                                "SELECT id, source, confidence FROM ticker_events WHERE ticker = ? AND event_type = 'earnings' AND event_date = ?",
                                (ticker, event_date)
                            ).fetchone()

                        if existing:
                            # Upgrade confidence: Finnhub confirmed data beats LLM estimated/rumored
                            if existing["source"] == "llm_extracted" and existing.get("confidence") != "confirmed":
                                with self.db.connection() as conn:
                                    conn.execute(
                                        "UPDATE ticker_events SET source = 'finnhub', confidence = 'confirmed', notes = ? WHERE id = ?",
                                        (f"Upgraded from {existing['source']} to finnhub confirmed", existing["id"])
                                    )
                                log.info("event_tracker.dedup_upgraded", ticker=ticker, date=event_date)
                            continue

                        with self.db.connection() as conn:
                            conn.execute(
                                """
                                INSERT INTO ticker_events
                                    (ticker, event_type, event_date, event_title, confidence, source)
                                VALUES (?, 'earnings', ?, ?, 'confirmed', 'finnhub')
                                """,
                                (ticker, event_date, f"{ticker} Earnings Report")
                            )
                        detected.append({
                            "ticker": ticker,
                            "event_type": "earnings",
                            "event_date": event_date,
                            "event_title": f"{ticker} Earnings Report",
                            "confidence": "confirmed",
                            "source": "finnhub",
                        })
                except Exception as e:
                    log.warning("event_tracker.earnings_failed", ticker=ticker, error=str(e))

        return detected

    async def _scan_news_for_events(self) -> list[dict]:
        """
        Scan recent high-importance articles for event mentions (product launches,
        FDA decisions, conferences, etc.) using DeepSeek extraction.
        """
        with self.db.connection() as conn:
            rows = conn.execute(
                """
                SELECT a.* FROM articles a
                LEFT JOIN ticker_events te ON te.source_article_id = a.id
                WHERE te.id IS NULL
                  AND a.importance_score >= 5.0
                  AND a.published_at >= datetime('now', '-72 hours')
                  AND (a.event_type IS NULL OR a.event_type != 'noise')
                ORDER BY a.importance_score DESC
                LIMIT 20
                """
            ).fetchall()

        if not rows:
            return []

        client = get_deepseek_client()
        if not client:
            return []

        detected = []
        for row in rows:
            article = self.db.row_to_article(row)
            text = f"{article.headline} {article.summary or ''}".lower()
            if not any(kw in text for kw in EVENT_KEYWORDS):
                continue

            result = await self._classify_event_article(article)
            if result:
                # Skip if no valid date was extracted
                if not result.get("event_date"):
                    log.info("event_tracker.skipped_no_date", ticker=result.get("ticker"), title=article.headline[:60])
                    continue

                result["source_article_id"] = article.id

                # Cross-source dedup: check if this exact event already exists
                with self.db.connection() as conn:
                    existing = conn.execute(
                        """SELECT id FROM ticker_events
                           WHERE ticker = ? AND event_date = ? AND event_type = ?""",
                        (result["ticker"], result["event_date"], result["event_type"])
                    ).fetchone()

                if existing:
                    log.info("event_tracker.dedup_skipped",
                             ticker=result["ticker"], event_type=result["event_type"],
                             event_date=result["event_date"])
                    continue

                with self.db.connection() as conn:
                    conn.execute(
                        """
                        INSERT INTO ticker_events
                            (ticker, event_type, event_date, event_title, confidence, source, source_article_id, notes)
                        VALUES (?, ?, ?, ?, ?, 'llm_extracted', ?, ?)
                        """,
                        (
                            result["ticker"], result["event_type"],
                            result["event_date"], result["event_title"],
                            result.get("confidence", "estimated"),
                            article.id, result.get("notes", ""),
                        )
                    )
                detected.append(result)

        return detected

    async def _classify_event_article(self, article: NewsArticle) -> Optional[dict]:
        """Use DeepSeek to extract event details from an article."""
        client = get_deepseek_client()
        if not client:
            return None

        prompt = (
            "Extract upcoming financial event information from this news article. "
            "Respond ONLY with a valid JSON object. No markdown, no backticks.\n\n"
            f"Headline: {article.headline}\n"
            f"Summary: {article.summary or ''}\n\n"
            "JSON Schema:\n"
            "{\n"
            '  "ticker": "main ticker symbol mentioned (string or null)",\n'
            '  "event_type": "earnings | product_launch | investor_day | fda_decision | conference | dividend | split | acquisition | other",\n'
            '  "event_date": "ISO date string if mentioned, or null",\n'
            '  "event_title": "short title (string)",\n'
            '  "confidence": "confirmed | estimated | rumored",\n'
            '  "notes": "any additional context (string)"\n'
            "}\n\n"
            "Set fields to null if not mentioned."
        )

        try:
            response = await client.chat.completions.create(
                model=settings.deepseek_model_classifier,
                messages=[
                    {"role": "system", "content": "You extract financial event details from news. Output JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content.strip())

            if result.get("event_type") and result.get("event_type") != "other":
                return {
                    "ticker": result.get("ticker") or "",
                    "event_type": result["event_type"],
                    "event_date": normalize_date(result.get("event_date") or ""),
                    "event_title": result.get("event_title", ""),
                    "confidence": result.get("confidence", "estimated"),
                    "notes": result.get("notes", ""),
                }
        except Exception as e:
            log.warning("event_tracker.classify_failed", error=str(e), headline=article.headline[:80])

        return None

    def get_ticker_events(self, ticker: str, days_ahead: int = 30) -> list[dict]:
        """Get all upcoming events for a specific ticker within N days."""
        end_date = (datetime.now(timezone.utc) + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        with self.db.connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM ticker_events
                WHERE ticker = ? AND event_date <= ?
                ORDER BY event_date ASC
                """,
                (ticker, end_date)
            ).fetchall()
            return [dict(row) for row in rows]

    def get_tracked_events_calendar(self, days_ahead: int = 14) -> list[dict]:
        """Get all upcoming events for all tracked tickers, sorted by date."""
        tracked = self.db.get_tracked_tickers()
        if not tracked:
            return []

        end_date = (datetime.now(timezone.utc) + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        placeholders = ",".join("?" for _ in tracked)

        with self.db.connection() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM ticker_events
                WHERE ticker IN ({placeholders}) AND event_date <= ?
                ORDER BY event_date ASC
                """,
                tracked + [end_date]
            ).fetchall()
            return [dict(row) for row in rows]

    def get_all_upcoming_events(self, days_ahead: int = 14) -> list[dict]:
        """Get all upcoming events (not limited to tracked tickers)."""
        end_date = (datetime.now(timezone.utc) + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        with self.db.connection() as conn:
            rows = conn.execute(
                """
                SELECT te.*, ti.sector
                FROM ticker_events te
                LEFT JOIN ticker_info ti ON ti.ticker = te.ticker
                WHERE te.event_date <= ?
                ORDER BY te.event_date ASC, te.confidence DESC
                """,
                (end_date,)
            ).fetchall()
            return [dict(row) for row in rows]
