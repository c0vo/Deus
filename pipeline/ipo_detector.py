"""
IPO Detector Component

Detects IPO mentions in the news feed and tracks upcoming/public offerings.
Runs after each pipeline classification pass.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone, timedelta
from typing import Optional
from config.logging_config import get_logger
from config.settings import settings
from config.llm import get_deepseek_client
from data.database import Database
from data.models import NewsArticle
from api.sse_manager import event_bus
from pipeline.date_utils import normalize_date, normalize_company_name

log = get_logger(__name__)

# Phrases that only ever describe a company coming to market. A headline
# containing one of these is enough on its own.
IPO_STRONG_KEYWORDS = [
    "ipo", "initial public offering", "going public", "public listing",
    "direct listing", "s-1 filing", "f-1 filing", "prices its offering",
    "stock market debut", "market debut", "trading debut",
]

# Phrases that describe listing events but are far too common in general
# business copy to trigger on their own — "debut" alone matched a Lockheed
# Martin story about a fighter jet. These only count alongside a strong term
# or an explicit new-issue word.
IPO_WEAK_KEYWORDS = [
    "debut", "listed on", "begins trading", "publicly traded", "spac",
    "offering price", "share sale",
]

IPO_CONTEXT_KEYWORDS = [
    "ipo", "offering", "listing", "flotation", "shares priced", "nasdaq debut",
    "nyse debut", "underwriter", "bookrunner", "prospectus",
]


class IPODetector:
    """Detects IPO mentions in news and tracks upcoming/public offerings."""

    def __init__(self, db: Database):
        self.db = db

    async def scan_for_ipos(self) -> list[dict]:
        """
        Search recent unclassified articles for IPO-related content.
        Returns list of newly detected IPO events.
        """
        with self.db.connection() as conn:
            rows = conn.execute(
                """
                SELECT a.* FROM articles a
                LEFT JOIN ipo_tracker ip ON ip.source_article_id = a.id
                WHERE ip.id IS NULL
                  AND a.published_at >= datetime('now', '-48 hours')
                  AND (a.event_type IS NULL OR a.event_type != 'noise')
                ORDER BY a.published_at DESC
                LIMIT 30
                """
            ).fetchall()

        if not rows:
            return []

        detected = []
        for row in rows:
            article = self.db.row_to_article(row)
            if self._is_ipo_related(article):
                result = await self._classify_ipo_article(article)
                if result:
                    result["source_article_id"] = article.id
                    self._store_ipo(result)
                    detected.append(result)

        if detected:
            log.info("ipo_detector.found", count=len(detected))
            # Publish to SSE event bus for real-time dashboard
            try:
                for ipo in detected:
                    await event_bus.publish("ipo_alert", ipo)
            except Exception as e:
                log.warning("ipo_detector.sse_publish_failed", error=str(e))

        return detected

    def _is_ipo_related(self, article: NewsArticle) -> bool:
        """
        Gate before spending an LLM call.

        A strong phrase stands alone. A weak one ("debut", "begins trading")
        only counts when something else in the text confirms we are talking
        about a new issue — otherwise any product launch trips the detector.
        """
        text = f"{article.headline} {article.summary or ''}".lower()

        if any(kw in text for kw in IPO_STRONG_KEYWORDS):
            return True

        if any(kw in text for kw in IPO_WEAK_KEYWORDS):
            return any(kw in text for kw in IPO_CONTEXT_KEYWORDS)

        return False

    def _is_plausible_new_listing(self, data: dict) -> bool:
        """
        Reject extractions that describe an already-public company.

        Two signals: the model itself saying the company was already listed,
        and an IPO date far enough in the past that it cannot be watchlist
        material. Both were letting long-established tickers through.
        """
        if data.get("already_public") is True:
            return False

        raw_date = data.get("expected_date")
        if raw_date:
            try:
                parsed = datetime.fromisoformat(str(raw_date).replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                age_days = (datetime.now(timezone.utc) - parsed).days
                if age_days > settings.ipo_max_backdate_days:
                    return False
            except (ValueError, TypeError):
                pass

        return True

    async def _classify_ipo_article(self, article: NewsArticle) -> Optional[dict]:
        """Use DeepSeek v4-flash to extract IPO details from an article."""
        client = get_deepseek_client()
        if not client:
            return None

        headline = article.headline
        summary = article.summary or ""

        prompt = (
            "Extract IPO information from the following news article. "
            "Respond ONLY with a valid JSON object. No markdown, no backticks.\n\n"
            f"Headline: {headline}\n"
            f"Summary: {summary}\n\n"
            "JSON Schema:\n"
            "{\n"
            '  "company_name": "string or null",\n'
            '  "ticker": "string or null",\n'
            '  "status": "upcoming | priced | listed | withdrawn | rumor",\n'
            '  "expected_price": "string or null",\n'
            '  "expected_date": "string (ISO date) or null",\n'
            '  "sector": "string or null",\n'
            '  "estimated_valuation": "string or null",\n'
            '  "already_public": true or false,\n'
            '  "notes": "string"\n'
            "}\n\n"
            "Set fields to null if not mentioned in the article.\n\n"
            "IMPORTANT: set already_public to true if the company has been "
            "listed on a public exchange for more than a year. An article that "
            "merely uses words like 'debut', 'begins trading' or 'publicly "
            "traded' about an established company is NOT an IPO — in that case "
            "set company_name to null."
        )

        try:
            response = await client.chat.completions.create(
                model=settings.deepseek_model_classifier,
                messages=[
                    {"role": "system", "content": "You extract IPO information from news articles. Output JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content.strip())

            if result.get("company_name") and result.get("company_name") != "null":
                extracted = {
                    "company_name": result["company_name"],
                    "ticker": result.get("ticker"),
                    "status": result.get("status") or "rumored",
                    "expected_price": result.get("expected_price"),
                    "expected_date": normalize_date(result.get("expected_date") or ""),
                    "sector": result.get("sector"),
                    "estimated_valuation": result.get("estimated_valuation"),
                    "already_public": result.get("already_public"),
                    "notes": result.get("notes", ""),
                }
                if not self._is_plausible_new_listing(extracted):
                    log.info(
                        "ipo_detector.rejected_established_company",
                        company=extracted["company_name"],
                        ticker=extracted.get("ticker"),
                        headline=headline[:80],
                    )
                    return None
                extracted.pop("already_public", None)
                return extracted
        except Exception as e:
            log.warning("ipo_detector.classify_failed", error=str(e), headline=headline[:80])

        return None

    def _store_ipo(self, data: dict) -> None:
        """Store or update an IPO record in the database."""
        with self.db.connection() as conn:
            # Fuzzy dedup: check exact match first, then normalized name
            existing = conn.execute(
                "SELECT id, status, company_name FROM ipo_tracker WHERE company_name = ?",
                (data["company_name"],)
            ).fetchone()

            if not existing:
                norm_name = normalize_company_name(data["company_name"])
                if norm_name:
                    all_rows = conn.execute(
                        "SELECT id, company_name, status FROM ipo_tracker"
                    ).fetchall()
                    for row in all_rows:
                        if normalize_company_name(row["company_name"]) == norm_name:
                            existing = row
                            break

            if existing:
                # Update existing record
                conn.execute(
                    """
                    UPDATE ipo_tracker SET ticker = ?, status = ?, ipo_date = ?,
                        offering_price = ?, sector = ?, estimated_valuation = ?,
                        notes = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (
                        data.get("ticker"), data.get("status"),
                        data.get("expected_date"), data.get("expected_price"),
                        data.get("sector"), data.get("estimated_valuation"),
                        data.get("notes", ""), existing["id"],
                    )
                )
            else:
                # Store source article ID
                source_id = data.get("source_article_id")
                conn.execute(
                    """
                    INSERT INTO ipo_tracker
                        (company_name, ticker, status, ipo_date, offering_price,
                         sector, estimated_valuation, source_article_id, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        data["company_name"], data.get("ticker"),
                        data.get("status", "rumored"), data.get("expected_date"),
                        data.get("expected_price"), data.get("sector"),
                        data.get("estimated_valuation"), source_id,
                        data.get("notes", ""),
                    )
                )

    def retire_stale(self, listed_retention_days: int = 30) -> int:
        """
        Remove IPOs that have finished being news.

        Three cases: a listing that completed more than ``listed_retention_days``
        ago, a withdrawn offering, and an entry whose IPO date is far enough in
        the past that it was almost certainly an established company misread as
        a new issue. Returns the number of rows removed.
        """
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=listed_retention_days)
        ).date().isoformat()
        backdate_cutoff = (
            datetime.now(timezone.utc)
            - timedelta(days=settings.ipo_max_backdate_days)
        ).date().isoformat()

        with self.db.connection() as conn:
            cursor = conn.execute(
                """
                DELETE FROM ipo_tracker
                WHERE (status = 'listed' AND ipo_date IS NOT NULL AND ipo_date < ?)
                   OR (status = 'withdrawn' AND detected_at < ?)
                   OR (ipo_date IS NOT NULL AND ipo_date < ?)
                """,
                (cutoff, cutoff, backdate_cutoff),
            )
            return cursor.rowcount or 0

    def get_ipo_watchlist(self, limit: int = 30) -> list[dict]:
        """
        Return tracked IPOs that are still forward-looking.

        Filters at read time as well as in ``retire_stale`` so a stale row never
        reaches the dashboard in the window before the daily job next runs.
        """
        backdate_cutoff = (
            datetime.now(timezone.utc)
            - timedelta(days=settings.ipo_max_backdate_days)
        ).date().isoformat()

        with self.db.connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM ipo_tracker
                WHERE status != 'withdrawn'
                  AND (ipo_date IS NULL OR ipo_date >= ?)
                ORDER BY
                    CASE status
                        WHEN 'upcoming' THEN 1
                        WHEN 'priced' THEN 2
                        WHEN 'listed' THEN 3
                        WHEN 'rumored' THEN 4
                        ELSE 5
                    END,
                    ipo_date ASC NULLS LAST,
                    detected_at DESC
                LIMIT ?
                """,
                (backdate_cutoff, limit),
            ).fetchall()
            return [dict(row) for row in rows]
