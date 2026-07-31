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

import httpx
from config.logging_config import get_logger
from config.settings import settings
from config.llm import get_deepseek_client
from config.usage import track_llm
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

# Finnhub's IPO status vocabulary mapped onto ours. 'filed' is deliberately
# absent — see _finnhub_to_record.
FINNHUB_IPO_STATUS = {
    "expected": "upcoming",
    "priced": "priced",
    "withdrawn": "withdrawn",
}


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
                  AND a.ipo_scanned_at IS NULL
                  AND a.published_at >= datetime('now', '-48 hours')
                  AND (a.event_type IS NULL OR a.event_type != 'noise')
                ORDER BY a.published_at DESC
                LIMIT 30
                """
            ).fetchall()

        if not rows:
            return []

        # Bail before the loop rather than letting _classify_ipo_article return
        # None per article — an unconfigured key would otherwise look like "the
        # model found no IPO" and stamp the whole window as scanned.
        if not get_deepseek_client():
            log.warning("ipo_detector.skipped", reason="DeepSeek not configured")
            return []

        detected = []
        for row in rows:
            article = self.db.row_to_article(row)

            # A keyword miss is deterministic — _is_ipo_related is pure text
            # matching, so a miss today is a miss forever. Stamp it and move on
            # without spending a call, otherwise these crowd the LIMIT 30 window.
            if not self._is_ipo_related(article):
                self.db.mark_scan_attempted(article.id, "ipo")
                continue

            try:
                result = await self._classify_ipo_article(article)
            except Exception as e:
                # Transient: leave unmarked so the next scan retries it. Marking
                # here would let one API outage silently discard a whole window.
                log.warning("ipo_detector.classify_failed", error=str(e), headline=article.headline[:80])
                continue

            # The model answered. That answer cost a call whether or not it
            # yielded a storable IPO, so the article is done either way.
            self.db.mark_scan_attempted(article.id, "ipo")

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

    async def scan_finnhub_ipos(self) -> list[dict]:
        """
        Pull the Finnhub IPO calendar into ipo_tracker.

        One request covers the whole window — unlike the earnings calendar there
        is no per-symbol scoping and no truncation to work around. Every failure
        mode degrades to an empty list: an IPO lane that is briefly stale is a
        far smaller problem than a scheduler job that raises every hour.
        """
        api_key = settings.finnhub_api_key
        if not api_key:
            log.info("ipo_detector.finnhub_skipped", reason="no api key")
            return []

        today = datetime.now(timezone.utc).date()
        start = (today - timedelta(days=settings.ipo_scan_days_back)).isoformat()
        end = (today + timedelta(days=settings.ipo_scan_days_ahead)).isoformat()
        url = (
            "https://finnhub.io/api/v1/calendar/ipo"
            f"?from={start}&to={end}&token={api_key}"
        )

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(url)
                if resp.status_code in (401, 403):
                    log.warning(
                        "ipo_detector.finnhub_forbidden",
                        status=resp.status_code,
                        hint="/calendar/ipo may not be on this plan",
                    )
                    return []
                if resp.status_code == 429:
                    log.warning("ipo_detector.finnhub_rate_limited")
                    return []
                resp.raise_for_status()
                rows = resp.json().get("ipoCalendar", []) or []
        except Exception as e:
            log.warning("ipo_detector.finnhub_failed", error=str(e))
            return []

        detected, updated = [], 0
        for entry in rows:
            record = self._finnhub_to_record(entry)
            if not record:
                continue

            with self.db.connection() as conn:
                existing = self._match_existing(conn, record["company_name"])

            # A withdrawn offering is only worth storing as a correction to a
            # row we already track — it is not watchlist material on its own.
            if record["status"] == "withdrawn" and not existing:
                continue

            self._store_ipo(record)
            if existing:
                updated += 1
            else:
                detected.append(record)

        if detected:
            try:
                for ipo in detected:
                    await event_bus.publish("ipo_alert", ipo)
            except Exception as e:
                log.warning("ipo_detector.sse_publish_failed", error=str(e))

        log.info(
            "ipo_detector.finnhub_scanned",
            fetched=len(rows), inserted=len(detected), updated=updated,
        )
        return detected

    @staticmethod
    def _finnhub_to_record(entry: dict) -> Optional[dict]:
        """
        Map one Finnhub IPO row onto the ipo_tracker shape.

        `filed` rows are skipped: their date is the S-1 filing date, not a
        listing date, so putting one on a calendar asserts something false.
        They are ~45% of the feed — flip this if the lane looks too sparse.
        """
        name = (entry.get("name") or "").strip()
        if not name:
            return None

        status = FINNHUB_IPO_STATUS.get((entry.get("status") or "").lower())
        if not status:
            return None

        exchange = (entry.get("exchange") or "").strip()
        shares = entry.get("numberOfShares")
        deal_value = entry.get("totalSharesValue")

        note_parts = [p for p in (
            exchange,
            f"{shares:,.0f} shares" if isinstance(shares, (int, float)) and shares else "",
            f"${deal_value / 1_000_000:,.1f}M deal" if isinstance(deal_value, (int, float)) and deal_value else "",
        ) if p]

        return {
            "company_name": name,
            "ticker": (entry.get("symbol") or "").strip() or None,
            "status": status,
            # Finnhub sends price as a string: "10.00" or a "14.00-16.00" range.
            # _store_ipo already writes strings here and the frontend types it
            # as string — do not coerce to the column's nominal REAL.
            "expected_price": (entry.get("price") or "").strip() or None,
            "expected_date": normalize_date(entry.get("date") or ""),
            # Finnhub supplies neither. Leaving them NULL lets the LLM path
            # contribute them later without being treated as a downgrade.
            "sector": None,
            "estimated_valuation": None,
            "notes": " · ".join(note_parts),
            "source": "finnhub",
            "metadata_json": json.dumps({
                "source": "finnhub",
                "exchange": exchange or None,
                "number_of_shares": shares,
                "total_shares_value": deal_value,
            }),
        }

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

        # The API call and the parse are kept in separate try blocks on purpose.
        # A failed call is transient and the article deserves another attempt on
        # the next scan; unparseable output at temperature=0.0 is deterministic
        # and will reproduce byte-for-byte, so it must not be retried. The caller
        # relies on that distinction to decide whether to mark the article
        # scanned — collapsing both into `return None` is what let an article be
        # re-extracted on every scan for its whole window.
        with track_llm(self.db, settings.deepseek_model_classifier, "ipo_extract") as u:
            u.response = response = await client.chat.completions.create(
                model=settings.deepseek_model_classifier,
                messages=[
                    {"role": "system", "content": "You extract IPO information from news articles. Output JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
                max_tokens=settings.extraction_max_output_tokens,
            )

        try:
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
        except (json.JSONDecodeError, AttributeError, IndexError, KeyError, TypeError) as e:
            log.warning("ipo_detector.parse_failed", error=str(e), headline=headline[:80])

        return None

    def _store_ipo(self, data: dict) -> None:
        """Store or update an IPO record in the database."""
        with self.db.connection() as conn:
            existing = self._match_existing(conn, data["company_name"])

            if existing:
                # Source precedence. Two writers touch this table now: the
                # hourly LLM scan and the Finnhub calendar. Finnhub knows the
                # ticker, date, price and status for certain; the LLM is
                # guessing them from prose. Letting the LLM overwrite a Finnhub
                # row would degrade confirmed data once an hour, so it may only
                # contribute the two fields Finnhub never supplies.
                if self._is_finnhub_row(conn, existing["id"]) and data.get("source") != "finnhub":
                    conn.execute(
                        """
                        UPDATE ipo_tracker
                        SET sector = COALESCE(?, sector),
                            estimated_valuation = COALESCE(?, estimated_valuation),
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                        """,
                        (data.get("sector"), data.get("estimated_valuation"), existing["id"])
                    )
                    return

                conn.execute(
                    """
                    UPDATE ipo_tracker SET ticker = ?, status = ?, ipo_date = ?,
                        offering_price = ?, sector = ?, estimated_valuation = ?,
                        notes = ?, metadata_json = COALESCE(?, metadata_json),
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (
                        data.get("ticker"), data.get("status"),
                        data.get("expected_date"), data.get("expected_price"),
                        data.get("sector"), data.get("estimated_valuation"),
                        data.get("notes", ""), data.get("metadata_json"),
                        existing["id"],
                    )
                )
            else:
                # Store source article ID
                source_id = data.get("source_article_id")
                conn.execute(
                    """
                    INSERT INTO ipo_tracker
                        (company_name, ticker, status, ipo_date, offering_price,
                         sector, estimated_valuation, source_article_id, notes,
                         metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        data["company_name"], data.get("ticker"),
                        data.get("status", "rumored"), data.get("expected_date"),
                        data.get("expected_price"), data.get("sector"),
                        data.get("estimated_valuation"), source_id,
                        data.get("notes", ""), data.get("metadata_json"),
                    )
                )

    @staticmethod
    def _match_existing(conn, company_name: str):
        """Exact company name first, then normalised — 'Apnimed, Inc.' == 'Apnimed'."""
        existing = conn.execute(
            "SELECT id, status, company_name FROM ipo_tracker WHERE company_name = ?",
            (company_name,)
        ).fetchone()
        if existing:
            return existing

        norm_name = normalize_company_name(company_name)
        if not norm_name:
            return None

        all_rows = conn.execute(
            "SELECT id, company_name, status FROM ipo_tracker"
        ).fetchall()
        for row in all_rows:
            if normalize_company_name(row["company_name"]) == norm_name:
                return row
        return None

    @staticmethod
    def _is_finnhub_row(conn, ipo_id: int) -> bool:
        row = conn.execute(
            "SELECT metadata_json FROM ipo_tracker WHERE id = ?", (ipo_id,)
        ).fetchone()
        if not row or not row["metadata_json"]:
            return False
        try:
            return json.loads(row["metadata_json"]).get("source") == "finnhub"
        except (json.JSONDecodeError, TypeError):
            return False

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
