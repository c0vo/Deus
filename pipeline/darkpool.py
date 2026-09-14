"""
Off-exchange ("dark pool") volume from FINRA.

Every US equity trade that does not execute on a lit exchange is printed to a
FINRA Trade Reporting Facility, and FINRA republishes those prints per symbol
per day, free and without credentials. That covers ATS/dark-pool crossing and
wholesaler internalisation alike — measured at 33-47% of consolidated volume
for large caps — which makes it the one genuinely free per-ticker view of where
size is trading.

Two numbers come out of it, and they answer different questions:

  total_volume / consolidated_volume   how much of the tape printed off-exchange
  short_volume / total_volume          how much of that off-exchange flow was short

The second needs care. It is NOT short interest and it is NOT a directional bet
count: roughly 40% of off-exchange volume is retail wholesaler internalisation
where the wholesaler marks its own side short, plus market-maker hedging. The
level is therefore a near-constant per ticker and carries no signal, while
comparing it across tickers carries less than none. Only the deviation from a
ticker's OWN trailing distribution is usable, which is why every feature built
on this is a z-score — see pipeline.features._darkpool_features.

This is not a NewsSource. That ABC is contractually `fetch() -> list[NewsArticle]`
and these are numeric rows, so the module follows the insider_tracker / kr_flows
shape instead: own class, own table, own scheduled job.

US-listed tickers only — the FINRA TRFs cover US equities and ETFs. A freshly
tracked ticker has no history until the sync accrues one, so features stay at
their defaults for the first few weeks.

Sourcing notes, each learned the hard way against the live endpoint:

  * One file carries EVERY symbol (~12,000 rows, ~540 KB), so a sync is one GET
    per session regardless of how many tickers are tracked. Rows are filtered to
    the watchlist before storage; keeping all of them would be ~24M rows.
  * The CDN keeps a rolling ~8 years and the floor moves forward over time.
    Nothing here hardcodes a start date; scripts/manual/backfill_darkpool.py
    discovers the floor by probing.
  * A missing file returns 403 (S3-style AccessDenied), never 404 — weekends,
    holidays and pre-retention dates are indistinguishable from each other. So
    403 means "no file", never "retry", and only 429/503 are retried.
  * HEAD returns 403 for every date including valid ones, so existence cannot be
    probed cheaply. GET is the only option.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo

import httpx

from config.logging_config import get_logger
from config.settings import settings
from data.database import Database
from data.tickers import US, classify_market

log = get_logger(__name__)

FINRA_CNMS_URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{stamp}.txt"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Deus/2.0"

# FINRA posts each session's file at ~18:00 ET the same day. That is 07:00 KST
# next morning under EDT and 08:00 under EST, which is what the two-shot
# scheduler job in orchestrator.scheduler is working around.
PUBLISH_HOUR_ET = 18
_ET = ZoneInfo("America/New_York")

# Be a considerate client of a public CDN object.
_REQUEST_DELAY = 0.3

# Re-read a few sessions either side of the high-water mark. FINRA restates:
# a file can be reposted with corrected figures after its first publication.
SYNC_OVERLAP_DAYS = 3

# Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market
_EXPECTED_FIELDS = 6


def _sessions_between(since: date, until: date):
    """Weekdays in [since, until], oldest first.

    Holidays are left in — they 403 like any other absent file and are counted
    as missing. Filtering them would need a market calendar dependency to save
    roughly nine requests a year.
    """
    day = since
    while day <= until:
        if day.weekday() < 5:
            yield day
        day += timedelta(days=1)


class FinraShortVolumeClient:
    """Thin reader for the FINRA consolidated short-volume files.

    Returns None on absence or failure rather than raising — callers are
    scheduled jobs that must degrade to "no new data this session", never take
    down the pipeline.
    """

    def __init__(self, timeout: float = 30.0):
        self._timeout = timeout

    def session(self) -> httpx.AsyncClient:
        """One client for a whole range sync, so connections are reused."""
        return httpx.AsyncClient(
            timeout=self._timeout,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"},
        )

    async def fetch_day(self, session_date: date,
                        client: Optional[httpx.AsyncClient] = None) -> Optional[list[dict[str, Any]]]:
        """Every symbol's off-exchange volume for one session, or None if absent."""
        if client is not None:
            return await self._fetch(session_date, client)
        async with self.session() as owned:
            return await self._fetch(session_date, owned)

    async def _fetch(self, session_date: date,
                     client: httpx.AsyncClient) -> Optional[list[dict[str, Any]]]:
        url = FINRA_CNMS_URL.format(stamp=session_date.strftime("%Y%m%d"))

        for attempt in range(3):
            try:
                response = await client.get(url)
            except Exception as e:
                log.warning("darkpool.request_failed", url=url,
                            error=str(e) or repr(e), error_type=type(e).__name__)
                return None

            if response.status_code == 200:
                return self.parse(response.text, expected_date=session_date)
            if response.status_code in (403, 404):
                # Weekend, market holiday, or past the CDN's retention floor.
                # Never retried: 403 is how this endpoint spells "no such file".
                log.debug("darkpool.no_file", url=url, status=response.status_code)
                return None
            if response.status_code in (429, 503):
                backoff = 2 ** attempt
                log.warning("darkpool.throttled", url=url,
                            status=response.status_code, backoff=backoff)
                await asyncio.sleep(backoff)
                continue

            log.warning("darkpool.bad_status", url=url, status=response.status_code)
            return None
        return None

    @staticmethod
    def parse(text: str, expected_date: Optional[date] = None) -> list[dict[str, Any]]:
        """Pipe-delimited FINRA rows -> storage dicts.

        Two lines in every file are not data and both would crash a naive
        positional read: the `Date|Symbol|...` header, and a bare record count
        on the final line (e.g. `12180`) which splits into a single field.

        Every numeric column is read with float(), never int(). FINRA publishes
        fractional share-adjusted volumes (5540409.463985), and
        ShortExemptVolume in particular mixes plain integers and decimals within
        the same file.
        """
        rows: list[dict[str, Any]] = []
        malformed = 0

        for line in text.splitlines():
            parts = line.strip().split("|")
            if len(parts) < _EXPECTED_FIELDS:
                continue  # trailing record count, or a blank line

            stamp, symbol = parts[0].strip(), parts[1].strip().upper()
            if len(stamp) != 8 or not stamp.isdigit():
                continue  # header row
            if not symbol:
                continue

            try:
                session_date = datetime.strptime(stamp, "%Y%m%d").date()
                short_volume = float(parts[2])
                short_exempt = float(parts[3])
                total_volume = float(parts[4])
            except ValueError:
                malformed += 1
                continue

            rows.append({
                "ticker": symbol,
                "session_date": session_date.isoformat(),
                "short_volume": short_volume,
                "short_exempt_volume": short_exempt,
                "total_volume": total_volume,
                "market_codes": parts[5].strip(),
                "published_at": FinraShortVolumeClient.published_at(session_date),
                "source": "finra_cnms",
            })

        if malformed:
            log.warning("darkpool.malformed_rows", count=malformed,
                        expected_date=expected_date.isoformat() if expected_date else None)
        return rows

    @staticmethod
    def published_at(session_date: date) -> str:
        """UTC instant the session's file became public.

        Built in America/New_York and converted, so the EDT/EST difference is
        handled by the tz database rather than a fixed offset: 18:00 ET is
        22:00 UTC in summer and 23:00 UTC in winter, and the predictor's as-of
        cut sits at 23:59:59 UTC. A hardcoded offset would put winter sessions
        on the wrong side of that boundary.
        """
        stamp = datetime.combine(session_date, time(hour=PUBLISH_HOUR_ET), tzinfo=_ET)
        return stamp.astimezone(timezone.utc).isoformat()


class DarkPoolTracker:
    """Pulls FINRA off-exchange volume into the offexchange_volume table."""

    def __init__(self, db: Database, client: Optional[FinraShortVolumeClient] = None):
        self.db = db
        self.client = client or FinraShortVolumeClient()

    # ── Sync ─────────────────────────────────────────────────────────────

    async def sync_recent(self, tickers: list[str],
                          days: Optional[int] = None) -> dict[str, int]:
        """Fetch and store any sessions newer than what is already stored."""
        since = self._resolve_since(days)
        until = datetime.now(timezone.utc).date()
        return await self.sync_range(tickers, since, until)

    def _resolve_since(self, days: Optional[int]) -> date:
        """First session to read.

        Resumes from the newest stored session across ALL tickers, not per
        ticker: one file carries every symbol, so a per-ticker high-water mark
        would re-download the same file once per ticker to learn it already had
        the rows.
        """
        today = datetime.now(timezone.utc).date()
        if days is not None:
            return today - timedelta(days=days)

        last = self.db.get_last_offexchange_date()
        if last:
            try:
                return date.fromisoformat(str(last)[:10]) - timedelta(days=SYNC_OVERLAP_DAYS)
            except ValueError:
                pass
        return today - timedelta(days=settings.darkpool_backfill_days)

    async def sync_range(self, tickers: list[str], since: date,
                         until: date) -> dict[str, int]:
        """Read every weekday session in [since, until] and store the tracked rows."""
        wanted = {t.upper().strip() for t in tickers if classify_market(t) == US}
        totals = {"sessions": 0, "rows": 0, "missing": 0}
        if not wanted:
            log.info("darkpool.no_us_tickers")
            return totals

        log.info("darkpool.sync_start", since=since.isoformat(),
                 until=until.isoformat(), tickers=len(wanted))

        async with self.client.session() as http:
            for session_date in _sessions_between(since, until):
                try:
                    rows = await self.client.fetch_day(session_date, http)
                except Exception as e:
                    log.error("darkpool.session_failed", session=session_date.isoformat(),
                              error=str(e) or repr(e), error_type=type(e).__name__)
                    totals["missing"] += 1
                    continue

                if rows is None:
                    totals["missing"] += 1
                    continue

                tracked = [r for r in rows if r["ticker"] in wanted]
                totals["rows"] += self.db.upsert_offexchange_volume(tracked)
                totals["sessions"] += 1
                await asyncio.sleep(_REQUEST_DELAY)

        # Every session missing is the signature of an IP block or a moved
        # retention floor, not of a quiet week — worth separating in the logs.
        if totals["sessions"] == 0 and totals["missing"] > 0:
            log.warning("darkpool.all_sessions_missing", missing=totals["missing"],
                        since=since.isoformat(), until=until.isoformat())
        else:
            log.info("darkpool.sync_complete", **totals)
        return totals

    # ── Reporting (debate context / API) ─────────────────────────────────

    def get_summary(self, ticker: str, days: int = 20) -> dict[str, Any]:
        """Recent off-exchange activity for one ticker.

        Reports each ratio against its own trailing mean rather than in
        isolation, because the level is a ticker-specific constant and only the
        deviation says anything — see the module docstring.
        """
        ticker = ticker.upper().strip()
        rows = self.db.get_recent_offexchange(ticker, days=days)

        short_ratios: list[float] = []
        shares: list[float] = []
        for r in rows:
            total = r.get("total_volume") or 0.0
            if total > 0 and r.get("short_volume") is not None:
                short_ratios.append(r["short_volume"] / total)
            consolidated = r.get("consolidated_volume") or 0.0
            if consolidated > 0 and total > 0:
                shares.append(total / consolidated)

        def _mean(values: list[float]) -> Optional[float]:
            return (sum(values) / len(values)) if values else None

        return {
            "ticker": ticker,
            "window_days": days,
            "sessions": len(rows),
            "latest_session": rows[0]["session_date"] if rows else None,
            "short_ratio": short_ratios[0] if short_ratios else None,
            "short_ratio_avg": _mean(short_ratios),
            "offexch_share": shares[0] if shares else None,
            "offexch_share_avg": _mean(shares),
        }

    def get_report(self, ticker: str, days: int = 20) -> str:
        """Plain-text block for the Bull/Bear debate prompt.

        Returns an explicit "no data" line rather than an empty string, and
        spells out what the short ratio is not: an LLM handed "short volume
        62%" with no framing will read it as bearish positioning, when most of
        it is market-maker and wholesaler hedging.
        """
        s = self.get_summary(ticker, days=days)
        if not s["sessions"]:
            return (f"No FINRA off-exchange volume data stored for {ticker} "
                    f"(needs a few sessions of history before it reports).")

        lines = [
            f"Off-exchange (dark pool) activity for {ticker} "
            f"(FINRA TRF prints, last {s['sessions']} sessions to {s['latest_session']}):"
        ]

        if s["offexch_share"] is not None and s["offexch_share_avg"] is not None:
            delta = s["offexch_share"] - s["offexch_share_avg"]
            lines.append(
                f"  Off-exchange share of volume: {s['offexch_share']:.1%} "
                f"vs {s['offexch_share_avg']:.1%} average "
                f"({delta:+.1%} vs its own norm)."
            )
        if s["short_ratio"] is not None and s["short_ratio_avg"] is not None:
            delta = s["short_ratio"] - s["short_ratio_avg"]
            lines.append(
                f"  Short share of off-exchange volume: {s['short_ratio']:.1%} "
                f"vs {s['short_ratio_avg']:.1%} average ({delta:+.1%})."
            )

        lines.append(
            "  Note: off-exchange short volume is not short interest. Most of it "
            "is market-maker and retail-wholesaler hedging, so only the deviation "
            "from this ticker's own average is meaningful — not the level."
        )
        return "\n".join(lines)
