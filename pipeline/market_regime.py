"""
Market-wide regime series: dark-pool index, gamma exposure, and put/call ratio.

Three free daily series that describe the whole market rather than any one
ticker, and slot in beside the existing VIX / SPX / 10Y regime features:

  dix                 SqueezeMetrics' Dark Index — the dollar-volume-weighted
                      short share of the entire off-exchange tape. High DIX
                      means off-exchange buyers are absorbing supply.
  gex                 Gamma exposure. Negative means dealers are short gamma and
                      must trade with the move, amplifying it; positive means
                      they dampen it.
  occ_put_call_ratio  Put/call volume across all 18 US options exchanges, from
                      the Options Clearing Corporation.

The OCC ratio is worth preferring over the Cboe number that most sentiment
sites quote: Cboe is a single venue with ~18% market share, while OCC clears
every US listed option, so this is the actual market-wide figure rather than
one venue's slice.

Note that dix is NOT independent of the per-ticker dark-pool features in
pipeline.darkpool — it is the index-level aggregate of the same short-volume
ratio those are built from. The pair is useful as regime-plus-idiosyncratic,
but agreement between them is not confirmation from two sources.

Publication timing is deliberately conservative. Both feeds derive from the
same US session close as the FINRA file, but neither publishes a documented
timestamp, so rows are stamped as public at 00:00 UTC the following day. That
costs one session of freshness on series that move slowly and removes any doubt
about reading a value before it existed. The FINRA data does NOT get this
treatment — there the per-ticker signal is the point and a day matters.

This is not a NewsSource — see the pipeline.darkpool docstring for why.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Optional

import httpx

from config.logging_config import get_logger
from data.database import Database

log = get_logger(__name__)

DIX_CSV_URL = "https://squeezemetrics.com/monitor/static/DIX.csv"
OCC_VOLUME_URL = "https://marketdata.theocc.com/mdapi/daily-volume-totals"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Deus/2.0"

METRIC_DIX = "dix"
METRIC_GEX = "gex"
METRIC_PUT_CALL = "occ_put_call_ratio"

_REQUEST_DELAY = 0.3

# Re-read a few sessions either side of the high-water mark.
SYNC_OVERLAP_DAYS = 3

# Cold-start window for the OCC series, which is one request per session. The
# full history lives in scripts/manual/backfill_market_regime.py.
OCC_DEFAULT_BACKFILL_DAYS = 30


def published_at(session_date: date) -> str:
    """00:00 UTC the day after the session — see the module docstring."""
    stamp = datetime.combine(session_date + timedelta(days=1), time(0),
                             tzinfo=timezone.utc)
    return stamp.isoformat()


def _sessions_between(since: date, until: date):
    """Weekdays in [since, until], oldest first."""
    day = since
    while day <= until:
        if day.weekday() < 5:
            yield day
        day += timedelta(days=1)


class SqueezeMetricsProvider:
    """Reads the free daily DIX/GEX history.

    The whole series is one 220 KB CSV going back to 2011, so there is no
    incremental mode: re-fetching it costs one request and INSERT OR REPLACE
    makes re-ingestion idempotent. That is simpler than tracking a high-water
    mark, and it self-heals if the vendor restates a past value.
    """

    name = "squeezemetrics"

    def __init__(self, timeout: float = 30.0):
        self._timeout = timeout

    async def fetch(self) -> list[dict[str, Any]]:
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout, follow_redirects=True,
                headers={"User-Agent": USER_AGENT},
            ) as client:
                response = await client.get(DIX_CSV_URL)
                response.raise_for_status()
        except Exception as e:
            log.warning("regime.dix_fetch_failed",
                        error=str(e) or repr(e), error_type=type(e).__name__)
            return []
        return self.parse(response.text)

    @staticmethod
    def parse(text: str) -> list[dict[str, Any]]:
        """`date,price,dix,gex` -> two storage rows per session.

        Emitted long rather than wide (one row per metric) so a future series
        is a new metric value instead of a schema change.
        """
        rows: list[dict[str, Any]] = []
        malformed = 0

        for line in text.splitlines():
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) < 4:
                continue
            try:
                session_date = date.fromisoformat(parts[0])
                dix = float(parts[2])
                gex = float(parts[3])
            except ValueError:
                # The header row lands here on the first pass; anything else is
                # a genuinely malformed line.
                if parts[0] != "date":
                    malformed += 1
                continue

            stamp = published_at(session_date)
            iso = session_date.isoformat()
            rows.append({"metric": METRIC_DIX, "session_date": iso, "value": dix,
                         "published_at": stamp, "source": SqueezeMetricsProvider.name})
            rows.append({"metric": METRIC_GEX, "session_date": iso, "value": gex,
                         "published_at": stamp, "source": SqueezeMetricsProvider.name})

        if malformed:
            log.warning("regime.dix_malformed_rows", count=malformed)
        return rows


class OccVolumeProvider:
    """Reads OCC cleared options volume for one session at a time."""

    name = "occ"

    def __init__(self, timeout: float = 20.0):
        self._timeout = timeout

    def session(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=self._timeout, follow_redirects=True,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        )

    async def fetch_day(self, session_date: date,
                        client: Optional[httpx.AsyncClient] = None) -> Optional[dict[str, Any]]:
        if client is not None:
            return await self._fetch(session_date, client)
        async with self.session() as owned:
            return await self._fetch(session_date, owned)

    async def _fetch(self, session_date: date,
                     client: httpx.AsyncClient) -> Optional[dict[str, Any]]:
        try:
            response = await client.get(
                OCC_VOLUME_URL, params={"report_date": session_date.isoformat()}
            )
        except Exception as e:
            log.warning("regime.occ_request_failed", session=session_date.isoformat(),
                        error=str(e) or repr(e), error_type=type(e).__name__)
            return None

        if response.status_code != 200:
            log.debug("regime.occ_no_data", session=session_date.isoformat(),
                      status=response.status_code)
            return None
        try:
            return self.parse(response.json(), session_date)
        except Exception as e:
            log.warning("regime.occ_unparseable", session=session_date.isoformat(),
                        error=str(e) or repr(e), error_type=type(e).__name__)
            return None

    @staticmethod
    def parse(payload: dict, session_date: date) -> Optional[dict[str, Any]]:
        """Per-exchange volumes -> one market-wide put/call ratio row.

        The ratio is recomputed from summed contract counts rather than
        averaging the per-exchange `ratio` field. Those span 0.56-0.80 across
        venues whose market shares range from 2% to 18%, so an unweighted mean
        of them is not the market's ratio and is not anyone else's number
        either.

        Exchange names arrive space-padded ("AMEX ", "C2   "), so anything
        keyed off them needs stripping.
        """
        venues = (payload.get("entity") or {}).get("total_volume") or []
        calls = puts = 0.0
        counted = 0

        for venue in venues:
            try:
                venue_calls = float(venue.get("calls") or 0)
                venue_puts = float(venue.get("puts") or 0)
            except (TypeError, ValueError):
                continue
            if venue_calls <= 0 and venue_puts <= 0:
                continue
            calls += venue_calls
            puts += venue_puts
            counted += 1

        if counted == 0 or calls <= 0:
            return None

        return {
            "metric": METRIC_PUT_CALL,
            "session_date": session_date.isoformat(),
            "value": puts / calls,
            "published_at": published_at(session_date),
            "source": OccVolumeProvider.name,
            "exchanges": counted,
        }


class MarketRegimeTracker:
    """Pulls DIX/GEX and the OCC put/call ratio into market_regime_daily."""

    def __init__(self, db: Database,
                 dix: Optional[SqueezeMetricsProvider] = None,
                 occ: Optional[OccVolumeProvider] = None):
        self.db = db
        self.dix = dix or SqueezeMetricsProvider()
        self.occ = occ or OccVolumeProvider()

    # ── Sync ─────────────────────────────────────────────────────────────

    async def sync(self, occ_days: Optional[int] = None) -> dict[str, int]:
        """Refresh both feeds. Failures are per-feed, never fatal."""
        totals = {"dix_rows": 0, "occ_rows": 0, "occ_missing": 0}

        try:
            totals["dix_rows"] = self.db.upsert_market_regime(await self.dix.fetch())
        except Exception as e:
            log.error("regime.dix_sync_failed",
                      error=str(e) or repr(e), error_type=type(e).__name__)

        try:
            occ = await self.sync_occ(days=occ_days)
            totals["occ_rows"] = occ["rows"]
            totals["occ_missing"] = occ["missing"]
        except Exception as e:
            log.error("regime.occ_sync_failed",
                      error=str(e) or repr(e), error_type=type(e).__name__)

        log.info("regime.sync_complete", **totals)
        return totals

    async def sync_occ(self, days: Optional[int] = None,
                       since: Optional[date] = None,
                       until: Optional[date] = None) -> dict[str, int]:
        """Read OCC volume for each weekday session in the resolved window."""
        until = until or datetime.now(timezone.utc).date()
        since = since or self._resolve_occ_since(days)
        totals = {"rows": 0, "missing": 0}

        rows: list[dict[str, Any]] = []
        async with self.occ.session() as http:
            for session_date in _sessions_between(since, until):
                row = await self.occ.fetch_day(session_date, http)
                if row is None:
                    totals["missing"] += 1
                    continue
                row.pop("exchanges", None)
                rows.append(row)
                await asyncio.sleep(_REQUEST_DELAY)

        totals["rows"] = self.db.upsert_market_regime(rows)
        return totals

    def _resolve_occ_since(self, days: Optional[int]) -> date:
        today = datetime.now(timezone.utc).date()
        if days is not None:
            return today - timedelta(days=days)

        last = self.db.get_last_market_regime_date(METRIC_PUT_CALL)
        if last:
            try:
                return date.fromisoformat(str(last)[:10]) - timedelta(days=SYNC_OVERLAP_DAYS)
            except ValueError:
                pass
        return today - timedelta(days=OCC_DEFAULT_BACKFILL_DAYS)

    # ── Reporting (debate context / API) ─────────────────────────────────

    def get_summary(self, days: int = 60) -> dict[str, Any]:
        """Latest value and trailing mean for each regime metric."""
        out: dict[str, Any] = {"window_days": days}
        for metric in (METRIC_DIX, METRIC_GEX, METRIC_PUT_CALL):
            rows = self.db.get_recent_market_regime(metric, days=days)
            values = [r["value"] for r in rows if r["value"] is not None]
            out[metric] = {
                "latest": values[0] if values else None,
                "average": (sum(values) / len(values)) if values else None,
                "sessions": len(values),
                "latest_session": rows[0]["session_date"] if rows else None,
            }
        return out

    def get_report(self, days: int = 60) -> str:
        """Plain-text block for the Bull/Bear debate prompt."""
        s = self.get_summary(days=days)
        dix, gex, pcr = s[METRIC_DIX], s[METRIC_GEX], s[METRIC_PUT_CALL]

        if not any(m["sessions"] for m in (dix, gex, pcr)):
            return "No market-wide regime data (DIX/GEX/put-call) stored yet."

        lines = [f"Market-wide regime (trailing {days} sessions):"]

        if dix["latest"] is not None and dix["average"] is not None:
            lean = "above" if dix["latest"] > dix["average"] else "below"
            lines.append(
                f"  Dark Index (DIX): {dix['latest']:.3f}, {lean} its "
                f"{dix['average']:.3f} average. Higher means off-exchange "
                f"buyers are absorbing more of the tape."
            )
        if gex["latest"] is not None:
            posture = ("negative — dealers are short gamma and trade with the "
                       "move, amplifying volatility"
                       if gex["latest"] < 0 else
                       "positive — dealers are long gamma and dampen moves")
            lines.append(f"  Gamma exposure (GEX): {gex['latest']:,.0f}, {posture}.")
        if pcr["latest"] is not None and pcr["average"] is not None:
            lean = "more" if pcr["latest"] > pcr["average"] else "less"
            lines.append(
                f"  OCC put/call volume ratio: {pcr['latest']:.2f} vs "
                f"{pcr['average']:.2f} average — {lean} put-heavy than usual "
                f"(all US options exchanges, not just Cboe)."
            )

        return "\n".join(lines)
