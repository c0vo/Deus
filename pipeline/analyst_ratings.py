"""
Analyst consensus and price targets.

Nothing here is computed. These are other people's published opinions —
sell-side analysts at banks writing "Overweight, target $340" — aggregated by a
vendor into bucket counts and target statistics. Webull and every retail app
show the same aggregate under licence; this reads the free copy of it.

Source is Yahoo's `quoteSummary`, via yfinance. Two modules carry everything
needed:

  recommendationTrend   strongBuy/buy/hold/sell/strongSell counts for the
                        current month plus the three preceding it.
  financialData         targetHigh/Low/Mean/MedianPrice, recommendationMean,
                        recommendationKey, numberOfAnalystOpinions.

yfinance is used rather than a direct httpx call for one reason: the endpoint
requires a cookie-and-crumb handshake (`fc.yahoo.com` for cookies, then
`v1/test/getcrumb`, then the request, with a browser User-Agent and a persistent
cookie jar), and bare requests get `401 Invalid Crumb`. yfinance implements and
maintains that handshake, and is already a dependency. If its wrapper ever
breaks, the raw path is:

    v10/finance/quoteSummary/{sym}?modules=recommendationTrend,financialData&crumb=...

Two properties of this data shape the design:

  No history.  Yahoo returns only the CURRENT consensus. The four monthly
               buckets look like history but they move with each refresh and
               carry no as-of date, so they cannot be trusted as a point-in-time
               series. This therefore accrues one session per day exactly like
               pipeline.options_flow, and a missed session is gone.

  Not a feature. Because of the above, no predictor feature reads this table.
               Stamping today's consensus across historical training rows would
               leak future information into the past; features wait until real
               point-in-time history has accrued.

Sources deliberately not used, recorded so they are not reconsidered:
api.nasdaq.com has the best-shaped free payload but its terms forbid extraction
for training machine-learning models, which is exactly what this project does;
TipRanks disallows /api/* in robots.txt on one host and everything on the other.

This is not a NewsSource — see the pipeline.darkpool docstring for why.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo

from config.logging_config import get_logger
from config.settings import settings
from data.database import Database
from data.tickers import US, classify_market

log = get_logger(__name__)

# Seconds between tickers. yfinance is an unofficial scraper of Yahoo endpoints
# and blocks IPs on bursts; a block here would also break the predictor's price
# fetching and the option-chain snapshot, which share the same host. Serial with
# a delay, never gather().
TICKER_DELAY = 2.0

# The session a post-close snapshot belongs to is the US trading day, not the
# UTC date — the job runs at 06:45 KST, which is the previous evening in NY.
_ET = ZoneInfo("America/New_York")

# Yahoo's recommendationMean scale, for translating a number into words when the
# textual key is absent. Low is bullish: 1.0 is unanimous strong buy.
_MEAN_BANDS = (
    (1.5, "strong_buy"),
    (2.5, "buy"),
    (3.5, "hold"),
    (4.5, "sell"),
)


def _num(value: Any) -> Optional[float]:
    """Float coercion that maps NaN and junk to None rather than 0.0.

    pandas NaN is truthy and unequal to itself, so `value or None` keeps it and
    a NaN target silently becomes a real-looking price target of nan. Unlike the
    options collector, absent here must stay absent: a missing price target is
    not a target of zero.
    """
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if out != out else out


def _int(value: Any) -> Optional[int]:
    out = _num(value)
    return None if out is None else int(out)


def key_from_mean(mean: Optional[float]) -> Optional[str]:
    """Yahoo's textual recommendation from its numeric mean, when only one exists."""
    if mean is None:
        return None
    for ceiling, label in _MEAN_BANDS:
        if mean < ceiling:
            return label
    return "strong_sell"


class AnalystRatingsTracker:
    """Captures one analyst consensus row per ticker per session."""

    def __init__(self, db: Database):
        self.db = db

    @property
    def enabled(self) -> bool:
        return settings.analyst_ratings_enabled

    # ── Capture ──────────────────────────────────────────────────────────

    async def snapshot_all(self, tickers: list[str]) -> dict[str, int]:
        """Snapshot every US ticker in the list, one at a time.

        Serial by design, for the same reason as the option-chain snapshot:
        yfinance is synchronous and blocking, so each call goes to the default
        executor, and running them concurrently would both saturate that pool —
        which the predictor also uses — and present Yahoo with the burst pattern
        that gets an IP blocked.
        """
        totals = {"tickers": 0, "rows": 0, "no_coverage": 0, "failed": 0}
        if not self.enabled:
            log.info("analyst.skipped_disabled")
            return totals

        # Consensus barely moves intraday and not at all at the weekend, but
        # Yahoo keeps serving Friday's figures, so a Saturday run would store
        # them under a Saturday date and do it again on Sunday — two rows a week
        # that look like real sessions to anything reading the table later.
        session = datetime.now(timezone.utc).astimezone(_ET).date()
        if session.weekday() >= 5:
            log.info("analyst.skipped_weekend", session=session.isoformat())
            return totals

        loop = asyncio.get_running_loop()
        rows: list[dict[str, Any]] = []

        for ticker in tickers:
            if classify_market(ticker) != US:
                continue
            try:
                row = await loop.run_in_executor(None, self._snapshot_sync, ticker)
            except Exception as e:
                log.error("analyst.snapshot_failed", ticker=ticker,
                          error=str(e) or repr(e), error_type=type(e).__name__)
                totals["failed"] += 1
                continue

            totals["tickers"] += 1
            if row is None:
                totals["no_coverage"] += 1
            else:
                rows.append(row)
            await asyncio.sleep(TICKER_DELAY)

        totals["rows"] = self.db.upsert_analyst_consensus(rows)
        log.info("analyst.snapshot_complete", **totals)
        return totals

    def _snapshot_sync(self, ticker: str) -> Optional[dict[str, Any]]:
        """Blocking yfinance read for one ticker. Runs in an executor."""
        import yfinance as yf

        ticker = ticker.upper().strip()
        handle = yf.Ticker(ticker)

        # .info is one request that carries every target field plus the analyst
        # count; analyst_price_targets is a subset of it and omits the count, so
        # it is only a fallback when .info comes back thin.
        try:
            info = handle.info or {}
        except Exception as e:
            log.warning("analyst.info_failed", ticker=ticker,
                        error=str(e) or repr(e), error_type=type(e).__name__)
            info = {}

        targets_fallback = None
        if info.get("targetMeanPrice") is None:
            try:
                targets_fallback = getattr(handle, "analyst_price_targets", None)
            except Exception as e:
                log.debug("analyst.targets_fallback_failed", ticker=ticker,
                          error=str(e) or repr(e))

        try:
            recommendations = handle.recommendations
        except Exception as e:
            log.warning("analyst.recommendations_failed", ticker=ticker,
                        error=str(e) or repr(e), error_type=type(e).__name__)
            recommendations = None

        return self.build_row(ticker, info, self.latest_trend(recommendations),
                              targets_fallback)

    @staticmethod
    def latest_trend(recommendations: Any) -> dict[str, Any]:
        """Current-month bucket counts from the recommendations frame.

        Yahoo labels the current month '0m' and the preceding ones '-1m'..'-3m'.
        Selecting by label rather than by position matters because the frame
        arrives with the period as either a column or the index depending on the
        yfinance release, and a positional read silently picks a stale month
        when the ordering flips.
        """
        if recommendations is None:
            return {}
        try:
            if hasattr(recommendations, "reset_index"):
                frame = recommendations.reset_index()
                records = frame.to_dict("records")
            else:
                records = list(recommendations)
        except Exception:
            return {}
        if not records:
            return {}

        current = next(
            (r for r in records if str(r.get("period", "")).strip() == "0m"),
            records[0],
        )
        return {
            "strong_buy": _int(current.get("strongBuy")),
            "buy": _int(current.get("buy")),
            "hold": _int(current.get("hold")),
            "sell": _int(current.get("sell")),
            "strong_sell": _int(current.get("strongSell")),
        }

    @classmethod
    def build_row(cls, ticker: str, info: dict, buckets: dict,
                  targets_fallback: Optional[dict] = None) -> Optional[dict[str, Any]]:
        """Fold one ticker's payloads into a storage row, or None if uncovered.

        Pure: takes plain dicts so the whole shape is testable against a saved
        payload with no network call, matching the convention in
        tests/test_flow_sources.py.

        Returns None when there is neither a bucket count nor a price target.
        Small caps genuinely have no coverage, and storing a row of nulls for
        them would make "no analysts follow this" indistinguishable from "the
        fetch failed".
        """
        info = info or {}
        fallback = targets_fallback or {}

        target_mean = _num(info.get("targetMeanPrice")) or _num(fallback.get("mean"))
        target_high = _num(info.get("targetHighPrice")) or _num(fallback.get("high"))
        target_low = _num(info.get("targetLowPrice")) or _num(fallback.get("low"))
        target_median = _num(info.get("targetMedianPrice")) or _num(fallback.get("median"))

        counts = [buckets.get(k) for k in
                  ("strong_buy", "buy", "hold", "sell", "strong_sell")]
        has_buckets = any(c for c in counts)
        if not has_buckets and target_mean is None:
            return None

        mean = _num(info.get("recommendationMean"))
        # analyst_count prefers Yahoo's own figure, but falls back to the bucket
        # sum: they disagree slightly because the buckets count ratings and the
        # count counts analysts, and a panel showing "based on 0 analysts" beside
        # 44 ratings reads as broken.
        analyst_count = _int(info.get("numberOfAnalystOpinions"))
        if analyst_count is None and has_buckets:
            analyst_count = sum(c or 0 for c in counts)

        now = datetime.now(timezone.utc)
        return {
            "ticker": ticker.upper(),
            "session_date": now.astimezone(_ET).date().isoformat(),
            "strong_buy": buckets.get("strong_buy"),
            "buy": buckets.get("buy"),
            "hold": buckets.get("hold"),
            "sell": buckets.get("sell"),
            "strong_sell": buckets.get("strong_sell"),
            "analyst_count": analyst_count,
            "recommendation_key": info.get("recommendationKey") or key_from_mean(mean),
            "recommendation_mean": mean,
            "target_mean": target_mean,
            "target_high": target_high,
            "target_low": target_low,
            "target_median": target_median,
            "spot_price": (_num(info.get("currentPrice"))
                           or _num(info.get("regularMarketPrice"))
                           or _num(fallback.get("current"))),
            "published_at": now.isoformat(),
            "source": "yfinance",
        }

    # ── Reporting ────────────────────────────────────────────────────────

    def get_summary(self, ticker: str) -> dict[str, Any]:
        """Latest stored consensus for one ticker, with derived upside."""
        ticker = ticker.upper().strip()
        latest = self.db.get_latest_analyst_consensus(ticker)
        if not latest:
            return {"ticker": ticker, "covered": False, "latest": None}

        spot = latest.get("spot_price")
        mean = latest.get("target_mean")
        return {
            "ticker": ticker,
            "covered": True,
            "latest": latest,
            # Computed on read rather than stored: spot moves every minute and a
            # stored percentage would be stale within the hour.
            "upside_pct": ((mean - spot) / spot * 100.0)
                          if (spot and mean and spot > 0) else None,
            "total_ratings": sum(
                latest.get(k) or 0
                for k in ("strong_buy", "buy", "hold", "sell", "strong_sell")
            ),
        }

    def get_report(self, ticker: str, spot: Optional[float] = None) -> str:
        """Plain-text block for the Bull/Bear debate prompt.

        Says so explicitly when a ticker has no coverage. An empty section
        invites the model to invent a consensus, which is the failure mode every
        report in this package is written to avoid.
        """
        summary = self.get_summary(ticker)
        if not summary["covered"]:
            return f"No analyst coverage on record for {ticker}."

        row = summary["latest"]
        lines = [f"Analyst consensus for {ticker} "
                 f"(as of {str(row['published_at'])[:10]}):"]

        if summary["total_ratings"]:
            lines.append(
                f"  Ratings: {row.get('strong_buy') or 0} strong buy, "
                f"{row.get('buy') or 0} buy, {row.get('hold') or 0} hold, "
                f"{row.get('sell') or 0} sell, "
                f"{row.get('strong_sell') or 0} strong sell "
                f"({summary['total_ratings']} total)."
            )
        if row.get("recommendation_key"):
            mean = row.get("recommendation_mean")
            mean_text = f", mean {mean:.2f}/5 where 1 is most bullish" if mean else ""
            lines.append(f"  Consensus: {row['recommendation_key'].replace('_', ' ').upper()}"
                         f"{mean_text}.")

        if row.get("target_mean"):
            lines.append(
                f"  Price target: ${row['target_mean']:,.2f} mean "
                f"(${row.get('target_low') or 0:,.2f} low to "
                f"${row.get('target_high') or 0:,.2f} high)"
                + (f", {row['analyst_count']} analysts." if row.get("analyst_count")
                   else ".")
            )
            # Prefer a live spot when the caller has one; the stored figure is
            # from the morning snapshot and a debate may run hours later.
            reference = spot or row.get("spot_price")
            if reference:
                upside = (row["target_mean"] - reference) / reference * 100.0
                lines.append(f"  Implied move to mean target: {upside:+.1f}% "
                             f"from ${reference:,.2f}.")

        lines.append("  Note: sell-side targets are typically 12-month and skew "
                     "optimistic; treat as positioning context, not a forecast.")
        return "\n".join(lines)
