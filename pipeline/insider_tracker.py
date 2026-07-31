"""
Insider and >5%-stake tracking from SEC EDGAR.

Two distinct signals, both free and both far timelier than 13F:

  Form 4      — an insider bought or sold, disclosed within 2 business days.
  SC 13D/13G  — someone crossed 5% ownership, disclosed within 10 days.
                13D means they intend to influence the company; 13G is passive.

This is not a NewsSource. That ABC is contractually `fetch() -> list[NewsArticle]`
and these are numeric rows, so the module follows the ipo_detector / event_tracker
shape instead: own class, own tables, own scheduled job.

Only US-listed tickers are in scope — see data.tickers.classify_market. Korean
holdings are covered by pipeline.kr_flows, which gets genuinely daily
institutional flow that no US free source provides.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from config.logging_config import get_logger
from config.settings import settings
from data.database import Database
from data.tickers import US, classify_market
from pipeline.edgar import (
    CODE_BUY,
    CODE_SELL,
    EdgarClient,
    parse_filing_date,
    parse_form4,
)

log = get_logger(__name__)

FORM4_TYPES = ("4",)
STAKE_TYPES = ("SC 13D", "SC 13G")

# Re-read a few days either side of the last sync. Filings can be accepted out
# of order and amendments land late, so a hard high-water mark would miss rows.
SYNC_OVERLAP_DAYS = 5


class InsiderTracker:
    """Pulls Form 4 and 13D/G disclosures into the smart-money tables."""

    def __init__(self, db: Database, client: Optional[EdgarClient] = None):
        self.db = db
        self.client = client or EdgarClient()

    @property
    def enabled(self) -> bool:
        return self.client.configured

    # ── Sync ─────────────────────────────────────────────────────────────

    async def sync_ticker(self, ticker: str, days: Optional[int] = None) -> dict[str, int]:
        """Fetch and store new disclosures for one ticker.

        Args:
            days: How far back to look. Defaults to resuming from the newest
                stored filing (minus an overlap), or the configured backfill
                window on a cold start.
        """
        ticker = ticker.upper().strip()
        result = {"insider_rows": 0, "stake_rows": 0, "filings_read": 0}

        if classify_market(ticker) != US:
            return result
        if not self.enabled:
            log.info("insider.skipped_no_user_agent", ticker=ticker)
            return result

        cik = await self.client.resolve_cik(ticker)
        if not cik:
            log.info("insider.no_cik", ticker=ticker)
            return result

        since = self._resolve_since(ticker, days)
        log.info("insider.sync_start", ticker=ticker, cik=cik, since=since)

        result["insider_rows"] = await self._sync_form4(ticker, cik, since, result)
        result["stake_rows"] = await self._sync_stakes(ticker, cik, since)

        log.info("insider.sync_complete", ticker=ticker, **result)
        return result

    def _resolve_since(self, ticker: str, days: Optional[int]) -> str:
        """ISO date to start reading from."""
        if days is not None:
            start = datetime.now(timezone.utc) - timedelta(days=days)
            return start.strftime("%Y-%m-%d")

        last = self.db.get_last_insider_filed_at(ticker)
        if last:
            try:
                anchor = datetime.fromisoformat(last)
                if anchor.tzinfo is None:
                    anchor = anchor.replace(tzinfo=timezone.utc)
                return (anchor - timedelta(days=SYNC_OVERLAP_DAYS)).strftime("%Y-%m-%d")
            except ValueError:
                pass
        start = datetime.now(timezone.utc) - timedelta(days=settings.insider_backfill_days)
        return start.strftime("%Y-%m-%d")

    async def _sync_form4(self, ticker: str, cik: str, since: str,
                          result: dict[str, int]) -> int:
        filings = await self.client.list_filings(
            cik, forms=FORM4_TYPES, since=since, limit=400
        )
        if not filings:
            return 0

        rows: list[dict[str, Any]] = []
        for filing in filings:
            filed_at = parse_filing_date(filing["filed_at"])
            if not filed_at:
                continue
            xml = await self.client.fetch_ownership_xml(
                cik, filing["accession_no"], filing["primary_document"]
            )
            result["filings_read"] += 1
            if not xml:
                continue
            for row in parse_form4(xml, filing["accession_no"]):
                # The XML carries the issuer's own symbol; trust the ticker we
                # queried when it is missing, but never silently relabel a row
                # that belongs to a different issuer.
                if row["ticker"] and row["ticker"] != ticker:
                    continue
                row["ticker"] = ticker
                row["filed_at"] = filed_at
                rows.append(row)

        return self.db.upsert_insider_transactions(rows)

    async def _sync_stakes(self, ticker: str, cik: str, since: str) -> int:
        filings = await self.client.list_filings(
            cik, forms=STAKE_TYPES, since=since, limit=100
        )
        if not filings:
            return 0

        rows: list[dict[str, Any]] = []
        for filing in filings:
            filed_at = parse_filing_date(filing["filed_at"])
            if not filed_at:
                continue
            form = filing["form"]
            base_form = form.split("/")[0].strip().upper()
            # An accession number is prefixed with the CIK of whoever submitted
            # it, which for a 13D/G is the stake-holder, not the issuer.
            filer_cik = str(filing["accession_no"]).split("-")[0]
            filer_name = await self.client.get_entity_name(filer_cik)

            rows.append({
                "id": filing["accession_no"],
                "ticker": ticker,
                "filer_name": filer_name,
                "filer_cik": filer_cik,
                "form_type": form,
                # 13D = intent to influence (activist). 13G = passive holder.
                "is_activist": 1 if base_form == "SC 13D" else 0,
                "is_amendment": 1 if "/A" in form else 0,
                # pct_of_class lives inside the document body. Structured XML is
                # only mandatory for filings from Dec 2024 on, and scraping the
                # older HTML is unreliable, so this stays NULL for now — the
                # filing event itself is the signal being tracked.
                "pct_of_class": None,
                "shares": None,
                "event_date": filing.get("report_date") or None,
                "filed_at": filed_at,
                "accession_no": filing["accession_no"],
            })

        return self.db.upsert_institutional_stakes(rows)

    async def sync_all(self, tickers: list[str], days: Optional[int] = None) -> dict[str, int]:
        """Sync every US ticker in the list, sequentially.

        Sequential on purpose: EdgarClient throttles globally anyway, and
        running these concurrently would only queue behind the same limiter
        while multiplying open connections.
        """
        totals = {"tickers": 0, "insider_rows": 0, "stake_rows": 0, "filings_read": 0}
        if not self.enabled:
            log.warning("insider.disabled_no_user_agent")
            return totals

        for ticker in tickers:
            if classify_market(ticker) != US:
                continue
            try:
                r = await self.sync_ticker(ticker, days=days)
            except Exception as e:
                log.error("insider.sync_failed", ticker=ticker,
                          error=str(e) or repr(e), error_type=type(e).__name__)
                continue
            totals["tickers"] += 1
            for k in ("insider_rows", "stake_rows", "filings_read"):
                totals[k] += r[k]
        log.info("insider.sync_all_complete", **totals)
        return totals

    # ── Reporting (debate context / API) ─────────────────────────────────

    def get_summary(self, ticker: str, days: int = 90) -> dict[str, Any]:
        """Aggregate recent insider conviction for one ticker.

        Only open-market buys and sales count. Grants, option exercises and
        tax withholding are mechanical compensation events and say nothing
        about what the insider actually thinks.
        """
        ticker = ticker.upper().strip()
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        series = [
            r for r in self.db.get_insider_series(ticker)
            if r["filed_at"] >= cutoff and r["is_discretionary"]
        ]

        buys = [r for r in series if r["transaction_code"] == CODE_BUY]
        sells = [r for r in series if r["transaction_code"] == CODE_SELL]
        buy_value = sum(abs(r["value_usd"] or 0) for r in buys)
        sell_value = sum(abs(r["value_usd"] or 0) for r in sells)
        denom = buy_value + sell_value

        stakes = [r for r in self.db.get_stakes_series(ticker) if r["filed_at"] >= cutoff]

        return {
            "ticker": ticker,
            "window_days": days,
            "buy_count": len(buys),
            "sell_count": len(sells),
            "buy_value_usd": buy_value,
            "sell_value_usd": sell_value,
            "net_value_usd": buy_value - sell_value,
            "buy_ratio": (buy_value / denom) if denom > 0 else None,
            "distinct_buyers": len({r["insider_name"] for r in buys if r["insider_name"]}),
            "distinct_sellers": len({r["insider_name"] for r in sells if r["insider_name"]}),
            "activist_filings": sum(1 for s in stakes if s["is_activist"]),
            "passive_filings": sum(1 for s in stakes if not s["is_activist"]),
            "recent_stakes": stakes[-5:],
            # Series is ordered by disclosure date; present it by trade date so
            # the reader sees a chronology of what happened, not of what filed.
            "transactions": sorted(series, key=lambda r: r["transaction_date"])[-15:],
        }

    def get_report(self, ticker: str, days: int = 90) -> str:
        """Plain-text block for the Bull/Bear debate prompt.

        Returns an explicit 'no disclosures' line rather than an empty string —
        absence of insider buying is itself information, and a blank section
        invites the model to invent one.
        """
        s = self.get_summary(ticker, days=days)
        if not s["buy_count"] and not s["sell_count"] and not s["recent_stakes"]:
            return (f"No Form 4 insider transactions or 13D/G stake filings "
                    f"disclosed for {ticker} in the last {days} days.")

        lines = [f"Insider activity for {ticker} (last {days} days, open-market only):"]
        if s["buy_count"] or s["sell_count"]:
            lines.append(
                f"  Buys: {s['buy_count']} totalling ${s['buy_value_usd']:,.0f} "
                f"across {s['distinct_buyers']} insider(s)."
            )
            lines.append(
                f"  Sells: {s['sell_count']} totalling ${s['sell_value_usd']:,.0f} "
                f"across {s['distinct_sellers']} insider(s)."
            )
            lines.append(f"  Net: ${s['net_value_usd']:,.0f}.")
            if s["buy_ratio"] is not None:
                lines.append(f"  Buy ratio: {s['buy_ratio']:.0%} of traded value.")

        for txn in s["transactions"][-5:]:
            who = txn["insider_name"] or "unknown"
            role = txn["insider_title"] or (
                "Director" if txn["is_director"] else
                "Officer" if txn["is_officer"] else
                "10% owner" if txn["is_ten_pct_owner"] else "insider"
            )
            action = "BUY" if txn["transaction_code"] == CODE_BUY else "SELL"
            planned = " (10b5-1 pre-scheduled)" if txn["is_10b5_1"] == 1 else ""
            lines.append(
                f"  {txn['transaction_date']}  {action}  {who} ({role})  "
                f"{txn['shares']:,.0f} sh @ ${txn['price_per_share']:,.2f}{planned}"
            )

        for st in s["recent_stakes"]:
            kind = "13D (activist)" if st["is_activist"] else "13G (passive)"
            amend = " amendment" if st["is_amendment"] else ""
            lines.append(
                f"  {str(st['filed_at'])[:10]}  {kind}{amend} filed by "
                f"{st['filer_name'] or st.get('filer_cik') or 'unknown filer'}"
            )

        return "\n".join(lines)
