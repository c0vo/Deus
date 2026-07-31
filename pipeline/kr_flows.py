"""
Korean daily investor flows — who is actually buying KOSPI/KOSDAQ names.

Korea discloses, per ticker per day, how much 기관 (institutions) and 외국인
(foreign investors) net bought or sold. There is no free US equivalent: 13F is
quarterly and 45 days stale, so for Korean holdings this is the one place the
"what is institutional money doing" question has a genuinely daily answer.

Sourcing note — KRX moved its investor-flow endpoint behind an account login,
which is why pykrx's get_market_trading_value_by_investor now fails without
KRX_ID/KRX_PW (and why pykrx is not a dependency here: it also imports
matplotlib at module load, which the Termux deploy target does not want).

So flows are read from Naver Finance, which is open and needs no credentials.
The trade-off is deliberate and recorded in the data: Naver publishes share
counts for 기관 and 외국인 only, where KRX publishes KRW and breaks
institutions down into 연기금 / 투신 / 금융투자. Providers are pluggable so a
KRX Open API key can be swapped in later without touching the tracker, and
every row records its flow_unit and source so the two never get mixed.
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Protocol

import httpx

from config.logging_config import get_logger
from data.database import Database
from data.tickers import KR, classify_market, to_krx_code

log = get_logger(__name__)

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Deus/2.0"

# Naver paginates this table 20 sessions at a time.
ROWS_PER_PAGE = 20
NAVER_FRGN_URL = "https://finance.naver.com/item/frgn.naver"

# Be a considerate scraper: this is an unauthenticated public page.
_REQUEST_DELAY = 0.4


class KrFlowProvider(Protocol):
    """Supplies daily investor flows for one Korean ticker."""

    name: str
    flow_unit: str  # 'shares' | 'krw'

    async def fetch(self, krx_code: str, days: int) -> list[dict[str, Any]]:
        ...


class NaverFlowProvider:
    """Reads 기관/외국인 순매매 from Naver Finance's per-stock foreign/institution page.

    Returns share counts, not KRW — see the module docstring.
    """

    name = "naver"
    flow_unit = "shares"

    # Each session row leads with a date cell, then numeric cells in fixed order:
    # 종가, 전일비, 거래량, 기관 순매매량, 외국인 순매매량, 외국인 보유주수.
    _ROW = re.compile(
        r'<span class="tah p10 gray03">(\d{4}\.\d{2}\.\d{2})</span>(.*?)</tr>',
        re.S,
    )
    _NUM = re.compile(r'<span class="tah p11[^"]*">\s*([\-\+0-9,]+)\s*</span>')

    def __init__(self, timeout: float = 20.0):
        self._timeout = timeout

    @staticmethod
    def _num(raw: str) -> Optional[float]:
        cleaned = raw.replace(",", "").replace("+", "").strip()
        if not cleaned or cleaned == "-":
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None

    async def fetch(self, krx_code: str, days: int) -> list[dict[str, Any]]:
        pages = max(1, min(100, (days // ROWS_PER_PAGE) + 1))
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()

        async with httpx.AsyncClient(
            timeout=self._timeout,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT, "Referer": NAVER_FRGN_URL},
        ) as client:
            for page in range(1, pages + 1):
                try:
                    response = await client.get(
                        NAVER_FRGN_URL, params={"code": krx_code, "page": page}
                    )
                    response.raise_for_status()
                except Exception as e:
                    log.warning("kr_flows.page_failed", ticker=krx_code, page=page,
                                error=str(e) or repr(e), error_type=type(e).__name__)
                    break

                # Naver still serves this page as EUC-KR.
                html = response.content.decode("euc-kr", errors="replace")
                page_rows = self._parse(html, krx_code)
                if not page_rows:
                    break  # ran past the end of the history

                fresh = [r for r in page_rows if r["trade_date"] not in seen]
                if not fresh:
                    break  # pagination looped — stop rather than spin
                seen.update(r["trade_date"] for r in fresh)
                rows.extend(fresh)

                await asyncio.sleep(_REQUEST_DELAY)

        return rows

    def _parse(self, html: str, krx_code: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for raw_date, blob in self._ROW.findall(html):
            nums = self._NUM.findall(blob)
            # Guard the positional read: if Naver reorders or adds a column,
            # skip rather than silently writing the wrong number into inst_net.
            if len(nums) < 5:
                continue
            volume = self._num(nums[2])
            inst_net = self._num(nums[3])
            foreign_net = self._num(nums[4])
            if inst_net is None and foreign_net is None:
                continue
            out.append({
                "ticker": krx_code,
                "trade_date": raw_date.replace(".", "-"),
                "inst_net": inst_net,
                "foreign_net": foreign_net,
                "retail_net": None,       # not published by Naver
                "pension_net": None,
                "financial_inv_net": None,
                "trust_net": None,
                "total_value": volume,
                "flow_unit": self.flow_unit,
                "source": self.name,
            })
        return out


class KrFlowTracker:
    """Loads Korean investor flows into kr_investor_flows."""

    def __init__(self, db: Database, provider: Optional[KrFlowProvider] = None):
        self.db = db
        self.provider = provider or NaverFlowProvider()

    async def sync_ticker(self, ticker: str, days: int = 30) -> int:
        """Fetch and store recent flows for one Korean ticker. Returns rows written."""
        if classify_market(ticker) != KR:
            return 0
        code = to_krx_code(ticker)
        if not code:
            log.info("kr_flows.not_a_krx_code", ticker=ticker)
            return 0

        try:
            rows = await self.provider.fetch(code, days=days)
        except Exception as e:
            log.error("kr_flows.fetch_failed", ticker=ticker,
                      error=str(e) or repr(e), error_type=type(e).__name__)
            return 0

        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
        rows = [r for r in rows if r["trade_date"] >= cutoff]
        written = self.db.upsert_kr_flows(rows)
        log.info("kr_flows.synced", ticker=ticker, code=code,
                 rows=written, provider=self.provider.name)
        return written

    async def sync_all(self, tickers: list[str], days: int = 30) -> dict[str, int]:
        totals = {"tickers": 0, "rows": 0}
        for ticker in tickers:
            if classify_market(ticker) != KR:
                continue
            written = await self.sync_ticker(ticker, days=days)
            totals["tickers"] += 1
            totals["rows"] += written
        if totals["tickers"]:
            log.info("kr_flows.sync_all_complete", **totals)
        return totals

    # ── Reporting ────────────────────────────────────────────────────────

    def get_summary(self, ticker: str, days: int = 20) -> dict[str, Any]:
        """Net institutional and foreign flow over a recent window."""
        code = to_krx_code(ticker)
        if not code:
            return {}
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
        series = [r for r in self.db.get_kr_flow_series(code) if r["trade_date"] >= cutoff]
        if not series:
            return {}

        inst = sum(r["inst_net"] or 0 for r in series)
        foreign = sum(r["foreign_net"] or 0 for r in series)
        volume = sum(r["total_value"] or 0 for r in series)
        return {
            "ticker": ticker,
            "krx_code": code,
            "sessions": len(series),
            "window_days": days,
            "inst_net": inst,
            "foreign_net": foreign,
            "combined_net": inst + foreign,
            "total_volume": volume,
            "inst_net_pct_volume": (inst / volume) if volume else None,
            "foreign_net_pct_volume": (foreign / volume) if volume else None,
            "unit": series[-1].get("flow_unit", "shares"),
            "latest_date": series[-1]["trade_date"],
        }

    def get_report(self, ticker: str, days: int = 20) -> str:
        """Plain-text block for the Bull/Bear debate prompt."""
        s = self.get_summary(ticker, days=days)
        if not s:
            return f"No Korean investor-flow data available for {ticker}."

        unit = "shares" if s["unit"] == "shares" else "KRW"

        def direction(v: float) -> str:
            return "net BUYING" if v > 0 else "net SELLING" if v < 0 else "flat"

        lines = [
            f"Korean investor flows for {ticker} "
            f"(last {s['sessions']} sessions to {s['latest_date']}, in {unit}):",
            f"  Institutions (기관): {direction(s['inst_net'])} {abs(s['inst_net']):,.0f}",
            f"  Foreign investors (외국인): {direction(s['foreign_net'])} "
            f"{abs(s['foreign_net']):,.0f}",
            f"  Combined smart-money net: {s['combined_net']:+,.0f}",
        ]
        if s["inst_net_pct_volume"] is not None:
            lines.append(
                f"  As share of traded volume — institutions "
                f"{s['inst_net_pct_volume']:+.2%}, foreign "
                f"{s['foreign_net_pct_volume']:+.2%}"
            )
        return "\n".join(lines)
