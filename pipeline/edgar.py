"""
SEC EDGAR client — Form 4 (insider transactions) and Schedule 13D/13G (>5% stakes).

EDGAR is free and needs no API key, but it does require a User-Agent carrying a
contact address (it returns 403 without one) and asks callers to stay under
10 requests/second. Both are enforced here rather than at the call sites.

Implemented against httpx + xml.etree directly instead of a library like
edgartools: that package pulls in pyarrow and lxml, which are painful to build
on the Termux/ARM target deploy.sh installs onto, and the slice of the ownership
schema this project needs is small and stable (unchanged since 2003).

Filing dates matter more than transaction dates here. A Form 4 covers a trade
that happened up to two business days before it was filed, and the trade is not
public knowledge until the filing lands — so `filed_at` is what downstream
as-of filtering keys on. See InsiderTracker for how that is consumed.
"""

from __future__ import annotations

import asyncio
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from config.logging_config import get_logger
from config.settings import settings

log = get_logger(__name__)

SEC_BASE = "https://www.sec.gov"
SEC_DATA_BASE = "https://data.sec.gov"
COMPANY_TICKERS_URL = f"{SEC_BASE}/files/company_tickers.json"

# SEC's published ceiling is 10 req/s. Stay under it — a ban is IP-wide and
# manual to unwind, and nothing here is latency-sensitive.
MAX_REQUESTS_PER_SECOND = 8.0
_MIN_INTERVAL = 1.0 / MAX_REQUESTS_PER_SECOND

# Open-market buy / sale. The rest of the transaction codes are compensation
# mechanics (A=grant, M=option exercise, F=tax withholding, G=gift) and carry
# no directional view, so they are kept but flagged as non-discretionary.
CODE_BUY = "P"
CODE_SELL = "S"
DISCRETIONARY_CODES = frozenset({CODE_BUY, CODE_SELL})


class SecRateLimiter:
    """Process-wide throttle shared by every EDGAR request."""

    def __init__(self, min_interval: float = _MIN_INTERVAL):
        self._min_interval = min_interval
        self._lock = asyncio.Lock()
        self._last = 0.0

    async def acquire(self) -> None:
        async with self._lock:
            delta = time.monotonic() - self._last
            if delta < self._min_interval:
                await asyncio.sleep(self._min_interval - delta)
            self._last = time.monotonic()


_limiter = SecRateLimiter()


class EdgarClient:
    """Thin, rate-limited EDGAR reader.

    Returns None on failure rather than raising — callers are scheduled jobs
    that must degrade to "no new data this cycle", never take down the pipeline.
    """

    def __init__(self, user_agent: Optional[str] = None, timeout: float = 20.0):
        self._user_agent = user_agent if user_agent is not None else settings.sec_user_agent
        self._timeout = timeout
        self._cik_map: Optional[dict[str, str]] = None
        self._entity_names: dict[str, Optional[str]] = {}

    @property
    def configured(self) -> bool:
        """EDGAR rejects requests without a contact User-Agent, so treat an
        unset one the same way the API-keyed sources treat a missing key."""
        return bool(self._user_agent and "@" in self._user_agent)

    async def _get(self, url: str, client: httpx.AsyncClient) -> Optional[httpx.Response]:
        """One throttled GET, with a bounded retry on the transient statuses."""
        for attempt in range(3):
            await _limiter.acquire()
            try:
                response = await client.get(url)
            except Exception as e:
                log.warning("edgar.request_failed", url=url,
                            error=str(e) or repr(e), error_type=type(e).__name__)
                return None

            if response.status_code == 200:
                return response
            if response.status_code == 404:
                return None  # Expected: not every filing has every document.
            if response.status_code in (429, 503):
                backoff = 2 ** attempt
                log.warning("edgar.throttled", url=url,
                            status=response.status_code, backoff=backoff)
                await asyncio.sleep(backoff)
                continue
            log.warning("edgar.bad_status", url=url, status=response.status_code)
            return None
        return None

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=self._timeout,
            follow_redirects=True,
            headers={
                "User-Agent": self._user_agent,
                "Accept-Encoding": "gzip, deflate",
            },
        )

    # ── CIK lookup ───────────────────────────────────────────────────────

    async def get_cik_map(self) -> dict[str, str]:
        """Ticker -> zero-padded 10-digit CIK. Cached for the client's lifetime."""
        if self._cik_map is not None:
            return self._cik_map

        mapping: dict[str, str] = {}
        async with self._client() as client:
            response = await self._get(COMPANY_TICKERS_URL, client)
            if response is None:
                return {}
            try:
                payload = response.json()
            except Exception as e:
                log.warning("edgar.cik_map_unparseable", error=str(e))
                return {}

        # Shape is {"0": {"cik_str": 320193, "ticker": "AAPL", ...}, ...}
        for entry in payload.values():
            ticker = str(entry.get("ticker", "")).upper()
            cik = entry.get("cik_str")
            if ticker and cik is not None:
                mapping[ticker] = str(cik).zfill(10)

        self._cik_map = mapping
        log.info("edgar.cik_map_loaded", count=len(mapping))
        return mapping

    async def resolve_cik(self, ticker: str) -> Optional[str]:
        return (await self.get_cik_map()).get(str(ticker).strip().upper())

    async def get_entity_name(self, cik: str) -> Optional[str]:
        """Registered name for a CIK, memoised.

        Used to label 13D/G filers: an accession number is prefixed with the
        submitting entity's CIK, but carries no name, and the same handful of
        institutions file repeatedly.
        """
        cik = str(cik).zfill(10)
        if cik in self._entity_names:
            return self._entity_names[cik]

        name: Optional[str] = None
        async with self._client() as client:
            response = await self._get(f"{SEC_DATA_BASE}/submissions/CIK{cik}.json", client)
            if response is not None:
                try:
                    name = response.json().get("name") or None
                except Exception:
                    name = None

        self._entity_names[cik] = name
        return name

    # ── Filing index ─────────────────────────────────────────────────────

    async def list_filings(
        self,
        cik: str,
        forms: tuple[str, ...],
        since: Optional[str] = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Recent filings of the given form types for one company CIK.

        Reads the `filings.recent` block of the submissions JSON, which holds
        roughly the last 1000 filings as column-oriented parallel arrays.

        Args:
            since: ISO date (YYYY-MM-DD); filings filed before it are dropped.
        """
        url = f"{SEC_DATA_BASE}/submissions/CIK{cik}.json"
        async with self._client() as client:
            response = await self._get(url, client)
            if response is None:
                return []
            try:
                payload = response.json()
            except Exception as e:
                log.warning("edgar.submissions_unparseable", cik=cik, error=str(e))
                return []

        recent = (payload.get("filings") or {}).get("recent") or {}
        accessions = recent.get("accessionNumber") or []
        form_types = recent.get("form") or []
        filing_dates = recent.get("filingDate") or []
        report_dates = recent.get("reportDate") or []
        primary_docs = recent.get("primaryDocument") or []

        wanted = {f.upper() for f in forms}
        out: list[dict[str, Any]] = []
        for i, accession in enumerate(accessions):
            form = str(form_types[i]).upper() if i < len(form_types) else ""
            # Match amendments too: '4/A' for '4', 'SC 13D/A' for 'SC 13D'.
            if form not in wanted and form.split("/")[0].strip() not in wanted:
                continue
            filed = filing_dates[i] if i < len(filing_dates) else ""
            if since and filed and filed < since:
                continue
            out.append({
                "accession_no": accession,
                "form": form,
                "filed_at": filed,
                "report_date": report_dates[i] if i < len(report_dates) else "",
                "primary_document": primary_docs[i] if i < len(primary_docs) else "",
                "cik": cik,
            })
            if len(out) >= limit:
                break
        return out

    async def fetch_ownership_xml(
        self, cik: str, accession_no: str, primary_document: str
    ) -> Optional[bytes]:
        """Raw ownership XML for a Form 3/4/5 filing.

        `primaryDocument` usually points at the XSL-rendered view
        ('xslF345X05/wf-form4_123.xml'); the machine-readable original sits at
        the same path with that prefix removed. If that misses, fall back to the
        filing's index.json and pick the ownership document out of it.
        """
        stripped = str(accession_no).replace("-", "")
        numeric_cik = str(cik).lstrip("0") or "0"
        base = f"{SEC_BASE}/Archives/edgar/data/{numeric_cik}/{stripped}"

        candidates: list[str] = []
        if primary_document:
            if "/" in primary_document:
                candidates.append(f"{base}/{primary_document.split('/', 1)[1]}")
            candidates.append(f"{base}/{primary_document}")

        async with self._client() as client:
            for url in candidates:
                if not url.endswith(".xml"):
                    continue
                response = await self._get(url, client)
                if response is not None and b"<ownershipDocument" in response.content:
                    return response.content

            index = await self._get(f"{base}/index.json", client)
            if index is None:
                return None
            try:
                items = ((index.json().get("directory") or {}).get("item") or [])
            except Exception:
                return None
            for item in items:
                name = str(item.get("name", ""))
                if name.endswith(".xml") and not name.startswith("xsl"):
                    response = await self._get(f"{base}/{name}", client)
                    if response is not None and b"<ownershipDocument" in response.content:
                        return response.content
        return None


# ── Form 4 parsing ───────────────────────────────────────────────────────

def _text(node: Optional[ET.Element]) -> str:
    """Ownership XML wraps most leaves in <value>, sometimes alongside a
    <footnoteId>. Prefer the value child, fall back to the node's own text."""
    if node is None:
        return ""
    value = node.find("value")
    if value is not None and value.text:
        return value.text.strip()
    return (node.text or "").strip()


def _float(node: Optional[ET.Element]) -> Optional[float]:
    raw = _text(node)
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _flag(node: Optional[ET.Element]) -> int:
    """Booleans appear as '1'/'0' and occasionally 'true'/'false'."""
    raw = _text(node).lower()
    return 1 if raw in ("1", "true", "y", "yes") else 0


def parse_form4(xml_bytes: bytes, accession_no: str = "") -> list[dict[str, Any]]:
    """Flatten a Form 4 ownership document into one row per non-derivative transaction.

    Derivative transactions (options, RSUs) are deliberately skipped: they are
    dominated by scheduled compensation and would swamp the open-market signal.
    """
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as e:
        log.warning("edgar.form4_unparseable", accession=accession_no, error=str(e))
        return []

    issuer = root.find("issuer")
    ticker = _text(issuer.find("issuerTradingSymbol")) if issuer is not None else ""
    issuer_cik = _text(issuer.find("issuerCik")) if issuer is not None else ""

    owner = root.find("reportingOwner")
    insider_name, is_officer, is_director, is_ten_pct, title = "", 0, 0, 0, ""
    if owner is not None:
        owner_id = owner.find("reportingOwnerId")
        if owner_id is not None:
            insider_name = _text(owner_id.find("rptOwnerName"))
        rel = owner.find("reportingOwnerRelationship")
        if rel is not None:
            is_officer = _flag(rel.find("isOfficer"))
            is_director = _flag(rel.find("isDirector"))
            is_ten_pct = _flag(rel.find("isTenPercentOwner"))
            title = _text(rel.find("officerTitle"))

    # The 10b5-1 checkbox only became mandatory in April 2023. Absent means
    # "unknown", not "no" — storing it as NULL keeps those two apart, because
    # treating pre-2023 rows as 'not pre-scheduled' would make the column mean
    # two different things across a multi-year training window.
    aff = root.find("aff10b5One")
    is_10b5_1: Optional[int] = _flag(aff) if aff is not None else None

    rows: list[dict[str, Any]] = []
    table = root.find("nonDerivativeTable")
    if table is None:
        return rows

    for seq, txn in enumerate(table.findall("nonDerivativeTransaction")):
        coding = txn.find("transactionCoding")
        code = _text(coding.find("transactionCode")) if coding is not None else ""

        amounts = txn.find("transactionAmounts")
        shares = price = None
        acquired_disposed = ""
        if amounts is not None:
            shares = _float(amounts.find("transactionShares"))
            price = _float(amounts.find("transactionPricePerShare"))
            acquired_disposed = _text(amounts.find("transactionAcquiredDisposedCode"))

        post = txn.find("postTransactionAmounts")
        owned_after = (
            _float(post.find("sharesOwnedFollowingTransaction")) if post is not None else None
        )

        txn_date = _text(txn.find("transactionDate"))
        if not txn_date:
            continue

        value_usd = (shares * price) if (shares is not None and price is not None) else None
        # 'D' (disposed) is a sale regardless of code; sign the value so callers
        # can sum a net figure without re-deriving direction.
        if value_usd is not None and acquired_disposed.upper() == "D":
            value_usd = -value_usd

        rows.append({
            "id": f"{accession_no}_{seq}",
            "ticker": ticker.upper(),
            "issuer_cik": issuer_cik,
            "insider_name": insider_name,
            "insider_title": title,
            "is_officer": is_officer,
            "is_director": is_director,
            "is_ten_pct_owner": is_ten_pct,
            "transaction_date": txn_date,
            "transaction_code": code,
            "shares": shares,
            "price_per_share": price,
            "value_usd": value_usd,
            "shares_owned_after": owned_after,
            "is_10b5_1": is_10b5_1,
            "accession_no": accession_no,
            "is_discretionary": 1 if code in DISCRETIONARY_CODES else 0,
        })
    return rows


def parse_filing_date(raw: str) -> Optional[str]:
    """EDGAR filing dates are plain YYYY-MM-DD. Normalise to an aware ISO stamp.

    Filings are accepted until 22:00 ET, so a same-day filing is not reliably
    tradeable that session. Stamping end-of-day keeps as-of comparisons using
    strict '<' from treating it as knowable intraday.
    """
    if not raw:
        return None
    try:
        d = datetime.strptime(str(raw)[:10], "%Y-%m-%d")
    except ValueError:
        return None
    return d.replace(hour=23, minute=59, second=59, tzinfo=timezone.utc).isoformat()
