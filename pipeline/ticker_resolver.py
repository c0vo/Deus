"""
Deus — Company name to ticker resolution

The discovery step returns company names and a guessed ticker. Those guesses
are wrong often enough to matter — models confidently emit plausible symbols,
and non-US names are worse still — so nothing reaches the scorer without being
confirmed against a live quote.

Resolution runs through Yahoo's chart endpoint rather than yfinance's
`Ticker().info`, for three reasons: `.info` is a blocking scrape (the exact
thing price_feed.py exists to avoid), the chart endpoint is already the
project's rate-limited async path, and its `meta` block carries symbol,
longName, exchangeName and currency — so confirming a symbol and warming its
price history are the same request.

The tradeable universe is US-listed. Companies that are not (SK hynix, a
private supplier) are still recorded, with `listing_status` marking why and
`us_proxy` free to point at the nearest tradeable name: a chain that omits the
actual chokepoint owner is a wrong chain even when you cannot buy it. Only the
*candidate* list is US-restricted, never the reasoning.
"""

from __future__ import annotations

import asyncio
import re
from typing import Optional

import httpx

from config.logging_config import get_logger
from config.settings import settings
from data.database import Database
from pipeline.price_feed import (
    HISTORY_TIMEOUT,
    MAX_CONCURRENT_FETCHES,
    PriceFeed,
)

log = get_logger(__name__)

YAHOO_SEARCH_URL = "https://query2.finance.yahoo.com/v1/finance/search"

# Yahoo exchange codes for the US venues. ADRs trade on these too, which is
# the point — TSM and ASML are reachable from a US broker.
US_EXCHANGES = {
    "NMS", "NYQ", "ASE", "PCX", "BTS", "NCM", "NGM", "NYS", "NAS", "AMEX",
}

LISTING_US = "us_listed"
LISTING_ADR = "adr"
LISTING_FOREIGN = "foreign_unlisted"
LISTING_PRIVATE = "private"
LISTING_UNRESOLVED = "unresolved"

STATUS_RESOLVED = "resolved"
STATUS_UNRESOLVED = "unresolved"
STATUS_UNLISTED = "unlisted"

_ADR_HINT = re.compile(r"\b(ADR|American Deposita)", re.IGNORECASE)
_SUFFIX_NOISE = re.compile(
    r"\b(inc|corp|corporation|co|ltd|limited|plc|holdings?|group|technologies|"
    r"technology|the|sa|nv|ag|se|company)\b\.?",
    re.IGNORECASE,
)


def name_key(name: str) -> str:
    """Cache key: lowercased, punctuation-stripped, legal suffixes removed.

    Periods are deleted rather than replaced with a space, so "N.V." collapses
    to "nv" and matches the suffix list; splitting it into "n v" first would
    leave stray initials behind on every European listing.
    """
    cleaned = (name or "").lower().replace(".", "")
    cleaned = re.sub(r"[^a-z0-9 ]+", " ", cleaned)
    cleaned = _SUFFIX_NOISE.sub(" ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def names_match(a: str, b: str) -> bool:
    """Loose containment check between a claimed and a returned company name."""
    ka, kb = name_key(a), name_key(b)
    if not ka or not kb:
        return False
    if ka in kb or kb in ka:
        return True
    ta, tb = set(ka.split()), set(kb.split())
    if not ta or not tb:
        return False
    return len(ta & tb) / min(len(ta), len(tb)) >= 0.6


class TickerResolver:
    """Confirms company names against live quotes. No LLM calls."""

    def __init__(self, db: Database):
        self.db = db
        self.price_feed = PriceFeed(db)

    async def resolve_many(self, candidates: list[dict]) -> list[dict]:
        """Resolve a batch of discovery candidates in place.

        Each input dict needs `company_name` and may carry `ticker_guess` and
        `is_listed`. Returns the same dicts with ticker / listing_status /
        resolution_status / market filled in.
        """
        if not candidates:
            return []

        semaphore = asyncio.Semaphore(MAX_CONCURRENT_FETCHES)
        to_write: list[dict] = []

        async with httpx.AsyncClient(timeout=HISTORY_TIMEOUT) as client:
            results = await asyncio.gather(
                *(self._resolve_one(client, semaphore, c) for c in candidates),
                return_exceptions=True,
            )

        for cand, res in zip(candidates, results):
            if isinstance(res, Exception):
                log.warning("ticker_resolver.failed",
                            company=cand.get("company_name"), error=str(res))
                res = self._unresolved(cand.get("company_name", ""))
            cand.update(res)
            to_write.append({
                "company_name_key": name_key(cand.get("company_name", "")),
                "company_name": cand.get("company_name"),
                "ticker": cand.get("ticker"),
                "listing_status": cand.get("listing_status"),
                "exchange": cand.get("exchange"),
                "us_proxy": cand.get("us_proxy"),
            })

        if to_write:
            await asyncio.to_thread(self.db.upsert_resolutions, to_write)

        resolved = sum(1 for c in candidates if c.get("ticker"))
        log.info("ticker_resolver.done", total=len(candidates), resolved=resolved)
        return candidates

    async def _resolve_one(
        self, client: httpx.AsyncClient, semaphore: asyncio.Semaphore, cand: dict
    ) -> dict:
        name = (cand.get("company_name") or "").strip()
        if not name:
            return self._unresolved("")

        key = name_key(name)
        cached = await asyncio.to_thread(
            self.db.get_cached_resolution,
            key,
            settings.thesis_ticker_cache_days,
            settings.thesis_ticker_cache_negative_days,
        )
        if cached:
            return {
                "ticker": cached.get("ticker"),
                "listing_status": cached.get("listing_status") or LISTING_UNRESOLVED,
                "resolution_status": (
                    STATUS_RESOLVED if cached.get("ticker") else STATUS_UNRESOLVED
                ),
                "exchange": cached.get("exchange"),
                "us_proxy": cached.get("us_proxy"),
                "market": "US" if cached.get("ticker") else None,
            }

        # The model already told us this one is not investable; believe it
        # rather than spending three probes proving a negative.
        if cand.get("is_listed") is False:
            return {
                "ticker": None, "listing_status": LISTING_PRIVATE,
                "resolution_status": STATUS_UNLISTED, "exchange": None,
                "market": None, "us_proxy": cand.get("us_proxy"),
            }

        symbols: list[str] = []
        guess = (cand.get("ticker_guess") or "").strip().upper()
        if guess:
            symbols.append(guess)

        found = await self._search_symbols(client, semaphore, name)
        symbols.extend(s for s in found if s not in symbols)

        alt: list[str] = []
        for symbol in symbols[:3]:  # hard probe cap
            meta = await self._probe(client, semaphore, symbol, name)
            if meta is None:
                continue
            if meta["matched"]:
                if alt:
                    meta["alt_tickers"] = alt
                return meta
            alt.append(symbol)

        return self._unresolved(name, alt)

    async def _probe(
        self,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        symbol: str,
        expected_name: str,
    ) -> Optional[dict]:
        """Confirm one symbol, warming its price history on the way through."""
        fetched = await self.price_feed._fetch_chart(client, semaphore, symbol, "1y")
        if not fetched:
            return None
        meta, rows = fetched
        resolved_symbol = (meta.get("symbol") or symbol).upper()
        long_name = meta.get("longName") or meta.get("shortName") or ""
        exchange = (meta.get("exchangeName") or "").upper()
        is_us = exchange in US_EXCHANGES

        # A symbol that resolves to some other company is worse than no symbol:
        # it would silently attach a real price series to the wrong thesis.
        matched = bool(long_name) and names_match(expected_name, long_name)
        if not matched and not long_name:
            matched = bool(rows)  # quote exists but Yahoo gave no name

        if matched and rows and is_us:
            await asyncio.to_thread(
                self.db.upsert_price_history, resolved_symbol, rows
            )

        if is_us:
            status = LISTING_ADR if _ADR_HINT.search(long_name) else LISTING_US
        else:
            status = LISTING_FOREIGN

        return {
            "ticker": resolved_symbol if (matched and is_us) else None,
            "listing_status": status if matched else LISTING_UNRESOLVED,
            "resolution_status": STATUS_RESOLVED if matched else STATUS_UNRESOLVED,
            "exchange": exchange or None,
            "market": "US" if is_us else "OTHER",
            "long_name": long_name,
            "matched": matched,
            "alt_tickers": [],
        }

    async def _search_symbols(
        self, client: httpx.AsyncClient, semaphore: asyncio.Semaphore, name: str
    ) -> list[str]:
        """Yahoo's own name search, best matches first."""
        async with semaphore:
            try:
                resp = await client.get(
                    YAHOO_SEARCH_URL,
                    params={"q": name, "quotesCount": 5, "newsCount": 0},
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                resp.raise_for_status()
                quotes = resp.json().get("quotes", []) or []
            except Exception as e:
                log.debug("ticker_resolver.search_failed", name=name, error=str(e))
                return []

        equities = [
            q for q in quotes
            if (q.get("quoteType") or "").upper() == "EQUITY" and q.get("symbol")
        ]
        # US venues first: the tradeable universe is US-listed, so a US match
        # should win over a foreign primary listing of the same company.
        equities.sort(key=lambda q: 0 if (q.get("exchange") or "").upper()
                      in US_EXCHANGES else 1)
        return [q["symbol"].upper() for q in equities]

    @staticmethod
    def _unresolved(name: str, alt: Optional[list[str]] = None) -> dict:
        """Kept, not dropped — an unresolvable name is still chain evidence."""
        return {
            "ticker": None,
            "listing_status": LISTING_UNRESOLVED,
            "resolution_status": STATUS_UNRESOLVED,
            "exchange": None,
            "market": None,
            "alt_tickers": alt or [],
        }
