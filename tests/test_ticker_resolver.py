"""Tests for company-name to ticker resolution.

No network: the Yahoo chart and search calls are both mocked. What is being
checked is the decision logic — that a symbol resolving to the wrong company is
rejected, and that names outside the tradeable universe are retained as chain
evidence rather than dropped.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pipeline.ticker_resolver import (
    LISTING_FOREIGN,
    LISTING_PRIVATE,
    LISTING_UNRESOLVED,
    LISTING_US,
    STATUS_RESOLVED,
    STATUS_UNLISTED,
    STATUS_UNRESOLVED,
    TickerResolver,
    name_key,
    names_match,
)


@pytest.fixture
def db():
    d = MagicMock()
    d.get_cached_resolution.return_value = None
    d.upsert_resolutions.return_value = 0
    d.upsert_price_history.return_value = None
    return d


def chart(symbol, long_name, exchange, bars=3):
    meta = {"symbol": symbol, "longName": long_name, "exchangeName": exchange}
    rows = [{"date": f"2026-01-0{i+1}", "close": 10.0 + i, "volume": 100}
            for i in range(bars)]
    return meta, rows


# ── Name normalisation ───────────────────────────────────────────────────


def test_name_key_strips_legal_suffixes():
    assert name_key("Amkor Technology, Inc.") == name_key("Amkor Technology")
    assert name_key("ASML Holding N.V.") == "asml"


def test_names_match_tolerates_suffixes_and_word_order():
    assert names_match("Amkor Technology", "Amkor Technology, Inc.")
    assert names_match("Micron", "Micron Technology Inc")
    assert not names_match("Amkor Technology", "Applied Materials")


# ── Resolution outcomes ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_resolves_a_correct_us_guess(db):
    r = TickerResolver(db)
    with patch.object(r.price_feed, "_fetch_chart",
                      new=AsyncMock(return_value=chart("AMKR", "Amkor Technology, Inc.", "NMS"))):
        out = await r.resolve_many(
            [{"company_name": "Amkor Technology", "ticker_guess": "AMKR"}]
        )
    assert out[0]["ticker"] == "AMKR"
    assert out[0]["listing_status"] == LISTING_US
    assert out[0]["resolution_status"] == STATUS_RESOLVED
    # Resolution doubles as the price warm-up.
    db.upsert_price_history.assert_called_once()


@pytest.mark.asyncio
async def test_rejects_a_symbol_that_is_a_different_company(db):
    """A wrong symbol is worse than none — it would attach a real price series
    to the wrong thesis."""
    r = TickerResolver(db)
    with patch.object(r.price_feed, "_fetch_chart",
                      new=AsyncMock(return_value=chart("AMAT", "Applied Materials, Inc.", "NMS"))), \
         patch.object(r, "_search_symbols", new=AsyncMock(return_value=[])):
        out = await r.resolve_many(
            [{"company_name": "Amkor Technology", "ticker_guess": "AMAT"}]
        )
    assert out[0]["ticker"] is None
    assert out[0]["resolution_status"] == STATUS_UNRESOLVED
    db.upsert_price_history.assert_not_called()


@pytest.mark.asyncio
async def test_foreign_listing_is_kept_but_not_tradeable(db):
    """SK hynix is the chokepoint owner and must survive into the chain, even
    though the tradeable universe is US-listed."""
    r = TickerResolver(db)
    with patch.object(r.price_feed, "_fetch_chart",
                      new=AsyncMock(return_value=chart("000660.KS", "SK hynix Inc.", "KSC"))):
        out = await r.resolve_many(
            [{"company_name": "SK hynix", "ticker_guess": "000660.KS"}]
        )
    assert out[0]["ticker"] is None            # not in the tradeable universe
    assert out[0]["listing_status"] == LISTING_FOREIGN
    assert out[0]["company_name"] == "SK hynix"  # but retained as chain evidence


@pytest.mark.asyncio
async def test_private_company_is_not_probed(db):
    r = TickerResolver(db)
    fetch = AsyncMock()
    with patch.object(r.price_feed, "_fetch_chart", new=fetch):
        out = await r.resolve_many(
            [{"company_name": "Some Private Supplier", "is_listed": False}]
        )
    assert out[0]["listing_status"] == LISTING_PRIVATE
    assert out[0]["resolution_status"] == STATUS_UNLISTED
    fetch.assert_not_called()


@pytest.mark.asyncio
async def test_falls_back_to_name_search_when_guess_is_absent(db):
    r = TickerResolver(db)
    with patch.object(r, "_search_symbols", new=AsyncMock(return_value=["MU"])), \
         patch.object(r.price_feed, "_fetch_chart",
                      new=AsyncMock(return_value=chart("MU", "Micron Technology, Inc.", "NMS"))):
        out = await r.resolve_many([{"company_name": "Micron Technology"}])
    assert out[0]["ticker"] == "MU"


@pytest.mark.asyncio
async def test_unresolvable_name_is_retained_not_dropped(db):
    r = TickerResolver(db)
    with patch.object(r, "_search_symbols", new=AsyncMock(return_value=[])), \
         patch.object(r.price_feed, "_fetch_chart", new=AsyncMock(return_value=None)):
        out = await r.resolve_many([{"company_name": "Obscure Ltd", "ticker_guess": "ZZZZ"}])
    assert len(out) == 1
    assert out[0]["ticker"] is None
    assert out[0]["listing_status"] == LISTING_UNRESOLVED


@pytest.mark.asyncio
async def test_cache_hit_skips_the_network(db):
    db.get_cached_resolution.return_value = {
        "ticker": "AMKR", "listing_status": LISTING_US, "exchange": "NMS",
        "us_proxy": None,
    }
    r = TickerResolver(db)
    fetch = AsyncMock()
    with patch.object(r.price_feed, "_fetch_chart", new=fetch):
        out = await r.resolve_many([{"company_name": "Amkor Technology"}])
    assert out[0]["ticker"] == "AMKR"
    fetch.assert_not_called()


@pytest.mark.asyncio
async def test_probe_count_is_capped(db):
    """Three probes maximum, regardless of how many symbols search returns."""
    r = TickerResolver(db)
    fetch = AsyncMock(return_value=chart("XXXX", "Totally Different Corp", "NMS"))
    with patch.object(r, "_search_symbols",
                      new=AsyncMock(return_value=["A", "B", "C", "D", "E"])), \
         patch.object(r.price_feed, "_fetch_chart", new=fetch):
        await r.resolve_many([{"company_name": "Amkor Technology", "ticker_guess": "AMKR"}])
    assert fetch.await_count <= 3
