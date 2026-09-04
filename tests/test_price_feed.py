"""
Tests for pipeline.price_feed.

Focused on parsing the Yahoo chart payload into price_history rows. That
mapping is the part with real failure modes — null-padded indicator arrays and
the session-date timezone — and it is what the dark-pool join depends on, since
a row keyed to the wrong date is indistinguishable from a missing row.

No network: every test feeds a canned payload through a stub httpx client.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from pipeline.price_feed import PriceFeed, _at


# Two US sessions, stamped 09:30 America/New_York as Yahoo does it.
# 1786109400 = 2026-08-07 13:30Z = 09:30 EDT; 1786023000 is the session before.
EDT_OFFSET = -14400


def _payload(stamps, closes, volumes, *, opens=None, highs=None, lows=None,
             gmtoffset=EDT_OFFSET):
    return {
        "chart": {
            "result": [
                {
                    "meta": {"gmtoffset": gmtoffset},
                    "timestamp": stamps,
                    "indicators": {
                        "quote": [
                            {
                                "open": opens if opens is not None else closes,
                                "high": highs if highs is not None else closes,
                                "low": lows if lows is not None else closes,
                                "close": closes,
                                "volume": volumes,
                            }
                        ]
                    },
                }
            ]
        }
    }


class _StubClient:
    """Stands in for httpx.AsyncClient.get, returning a canned payload."""

    def __init__(self, payload=None, raise_on_get=None):
        self._payload = payload
        self._raise = raise_on_get
        self.calls = []

    async def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if self._raise:
            raise self._raise
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json = MagicMock(return_value=self._payload)
        return resp


def _fetch(payload=None, ticker="TSLA", raise_on_get=None):
    feed = PriceFeed(db=MagicMock())
    client = _StubClient(payload, raise_on_get)
    return asyncio.run(
        feed._fetch_history(client, asyncio.Semaphore(1), ticker)
    ), client


# ── _at ──────────────────────────────────────────────────────────────


def test_at_handles_short_and_null_arrays():
    assert _at([1, 2, 3], 1) == 2
    assert _at([1, 2], 5) is None      # array shorter than timestamp
    assert _at(None, 0) is None        # array absent entirely
    assert _at([], 0) is None
    assert _at([None, 4], 0) is None   # Yahoo's null padding


# ── Session date ─────────────────────────────────────────────────────


def test_session_date_uses_the_exchange_timezone():
    """The date is the FINRA join key, so it must be the exchange's date."""
    rows, _ = _fetch(_payload([1786109400], [420.5], [39370100]))
    assert [r["date"] for r in rows] == ["2026-08-07"]


def test_session_date_respects_a_non_us_offset():
    """A KRX listing (+09:00) opening 09:00 local is still that local date."""
    # 1786060800 = 2026-08-07 00:00Z, which is 09:00 KST on the 7th.
    rows, _ = _fetch(
        _payload([1786060800], [70000.0], [12345], gmtoffset=32400)
    )
    assert [r["date"] for r in rows] == ["2026-08-07"]


def test_session_date_would_be_wrong_without_the_offset():
    """Guards the reason the offset exists: UTC alone straddles the date line.

    A 20:00 KST close stamped in UTC lands on the previous day, which is how an
    off-by-one join key gets introduced without anything raising.
    """
    # 1786143600 = 2026-08-07 23:00Z = 2026-08-08 08:00 KST.
    rows, _ = _fetch(
        _payload([1786143600], [70000.0], [12345], gmtoffset=32400)
    )
    assert [r["date"] for r in rows] == ["2026-08-08"]


# ── Row mapping ──────────────────────────────────────────────────────


def test_maps_ohlcv_and_sorts_oldest_first():
    rows, _ = _fetch(
        _payload(
            [1786109400, 1786023000],          # newest first on the wire
            [420.5, 415.0],
            [39370100, 31000000],
            opens=[410.0, 412.0],
            highs=[425.0, 418.0],
            lows=[405.0, 409.0],
        )
    )
    assert [r["date"] for r in rows] == ["2026-08-06", "2026-08-07"]
    newest = rows[-1]
    assert newest["open"] == 410.0
    assert newest["high"] == 425.0
    assert newest["low"] == 405.0
    assert newest["close"] == 420.5
    assert newest["volume"] == 39370100
    assert isinstance(newest["volume"], int)


def test_skips_sessions_with_a_null_close():
    """A null close would violate price_history's NOT NULL constraint."""
    rows, _ = _fetch(
        _payload([1786023000, 1786109400], [None, 420.5], [0, 39370100])
    )
    assert [r["date"] for r in rows] == ["2026-08-07"]


def test_null_volume_becomes_zero_not_a_dropped_row():
    """0 reads downstream as unusable, exactly like a missing row."""
    rows, _ = _fetch(_payload([1786109400], [420.5], [None]))
    assert rows[0]["volume"] == 0


def test_volume_array_shorter_than_timestamps_does_not_raise():
    rows, _ = _fetch(_payload([1786023000, 1786109400], [415.0, 420.5], [31000000]))
    assert len(rows) == 2
    assert rows[1]["volume"] == 0


# ── Failure handling ─────────────────────────────────────────────────


def test_returns_none_on_transport_error():
    """One bad symbol must not abort the rest of the watchlist."""
    rows, _ = _fetch(raise_on_get=RuntimeError("connection reset"))
    assert rows is None


def test_returns_none_when_yahoo_has_no_result():
    rows, _ = _fetch({"chart": {"result": []}})
    assert rows is None


def test_empty_timestamps_yield_no_rows():
    rows, _ = _fetch(_payload([], [], []))
    assert rows == []


def test_requests_a_wide_range_so_gaps_self_heal():
    """A single run has to repair what earlier runs missed, not just add a day."""
    _, client = _fetch(_payload([1786109400], [420.5], [39370100]))
    params = client.calls[0][1]["params"]
    assert params["interval"] == "1d"
    assert params["range"].endswith("mo")


# ── refresh_history ──────────────────────────────────────────────────


def test_refresh_history_upserts_per_ticker_and_counts_rows(monkeypatch):
    db = MagicMock()
    db.get_tracked_tickers.return_value = ["TSLA", "MU", "BADSYM"]
    feed = PriceFeed(db=db)

    async def fake_fetch(client, semaphore, ticker):
        if ticker == "BADSYM":
            return None
        return [{"date": "2026-08-07", "open": 1.0, "high": 1.0,
                 "low": 1.0, "close": 1.0, "volume": 10}]

    monkeypatch.setattr(feed, "_fetch_history", fake_fetch)

    stored = asyncio.run(feed.refresh_history())

    assert stored == 2
    upserted = [c.args[0] for c in db.upsert_price_history.call_args_list]
    assert upserted == ["TSLA", "MU"]      # the failed symbol is skipped
    assert db.upsert_price_history.call_count == 2


def test_refresh_history_falls_back_when_watchlist_is_empty(monkeypatch):
    db = MagicMock()
    db.get_tracked_tickers.return_value = []
    feed = PriceFeed(db=db)

    seen = []

    async def fake_fetch(client, semaphore, ticker):
        seen.append(ticker)
        return None

    monkeypatch.setattr(feed, "_fetch_history", fake_fetch)
    asyncio.run(feed.refresh_history())

    assert seen  # fell back rather than doing nothing


def test_refresh_history_survives_a_raising_fetch(monkeypatch):
    """gather(return_exceptions=True) means an exception is a skipped ticker."""
    db = MagicMock()
    db.get_tracked_tickers.return_value = ["TSLA", "MU"]
    feed = PriceFeed(db=db)

    async def fake_fetch(client, semaphore, ticker):
        if ticker == "TSLA":
            raise RuntimeError("boom")
        return [{"date": "2026-08-07", "open": 1.0, "high": 1.0,
                 "low": 1.0, "close": 1.0, "volume": 10}]

    monkeypatch.setattr(feed, "_fetch_history", fake_fetch)

    stored = asyncio.run(feed.refresh_history())
    assert stored == 1
    assert [c.args[0] for c in db.upsert_price_history.call_args_list] == ["MU"]


# ── Scheduler wiring ─────────────────────────────────────────────────


def test_scheduler_registers_the_price_history_job():
    """A job with no trigger registered is the failure mode this whole fix is about."""
    import inspect

    from orchestrator import scheduler as scheduler_module

    source = inspect.getsource(scheduler_module)
    assert "id='price_history_sync'" in source
    assert "self.sync_price_history" in source
