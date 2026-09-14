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

from data.watchlist import (
    DEFAULT_WATCHLIST,
    INDEX_TICKERS,
    MARKET_INPUT_TICKERS,
    SECTOR_ETFS,
    TRAINING_BACKBONE,
)
from pipeline.price_feed import FALLBACK_TICKERS, HISTORY_RANGE, PriceFeed, _at


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


def _fetch_with_splits(payload=None, ticker="TSLA", raise_on_get=None,
                       history_range=HISTORY_RANGE):
    """(rows, splits) from _fetch_history, or None, plus the stub client."""
    feed = PriceFeed(db=MagicMock())
    client = _StubClient(payload, raise_on_get)
    return asyncio.run(
        feed._fetch_history(client, asyncio.Semaphore(1), ticker, history_range)
    ), client


def _fetch(payload=None, ticker="TSLA", raise_on_get=None):
    """Just the bar rows (or None), for the tests that are about rows."""
    fetched, client = _fetch_with_splits(payload, ticker, raise_on_get)
    return (fetched[0] if fetched is not None else None), client


def _fetch_quote(payload=None, ticker="TSLA", raise_on_get=None):
    feed = PriceFeed(db=MagicMock())
    client = _StubClient(payload, raise_on_get)
    return asyncio.run(
        feed._fetch_quote(client, asyncio.Semaphore(1), ticker)
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
    """A single run has to repair what earlier runs missed, not just add a day.

    The range also has to warm a 200-period daily moving average for the
    technical rating, which is why it is years rather than the 3mo it started
    as: 3mo is ~63 sessions, short of a single timeframe let alone the
    resampled weekly and monthly legs.
    """
    _, client = _fetch(_payload([1786109400], [420.5], [39370100]))
    params = client.calls[0][1]["params"]
    assert params["interval"] == "1d"
    assert params["range"] == HISTORY_RANGE
    assert HISTORY_RANGE.endswith("y")   # years, not days or months


def test_requests_split_events_with_the_bars():
    _, client = _fetch(_payload([1786109400], [420.5], [39370100]))
    assert client.calls[0][1]["params"]["events"] == "splits"


def test_max_range_is_requested_as_an_explicit_daily_window():
    """range=max makes Yahoo return MONTHLY bars even at interval=1d."""
    _, client = _fetch_with_splits(
        _payload([1786109400], [420.5], [39370100]), history_range="max")
    params = client.calls[0][1]["params"]
    assert "range" not in params
    assert params["period1"] == 0
    assert params["period2"] > 1786109400 - 86400 * 365 * 10
    assert params["interval"] == "1d"


def test_refuses_a_non_daily_response():
    """Monthly rows mixed into price_history poison every rolling window."""
    payload = _payload([1786109400], [420.5], [39370100])
    payload["chart"]["result"][0]["meta"]["dataGranularity"] = "1mo"
    fetched, _ = _fetch_with_splits(payload)
    assert fetched is None


def test_accepts_an_explicit_daily_granularity():
    payload = _payload([1786109400], [420.5], [39370100])
    payload["chart"]["result"][0]["meta"]["dataGranularity"] = "1d"
    rows, _ = _fetch(payload)
    assert [r["date"] for r in rows] == ["2026-08-07"]


def test_parses_split_events_as_new_shares_per_old():
    payload = _payload([1786109400], [420.5], [39370100])
    # Keyed by an unrelated stamp on purpose: the event's own date must win.
    payload["chart"]["result"][0]["events"] = {"splits": {
        "1": {"date": 1786109400, "numerator": 4.0, "denominator": 1.0,
              "splitRatio": "4:1"},
        "2": {"date": 1786023000, "numerator": 1, "denominator": 10,
              "splitRatio": "1:10"},
        "3": {"date": 1786023000, "numerator": 1, "denominator": 0},
    }}
    fetched, _ = _fetch_with_splits(payload)
    rows, splits = fetched
    assert splits == [
        {"date": "2026-08-06", "ratio": 0.1},
        {"date": "2026-08-07", "ratio": 4.0},
    ]


def test_no_split_events_is_an_empty_list():
    fetched, _ = _fetch_with_splits(_payload([1786109400], [420.5], [39370100]))
    assert fetched[1] == []


# ── universe ─────────────────────────────────────────────────────────


def _universe(tracked):
    db = MagicMock()
    db.get_tracked_tickers.return_value = tracked
    return PriceFeed(db=db).universe()


def test_universe_is_tracked_plus_the_default_watchlist():
    symbols = _universe(["MU", "AAPL"])
    assert "MU" in symbols
    assert set(DEFAULT_WATCHLIST) <= set(symbols)


def test_universe_is_sorted_and_deduped():
    """AAPL is both tracked and on the default list; it appears once."""
    symbols = _universe(["MU", "AAPL"])
    assert symbols == sorted(set(symbols))
    assert symbols.count("AAPL") == 1


def test_universe_always_warms_the_index_symbols():
    """A price alert cannot say market-wide vs idiosyncratic without these.

    They were absent before, which is why the scanner fetched Yahoo itself:
    SPY and QQQ had no latest_prices row unless somebody happened to track them.
    """
    assert set(INDEX_TICKERS) <= set(_universe(["MU"]))


def test_universe_falls_back_only_when_nothing_is_tracked():
    """GOOGL is /api/markets' cold-start fallback and not on the default list."""
    assert set(FALLBACK_TICKERS) <= set(_universe([]))
    assert "GOOGL" not in _universe(["MU"])


def test_universe_handles_a_none_from_the_database():
    assert _universe(None)


def test_history_universe_adds_the_model_inputs():
    """Bars are kept for the training backbone and market inputs; quotes are not."""
    db = MagicMock()
    db.get_tracked_tickers.return_value = ["MU"]
    feed = PriceFeed(db=db)
    history = feed.history_universe()

    assert set(feed.universe()) <= set(history)
    assert set(TRAINING_BACKBONE) <= set(history)
    assert set(SECTOR_ETFS) <= set(history)
    assert set(MARKET_INPUT_TICKERS) <= set(history)
    assert history == sorted(set(history))
    assert "^VIX" not in feed.universe()


def test_refresh_fetches_the_universe(monkeypatch):
    db = MagicMock()
    db.get_tracked_tickers.return_value = ["MU"]
    db.upsert_latest_prices.return_value = 1
    feed = PriceFeed(db=db)

    seen = []

    async def fake_fetch(client, semaphore, ticker):
        seen.append(ticker)
        return {"ticker": ticker, "price": 1.0, "previous_close": 1.0,
                "daily_change_pct": 0.0, "volume": 10.0}

    monkeypatch.setattr(feed, "_fetch_quote", fake_fetch)
    asyncio.run(feed.refresh())

    assert sorted(seen) == feed.universe()


def test_refresh_history_fetches_the_history_universe(monkeypatch):
    db = MagicMock()
    db.get_tracked_tickers.return_value = ["MU"]
    feed = PriceFeed(db=db)

    seen = []

    async def fake_fetch(client, semaphore, ticker):
        seen.append(ticker)
        return None

    monkeypatch.setattr(feed, "_fetch_history", fake_fetch)
    asyncio.run(feed.refresh_history())

    assert sorted(seen) == feed.history_universe()


# ── _fetch_quote ─────────────────────────────────────────────────────


def test_quote_reports_the_last_bar_and_the_one_before():
    quote, _ = _fetch_quote(
        _payload([1786023000, 1786109400], [415.0, 420.5], [31000000, 39370100])
    )
    assert quote["ticker"] == "TSLA"
    assert quote["price"] == 420.5
    assert quote["previous_close"] == 415.0
    assert quote["daily_change_pct"] == pytest.approx(1.3253, abs=1e-4)


def test_quote_carries_the_volume_off_the_same_bar_as_the_close():
    """What the anomalous-volume alert divides by the 20-session average.

    Index-aligned on purpose: compacting the close array first — as this did —
    throws away the index that says which volume belongs to the last close, so
    a null-padded session silently shifted the figure by one bar.
    """
    quote, _ = _fetch_quote(
        _payload(
            [1786023000, 1786066200, 1786109400],
            [415.0, None, 420.5],            # middle session untraded
            [31000000, 999, 39370100],
        )
    )
    assert quote["price"] == 420.5
    assert quote["previous_close"] == 415.0   # not the null bar
    assert quote["volume"] == 39370100.0      # not the 999 padding bar


def test_quote_volume_is_a_float():
    quote, _ = _fetch_quote(_payload([1786109400], [420.5], [39370100]))
    assert isinstance(quote["volume"], float)


def test_quote_null_volume_is_none_not_zero():
    """0 would read as 'no volume today' and the upsert COALESCEs on None."""
    quote, _ = _fetch_quote(_payload([1786109400], [420.5], [None]))
    assert quote["volume"] is None


def test_quote_missing_volume_array_is_none():
    quote, _ = _fetch_quote(
        _payload([1786023000, 1786109400], [415.0, 420.5], [31000000])
    )
    assert quote["volume"] is None


def test_quote_with_a_single_traded_bar_reports_no_change():
    quote, _ = _fetch_quote(_payload([1786109400], [420.5], [39370100]))
    assert quote["previous_close"] == 420.5
    assert quote["daily_change_pct"] == 0.0


def test_quote_requests_only_a_few_days():
    """The quote needs two closes, not two years of them."""
    _, client = _fetch_quote(_payload([1786109400], [420.5], [39370100]))
    params = client.calls[0][1]["params"]
    assert params == {"range": "5d", "interval": "1d"}


def test_quote_returns_none_when_nothing_traded():
    quote, _ = _fetch_quote(_payload([1786109400], [None], [None]))
    assert quote is None


def test_quote_returns_none_on_transport_error():
    quote, _ = _fetch_quote(raise_on_get=RuntimeError("connection reset"))
    assert quote is None


def test_refresh_drops_failed_quotes_without_dropping_the_batch(monkeypatch):
    db = MagicMock()
    db.get_tracked_tickers.return_value = ["MU", "TSLA"]
    db.upsert_latest_prices.side_effect = lambda quotes: len(quotes)
    feed = PriceFeed(db=db)

    async def fake_fetch(client, semaphore, ticker):
        if ticker == "MU":
            return None
        if ticker == "TSLA":
            raise RuntimeError("boom")
        return {"ticker": ticker, "price": 1.0, "previous_close": 1.0,
                "daily_change_pct": 0.0, "volume": None}

    monkeypatch.setattr(feed, "_fetch_quote", fake_fetch)
    stored = asyncio.run(feed.refresh())

    persisted = db.upsert_latest_prices.call_args.args[0]
    assert stored == len(persisted)
    assert {q["ticker"] for q in persisted}.isdisjoint({"MU", "TSLA"})


# ── refresh_history ──────────────────────────────────────────────────


_ONE_BAR = [{"date": "2026-08-07", "open": 1.0, "high": 1.0,
             "low": 1.0, "close": 1.0, "volume": 10}]


def test_refresh_history_upserts_per_ticker_and_counts_rows(monkeypatch):
    db = MagicMock()
    db.get_tracked_tickers.return_value = ["TSLA", "MU", "BADSYM"]
    feed = PriceFeed(db=db)
    expected = [t for t in feed.history_universe() if t != "BADSYM"]

    async def fake_fetch(client, semaphore, ticker):
        if ticker == "BADSYM":
            return None
        return _ONE_BAR, []

    monkeypatch.setattr(feed, "_fetch_history", fake_fetch)

    stored = asyncio.run(feed.refresh_history())

    assert stored == len(expected)
    upserted = [c.args[0] for c in db.upsert_price_history.call_args_list]
    assert upserted == expected            # the failed symbol is skipped
    assert db.upsert_price_history.call_count == len(expected)
    db.upsert_price_splits.assert_not_called()   # nobody split


def test_refresh_history_stores_split_events(monkeypatch):
    db = MagicMock()
    db.get_tracked_tickers.return_value = ["NVDA"]
    feed = PriceFeed(db=db)
    split = [{"date": "2024-06-10", "ratio": 10.0}]

    async def fake_fetch(client, semaphore, ticker):
        return _ONE_BAR, (split if ticker == "NVDA" else [])

    monkeypatch.setattr(feed, "_fetch_history", fake_fetch)
    asyncio.run(feed.refresh_history())

    db.upsert_price_splits.assert_called_once_with("NVDA", split)


def test_refresh_history_for_stores_split_events(monkeypatch):
    db = MagicMock()
    feed = PriceFeed(db=db)
    split = [{"date": "2024-06-10", "ratio": 10.0}]
    ranges = []

    async def fake_fetch(client, semaphore, ticker, history_range):
        ranges.append(history_range)
        return _ONE_BAR, split

    monkeypatch.setattr(feed, "_fetch_history", fake_fetch)
    stored = asyncio.run(feed.refresh_history_for(["nvda"], history_range="max"))

    assert stored == 1
    assert ranges == ["max"]
    db.upsert_price_history.assert_called_once_with("NVDA", _ONE_BAR)
    db.upsert_price_splits.assert_called_once_with("NVDA", split)


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
    expected = [t for t in feed.history_universe() if t != "TSLA"]

    async def fake_fetch(client, semaphore, ticker):
        if ticker == "TSLA":
            raise RuntimeError("boom")
        return _ONE_BAR, []

    monkeypatch.setattr(feed, "_fetch_history", fake_fetch)

    stored = asyncio.run(feed.refresh_history())
    assert stored == len(expected)
    assert [c.args[0] for c in db.upsert_price_history.call_args_list] == expected


# ── Scheduler wiring ─────────────────────────────────────────────────


def test_scheduler_registers_the_price_history_job():
    """A job with no trigger registered is the failure mode this whole fix is about."""
    import inspect

    from orchestrator import scheduler as scheduler_module

    source = inspect.getsource(scheduler_module)
    assert "id='price_history_sync'" in source
    assert "self.sync_price_history" in source
