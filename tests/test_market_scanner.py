"""
Tests for the rewritten market scanner.

Four behaviours the old scanner got wrong, and which a passing test suite
previously said nothing about:

1. `run_scan` reads stored quotes and makes no outbound HTTP call. The httpx
   module is patched and asserted untouched rather than trusted.
2. The threshold is asymmetric — a tracked position falling has a lower bar than
   anything else moving — and a deepening drop re-alerts past a step instead of
   being suppressed by "already alerted today".
3. Every alert is persisted and published, not Telegram-only.
4. The volume check's dedup key carries the day, so it can fire more than once
   in the lifetime of the database.

Every LLM and web path is patched: `explain_move` is the single seam the scanner
reaches reasoning through, so patching it covers the grader, the search and the
model call at once.
"""

from __future__ import annotations

import datetime as dt
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from data.database import Database
from pipeline import market_scanner as scanner_mod
from pipeline.grounded_answer import GroundedAnswer, MoveExplanation
from pipeline.macro_calendar import today_et
from pipeline.market_scanner import MOVE_ALERT_KINDS, MarketScanner
from tests.conftest import make_llm_response

# A tracked name deliberately absent from DEFAULT_WATCHLIST, so "tracked" and
# "on the default list" stay separable in the threshold tests.
TRACKED = "AMD"
# On the default list, never tracked here.
UNTRACKED = "MSFT"


@pytest.fixture
def db(tmp_path):
    database = Database(db_path=str(tmp_path / "test.db"))
    database.initialize()
    return database


@pytest.fixture(autouse=True)
def thresholds():
    """
    Pin the alert knobs.

    These are `.env`-configurable, and the worktree has a real `.env`: without
    pinning, "−3.2% alerts" asserts the developer's current settings rather than
    the rule.
    """
    with patch.object(scanner_mod.settings, "alert_drop_pct_tracked", 3.0), \
         patch.object(scanner_mod.settings, "alert_move_pct", 5.0), \
         patch.object(scanner_mod.settings, "alert_escalation_step_pct", 3.0), \
         patch.object(scanner_mod.settings, "alert_volume_multiple", 3.0), \
         patch.object(scanner_mod.settings, "price_refresh_seconds", 60):
        yield


def seed_quote(db, ticker, pct, *, price=100.0, volume=None):
    """One fresh `latest_prices` row — `updated_at` is CURRENT_TIMESTAMP."""
    db.upsert_latest_prices([{
        "ticker": ticker,
        "price": price,
        "previous_close": price / (1 + pct / 100.0),
        "daily_change_pct": pct,
        "volume": volume,
    }])


def age_quote(db, ticker, seconds):
    """Backdate a stored quote, to exercise the staleness guard."""
    stamp = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=seconds)
    with db.connection() as conn:
        conn.execute(
            "UPDATE latest_prices SET updated_at = ? WHERE ticker = ?",
            (stamp.strftime("%Y-%m-%d %H:%M:%S"), ticker),
        )


def seed_volume_history(db, ticker, *, sessions=20, volume=1_000_000.0):
    """Completed sessions for `get_avg_volume`, all strictly before UTC today."""
    start = dt.datetime.now(dt.timezone.utc).date() - dt.timedelta(days=sessions + 5)
    db.upsert_price_history(ticker, [
        {
            "date": (start + dt.timedelta(days=i)).isoformat(),
            "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
            "volume": volume,
        }
        for i in range(sessions)
    ])


def grounded(*, cause="Guidance cut before the open.", grounded_by="db",
             catalyst_found=True, sources=None):
    """A `GroundedAnswer` shaped the way `explain_move` returns one."""
    return GroundedAnswer(
        text=cause,
        sources=sources if sources is not None else [{
            "title": "Chipmaker cuts full-year guidance",
            "url": "https://example.com/amd-guidance",
            "source": "reuters",
            "published_at": "2026-09-12",
            "kind": "db",
        }],
        grounded_by=grounded_by,
        explanation=MoveExplanation(
            catalyst_found=catalyst_found,
            catalyst_kind="company",
            cause=cause,
            sustainability="Reads durable until the next print.",
            what_to_watch="Whether the guide is restated at the analyst day.",
            source_indices=[1],
        ),
    )


async def run_scan(db, *, answer=None, publish_raises=False):
    """
    One scan with every outbound seam patched, returning the mocks.

    `httpx` is patched at the module attribute so "makes no HTTP call" is an
    assertion rather than a claim in a docstring.
    """
    manager = MagicMock()
    manager.send_html = AsyncMock()

    explain = AsyncMock(return_value=answer or grounded())
    bus = MagicMock()
    bus.publish = AsyncMock(
        side_effect=RuntimeError("bus down") if publish_raises else None
    )

    scanner = MarketScanner(db=db, alert_manager=manager)
    with patch.object(scanner_mod, "explain_move", explain), \
         patch.object(scanner_mod, "event_bus", bus), \
         patch.object(scanner_mod.httpx, "AsyncClient") as http_client:
        await scanner.run_scan()

    return SimpleNamespace(manager=manager, explain=explain, bus=bus,
                           http_client=http_client, scanner=scanner)


def sent_texts(result):
    return [c.args[0] for c in result.manager.send_html.await_args_list]


# ── Thresholds ──────────────────────────────────────────────────────────────

class TestThresholdAsymmetry:
    """
    A tracked position falling is not the same event as anything else moving.

    The old rule was a symmetric hardcoded 5%, so a held name could fall 4.9%
    without a word while an unheld one popping 5% got a message.
    """

    def test_threshold_is_lower_for_a_tracked_drop(self, db):
        scanner = MarketScanner(db=db, alert_manager=MagicMock())
        assert scanner._threshold(TRACKED, -3.2, {TRACKED}) == 3.0

    def test_threshold_is_the_wide_one_for_a_tracked_rise(self, db):
        scanner = MarketScanner(db=db, alert_manager=MagicMock())
        assert scanner._threshold(TRACKED, 3.2, {TRACKED}) == 5.0

    def test_threshold_is_the_wide_one_for_an_untracked_drop(self, db):
        scanner = MarketScanner(db=db, alert_manager=MagicMock())
        assert scanner._threshold(UNTRACKED, -3.2, {TRACKED}) == 5.0

    async def test_tracked_drop_of_three_point_two_alerts(self, db):
        db.add_tracked_ticker(TRACKED)
        seed_quote(db, TRACKED, -3.2)

        result = await run_scan(db)

        assert result.manager.send_html.await_count == 1
        rows = db.get_recent_alerts()
        assert [r["ticker"] for r in rows] == [TRACKED]
        assert rows[0]["kind"] == "price_drop"

    async def test_tracked_rise_of_three_point_two_does_not(self, db):
        db.add_tracked_ticker(TRACKED)
        seed_quote(db, TRACKED, 3.2)

        result = await run_scan(db)

        assert result.manager.send_html.await_count == 0
        assert db.get_recent_alerts() == []

    async def test_untracked_drop_of_four_percent_does_not(self, db):
        seed_quote(db, UNTRACKED, -4.0)

        result = await run_scan(db)

        assert result.manager.send_html.await_count == 0
        assert db.get_recent_alerts() == []

    async def test_untracked_move_past_five_percent_alerts_as_a_move(self, db):
        seed_quote(db, UNTRACKED, -5.4)

        result = await run_scan(db)

        assert result.manager.send_html.await_count == 1
        assert db.get_recent_alerts()[0]["kind"] == "price_move"

    async def test_tracked_rise_past_five_percent_alerts_as_a_move(self, db):
        db.add_tracked_ticker(TRACKED)
        seed_quote(db, TRACKED, 6.1)

        result = await run_scan(db)

        assert result.manager.send_html.await_count == 1
        assert db.get_recent_alerts()[0]["kind"] == "price_move"

    async def test_a_quote_with_no_change_is_quiet(self, db):
        db.add_tracked_ticker(TRACKED)
        seed_quote(db, TRACKED, 0.4)

        result = await run_scan(db)

        assert result.manager.send_html.await_count == 0


# ── Escalation ──────────────────────────────────────────────────────────────

class TestEscalation:
    """
    A drop that deepens re-alerts; one that merely continues does not.

    The rule it replaces sent exactly one message per ticker per day, at the
    shallowest point of the day — a position that opened down 3% and closed down
    11% produced a single alert saying 3%.
    """

    async def test_deepening_past_the_step_re_alerts_but_not_before(self, db):
        db.add_tracked_ticker(TRACKED)

        seed_quote(db, TRACKED, -3.1)
        first = await run_scan(db)
        assert first.manager.send_html.await_count == 1

        # −5.0% is deeper, but not 3.0 points deeper than the −3.1% already sent.
        seed_quote(db, TRACKED, -5.0)
        second = await run_scan(db)
        assert second.manager.send_html.await_count == 0

        # −6.2% clears 3.1 + 3.0.
        seed_quote(db, TRACKED, -6.2)
        third = await run_scan(db)
        assert third.manager.send_html.await_count == 1

        pcts = sorted(r["pct"] for r in db.get_recent_alerts())
        assert pcts == pytest.approx([-6.2, -3.1])

    async def test_first_alert_of_the_day_always_goes(self, db):
        db.add_tracked_ticker(TRACKED)
        scanner = MarketScanner(db=db, alert_manager=MagicMock())
        assert await scanner._should_alert(TRACKED, -3.1) is True

    async def test_a_volume_alert_does_not_raise_the_bar_for_a_price_alert(self, db):
        """
        MOVE_ALERT_KINDS exists for this: a volume row carries the session's
        small pct, and counting it would suppress the real move alert later.
        """
        db.insert_alert({
            "ticker": TRACKED, "kind": "volume", "pct": 0.4, "price": 100.0,
            "severity": "medium", "title": "volume", "summary": "",
            "body_html": "", "sources_json": [], "grounded_by": "none",
        })
        scanner = MarketScanner(db=db, alert_manager=MagicMock())
        assert "volume" not in MOVE_ALERT_KINDS
        assert await scanner._should_alert(TRACKED, -3.2) is True


# ── Persistence, publish, send ──────────────────────────────────────────────

class TestPersistAndPublish:
    """Alerts are a row and an SSE event, not only a Telegram message."""

    async def test_the_row_carries_the_citation_and_the_grounding(self, db):
        db.add_tracked_ticker(TRACKED)
        seed_quote(db, TRACKED, -3.4, price=142.5)

        await run_scan(db)

        row = db.get_recent_alerts()[0]
        assert row["ticker"] == TRACKED
        assert row["kind"] == "price_drop"
        assert row["pct"] == pytest.approx(-3.4)
        assert row["price"] == pytest.approx(142.5)
        assert row["grounded_by"] == "db"
        assert row["severity"] == "medium"
        assert "down 3.40%" in row["title"]
        assert "Guidance cut" in row["summary"]
        assert row["body_html"]
        # _decode_alert parses sources_json into a list for the dashboard card.
        assert [s["url"] for s in row["sources"]] == [
            "https://example.com/amd-guidance"
        ]

    async def test_publish_is_awaited_with_the_stored_row(self, db):
        db.add_tracked_ticker(TRACKED)
        seed_quote(db, TRACKED, -3.4)

        result = await run_scan(db)

        result.bus.publish.assert_awaited_once()
        topic, payload = result.bus.publish.await_args.args
        assert topic == "alert"
        # The stored row, not the dict built in the scanner: the dashboard card
        # prepends live events into the list it fetched from /api/alerts, so the
        # two shapes have to match down to id/created_at.
        assert payload["id"] == db.get_recent_alerts()[0]["id"]
        assert payload["created_at"]
        assert isinstance(payload["sources"], list)

    async def test_a_bus_failure_does_not_swallow_the_telegram_send(self, db):
        db.add_tracked_ticker(TRACKED)
        seed_quote(db, TRACKED, -3.4)

        result = await run_scan(db, publish_raises=True)

        assert result.manager.send_html.await_count == 1
        assert len(db.get_recent_alerts()) == 1

    async def test_the_message_goes_through_send_html(self, db):
        db.add_tracked_ticker(TRACKED)
        seed_quote(db, TRACKED, -3.4, price=142.5)
        seed_quote(db, "SPY", -0.2)
        seed_quote(db, "QQQ", -0.3)

        result = await run_scan(db)

        # send_html, not bot.send_message: on a MagicMock the latter is a
        # non-awaitable, the scanner logs the TypeError and a test asserting
        # only on the DB row would keep passing with nothing ever sent.
        text = sent_texts(result)[0]
        assert "PRICE DROP: AMD" in text
        assert "down <b>3.40%</b>" in text
        assert "$142.50" in text
        assert "<b>Session:</b>" in text
        assert "https://example.com/amd-guidance" in text
        assert "What to watch" in text

    async def test_an_uncited_answer_says_so_in_the_message(self, db):
        db.add_tracked_ticker(TRACKED)
        seed_quote(db, TRACKED, -3.4)

        result = await run_scan(db, answer=grounded(
            cause="No dated catalyst; the move looks market-wide.",
            grounded_by="none", catalyst_found=False, sources=[],
        ))

        text = sent_texts(result)[0]
        assert "No dated source supports a cause" in text
        assert db.get_recent_alerts()[0]["grounded_by"] == "none"

    async def test_one_failing_ticker_does_not_end_the_scan(self, db):
        db.add_tracked_ticker(TRACKED)
        seed_quote(db, TRACKED, -4.0)
        seed_quote(db, UNTRACKED, -6.0)

        manager = MagicMock()
        manager.send_html = AsyncMock()
        bus = MagicMock()
        bus.publish = AsyncMock()

        def explode(db_, ticker, *a, **kw):
            if ticker == TRACKED:
                raise RuntimeError("grader exploded")
            return grounded()

        scanner = MarketScanner(db=db, alert_manager=manager)
        with patch.object(scanner_mod, "explain_move", AsyncMock(side_effect=explode)), \
             patch.object(scanner_mod, "event_bus", bus):
            await scanner.run_scan()

        assert [r["ticker"] for r in db.get_recent_alerts()] == [UNTRACKED]

    async def test_no_alert_manager_means_no_scan(self, db):
        db.add_tracked_ticker(TRACKED)
        seed_quote(db, TRACKED, -9.0)

        scanner = MarketScanner(db=db, alert_manager=None)
        with patch.object(scanner_mod, "explain_move", AsyncMock()) as explain:
            await scanner.run_scan()

        explain.assert_not_awaited()
        assert db.get_recent_alerts() == []

    def test_severity_bands(self, db):
        assert MarketScanner._severity(-3.2) == "medium"
        assert MarketScanner._severity(-5.0) == "high"
        assert MarketScanner._severity(12.0) == "critical"


# ── Cost control and freshness ──────────────────────────────────────────────

class TestWebSearchBudget:
    """Only a tracked drop is worth a Tavily search."""

    async def test_a_tracked_drop_allows_the_web(self, db):
        db.add_tracked_ticker(TRACKED)
        seed_quote(db, TRACKED, -3.4)

        result = await run_scan(db)

        assert result.explain.await_args.kwargs["allow_web"] is True

    async def test_an_untracked_move_does_not(self, db):
        seed_quote(db, UNTRACKED, -6.0)

        result = await run_scan(db)

        assert result.explain.await_args.kwargs["allow_web"] is False

    async def test_the_index_context_is_passed_down(self, db):
        db.add_tracked_ticker(TRACKED)
        seed_quote(db, TRACKED, -3.4)
        seed_quote(db, "SPY", -2.1)
        seed_quote(db, "QQQ", -2.8)

        result = await run_scan(db)

        amd_call = next(c for c in result.explain.await_args_list
                        if c.args[1] == TRACKED)
        assert amd_call.kwargs["index_context"] == {"SPY": -2.1, "QQQ": -2.8}


class TestStoredQuotesOnly:
    """`run_scan` reads the feed's table; it does not fetch."""

    async def test_no_outbound_http_call(self, db):
        db.add_tracked_ticker(TRACKED)
        seed_quote(db, TRACKED, -4.0, volume=9_000_000.0)
        seed_volume_history(db, TRACKED)

        result = await run_scan(db)

        # Two alerts fired off this one quote, both without touching the network.
        result.http_client.assert_not_called()
        assert result.manager.send_html.await_count == 2

    async def test_a_stale_quote_is_skipped(self, db):
        db.add_tracked_ticker(TRACKED)
        seed_quote(db, TRACKED, -9.0)
        age_quote(db, TRACKED, 3600)

        result = await run_scan(db)

        assert result.manager.send_html.await_count == 0

    async def test_a_quote_just_inside_the_window_is_used(self, db):
        db.add_tracked_ticker(TRACKED)
        seed_quote(db, TRACKED, -9.0)
        age_quote(db, TRACKED, 120)  # under 3 x price_refresh_seconds

        result = await run_scan(db)

        assert result.manager.send_html.await_count == 1

    async def test_a_quote_with_no_timestamp_is_skipped(self, db):
        db.add_tracked_ticker(TRACKED)
        seed_quote(db, TRACKED, -9.0)
        with db.connection() as conn:
            conn.execute("UPDATE latest_prices SET updated_at = NULL "
                         "WHERE ticker = ?", (TRACKED,))

        result = await run_scan(db)

        assert result.manager.send_html.await_count == 0

    async def test_a_quote_missing_a_price_is_skipped(self, db):
        db.add_tracked_ticker(TRACKED)
        seed_quote(db, TRACKED, -9.0)
        with db.connection() as conn:
            conn.execute("UPDATE latest_prices SET daily_change_pct = NULL "
                         "WHERE ticker = ?", (TRACKED,))

        result = await run_scan(db)

        assert result.manager.send_html.await_count == 0


# ── Volume ──────────────────────────────────────────────────────────────────

class TestVolumeAlerts:
    """
    The check that could never fire.

    It read volumes out of a `range=2d` Yahoo response — two bars at most —
    behind a `len(volumes) >= 20` guard, and keyed its dedup on the bare ticker
    in `sent_alerts`, which has no date column.
    """

    async def test_a_spike_alerts_and_is_keyed_to_today(self, db):
        db.add_tracked_ticker(TRACKED)
        seed_quote(db, TRACKED, 0.4, volume=5_000_000.0)
        seed_volume_history(db, TRACKED, volume=1_000_000.0)

        result = await run_scan(db)

        assert result.manager.send_html.await_count == 1
        row = db.get_recent_alerts()[0]
        assert row["kind"] == "volume"
        assert "5.0x average" in row["title"]
        assert "VOLUME SPIKE" in sent_texts(result)[0]
        assert db.was_alert_sent(f"{TRACKED}_{today_et().isoformat()}",
                                 "anomalous_volume")

    async def test_it_does_not_repeat_within_the_day(self, db):
        db.add_tracked_ticker(TRACKED)
        seed_quote(db, TRACKED, 0.4, volume=5_000_000.0)
        seed_volume_history(db, TRACKED, volume=1_000_000.0)

        await run_scan(db)
        second = await run_scan(db)

        assert second.manager.send_html.await_count == 0
        assert len(db.get_recent_alerts()) == 1

    async def test_yesterdays_key_does_not_suppress_today(self, db):
        """The bug the date in the key fixes: once-ever, not once-a-day."""
        db.add_tracked_ticker(TRACKED)
        seed_quote(db, TRACKED, 0.4, volume=5_000_000.0)
        seed_volume_history(db, TRACKED, volume=1_000_000.0)
        yesterday = (today_et() - dt.timedelta(days=1)).isoformat()
        db.record_alert(f"{TRACKED}_{yesterday}", "anomalous_volume")

        result = await run_scan(db)

        assert result.manager.send_html.await_count == 1

    async def test_volume_inside_the_multiple_is_quiet(self, db):
        db.add_tracked_ticker(TRACKED)
        seed_quote(db, TRACKED, 0.4, volume=2_500_000.0)
        seed_volume_history(db, TRACKED, volume=1_000_000.0)

        result = await run_scan(db)

        assert result.manager.send_html.await_count == 0

    async def test_too_little_history_means_no_baseline_and_no_alert(self, db):
        db.add_tracked_ticker(TRACKED)
        seed_quote(db, TRACKED, 0.4, volume=5_000_000.0)
        seed_volume_history(db, TRACKED, sessions=4, volume=1_000_000.0)

        result = await run_scan(db)

        assert db.get_avg_volume(TRACKED) is None
        assert result.manager.send_html.await_count == 0

    async def test_a_quote_without_volume_is_quiet(self, db):
        db.add_tracked_ticker(TRACKED)
        seed_quote(db, TRACKED, 0.4, volume=None)
        seed_volume_history(db, TRACKED, volume=1_000_000.0)

        result = await run_scan(db)

        assert result.manager.send_html.await_count == 0


# ── Earnings whisper ────────────────────────────────────────────────────────

class TestEarningsWhisper:
    """The whisper is persisted and published like every other alert."""

    async def test_no_api_key_means_no_http_client(self, db):
        manager = MagicMock()
        manager.send_html = AsyncMock()
        scanner = MarketScanner(db=db, alert_manager=manager)

        with patch.object(scanner_mod.settings, "finnhub_api_key", ""), \
             patch.object(scanner_mod.httpx, "AsyncClient") as http_client:
            await scanner.check_earnings()

        http_client.assert_not_called()

    async def test_the_whisper_is_stored_and_published(self, db):
        manager = MagicMock()
        manager.send_html = AsyncMock()
        scanner = MarketScanner(db=db, alert_manager=manager)
        bus = MagicMock()
        bus.publish = AsyncMock()

        with patch.object(scanner_mod, "event_bus", bus), \
             patch.object(scanner_mod, "is_llm_configured", return_value=True), \
             patch.object(scanner_mod.settings, "model_market_scanner", "t/scanner"), \
             patch.object(scanner_mod, "complete",
                          AsyncMock(return_value=make_llm_response("Anxious."))):
            await scanner._send_earnings_whisper(TRACKED, "2026-09-14")

        row = db.get_recent_alerts()[0]
        assert row["kind"] == "earnings_whisper"
        assert row["ticker"] == TRACKED
        assert row["pct"] is None
        assert row["grounded_by"] == "none"  # no stored articles on a fresh db
        bus.publish.assert_awaited_once()
        assert bus.publish.await_args.args[0] == "alert"
        assert manager.send_html.await_count == 1
        assert "EARNINGS WHISPER" in manager.send_html.await_args.args[0]

    async def test_an_unset_model_sends_nothing(self, db):
        manager = MagicMock()
        manager.send_html = AsyncMock()
        scanner = MarketScanner(db=db, alert_manager=manager)

        with patch.object(scanner_mod, "is_llm_configured", return_value=True), \
             patch.object(scanner_mod.settings, "model_market_scanner", ""):
            await scanner._send_earnings_whisper(TRACKED, "2026-09-14")

        assert manager.send_html.await_count == 0
        assert db.get_recent_alerts() == []
