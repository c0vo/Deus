"""Tests for SSE (Server-Sent Events) streaming format and parsing."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from api import server
from api.server import BRAIN_STREAM_TOPICS, _sse_event


class TestSSEEventFormatting:
    """_sse_event() helper formatting."""

    def test_simple_string(self):
        result = _sse_event("message", "hello")
        assert "event: message" in result
        assert "data: hello" in result
        assert result.endswith("\n\n")

    def test_event_name_included(self):
        result = _sse_event("progress", "working")
        assert result.startswith("event: progress\n")

    def test_dict_data_is_json_serialized(self):
        data = {"type": "progress", "step": 1}
        result = _sse_event("update", data)
        lines = result.strip().split("\n")
        assert lines[0] == "event: update"
        parsed = json.loads(lines[1].replace("data: ", ""))
        assert parsed == data

    def test_multi_line_data(self):
        data = "line1\nline2\nline3"
        result = _sse_event("message", data)
        lines = result.strip().split("\n")
        assert "data: line1" in lines
        assert "data: line2" in lines
        assert "data: line3" in lines

    def test_trailing_newline(self):
        """SSE events should end with double newline."""
        result = _sse_event("test", "data")
        assert result.endswith("\n\n")

    def test_empty_data(self):
        result = _sse_event("test", "")
        assert "data: " in result

    def test_numeric_data(self):
        result = _sse_event("count", 42)
        assert "data: 42" in result


class TestSSEParsing:
    """Round-trip: format then parse SSE events."""

    def _parse_events(self, raw: str) -> list[dict]:
        """Simple SSE parser for testing."""
        events = []
        for block in raw.strip().split("\n\n"):
            if not block.strip():
                continue
            event_name = "message"
            data = ""
            for line in block.split("\n"):
                if line.startswith("event:"):
                    event_name = line[6:].strip()
                elif line.startswith("data:"):
                    data_str = line[5:].strip()
                    try:
                        data = json.loads(data_str)
                    except (json.JSONDecodeError, TypeError):
                        data = data_str
            events.append({"event": event_name, "data": data})
        return events

    def test_round_trip_string(self):
        raw = _sse_event("test", "hello world")
        events = self._parse_events(raw)
        assert len(events) == 1
        assert events[0]["event"] == "test"
        assert events[0]["data"] == "hello world"

    def test_round_trip_json(self):
        payload = {"step": 1, "total": 5, "status": "running"}
        raw = _sse_event("progress", payload)
        events = self._parse_events(raw)
        assert len(events) == 1
        assert events[0]["event"] == "progress"
        assert events[0]["data"] == payload

    def test_multiple_events(self):
        raw = (
            _sse_event("start", "beginning")
            + _sse_event("progress", {"pct": 50})
            + _sse_event("done", "finished")
        )
        events = self._parse_events(raw)
        assert len(events) == 3
        assert events[0]["event"] == "start"
        assert events[2]["event"] == "done"


class TestPredictStreamFormat:
    """Predict SSE stream sequence validation."""

    def test_progress_event_format(self):
        progress_data = {"step": 1, "total": 5, "message": "Analyzing news..."}
        raw = _sse_event("progress", progress_data)
        assert "event: progress" in raw
        assert '"step": 1' in raw
        assert '"total": 5' in raw

    def test_debate_event_format(self):
        debate_data = {"speaker": "bull", "round": 1, "content": "Bullish case for AAPL"}
        raw = _sse_event("debate", debate_data)
        assert "event: debate" in raw
        assert '"speaker": "bull"' in raw
        assert '"content": "Bullish case for AAPL"' in raw

    def test_done_event_format(self):
        done_data = {"type": "done", "advisory": "Final result"}
        raw = _sse_event("done", done_data)
        assert "event: done" in raw
        assert '"type": "done"' in raw

    def test_full_predict_stream_sequence(self):
        """Simulate the full predict stream sequence."""
        events = [
            _sse_event("progress", {"step": 1, "total": 5}),
            _sse_event("progress", {"step": 2, "total": 5}),
            _sse_event("debate", {"speaker": "bull", "round": 1, "content": "Bull case"}),
            _sse_event("debate", {"speaker": "bear", "round": 1, "content": "Bear case"}),
            _sse_event("done", {"type": "done", "advisory": "BUY"}),
        ]
        stream = "".join(events)
        assert stream.count("event: progress") == 2
        assert stream.count("event: debate") == 2
        assert stream.count("event: done") == 1

    def test_predict_stream_no_truncation(self):
        """Verify long debate content doesn't break SSE format."""
        long_content = "word " * 1000
        raw = _sse_event("debate", {"speaker": "bull", "round": 1, "content": long_content})
        assert raw.count("\n") > 1
        assert "event: debate" in raw


class TestChatStreamFormat:
    """Chat SSE stream sequence validation."""

    def test_chunk_event_format(self):
        chunk_data = {"type": "chunk", "content": "Apple"}
        raw = _sse_event("chunk", chunk_data)
        assert "event: chunk" in raw
        assert '"content": "Apple"' in raw

    def test_chat_done_event_format(self):
        done_data = {"type": "done"}
        raw = _sse_event("done", done_data)
        assert "event: done" in raw

    def test_full_chat_stream_sequence(self):
        events = [
            _sse_event("chunk", {"type": "chunk", "content": "Apple"}),
            _sse_event("chunk", {"type": "chunk", "content": " is"}),
            _sse_event("chunk", {"type": "chunk", "content": " a"}),
            _sse_event("chunk", {"type": "chunk", "content": " good"}),
            _sse_event("chunk", {"type": "chunk", "content": " company"}),
            _sse_event("done", {"type": "done"}),
        ]
        stream = "".join(events)
        assert stream.count("event: chunk") == 5
        assert stream.count("event: done") == 1


class TestConcurrentStreams:
    """Multiple concurrent SSE streams."""

    def test_interleaved_streams_parse_independently(self):
        """Interleave two streams and verify each parses correctly."""
        stream_a_parts = [
            _sse_event("progress", {"id": "A", "step": 1}),
            _sse_event("progress", {"id": "A", "step": 2}),
            _sse_event("done", {"id": "A"}),
        ]
        stream_b_parts = [
            _sse_event("progress", {"id": "B", "step": 1}),
            _sse_event("done", {"id": "B"}),
        ]

        # Interleave
        combined = ""
        for i in range(max(len(stream_a_parts), len(stream_b_parts))):
            if i < len(stream_a_parts):
                combined += stream_a_parts[i]
            if i < len(stream_b_parts):
                combined += stream_b_parts[i]

        # Both "done" events should be present
        assert combined.count("event: done") == 2
        assert '"id": "A"' in combined
        assert '"id": "B"' in combined


class TestEdgeCases:
    """Edge cases in SSE formatting."""

    def test_special_characters_in_data(self):
        data = {"text": "new\nline", "price": "$100.50"}
        raw = _sse_event("data", data)
        # The newline inside JSON should be escaped as \\n not actual newline
        assert 'new\\nline' in raw or 'new\n' not in raw.split("data: ")[1].split("\\n")[0]

    def test_unicode_in_data(self):
        data = {"emoji": "🚀📈"}
        raw = _sse_event("data", data)
        assert "🚀" in raw or "\\ud83d\\ude80" in raw

    def test_none_data_conversion(self):
        """None should be serialized as JSON null."""
        raw = _sse_event("test", None)
        assert "data: null" in raw

    def test_very_large_data_block(self):
        """Large data blocks should still be valid SSE."""
        large_list = list(range(1000))
        raw = _sse_event("data", large_list)
        assert raw.count("\n") > 1
        assert raw.startswith("event: data")


class TestBrainStreamTopics:
    """
    The dashboard's live-update contract.

    A topic published in the worker but missing from this tuple is delivered to
    nobody, and nothing anywhere else fails — which is how macro_themes stayed
    orphaned. Asserting the membership is the only cheap guard against it.
    """

    def test_macro_themes_is_subscribed(self):
        assert "macro_themes" in BRAIN_STREAM_TOPICS

    def test_classification_status_is_subscribed(self):
        """Published at the end of every classify_backlog run."""
        assert "classification_status" in BRAIN_STREAM_TOPICS

    def test_alert_is_subscribed(self):
        """Published by the market scanner and by breaking-news alerts.

        Without it the AlertsCard only ever shows what its own page-load fetch
        returned, and an alert that fired while the tab was open never appears.
        """
        assert "alert" in BRAIN_STREAM_TOPICS

    def test_weekly_tip_is_subscribed(self):
        assert "weekly_tip" in BRAIN_STREAM_TOPICS

    def test_no_duplicate_topics(self):
        assert len(BRAIN_STREAM_TOPICS) == len(set(BRAIN_STREAM_TOPICS))

    def test_existing_topics_are_all_still_present(self):
        expected = {
            "pipeline_status", "new_articles", "sector_heatmap",
            "rotation_signal", "ipo_alert", "trend_forecast",
            "hot_tickers", "market_ticker", "sentiment_distribution",
            "embedding_status", "events_updated", "thesis_update",
        }
        assert expected <= set(BRAIN_STREAM_TOPICS)


# ── /api/predict/{ticker}/stream: who owns the debate ───────────────────────

ADVISORY = {
    "debate_history": [
        "Bull: Margins are expanding faster than guidance.",
        "Bear: The multiple already prices that in.",
    ],
    "final_advisory": "Hold into the print.",
    "trader_direction": "HOLD",
    "trader_conviction": "Medium",
    "ml_prediction": {"predicted_direction": "UP", "confidence": 0.61},
    "executive_summary": "Balanced risk into earnings.",
}

VERDICT_KEYS = {
    "ticker", "predicted_direction", "confidence", "advisory_direction",
    "advisory_conviction", "final_advisory", "bull_report", "bear_report",
    "ml_prediction", "debate_history", "executive_summary",
}


class _Request:
    """Just enough of a Starlette Request for predict_stream."""

    def __init__(self, db):
        self.app = SimpleNamespace(state=SimpleNamespace(db=db))

    async def is_disconnected(self) -> bool:
        return False


def _db(cached=None):
    db = MagicMock()
    db.get_cached_advisory.return_value = cached
    return db


def _parse(raw: str) -> tuple[str, str]:
    """One chunk from the stream is exactly one `_sse_event`."""
    lines = raw.rstrip("\n").split("\n")
    event = lines[0][len("event: "):]
    data = "\n".join(line[len("data: "):] for line in lines[1:])
    return event, data


async def _drain(response) -> list[tuple[str, str]]:
    return [_parse(raw) async for raw in response.body_iterator]


@pytest.fixture
def debate(monkeypatch):
    """
    StockPredictor replaced by a debate that runs until `release` is set.

    It writes the cache the way the real predict_with_agents does, just before
    returning, so a test can tell a debate that finished from one that was
    abandoned.
    """
    gate = SimpleNamespace(release=asyncio.Event(), started=[], error=None)

    class FakePredictor:
        def __init__(self, db):
            self.db = db

        async def predict_with_agents(self, ticker, **callbacks):
            gate.started.append(ticker)
            await callbacks["progress_callback"]("Aggregating data...")
            await gate.release.wait()
            if gate.error is not None:
                raise gate.error
            self.db.set_cached_advisory(ticker, "2026-09-13", json.dumps(ADVISORY))
            return dict(ADVISORY)

    monkeypatch.setattr("api.server.StockPredictor", FakePredictor)
    server._running_debates.clear()
    yield gate
    server._running_debates.clear()


class TestDebateStreamOwnership:
    """
    A debate is up to five paid model turns. It used to be cancelled with the
    stream that started it, so a page refresh discarded everything already paid
    for and cached nothing, and two tabs on one ticker each ran their own.
    """

    async def test_disconnect_does_not_cancel_the_debate(self, debate):
        db = _db()
        response = await server.predict_stream(_Request(db), "nvda", refresh=True)
        stream = response.body_iterator

        assert _parse(await anext(stream)) == ("agent_update", "Aggregating data...")
        running = server._running_debates["NVDA"]

        # The page is refreshed mid-debate: the server stops iterating the
        # stream, which runs its cleanup.
        await stream.aclose()
        await asyncio.sleep(0.01)
        assert not running.done()

        debate.release.set()
        assert await asyncio.wait_for(running, timeout=5) == ADVISORY
        db.set_cached_advisory.assert_called_once()

        await asyncio.sleep(0)
        assert "NVDA" not in server._running_debates

    @pytest.mark.parametrize("refresh", [False, True])
    async def test_concurrent_request_attaches_instead_of_starting_another(
        self, debate, refresh
    ):
        # An older advisory is cached, so a refresh=False request that ignored
        # the running debate would visibly serve the stale one.
        db = _db(cached={**ADVISORY, "final_advisory": "stale", "_cache_date": "2026-09-10"})

        first = await server.predict_stream(_Request(db), "NVDA", refresh=True)
        first_drain = asyncio.create_task(_drain(first))
        second = await server.predict_stream(_Request(db), "NVDA", refresh=refresh)
        second_drain = asyncio.create_task(_drain(second))

        await asyncio.sleep(0.01)
        debate.release.set()
        first_events = await asyncio.wait_for(first_drain, timeout=5)
        second_events = await asyncio.wait_for(second_drain, timeout=5)

        assert debate.started == ["NVDA"]
        db.get_cached_advisory.assert_not_called()

        event, data = second_events[0]
        assert event == "agent_update"
        assert "already running" in data

        for events in (first_events, second_events):
            verdict = json.loads(next(d for e, d in events if e == "verdict"))
            assert set(verdict) == VERDICT_KEYS
            assert verdict["final_advisory"] == "Hold into the print."
            assert verdict["advisory_direction"] == "HOLD"
            assert events[-1] == ("done", "")

        # The attached client gets the finished debate through the replay path.
        assert any(e == "debate_chunk" for e, _ in second_events)
        await asyncio.sleep(0)
        assert "NVDA" not in server._running_debates

    async def test_failed_debate_reports_the_error_and_leaves_the_registry(self, debate):
        debate.error = RuntimeError("upstream 502")
        db = _db()
        response = await server.predict_stream(_Request(db), "NVDA", refresh=True)
        drain = asyncio.create_task(_drain(response))

        await asyncio.sleep(0.01)
        assert "NVDA" in server._running_debates

        debate.release.set()
        events = await asyncio.wait_for(drain, timeout=5)

        assert ("error", "upstream 502") in events
        db.set_cached_advisory.assert_not_called()
        await asyncio.sleep(0)
        assert "NVDA" not in server._running_debates

    async def test_a_finished_debate_does_not_block_the_next_one(self, debate):
        db = _db()
        debate.release.set()
        first = await server.predict_stream(_Request(db), "NVDA", refresh=True)
        await asyncio.wait_for(_drain(first), timeout=5)
        await asyncio.sleep(0)

        second = await server.predict_stream(_Request(db), "NVDA", refresh=True)
        await asyncio.wait_for(_drain(second), timeout=5)

        assert debate.started == ["NVDA", "NVDA"]
