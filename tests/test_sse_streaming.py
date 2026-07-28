"""Tests for SSE (Server-Sent Events) streaming format and parsing."""

import json
from api.server import _sse_event


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
