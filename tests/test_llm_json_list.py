"""
Tests for `parse_json_list` — the json_mode array contract.

`response_format={"type": "json_object"}` forces an object at the root, so
every prompt in this codebase that asks for a bare JSON array can get one back
wrapped under a key the model invents. These pin the tolerance down.
"""

import json
import pytest

from config.llm import parse_json_list


class TestParseJsonList:

    def test_bare_array_passes_through(self):
        assert parse_json_list('[{"id": "a1"}, {"id": "a2"}]') == [
            {"id": "a1"}, {"id": "a2"}
        ]

    @pytest.mark.parametrize("key", ["events", "results", "rankings", "articles"])
    def test_unwraps_whatever_key_the_model_picked(self, key):
        """The wrapper key is the model's choice, so nothing may depend on it."""
        payload = json.dumps({key: [{"id": "a1", "importance_score": 8.5}]})
        assert parse_json_list(payload) == [{"id": "a1", "importance_score": 8.5}]

    def test_unwraps_alongside_scalar_siblings(self):
        """A lone list is still unambiguous next to non-list metadata."""
        payload = json.dumps({"model": "x", "count": 1, "events": [{"id": "a1"}]})
        assert parse_json_list(payload) == [{"id": "a1"}]

    def test_strips_code_fence(self):
        assert parse_json_list('```json\n[{"id": "a1"}]\n```') == [{"id": "a1"}]

    def test_fenced_and_wrapped(self):
        assert parse_json_list('```json\n{"events": [{"id": "a1"}]}\n```') == [
            {"id": "a1"}
        ]

    def test_empty_array_is_a_valid_result(self):
        assert parse_json_list("[]") == []
        assert parse_json_list('{"events": []}') == []

    def test_two_lists_is_ambiguous_and_raises(self):
        """Guessing here would attach the wrong scores to the wrong articles."""
        payload = json.dumps({"events": [{"id": "a1"}], "errors": [{"id": "a2"}]})
        with pytest.raises(ValueError):
            parse_json_list(payload)

    def test_object_with_no_list_raises(self):
        with pytest.raises(ValueError):
            parse_json_list('{"id": "a1", "importance_score": 8.5}')

    def test_scalar_root_raises(self):
        with pytest.raises(ValueError):
            parse_json_list('"just a string"')

    def test_unparseable_raises(self):
        with pytest.raises(Exception):
            parse_json_list("not json at all")

    def test_newline_delimited_objects(self):
        """"One result per article" gets read as one object per line."""
        payload = '{"id": "a1"}\n{"id": "a2"}\n{"id": "a3"}'
        assert parse_json_list(payload) == [
            {"id": "a1"}, {"id": "a2"}, {"id": "a3"}
        ]

    def test_concatenated_objects_without_newlines(self):
        assert parse_json_list('{"id": "a1"}{"id": "a2"}') == [
            {"id": "a1"}, {"id": "a2"}
        ]

    def test_truncated_tail_keeps_the_whole_values_before_it(self):
        """A response cut off by the token cap still carries usable results."""
        payload = '{"id": "a1"}\n{"id": "a2"}\n{"id": "a3", "summary": "cut off'
        assert parse_json_list(payload) == [{"id": "a1"}, {"id": "a2"}]

    def test_single_truncated_object_still_raises(self):
        """One unusable value must not masquerade as a one-item result."""
        with pytest.raises(Exception):
            parse_json_list('{"id": "a1", "summary": "cut off')
