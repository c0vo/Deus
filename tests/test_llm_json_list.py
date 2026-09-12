"""
Tests for `parse_json_list` — the json_mode array contract.

`response_format={"type": "json_object"}` forces an object at the root, so
every prompt in this codebase that asks for a bare JSON array can get one back
wrapped under a key the model invents. These pin the tolerance down.
"""

import json
from unittest.mock import patch

import pytest

from config.llm import parse_json_list, salvage_json_array


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


class TestSalvageJsonArray:
    """
    Recovery from an envelope whose array is cut off mid-element.

    `parse_json_list` cannot help here: the text is one JSON value that happens
    to be incomplete, so `raw_decode` fails at offset zero and its
    concatenated-objects path finds nothing. The elements inside the array are
    nonetheless whole and paid for.
    """

    def test_recovers_whole_items_before_a_truncation(self):
        payload = (
            '{"items": [{"id": "item_1", "event_type": "earnings"},'
            ' {"id": "item_2", "event_type": "macro"},'
            ' {"id": "item_3", "event_type": "mer'
        )
        assert salvage_json_array(payload, key="items") == [
            {"id": "item_1", "event_type": "earnings"},
            {"id": "item_2", "event_type": "macro"},
        ]

    def test_recovers_a_degenerate_repeated_id(self):
        """The observed failure: the id itself ran away until the token cap."""
        payload = (
            '{"items": [{"id": "item_1", "event_type": "earnings"},'
            ' {"id": "item_2item_2item_2item_2item_2item_2item_2item_2'
        )
        assert salvage_json_array(payload, key="items") == [
            {"id": "item_1", "event_type": "earnings"}
        ]

    def test_complete_array_is_returned_in_full(self):
        payload = '{"items": [{"id": "item_1"}, {"id": "item_2"}]}'
        assert salvage_json_array(payload, key="items") == [
            {"id": "item_1"}, {"id": "item_2"}
        ]

    def test_falls_back_to_the_first_bracket_without_a_key(self):
        assert salvage_json_array('[{"id": "item_1"}, {"id": "item_2') == [
            {"id": "item_1"}
        ]

    def test_code_fenced_envelope_is_handled(self):
        payload = '```json\n{"items": [{"id": "item_1"}, {"id": "it'
        assert salvage_json_array(payload, key="items") == [{"id": "item_1"}]

    def test_nothing_whole_returns_empty(self):
        """Callers treat [] exactly as a parse failure, so it must not guess."""
        assert salvage_json_array('{"items": [{"id": "ite', key="items") == []
        assert salvage_json_array("not json at all", key="items") == []
        assert salvage_json_array("", key="items") == []


class TestShortBatchResponsesStillParse:
    """
    The shapes the batch-classifier regression produced, pinned here.

    `exact_items` and an all-required item schema change what the *request* asks
    for; they must change nothing about how a response that ignores it is read.
    An under-length or field-starved response is a matching problem for the
    caller to count and retry, never a parse error that costs the batch.
    """

    def test_one_item_envelope_parses_as_a_one_item_list(self):
        text = json.dumps(
            {"items": [{"id": "item_1", "sentiment_score": -0.2, "urgency": "medium"}]}
        )
        assert parse_json_list(text) == [
            {"id": "item_1", "sentiment_score": -0.2, "urgency": "medium"}
        ]

    def test_id_only_items_parse_without_complaint(self):
        """Schema-valid and useless is still the parser's job to hand over."""
        text = json.dumps({"items": [{"id": f"item_{i}"} for i in range(1, 11)]})
        assert len(parse_json_list(text)) == 10

    def test_salvage_is_unaffected_by_the_request_side_change(self):
        truncated = (
            '{"items": [{"id": "item_1", "event_type": "earnings"}, '
            '{"id": "item_2", "event_type": "mac'
        )
        assert salvage_json_array(truncated, key="items") == [
            {"id": "item_1", "event_type": "earnings"}
        ]


class TestExactItems:
    """
    `exact_items` — the array length a Python type cannot carry.

    `list[Model]` says "an array of these" and never "exactly ten of these", so
    one object satisfied the batch classifier's schema and one object is what
    came back. It lives in `config/llm.py` because `minItems`/`maxItems` is JSON
    Schema dialect, which no call site is allowed to speak.
    """

    @staticmethod
    def _items(**kwargs):
        from pydantic import BaseModel

        from config.llm import _build_response_format

        class Row(BaseModel):
            name: str = ""

        response_format, enveloped = _build_response_format(
            list[Row], False, kwargs.get("exact_items")
        )
        assert enveloped is True
        return response_format["json_schema"]["schema"]["properties"]["items"]

    def test_pins_both_bounds(self):
        items = self._items(exact_items=10)
        assert items["minItems"] == 10
        assert items["maxItems"] == 10

    def test_omitted_leaves_the_array_unbounded(self):
        items = self._items()
        assert "minItems" not in items
        assert "maxItems" not in items

    @pytest.mark.parametrize("value", [0, -1])
    def test_non_positive_is_ignored_rather_than_emitted(self, value):
        """A zero minimum is not a constraint, and a negative one is invalid."""
        items = self._items(exact_items=value)
        assert "minItems" not in items

    def test_warns_rather_than_silently_dropping_it_on_a_bare_model(self):
        """A caller that asked for a length must not be told it got one."""
        from pydantic import BaseModel

        from config.llm import _json_schema_for

        class Row(BaseModel):
            name: str = ""

        with patch("config.llm.log") as mock_log:
            body, enveloped = _json_schema_for(Row, 5)

        assert enveloped is False
        assert "minItems" not in body
        assert mock_log.warning.call_args.args[0] == "llm.exact_items_ignored"
