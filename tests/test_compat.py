"""Tests for compat shims (json repair, arg parsing, helpers)."""

import pytest

from bluellm._compat import (
    PolyfillResult,
    _attempt_json_repair,
    is_reasoning_auto_summary_enabled,
    parse_tool_call_arguments,
)


def test_auto_summary_flag(monkeypatch):
    monkeypatch.delenv("BLUELLM_REASONING_AUTO_SUMMARY", raising=False)
    assert is_reasoning_auto_summary_enabled() is False
    monkeypatch.setenv("BLUELLM_REASONING_AUTO_SUMMARY", "true")
    assert is_reasoning_auto_summary_enabled() is True


def test_polyfill_result_applied_edits_none():
    assert PolyfillResult().applied_edits_for_response() is None


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("", None),  # empty
        ("{\"a\": 1}", None),  # already balanced -> nothing to repair
        ("{\"a\": 1", {"a": 1}),  # truncated object
        ("{\"a\": 1,", {"a": 1}),  # trailing comma stripped
        ("[1, 2", [1, 2]),  # truncated array
        ("{\"a\": [1, 2", {"a": [1, 2]}),  # nested truncation
        ('{"a": "b\\"c"', {"a": 'b"c'}),  # escaped quote inside string
        ("{\"a\":", None),  # unrepairable (dangling colon)
        ("}", None),  # unmatched closer, empty stack
        ("\\[1", None),  # leading backslash outside string -> still unrepairable
    ],
)
def test_attempt_json_repair(raw, expected):
    assert _attempt_json_repair(raw) == expected


def test_parse_tool_call_arguments_valid():
    assert parse_tool_call_arguments('{"x": 1}') == {"x": 1}


def test_parse_tool_call_arguments_empty():
    assert parse_tool_call_arguments("") == {}
    assert parse_tool_call_arguments(None) == {}


def test_parse_tool_call_arguments_repairs_truncation(caplog):
    with caplog.at_level("WARNING"):
        assert parse_tool_call_arguments('{"x": 1', tool_name="Read", context="ctx") == {
            "x": 1
        }
    assert any("Repaired" in r.getMessage() for r in caplog.records)


def test_parse_tool_call_arguments_unrepairable_raises():
    with pytest.raises(ValueError):
        parse_tool_call_arguments('{"x":', tool_name="Read", context="ctx")
