"""Unit tests for NormalizedResponse / NormalizedToolCall / NormalizedUsage."""

from __future__ import annotations

import pytest

from qed_swe_bench.runner.llm.base import (
    NormalizedResponse,
    NormalizedToolCall,
    NormalizedUsage,
)


def test_tool_call_is_immutable() -> None:
    tc = NormalizedToolCall(id="toolu_1", name="setup", arguments={})
    with pytest.raises(Exception):  # noqa: B017
        tc.id = "other"  # type: ignore[misc]


def test_usage_defaults() -> None:
    u = NormalizedUsage()
    assert u.input_tokens == 0
    assert u.cache_read_tokens == 0


def test_response_holds_tool_calls() -> None:
    resp = NormalizedResponse(
        text=None,
        tool_calls=(
            NormalizedToolCall(id="toolu_1", name="setup", arguments={}),
            NormalizedToolCall(id="toolu_2", name="exec", arguments={"cmd": "ls"}),
        ),
        usage=NormalizedUsage(input_tokens=10, output_tokens=5),
        stop_reason="tool_use",
        model="claude-sonnet-4-5",
    )
    assert len(resp.tool_calls) == 2
    assert resp.tool_calls[0].name == "setup"
    assert resp.tool_calls[1].arguments["cmd"] == "ls"
    assert resp.text is None


def test_response_text_only() -> None:
    resp = NormalizedResponse(
        text="hello",
        tool_calls=(),
        usage=NormalizedUsage(),
        stop_reason="end_turn",
        model="x",
    )
    assert resp.text == "hello"
    assert resp.tool_calls == ()
