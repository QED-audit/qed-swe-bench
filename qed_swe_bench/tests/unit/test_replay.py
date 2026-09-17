"""Replay module: feed recorded tool_calls back through MCP verbatim.

Stub the session so we don't need docker. Coverage focus: the replay's
own state machine, not the MCP layer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from qed_swe_bench.runner.replay import (
    ReplayedCall,
    read_tool_calls_jsonl,
    replay_tool_calls,
)


class _StubResult:
    def __init__(
        self, *, is_error: bool = False, text: str = "",
        structured: dict | None = None,
    ) -> None:
        self.is_error = is_error
        self.text = text
        self.structured = structured


class _RecordingSession:
    """Records every (name, args) it receives; returns canned results."""

    def __init__(
        self, *,
        canned: dict[str, _StubResult] | None = None,
        raise_on: dict[str, Exception] | None = None,
    ) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._canned = canned or {}
        self._raise = raise_on or {}

    async def call_tool(self, name: str, args: dict[str, Any]):
        self.calls.append((name, dict(args)))
        if name in self._raise:
            raise self._raise[name]
        return self._canned.get(name, _StubResult(text=f"{name} ok"))


# ---------------- read_tool_calls_jsonl ----------------


def test_read_tool_calls_jsonl_handles_missing_file(tmp_path: Path) -> None:
    assert read_tool_calls_jsonl(tmp_path / "missing.jsonl") == []


def test_read_tool_calls_jsonl_skips_blank_and_malformed(tmp_path: Path) -> None:
    p = tmp_path / "tc.jsonl"
    p.write_text(
        json.dumps({"tool": "exec", "args": {"cmd": "ls"}}) + "\n"
        "\n"  # blank
        "{this is not json}\n"
        + json.dumps({"tool": "grade", "args": {"path": "/p"}}) + "\n"
    )
    out = read_tool_calls_jsonl(p)
    assert [e["tool"] for e in out] == ["exec", "grade"]


# ---------------- replay_tool_calls ----------------


@pytest.mark.asyncio
async def test_replay_forwards_args_verbatim() -> None:
    """The whole point: regardless of what the args are, they get
    handed to MCP unchanged. No regex, no parsing."""
    sess = _RecordingSession()
    calls = [
        {"tool": "exec", "args": {"cmd": "sed -i 's/a/b/' /p.js"}},
        {"tool": "write_file", "args": {"path": "/q.js", "contents": "..."}},
        {"tool": "grade", "args": {"path": "/p.js"}},
    ]
    out = await replay_tool_calls(sess, calls)
    assert len(out) == 3
    assert sess.calls == [
        ("exec", {"cmd": "sed -i 's/a/b/' /p.js"}),
        ("write_file", {"path": "/q.js", "contents": "..."}),
        ("grade", {"path": "/p.js"}),
    ]


@pytest.mark.asyncio
async def test_replay_skip_bypasses_listed_tools() -> None:
    """`skip={'grade'}` lets resume reconstruct fs state without
    re-running slow grade calls."""
    sess = _RecordingSession()
    calls = [
        {"tool": "exec", "args": {"cmd": "ls"}},
        {"tool": "grade", "args": {"path": "/p.js"}},
        {"tool": "exec", "args": {"cmd": "echo done"}},
    ]
    out = await replay_tool_calls(sess, calls, skip={"grade"})
    assert [c[0] for c in sess.calls] == ["exec", "exec"]
    assert [r.tool for r in out] == ["exec", "grade", "exec"]
    # The skipped entry is recorded but marked skipped + not invoked.
    assert out[1].skipped is True
    assert out[1].is_error is False


@pytest.mark.asyncio
async def test_replay_continues_after_per_call_exception() -> None:
    """If one MCP call raises (e.g. a transient error), we capture it
    and keep going — the remaining sequence may still rebuild the
    state we need."""
    sess = _RecordingSession(raise_on={"grade": RuntimeError("server gone")})
    calls = [
        {"tool": "exec", "args": {"cmd": "x"}},
        {"tool": "grade", "args": {"path": "/p.js"}},
        {"tool": "exec", "args": {"cmd": "y"}},
    ]
    out = await replay_tool_calls(sess, calls)
    assert [r.tool for r in out] == ["exec", "grade", "exec"]
    assert out[1].error is not None
    assert "RuntimeError: server gone" in out[1].error
    # Subsequent call still executed.
    assert sess.calls[-1] == ("exec", {"cmd": "y"})


@pytest.mark.asyncio
async def test_replay_records_recorded_text_for_diffing() -> None:
    """The original stdout/grade-result string from tool_calls.jsonl is
    surfaced on each ReplayedCall so callers can diff it themselves."""
    sess = _RecordingSession()
    calls = [
        {"tool": "exec", "args": {"cmd": "ls"}, "result": "a\nb\nc"},
        {"tool": "exec", "args": {"cmd": "pwd"}},  # no recorded result
    ]
    out = await replay_tool_calls(sess, calls)
    assert out[0].recorded_text == "a\nb\nc"
    assert out[1].recorded_text is None


@pytest.mark.asyncio
async def test_replay_stop_after_index_truncates_sequence() -> None:
    """Audit uses this to halt at a specific grade call after capturing
    its result."""
    sess = _RecordingSession()
    calls = [
        {"tool": "exec", "args": {}},
        {"tool": "grade", "args": {}},
        {"tool": "exec", "args": {}},
    ]
    out = await replay_tool_calls(sess, calls, stop_after_index=1)
    assert len(out) == 2
    assert [c[0] for c in sess.calls] == ["exec", "grade"]


@pytest.mark.asyncio
async def test_replay_skips_malformed_entries() -> None:
    """If tool_calls.jsonl somehow has entries without `tool` or
    `args`, skip rather than crash. Preserves replay over partial /
    corrupted runs."""
    sess = _RecordingSession()
    calls = [
        {"tool": "exec", "args": {"cmd": "ok"}},
        {"args": {"cmd": "missing tool name"}},
        {"tool": "exec"},                       # missing args is OK (defaults to {})
        {"tool": 123, "args": {}},               # non-string tool
    ]
    out = await replay_tool_calls(sess, calls)
    assert [c[0] for c in sess.calls] == ["exec", "exec"]
    assert len(out) == 2


@pytest.mark.asyncio
async def test_replay_propagates_structured_grade_result() -> None:
    """Grade results carry a structured capabilities dict — that's what
    the audit comparison consumes, so it has to flow through."""
    sess = _RecordingSession(canned={
        "grade": _StubResult(structured={"capabilities": {"cov_func": True}}),
    })
    calls = [{"tool": "grade", "args": {"path": "/p.js"}}]
    out = await replay_tool_calls(sess, calls)
    assert out[0].structured == {"capabilities": {"cov_func": True}}
