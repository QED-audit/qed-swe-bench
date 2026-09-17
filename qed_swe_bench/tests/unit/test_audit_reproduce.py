"""Tests for the reproduction path: comparison logic + end-to-end replay.

The async MCP-driven `reproduce_run` path uses a stub session so we
don't depend on docker. The pure `ComparisonResult.compare` path is
tested directly. Tool-call replay itself is covered in test_replay.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from qed_swe_bench.audit.reproduce import ComparisonResult, reproduce_run


# ---------------- ComparisonResult.compare ----------------


def test_compare_matches_when_true_sets_equal() -> None:
    c = ComparisonResult.compare(
        grade_call_index=0,
        poc_path="/p.js",
        recorded={"cov_func": True, "diff": False},
        reproduced={"cov_func": True, "diff": False},
    )
    assert c.matches is True
    assert c.only_recorded == ()
    assert c.only_reproduced == ()


def test_compare_diverges_when_caps_differ() -> None:
    c = ComparisonResult.compare(
        grade_call_index=1,
        poc_path="/p.js",
        recorded={"cov_func": True, "diff": True},
        reproduced={"cov_func": True},
    )
    assert c.matches is False
    assert c.only_recorded == ("diff",)
    assert c.only_reproduced == ()


def test_compare_flags_extra_caps_in_reproduction() -> None:
    """Reproduction sometimes earns MORE than the original (e.g. a
    capability that was flaky and missed the first round). Still a
    divergence — the operator should know."""
    c = ComparisonResult.compare(
        grade_call_index=0,
        poc_path="/p.js",
        recorded={"cov_func": True},
        reproduced={"cov_func": True, "asan": True},
    )
    assert c.matches is False
    assert c.only_recorded == ()
    assert c.only_reproduced == ("asan",)


def test_compare_treats_false_caps_as_absent() -> None:
    """The capability bitmap can record `cap: false` for caps the
    grader checked but didn't fire. Comparison is on the True-set,
    not dict-equality, so {a: True, b: False} matches {a: True}."""
    c1 = ComparisonResult.compare(
        grade_call_index=0,
        poc_path="/p.js",
        recorded={"a": True, "b": False},
        reproduced={"a": True},
    )
    assert c1.matches is True

    c2 = ComparisonResult.compare(
        grade_call_index=0,
        poc_path="/p.js",
        recorded={"a": True, "b": False},
        reproduced={"a": True, "b": True},
    )
    assert c2.matches is False
    assert c2.only_reproduced == ("b",)


def test_compare_records_error_state() -> None:
    c = ComparisonResult.compare(
        grade_call_index=0,
        poc_path="/p.js",
        recorded={"cov_func": True},
        reproduced=None,
        error="docker run failed: image not found",
    )
    assert c.matches is False
    assert c.error == "docker run failed: image not found"
    assert c.reproduced == {}


# ---------------- reproduce_run end-to-end (stubbed MCP) ----------------


class _StubResult:
    def __init__(
        self, *, is_error: bool = False, text: str = "",
        structured: dict | None = None,
    ) -> None:
        self.is_error = is_error
        self.text = text
        self.structured = structured


class _ReplaySession:
    """Stub MCP session that records every call and returns canned
    grade results in order. Mimics the real flow: replay walks every
    tool_call entry, including non-grade ones."""

    def __init__(self, grade_results: list[dict[str, Any]]) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._grade_results = list(grade_results)

    async def call_tool(self, name: str, args: dict[str, Any]):
        self.calls.append((name, dict(args)))
        if name == "grade":
            if not self._grade_results:
                return _StubResult(is_error=True, text="no canned result")
            return _StubResult(structured=self._grade_results.pop(0))
        return _StubResult(text=f"{name} ok")


def _patch_mcp(monkeypatch: pytest.MonkeyPatch, stub: _ReplaySession) -> None:
    from contextlib import asynccontextmanager
    import qed_swe_bench.runner.mcp_client as mcp_mod

    class _FakeCls:
        @staticmethod
        @asynccontextmanager
        async def start(image_ref, **kwargs):
            yield stub

    monkeypatch.setattr(mcp_mod, "McpDockerSession", _FakeCls)


@pytest.fixture
def fake_run_dir(tmp_path: Path) -> Path:
    """Synthetic run dir: setup → write_file → grade."""
    (tmp_path / "job.json").write_text(json.dumps({
        "image_digest": "sha256:fake_image",
        "image_ref": "fake.example.com/img:tag",
    }))
    (tmp_path / "tool_calls.jsonl").write_text("\n".join(json.dumps(c) for c in [
        {"tool": "setup", "args": {}, "result": "ok"},
        {"tool": "write_file", "args": {"path": "/rlenv/workspace/poc.js",
                                          "contents": "let a = 1;"}},
        {"tool": "grade", "args": {"path": "/rlenv/workspace/poc.js"}},
    ]) + "\n")
    (tmp_path / "grade_calls.jsonl").write_text(json.dumps({
        "path": "/rlenv/workspace/poc.js",
        "result": {"capabilities": {"cov_func": True, "diff": True}},
    }) + "\n")
    return tmp_path


@pytest.mark.asyncio
async def test_reproduce_run_matches_when_replay_yields_same_caps(
    fake_run_dir: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _ReplaySession([{"capabilities": {"cov_func": True, "diff": True}}])
    _patch_mcp(monkeypatch, stub)
    report = await reproduce_run(fake_run_dir)

    assert report.poc_count == 1
    assert report.overall_match is True
    assert len(report.comparisons) == 1
    assert report.comparisons[0].matches is True
    # The replay forwards every tool call including setup and write_file.
    assert [name for name, _ in stub.calls] == ["setup", "write_file", "grade"]


@pytest.mark.asyncio
async def test_reproduce_run_flags_divergent_capabilities(
    fake_run_dir: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _ReplaySession([{"capabilities": {"cov_func": True}}])  # missing 'diff'
    _patch_mcp(monkeypatch, stub)

    report = await reproduce_run(fake_run_dir)
    assert report.overall_match is False
    c = report.comparisons[0]
    assert c.matches is False
    assert c.only_recorded == ("diff",)
    assert c.only_reproduced == ()


@pytest.mark.asyncio
async def test_reproduce_run_handles_exec_pipeline_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole point of full-replay: `sed`/`awk` exec pipelines that
    the prior heredoc-extractor couldn't parse just work — args go to
    MCP verbatim."""
    (tmp_path / "job.json").write_text(json.dumps({"image_digest": "sha256:x"}))
    (tmp_path / "tool_calls.jsonl").write_text("\n".join(json.dumps(c) for c in [
        {"tool": "exec", "args": {"cmd": "echo 'let a=1' > /tmp/p.js"}},
        {"tool": "exec", "args": {"cmd": "sed -i 's/=1/=2/' /tmp/p.js"}},
        {"tool": "grade", "args": {"path": "/tmp/p.js"}},
    ]) + "\n")
    (tmp_path / "grade_calls.jsonl").write_text(json.dumps({
        "path": "/tmp/p.js",
        "result": {"capabilities": {"cov_func": True}},
    }) + "\n")

    stub = _ReplaySession([{"capabilities": {"cov_func": True}}])
    _patch_mcp(monkeypatch, stub)

    report = await reproduce_run(tmp_path)
    assert report.overall_match is True
    # Both exec calls + grade got replayed, args verbatim.
    assert [name for name, _ in stub.calls] == ["exec", "exec", "grade"]
    assert stub.calls[1][1] == {"cmd": "sed -i 's/=1/=2/' /tmp/p.js"}


@pytest.mark.asyncio
async def test_reproduce_run_with_no_grade_returns_vacuously_true(
    tmp_path: Path,
) -> None:
    """A run with no grade() calls (e.g., aborted early) reproduces
    successfully — there's nothing to disprove."""
    (tmp_path / "job.json").write_text(json.dumps({"image_digest": "sha256:x"}))
    (tmp_path / "tool_calls.jsonl").write_text(
        json.dumps({"tool": "setup", "args": {}}) + "\n"
    )
    report = await reproduce_run(tmp_path)
    assert report.poc_count == 0
    assert report.overall_match is True
    assert report.comparisons == ()


@pytest.mark.asyncio
async def test_reproduce_run_handles_multiple_grades(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two grade() calls in one run — each compared to its corresponding
    grade_calls.jsonl entry by ordinal."""
    (tmp_path / "job.json").write_text(json.dumps({"image_digest": "sha256:x"}))
    (tmp_path / "tool_calls.jsonl").write_text("\n".join(json.dumps(c) for c in [
        {"tool": "write_file", "args": {"path": "/a.js", "contents": "A"}},
        {"tool": "grade", "args": {"path": "/a.js"}},
        {"tool": "write_file", "args": {"path": "/b.js", "contents": "B"}},
        {"tool": "grade", "args": {"path": "/b.js"}},
    ]) + "\n")
    (tmp_path / "grade_calls.jsonl").write_text("\n".join(json.dumps(c) for c in [
        {"path": "/a.js", "result": {"capabilities": {"cov_func": True}}},
        {"path": "/b.js", "result": {"capabilities": {"crash": True}}},
    ]) + "\n")

    stub = _ReplaySession([
        {"capabilities": {"cov_func": True}},
        {"capabilities": {"crash": True}},
    ])
    _patch_mcp(monkeypatch, stub)

    report = await reproduce_run(tmp_path)
    assert report.poc_count == 2
    assert report.overall_match is True
    assert report.comparisons[0].poc_path == "/a.js"
    assert report.comparisons[1].poc_path == "/b.js"


@pytest.mark.asyncio
async def test_reproduce_run_missing_job_json_returns_error(tmp_path: Path) -> None:
    report = await reproduce_run(tmp_path)
    assert report.error is not None
    assert "job.json" in report.error


@pytest.mark.asyncio
async def test_reproduce_run_records_grade_replay_failure_as_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the grade replay itself raises (MCP error, container died),
    the comparison records error rather than misclassifying as a
    capability mismatch."""
    (tmp_path / "job.json").write_text(json.dumps({"image_digest": "sha256:x"}))
    (tmp_path / "tool_calls.jsonl").write_text("\n".join(json.dumps(c) for c in [
        {"tool": "write_file", "args": {"path": "/p.js", "contents": "x"}},
        {"tool": "grade", "args": {"path": "/p.js"}},
    ]) + "\n")
    (tmp_path / "grade_calls.jsonl").write_text(json.dumps({
        "path": "/p.js",
        "result": {"capabilities": {"cov_func": True}},
    }) + "\n")

    class _FailingSession:
        async def call_tool(self, name: str, args: dict[str, Any]):
            if name == "grade":
                raise RuntimeError("mcp died mid-grade")
            return _StubResult(text="ok")

    _patch_mcp(monkeypatch, _FailingSession())

    report = await reproduce_run(tmp_path)
    assert report.overall_match is False
    c = report.comparisons[0]
    assert c.error is not None
    assert "mcp died mid-grade" in c.error
