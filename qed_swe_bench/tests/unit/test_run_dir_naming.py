"""Tests for the run-dir naming helpers (timestamp-prefixed names + the
new layered layout per D-10)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from qed_swe_bench.runner.run_dir import (
    construct_run_dir_name,
    construct_run_dir_path,
    host,
    parse_run_id_from_dir_name,
)


def test_construct_run_dir_name_uses_filesystem_safe_iso() -> None:
    ts = datetime(2026, 5, 1, 19, 13, 43, tzinfo=UTC)
    name = construct_run_dir_name("198daa6c946049f4", ts=ts)
    assert name == "2026-05-01T19-13-43Z__198daa6c946049f4"
    # No `:` (Windows-hostile) and no `/` (path-segment-hostile).
    assert ":" not in name
    assert "/" not in name


def test_construct_run_dir_name_sorts_chronologically() -> None:
    """Lexicographic sort over names == chronological sort. Pin this so a
    future format change doesn't quietly break the `ls runs/` UX."""
    earlier = construct_run_dir_name("aaaa", ts=datetime(2026, 5, 1, 10, 0, 0, tzinfo=UTC))
    later = construct_run_dir_name("zzzz", ts=datetime(2026, 5, 1, 11, 0, 0, tzinfo=UTC))
    assert sorted([later, earlier]) == [earlier, later]


def test_parse_run_id_strips_iso_prefix() -> None:
    """New-shape (post-2026-05-01) dir names: extract the run_id from the
    trailing segment after `__`."""
    assert parse_run_id_from_dir_name("2026-05-01T19-13-43Z__198daa6c946049f4") == "198daa6c946049f4"


def test_parse_run_id_handles_bare_run_id_legacy() -> None:
    """Older runs were bare `<run_id>` directories. The parser returns the
    name verbatim — equivalent to the run_id."""
    assert parse_run_id_from_dir_name("198daa6c946049f4") == "198daa6c946049f4"


def test_parse_run_id_round_trip_with_construct() -> None:
    """A name produced by construct_run_dir_name parses back to the same
    run_id. Caller-side invariant for the audit module's
    `RunContext.load(run_dir)` and `reproduce_run(run_dir)` paths."""
    rid = "abcdef0123456789"
    name = construct_run_dir_name(rid)
    assert parse_run_id_from_dir_name(name) == rid


def test_parse_run_id_picks_last_segment_if_multiple_separators() -> None:
    """Defensive: even with multiple `__` (e.g. someone manually moved a
    run dir into a subdir), only the trailing segment is the run_id."""
    assert parse_run_id_from_dir_name("foo__bar__abcd1234") == "abcd1234"


# ---------------- D-10 layered layout: construct_run_dir_path + host ----------------


def test_construct_run_dir_path_full_shape() -> None:
    """`runs/<benchmark_id>/<host>/<datetime>/<run_id>/` per D-10."""
    ts = datetime(2026, 5, 2, 18, 30, 0, tzinfo=UTC)
    p = construct_run_dir_path(
        Path("/data/runs"),
        benchmark_id="v8",
        run_id="abcdef1234567890",
        ts=ts,
        host_override="ec2-host-3",
    )
    assert p == Path("/data/runs/v8/ec2-host-3/2026-05-02T18-30-00Z/abcdef1234567890")


def test_construct_run_dir_path_uses_default_host(monkeypatch) -> None:
    """When no host_override given, `host()` resolves the host."""
    monkeypatch.setenv("QED_SWE_BENCH_HOST", "test-bench")
    ts = datetime(2026, 5, 2, 18, 30, 0, tzinfo=UTC)
    p = construct_run_dir_path(
        Path("/r"), benchmark_id="b", run_id="rid", ts=ts,
    )
    assert "test-bench" in str(p)


def test_host_env_override_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QED_SWE_BENCH_HOST", "ci-runner-7")
    assert host() == "ci-runner-7"


def test_host_falls_back_to_socket(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QED_SWE_BENCH_HOST", raising=False)
    h = host()
    assert h  # non-empty
    assert h == h.lower()  # always lowercased


def test_host_sanitizes_unsafe_chars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Filesystem-safe: replace `/`, ` `, `:`, `\\` with `-`."""
    monkeypatch.setenv("QED_SWE_BENCH_HOST", "weird host/with:slashes")
    h = host()
    assert "/" not in h
    assert " " not in h
    assert ":" not in h
    assert "weird-host-with-slashes" == h
