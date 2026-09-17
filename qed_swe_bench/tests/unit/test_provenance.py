"""Tests for `runner/provenance.py` — git_sha, env-var snapshot, repro_cmd."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from qed_swe_bench.runner.provenance import (
    env_overrides_snapshot,
    git_sha,
    synthesize_repro_cmd,
)


# ---------------- git_sha ----------------


def test_git_sha_returns_hex_in_a_repo(tmp_path: Path) -> None:
    """In a real git repo, returns a hex sha string."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@e.x"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    (repo / "x").write_text("x")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "init", "--no-verify"], cwd=repo, check=True
    )
    sha = git_sha(repo_root=repo)
    assert sha is not None
    assert len(sha) == 40
    assert all(c in "0123456789abcdef" for c in sha)


def test_git_sha_returns_none_outside_repo(tmp_path: Path) -> None:
    """Pointed at a non-git dir, returns None rather than crashing."""
    assert git_sha(repo_root=tmp_path) is None


# ---------------- env_overrides_snapshot ----------------


def test_env_overrides_picks_up_qed_swe_bench_prefix() -> None:
    snap = env_overrides_snapshot(
        env={
            "QED_SWE_BENCH_RATE_LIMIT_MAX_RETRIES": "10",
            "QED_SWE_BENCH_RUN_REAL_API": "1",
            "PATH": "/usr/bin",  # not picked up
            "HOME": "/x",        # not picked up
        }
    )
    assert snap == {
        "QED_SWE_BENCH_RATE_LIMIT_MAX_RETRIES": "10",
        "QED_SWE_BENCH_RUN_REAL_API": "1",
    }


def test_env_overrides_picks_up_litellm_prefix() -> None:
    snap = env_overrides_snapshot(
        env={
            "LITELLM_REQUEST_TIMEOUT_S": "300",
            "OTHER": "x",
        }
    )
    assert snap == {"LITELLM_REQUEST_TIMEOUT_S": "300"}


def test_env_overrides_drops_sensitive_substrings() -> None:
    """API_KEY / TOKEN / SECRET / PASSWORD substrings are excluded even
    when the prefix would otherwise match."""
    snap = env_overrides_snapshot(
        env={
            "QED_SWE_BENCH_VENDOR_API_KEY": "sk-secret",
            "QED_SWE_BENCH_FOO_TOKEN": "abc",
            "QED_SWE_BENCH_BAR_SECRET": "xyz",
            "QED_SWE_BENCH_BAZ_PASSWORD": "p",
            "QED_SWE_BENCH_QUX_PASSWD": "p",
            "QED_SWE_BENCH_REAL_OVERRIDE": "kept",  # this one stays
        }
    )
    assert snap == {"QED_SWE_BENCH_REAL_OVERRIDE": "kept"}


def test_env_overrides_empty_when_no_matches() -> None:
    """No QED_SWE_BENCH_/LITELLM_ vars → empty dict (not None)."""
    snap = env_overrides_snapshot(env={"PATH": "/x", "HOME": "/y"})
    assert snap == {}


# ---------------- synthesize_repro_cmd ----------------
#
# Per D-13, the repro command is a literal `qed_swe_bench rerun
# <run_id>`. The subcommand reads the row's `runs.config_snapshot`
# (full source YAML, comments preserved) so it doesn't need any
# per-cell arguments.


def test_repro_cmd_is_self_contained_rerun_invocation() -> None:
    cmd = synthesize_repro_cmd(run_id="abc123def4567890")
    assert cmd == "qed_swe_bench rerun abc123def4567890"


def test_repro_cmd_quotes_run_id_with_special_chars() -> None:
    """run_ids should always be 16-hex but defensively quote — if a
    pathological caller ever passes a value with shell metachars, it
    must not let those leak through."""
    cmd = synthesize_repro_cmd(run_id="abc def")
    # shlex.quote wraps in single quotes when it contains spaces
    assert "'abc def'" in cmd
