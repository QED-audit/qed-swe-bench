"""Per-run provenance helpers — git_sha, env-var fingerprint, reproduce-with cmd.

Captured once at orchestrator startup and threaded through to per-tuple
artifacts so each `runs/<id>/<run_id>/job.json` records exactly which
code SHA, which env-var overrides, and which copy-pasteable single-tuple
invocation produced this run. Recovery from "what did we change between
v8-r1 and v8-r2" relies on these fields existing.

All three are best-effort and may be `None` (no git, no overrides set).
The orchestrator never blocks on a failed capture.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

# Env-var prefixes whose values affect benchmark execution. Kept tight so
# we don't dump the entire environment into job.json (privacy + noise).
_PROVENANCE_PREFIXES: tuple[str, ...] = (
    "QED_SWE_BENCH_",
    "LITELLM_",
)

# Substrings that signal a sensitive value (key, token, password). Any
# var matching one of these is silently dropped from the snapshot, even
# if its prefix is in _PROVENANCE_PREFIXES — defense-in-depth so a future
# `QED_SWE_BENCH_VENDOR_API_KEY` doesn't end up in a committed run-dir.
_SENSITIVE_SUBSTRINGS: tuple[str, ...] = (
    "API_KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL",
)


def git_sha(repo_root: Path | str | None = None) -> str | None:
    """Return current git HEAD sha, or None if not in a git repo / git
    not installed.

    `repo_root` is injectable for testing; defaults to the current
    working directory at call time.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=str(repo_root) if repo_root else None,
            timeout=5,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None
    sha = result.stdout.strip()
    return sha or None


def env_overrides_snapshot(env: dict[str, str] | None = None) -> dict[str, str]:
    """Return QED_SWE_BENCH_* + LITELLM_* env vars affecting this run.

    Sensitive (key/token/password-shaped) keys are dropped even when
    their prefix matches. `env` is injectable for testing; defaults to
    `os.environ`.
    """
    src = env if env is not None else os.environ
    out: dict[str, str] = {}
    for k, v in src.items():
        if not any(k.startswith(p) for p in _PROVENANCE_PREFIXES):
            continue
        if any(s in k for s in _SENSITIVE_SUBSTRINGS):
            continue
        out[k] = v
    return out


def synthesize_repro_cmd(*, run_id: str) -> str:
    """Build the copy-pasteable invocation that re-runs this cell.

    The row's `runs.config_snapshot` column carries the full source
    YAML (verbatim, comments preserved — D-13 in docs/decisions.md),
    so the command is fully self-contained: `qed_swe_bench rerun
    <run_id>` reads the snapshot from the DB, narrows to this row's
    (model, env, seed), and replays.
    """
    return f"qed_swe_bench rerun {shlex.quote(run_id)}"
