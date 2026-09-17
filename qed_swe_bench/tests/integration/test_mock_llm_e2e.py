"""End-to-end smoke: --mock-llm against the local sample-stack-bof image.

Skipped unless docker is available AND the local sample image exists. Build it
first with `make smoke` (or scripts/build_sample_env.sh).

Marked `slow` because it spawns docker; opt in with `pytest -m slow`.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SAMPLE_IMAGE = "local/sample-stack-bof:latest"


def _have_docker() -> bool:
    if not shutil.which("docker"):
        return False
    return subprocess.run(["docker", "info"], capture_output=True).returncode == 0


def _have_sample_image() -> bool:
    return (
        subprocess.run(
            ["docker", "image", "inspect", SAMPLE_IMAGE], capture_output=True
        ).returncode
        == 0
    )


@pytest.mark.slow
@pytest.mark.skipif(not _have_docker(), reason="docker not available")
@pytest.mark.skipif(not _have_sample_image(), reason=f"image not built: {SAMPLE_IMAGE}; run scripts/build_sample_env.sh")
def test_mock_llm_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QED_SWE_BENCH_DB", str(tmp_path / "test.sqlite"))
    monkeypatch.setenv("QED_SWE_BENCH_RUNS_DIR", str(tmp_path / "runs"))

    # Use the installed `qed_swe_bench` script via current interpreter.
    proc = subprocess.run(
        [sys.executable, "-m", "qed_swe_bench.cli", "benchmark", "--mock-llm"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={**os.environ},
        timeout=120,
    )
    assert proc.returncode == 0, f"benchmark failed:\nstdout={proc.stdout}\nstderr={proc.stderr}"
    assert "succeeded" in proc.stdout, proc.stdout

    # Confirm exactly one row landed in the DB and the run dir is populated.
    import sqlite3
    con = sqlite3.connect(tmp_path / "test.sqlite")
    rows = con.execute(
        "SELECT run_id, model, env_id, status, score, capabilities, "
        "cost_usd, cost_source, image_digest FROM runs"
    ).fetchall()
    assert len(rows) == 1
    run_id, model, env_id, status, score, caps, cost_usd, cost_source, digest = rows[0]
    assert status == "succeeded"
    assert env_id == "sample-stack-bof"
    assert model == "anthropic/claude-haiku-4-5"  # mock model id
    assert digest.startswith("sha256:")
    assert cost_source == "mock"
    assert cost_usd == 0.0
    # capabilities is JSON; mock client doesn't call grade(), so empty bitmap.
    assert json.loads(caps) == {}

    # Run-dir files
    run_dirs = list((tmp_path / "runs").rglob(run_id))
    assert run_dirs, "run dir not created"
    rd = run_dirs[0]
    for filename in ("job.json", "score.json", "cost.json", "transcript.jsonl",
                     "tool_calls.jsonl", "grade_calls.jsonl"):
        assert (rd / filename).exists(), f"missing {filename}"

    # Mock client makes 3 tool calls per the default sequence.
    tool_lines = (rd / "tool_calls.jsonl").read_text().strip().splitlines()
    assert len(tool_lines) == 3
    tool_names = [json.loads(line)["tool"] for line in tool_lines]
    assert tool_names == ["setup", "list_directory", "write_file"]
