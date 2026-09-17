"""--resume idempotency: running the same config twice doesn't duplicate runs."""

from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SAMPLE_IMAGE = "local/sample-stack-bof:latest"


def _have_docker_image() -> bool:
    if not shutil.which("docker"):
        return False
    return (
        subprocess.run(
            ["docker", "image", "inspect", SAMPLE_IMAGE], capture_output=True
        ).returncode
        == 0
    )


@pytest.mark.slow
@pytest.mark.skipif(not _have_docker_image(), reason=f"image not built: {SAMPLE_IMAGE}")
def test_double_mock_llm_run_keeps_one_row_per_tuple(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Running --mock-llm twice produces exactly the same number of rows
    *plus* the unique constraint blocks duplicates within a benchmark.

    --mock-llm chooses a fresh benchmark_id each time (timestamp), so each
    invocation is its own benchmark. The right invariant is: within a single
    benchmark_id, each (model, env, seed) appears at most once.
    """
    monkeypatch.setenv("QED_SWE_BENCH_DB", str(tmp_path / "test.sqlite"))
    monkeypatch.setenv("QED_SWE_BENCH_RUNS_DIR", str(tmp_path / "runs"))

    for _ in range(2):
        proc = subprocess.run(
            [sys.executable, "-m", "qed_swe_bench.cli", "benchmark", "--mock-llm"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            env={**os.environ},
            timeout=120,
        )
        assert proc.returncode == 0, proc.stderr

    con = sqlite3.connect(tmp_path / "test.sqlite")
    # Two distinct benchmarks, one row each.
    rows = con.execute(
        "SELECT benchmark_id, count(*) FROM runs GROUP BY benchmark_id"
    ).fetchall()
    assert len(rows) == 2
    assert all(count == 1 for _, count in rows)

    # Within any benchmark_id, the UNIQUE constraint holds.
    dupes = con.execute(
        "SELECT benchmark_id, model, env_id, seed, count(*) AS n FROM runs "
        "GROUP BY benchmark_id, model, env_id, seed HAVING n > 1"
    ).fetchall()
    assert dupes == []
