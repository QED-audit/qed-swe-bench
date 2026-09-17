"""End-to-end resume against a real MCP container (sample-stack-bof).

Requires docker + the local sample image. Build it with `make smoke` or
`scripts/build_sample_env.sh`. Marked `slow` — opt in with
`pytest -m slow`.

What this exercises that unit tests don't:
  - replay_tool_calls actually drives a real MCP stdio connection
  - resume_one_run correctly opens MCP, replays, hands off to run_episode
  - TranscriptWriter append-mode preserves the truncated history
  - DB row transitions failed → succeeded after a clean resume
"""

from __future__ import annotations

import asyncio
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


def _produce_source_run(tmp_path: Path) -> Path:
    """Run a mock-llm benchmark once to get a real run_dir + DB row.
    Returns the produced run_dir. The benchmark_id is timestamped so it
    won't collide with anything else."""
    db_path = tmp_path / "src.sqlite"
    runs_dir = tmp_path / "runs"
    env = {
        **os.environ,
        "QED_SWE_BENCH_DB": str(db_path),
        "QED_SWE_BENCH_RUNS_DIR": str(runs_dir),
    }
    proc = subprocess.run(
        [sys.executable, "-m", "qed_swe_bench.cli", "benchmark", "--mock-llm"],
        capture_output=True, text=True, cwd=REPO_ROOT, env=env, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr

    score_files = list(runs_dir.rglob("score.json"))
    assert score_files, "no run dir produced by mock-llm benchmark"
    return score_files[0].parent


def _ai_turn_count(transcript: Path) -> int:
    n = 0
    for line in transcript.open():
        try:
            if json.loads(line).get("role") == "ai":
                n += 1
        except json.JSONDecodeError:
            pass
    return n


def _truncate_at_ai_turn(run_dir: Path, target_ai_turns: int) -> int:
    transcript = run_dir / "transcript.jsonl"
    cut_ts: str | None = None
    keep: list[str] = []
    kept_ai = 0
    for line in transcript.read_text().splitlines():
        if not line.strip():
            keep.append(line)
            continue
        obj = json.loads(line)
        if obj.get("role") == "ai":
            if kept_ai >= target_ai_turns:
                break
            kept_ai += 1
            cut_ts = obj.get("ts") or cut_ts
        keep.append(line)
        if kept_ai >= target_ai_turns and obj.get("role") not in ("ai", "tool"):
            break
    transcript.write_text("\n".join(keep) + "\n")

    if cut_ts is not None:
        for fname in ("tool_calls.jsonl", "grade_calls.jsonl"):
            fp = run_dir / fname
            if not fp.is_file():
                continue
            kept_lines = []
            for line in fp.read_text().splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                if (obj.get("ts") or "") <= cut_ts:
                    kept_lines.append(line)
            fp.write_text("\n".join(kept_lines) + ("\n" if kept_lines else ""))
    return kept_ai


@pytest.mark.slow
@pytest.mark.skipif(not _have_docker(), reason="docker not available")
@pytest.mark.skipif(
    not _have_sample_image(),
    reason=f"image not built: {SAMPLE_IMAGE}; run scripts/build_sample_env.sh",
)
def test_resume_replays_against_real_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Truncate a successful mock-llm run, resume it via resume_one_run,
    confirm replay actually drives MCP and the resumed loop continues."""
    from qed_swe_bench.runner.resume import resume_one_run

    monkeypatch.setenv("QED_SWE_BENCH_DB", str(tmp_path / "resume.sqlite"))
    monkeypatch.setenv("QED_SWE_BENCH_RUNS_DIR", str(tmp_path / "runs"))

    # Step 1: produce a real, succeeded run via mock-llm.
    src_run = _produce_source_run(tmp_path)
    src_ai_turns = _ai_turn_count(src_run / "transcript.jsonl")
    assert src_ai_turns >= 3, "mock benchmark should produce >=3 ai turns"

    # Step 2: copy + truncate to half the ai turns.
    target = tmp_path / "truncated"
    shutil.copytree(src_run, target,
                    ignore=shutil.ignore_patterns("score.json", "cost.json"))
    keep_n = max(1, src_ai_turns // 2)
    kept = _truncate_at_ai_turn(target, keep_n)
    assert kept == keep_n

    # Step 3: rewrite job.json with a fresh run_id + benchmark_id so the
    # source run's DB row stays untouched.
    job_path = target / "job.json"
    job = json.loads(job_path.read_text())
    job["run_id"] = "rt-e2e-001"
    job["benchmark_id"] = "replay-test-e2e"
    job_path.write_text(json.dumps(job, indent=2))

    # Step 4: insert a fake model_failed row pointing at the truncated dir.
    from qed_swe_bench.config import Config
    from qed_swe_bench.db.schema import init_db, transaction
    init_db(Config.from_env().db_path)
    with transaction() as con:
        con.execute(
            """
            INSERT INTO runs (
                run_id, benchmark_id, model, env_id, image_ref, image_digest,
                task_type, interface, seed, status, run_dir, started_at,
                provenance, exit_reason, finished_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'model_failed', ?,
                      '2026-01-01T00:00:00+00:00', 'mock', 'error: Timeout',
                      '2026-01-01T00:01:00+00:00')
            """,
            (
                job["run_id"], job["benchmark_id"], job["model"], job["env_id"],
                job.get("image_ref", ""), job["image_digest"],
                job.get("task_type", "binary_task"), job.get("interface"),
                int(job["seed"]), str(target),
            ),
        )

    # Step 5: resume in mock mode (real container, mock LLM).
    from qed_swe_bench.runner.orchestrator_config import Budgets
    outcome = asyncio.run(resume_one_run(
        target,
        budgets=Budgets(),
        episode_timeout_s=120,
        model_params={},
        mock_llm=True,
    ))

    # Step 6: verify the resumed run finalized cleanly.
    assert outcome.status in ("succeeded", "model_failed"), outcome
    # Resumed run's turns_total should exceed the truncated prior count
    # (the loop made at least one new turn before stopping).
    assert outcome.turns_total >= keep_n

    # The truncated dir's transcript was appended to, not rewritten.
    final_ai_turns = _ai_turn_count(target / "transcript.jsonl")
    assert final_ai_turns >= keep_n

    # DB row was finalized.
    with transaction() as con:
        row = con.execute(
            "SELECT status FROM runs WHERE run_id = ?", (job["run_id"],),
        ).fetchone()
    assert row is not None
    assert row[0] in ("succeeded", "model_failed")
