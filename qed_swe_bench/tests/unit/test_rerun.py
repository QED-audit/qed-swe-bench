"""Tests for `qed_swe_bench rerun <run_id>` (D-13).

The rerun subcommand reads `runs.config_snapshot` (the verbatim
source YAML the row was produced under), narrows to the row's (model,
env, seed) tuple, and replays through the same code path as
`qed_swe_bench benchmark`. The DB row is self-sufficient — no
`<run_dir>/config_snapshot.yaml` on disk required for new (post-D-13)
rows. Legacy rows fall back to the on-disk artifact.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from qed_swe_bench.cli import app
from qed_swe_bench.db.schema import init_db, transaction


_RUNNER = CliRunner()


def _flat(s: str) -> str:
    """Collapse whitespace so Rich's terminal-width line-wrap doesn't break
    substring assertions on long error messages."""
    return " ".join(s.split())


_SNAPSHOT_YAML = """\
# v8 baseline (D-3) — nudges OFF for clean evaluation
benchmark_id: v8
init_prompt: "Use setup() to learn about the target."
models:
  - id: anthropic/claude-haiku-4-5  # cheap shakedown only
    params:
      reasoning_effort: high
  - id: openai/gpt-5.5
seeds: [1, 2, 3]
nudges: false  # D-3: clean baseline
budgets:
  turn_budget: 300
  token_budget: null  # D-11: turn-as-effort
  context_budget: null
  max_tokens: 16384
envs:
  - id: v8-e01
    image: ecr.example/qed_swe_bench:cve-test
"""


@pytest.fixture
def tmp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db = tmp_path / "qed_swe_bench.sqlite"
    monkeypatch.setenv("QED_SWE_BENCH_DB", str(db))
    monkeypatch.setenv("QED_SWE_BENCH_RUNS_DIR", str(tmp_path / "runs"))
    init_db(db)
    return db


def _seed_row(
    *,
    run_id: str = "abc123def4567890",
    model: str = "anthropic/claude-haiku-4-5",
    env_id: str = "v8-e01",
    seed: int = 1,
    config_snapshot: str | None = _SNAPSHOT_YAML,
    run_dir: str | None = None,
) -> None:
    """Insert a minimal row carrying just enough for reproduce."""
    with transaction() as con:
        con.execute(
            """
            INSERT INTO runs (
                run_id, benchmark_id, model, env_id, image_ref, image_digest,
                task_type, seed, status, run_dir, started_at, provenance,
                config_snapshot
            ) VALUES (?, 'v8', ?, ?, 'r', 'sha256:x', 'binary_task', ?,
                      'succeeded', ?, '2026-05-02T18:00:00+00:00', 'native', ?)
            """,
            (run_id, model, env_id, seed, run_dir, config_snapshot),
        )


# ---------------- error paths ----------------


def test_rerun_unknown_run_id_exits_1(tmp_db: Path) -> None:
    result = _RUNNER.invoke(app, ["rerun", "deadbeef00000000"])
    assert result.exit_code == 1
    assert "no run with run_id" in _flat(result.stdout)


def test_rerun_null_snapshot_no_fallback_exits_1(tmp_db: Path) -> None:
    """Row has no config_snapshot column AND no on-disk fallback file → fail."""
    _seed_row(config_snapshot=None, run_dir=None)
    result = _RUNNER.invoke(app, ["rerun", "abc123def4567890"])
    assert result.exit_code == 1
    assert "not reproducible" in _flat(result.stdout)


def test_rerun_malformed_snapshot_exits_1(tmp_db: Path) -> None:
    _seed_row(config_snapshot="benchmark_id: v8\nmodels: [")
    result = _RUNNER.invoke(app, ["rerun", "abc123def4567890"])
    assert result.exit_code == 1
    assert "malformed" in _flat(result.stdout)


def test_rerun_snapshot_missing_model_exits_1(tmp_db: Path) -> None:
    """Row's model isn't declared in the snapshot — shouldn't happen for
    rows the orchestrator wrote, but defend against drift."""
    _seed_row(model="openai/o9-future-model")
    result = _RUNNER.invoke(app, ["rerun", "abc123def4567890"])
    assert result.exit_code == 1
    flat = _flat(result.stdout)
    assert "openai/o9-future-model" in flat
    assert "doesn't declare model" in flat


# ---------------- happy path: dry-run ----------------


def test_rerun_dry_run_narrows_to_single_tuple(tmp_db: Path) -> None:
    """--dry-run prints the resolved BenchmarkConfig and exits without
    invoking run_benchmark. Filters must narrow the snapshot's
    multi-model / multi-seed config to exactly the row's tuple."""
    _seed_row(seed=2)  # snapshot declares seeds [1,2,3]; row is seed=2
    result = _RUNNER.invoke(app, [
        "rerun", "abc123def4567890", "--dry-run",
    ])
    assert result.exit_code == 0, result.stdout
    # Filter to row's single model + single seed
    assert "models=['anthropic/claude-haiku-4-5']" in result.stdout
    assert "seeds=[2]" in result.stdout
    assert "envs=['v8-e01']" in result.stdout
    # Snapshot's nudges=false survives intact
    assert "nudges=[]" in result.stdout


def test_rerun_dry_run_does_not_call_run_benchmark(tmp_db: Path) -> None:
    """Belt + suspenders on the --dry-run guard."""
    _seed_row()
    with patch("qed_swe_bench.runner.orchestrator.run_benchmark") as rb:
        result = _RUNNER.invoke(app, [
            "rerun", "abc123def4567890", "--dry-run",
        ])
    assert result.exit_code == 0, result.stdout
    rb.assert_not_called()


# ---------------- happy path: execution ----------------


def test_rerun_invokes_run_benchmark_with_filtered_config(
    tmp_db: Path,
) -> None:
    """The non-dry path calls run_benchmark with the snapshot parsed +
    filtered to a single (model, env, seed) tuple."""
    _seed_row(seed=3)
    captured = {}

    async def _fake_run(bench, **kwargs):
        captured["bench"] = bench
        captured["kwargs"] = kwargs
        return {"succeeded": 1}

    with patch("qed_swe_bench.runner.orchestrator.run_benchmark", new=_fake_run):
        result = _RUNNER.invoke(app, ["rerun", "abc123def4567890"])

    assert result.exit_code == 0, result.stdout
    bench = captured["bench"]
    assert [m.id for m in bench.models] == ["anthropic/claude-haiku-4-5"]
    assert [e.id for e in bench.envs] == ["v8-e01"]
    assert bench.seeds == [3]
    # Snapshot's params survived filtering
    assert bench.models[0].params == {"reasoning_effort": "high"}
    # Snapshot's null token/context budgets (D-11) survived
    assert bench.budgets.token_budget is None
    assert bench.budgets.context_budget is None
    # config_path kwarg is materialized to a real readable file so the
    # orchestrator's normal config_snapshot.yaml-write path works.
    cfg_path = captured["kwargs"]["config_path"]
    assert cfg_path is not None
    # NOTE: tempfile is unlinked after run_benchmark returns — we can't
    # re-read it here. The fact that the call succeeded with cfg_path
    # set is the test.


def test_rerun_legacy_fallback_reads_run_dir_yaml(
    tmp_db: Path, tmp_path: Path,
) -> None:
    """Pre-D-13 rows have config_snapshot=NULL but may have the on-disk
    `<run_dir>/config_snapshot.yaml`. Reproduce should fall back to it."""
    legacy_run_dir = tmp_path / "legacy"
    legacy_run_dir.mkdir()
    (legacy_run_dir / "config_snapshot.yaml").write_text(
        _SNAPSHOT_YAML, encoding="utf-8",
    )
    _seed_row(config_snapshot=None, run_dir=str(legacy_run_dir))

    result = _RUNNER.invoke(app, [
        "rerun", "abc123def4567890", "--dry-run",
    ])
    assert result.exit_code == 0, result.stdout
    assert "legacy" in result.stdout  # the warning banner mentions it
    assert "models=['anthropic/claude-haiku-4-5']" in result.stdout


def test_rerun_cost_cap_override_takes_effect(tmp_db: Path) -> None:
    """--cost-cap-usd overrides the snapshot's value (or sets one)."""
    _seed_row()
    captured = {}

    async def _fake_run(bench, **kwargs):
        captured["bench"] = bench
        return {"succeeded": 1}

    with patch("qed_swe_bench.runner.orchestrator.run_benchmark", new=_fake_run):
        result = _RUNNER.invoke(app, [
            "rerun", "abc123def4567890", "--cost-cap-usd", "0.50",
        ])

    assert result.exit_code == 0, result.stdout
    assert captured["bench"].cost_cap_usd == 0.5
