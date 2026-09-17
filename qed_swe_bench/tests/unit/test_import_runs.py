"""Tests for `qed_swe_bench import` and `qed_swe_bench export` — the
native FS↔DB bijection (D-10).

Import: run-dir → runs row.
Export: runs row → run-dir.
Round-trip: import then export should reproduce the source artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from qed_swe_bench.db.schema import connect, init_db
from qed_swe_bench.historical import (
    export_native_run,
    export_native_runs,
    import_native_run_dir,
    import_native_runs,
    migrate_runs_to_layered_layout,
)


@pytest.fixture
def tmp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db = tmp_path / "qed_swe_bench.sqlite"
    monkeypatch.setenv("QED_SWE_BENCH_DB", str(db))
    init_db(db)
    return db


def _write_synthetic_run_dir(
    base: Path,
    *,
    run_id: str = "abc123def4567890",
    benchmark_id: str = "v8",
    model: str = "anthropic/claude-haiku-4-5",
    env_id: str = "v8-e01",
    seed: int = 1,
    score: float = 3.0,
    status: str = "succeeded",
    capabilities: dict | None = None,
    write_cost: bool = True,
    write_score: bool = True,
    config_snapshot_yaml: str | None = None,
) -> Path:
    """Synthesize a run-dir with job/score/cost.json (+ optional snapshot)."""
    run_dir = base / run_id
    run_dir.mkdir(parents=True)
    if config_snapshot_yaml is not None:
        (run_dir / "config_snapshot.yaml").write_text(
            config_snapshot_yaml, encoding="utf-8",
        )
    (run_dir / "job.json").write_text(json.dumps({
        "run_id": run_id,
        "benchmark_id": benchmark_id,
        "model": model,
        "env_id": env_id,
        "image_ref": "ecr.example/qed_swe_bench:cve-test",
        "image_digest": "sha256:" + "a" * 64,
        "image_pulled": False,
        "task_type": "binary_task",
        "interface": "rl.mcp.v8_task.v1",
        "seed": seed,
        "budgets": {"turn_budget": 300, "token_budget": 2500000,
                    "context_budget": 180000, "max_tokens": 16384},
        "started_at": "2026-05-02T18:00:00+00:00",
        "git_sha": "abcdef1234567890",
        "repro_cmd": "qed_swe_bench benchmark ...",
    }))
    if write_score:
        (run_dir / "score.json").write_text(json.dumps({
            "capabilities": capabilities or {"cov_func": True, "cov_line": True, "diff": True},
            "score": score,
            "exit_reason": "budget: token_budget (2500000 >= 2500000)",
            "finished_at": "2026-05-02T19:00:00+00:00",
            "runtime_s": 3600.5,
            "turns_used": 187,
            "status": status,
            "weighted_tokens_used": 1_200_000,
            "peak_per_turn_context": 165_000,
        }))
    if write_cost:
        (run_dir / "cost.json").write_text(json.dumps({
            "model": model,
            "served_model": model.rsplit("/", 1)[-1],
            "tokens_in": 5000000,
            "tokens_out": 80000,
            "tokens_cache_read": 4500000,
            "tokens_cache_creation": 0,
            "tokens_reasoning": 0,
            "weighted_tokens_used": 1200000,
            "cost_usd": 5.49,
            "cost_source": "pricing_table",
            "llm_route": "litellm",
            "api_base": None,
        }))
    return run_dir


# ---------------- import_native_run_dir ----------------


def test_import_run_dir_reads_all_artifacts(tmp_path: Path) -> None:
    run_dir = _write_synthetic_run_dir(tmp_path)
    row = import_native_run_dir(run_dir)
    assert row["run_id"] == "abc123def4567890"
    assert row["benchmark_id"] == "v8"
    assert row["model"] == "anthropic/claude-haiku-4-5"
    assert row["env_id"] == "v8-e01"
    assert row["seed"] == 1
    assert row["score"] == 3.0
    assert row["status"] == "succeeded"
    assert row["tokens_in"] == 5000000
    assert row["cost_usd"] == 5.49
    assert row["llm_route"] == "litellm"
    assert row["interface"] == "rl.mcp.v8_task.v1"
    assert row["git_sha"] == "abcdef1234567890"
    assert row["finished_at"] == "2026-05-02T19:00:00+00:00"
    assert row["runtime_s"] == 3600.5
    assert row["turns_used"] == 187
    assert row["provenance"] == "native"
    # Always-on diagnostics from score.json (turn-as-effort).
    assert row["weighted_tokens_used"] == 1_200_000
    assert row["peak_per_turn_context"] == 165_000
    # capabilities is JSON-encoded
    assert json.loads(row["capabilities"]) == {
        "cov_func": True, "cov_line": True, "diff": True,
    }


def test_import_run_dir_diagnostics_default_to_none(tmp_path: Path) -> None:
    """Legacy / pre-2026-05 score.json files won't have the new
    diagnostic fields. Importer must tolerate their absence and write
    NULL into the corresponding columns."""
    run_dir = tmp_path / "legacy_run"
    run_dir.mkdir()
    (run_dir / "job.json").write_text(json.dumps({
        "run_id": "legacy00abcdef00",
        "benchmark_id": "v8",
        "model": "anthropic/claude-opus-4-7",
        "env_id": "v8-e01",
        "image_ref": "ecr/x:tag",
        "image_digest": "sha256:" + "b" * 64,
        "image_pulled": False,
        "task_type": "binary_task",
        "interface": "rl.mcp.v8_task.v1",
        "seed": 1,
        "budgets": {"turn_budget": 300},
        "started_at": "2026-04-01T00:00:00+00:00",
    }))
    (run_dir / "score.json").write_text(json.dumps({
        "capabilities": {"crash": True},
        "score": 1.0,
        "exit_reason": "no_tool_calls",
        "status": "succeeded",
    }))
    row = import_native_run_dir(run_dir)
    assert row["weighted_tokens_used"] is None
    assert row["peak_per_turn_context"] is None


def test_import_run_dir_missing_score_yields_queued(tmp_path: Path) -> None:
    """A run-dir without score.json (crashed mid-episode) imports as queued."""
    run_dir = _write_synthetic_run_dir(tmp_path, write_score=False)
    row = import_native_run_dir(run_dir)
    assert row["status"] == "queued"
    assert row["score"] is None
    assert row["finished_at"] is None


def test_import_run_dir_missing_cost_yields_null_telemetry(tmp_path: Path) -> None:
    """A run-dir without cost.json imports with NULL token/cost columns."""
    run_dir = _write_synthetic_run_dir(tmp_path, write_cost=False)
    row = import_native_run_dir(run_dir)
    assert row["tokens_in"] is None
    assert row["cost_usd"] is None
    assert row["llm_route"] is None


def test_import_run_dir_missing_job_raises(tmp_path: Path) -> None:
    """job.json is the only required artifact; absence is an error."""
    run_dir = tmp_path / "no_job"
    run_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="no job.json"):
        import_native_run_dir(run_dir)


# ---------------- import_native_runs (DB integration) ----------------


def test_import_native_runs_walks_tree(tmp_db: Path, tmp_path: Path) -> None:
    """Multiple run-dirs at different depths are all imported."""
    _write_synthetic_run_dir(tmp_path / "v8/host-a/2026-05-02T18-00-00Z",
                             run_id="11111111aaaaaaaa", seed=1)
    _write_synthetic_run_dir(tmp_path / "v8/host-a/2026-05-02T19-00-00Z",
                             run_id="22222222bbbbbbbb", seed=2)
    _write_synthetic_run_dir(tmp_path / "v8/host-b/2026-05-02T20-00-00Z",
                             run_id="33333333cccccccc", seed=3)

    histogram = import_native_runs(tmp_path)
    assert histogram["imported"] == 3
    assert histogram["skipped_duplicate"] == 0
    assert histogram["unparseable"] == 0

    with connect(tmp_db) as con:
        rows = con.execute("SELECT run_id FROM runs ORDER BY run_id").fetchall()
    assert {r["run_id"] for r in rows} == {
        "11111111aaaaaaaa", "22222222bbbbbbbb", "33333333cccccccc"
    }


def test_import_native_runs_idempotent(tmp_db: Path, tmp_path: Path) -> None:
    """Running import twice on the same tree is a no-op the second time."""
    _write_synthetic_run_dir(tmp_path / "v8", seed=1)
    h1 = import_native_runs(tmp_path)
    h2 = import_native_runs(tmp_path)
    assert h1["imported"] == 1
    assert h2["imported"] == 0
    assert h2["skipped_duplicate"] == 1


def test_import_native_runs_unparseable_counted(
    tmp_db: Path, tmp_path: Path,
) -> None:
    """A run-dir with malformed job.json is counted as unparseable, not crashing."""
    bad_dir = tmp_path / "broken/run_id_xyz"
    bad_dir.mkdir(parents=True)
    (bad_dir / "job.json").write_text("not valid json {{{")
    histogram = import_native_runs(tmp_path)
    assert histogram["unparseable"] == 1
    assert histogram["imported"] == 0


def test_import_native_runs_round_trip_preserves_columns(
    tmp_db: Path, tmp_path: Path,
) -> None:
    """The DB row produced by import matches the run-dir artifacts
    field-by-field — this is the bijection contract."""
    _write_synthetic_run_dir(tmp_path / "v8", seed=1)
    import_native_runs(tmp_path)

    with connect(tmp_db) as con:
        row = dict(con.execute("SELECT * FROM runs").fetchone())

    assert row["run_id"] == "abc123def4567890"
    assert row["benchmark_id"] == "v8"
    assert row["model"] == "anthropic/claude-haiku-4-5"
    assert row["score"] == 3.0
    assert row["status"] == "succeeded"
    assert row["tokens_in"] == 5000000
    assert row["cost_usd"] == 5.49
    assert row["llm_route"] == "litellm"
    assert row["interface"] == "rl.mcp.v8_task.v1"
    assert row["finished_at"] == "2026-05-02T19:00:00+00:00"
    assert row["turns_used"] == 187
    assert row["provenance"] == "native"
    assert json.loads(row["capabilities"]) == {
        "cov_func": True, "cov_line": True, "diff": True,
    }


# ---------------- export_native_run / export_native_runs ----------------


def test_export_run_writes_three_artifacts(
    tmp_db: Path, tmp_path: Path,
) -> None:
    """Exporting a row produces job.json + score.json + cost.json."""
    _write_synthetic_run_dir(tmp_path / "src" / "v8", seed=1)
    import_native_runs(tmp_path / "src")

    with connect(tmp_db) as con:
        row = dict(con.execute("SELECT * FROM runs").fetchone())

    target = tmp_path / "exported"
    export_native_run(row, target)
    assert (target / "job.json").is_file()
    assert (target / "score.json").is_file()
    assert (target / "cost.json").is_file()


def test_bijection_round_trip_preserves_diagnostic_fields(
    tmp_db: Path, tmp_path: Path,
) -> None:
    """Import then export: weighted_tokens_used and peak_per_turn_context
    survive both legs of the bijection (D-10 contract for the new
    always-on diagnostic columns)."""
    _write_synthetic_run_dir(tmp_path / "src" / "v8", seed=1)
    import_native_runs(tmp_path / "src")

    with connect(tmp_db) as con:
        row = dict(con.execute("SELECT * FROM runs").fetchone())
    assert row["weighted_tokens_used"] == 1_200_000
    assert row["peak_per_turn_context"] == 165_000

    target = tmp_path / "exported"
    export_native_run(row, target)
    score = json.loads((target / "score.json").read_text())
    assert score["weighted_tokens_used"] == 1_200_000
    assert score["peak_per_turn_context"] == 165_000


def test_bijection_round_trip_preserves_config_snapshot(
    tmp_db: Path, tmp_path: Path,
) -> None:
    """D-13: import → export of a run-dir with config_snapshot.yaml
    produces a byte-identical snapshot file. The bytes flow:
      source YAML on disk → import reads → DB column → export writes →
      target YAML on disk (must equal source byte-for-byte, comments
      preserved).
    """
    snapshot_with_comments = (
        "# v8 baseline (D-3) — nudges OFF for clean evaluation\n"
        "benchmark_id: v8\n"
        "models:\n"
        "  - id: anthropic/claude-haiku-4-5  # cheap shakedown only\n"
        "envs:\n"
        "  - id: v8-e01\n"
        "    image: ecr.example/qed_swe_bench:cve-test\n"
        "seeds: [1]\n"
        "nudges: false  # D-3: clean baseline\n"
    )
    src_dir = _write_synthetic_run_dir(
        tmp_path / "src" / "v8",
        seed=1,
        config_snapshot_yaml=snapshot_with_comments,
    )
    # Import reads the snapshot off disk into the column.
    row = import_native_run_dir(src_dir)
    assert row["config_snapshot"] == snapshot_with_comments

    # And import_native_runs writes the row to the DB end-to-end.
    import_native_runs(tmp_path / "src")
    with connect(tmp_db) as con:
        db_row = dict(con.execute("SELECT * FROM runs").fetchone())
    assert db_row["config_snapshot"] == snapshot_with_comments

    # Export writes the column back to a target run-dir. Bytes must
    # be byte-identical to source.
    target = tmp_path / "exported"
    export_native_run(db_row, target)
    exported_yaml = (target / "config_snapshot.yaml").read_text(encoding="utf-8")
    assert exported_yaml == snapshot_with_comments


def test_import_run_dir_without_snapshot_yields_null(tmp_path: Path) -> None:
    """Pre-D-13 / partial run-dirs (no config_snapshot.yaml) import
    cleanly with the column NULL. Reproduce treats NULL as "this row
    is unreproducible" unless the on-disk fallback works."""
    run_dir = _write_synthetic_run_dir(tmp_path)  # no snapshot kwarg
    row = import_native_run_dir(run_dir)
    assert row["config_snapshot"] is None


def test_export_skips_snapshot_when_column_is_null(
    tmp_db: Path, tmp_path: Path,
) -> None:
    """Symmetric to the import-without-snapshot case: a row with
    config_snapshot=NULL must NOT produce an empty config_snapshot.yaml
    on export — that would be a lie about what was actually run."""
    _write_synthetic_run_dir(tmp_path / "src" / "v8", seed=1)  # no snapshot
    import_native_runs(tmp_path / "src")
    with connect(tmp_db) as con:
        row = dict(con.execute("SELECT * FROM runs").fetchone())
    assert row["config_snapshot"] is None

    target = tmp_path / "exported"
    export_native_run(row, target)
    assert not (target / "config_snapshot.yaml").exists()


def test_bijection_round_trip_preserves_mock_provenance(
    tmp_db: Path, tmp_path: Path,
) -> None:
    """A mock-LLM run-dir round-trips with `provenance=mock` intact.
    Pre-fix, import_native_run_dir hard-coded `provenance=native`,
    losing the mock label on every re-import. job.json now carries
    provenance directly."""
    src = tmp_path / "src" / "smoke" / "abc123def4567890"
    src.mkdir(parents=True)
    (src / "job.json").write_text(json.dumps({
        "run_id": "abc123def4567890",
        "benchmark_id": "smoke",
        "model": "mock/test",
        "env_id": "sample-stack-bof",
        "image_ref": "local/sample-stack-bof:latest",
        "image_digest": "sha256:" + "a" * 64,
        "image_pulled": False,
        "task_type": "binary_task",
        "interface": "rl.mcp.v8_task.v1",
        "seed": 1,
        "budgets": {"turn_budget": 5},
        "started_at": "2026-05-02T22:57:15+00:00",
        "provenance": "mock",
    }))
    (src / "score.json").write_text(json.dumps({
        "capabilities": {},
        "score": 0.0,
        "exit_reason": "no_tool_calls",
        "status": "succeeded",
    }))
    row = import_native_run_dir(src)
    assert row["provenance"] == "mock"

    # Round-trip through export: new job.json should preserve mock.
    target = tmp_path / "exported"
    export_native_run(row, target)
    job = json.loads((target / "job.json").read_text())
    assert job["provenance"] == "mock"


def test_bijection_legacy_job_json_defaults_to_native(tmp_path: Path) -> None:
    """A pre-2026-05 job.json with no `provenance` field still imports
    cleanly and gets the historical default `native` (preserves
    backward compat for existing runs/ trees)."""
    src = tmp_path / "legacy"
    src.mkdir()
    (src / "job.json").write_text(json.dumps({
        "run_id": "legacy00abcdef00",
        "benchmark_id": "v8",
        "model": "anthropic/claude-opus-4-7",
        "env_id": "v8-e01",
        "image_ref": "ecr/x:tag",
        "image_digest": "sha256:" + "b" * 64,
        "image_pulled": False,
        "task_type": "binary_task",
        "interface": "rl.mcp.v8_task.v1",
        "seed": 1,
        "budgets": {"turn_budget": 300},
        "started_at": "2026-04-01T00:00:00+00:00",
        # No `provenance` field — pre-fix shape.
    }))
    row = import_native_run_dir(src)
    assert row["provenance"] == "native"


def test_export_skips_rows_with_existing_run_dir(
    tmp_db: Path, tmp_path: Path,
) -> None:
    """By default, export skips rows whose run_dir is populated on disk."""
    _write_synthetic_run_dir(tmp_path / "src" / "v8", seed=1)
    import_native_runs(tmp_path / "src")
    # The row's run_dir points at the source directory which has job.json.

    target = tmp_path / "exported"
    histogram = export_native_runs(target)
    assert histogram["skipped_has_dir"] == 1
    assert histogram["exported"] == 0


def test_export_force_all_overrides_skip(
    tmp_db: Path, tmp_path: Path,
) -> None:
    """only_missing_run_dir=False exports every matching row."""
    _write_synthetic_run_dir(tmp_path / "src" / "v8", seed=1)
    import_native_runs(tmp_path / "src")

    target = tmp_path / "exported"
    histogram = export_native_runs(target, only_missing_run_dir=False)
    assert histogram["exported"] == 1
    expected = target / "v8" / "exported" / "abc123def4567890" / "job.json"
    assert expected.is_file()


# ---------------- migrate_runs_to_layered_layout ----------------


def test_migrate_moves_legacy_to_layered(
    tmp_db: Path, tmp_path: Path,
) -> None:
    """Legacy `runs/<benchmark_id>/<utc>__<run_id>/` moves to
    `runs/<benchmark_id>/<host>/<datetime>/<run_id>/`."""
    runs_root = tmp_path / "runs"
    legacy_dir = runs_root / "v8" / "2026-05-02T18-00-00Z__abc123def4567890"
    _write_synthetic_run_dir(legacy_dir.parent,
                             run_id="2026-05-02T18-00-00Z__abc123def4567890",
                             seed=1)
    # The synthetic-dir helper used run_id as the dir name; rename to
    # the legacy-shape we expect.
    src = legacy_dir.parent / "2026-05-02T18-00-00Z__abc123def4567890"
    # Update the run_id inside job.json so the import recognizes it.
    job = json.loads((src / "job.json").read_text())
    job["run_id"] = "abc123def4567890"
    (src / "job.json").write_text(json.dumps(job, indent=2))
    # Also update score.json so the row gets imported as succeeded.
    import_native_runs(runs_root)

    # Now migrate.
    histogram = migrate_runs_to_layered_layout(
        runs_root, legacy_host="testbox",
    )
    assert histogram["migrated"] == 1
    # Old path should be gone.
    assert not src.exists()
    # New path should exist with job.json.
    new_dirs = list(runs_root.glob("v8/testbox/*/abc123def4567890"))
    assert len(new_dirs) == 1
    assert (new_dirs[0] / "job.json").is_file()

    # DB row should now point at the new path.
    with connect(tmp_db) as con:
        row = con.execute(
            "SELECT run_dir FROM runs WHERE run_id='abc123def4567890'"
        ).fetchone()
    assert "testbox" in row["run_dir"]


def test_migrate_skips_already_layered(
    tmp_db: Path, tmp_path: Path,
) -> None:
    """If the path already has 4 segments, skip — already migrated."""
    runs_root = tmp_path / "runs"
    new_path = runs_root / "v8" / "host-a" / "2026-05-02T18-00-00Z" / "abc123def4567890"
    _write_synthetic_run_dir(new_path.parent, run_id=new_path.name, seed=1)
    src = new_path.parent / new_path.name
    job = json.loads((src / "job.json").read_text())
    job["run_id"] = "abc123def4567890"
    (src / "job.json").write_text(json.dumps(job, indent=2))
    import_native_runs(runs_root)

    histogram = migrate_runs_to_layered_layout(runs_root, legacy_host="legacy")
    assert histogram["skipped_already_new"] == 1
    assert histogram["migrated"] == 0


def test_migrate_dry_run_does_not_touch_anything(
    tmp_db: Path, tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    legacy_dir = runs_root / "v8"
    _write_synthetic_run_dir(legacy_dir,
                             run_id="2026-05-02T18-00-00Z__abc123def4567890",
                             seed=1)
    src = legacy_dir / "2026-05-02T18-00-00Z__abc123def4567890"
    job = json.loads((src / "job.json").read_text())
    job["run_id"] = "abc123def4567890"
    (src / "job.json").write_text(json.dumps(job, indent=2))
    import_native_runs(runs_root)

    histogram = migrate_runs_to_layered_layout(
        runs_root, legacy_host="legacy", dry_run=True,
    )
    assert histogram["dry_run_only"] == 1
    assert histogram["migrated"] == 0
    # Source should still exist (not moved).
    assert src.exists()


def test_migrate_refuses_when_orchestrator_is_active(
    tmp_db: Path, tmp_path: Path,
) -> None:
    """A queued row with a recent heartbeat = a live orchestrator.
    Don't migrate; raise RuntimeError."""
    from datetime import UTC, datetime as _dt
    with connect(tmp_db) as con:
        con.execute(
            """INSERT INTO runs (run_id, benchmark_id, model, env_id,
                image_ref, image_digest, task_type, seed, status, run_dir,
                started_at, last_heartbeat, provenance)
               VALUES ('live', 'v8', 'm', 'e', 'r', 'sha256:x',
                'binary_task', 1, 'queued', '/tmp/x',
                ?, ?, 'native')""",
            (_dt.now(UTC).isoformat(), _dt.now(UTC).isoformat()),
        )
    with pytest.raises(RuntimeError, match="recent heartbeat"):
        migrate_runs_to_layered_layout(tmp_path / "runs")
