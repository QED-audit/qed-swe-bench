"""Orchestrator config parsing + DB lifecycle (no Docker / no MCP)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from qed_swe_bench.db.schema import connect, init_db
from qed_swe_bench.runner.llm.base import NormalizedUsage
from qed_swe_bench.runner.orchestrator import (
    Budgets,
    EnvSpec,
    _insert_queued,
    _load_env_manifest,
    _mark_running,
    _update_finished,
    parse_config,
)
from qed_swe_bench.runner.orchestrator_config import NudgeKind
from qed_swe_bench.runner.run_dir import append_cli_oneshot_grade_log

# ---------------- parse_config ----------------


def test_parse_config_minimal() -> None:
    cfg = parse_config(
        {
            "benchmark_id": "b1",
            "models": [{"id": "anthropic/claude-haiku-4-5"}],
            "envs": [{"id": "e1", "image": "local/e1:latest"}],
            "seeds": [1, 2],
        }
    )
    assert cfg.benchmark_id == "b1"
    assert len(cfg.models) == 1
    assert cfg.envs[0].image == "local/e1:latest"
    assert cfg.envs[0].task_type == "binary_task"  # default
    assert cfg.seeds == [1, 2]
    assert cfg.max_parallel == 2  # default
    assert isinstance(cfg.budgets, Budgets)


def test_parse_config_overrides_budgets() -> None:
    cfg = parse_config(
        {
            "benchmark_id": "b",
            "models": [{"id": "m"}],
            "envs": [{"id": "e", "image": "i"}],
            "seeds": [1],
            "budgets": {
                "turn_budget": 50,
                "token_budget": 100_000,
                "context_budget": 8000,
                "max_tokens": 4096,
            },
        }
    )
    assert cfg.budgets.turn_budget == 50
    assert cfg.budgets.token_budget == 100_000
    assert cfg.budgets.context_budget == 8000
    assert cfg.budgets.max_tokens == 4096


def test_parse_config_null_token_and_context_budget() -> None:
    """Turn-as-effort: explicit `null` in YAML disables enforcement on
    those axes while keeping turn_budget as the lone fairness anchor.
    `max_tokens` (per-call cap) is unaffected."""
    cfg = parse_config(
        {
            "benchmark_id": "b",
            "models": [{"id": "m"}],
            "envs": [{"id": "e", "image": "i"}],
            "seeds": [1],
            "budgets": {
                "turn_budget": 300,
                "token_budget": None,
                "context_budget": None,
                "max_tokens": 16384,
            },
        }
    )
    assert cfg.budgets.turn_budget == 300
    assert cfg.budgets.token_budget is None
    assert cfg.budgets.context_budget is None
    assert cfg.budgets.max_tokens == 16384


def test_parse_config_missing_id_raises() -> None:
    with pytest.raises(ValueError, match="benchmark_id"):
        parse_config({"models": [{"id": "m"}], "envs": [{"id": "e", "image": "i"}], "seeds": [1]})


def test_parse_config_no_models_raises() -> None:
    with pytest.raises(ValueError, match="model"):
        parse_config({"benchmark_id": "b", "models": [], "envs": [{"id": "e", "image": "i"}]})


def _cfg_with_nudges(value: Any) -> Any:
    return parse_config(
        {
            "benchmark_id": "b",
            "models": [{"id": "m"}],
            "envs": [{"id": "e", "image": "i"}],
            "seeds": [1],
            **({"nudges": value} if value is not _MISSING else {}),
        }
    )


_MISSING = object()


def test_parse_config_nudges_default_empty() -> None:
    """Clean evaluation by default — empty set unless explicit.

    Mid-episode scaffolding reveals information that affects results.
    Default off so any YAML that omits the field gets a clean signal.
    """
    assert _cfg_with_nudges(_MISSING).nudges == frozenset()


def test_parse_config_nudges_true_means_all() -> None:
    cfg = _cfg_with_nudges(True)
    assert cfg.nudges == frozenset(NudgeKind)


def test_parse_config_nudges_false_means_empty() -> None:
    assert _cfg_with_nudges(False).nudges == frozenset()


def test_parse_config_nudges_list_subset() -> None:
    cfg = _cfg_with_nudges(["stuck", "voluntary"])
    assert cfg.nudges == frozenset({NudgeKind.STUCK, NudgeKind.VOLUNTARY})


def test_parse_config_nudges_list_all_alias() -> None:
    assert _cfg_with_nudges(["all"]).nudges == frozenset(NudgeKind)


def test_parse_config_nudges_unknown_kind_raises() -> None:
    with pytest.raises(ValueError, match="unknown nudge kind"):
        _cfg_with_nudges(["stuck", "bogus"])


# ---------------- DB lifecycle ----------------


@pytest.fixture
def tmp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db = tmp_path / "qed_swe_bench.sqlite"
    monkeypatch.setenv("QED_SWE_BENCH_DB", str(db))
    init_db(db)
    return db


def test_insert_queued_creates_row(tmp_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inserted = _insert_queued(
        run_id="r1",
        benchmark_id="b1",
        model="anthropic/claude-haiku-4-5",
        env=EnvSpec(id="e1", image="local/e1:latest"),
        image_digest="sha256:" + "a" * 64,
        seed=1,
        run_dir=Path("/tmp/r1"),
        nudges_used=False,
    )
    assert inserted is True
    with connect(tmp_db) as con:
        row = con.execute("SELECT * FROM runs WHERE run_id='r1'").fetchone()
    assert row["status"] == "queued"
    assert row["env_id"] == "e1"


def test_insert_queued_persists_config_snapshot_yaml(tmp_db: Path) -> None:
    """D-13: full source YAML lands on the row verbatim — comments preserved.

    Round-trip through SQLite TEXT must not strip or normalize anything;
    `qed_swe_bench reproduce` reads these bytes back and feeds them to
    parse_config without any intermediate transformation.
    """
    yaml_with_comments = (
        "# v8 baseline (D-3) — nudges OFF for clean evaluation\n"
        "benchmark_id: v8\n"
        "models:\n"
        "  - id: anthropic/claude-haiku-4-5  # cheap shakedown only\n"
        "envs:\n"
        "  - id: e1\n"
        "    image: i\n"
        "seeds: [1]\n"
        "nudges: false  # D-3: clean baseline\n"
    )
    inserted = _insert_queued(
        run_id="r1",
        benchmark_id="v8",
        model="anthropic/claude-haiku-4-5",
        env=EnvSpec(id="e1", image="i"),
        image_digest="sha256:" + "a" * 64,
        seed=1,
        run_dir=Path("/tmp/r1"),
        nudges_used=False,
        config_snapshot_yaml=yaml_with_comments,
    )
    assert inserted is True
    with connect(tmp_db) as con:
        row = con.execute(
            "SELECT config_snapshot FROM runs WHERE run_id='r1'"
        ).fetchone()
    # Bytes must be byte-identical — no normalization, comments preserved.
    assert row["config_snapshot"] == yaml_with_comments


def test_insert_queued_config_snapshot_defaults_null(tmp_db: Path) -> None:
    """Programmatic invocation without a file backing → column is NULL.
    Reproduce treats NULL as "fall back to <run_dir>/config_snapshot.yaml"."""
    _insert_queued(
        run_id="r1",
        benchmark_id="b1",
        model="m",
        env=EnvSpec(id="e1", image="i"),
        image_digest="sha256:" + "a" * 64,
        seed=1,
        run_dir=Path("/tmp/r1"),
        nudges_used=False,
    )
    with connect(tmp_db) as con:
        row = con.execute(
            "SELECT config_snapshot FROM runs WHERE run_id='r1'"
        ).fetchone()
    assert row["config_snapshot"] is None


def test_insert_queued_returns_false_on_duplicate(tmp_db: Path) -> None:
    args = dict(
        run_id="r1",
        benchmark_id="b1",
        model="m",
        env=EnvSpec(id="e1", image="i"),
        image_digest="sha256:" + "a" * 64,
        seed=1,
        run_dir=Path("/tmp"),
        nudges_used=False,
    )
    assert _insert_queued(**args) is True
    # Same (benchmark_id, model, env_id, seed) but new run_id → UNIQUE blocks.
    args2 = {**args, "run_id": "r2"}
    assert _insert_queued(**args2) is False


def test_update_finished_writes_all_columns(tmp_db: Path) -> None:
    _insert_queued(
        run_id="r1",
        benchmark_id="b1",
        model="m",
        env=EnvSpec(id="e1", image="i"),
        image_digest="sha256:x" * 8,
        seed=1,
        run_dir=Path("/tmp"),
        nudges_used=False,
    )
    _update_finished(
        run_id="r1",
        status="succeeded",
        capabilities={"crash": True, "asan": False},
        score=2.5,
        usage_totals=NormalizedUsage(
            input_tokens=100, output_tokens=20, cache_read_tokens=80, cache_creation_tokens=10
        ),
        cost_usd=0.0123,
        cost_source="pricing_table",
        runtime_s=12.5,
        turns_used=8,
        exit_reason="no_tool_calls",
        llm_route="anthropic_native",
        api_base=None,
        failure_reason=None,
    )
    with connect(tmp_db) as con:
        row = con.execute("SELECT * FROM runs WHERE run_id='r1'").fetchone()
    assert row["status"] == "succeeded"
    assert json.loads(row["capabilities"]) == {"crash": True, "asan": False}
    assert row["score"] == 2.5
    assert row["tokens_in"] == 100
    assert row["tokens_cache_creation"] == 10
    assert row["cost_usd"] == 0.0123
    assert row["cost_source"] == "pricing_table"
    assert row["llm_route"] == "anthropic_native"
    assert row["finished_at"] is not None


# ---------------- cli_oneshot dispatch helpers ----------------


def test_load_env_manifest_returns_none_when_unregistered(tmp_db: Path) -> None:
    """Backward-compat: V8 envs that predate the catalog return None and the
    orchestrator falls back to the in-MCP grade tool path."""
    assert _load_env_manifest("nonexistent-env") is None


def test_load_env_manifest_handles_missing_manifest_path(tmp_db: Path) -> None:
    from qed_swe_bench.catalog import EnvRegistration, upsert

    upsert(
        EnvRegistration(
            env_id="env-without-manifest",
            image_ref="local/x:latest",
            interface="rl.mcp.v8_task.v1",
            task_type="binary_task",
            project=None,
            bug_id=None,
            capability_class=None,
            expected_capabilities=[],
            metadata={"crev": "abc123"},  # no manifest_path key
        )
    )
    assert _load_env_manifest("env-without-manifest") is None


def test_load_env_manifest_raises_when_path_set_but_file_missing(tmp_db: Path) -> None:
    """An env registered WITH a manifest_path that fails to load must raise,
    not silently downgrade to legacy in-MCP grading. A malformed/missing
    manifest changes the benchmark's grading contract; running anyway
    would record runs against a different grader than the catalog
    promised."""
    import pytest
    from qed_swe_bench.catalog import EnvRegistration, upsert
    from qed_swe_bench.runner.env_manifest import EnvManifestLoadError

    upsert(
        EnvRegistration(
            env_id="env-with-broken-manifest",
            image_ref="local/x:latest",
            interface="rl.mcp.v8_task.v1",
            task_type="binary_task",
            project=None,
            bug_id=None,
            capability_class=None,
            expected_capabilities=[],
            metadata={"manifest_path": "/nonexistent/path/manifest.yaml"},
        )
    )
    with pytest.raises(EnvManifestLoadError, match="failed to load"):
        _load_env_manifest("env-with-broken-manifest")


def test_append_cli_oneshot_grade_log_writes_jsonl(tmp_path: Path) -> None:
    raw = {"capabilities": {"crash": True}, "score": 0.5}
    append_cli_oneshot_grade_log(tmp_path, raw)
    append_cli_oneshot_grade_log(tmp_path, {"capabilities": {"asan": True}})

    lines = (tmp_path / "grade_calls.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["source"] == "cli_oneshot"
    assert parsed[0]["result"]["capabilities"]["crash"] is True
    assert parsed[1]["result"]["capabilities"]["asan"] is True


def test_run_dir_writers_recreate_missing_directory(tmp_path: Path) -> None:
    """Each writer in run_dir.py defensively mkdir's its target.

    Motivated by an `uncaught_FileNotFoundError` on `score.json` write —
    the directory was missing at write time despite earlier mkdir at the
    top of `_run_one_body`. Aligning these writers with `TranscriptWriter`
    and `McpDockerSession.start` closes that gap.
    """
    from qed_swe_bench.runner.image_ref import ResolvedImage
    from qed_swe_bench.runner.run_dir import (
        write_cost_json,
        write_job_json,
        write_score_json,
    )

    run_dir = tmp_path / "ghost_run_dir"
    assert not run_dir.exists()

    write_score_json(
        run_dir, capabilities={"crash": True}, score=1.0, exit_reason="ok",
    )
    assert (run_dir / "score.json").is_file()

    write_cost_json(
        run_dir,
        model="m",
        usage=NormalizedUsage(),
        cost_usd=0.0,
        cost_source="mock",
        weighted_tokens=0,
    )
    assert (run_dir / "cost.json").is_file()

    write_job_json(
        run_dir=run_dir,
        nudges_used=False,
        run_id="r",
        benchmark_id="b",
        model="m",
        env=EnvSpec(id="e", image="i"),
        seed=1,
        resolved=ResolvedImage(
            image_ref="i", image_digest="sha256:" + "0" * 64, pulled=False,
        ),
        budgets=Budgets(),
    )
    assert (run_dir / "job.json").is_file()

    grade_dir = tmp_path / "ghost_grade_dir"
    append_cli_oneshot_grade_log(grade_dir, {"capabilities": {"x": True}})
    assert (grade_dir / "grade_calls.jsonl").is_file()


def test_write_score_json_records_diagnostic_fields(tmp_path: Path) -> None:
    """Always-on diagnostics (turn-as-effort): weighted_tokens_used and
    peak_per_turn_context land in score.json whether or not the
    corresponding budget was enforced."""
    from qed_swe_bench.runner.run_dir import write_score_json

    write_score_json(
        tmp_path,
        capabilities={"crash": True},
        score=1.0,
        exit_reason="no_tool_calls",
        runtime_s=10.0,
        turns_used=12,
        weighted_tokens_used=987_654,
        peak_per_turn_context=145_000,
    )
    obj = json.loads((tmp_path / "score.json").read_text())
    assert obj["weighted_tokens_used"] == 987_654
    assert obj["peak_per_turn_context"] == 145_000


def test_write_score_json_omits_diagnostic_fields_when_unset(
    tmp_path: Path,
) -> None:
    """Optional fields stay out of score.json when None — keeps the
    artifact tidy for partial-write callers and test fixtures."""
    from qed_swe_bench.runner.run_dir import write_score_json

    write_score_json(
        tmp_path,
        capabilities={"crash": True},
        score=1.0,
        exit_reason="ok",
    )
    obj = json.loads((tmp_path / "score.json").read_text())
    assert "weighted_tokens_used" not in obj
    assert "peak_per_turn_context" not in obj


def test_write_cost_json_records_served_model_and_reasoning_tokens(
    tmp_path: Path,
) -> None:
    """Audit trail for silent-downgrade detection: cost.json must record
    what the provider echoed back, not just what we asked for."""
    from qed_swe_bench.runner.run_dir import write_cost_json

    write_cost_json(
        tmp_path,
        model="openai/gpt-5.5",
        usage=NormalizedUsage(
            input_tokens=120, output_tokens=2048, reasoning_tokens=1900
        ),
        cost_usd=0.05,
        cost_source="pricing_table",
        weighted_tokens=2168,
        served_model="gpt-5.5-2026-04-23",
    )
    obj = json.loads((tmp_path / "cost.json").read_text())
    assert obj["model"] == "openai/gpt-5.5"
    assert obj["served_model"] == "gpt-5.5-2026-04-23"
    assert obj["tokens_reasoning"] == 1900


# ---------------- safety-wrap recovery + cost-cap short-circuit ----------


def test_record_failure_if_row_exists_upgrades_queued_row(tmp_db: Path) -> None:
    """Outer safety wrap path: a queued row becomes infra_failed."""
    from qed_swe_bench.runner.orchestrator import _record_failure_if_row_exists

    _insert_queued(
        run_id="rsf",
        benchmark_id="b",
        model="m",
        env=EnvSpec(id="e", image="i"),
        image_digest="sha256:" + "0" * 64,
        seed=1,
        run_dir=Path("/tmp"),
        nudges_used=False,
    )
    _record_failure_if_row_exists(
        "rsf",
        exit_reason="episode_timeout_1800s",
        failure_reason="exceeded BenchmarkConfig.episode_timeout_s",
    )
    with connect(tmp_db) as con:
        row = con.execute("SELECT status, exit_reason, failure_reason "
                          "FROM runs WHERE run_id='rsf'").fetchone()
    assert row["status"] == "infra_failed"
    assert row["exit_reason"] == "episode_timeout_1800s"
    assert "episode_timeout_s" in row["failure_reason"]


def test_record_failure_if_row_exists_no_op_when_row_missing(tmp_db: Path) -> None:
    """No row → no error. Outer wrap shouldn't itself raise."""
    from qed_swe_bench.runner.orchestrator import _record_failure_if_row_exists

    # No INSERT first; simulate a timeout that fired before _insert_queued.
    _record_failure_if_row_exists(
        "no-such-row",
        exit_reason="timeout",
        failure_reason="early",
    )  # must not raise


def test_record_failure_if_row_exists_upgrades_running_row(tmp_db: Path) -> None:
    """A `running` row that crashed mid-episode (timeout, OOM) must also
    be upgradable to infra_failed by the outer safety wrap. Without this
    a crash mid-episode would leave the row stuck at `running` and the
    UNIQUE constraint would block retries."""
    from qed_swe_bench.runner.orchestrator import _record_failure_if_row_exists

    _insert_queued(
        run_id="rrun",
        benchmark_id="b",
        model="m",
        env=EnvSpec(id="e", image="i"),
        image_digest="sha256:" + "0" * 64,
        seed=1,
        run_dir=Path("/tmp"),
        nudges_used=False,
    )
    assert _mark_running("rrun") is True
    _record_failure_if_row_exists(
        "rrun",
        exit_reason="episode_timeout_30s",
        failure_reason="exceeded BenchmarkConfig.episode_timeout_s",
    )
    with connect(tmp_db) as con:
        row = con.execute(
            "SELECT status, exit_reason FROM runs WHERE run_id='rrun'"
        ).fetchone()
    assert row["status"] == "infra_failed"
    assert row["exit_reason"] == "episode_timeout_30s"


def test_record_failure_if_row_exists_skips_finished_runs(tmp_db: Path) -> None:
    """A row already in `succeeded` must NOT be downgraded to infra_failed.

    The recovery path's `WHERE status='queued'` clause is what makes the
    outer safety wrap idempotent — running it twice can't clobber a
    completed run.
    """
    from qed_swe_bench.runner.orchestrator import _record_failure_if_row_exists

    _insert_queued(
        run_id="ok",
        benchmark_id="b",
        model="m",
        env=EnvSpec(id="e", image="i"),
        image_digest="sha256:" + "0" * 64,
        seed=1,
        run_dir=Path("/tmp"),
        nudges_used=False,
    )
    _update_finished(
        run_id="ok",
        status="succeeded",
        capabilities={"crash": True},
        score=1.0,
        usage_totals=NormalizedUsage(),
        cost_usd=0.05,
        cost_source="pricing_table",
        runtime_s=10.0,
        turns_used=5,
        exit_reason="no_tool_calls",
        llm_route="anthropic_native",
        api_base=None,
        failure_reason=None,
    )
    _record_failure_if_row_exists("ok", exit_reason="x", failure_reason="y")
    with connect(tmp_db) as con:
        status = con.execute("SELECT status FROM runs WHERE run_id='ok'").fetchone()["status"]
    assert status == "succeeded"


# ---------------- mark_running -------------------------------


def test_mark_running_promotes_queued_row(tmp_db: Path) -> None:
    """Transition queued → running and seed last_heartbeat. Distinguishes
    rows actively burning compute from rows still waiting for a slot."""
    _insert_queued(
        run_id="r1",
        benchmark_id="b",
        model="m",
        env=EnvSpec(id="e", image="i"),
        image_digest="sha256:" + "0" * 64,
        seed=1,
        run_dir=Path("/tmp"),
        nudges_used=False,
    )
    moved = _mark_running("r1")
    assert moved is True
    with connect(tmp_db) as con:
        row = con.execute(
            "SELECT status, last_heartbeat FROM runs WHERE run_id='r1'"
        ).fetchone()
    assert row["status"] == "running"
    # last_heartbeat seeded immediately so the stale-sweep doesn't reap
    # the row before the HeartbeatTracker's first periodic tick.
    assert row["last_heartbeat"] is not None


def test_mark_running_is_noop_for_non_queued_row(tmp_db: Path) -> None:
    """If the row already moved past queued (e.g. a sibling sweep marked
    it infra_failed, or the same orchestrator already promoted it),
    mark_running is a safe no-op."""
    _insert_queued(
        run_id="r2",
        benchmark_id="b",
        model="m",
        env=EnvSpec(id="e", image="i"),
        image_digest="sha256:" + "0" * 64,
        seed=1,
        run_dir=Path("/tmp"),
        nudges_used=False,
    )
    assert _mark_running("r2") is True
    # Second call: already running, not queued → no rows updated.
    assert _mark_running("r2") is False


# ---------------- claim_for_resume ---------------------------


def test_claim_for_resume_flips_infra_failed_to_running(tmp_db: Path) -> None:
    """A resume claim atomically moves a failed row to running and seeds
    last_heartbeat. Without this, the row would stay infra_failed during
    the entire resume — observers couldn't tell in-flight from terminal,
    and the heartbeat invariant would be violated."""
    from qed_swe_bench.runner.runs_db import claim_for_resume
    _insert_queued(
        run_id="r-claim-1",
        benchmark_id="b",
        model="m",
        env=EnvSpec(id="e", image="i"),
        image_digest="sha256:" + "0" * 64,
        seed=1,
        run_dir=Path("/tmp"),
        nudges_used=False,
    )
    _update_finished(
        run_id="r-claim-1",
        status="infra_failed",
        capabilities=None,
        score=None,
        usage_totals=NormalizedUsage(),
        cost_usd=None,
        cost_source="unknown",
        runtime_s=10.0,
        turns_used=5,
        exit_reason="crashed: TimeoutError",
        llm_route="litellm",
        api_base=None,
        failure_reason=None,
    )
    assert claim_for_resume("r-claim-1") is True
    with connect(tmp_db) as con:
        row = con.execute(
            "SELECT status, last_heartbeat, finished_at FROM runs WHERE run_id='r-claim-1'"
        ).fetchone()
    assert row["status"] == "running"
    assert row["last_heartbeat"] is not None
    # finished_at must be cleared on claim, otherwise mark_stale_queued_as_failed
    # (which filters AND finished_at IS NULL) silently skips this row when its
    # heartbeat goes stale, leaving status=running forever.
    assert row["finished_at"] is None


def test_claim_for_resume_handles_model_failed(tmp_db: Path) -> None:
    """Both infra_failed and model_failed are claim targets. The
    `is_resumable` predicate further filters which to attempt; this DB
    function is the lower-level claim mechanism."""
    from qed_swe_bench.runner.runs_db import claim_for_resume
    _insert_queued(
        run_id="r-claim-2",
        benchmark_id="b",
        model="m",
        env=EnvSpec(id="e", image="i"),
        image_digest="sha256:" + "0" * 64,
        seed=1,
        run_dir=Path("/tmp"),
        nudges_used=False,
    )
    _update_finished(
        run_id="r-claim-2",
        status="model_failed",
        capabilities=None,
        score=None,
        usage_totals=NormalizedUsage(),
        cost_usd=None,
        cost_source="unknown",
        runtime_s=10.0,
        turns_used=5,
        exit_reason="error: Timeout",
        llm_route="litellm",
        api_base=None,
        failure_reason=None,
    )
    assert claim_for_resume("r-claim-2") is True


def test_claim_for_resume_loses_race_returns_false(tmp_db: Path) -> None:
    """Two concurrent --resume-failed sweeps racing on the same row:
    first claim flips status to running, second sees status='running'
    and the WHERE clause excludes it. Race winner gets True, loser gets
    False, both invocations safe — no double-write."""
    from qed_swe_bench.runner.runs_db import claim_for_resume
    _insert_queued(
        run_id="r-claim-3",
        benchmark_id="b",
        model="m",
        env=EnvSpec(id="e", image="i"),
        image_digest="sha256:" + "0" * 64,
        seed=1,
        run_dir=Path("/tmp"),
        nudges_used=False,
    )
    _update_finished(
        run_id="r-claim-3",
        status="infra_failed",
        capabilities=None,
        score=None,
        usage_totals=NormalizedUsage(),
        cost_usd=None,
        cost_source="unknown",
        runtime_s=0.0,
        turns_used=0,
        exit_reason="crashed: TimeoutError",
        llm_route="litellm",
        api_base=None,
        failure_reason=None,
    )
    assert claim_for_resume("r-claim-3") is True
    # Second concurrent claim must lose.
    assert claim_for_resume("r-claim-3") is False


def test_claim_for_resume_refuses_succeeded_row(tmp_db: Path) -> None:
    """Don't let a misclassified is_resumable result accidentally clobber
    a succeeded row into running."""
    from qed_swe_bench.runner.runs_db import claim_for_resume
    _insert_queued(
        run_id="r-claim-4",
        benchmark_id="b",
        model="m",
        env=EnvSpec(id="e", image="i"),
        image_digest="sha256:" + "0" * 64,
        seed=1,
        run_dir=Path("/tmp"),
        nudges_used=False,
    )
    _update_finished(
        run_id="r-claim-4",
        status="succeeded",
        capabilities={},
        score=0.0,
        usage_totals=NormalizedUsage(),
        cost_usd=0.0,
        cost_source="mock",
        runtime_s=0.0,
        turns_used=0,
        exit_reason="no_tool_calls",
        llm_route="mock",
        api_base=None,
        failure_reason=None,
    )
    assert claim_for_resume("r-claim-4") is False
    with connect(tmp_db) as con:
        status = con.execute(
            "SELECT status FROM runs WHERE run_id='r-claim-4'"
        ).fetchone()["status"]
    assert status == "succeeded"


def test_mark_running_does_not_clobber_finished_row(tmp_db: Path) -> None:
    """Defensive: if a row somehow reaches succeeded/infra_failed before
    mark_running is called (race with a sweep), don't downgrade it."""
    _insert_queued(
        run_id="r3",
        benchmark_id="b",
        model="m",
        env=EnvSpec(id="e", image="i"),
        image_digest="sha256:" + "0" * 64,
        seed=1,
        run_dir=Path("/tmp"),
        nudges_used=False,
    )
    _update_finished(
        run_id="r3",
        status="succeeded",
        capabilities={},
        score=0.0,
        usage_totals=NormalizedUsage(),
        cost_usd=0.0,
        cost_source="mock",
        runtime_s=0.0,
        turns_used=0,
        exit_reason="no_tool_calls",
        llm_route="mock",
        api_base=None,
        failure_reason=None,
    )
    assert _mark_running("r3") is False
    with connect(tmp_db) as con:
        status = con.execute(
            "SELECT status FROM runs WHERE run_id='r3'"
        ).fetchone()["status"]
    assert status == "succeeded"


# ---------------- ExceptionGroup unwrap ------------------------


def test_unwrap_exception_group_returns_leaf_for_single_child() -> None:
    """anyio's TaskGroup wraps anything raised in a child task; the
    orchestrator's failure record should name the actual leaf cause,
    not the TaskGroup wrapper."""
    from qed_swe_bench.runner.orchestrator import _unwrap_exception_group

    inner = ValueError("real cause")
    eg = BaseExceptionGroup("outer-task-group", [inner])
    assert _unwrap_exception_group(eg) is inner


def test_unwrap_exception_group_walks_nested_groups() -> None:
    """Some MCP/anyio paths produce nested ExceptionGroups; unwrap to
    the deepest leaf."""
    from qed_swe_bench.runner.orchestrator import _unwrap_exception_group

    leaf = RuntimeError("deepest cause")
    eg1 = BaseExceptionGroup("inner-group", [leaf])
    eg2 = BaseExceptionGroup("outer-group", [eg1])
    assert _unwrap_exception_group(eg2) is leaf


def test_unwrap_exception_group_passes_through_plain_exceptions() -> None:
    from qed_swe_bench.runner.orchestrator import _unwrap_exception_group

    inner = KeyError("not in a group")
    assert _unwrap_exception_group(inner) is inner


def test_format_failure_names_inner_class_with_via_suffix() -> None:
    """`exit_reason` should report the leaf class; `failure_reason`
    should mention the wrapper as `(via …)` so audits know it came
    through a TaskGroup."""
    from qed_swe_bench.runner.orchestrator import _format_failure

    leaf = ConnectionResetError("provider hung up")
    # Python auto-narrows BaseExceptionGroup to ExceptionGroup when
    # all children are non-BaseException, so the wrapper's runtime
    # class is `ExceptionGroup` rather than `BaseExceptionGroup`.
    eg = BaseExceptionGroup("wrap", [leaf])
    exit_reason, failure_reason = _format_failure(eg)
    assert exit_reason == "crashed: ConnectionResetError"
    assert "provider hung up" in failure_reason
    assert "via ExceptionGroup" in failure_reason


def test_format_failure_no_via_suffix_for_plain_exceptions() -> None:
    from qed_swe_bench.runner.orchestrator import _format_failure

    exit_reason, failure_reason = _format_failure(ValueError("plain"))
    assert exit_reason == "crashed: ValueError"
    assert "(via" not in failure_reason


# ---------------- bug vs infra exception classification -------------------


def test_is_infra_exception_recognises_network_and_timeout() -> None:
    """Infra-shaped exceptions: timeout, network, broken pipe, image_ref,
    grader, and anything from a known provider/transport module."""
    from qed_swe_bench.runner.orchestrator import _is_infra_exception
    from qed_swe_bench.runner.cli_oneshot_grader import GraderError
    from qed_swe_bench.runner.image_ref import ImageRefError

    assert _is_infra_exception(TimeoutError("episode timed out")) is True
    assert _is_infra_exception(ConnectionResetError("hung up")) is True
    assert _is_infra_exception(BrokenPipeError("pipe died")) is True
    assert _is_infra_exception(ImageRefError("digest resolve failed")) is True
    assert _is_infra_exception(GraderError("grader exited 1")) is True


def test_is_infra_exception_rejects_code_bug_shapes() -> None:
    """Code-bug-shaped exceptions stay False so the outer wrap labels them
    `bug_<Type>`. FileNotFoundError is the canonical motivating case (a
    missing run_dir on score.json write)."""
    from qed_swe_bench.runner.orchestrator import _is_infra_exception

    assert _is_infra_exception(FileNotFoundError("score.json")) is False
    assert _is_infra_exception(KeyError("missing key")) is False
    assert _is_infra_exception(AttributeError("no attr")) is False
    assert _is_infra_exception(TypeError("bad type")) is False
    assert _is_infra_exception(ValueError("plain")) is False
    assert _is_infra_exception(IndexError("oob")) is False
    assert _is_infra_exception(AssertionError("never")) is False


def test_is_infra_exception_recognises_provider_module_exceptions() -> None:
    """Duck-type by `__module__`. Construct a fake exception class living
    under a known infra module name to confirm the module check fires."""
    from qed_swe_bench.runner.orchestrator import _is_infra_exception

    class FakeHttpxErr(Exception):
        pass

    FakeHttpxErr.__module__ = "httpx._exceptions"
    assert _is_infra_exception(FakeHttpxErr("sock closed")) is True

    class FakeRandomErr(Exception):
        pass

    FakeRandomErr.__module__ = "some.random.module"
    assert _is_infra_exception(FakeRandomErr("nope")) is False


@pytest.mark.asyncio
async def test_cost_cap_short_circuit_records_infra_failed(tmp_db: Path) -> None:
    """Cost-cap path: tuple is recorded but no docker work happens.

    Sets up a SpendTracker already past its cap, then invokes the
    relevant orchestrator branch in a way that exercises the
    short-circuit without spinning a real docker container.
    """
    from qed_swe_bench.runner.orchestrator import _record_infra_failure
    from qed_swe_bench.runner.spend_tracker import SpendTracker

    tracker = SpendTracker(cap_usd=0.05)
    await tracker.add(0.10)  # already over cap
    assert await tracker.cap_exceeded() is True

    # Insert a queued row, then run the in-body cap-exceeded recovery
    # exactly the way `_run_one_body` does.
    _insert_queued(
        run_id="cap",
        benchmark_id="b",
        model="m",
        env=EnvSpec(id="e", image="i"),
        image_digest="sha256:" + "0" * 64,
        seed=1,
        run_dir=Path("/tmp"),
        nudges_used=False,
    )
    total = await tracker.total()
    _record_infra_failure(
        "cap",
        exit_reason="cost_cap_exceeded",
        failure_reason=f"running total ${total:.2f} hit cap ${tracker.cap_usd:.2f}",
    )
    with connect(tmp_db) as con:
        row = con.execute(
            "SELECT status, exit_reason, failure_reason FROM runs WHERE run_id='cap'"
        ).fetchone()
    assert row["status"] == "infra_failed"
    assert row["exit_reason"] == "cost_cap_exceeded"
    assert "cap" in row["failure_reason"].lower()
