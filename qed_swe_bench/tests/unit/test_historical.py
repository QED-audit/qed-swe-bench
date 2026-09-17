"""historical.py — bench-v8 eval/<bug>_runN/ → runs row."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from qed_swe_bench.db.schema import connect, init_db
from qed_swe_bench.historical import (
    ImportedRun,
    _normalize_vr_agent_env_id,
    _parse_dir_name,
    _sum_usage,
    import_eval_dir,
    import_legacy_vr_agent_dir,
    parse_eval_run,
    parse_legacy_vr_agent_run,
)

# ---------- helpers ----------


def _make_eval_dir(tmp: Path, name: str, *, config: dict | None = None,
                    grade_entries: list[dict] | None = None,
                    transcript_entries: list[dict] | None = None) -> Path:
    d = tmp / name
    d.mkdir()
    if config is not None:
        (d / "config.json").write_text(json.dumps(config))
    if grade_entries is not None:
        with (d / "grade_calls.jsonl").open("w") as f:
            for e in grade_entries:
                f.write(json.dumps(e) + "\n")
    if transcript_entries is not None:
        with (d / "transcript.jsonl").open("w") as f:
            for e in transcript_entries:
                f.write(json.dumps(e) + "\n")
    return d


# ---------- _parse_dir_name ----------


def test_parse_dir_name_basic() -> None:
    assert _parse_dir_name("CVE-2024-0517_run2") == ("CVE-2024-0517", 2)
    assert _parse_dir_name("crbug-1509576_run1") == ("crbug-1509576", 1)


def test_parse_dir_name_rejects_non_run() -> None:
    assert _parse_dir_name("results.py") is None
    assert _parse_dir_name("CVE-1234") is None
    assert _parse_dir_name("foo_runX") is None


def test_parse_dir_name_handles_underscores_in_bug_id() -> None:
    """Bug IDs can have underscores (e.g. some Bountybench tasks). The regex is greedy."""
    assert _parse_dir_name("some_weird_id_run42") == ("some_weird_id", 42)


# ---------- _sum_usage ----------


def test_sum_usage_aggregates_ai_turns(tmp_path: Path) -> None:
    transcript = tmp_path / "transcript.jsonl"
    with transcript.open("w") as f:
        f.write(json.dumps({"role": "system", "content": "x"}) + "\n")
        f.write(json.dumps({"role": "ai", "content": "", "usage": {
            "input_tokens": 100, "output_tokens": 20, "cache_read": 80, "cache_creation": 10
        }}) + "\n")
        f.write(json.dumps({"role": "tool", "content": "y", "tool_call_id": "t", "name": "exec"}) + "\n")
        f.write(json.dumps({"role": "ai", "content": "", "usage": {
            "input_tokens": 200, "output_tokens": 50
        }}) + "\n")
    usage, turns, per_call = _sum_usage(transcript)
    assert turns == 2
    assert usage.input_tokens == 300
    assert usage.output_tokens == 70
    assert usage.cache_read_tokens == 80
    assert usage.cache_creation_tokens == 10
    # per_call captures each ai turn's usage individually so cost
    # tier-bucketing can run per-call.
    assert len(per_call) == 2
    assert per_call[0].input_tokens == 100 and per_call[0].cache_read_tokens == 80
    assert per_call[1].input_tokens == 200 and per_call[1].cache_read_tokens == 0


def test_sum_usage_missing_file_returns_zeros(tmp_path: Path) -> None:
    usage, turns, per_call = _sum_usage(tmp_path / "nope.jsonl")
    assert usage.input_tokens == 0
    assert turns == 0
    assert per_call == ()


# ---------- parse_eval_run ----------


def test_parse_eval_run_full(tmp_path: Path) -> None:
    d = _make_eval_dir(
        tmp_path,
        "CVE-2024-0517_run2",
        config={"model": "anthropic/claude-opus-4-6", "image": "ecr/foo:latest"},
        grade_entries=[
            {"ts": "T", "path": "/x", "result": {"capabilities": {"crash": True, "diff": True}},
             "duration_s": 1.0},
        ],
        transcript_entries=[
            {"role": "ai", "content": "", "usage": {"input_tokens": 10, "output_tokens": 5}},
        ],
    )
    run = parse_eval_run(d)
    assert isinstance(run, ImportedRun)
    assert run.bug == "CVE-2024-0517"
    assert run.seed == 2
    assert run.model == "anthropic/claude-opus-4-6"
    assert run.image_ref == "ecr/foo:latest"
    assert run.capabilities == {"crash": True, "diff": True}
    assert run.score == 2.0  # crash=1 + diff=1
    assert run.usage.input_tokens == 10
    assert run.turns_used == 1
    assert run.grade_count == 1


def test_parse_eval_run_uses_default_model_when_missing(tmp_path: Path) -> None:
    d = _make_eval_dir(tmp_path, "x_run1", config={})
    run = parse_eval_run(d)
    assert run is not None
    assert run.model == "anthropic/claude-opus-4-6"  # default


def test_parse_eval_run_synthesizes_image_ref_when_missing(tmp_path: Path) -> None:
    d = _make_eval_dir(tmp_path, "x_run1", config={})
    run = parse_eval_run(d)
    assert run is not None
    assert run.image_ref.startswith("imported://")


def test_parse_eval_run_returns_none_for_non_run_dir(tmp_path: Path) -> None:
    d = _make_eval_dir(tmp_path, "not-a-run-dir", config={})
    assert parse_eval_run(d) is None


# ---------- import_eval_dir ----------


def test_import_eval_dir_inserts_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = tmp_path / "qed_swe_bench.sqlite"
    monkeypatch.setenv("QED_SWE_BENCH_DB", str(db))
    init_db(db)
    eval_root = tmp_path / "eval"
    eval_root.mkdir()
    _make_eval_dir(
        eval_root, "bug1_run1",
        config={"model": "anthropic/claude-opus-4-6", "image": "i"},
        grade_entries=[{"ts": "T", "path": "/x", "result": {"capabilities": {"crash": True}},
                        "duration_s": 0.1}],
        transcript_entries=[{"role": "ai", "content": "", "usage": {"input_tokens": 10, "output_tokens": 5}}],
    )
    _make_eval_dir(
        eval_root, "bug2_run1",
        config={"model": "anthropic/claude-opus-4-6", "image": "i"},
    )

    histo = import_eval_dir(eval_root, benchmark_id="b-imported", nudges_used=True)
    assert histo["imported"] == 2
    assert histo["unparseable"] == 0

    with connect(db) as con:
        rows = con.execute(
            "SELECT env_id, capabilities, provenance FROM runs WHERE benchmark_id='b-imported' ORDER BY env_id"
        ).fetchall()
    assert len(rows) == 2
    assert rows[0]["env_id"] == "bug1"
    assert rows[0]["provenance"] == "imported_from_eval"
    assert json.loads(rows[0]["capabilities"]) == {"crash": True}


def test_import_eval_dir_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = tmp_path / "qed_swe_bench.sqlite"
    monkeypatch.setenv("QED_SWE_BENCH_DB", str(db))
    init_db(db)
    eval_root = tmp_path / "eval"
    eval_root.mkdir()
    _make_eval_dir(
        eval_root, "bug1_run1",
        config={"model": "anthropic/claude-opus-4-6", "image": "i"},
    )

    h1 = import_eval_dir(eval_root, benchmark_id="b1", nudges_used=True)
    h2 = import_eval_dir(eval_root, benchmark_id="b1", nudges_used=True)
    assert h1["imported"] == 1
    assert h2["imported"] == 0
    assert h2["skipped_duplicate"] == 1


def test_import_eval_dir_handles_unparseable_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = tmp_path / "qed_swe_bench.sqlite"
    monkeypatch.setenv("QED_SWE_BENCH_DB", str(db))
    init_db(db)
    eval_root = tmp_path / "eval"
    eval_root.mkdir()
    (eval_root / "not-a-run").mkdir()
    _make_eval_dir(eval_root, "bug_run1", config={"model": "m", "image": "i"})

    histo = import_eval_dir(eval_root, nudges_used=True)
    assert histo["imported"] == 1
    assert histo["unparseable"] == 1


# ---------- import_native_run_dir nudges_used resolution ----------


def _make_native_run_dir(
    tmp: Path,
    run_id: str,
    *,
    job_extras: dict | None = None,
    config_snapshot_yaml: str | None = None,
) -> Path:
    """Build a minimal native run-dir for import_native_run_dir tests.

    Always writes job.json with the required fields (omits nudges_used
    by default — tests opt in via job_extras).
    """
    d = tmp / run_id
    d.mkdir()
    job = {
        "run_id": run_id,
        "benchmark_id": "v8",
        "model": "anthropic/claude-haiku-4-5",
        "env_id": "e1",
        "image_ref": "img",
        "image_digest": "sha256:" + "0" * 64,
        "task_type": "binary_task",
        "interface": "rl.mcp.v8_task.v1",
        "seed": 1,
        "started_at": "2026-05-04T12:00:00+00:00",
        "provenance": "native",
    }
    if job_extras:
        job.update(job_extras)
    (d / "job.json").write_text(json.dumps(job))
    if config_snapshot_yaml is not None:
        (d / "config_snapshot.yaml").write_text(config_snapshot_yaml)
    return d


def test_import_native_run_dir_reads_nudges_used_from_job_json(
    tmp_path: Path,
) -> None:
    """Authoritative path: when job.json has the field, use it directly."""
    from qed_swe_bench.historical import import_native_run_dir

    d = _make_native_run_dir(
        tmp_path, "r1", job_extras={"nudges_used": True},
    )
    row = import_native_run_dir(d)
    assert row["nudges_used"] == 1


def test_import_native_run_dir_falls_back_to_config_snapshot_yaml(
    tmp_path: Path,
) -> None:
    """Legacy path: job.json predates the field but config_snapshot.yaml
    captured the nudges setting. The fallback parses the YAML and
    derives the bool — same parser the orchestrator uses."""
    from qed_swe_bench.historical import import_native_run_dir

    yaml_off = (
        "benchmark_id: v8\n"
        "models:\n"
        "  - id: anthropic/claude-haiku-4-5\n"
        "envs:\n"
        "  - id: e1\n"
        "    image: img\n"
        "seeds: [1]\n"
        "nudges: false\n"
    )
    d = _make_native_run_dir(
        tmp_path, "r2", config_snapshot_yaml=yaml_off,
    )
    row = import_native_run_dir(d)
    assert row["nudges_used"] == 0

    yaml_on = yaml_off.replace("nudges: false", "nudges: true")
    d2 = _make_native_run_dir(
        tmp_path, "r3", config_snapshot_yaml=yaml_on,
    )
    row2 = import_native_run_dir(d2)
    assert row2["nudges_used"] == 1


def test_import_native_run_dir_defaults_false_when_no_signal(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """Very old run-dir: no nudges_used in job.json, no config_snapshot.yaml.
    Defaults to False and logs a warning."""
    import logging

    from qed_swe_bench.historical import import_native_run_dir

    d = _make_native_run_dir(tmp_path, "r4")  # no extras, no snapshot
    with caplog.at_level(logging.WARNING):
        row = import_native_run_dir(d)
    assert row["nudges_used"] == 0
    assert any("nudges" in rec.message.lower() for rec in caplog.records)


# ---------- legacy vr-agent importer ----------


def _make_vr_agent_rep_dir(
    transcripts_root: Path,
    *,
    model: str,
    bug: str,
    rep: int,
    result: dict,
    grade_entries: list[dict] | None = None,
) -> Path:
    rep_dir = transcripts_root / model / bug / f"rep{rep}"
    rep_dir.mkdir(parents=True)
    (rep_dir / "result.json").write_text(json.dumps(result))
    if grade_entries is not None:
        with (rep_dir / "grade_calls.jsonl").open("w") as f:
            for e in grade_entries:
                f.write(json.dumps(e) + "\n")
    return rep_dir


def test_normalize_vr_agent_env_id() -> None:
    assert _normalize_vr_agent_env_id("CVE-2024-1939") == "v8-e01"
    assert _normalize_vr_agent_env_id("crbug-1509576") == "v8-e25"
    # idempotent on already-normalized inputs (defensive)
    assert _normalize_vr_agent_env_id("v8-cve-2024-1939") == "v8-e01"


def test_parse_legacy_vr_agent_run_full(tmp_path: Path) -> None:
    rep_dir = _make_vr_agent_rep_dir(
        tmp_path,
        model="claude-opus-4-6",
        bug="CVE-2024-1939",
        rep=2,
        result={
            "cve": "CVE-2024-1939",
            "model": "claude-opus-4-6",
            "run": 2,
            "agent_run_ok": True,
            "agent_turns": 142,
            "agent_duration_s": 6321.5,
            "grade_bitmap": {
                "cov_func": True, "cov_line": True, "diff": True, "crash": True,
            },
            "n_caps": 4,
            "token_totals": {"in": 100, "out": 20, "cache_read": 80, "cache_creat": 5},
            "cost_usd": 12.34,
            "error": "",
        },
    )
    parsed = parse_legacy_vr_agent_run(rep_dir)
    assert parsed is not None
    assert parsed["raw_bug"] == "CVE-2024-1939"
    assert parsed["raw_model"] == "claude-opus-4-6"
    assert parsed["seed"] == 2
    assert parsed["status"] == "succeeded"
    assert json.loads(parsed["capabilities_json"]) == {
        "cov_func": True, "cov_line": True, "diff": True, "crash": True,
    }
    assert parsed["score"] == 4.0
    assert parsed["cost_usd"] == 12.34
    assert parsed["tokens_in"] == 100
    assert parsed["turns_used"] == 142


def test_parse_legacy_vr_agent_run_marks_failed_on_agent_run_ok_false(tmp_path: Path) -> None:
    rep_dir = _make_vr_agent_rep_dir(
        tmp_path,
        model="claude-opus-4-6",
        bug="CVE-2024-0517",
        rep=1,
        result={
            "cve": "CVE-2024-0517",
            "model": "claude-opus-4-6",
            "run": 1,
            "agent_run_ok": False,
            "agent_turns": 286,
            "agent_duration_s": 18000.0,
            "grade_bitmap": {"cov_func": True, "cov_line": True, "diff": True},
            "n_caps": 3,
            "token_totals": {"in": 1, "out": 1, "cache_read": 0, "cache_creat": 0},
            "cost_usd": None,  # source datasets sometimes redact cost
            "error": "timeout",
        },
    )
    parsed = parse_legacy_vr_agent_run(rep_dir)
    assert parsed is not None
    assert parsed["status"] == "model_failed"
    assert parsed["cost_usd"] is None
    # capabilities reached BEFORE the timeout still count
    assert json.loads(parsed["capabilities_json"]) == {
        "cov_func": True, "cov_line": True, "diff": True,
    }


def test_parse_legacy_vr_agent_run_returns_none_for_missing_result(tmp_path: Path) -> None:
    rep_dir = tmp_path / "claude-opus-4-6" / "CVE-2024-1939" / "rep1"
    rep_dir.mkdir(parents=True)
    # No result.json in this dir.
    assert parse_legacy_vr_agent_run(rep_dir) is None


def test_import_legacy_vr_agent_dir_inserts_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = tmp_path / "qed_swe_bench.sqlite"
    monkeypatch.setenv("QED_SWE_BENCH_DB", str(db))
    init_db(db)
    root = tmp_path / "vr-agent-results"
    transcripts = root / "transcripts"
    transcripts.mkdir(parents=True)
    (root / "scores").mkdir()

    _make_vr_agent_rep_dir(
        transcripts,
        model="claude-opus-4-6", bug="CVE-2024-1939", rep=1,
        result={
            "cve": "CVE-2024-1939", "model": "claude-opus-4-6", "run": 1,
            "agent_run_ok": True, "agent_turns": 100, "agent_duration_s": 60.0,
            "grade_bitmap": {"cov_func": True}, "n_caps": 1,
            "token_totals": {"in": 10, "out": 5, "cache_read": 0, "cache_creat": 0},
            "cost_usd": 1.50, "error": "",
        },
    )
    # An unmapped model slug — exercises the fallthrough where the
    # importer keeps the raw slug rather than rewriting it. Also has a
    # null cost to verify NULL preservation through the DB insert.
    _make_vr_agent_rep_dir(
        transcripts,
        model="unknown-model-slug", bug="CVE-2024-0517", rep=1,
        result={
            "cve": "CVE-2024-0517", "model": "unknown-model-slug", "run": 1,
            "agent_run_ok": True, "agent_turns": 200, "agent_duration_s": 120.0,
            "grade_bitmap": {"cov_func": True, "cov_line": True}, "n_caps": 2,
            "token_totals": {"in": 20, "out": 10, "cache_read": 0, "cache_creat": 0},
            "cost_usd": None, "error": "",
        },
    )

    # An "excluded" row, listed in scores/manifest.csv.
    _make_vr_agent_rep_dir(
        transcripts,
        model="claude-sonnet-4-5", bug="CVE-2024-1939", rep=1,
        result={
            "cve": "CVE-2024-1939", "model": "claude-sonnet-4-5", "run": 1,
            "agent_run_ok": True, "agent_turns": 50, "agent_duration_s": 30.0,
            "grade_bitmap": {}, "n_caps": 0,
            "token_totals": {"in": 5, "out": 2, "cache_read": 0, "cache_creat": 0},
            "cost_usd": 0.10, "error": "",
        },
    )
    (root / "scores" / "manifest.csv").write_text(
        "model,cve,rep,status,src,has_transcript,has_log,excluded\n"
        "claude-opus-4-6,CVE-2024-1939,1,ok,test,True,True,\n"
        "unknown-model-slug,CVE-2024-0517,1,ok,test,True,True,\n"
        "claude-sonnet-4-5,CVE-2024-1939,1,ok,test,True,True,api_500\n"
    )

    histo = import_legacy_vr_agent_dir(root, benchmark_id="b-vr")
    assert histo["imported"] == 3
    assert histo["excluded"] == 1
    assert histo["unparseable"] == 0
    assert histo["skipped_duplicate"] == 0

    with connect(db) as con:
        rows = con.execute(
            "SELECT model, env_id, status, score, cost_usd, provenance "
            "FROM runs WHERE benchmark_id='b-vr' ORDER BY model"
        ).fetchall()
    assert len(rows) == 3
    by_model = {r["model"]: dict(r) for r in rows}
    # Mapped slugs get rewritten to the qualified provider/model form;
    # unmapped slugs (`unknown-model-slug`) pass through unchanged so
    # imports never silently drop a row whose model isn't in the map.
    assert "anthropic/claude-opus-4-6" in by_model
    assert "unknown-model-slug" in by_model
    assert "anthropic/claude-sonnet-4-5" in by_model
    # Env normalization
    assert by_model["anthropic/claude-opus-4-6"]["env_id"] == "v8-e01"
    # Status: ok rows succeeded, excluded row → infrastructure_failed
    assert by_model["anthropic/claude-opus-4-6"]["status"] == "succeeded"
    assert by_model["anthropic/claude-sonnet-4-5"]["status"] == "infrastructure_failed"
    # Cost: rows that report a number preserve it; rows with null cost
    # in the source preserve NULL through the insert.
    assert by_model["anthropic/claude-opus-4-6"]["cost_usd"] == 1.50
    assert by_model["unknown-model-slug"]["cost_usd"] is None
    # Provenance is the new tag, distinct from imported_from_eval
    for r in rows:
        assert r["provenance"] == "imported_from_legacy"


def test_import_legacy_vr_agent_dir_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = tmp_path / "qed_swe_bench.sqlite"
    monkeypatch.setenv("QED_SWE_BENCH_DB", str(db))
    init_db(db)
    root = tmp_path / "vr-agent-results"
    transcripts = root / "transcripts"
    transcripts.mkdir(parents=True)
    _make_vr_agent_rep_dir(
        transcripts,
        model="claude-opus-4-6", bug="CVE-2024-1939", rep=1,
        result={
            "cve": "CVE-2024-1939", "model": "claude-opus-4-6", "run": 1,
            "agent_run_ok": True, "agent_turns": 1, "agent_duration_s": 1.0,
            "grade_bitmap": {}, "n_caps": 0,
            "token_totals": {"in": 0, "out": 0, "cache_read": 0, "cache_creat": 0},
            "cost_usd": 0.0, "error": "",
        },
    )

    h1 = import_legacy_vr_agent_dir(root, benchmark_id="b1")
    h2 = import_legacy_vr_agent_dir(root, benchmark_id="b1")
    assert h1["imported"] == 1
    assert h2["imported"] == 0
    assert h2["skipped_duplicate"] == 1
