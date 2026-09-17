"""Unit tests for the `qed_swe_bench publish` pipeline.

Covers:
  - selection.select_canonical: status rank, tie-break, --exclude-model
  - selection.model_slug
  - revision.compute_revision: sha7 of present + missing files
  - card.write_manifest: for_upload omits excluded_models; local keeps it
  - audit_gate.gate: counts + run-id bucketing
  - bundle.build: parquet roundtrip + sidecar compression

Tests synthesize a tiny SQLite DB and fake run dirs under tmp_path; no
real qed_swe_bench DB or network required.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from qed_swe_bench.publish import (
    audit_gate,
    bundle,
    card,
    revision,
    selection,
)
from qed_swe_bench.publish.selection import model_slug, select_canonical


# ---------------- DB fixture ----------------


def _create_runs_table(con: sqlite3.Connection) -> None:
    """Mirror the columns publish/selection.py SELECTs.

    We don't run init_db() because that pulls in Config + the rlenv_images
    table; this table is the minimum surface select_canonical needs.
    """
    con.execute(
        """
        CREATE TABLE runs (
            run_id TEXT PRIMARY KEY,
            benchmark_id TEXT NOT NULL,
            model TEXT NOT NULL,
            env_id TEXT NOT NULL,
            seed INTEGER NOT NULL,
            status TEXT NOT NULL,
            score REAL,
            capabilities TEXT,
            run_dir TEXT,
            started_at TEXT,
            finished_at TEXT,
            runtime_s REAL,
            turns_used INTEGER,
            exit_reason TEXT,
            image_ref TEXT,
            image_digest TEXT,
            git_sha TEXT,
            weighted_tokens_used INTEGER,
            peak_per_turn_context INTEGER,
            cost_usd REAL,
            cost_source TEXT,
            tokens_in INTEGER,
            tokens_out INTEGER,
            tokens_cache_read INTEGER,
            tokens_cache_creation INTEGER,
            tokens_reasoning INTEGER,
            served_model TEXT
        )
        """
    )


def _insert_run(
    con: sqlite3.Connection,
    *,
    run_id: str,
    benchmark_id: str = "v8",
    model: str,
    env_id: str,
    seed: int,
    status: str,
    started_at: str,
    run_dir: Path | None = None,
    capabilities: dict[str, bool] | None = None,
    score: float | None = 1.0,
) -> None:
    con.execute(
        """
        INSERT INTO runs (
            run_id, benchmark_id, model, env_id, seed, status, score,
            capabilities, run_dir, started_at, image_ref, image_digest,
            cost_usd, cost_source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id, benchmark_id, model, env_id, seed, status, score,
            json.dumps(capabilities or {}),
            str(run_dir) if run_dir else None,
            started_at,
            "registry.example/v8:latest",
            "sha256:deadbeef",
            0.12,
            "computed",
        ),
    )


def _make_run_dir(
    tmp_path: Path,
    *,
    name: str,
    transcript_lines: int = 3,
    tool_calls_lines: int = 2,
    grade_calls_lines: int = 1,
    config_snapshot: str = "version: 1\n",
) -> Path:
    rd = tmp_path / name
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "transcript.jsonl").write_text(
        "\n".join(json.dumps({"role": "ai", "content": f"t{i}"}) for i in range(transcript_lines))
        + "\n"
    )
    (rd / "tool_calls.jsonl").write_text(
        "\n".join(json.dumps({"tool": "exec", "args": {"cmd": f"c{i}"}}) for i in range(tool_calls_lines))
        + "\n"
    )
    (rd / "grade_calls.jsonl").write_text(
        "\n".join(
            json.dumps({"path": f"poc{i}.js", "result": {"capabilities": {"cov_func": True}}})
            for i in range(grade_calls_lines)
        )
        + "\n"
    )
    (rd / "config_snapshot.yaml").write_text(config_snapshot)
    (rd / "job.json").write_text(json.dumps({"run_id": name}))
    (rd / "score.json").write_text(json.dumps({"status": "succeeded"}))
    return rd


@pytest.fixture
def db_with_runs(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    """A SQLite DB pre-populated with cells across two models, two seeds.

    Returns (db_path, {run_id: run_dir}). The db has duplicates per
    (model, env_id, seed) so tests can verify the canonical pick.
    """
    rdirs: dict[str, Path] = {
        "rA": _make_run_dir(tmp_path, name="rA"),
        "rB": _make_run_dir(tmp_path, name="rB"),
        "rC": _make_run_dir(tmp_path, name="rC"),
        "rD": _make_run_dir(tmp_path, name="rD"),
        "rE": _make_run_dir(tmp_path, name="rE"),
        "rF": _make_run_dir(tmp_path, name="rF"),
    }
    db_path = tmp_path / "test.sqlite"
    with sqlite3.connect(str(db_path)) as con:
        _create_runs_table(con)
        # Two candidates for (publicmodel, env_a, seed=1): an older
        # succeeded and a newer model_failed. succeeded should win
        # despite being older (status rank > recency).
        _insert_run(
            con, run_id="rA", model="anthropic/public-1", env_id="env_a",
            seed=1, status="succeeded", started_at="2026-05-01T00:00:00Z",
            run_dir=rdirs["rA"], capabilities={"cov_func": True},
        )
        _insert_run(
            con, run_id="rB", model="anthropic/public-1", env_id="env_a",
            seed=1, status="model_failed", started_at="2026-05-02T00:00:00Z",
            run_dir=rdirs["rB"], capabilities={},
            score=0.0,
        )
        # Single succeeded for (publicmodel, env_b, seed=1)
        _insert_run(
            con, run_id="rC", model="anthropic/public-1", env_id="env_b",
            seed=1, status="succeeded", started_at="2026-05-01T00:00:00Z",
            run_dir=rdirs["rC"], capabilities={"cov_func": True, "ace": True},
            score=16.0,
        )
        # Two model_failed for (publicmodel, env_a, seed=2): newer wins
        _insert_run(
            con, run_id="rD", model="anthropic/public-1", env_id="env_a",
            seed=2, status="model_failed", started_at="2026-05-01T00:00:00Z",
            run_dir=rdirs["rD"], capabilities={}, score=0.0,
        )
        _insert_run(
            con, run_id="rE", model="anthropic/public-1", env_id="env_a",
            seed=2, status="model_failed", started_at="2026-05-03T00:00:00Z",
            run_dir=rdirs["rE"], capabilities={}, score=0.0,
        )
        # Private model — should be excluded by --exclude-model
        _insert_run(
            con, run_id="rF", model="anthropic/private-preview", env_id="env_a",
            seed=1, status="succeeded", started_at="2026-05-01T00:00:00Z",
            run_dir=rdirs["rF"], capabilities={"cov_func": True},
        )
        # Queued row (ineligible) — should never appear regardless
        _insert_run(
            con, run_id="rG", model="anthropic/public-1", env_id="env_c",
            seed=1, status="queued", started_at="2026-05-01T00:00:00Z",
            run_dir=None, capabilities=None, score=None,
        )
    return db_path, rdirs


# ---------------- selection ----------------


class TestModelSlug:
    def test_strips_provider_prefix(self) -> None:
        assert model_slug("anthropic/claude-opus-4-7") == "claude-opus-4-7"

    def test_preserves_dots_and_digits(self) -> None:
        assert model_slug("minimax/MiniMax-M2.7") == "minimax-m2.7"

    def test_handles_no_prefix(self) -> None:
        assert model_slug("gpt-5.5") == "gpt-5.5"

    def test_collapses_other_punctuation(self) -> None:
        assert model_slug("vendor/Some_Weird Name!!") == "some-weird-name"

    def test_empty_input(self) -> None:
        assert model_slug("") == "unknown-model"


class TestSelectCanonical:
    def test_picks_succeeded_over_model_failed(
        self, db_with_runs: tuple[Path, dict[str, Path]]
    ) -> None:
        db_path, _ = db_with_runs
        cells = select_canonical(db_path, "v8", exclude_models=("anthropic/private-preview",))
        # Find the (public-1, env_a, seed=1) cell
        match = [c for c in cells
                 if c.model == "anthropic/public-1"
                 and c.env_id == "env_a"
                 and c.seed == 1]
        assert len(match) == 1
        # Must be rA (succeeded, older) not rB (model_failed, newer)
        assert match[0].run_id == "rA"
        assert match[0].status == "succeeded"

    def test_tie_break_picks_most_recent(
        self, db_with_runs: tuple[Path, dict[str, Path]]
    ) -> None:
        db_path, _ = db_with_runs
        cells = select_canonical(db_path, "v8", exclude_models=("anthropic/private-preview",))
        match = [c for c in cells
                 if c.model == "anthropic/public-1"
                 and c.env_id == "env_a"
                 and c.seed == 2]
        assert len(match) == 1
        # Both are model_failed; rE (2026-05-03) is newer than rD (2026-05-01)
        assert match[0].run_id == "rE"

    def test_exclude_model_drops_all_matching_rows(
        self, db_with_runs: tuple[Path, dict[str, Path]]
    ) -> None:
        db_path, _ = db_with_runs
        cells = select_canonical(
            db_path, "v8", exclude_models=("anthropic/private-preview",)
        )
        models = {c.model for c in cells}
        assert "anthropic/private-preview" not in models
        assert "anthropic/public-1" in models

    def test_no_exclude_keeps_private(
        self, db_with_runs: tuple[Path, dict[str, Path]]
    ) -> None:
        db_path, _ = db_with_runs
        cells = select_canonical(db_path, "v8")
        models = {c.model for c in cells}
        assert "anthropic/private-preview" in models, (
            "Without --exclude-model, the private row must be included; "
            "default-publish-everything is the documented behavior."
        )

    def test_queued_rows_never_selected(
        self, db_with_runs: tuple[Path, dict[str, Path]]
    ) -> None:
        db_path, _ = db_with_runs
        cells = select_canonical(db_path, "v8")
        statuses = {c.status for c in cells}
        assert "queued" not in statuses
        # And the env_c (queued-only) cell should be absent entirely
        assert not any(c.env_id == "env_c" for c in cells)

    def test_empty_exclusion_list_returns_everything(
        self, db_with_runs: tuple[Path, dict[str, Path]]
    ) -> None:
        db_path, _ = db_with_runs
        cells = select_canonical(db_path, "v8", exclude_models=())
        # 3 (model, env, seed) tuples for public-1 + 1 for private-preview
        assert len(cells) == 4

    def test_results_are_sorted_deterministically(
        self, db_with_runs: tuple[Path, dict[str, Path]]
    ) -> None:
        db_path, _ = db_with_runs
        c1 = select_canonical(db_path, "v8")
        c2 = select_canonical(db_path, "v8")
        assert [c.run_id for c in c1] == [c.run_id for c in c2]


# ---------------- revision ----------------


class TestComputeRevision:
    def test_returns_missing_for_absent_files(self, tmp_path: Path) -> None:
        rev = revision.compute_revision(tmp_path)
        assert rev == "v8-missing-ptmissing"

    def test_includes_short_shas(self, tmp_path: Path) -> None:
        (tmp_path / "benchmarks").mkdir()
        (tmp_path / "benchmarks" / "v8.yaml").write_text("hello\n")
        (tmp_path / "benchmarks" / "bench-v8").mkdir()
        (tmp_path / "benchmarks" / "bench-v8" / "prompt-template").mkdir()
        (tmp_path / "benchmarks" / "bench-v8" / "prompt-template" / "v8.template").write_text("x\n")
        rev = revision.compute_revision(tmp_path)
        # Format: v8-<7chars>-pt<7chars>
        parts = rev.split("-")
        assert parts[0] == "v8"
        assert len(parts[1]) == 7
        assert parts[2].startswith("pt")
        assert len(parts[2]) == 9  # "pt" + 7 chars

    def test_changes_when_yaml_changes(self, tmp_path: Path) -> None:
        (tmp_path / "benchmarks").mkdir()
        yp = tmp_path / "benchmarks" / "v8.yaml"
        yp.write_text("a\n")
        rev_a = revision.compute_revision(tmp_path)
        yp.write_text("b\n")
        rev_b = revision.compute_revision(tmp_path)
        assert rev_a != rev_b


# ---------------- audit_gate ----------------


class TestAuditGate:
    def test_clean_run_dir_zero_findings(self, tmp_path: Path) -> None:
        rd = _make_run_dir(tmp_path, name="clean")
        result = audit_gate.gate([rd])
        assert result.counts == {"high": 0, "medium": 0, "info": 0}
        assert not result.has_high

    def test_synthetic_high_finding_is_counted(self, tmp_path: Path) -> None:
        # Off-workspace write triggers a HIGH finding (C2). We synthesize
        # a tool_calls entry that writes to /etc.
        rd = tmp_path / "highrun"
        rd.mkdir()
        (rd / "tool_calls.jsonl").write_text(
            json.dumps({
                "tool": "write_file",
                "args": {"path": "/etc/evil.txt", "content": "x"},
                "result": "ok",
            }) + "\n"
        )
        (rd / "transcript.jsonl").write_text("")
        (rd / "grade_calls.jsonl").write_text("")
        (rd / "score.json").write_text("{}")
        result = audit_gate.gate([rd])
        # Don't assert exactly 1 — multiple checks may fire — only that
        # at least one HIGH was found and the run_id is recorded.
        assert result.counts["high"] >= 1
        assert result.has_high
        assert "highrun" in result.high_run_ids


# ---------------- card / manifest hygiene (load-bearing) ----------------


class TestManifestHygiene:
    def _bundle_stub(self, tmp_path: Path) -> tuple[Path, list, bundle.BundleStats, audit_gate.GateResult]:
        rd = _make_run_dir(tmp_path, name="rOne")
        cell = selection.CellRecord(
            run_id="rOne", benchmark_id="v8",
            model="anthropic/public-1", env_id="env_a", seed=1,
            status="succeeded", score=1.0,
            capabilities={"cov_func": True},
            run_dir=rd,
            started_at="2026-05-01T00:00:00Z",
            finished_at="2026-05-01T00:30:00Z",
            runtime_s=1800.0, turns_used=42,
            exit_reason="solved",
            image_ref="r/v8:latest", image_digest="sha256:x",
            git_sha=None, weighted_tokens_used=None, peak_per_turn_context=None,
            cost_usd=0.10, cost_source="computed",
            tokens_in=100, tokens_out=200,
            tokens_cache_read=0, tokens_cache_creation=0, tokens_reasoning=None,
            served_model="anthropic/public-1",
        )
        dest = tmp_path / "dist" / "ds" / "rev"
        stats = bundle.build([cell], dest)
        gate = audit_gate.gate([cell.run_dir])
        return dest, [cell], stats, gate

    def test_for_upload_excludes_excluded_models_field(
        self, tmp_path: Path
    ) -> None:
        dest, cells, stats, gate = self._bundle_stub(tmp_path)
        path = card.write_manifest(
            dest, repo_id="qed_swe_bench/v8", revision="v8-x-pty",
            cells=cells, stats=stats, gate=gate,
            license_id="cc-by-4.0",
            excluded_models=["anthropic/private-preview"],
            for_upload=True,
            filename="manifest_upload.json",
        )
        data = json.loads(path.read_text())
        assert "excluded_models" not in data, (
            "Upload manifest must never name private models — that's "
            "the whole point of --exclude-model."
        )

    def test_local_manifest_records_excluded_models(self, tmp_path: Path) -> None:
        dest, cells, stats, gate = self._bundle_stub(tmp_path)
        path = card.write_manifest(
            dest, repo_id="qed_swe_bench/v8", revision="v8-x-pty",
            cells=cells, stats=stats, gate=gate,
            license_id="cc-by-4.0",
            excluded_models=["anthropic/private-preview", "anthropic/other"],
            for_upload=False,
            filename="manifest_local.json",
        )
        data = json.loads(path.read_text())
        assert data["excluded_models"] == [
            "anthropic/other", "anthropic/private-preview",
        ], "Local manifest is the operator's audit trail; field must be present and sorted."


# ---------------- bundle (parquet roundtrip) ----------------


class TestBundle:
    def test_parquet_roundtrip(self, tmp_path: Path) -> None:
        rd = _make_run_dir(tmp_path, name="rOne")
        cell = selection.CellRecord(
            run_id="rOne", benchmark_id="v8",
            model="anthropic/public-1", env_id="env_a", seed=3,
            status="succeeded", score=2.5,
            capabilities={"cov_func": True, "ace": True},
            run_dir=rd,
            started_at="2026-05-01T00:00:00Z",
            finished_at="2026-05-01T00:30:00Z",
            runtime_s=10.0, turns_used=5,
            exit_reason="solved",
            image_ref="r/v8:latest", image_digest="sha256:x",
            git_sha="abc1234", weighted_tokens_used=1000,
            peak_per_turn_context=500,
            cost_usd=0.10, cost_source="computed",
            tokens_in=100, tokens_out=200,
            tokens_cache_read=0, tokens_cache_creation=0, tokens_reasoning=None,
            served_model="anthropic/public-1",
        )
        dest = tmp_path / "dist" / "x"
        stats = bundle.build([cell], dest)
        assert stats.n_cells == 1
        assert stats.parquet_path.is_file()
        # Sidecars landed at expected paths
        slug = "public-1"
        assert (dest / "transcripts" / slug / "env_a" / "seed_3.jsonl.zst").is_file()
        assert (dest / "tool_calls" / slug / "env_a" / "seed_3.jsonl.zst").is_file()
        assert (dest / "grade_calls" / slug / "env_a" / "seed_3.jsonl.zst").is_file()
        # Roundtrip read
        import pyarrow.parquet as pq
        table = pq.read_table(stats.parquet_path)
        assert table.num_rows == 1
        cols = table.column_names
        assert "model" in cols
        assert "caps_cov_func" in cols
        assert "caps_ace" in cols
        assert "transcript_path" in cols
        rec = table.to_pylist()[0]
        assert rec["model"] == "anthropic/public-1"
        assert rec["caps_cov_func"] is True
        assert rec["caps_ace"] is True
        assert rec["transcript_path"] == "transcripts/public-1/env_a/seed_3.jsonl.zst"

    def test_missing_sidecar_yields_null_path(self, tmp_path: Path) -> None:
        # Run dir exists but has no transcript.jsonl
        rd = tmp_path / "thin"
        rd.mkdir()
        (rd / "tool_calls.jsonl").write_text("")
        # No transcript.jsonl, no grade_calls.jsonl
        cell = selection.CellRecord(
            run_id="thin", benchmark_id="v8",
            model="m/x", env_id="env_a", seed=1,
            status="model_failed", score=0.0,
            capabilities={},
            run_dir=rd,
            started_at="2026-05-01T00:00:00Z",
            finished_at=None, runtime_s=None, turns_used=None,
            exit_reason="error",
            image_ref=None, image_digest=None,
            git_sha=None, weighted_tokens_used=None, peak_per_turn_context=None,
            cost_usd=None, cost_source=None,
            tokens_in=None, tokens_out=None,
            tokens_cache_read=None, tokens_cache_creation=None, tokens_reasoning=None,
            served_model=None,
        )
        dest = tmp_path / "dist" / "y"
        bundle.build([cell], dest)
        import pyarrow.parquet as pq
        rec = pq.read_table(dest / "runs.parquet").to_pylist()[0]
        assert rec["transcript_path"] is None
        assert rec["grade_calls_path"] is None
        # tool_calls.jsonl was empty but exists — it gets compressed
        assert rec["tool_calls_path"] == "tool_calls/x/env_a/seed_1.jsonl.zst"
