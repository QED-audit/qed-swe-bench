"""Per-episode run-dir artifact writers.

Each tuple's `runs/<benchmark_id>/<run_id>/` gets a small set of
JSON / JSONL artifacts: `job.json` (provenance), `score.json` (final
result), `cost.json` (token + spend), and an appended row in
`grade_calls.jsonl` for cli_oneshot graders. Pure file I/O — no DB,
no docker, no async — carved out of `orchestrator.py` so the
orchestrator file can focus on per-tuple state-machine work.

The transcript / tool_calls / grade_calls (in-MCP path) JSONL streams
are owned by `runner/transcript.py:TranscriptWriter`, not this module.

Each writer defensively `mkdir(parents=True, exist_ok=True)`s its
target directory before writing. The orchestrator already mkdir's
the run_dir before any writers run, but we have observed cases where
the directory is missing when the writer runs (an `uncaught_FileNotFoundError`
escaped the outer wrap as `score.json` couldn't be written). The
existing `TranscriptWriter` and `McpDockerSession.start` already do
the same defensive mkdir — this module aligns with that pattern.
"""

from __future__ import annotations

import json
import os
import socket
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qed_swe_bench.runner.image_ref import ResolvedImage
from qed_swe_bench.runner.llm.base import NormalizedUsage
from qed_swe_bench.runner.orchestrator_config import Budgets, EnvSpec


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


# On-disk dir name format. Lexicographic sort = chronological sort, so
# `ls runs/v8/` lands the most-recent runs at the bottom (or top with
# `-r`). The 16-hex run_id stays the canonical identifier in the DB,
# API URLs (`/api/runs/{run_id}`), audit `--run-id`, and
# `safety_identifier`; only the directory name carries the timestamp
# prefix. A `__` separator is unambiguous since hex run_ids can't
# contain underscores.
_RUN_DIR_TS_FMT = "%Y-%m-%dT%H-%M-%SZ"  # filesystem-safe ISO-shape


def construct_run_dir_name(run_id: str, *, ts: datetime | None = None) -> str:
    """Build the on-disk dir name for a fresh run as `<iso>__<run_id>`.

    LEGACY layout, kept for backward compat reading. New runs use the
    layered path from `construct_run_dir_path()` per D-10.

    `ts` defaults to now-UTC; tests pass a fixed value for determinism.
    """
    when = ts or datetime.now(UTC)
    return f"{when.strftime(_RUN_DIR_TS_FMT)}__{run_id}"


def host() -> str:
    """Return the host identity for run-dir partitioning.

    `QED_SWE_BENCH_HOST` env override (preferred — lets EC2 instances
    set their host via instance metadata or systemd EnvironmentFile)
    or `socket.gethostname()` fallback. Sanitized to filesystem-safe
    chars: lowercased, non-alphanumeric replaced with `-`.
    """
    raw = os.environ.get("QED_SWE_BENCH_HOST") or socket.gethostname() or "unknown"
    sanitized = "".join(c if c.isalnum() or c in "-_." else "-" for c in raw.lower())
    return sanitized or "unknown"


def construct_run_dir_path(
    runs_root: Path,
    *,
    benchmark_id: str,
    run_id: str,
    ts: datetime | None = None,
    host_override: str | None = None,
) -> Path:
    """Build the per-run absolute path under the new layered layout
    (D-10): `runs/<benchmark_id>/<host>/<datetime>/<run_id>/`.

    - `benchmark_id` at top: aggregation by benchmark (the dominant
      access pattern; matches `make audit BENCHMARK_ID=v8` ergonomics).
    - `host` second: rsync-from-multiple-hosts isolation.
    - `<datetime>` third: chronological / archival.
    - `run_id` at the leaf: canonical 16-hex DB key, no timestamp prefix.

    `host_override` is for the migrate-runs script (which sets it to
    `legacy-pre-2026-05` or similar for runs whose host is unknown).
    Tests pass `ts` for determinism.
    """
    when = ts or datetime.now(UTC)
    h = host_override or host()
    return (
        runs_root
        / benchmark_id
        / h
        / when.strftime(_RUN_DIR_TS_FMT)
        / run_id
    )


def parse_run_id_from_dir_name(name: str) -> str:
    """Extract the run_id from a run-dir name.

    New shape (`<iso>__<run_id>`): split on `__` and take the trailing
    segment. Pre-2026-05-01 dirs are bare hex (`<run_id>`); we return
    the whole name in that case, which is the run_id verbatim. Imported
    or legacy paths can also be parsed safely — anything without `__`
    is treated as a bare run_id.
    """
    if "__" in name:
        return name.rsplit("__", 1)[-1]
    return name


def write_job_json(
    *,
    run_dir: Path,
    run_id: str,
    benchmark_id: str,
    model: str,
    env: EnvSpec,
    seed: int,
    resolved: ResolvedImage,
    budgets: Budgets,
    nudges_used: bool,
    git_sha: str | None = None,
    env_overrides: dict[str, str] | None = None,
    repro_cmd: str | None = None,
    provenance: str = "native",
    agent: str = "qed_swe_bench",
) -> None:
    """Provenance record. One per run-dir, written before docker spawn.

    `git_sha`, `env_overrides`, and `repro_cmd` are best-effort
    reproducibility metadata captured by the orchestrator. They're all
    optional so test fixtures and programmatic callers can omit them;
    real runs populate all three via `runner/provenance.py`.

    `provenance` distinguishes the source of the row. Values produced
    by writers in this repo: `native` (default — runner-produced),
    `mock` (mock-LLM smoke runs), and an open set of `imported_from_*`
    tags written by the import paths in `historical.py` and the
    one-off scripts under `scripts/`. Downstream gates use
    `provenance.startswith("imported_from_")` rather than a closed
    membership check so new importers can land without coordinating.
    Defaults to `'native'` so old callers and pre-existing job.json
    files (which omit the field) continue to import as native.

    `agent` is the identity of the agent harness that produced this
    run: `qed_swe_bench` for the native runner, `codex` for codex CLI
    sweeps, `vr-agent` for historical Anthropic imports. Distinct from
    `model` (which LLM) and `provenance` (data lineage); leaderboards
    group by `(model, agent)` so the same model under different
    harnesses doesn't average into one row.

    `nudges_used` records whether mid-episode scaffolding nudges
    (`build_stuck_nudge` / `build_wrapup_nudge` /
    `build_voluntary_exit_nudge` in `runner/loop.py`) were enabled for
    this run. Required (no NULL state) so the corresponding DB column
    can be `NOT NULL`; the orchestrator computes it as
    `bool(BenchmarkConfig.nudges)`.
    """
    obj: dict[str, Any] = {
        "run_id": run_id,
        "benchmark_id": benchmark_id,
        "model": model,
        "env_id": env.id,
        "image_ref": env.image,
        "image_digest": resolved.image_digest,
        "image_pulled": resolved.pulled,
        "task_type": env.task_type,
        "interface": env.interface,
        "seed": seed,
        "budgets": asdict(budgets),
        "nudges_used": nudges_used,
        "started_at": _now_iso(),
        "provenance": provenance,
        "agent": agent,
    }
    if git_sha is not None:
        obj["git_sha"] = git_sha
    if env_overrides:
        obj["env_overrides"] = env_overrides
    if repro_cmd is not None:
        obj["repro_cmd"] = repro_cmd
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "job.json").write_text(
        json.dumps(obj, indent=2) + "\n", encoding="utf-8",
    )


def write_score_json(
    run_dir: Path,
    *,
    capabilities: dict[str, bool],
    score: float,
    exit_reason: str,
    finished_at: str | None = None,
    runtime_s: float | None = None,
    turns_used: int | None = None,
    failure_reason: str | None = None,
    status: str | None = None,
    weighted_tokens_used: int | None = None,
    peak_per_turn_context: int | None = None,
) -> None:
    """Final capability bitmap + score + exit reason + timing + status.

    The extra fields beyond capabilities/score/exit_reason exist to
    close the FS↔DB bijection (D-10): every DB column that's not
    purely runtime / token telemetry needs a flat-text source so
    `qed_swe_bench import` can hydrate the row without inference.

    `finished_at` is ISO-8601 UTC string; `runtime_s` is wall-clock
    duration. `status` is the lifecycle terminal value (succeeded /
    model_failed / infra_failed) — same string the DB row carries.

    `weighted_tokens_used` and `peak_per_turn_context` are
    always-reported diagnostics (turn-as-effort methodology, see
    docs/decisions.md): the runner tracks them whether or not the
    corresponding budget is enforced, so the same cell looks
    identical across "budget on" and "budget off" runs except for
    the (optional) early termination. On `exit_reason ==
    context_window_exceeded` the provider's verbatim error message
    (which carries the live byte limit) lands in `failure_reason`.

    All optional so test fixtures and partial-write callers don't
    have to populate every field.
    """
    obj: dict[str, Any] = {
        "capabilities": capabilities,
        "score": score,
        "exit_reason": exit_reason,
    }
    if finished_at is not None:
        obj["finished_at"] = finished_at
    if runtime_s is not None:
        obj["runtime_s"] = runtime_s
    if turns_used is not None:
        obj["turns_used"] = turns_used
    if failure_reason is not None:
        obj["failure_reason"] = failure_reason
    if status is not None:
        obj["status"] = status
    if weighted_tokens_used is not None:
        obj["weighted_tokens_used"] = weighted_tokens_used
    if peak_per_turn_context is not None:
        obj["peak_per_turn_context"] = peak_per_turn_context
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "score.json").write_text(
        json.dumps(obj, indent=2) + "\n", encoding="utf-8",
    )


def write_cost_json(
    run_dir: Path,
    *,
    model: str,
    usage: NormalizedUsage,
    cost_usd: float | None,
    cost_source: str,
    weighted_tokens: int,
    served_model: str | None = None,
    llm_route: str | None = None,
    api_base: str | None = None,
    or_reconciliation: dict | None = None,
) -> None:
    """Token + spend summary. Mirrors the columns persisted on the runs row.

    `model` is the id we *requested*; `served_model` is what the provider
    echoed back. They diverge when the provider routes to a different
    snapshot (e.g. silent downgrade, alias resolution). Both are recorded
    so post-hoc audits can detect drift without needing the raw transcript.

    `llm_route` and `api_base` close the FS↔DB bijection (D-10) for
    the routing-related columns: `llm_route` is one of
    `anthropic_native | litellm | litellm_gateway | mock`; `api_base`
    is the gateway URL when set (rare; only OpenAI-compatible paths).

    `or_reconciliation` (when not None) is the dict produced by
    `runner/openrouter_recon.py:ReconciliationSummary.to_dict`. It
    sits alongside the inferred `cost_usd` so audits can compare
    inferred-vs-authoritative cost on OR-routed cells without losing
    either signal.
    """
    obj: dict[str, Any] = {
        "model": model,
        "served_model": served_model,
        "tokens_in": usage.input_tokens,
        "tokens_out": usage.output_tokens,
        "tokens_cache_read": usage.cache_read_tokens,
        "tokens_cache_creation": usage.cache_creation_tokens,
        "tokens_reasoning": usage.reasoning_tokens,
        "weighted_tokens_used": weighted_tokens,
        "cost_usd": cost_usd,
        "cost_source": cost_source,
    }
    if llm_route is not None:
        obj["llm_route"] = llm_route
    if api_base is not None:
        obj["api_base"] = api_base
    if or_reconciliation is not None:
        obj["or_reconciliation"] = or_reconciliation
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "cost.json").write_text(
        json.dumps(obj, indent=2) + "\n", encoding="utf-8",
    )


def append_cli_oneshot_grade_log(
    run_dir: Path, grade_raw: dict[str, Any],
) -> None:
    """Append one entry to grade_calls.jsonl shaped like the in-MCP `grade`
    tool's output, so the capability extractor and any future replay
    logic see a uniform stream regardless of grading path.
    """
    line = json.dumps({"source": "cli_oneshot", "result": grade_raw}) + "\n"
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "grade_calls.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(line)
