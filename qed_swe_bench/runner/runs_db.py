"""CRUD for the `runs` table.

All writes to the per-episode lifecycle row live here. The orchestrator
calls these from four phases of a tuple's life:

    queued (insert_queued)             — inserted but not yet started
        ↓
    running (mark_running)             — docker spawned + LLM in flight
        ↓
    succeeded / model_failed / infra_failed (update_finished)
        OR
    infra_failed via the in-body shortcut (record_infra_failure)
        OR
    infra_failed via the outer safety wrap (record_failure_if_row_exists)

The `queued` → `running` split (added 2026-05) disambiguates two
states that previously shared the `queued` label: rows truly waiting
for a `max_parallel` semaphore slot vs rows actively burning compute.
External tooling (CLI summary, webui, audits) can now tell them
apart at a glance instead of having to derive it from `last_heartbeat`
freshness.

Pure DB I/O — no docker, no async, no LLM. Carved out of `orchestrator.py`
so the orchestrator can focus on lifecycle orchestration. The companion
file `run_dir.py` owns the on-disk artifacts (job.json / score.json / …);
together they're the two halves of a single record (DB row + run-dir).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from qed_swe_bench.db.schema import transaction
from qed_swe_bench.runner.llm.base import NormalizedUsage
from qed_swe_bench.runner.orchestrator_config import EnvSpec

log = logging.getLogger(__name__)


def now_iso() -> str:
    """Single source of truth for `started_at` / `finished_at` formatting."""
    return datetime.now(UTC).isoformat()


def insert_queued(
    *,
    run_id: str,
    benchmark_id: str,
    model: str,
    env: EnvSpec,
    image_digest: str,
    seed: int,
    run_dir: Path,
    nudges_used: bool,
    git_sha: str | None = None,
    repro_cmd: str | None = None,
    config_snapshot_yaml: str | None = None,
) -> bool:
    """Insert a queued row. Returns True if inserted, False if duplicate (resume).

    Duplicate detection is `UNIQUE(benchmark_id, model, env_id, seed)` —
    the same logical tuple can only appear once per benchmark_id, so a
    repeat invocation against the same config short-circuits without
    spawning docker.

    `nudges_used` records whether mid-episode scaffolding nudges were
    enabled for this run. Required (no NULL state) so the DB column
    can be `NOT NULL`; the orchestrator computes it as
    `bool(BenchmarkConfig.nudges)`.

    `git_sha` and `repro_cmd` are reproducibility provenance — see
    `runner/provenance.py`. `config_snapshot_yaml` is the verbatim
    source YAML the row was produced under — see D-13 in
    `docs/decisions.md`; lets `qed_swe_bench rerun <run_id>` replay
    without the source file on disk. All three nullable so test
    fixtures and programmatic callers (no file backing) can omit them.
    """
    with transaction() as con:
        try:
            con.execute(
                """
                INSERT INTO runs (
                    run_id, benchmark_id, model, env_id, image_ref, image_digest,
                    task_type, interface, seed, status, run_dir, started_at, provenance,
                    git_sha, repro_cmd, config_snapshot, nudges_used, agent
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?, 'native', ?, ?, ?, ?, 'qed_swe_bench')
                """,
                (
                    run_id,
                    benchmark_id,
                    model,
                    env.id,
                    env.image,
                    image_digest,
                    env.task_type,
                    env.interface,
                    seed,
                    str(run_dir),
                    now_iso(),
                    git_sha,
                    repro_cmd,
                    config_snapshot_yaml,
                    1 if nudges_used else 0,
                ),
            )
            return True
        except sqlite3.IntegrityError:
            # UNIQUE(benchmark_id, model, env_id, seed) collision.
            return False


def mark_running(run_id: str) -> bool:
    """Transition a `queued` row to `running` and seed `last_heartbeat`.

    Called by the orchestrator right before `McpDockerSession.start`,
    once the row has cleared the cost-cap short-circuit and committed
    to actually executing the episode. Returns True if the row moved
    to `running`, False if it wasn't in `queued` (e.g. already
    promoted, or marked infra_failed by a concurrent sweep — both
    safe no-ops).

    Initializing `last_heartbeat=now` here closes a brief window where
    a `running` row would otherwise look like "no heartbeat" to a
    concurrent stale-sweep until the HeartbeatTracker's first periodic
    tick fires.
    """
    with transaction() as con:
        cur = con.execute(
            "UPDATE runs SET status='running', last_heartbeat=? "
            "WHERE run_id=? AND status='queued'",
            (now_iso(), run_id),
        )
        return cur.rowcount > 0


def claim_for_resume(run_id: str) -> bool:
    """Atomically claim a failed row for an in-progress resume.

    Flips `infra_failed`/`model_failed` → `running` and seeds
    `last_heartbeat=now`, mirroring `mark_running`'s contract for the
    fresh-run path: while a resume is in flight the row presents the
    same `running` + heartbeated lifecycle a fresh-run row does. Returns
    True if claimed; False if the row was already running (race lost
    to another resume invocation) or no longer in a resumable state.

    The WHERE clause keeps the flip atomic so two `--resume-failed`
    sweeps racing on the same row both see one success and one False;
    the losing invocation must skip the row.
    """
    with transaction() as con:
        cur = con.execute(
            "UPDATE runs SET status='running', last_heartbeat=?, finished_at=NULL "
            "WHERE run_id=? AND status IN ('infra_failed', 'model_failed')",
            (now_iso(), run_id),
        )
        return cur.rowcount > 0


def update_finished(
    *,
    run_id: str,
    status: str,
    capabilities: dict[str, bool] | None,
    score: float | None,
    usage_totals: NormalizedUsage,
    cost_usd: float | None,
    cost_source: str,
    runtime_s: float,
    turns_used: int,
    exit_reason: str,
    llm_route: str,
    api_base: str | None,
    failure_reason: str | None,
    provenance: str = "native",
    weighted_tokens_used: int | None = None,
    peak_per_turn_context: int | None = None,
) -> None:
    """Move a queued row to its terminal state with all measured fields.

    `weighted_tokens_used` and `peak_per_turn_context` are always-on
    diagnostics tracked by `Budget` regardless of whether the
    corresponding budget is enforced. See docs/decisions.md
    (turn-as-effort).
    """
    with transaction() as con:
        con.execute(
            """
            UPDATE runs SET
                status=?, capabilities=?, score=?,
                tokens_in=?, tokens_out=?, tokens_cache_read=?, tokens_cache_creation=?,
                cost_usd=?, cost_source=?, runtime_s=?, turns_used=?,
                exit_reason=?, finished_at=?,
                llm_route=?, api_base=?, failure_reason=?, provenance=?,
                weighted_tokens_used=?, peak_per_turn_context=?
            WHERE run_id=?
            """,
            (
                status,
                json.dumps(capabilities) if capabilities is not None else None,
                score,
                usage_totals.input_tokens,
                usage_totals.output_tokens,
                usage_totals.cache_read_tokens,
                usage_totals.cache_creation_tokens,
                cost_usd,
                cost_source,
                runtime_s,
                turns_used,
                exit_reason,
                now_iso(),
                llm_route,
                api_base,
                failure_reason,
                provenance,
                weighted_tokens_used,
                peak_per_turn_context,
                run_id,
            ),
        )


def record_infra_failure(
    run_id: str,
    *,
    exit_reason: str,
    failure_reason: str | None,
    runtime_s: float = 0.0,
    provenance: str = "native",
) -> None:
    """Record a failure-with-no-result row.

    Used by the in-body failure paths (image-resolve fail, cost-cap
    short-circuit, mid-episode crash) — places where we know a queued
    row exists and want to upgrade it to `infra_failed` with empty
    usage / cost fields. Not used by the outer safety wrap (see
    `record_failure_if_row_exists` for that — same shape but raw SQL
    so it can no-op when no row exists).
    """
    update_finished(
        run_id=run_id,
        status="infra_failed",
        capabilities=None,
        score=None,
        usage_totals=NormalizedUsage(),
        cost_usd=None,
        cost_source="unknown",
        runtime_s=runtime_s,
        turns_used=0,
        exit_reason=exit_reason,
        llm_route="unknown",
        api_base=None,
        failure_reason=failure_reason,
        provenance=provenance,
    )


def record_failure_if_row_exists(
    run_id: str, *, exit_reason: str, failure_reason: str
) -> None:
    """If a queued OR running row for `run_id` exists, mark it infra_failed.

    Called from the outer safety wrap when a per-tuple exception
    escapes (timeout, uncaught exception). The row may be in either
    `queued` (timeout fired before docker spawn) or `running` (timeout
    fired mid-episode); both are valid in-flight states and either
    should upgrade to `infra_failed`. If the row was never inserted
    (e.g., insert_queued itself raised), there's nothing to update
    and we silently move on; the next sweep will retry the tuple
    cleanly.
    """
    try:
        with transaction() as con:
            con.execute(
                """
                UPDATE runs SET status='infra_failed',
                                exit_reason=COALESCE(exit_reason, ?),
                                failure_reason=COALESCE(failure_reason, ?),
                                finished_at=?
                 WHERE run_id=? AND status IN ('queued', 'running')
                """,
                (exit_reason, failure_reason, now_iso(), run_id),
            )
    except sqlite3.DatabaseError:
        # Don't let a recovery failure mask the original error.
        log.exception("[%s] recovery UPDATE failed", run_id)
