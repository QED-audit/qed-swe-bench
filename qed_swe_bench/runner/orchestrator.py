"""Drive a benchmark run: enumerate (model, env, seed) tuples, run them with
controlled parallelism, persist results.

This module is the lifecycle + driver only. Supporting concerns live in
single-purpose siblings:

  runner/runs_db.py        — `runs` table CRUD (insert / update / record_failure)
  runner/exceptions.py     — bug vs infra classification of caught exceptions
  runner/prompts.py        — default system / init prompts
  runner/env_manifest.py   — catalog → manifest lookup for the cli_oneshot path
  runner/orchestrator_config.py — BenchmarkConfig / EnvSpec / ModelSpec / parse_config
  runner/run_dir.py        — on-disk artifacts (job.json / score.json / …)
  runner/resilience.py     — retry-failed, stale-queued recovery
  runner/spend_tracker.py  — running cost cap

For each tuple:
  1. INSERT runs(status='queued') — UNIQUE constraint dedupes on --resume.
  2. Resolve the env's image_ref to a digest.
  3. Open a docker MCP session, build an LLM client, run_episode().
  4. Write run-dir files (job.json, transcript/tool/grade jsonl, score.json,
     cost.json) and UPDATE runs(status='succeeded'|...).

A single asyncio.Semaphore caps concurrency so we don't spawn N V8 containers
at once on a small box.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic

from qed_swe_bench.config import Config
from qed_swe_bench.db.schema import init_db
from qed_swe_bench.runner.budget import Budget
from qed_swe_bench.runner.capabilities import (
    DEFAULT_SCORING_POLICY,
    compute_score,
)
from qed_swe_bench.runner.cli_oneshot_grader import GraderError, run_cli_oneshot_grader
from qed_swe_bench.runner.cost import compute_total_cost
from qed_swe_bench.runner.env_manifest import EnvManifestLoadError, load_env_manifest
from qed_swe_bench.runner.exceptions import (
    format_failure,
    is_infra_exception,
    unwrap_exception_group,
)
from qed_swe_bench.runner.image_ref import ImageRefError, resolve
from qed_swe_bench.runner.llm.base import LLMClient, NormalizedUsage
from qed_swe_bench.runner.llm.factory import build_client
from qed_swe_bench.runner.loop import EpisodeResult, run_episode
from qed_swe_bench.runner.mcp_client import McpDockerSession
from qed_swe_bench.runner.prompts import default_init_prompt

# orchestrator_config.py owns the dataclasses + parser; re-exported so
# external callers (cli.py, tests) can keep importing them from here.
from qed_swe_bench.runner.orchestrator_config import (
    DEFAULT_EPISODE_TIMEOUT_S,
    BenchmarkConfig,
    Budgets,
    EnvSpec,
    ModelSpec,
    parse_config,
)
from qed_swe_bench.runner.resilience import (
    HeartbeatTracker,
    delete_failed_rows,
    mark_stale_queued_as_failed,
)
from qed_swe_bench.runner.run_dir import (
    append_cli_oneshot_grade_log,
    construct_run_dir_path,
    write_cost_json,
    write_job_json,
    write_score_json,
)
from qed_swe_bench.runner.runs_db import (
    insert_queued,
    mark_running,
    record_failure_if_row_exists,
    record_infra_failure,
    update_finished,
)
from qed_swe_bench.runner.spend_tracker import SpendTracker
from qed_swe_bench.runner.transcript import TranscriptWriter

# Back-compat aliases. Tests and historical code import the underscore-
# prefixed names from this module. Re-export from the new modules so a
# single refactor pass can land without churning every test.
_insert_queued = insert_queued
_mark_running = mark_running
_update_finished = update_finished
_record_infra_failure = record_infra_failure
_record_failure_if_row_exists = record_failure_if_row_exists
_unwrap_exception_group = unwrap_exception_group
_format_failure = format_failure
_is_infra_exception = is_infra_exception
_load_env_manifest = load_env_manifest

__all__ = [
    "BenchmarkConfig",
    "Budgets",
    "DEFAULT_EPISODE_TIMEOUT_S",
    "EnvSpec",
    "ModelSpec",
    "RunOutcome",
    "parse_config",
    "run_benchmark",
]

log = logging.getLogger(__name__)


def _read_prompt_file(path: Path, field_name: str) -> str:
    """Resolve a YAML-supplied prompt file path and read it.

    Relative paths are resolved against CWD (the repo root when the CLI is
    invoked from a checkout), so YAMLs can carry stable paths like
    `benchmarks/prompts/init-v2-hint.template`.
    """
    resolved = path if path.is_absolute() else (Path.cwd() / path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"{field_name}={path} does not exist (resolved to {resolved})."
        )
    return resolved.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Per-tuple lifecycle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunOutcome:
    """What `_run_one` returns to the orchestrator driver.

    Carrying the cost out of the per-tuple call lets the SpendTracker
    update without a follow-up DB roundtrip, and the run_id keeps the
    safety-wrap path able to mark a row infra_failed if something
    escapes.
    """

    status: str
    cost_usd: float | None
    run_id: str


async def _run_one(
    *,
    cfg: Config,
    bench: BenchmarkConfig,
    model: str,
    env: EnvSpec,
    seed: int,
    mock_llm: bool,
    init_prompt: str,
    run_root: Path,
    spend_tracker: SpendTracker,
    config_path: Path | str | None = None,
    git_sha: str | None = None,
    env_overrides: dict[str, str] | None = None,
    heartbeat: HeartbeatTracker | None = None,
) -> RunOutcome:
    """Execute one (model, env, seed) tuple. Returns RunOutcome.

    Resilience layered on top of the per-tuple body:
      - Outer try/except converts any unexpected escape into a recorded
        infra_failed (via `_record_failure_if_row_exists`).
      - `asyncio.wait_for(bench.episode_timeout_s)` bounds the docker /
        MCP / episode block so a wedged container can't permanently
        occupy a concurrency slot.
      - Cost cap check after `_insert_queued`: if a previously-
        completed tuple already crossed the cap, this tuple short-
        circuits to infra_failed without paying the docker pull /
        MCP startup cost.
    """
    run_id = uuid.uuid4().hex[:16]
    try:
        try:
            return await _run_one_body(
                run_id=run_id,
                cfg=cfg,
                bench=bench,
                model=model,
                env=env,
                seed=seed,
                mock_llm=mock_llm,
                init_prompt=init_prompt,
                run_root=run_root,
                spend_tracker=spend_tracker,
                config_path=config_path,
                git_sha=git_sha,
                env_overrides=env_overrides,
                heartbeat=heartbeat,
            )
        finally:
            # Drop the run_id from the heartbeat tracker once the episode
            # body completes (succeeded / failed / unhandled-exception).
            # Idempotent: if the row never made it past _insert_queued the
            # set just doesn't contain it, and discard is a no-op.
            if heartbeat is not None:
                try:
                    await heartbeat.remove(run_id)
                except Exception:  # noqa: BLE001
                    log.exception("[%s] heartbeat.remove failed", run_id)
    except TimeoutError:
        log.error(
            "[%s] episode timed out after %ds (model=%s env=%s seed=%d)",
            run_id, bench.episode_timeout_s, model, env.id, seed,
        )
        _record_failure_if_row_exists(
            run_id,
            exit_reason=f"episode_timeout_{bench.episode_timeout_s}s",
            failure_reason="exceeded BenchmarkConfig.episode_timeout_s",
        )
        return RunOutcome("infra_failed", None, run_id)
    except Exception as exc:
        inner = unwrap_exception_group(exc)
        if is_infra_exception(inner):
            log.exception("[%s] infra exception escaping _run_one_body", run_id)
            kind = "infra"
        else:
            # A code bug — log louder and tag the row distinctly so the CLI
            # surfaces it under a separate "code bugs" header. The sweep
            # still continues so sibling tuples aren't penalised.
            log.exception(
                "[%s] CODE BUG escaping _run_one_body: %s",
                run_id, type(inner).__name__,
            )
            kind = "bug"
        _record_failure_if_row_exists(
            run_id,
            exit_reason=f"{kind}_{type(inner).__name__}",
            failure_reason=f"{type(inner).__name__}: {inner}",
        )
        return RunOutcome("infra_failed", None, run_id)


async def _run_one_body(
    *,
    run_id: str,
    cfg: Config,
    bench: BenchmarkConfig,
    model: str,
    env: EnvSpec,
    seed: int,
    mock_llm: bool,
    init_prompt: str,
    run_root: Path,
    spend_tracker: SpendTracker,
    config_path: Path | str | None = None,
    git_sha: str | None = None,
    env_overrides: dict[str, str] | None = None,
    heartbeat: HeartbeatTracker | None = None,
) -> RunOutcome:
    """Body of `_run_one`. Outer wrapper handles timeout + uncaught."""
    # The run_dir path is computed up-front but NOT mkdir'd here —
    # `_insert_queued` may return False on a duplicate (model, env, seed)
    # tuple, in which case we bail without writing anything and creating
    # an empty dir would leak one shell of garbage per re-invocation.
    # Every writer (run_dir.py, TranscriptWriter, McpDockerSession.start)
    # now defensively mkdirs its target, so this is safe.
    # Dir name is `<utc-iso>__<run_id>` so `ls runs/<benchmark_id>/`
    # sorts chronologically. The DB run_id column stays the bare
    # 16-hex value; only the on-disk path picks up the timestamp.
    run_dir = construct_run_dir_path(
        run_root,
        benchmark_id=bench.benchmark_id,
        run_id=run_id,
    )

    model_spec = next(
        (item for item in bench.models if item.id == model), ModelSpec(id=model)
    )

    # Synthesize the per-tuple repro command: a literal
    # `qed_swe_bench rerun <run_id>`. The row's `config_snapshot`
    # column carries the source YAML so rerun is fully self-
    # contained (D-13 in docs/decisions.md).
    from qed_swe_bench.runner.provenance import synthesize_repro_cmd
    repro_cmd = synthesize_repro_cmd(run_id=run_id)

    # Read the source YAML once up-front. Passed verbatim to
    # `_insert_queued` (DB column) and reused for the on-disk
    # `<run_dir>/config_snapshot.yaml` write below — same bytes both
    # places per the bijection contract. None when there's no file
    # backing (programmatic invocation, --test, --mock-llm).
    config_snapshot_yaml: str | None = None
    if config_path is not None:
        try:
            config_snapshot_yaml = Path(config_path).read_text(encoding="utf-8")
        except OSError:
            # Defensive: don't block the run on a transient FS hiccup;
            # the column stays NULL and reproduce can fall back to the
            # on-disk snapshot if it's eventually written.
            config_snapshot_yaml = None

    # 1. Resolve image ref → digest. If this fails it's an infra problem.
    try:
        resolved = resolve(env.image)
    except ImageRefError as exc:
        # Insert + immediate failure record.
        _insert_queued(
            run_id=run_id,
            benchmark_id=bench.benchmark_id,
            model=model,
            env=env,
            image_digest="<unresolved>",
            seed=seed,
            run_dir=run_dir,
            nudges_used=bool(bench.nudges),
            git_sha=git_sha,
            repro_cmd=repro_cmd,
            config_snapshot_yaml=config_snapshot_yaml,
        )
        _record_infra_failure(
            run_id,
            exit_reason="image_unresolved",
            failure_reason=str(exc),
        )
        log.error("[%s] image resolve failed: %s", run_id, exc)
        return RunOutcome("infra_failed", None, run_id)

    inserted = _insert_queued(
        run_id=run_id,
        benchmark_id=bench.benchmark_id,
        model=model,
        env=env,
        image_digest=resolved.image_digest,
        seed=seed,
        run_dir=run_dir,
        nudges_used=bool(bench.nudges),
        git_sha=git_sha,
        repro_cmd=repro_cmd,
        config_snapshot_yaml=config_snapshot_yaml,
    )
    if not inserted:
        log.info("[%s] skipped (already in DB; --resume)", run_id)
        return RunOutcome("resumed_skip", None, run_id)

    # Cost-cap short-circuit: if a previously-completed tuple already
    # crossed the cap, refuse to start the docker / MCP work for this
    # one. Recording an infra_failed row here keeps `--retry-failed`
    # able to resume the sweep later (e.g., after the user raises the
    # cap or kills costly tuples).
    if await spend_tracker.cap_exceeded():
        total = await spend_tracker.total()
        log.warning(
            "[%s] cost cap reached ($%.2f >= $%.2f); short-circuiting tuple",
            run_id, total, spend_tracker.cap_usd,
        )
        _record_infra_failure(
            run_id,
            exit_reason="cost_cap_exceeded",
            failure_reason=f"running total ${total:.2f} hit cap ${spend_tracker.cap_usd:.2f}",
            provenance="mock" if mock_llm else "native",
        )
        return RunOutcome("infra_failed", None, run_id)

    # Register with the heartbeat tracker now that the row is queued
    # past the early-fail filters. Background task will periodically
    # refresh `last_heartbeat` so a sibling process's startup sweep
    # sees recent liveness and skips this row. Removed in `_run_one`'s
    # finally block once the episode finishes (or this orchestrator
    # dies, in which case the heartbeat just stops refreshing and the
    # row eventually reaps).
    if heartbeat is not None:
        await heartbeat.add(run_id)

    write_job_json(
        run_dir=run_dir,
        run_id=run_id,
        benchmark_id=bench.benchmark_id,
        model=model,
        env=env,
        seed=seed,
        resolved=resolved,
        budgets=bench.budgets,
        nudges_used=bool(bench.nudges),
        git_sha=git_sha,
        env_overrides=env_overrides,
        repro_cmd=repro_cmd,
        provenance="mock" if mock_llm else "native",
        reasoning_replay=model_spec.reasoning_replay,
    )

    # Snapshot the benchmark config alongside job.json so a later edit
    # to v8.yaml doesn't erase history of what was actually run. We
    # write the same bytes we already stuffed into runs.config_snapshot
    # above, so the FS↔DB bijection on this artifact is guaranteed
    # byte-identical (D-13 in docs/decisions.md). No-op when there's
    # no source YAML (programmatic invocation, --test, --mock-llm).
    if config_snapshot_yaml is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config_snapshot.yaml").write_text(
            config_snapshot_yaml, encoding="utf-8",
        )

    # Transition queued -> running. From here on, an external observer
    # (CLI summary, webui) sees this row distinctly from rows still
    # waiting for a max_parallel slot. mark_running also seeds
    # last_heartbeat so the stale-sweep doesn't reap the row before
    # the HeartbeatTracker's first periodic tick fires.
    _mark_running(run_id)

    # 2. Build LLM client.
    # Look up per-model params (e.g. `reasoning_effort: xhigh` for gpt-5*)
    # from the BenchmarkConfig. Default to no overrides if a programmatic
    # caller built bench without ModelSpec.params populated.
    model_params = model_spec.params
    client: LLMClient = build_client(
        model, mock=mock_llm, config=cfg, params=model_params,
    )
    api_base = getattr(client, "api_base", None)

    # 3. Open MCP, run the episode. Bounded by `episode_timeout_s` so a
    # wedged container can't hold a concurrency slot indefinitely; a
    # TimeoutError here propagates to the outer wrap which records the
    # row as infra_failed and returns. Mid-episode failures (MCP crash,
    # provider error, etc.) are handled inline.
    started = monotonic()
    result: EpisodeResult | None = None

    # Load the env manifest once up-front so we can both filter the
    # agent's tool surface (evaluation_tools must NOT reach the model —
    # they're the post-episode grading shape) and dispatch the
    # cli_oneshot grader after the episode. None for legacy V8 envs that
    # predate the manifest contract; they grade via the in-MCP `grade`
    # tool instead and have no eval-tool segregation problem.
    # An EnvManifestLoadError means the env IS registered with a manifest
    # but the file is broken — fail closed (record infra_failed) rather
    # than silently downgrade to in-MCP grading, which would change the
    # benchmark's grading contract from what the catalog promised.
    try:
        manifest = load_env_manifest(env.id)
    except EnvManifestLoadError as exc:
        log.error("[%s] %s", run_id, exc)
        _record_infra_failure(
            run_id,
            exit_reason="manifest_load_failed",
            failure_reason=str(exc),
            provenance="mock" if mock_llm else "native",
        )
        return RunOutcome("infra_failed", None, run_id)
    excluded_tools = (
        frozenset(manifest.evaluation_tools) if manifest is not None else frozenset()
    )

    try:
        async with McpDockerSession.start(
            resolved.image_digest,
            stderr_path=run_dir / "mcp_stderr.log",
        ) as session:
            with TranscriptWriter(run_dir) as tw:
                budget = Budget(
                    turn_budget=bench.budgets.turn_budget,
                    token_budget=bench.budgets.token_budget,
                    context_budget=bench.budgets.context_budget,
                )
                result = await asyncio.wait_for(
                    run_episode(
                        client=client,
                        mcp_session=session,
                        transcript=tw,
                        budget=budget,
                        init_prompt=init_prompt,
                        seed=seed,
                        max_tokens=bench.budgets.max_tokens,
                        nudges=bench.nudges,
                        excluded_tools=excluded_tools,
                        # Per-run safety_identifier: OpenAI scopes a
                        # cyber_policy revocation to the identifier rather
                        # than the whole org. Prefix makes it grep-able in
                        # OpenAI's dashboard if we ever need to correlate
                        # blocks back to a run.
                        safety_identifier=f"qed_swe_bench-run-{run_id}",
                        reasoning_replay=model_spec.reasoning_replay,
                    ),
                    timeout=bench.episode_timeout_s,
                )
    except TimeoutError:
        # Re-raise so the outer wrap records + classifies uniformly.
        raise
    except Exception as exc:
        # If the loop already returned a valid EpisodeResult, the exception
        # came from MCP / docker / TaskGroup cleanup AFTER the agent finished.
        # Common case: the loop exits with `no_tool_calls`, the docker
        # container is torn down, and stdio_client's stdout_reader raises
        # BrokenResourceError on its closed channel. The episode itself
        # succeeded — log the cleanup error but use the agent's result.
        if result is not None:
            log.warning(
                "[%s] cleanup error after successful episode "
                "(result preserved): %s: %s",
                run_id, type(exc).__name__, exc,
            )
        else:
            runtime_s = monotonic() - started
            log.exception("[%s] episode crashed", run_id)
            exit_reason, failure_reason = format_failure(exc)
            _record_infra_failure(
                run_id,
                exit_reason=exit_reason,
                failure_reason=failure_reason,
                runtime_s=runtime_s,
                provenance="mock" if mock_llm else "native",
            )
            return RunOutcome("infra_failed", None, run_id)

    # 4. cli_oneshot grading (rl_mcp_v1 detect/solve/patch tasks) runs
    # after the episode in a fresh container. The agent never sees the
    # grader binary during the episode — that's the reward-hacking guard.
    # When no manifest is registered or the manifest uses the in-MCP
    # grade tool, this block is a no-op and we keep the capabilities
    # the loop accumulated from grade-tool calls.
    # `manifest` was loaded above (before run_episode) so we could plumb
    # `excluded_tools` into the loop's tool surface; reuse here.
    cli_oneshot_failure: str | None = None
    if manifest is not None and manifest.grader_kind == "cli_oneshot":
        timeout_s = int((manifest.evaluate or {}).get("timeout_s", 120))
        try:
            grade = await run_cli_oneshot_grader(
                image_ref=resolved.image_digest,
                command=manifest.grader_command or (),
                run_dir=run_dir,
                timeout_s=timeout_s,
            )
            result.capabilities = grade.capabilities
            append_cli_oneshot_grade_log(run_dir, grade.raw)
            log.info("[%s] cli_oneshot grade: %s", run_id, grade.capabilities)
        except (GraderError, ValueError) as exc:
            cli_oneshot_failure = str(exc)
            result.failure_reason = (
                f"cli_oneshot grader failed: {exc}" if not result.failure_reason
                else f"{result.failure_reason}; cli_oneshot grader failed: {exc}"
            )
            log.error("[%s] cli_oneshot grader failed: %s", run_id, exc)

    # 5. Compute cost + score, write files, update DB.
    # Aggregate kept for the tokens_* columns + write_cost_json. Cost
    # itself is computed per-call (not aggregate) so each call routes
    # to the correct long-context tier — see compute_total_cost.
    usage_totals = NormalizedUsage(
        input_tokens=result.tokens_in,
        output_tokens=result.tokens_out,
        cache_read_tokens=result.tokens_cache_read,
        cache_creation_tokens=result.tokens_cache_creation,
        reasoning_tokens=result.reasoning_tokens,
    )
    if mock_llm:
        cost_usd, cost_source = 0.0, "mock"
    else:
        cost_usd, cost_source = compute_total_cost(model, result.per_call_usages)

    score_value = compute_score(result.capabilities, DEFAULT_SCORING_POLICY)

    if cli_oneshot_failure is not None:
        terminal_status = "infra_failed"
    elif result.exit_reason.startswith("error:"):
        terminal_status = "model_failed"
    else:
        terminal_status = "succeeded"
    finished_at_iso = datetime.now(UTC).isoformat()

    write_score_json(
        run_dir,
        capabilities=result.capabilities,
        score=score_value,
        exit_reason=result.exit_reason,
        finished_at=finished_at_iso,
        runtime_s=result.runtime_s,
        turns_used=result.turns_used,
        failure_reason=result.failure_reason,
        status=terminal_status,
        weighted_tokens_used=result.weighted_tokens_used,
        peak_per_turn_context=result.peak_per_turn_context,
    )
    # OpenRouter cost reconciliation: for OR-routed cells, look up the
    # authoritative billed cost via OR's /generation endpoint per turn
    # and write the aggregate alongside the inferred cost_usd. Best-
    # effort — any HTTP error logs and is captured as `incomplete=True`
    # in the summary so downstream tooling can flag it. Skipped silently
    # for non-OR cells (no `gen-...` ids in transcript) and in mock-LLM
    # mode (no real API calls happened).
    or_recon_dict: dict | None = None
    if not mock_llm:
        try:
            from qed_swe_bench.runner.openrouter_recon import reconcile_run_dir
            recon = await asyncio.to_thread(reconcile_run_dir, run_dir)
            if recon is not None:
                or_recon_dict = recon.to_dict()
                log.info(
                    "[%s] OR cost recon: resolved=%d/%d "
                    "authoritative=$%.4f cache_discount=$%.4f%s",
                    run_id,
                    recon.n_turns_resolved,
                    recon.n_turns_recorded,
                    recon.cost_usd_authoritative,
                    recon.cache_discount_total,
                    " (incomplete)" if recon.incomplete else "",
                )
        except Exception:  # noqa: BLE001
            log.exception("[%s] OR cost reconciliation crashed; continuing", run_id)

    write_cost_json(
        run_dir,
        model=model,
        usage=usage_totals,
        cost_usd=cost_usd,
        cost_source=cost_source,
        weighted_tokens=result.weighted_tokens_used,
        served_model=result.served_model,
        llm_route=getattr(client, "route", None),
        api_base=api_base,
        or_reconciliation=or_recon_dict,
    )

    # Always surface what was actually served — divergence from `model` is
    # normal (OpenAI returns dated snapshots like `gpt-5.5-2026-04-23`,
    # Anthropic strips the `anthropic/` prefix), but the value is recorded
    # so you can detect silent downgrades after the fact.
    log.info(
        "[%s] requested=%s served=%s reasoning_tokens=%d",
        run_id,
        model,
        result.served_model or "<none>",
        result.reasoning_tokens,
    )

    _update_finished(
        run_id=run_id,
        status=terminal_status,
        capabilities=result.capabilities,
        score=score_value,
        usage_totals=usage_totals,
        cost_usd=cost_usd,
        cost_source=cost_source,
        runtime_s=result.runtime_s,
        turns_used=result.turns_used,
        exit_reason=result.exit_reason,
        llm_route=getattr(client, "route", "unknown"),
        api_base=api_base,
        failure_reason=result.failure_reason,
        provenance="mock" if mock_llm else "native",
        weighted_tokens_used=result.weighted_tokens_used,
        peak_per_turn_context=result.peak_per_turn_context,
    )
    # Tick the running spend total so subsequent tuples can short-
    # circuit when the cap is reached. Mock runs report cost=0 and
    # contribute nothing.
    await spend_tracker.add(cost_usd)

    log.info(
        "[%s] %s: %s, score=%s, cost=$%s",
        run_id,
        terminal_status,
        result.capabilities,
        score_value,
        cost_usd,
    )
    return RunOutcome(terminal_status, cost_usd, run_id)


# ---------------------------------------------------------------------------
# Driver entry point
# ---------------------------------------------------------------------------


async def run_benchmark(
    bench: BenchmarkConfig,
    *,
    cfg: Config | None = None,
    mock_llm: bool = False,
    init_prompt: str | None = None,
    retry_failed: bool = False,
    config_path: Path | str | None = None,
) -> dict[str, int]:
    """Execute a full benchmark. Returns a status histogram for the CLI summary.

    Resilience semantics:
      - `retry_failed=True` deletes prior infra_failed / model_failed rows
        for this benchmark_id so the sweep re-attempts them. succeeded
        rows are never touched.
      - `mark_stale_queued_as_failed()` always runs at startup, recovering
        any queued rows orphaned by a prior crash.
      - Per-tuple wallclock timeout is `bench.episode_timeout_s`.
      - Optional `bench.cost_cap_usd` short-circuits remaining tuples
        once the running total crosses the cap.
      - `asyncio.gather(..., return_exceptions=True)`: a bug that
        escapes `_run_one`'s outer wrap doesn't take down the whole
        sweep — sibling tuples keep running, and the escaped exception
        is logged + counted as infra_failed in the histogram.
    """
    cfg = cfg or Config.from_env()
    init_db(cfg.db_path)

    # Resilience: clean up before scheduling.
    if retry_failed:
        delete_failed_rows(bench.benchmark_id)
    mark_stale_queued_as_failed()

    # Init prompt: bench.init_prompt → bench.init_prompt_path → caller arg
    # → default. The container's MCP setup() carries all bug-specific
    # framing; this is just the first user turn that points the agent at it.
    if bench.init_prompt is not None:
        init_p = bench.init_prompt
    elif bench.init_prompt_path is not None:
        init_p = _read_prompt_file(bench.init_prompt_path, "init_prompt_path")
    elif init_prompt is not None:
        init_p = init_prompt
    else:
        init_p = default_init_prompt()

    # Optional hint: appended with a blank line between. Cheap experimental
    # knob for prompt-engineering studies without forking the init body.
    if bench.init_prompt_hint is not None:
        init_p = f"{init_p}\n\n{bench.init_prompt_hint}"
    elif bench.init_prompt_hint_path is not None:
        hint = _read_prompt_file(bench.init_prompt_hint_path, "init_prompt_hint_path")
        init_p = f"{init_p}\n\n{hint}"

    # New layered layout (D-10): runs/<benchmark_id>/<host>/<datetime>/<run_id>/.
    # We pass cfg.runs_dir (the unscoped root) down to _run_one_body and
    # let it construct the full per-run path via construct_run_dir_path.
    run_root = cfg.runs_dir
    run_root.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(bench.max_parallel)
    spend_tracker = SpendTracker(cap_usd=bench.cost_cap_usd)

    # Size the asyncio default thread pool so every concurrent episode's
    # `asyncio.to_thread(client.complete, ...)` (loop.py) gets its own
    # worker thread. The CPython default is min(32, cpu_count + 4); on
    # high-parallel runs (--max-parallel > 32) episodes would otherwise
    # queue behind the pool, partially re-introducing the very
    # serialization the to_thread wrap is meant to eliminate. Floor at
    # 32 to preserve the historical default for small runs.
    import concurrent.futures
    loop = asyncio.get_running_loop()
    loop.set_default_executor(
        concurrent.futures.ThreadPoolExecutor(
            max_workers=max(32, bench.max_parallel * 2),
            thread_name_prefix="llm-",
        )
    )

    # Reproducibility provenance — captured once per benchmark process,
    # threaded through to per-tuple job.json + repro_cmd column. See
    # docs/decisions.md (P-N) and docs/FINDINGS.md if/when we lock the
    # column shape further.
    from qed_swe_bench.runner.provenance import (
        env_overrides_snapshot,
        git_sha as _git_sha,
    )
    captured_git_sha = _git_sha()
    captured_env_overrides = env_overrides_snapshot()

    # Heartbeat tracker for stale-queued recovery. Each row this
    # orchestrator inserts is registered here so the background task
    # periodically refreshes its `last_heartbeat`. Sibling-process
    # startup sweeps see the recent heartbeat and skip our row.
    heartbeat = HeartbeatTracker()
    heartbeat.start()

    async def _bound(model: str, env: EnvSpec, seed: int) -> RunOutcome:
        async with sem:
            return await _run_one(
                cfg=cfg,
                bench=bench,
                model=model,
                env=env,
                seed=seed,
                mock_llm=mock_llm,
                init_prompt=init_p,
                run_root=run_root,
                spend_tracker=spend_tracker,
                config_path=config_path,
                git_sha=captured_git_sha,
                env_overrides=captured_env_overrides,
                heartbeat=heartbeat,
            )

    tasks = [
        _bound(m.id, e, s)
        for m in bench.models
        for e in bench.envs
        for s in bench.seeds
    ]
    # return_exceptions=True so an unexpected escape from a single
    # tuple doesn't cancel sibling work that's already running.
    try:
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        await heartbeat.stop()

    histogram: dict[str, int] = {}
    for r in raw_results:
        if isinstance(r, BaseException):
            log.exception(
                "tuple raised past _run_one's safety wrap (this is a bug); "
                "counting as infra_failed: %s", r,
            )
            histogram["infra_failed"] = histogram.get("infra_failed", 0) + 1
            continue
        histogram[r.status] = histogram.get(r.status, 0) + 1

    final_total = await spend_tracker.total()
    log.info("benchmark %s done; total spend $%.2f; histogram=%s",
             bench.benchmark_id, final_total, histogram)
    return histogram


