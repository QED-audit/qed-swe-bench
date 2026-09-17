"""Resume a partial / failed episode: replay tool sequence against a
fresh container, rehydrate LLM message history from transcript.jsonl,
continue `run_episode` from the next turn.

is_resumable matrix:

  status         exit_reason / failure_reason             why
  ------------   --------------------------------------   ----------------
  infra_failed   exit_reason='crashed: TimeoutError'      episode wallclock
                                                           ran out — agent
                                                           was progressing
  infra_failed   failure_reason='stale_queued_recovery'   orchestrator
                                                           crashed before
                                                           finalizing
  model_failed   error: Timeout / APIConnectionError /    transient
                  ServiceUnavailable / InternalServer /    transport or
                  RateLimit                                quota, retry
                                                           after wait
                                                           likely succeeds
  model_failed   error: BadRequestError                   default: NOT
                                                           resumable (context
                                                           overflow shape) —
                                                           but failure_reason
                                                           text can promote
                                                           wrapped 429s back
                                                           to resumable; see
                                                           _TRANSIENT_TEXT_MARKERS
  model_failed   transient class + "context window        DEMOTED to
                  exceeds limit" / "request too large"     not-resumable: the
                                                           class name lies, the
                                                           text reveals the
                                                           true cause is
                                                           overflow / single-
                                                           request too big;
                                                           same prompt → same
                                                           400 on retry. See
                                                           _CONTEXT_OVERFLOW_TEXT_MARKERS
  model_failed   error: ModelMismatchError                silent reroute is
                                                           an integrity concern,
                                                           NOT resumable
  succeeded      *                                         already finished
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qed_swe_bench.runner.budget import Budget, GRADE_NUDGE_INTERVAL, WRAPUP_FRACTION
from qed_swe_bench.runner.capabilities import (
    DEFAULT_SCORING_POLICY,
    compute_score,
)
from qed_swe_bench.runner.llm.base import (
    NormalizedResponse,
    NormalizedToolCall,
    NormalizedUsage,
)
from qed_swe_bench.runner.llm.messages import MessageList, build_messages_for

log = logging.getLogger(__name__)


# Lowercased substrings matched against `error: <ClassName>` exit_reasons.
_TRANSIENT_ERROR_SUFFIXES: tuple[str, ...] = (
    "timeout",
    "apiconnection",
    "serviceunavailable",
    "internalserver",
    "ratelimit",
)

# Lowercased substrings in failure_reason text that mean the underlying
# error is rate-limit / quota even when LiteLLM wrapped it as a non-transient
# class. Observed in the wild:
#   - Gemini 429s come back as `BadRequestError` with text containing
#     "Quota exceeded for metric: ..." / "RESOURCE_EXHAUSTED" / "Please
#     retry in Xs.". By class alone they look terminal; the text is
#     authoritative.
# Promotes `is_resumable` to True for such cases.
_TRANSIENT_TEXT_MARKERS: tuple[str, ...] = (
    "quota exceeded",
    "exceeded your current quota",
    "resource_exhausted",
    "please retry in",
    "rate limit reached",
)

# Lowercased substrings in failure_reason text that mean the underlying
# error is context-overflow / single-request-too-large — terminal, same
# prompt → same 400 on retry. Demotes `is_resumable` to False when paired
# with a transient class. All four are observed in the V8 sweep:
#   - "context window exceeds limit" — Minimax APIConnectionError wrapping
#     a 400 (the only one observed in the demote branch in current data).
#   - "request too large" — gpt-5.5 RateLimitError wrapping a single
#     request that exceeds the 400K-TPM long-context cap (can't fit in
#     ANY one minute window). Also fires the demote branch.
#   - "exceeded model token limit" — kimi/Moonshot phrasing.
#   - "exceeds max length" — zai/GLM phrasing.
_CONTEXT_OVERFLOW_TEXT_MARKERS: tuple[str, ...] = (
    "context window exceeds limit",
    "exceeded model token limit",
    "exceeds max length",
    "request too large",
)


def is_resumable(
    *,
    status: str,
    exit_reason: str | None,
    failure_reason: str | None = None,
) -> bool:
    """Decide whether a row is a candidate for resume."""
    if status == "infra_failed":
        if (failure_reason or "") == "stale_queued_recovery":
            return True
        if (exit_reason or "").startswith("crashed: TimeoutError"):
            return True
        # cap trip is a budget decision, not a property of the cell. After
        # the operator raises (or removes) the cap, the cell can resume
        # against its preserved transcript without re-running from scratch.
        if (exit_reason or "") == "cost_cap_exceeded":
            return True
        return False
    if status == "model_failed":
        er = (exit_reason or "").lower()
        fr = (failure_reason or "").lower()
        if not er.startswith("error: "):
            return False
        cls = er[len("error: "):]
        is_transient_class = any(suffix in cls for suffix in _TRANSIENT_ERROR_SUFFIXES)
        # Demote a "transient"-class error whose message reveals the
        # underlying cause is context overflow. Resume would re-feed the
        # same prompt → same 400.
        if is_transient_class and any(m in fr for m in _CONTEXT_OVERFLOW_TEXT_MARKERS):
            return False
        if is_transient_class:
            return True
        # Promote a non-transient-class error whose message contains
        # rate-limit / quota text (LiteLLM wrapping a 429 as
        # BadRequestError). After the quota window reopens, resume succeeds.
        if any(marker in fr for marker in _TRANSIENT_TEXT_MARKERS):
            return True
        return False
    return False


@dataclass(frozen=True)
class ResumeState:
    """Pre-built inputs for `run_episode` to continue from where the
    failed run left off."""

    messages: MessageList
    best_caps: dict[str, bool]
    served_model: str | None
    prior_turns: int
    prior_input_tokens: int
    prior_output_tokens: int
    prior_cache_read_tokens: int
    prior_cache_creation_tokens: int
    prior_reasoning_tokens: int
    prior_weighted_tokens: int
    prior_peak_per_turn_context: int
    prior_turns_since_grade: int
    image_digest: str
    init_prompt_seen: bool = field(default=True)
    # Per-call usage records from saved transcript ai entries, in order.
    # Used by `run_episode` to seed its own per_call_usages list so the
    # post-episode `compute_total_cost` sees both prior and new calls.
    prior_per_call_usages: tuple[NormalizedUsage, ...] = field(default_factory=tuple)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _drop_dangling_ai_turns(
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Drop trailing AI turns whose tool calls have no matching tool results.

    All chat providers reject a history whose last assistant turn has
    unmatched tool calls. Episodes that die mid-tool-call (hung grade,
    wallclock, etc.) leave that shape behind; the agent re-issues the
    dropped call live on resume."""
    while entries:
        idx = max(
            (i for i, e in enumerate(entries) if e.get("role") == "ai"),
            default=-1,
        )
        if idx == -1:
            return entries
        last_ai = entries[idx]
        tcs = last_ai.get("tool_calls") or ()
        if not tcs:
            return entries
        tc_ids = {str(tc.get("id", "")) for tc in tcs}
        following_results = {
            str(e.get("tool_call_id", ""))
            for e in entries[idx + 1:]
            if e.get("role") == "tool"
        }
        if tc_ids.issubset(following_results):
            return entries
        entries = entries[:idx]
    return entries


def _persist_transcript_trim(
    path: Path,
    trimmed: list[dict[str, Any]],
    original: list[dict[str, Any]],
) -> None:
    """Rewrite `path` with `trimmed`, after a sibling backup of the original.

    Called when `_drop_dangling_ai_turns` removed entries; keeps the on-disk
    transcript consistent with the in-memory state used for resume."""
    from datetime import UTC, datetime
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup = path.with_suffix(f".jsonl.pre-resume-trim-{ts}")
    backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    body = "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in trimmed)
    path.write_text(body, encoding="utf-8")
    log.info(
        "trimmed %d dangling entries from %s (backup: %s)",
        len(original) - len(trimmed), path.name, backup.name,
    )


def _rebuild_response_from_ai_entry(entry: dict[str, Any]) -> NormalizedResponse:
    """Recover a NormalizedResponse from a transcript ai entry.

    Loop wrote either `tool_calls: [{id, name, args}]` (compact form) or
    `content_blocks: [...]` (anthropic-native, when thinking is present).
    Both paths come back in here.
    """
    tool_calls = tuple(
        NormalizedToolCall(
            id=str(tc.get("id", "")),
            name=str(tc.get("name", "")),
            arguments=dict(tc.get("args") or {}),
        )
        for tc in entry.get("tool_calls") or ()
    )
    blocks_raw = entry.get("content_blocks")
    content_blocks = (
        tuple(dict(b) for b in blocks_raw) if blocks_raw else ()
    )

    usage_dict = entry.get("usage") or {}
    usage = NormalizedUsage(
        input_tokens=int(usage_dict.get("input_tokens", 0) or 0),
        output_tokens=int(usage_dict.get("output_tokens", 0) or 0),
        cache_read_tokens=int(usage_dict.get("cache_read", 0) or 0),
        cache_creation_tokens=int(usage_dict.get("cache_creation", 0) or 0),
        reasoning_tokens=int(usage_dict.get("reasoning_tokens", 0) or 0),
    )

    return NormalizedResponse(
        text=entry.get("content") or "",
        tool_calls=tool_calls,
        usage=usage,
        stop_reason=str(entry.get("stop_reason") or "unknown"),
        model=str(entry.get("served_model") or ""),
        content_blocks=content_blocks,
        reasoning_content=entry.get("reasoning_content"),
    )


def _rebuild_messages(
    transcript_entries: list[dict[str, Any]],
    *,
    route: str,
    model_id: str,
    reasoning_replay: str = "auto",
) -> MessageList:
    """Replay transcript entries into a `MessageList` builder.

    `system` entries are skipped (system prompts are rebuilt by
    `build_messages_for`). Consecutive `tool` entries batch into one
    `append_tool_results` call to match the provider message shape.
    """
    msgs = build_messages_for(
        route,
        model_id=model_id,
        reasoning_replay=reasoning_replay,
    )
    pending_tool_results: list[tuple[NormalizedToolCall, str, bool]] = []
    seen_initial = False

    def flush_tools() -> None:
        if pending_tool_results:
            msgs.append_tool_results(list(pending_tool_results))
            pending_tool_results.clear()

    for entry in transcript_entries:
        role = entry.get("role")
        if role == "system":
            continue
        if role == "marker":
            # Operational marker (e.g. `[resume]` checkpoint). Never feed
            # to the LLM — it's metadata for ops/audit only.
            continue
        if role == "human":
            flush_tools()
            text = entry.get("content") or ""
            # Legacy compat: pre-marker-role runs wrote `[resume] continuing
            # episode at turn N` as role=human. Treat it the same as a
            # marker entry so old transcripts also stop biasing future
            # resumes toward summary/wrap-up behavior.
            if isinstance(text, str) and text.startswith("[resume] "):
                continue
            if not seen_initial:
                msgs.initial(text)
                seen_initial = True
            else:
                msgs.append_user(text)
        elif role == "ai":
            flush_tools()
            msgs.append_assistant(_rebuild_response_from_ai_entry(entry))
        elif role == "tool":
            tc = NormalizedToolCall(
                id=str(entry.get("tool_call_id", "")),
                name=str(entry.get("name", "")),
                arguments={},
            )
            content = entry.get("content") or ""
            pending_tool_results.append((tc, content, False))

    flush_tools()
    return msgs


def _rebuild_best_caps(grade_calls: list[dict[str, Any]]) -> dict[str, bool]:
    """Cumulative-OR over every recorded grade result's capability bitmap."""
    best: dict[str, bool] = {}
    for entry in grade_calls:
        result = entry.get("result")
        if not isinstance(result, dict):
            continue
        caps = result.get("capabilities") or {}
        for k, v in caps.items():
            if v:
                best[str(k)] = True
            elif str(k) not in best:
                best[str(k)] = False
    return best


def _last_served_model(transcript_entries: list[dict[str, Any]]) -> str | None:
    """Most recent non-empty `served_model` across ai turns."""
    for entry in reversed(transcript_entries):
        if entry.get("role") != "ai":
            continue
        served = entry.get("served_model")
        if served:
            return str(served)
    return None


def _prior_token_totals(
    transcript_entries: list[dict[str, Any]],
) -> tuple[int, int, int, int, int, int, int]:
    """Sum (input, output, cache_read, cache_creation, reasoning,
    weighted, peak_per_turn) across ai turns. Weighted re-applies
    `Budget.tick_ai_turn`'s formula per-turn; peak_per_turn is the max
    of (input+output) across turns, matching `Budget.last_turn_context`
    semantics."""
    from qed_swe_bench.runner.budget import CACHE_READ_WEIGHT, OUTPUT_WEIGHT

    sums = [0, 0, 0, 0, 0, 0]
    peak = 0
    keys = ("input_tokens", "output_tokens", "cache_read",
            "cache_creation", "reasoning_tokens")
    for e in transcript_entries:
        if e.get("role") != "ai":
            continue
        usage = e.get("usage") or {}
        per_turn = [int(usage.get(k, 0) or 0) for k in keys]
        for i, v in enumerate(per_turn):
            sums[i] += v
        in_t, out_t, cr, cc, _ = per_turn
        base_in = max(0, in_t - cr - cc)
        sums[5] += base_in + cc + int(cr * CACHE_READ_WEIGHT) + OUTPUT_WEIGHT * out_t
        if in_t + out_t > peak:
            peak = in_t + out_t
    return (*sums, peak)


def _prior_per_call_usages(
    transcript_entries: list[dict[str, Any]],
) -> tuple[NormalizedUsage, ...]:
    """Reconstruct per-ai-turn NormalizedUsage records from saved transcript.

    Used to feed ResumeState.prior_per_call_usages so the resumed episode's
    final cost computation runs over the full call history at the correct
    per-call price tier. The mapping is the same one TranscriptWriter
    writes (input_tokens / output_tokens / cache_read / cache_creation /
    reasoning_tokens).
    """
    out: list[NormalizedUsage] = []
    for e in transcript_entries:
        if e.get("role") != "ai":
            continue
        u = e.get("usage") or {}
        out.append(NormalizedUsage(
            input_tokens=int(u.get("input_tokens", 0) or 0),
            output_tokens=int(u.get("output_tokens", 0) or 0),
            cache_read_tokens=int(u.get("cache_read", 0) or 0),
            cache_creation_tokens=int(u.get("cache_creation", 0) or 0),
            reasoning_tokens=int(u.get("reasoning_tokens", 0) or 0),
        ))
    return tuple(out)


def _read_image_digest(run_dir: Path) -> str | None:
    job = run_dir / "job.json"
    if not job.is_file():
        return None
    try:
        data = json.loads(job.read_text())
    except json.JSONDecodeError:
        return None
    digest = data.get("image_digest")
    return str(digest) if digest else None


def prepare_resume_state(
    run_dir: Path, *, route: str, model_id: str, reasoning_replay: str = "auto",
) -> ResumeState:
    """Read run_dir, return the inputs `run_episode` needs to continue."""
    image_digest = _read_image_digest(run_dir)
    if image_digest is None:
        raise ValueError(
            f"cannot resume {run_dir}: job.json missing image_digest"
        )

    transcript_path = run_dir / "transcript.jsonl"
    transcript_entries = _read_jsonl(transcript_path)
    trimmed = _drop_dangling_ai_turns(transcript_entries)
    if len(trimmed) < len(transcript_entries):
        _persist_transcript_trim(transcript_path, trimmed, transcript_entries)
        transcript_entries = trimmed
    grade_entries = _read_jsonl(run_dir / "grade_calls.jsonl")
    messages = _rebuild_messages(
        transcript_entries, route=route, model_id=model_id,
        reasoning_replay=reasoning_replay,
    )
    best_caps = _rebuild_best_caps(grade_entries)
    served_model = _last_served_model(transcript_entries)

    prior_turns = sum(1 for e in transcript_entries if e.get("role") == "ai")
    in_t, out_t, cr_t, cc_t, rt, wt, peak = _prior_token_totals(transcript_entries)
    prior_calls = _prior_per_call_usages(transcript_entries)

    return ResumeState(
        messages=messages,
        best_caps=best_caps,
        served_model=served_model,
        prior_turns=prior_turns,
        prior_input_tokens=in_t,
        prior_output_tokens=out_t,
        prior_cache_read_tokens=cr_t,
        prior_cache_creation_tokens=cc_t,
        prior_reasoning_tokens=rt,
        prior_weighted_tokens=wt,
        prior_peak_per_turn_context=peak,
        prior_turns_since_grade=_turns_since_last_grade(transcript_entries),
        image_digest=image_digest,
        init_prompt_seen=any(e.get("role") == "human" for e in transcript_entries),
        prior_per_call_usages=prior_calls,
    )


@dataclass(frozen=True)
class ResumeOutcome:
    """Return shape of `resume_one_run`."""

    run_id: str
    status: str
    exit_reason: str | None
    runtime_s: float
    turns_total: int
    cost_usd: float = 0.0
    error: str | None = None


def _turns_since_last_grade(entries: list[dict[str, Any]]) -> int:
    """Reconstruct the live turns_since_grade counter from the transcript.

    Mirrors the loop exactly: at the top of each ai turn the stuck nudge
    resets the counter once it reaches GRADE_NUDGE_INTERVAL (note_grade_called
    on fire), then the turn increments it, then a grade() call on that turn
    resets it. The stuck-reset is simulated unconditionally — when STUCK is
    disabled the counter is never read, so the value is harmless there."""
    tsg = 0
    for e in entries:
        if e.get("role") != "ai":
            continue
        if tsg >= GRADE_NUDGE_INTERVAL:
            tsg = 0
        tsg += 1
        if any(tc.get("name") == "grade" for tc in e.get("tool_calls") or []):
            tsg = 0
    return tsg


def prime_budget(budget: Budget, state: ResumeState) -> None:
    """Bump a fresh Budget's counters by prior-run consumption so the
    original turn / token / context limits and nudge state stay consistent
    across resume. `tokens_used` and `peak_per_turn_context` are
    reconstructed by re-applying tick_ai_turn's formulas per ai entry."""
    budget.turn = state.prior_turns
    budget.total_input_tokens = state.prior_input_tokens
    budget.total_output_tokens = state.prior_output_tokens
    budget.total_cache_read_tokens = state.prior_cache_read_tokens
    budget.total_cache_creation_tokens = state.prior_cache_creation_tokens
    budget.tokens_used = state.prior_weighted_tokens
    budget.peak_per_turn_context = state.prior_peak_per_turn_context
    budget.turns_since_grade = state.prior_turns_since_grade
    # The wrapup nudge is a one-shot fired the moment `turn` first crosses the
    # threshold; if we resumed past it, it already fired. Latch so it doesn't
    # re-fire.
    if budget.turn_budget:
        budget.sent_wrapup = state.prior_turns >= int(budget.turn_budget * WRAPUP_FRACTION)


def select_resumable_run_dirs(
    *,
    benchmark_id: str | None = None,
    model_ids: list[str] | None = None,
    env_ids: list[str] | None = None,
    seeds: list[int] | None = None,
) -> list[tuple[str, Path]]:
    """`(run_id, run_dir)` pairs for failed rows that pass `is_resumable`,
    oldest started_at first. `benchmark_id=None` selects across all
    benchmarks. `model_ids`, `env_ids`, and `seeds` further narrow the
    selection so `--resume-failed` honors the same `--models`/`--envs`/
    `--seeds` filters the fresh-run path uses; pass None (the default)
    to skip a filter."""
    from qed_swe_bench.db.schema import transaction
    where: list[str] = ["status IN ('infra_failed', 'model_failed')"]
    params: list[object] = []
    if benchmark_id is not None:
        where.append("benchmark_id=?")
        params.append(benchmark_id)
    if model_ids:
        where.append("model IN (" + ",".join("?" for _ in model_ids) + ")")
        params.extend(model_ids)
    if env_ids:
        where.append("env_id IN (" + ",".join("?" for _ in env_ids) + ")")
        params.extend(env_ids)
    if seeds:
        where.append("seed IN (" + ",".join("?" for _ in seeds) + ")")
        params.extend(seeds)
    sql = (
        "SELECT run_id, status, exit_reason, failure_reason, run_dir "
        "FROM runs WHERE " + " AND ".join(where) + " ORDER BY started_at"
    )
    rows: list[tuple[str, Path]] = []
    with transaction() as con:
        cur = con.execute(sql, params)
        for run_id, status, exit_reason, failure_reason, run_dir in cur.fetchall():
            if not run_dir:
                continue
            if not is_resumable(
                status=status,
                exit_reason=exit_reason,
                failure_reason=failure_reason,
            ):
                continue
            rows.append((run_id, Path(run_dir)))
    return rows


def _accumulate_and_persist_failure(
    *,
    run_id: str,
    new_runtime_s: float,
    exit_reason: str,
    failure_reason: str,
    mock_llm: bool,
) -> float:
    """Add `new_runtime_s` to the DB row's existing runtime_s, persist as
    infra_failed, and return the cumulative.

    Called from the failure paths in `_resume_one_run_post_claim` so a
    resume attempt that crashes/times out doesn't silently lose its
    wall-clock. Without this, the row's runtime_s stays at the
    pre-resume value and a subsequent successful resume would compute
    its own `prior + new` against that stale prior, dropping the failed
    resume's seconds entirely (under-count).

    Mirrors the cumulative-read logic in the success path at L799-806
    so both paths land an additive value via `update_finished`.
    """
    from qed_swe_bench.db.schema import transaction
    from qed_swe_bench.runner.runs_db import record_infra_failure

    prior_runtime_s = 0.0
    with transaction() as con:
        row = con.execute(
            "SELECT runtime_s FROM runs WHERE run_id=?", (run_id,),
        ).fetchone()
        if row and row[0] is not None:
            prior_runtime_s = float(row[0])
    cumulative = prior_runtime_s + new_runtime_s
    record_infra_failure(
        run_id,
        exit_reason=exit_reason,
        failure_reason=failure_reason,
        runtime_s=cumulative,
        provenance="mock" if mock_llm else "native",
    )
    return cumulative


async def resume_one_run(
    run_dir: Path,
    *,
    budgets: Any,
    episode_timeout_s: int,
    model_params: dict[str, Any] | None = None,
    nudges: frozenset | None = None,
    mock_llm: bool = False,
    cfg: Any = None,
    on_replay_progress: Any = None,
    heartbeat: Any = None,
    reasoning_replay: str = "auto",
) -> ResumeOutcome:
    """Resume one failed/partial run end-to-end.

    Reads job.json, builds an LLM client, opens MCP at the recorded
    image_digest, replays tool_calls (skipping `grade`) to rebuild fs
    state, calls `run_episode` with the rehydrated message history, and
    finalizes cost / score / DB. `budgets` and `episode_timeout_s` are
    required: resume must mirror the original run's limits, never
    invent its own — the caller (run_resume_batch from `bench`, or the
    standalone CLI from the run-dir's `config_snapshot.yaml`) is
    responsible for supplying them.
    """
    from qed_swe_bench.config import Config
    from qed_swe_bench.runner.runs_db import claim_for_resume

    cfg = cfg or Config.from_env()
    job_path = run_dir / "job.json"
    if not job_path.is_file():
        raise FileNotFoundError(f"job.json not found in {run_dir}")
    job = json.loads(job_path.read_text())
    run_id = str(job["run_id"])
    model = str(job["model"])
    env_id = str(job["env_id"])
    seed = int(job["seed"])
    image_digest = str(job["image_digest"])

    # Atomic claim: flips the row from infra_failed/model_failed → running
    # and seeds last_heartbeat. If another --resume-failed already claimed
    # the row, bail out without touching anything.
    if not claim_for_resume(run_id):
        log.info("[%s] resume skipped: row not in a resumable state", run_id)
        return ResumeOutcome(
            run_id=run_id, status="skipped",
            exit_reason="already_claimed_or_terminal",
            runtime_s=0.0, turns_total=0,
        )
    if heartbeat is not None:
        await heartbeat.add(run_id)
    try:
        return await _resume_one_run_post_claim(
            run_dir=run_dir,
            run_id=run_id,
            model=model,
            env_id=env_id,
            seed=seed,
            image_digest=image_digest,
            model_params=model_params or {},
            nudges=nudges or frozenset(),
            mock_llm=mock_llm,
            episode_timeout_s=episode_timeout_s,
            budgets=budgets,
            cfg=cfg,
            on_replay_progress=on_replay_progress,
        )
    finally:
        if heartbeat is not None:
            await heartbeat.remove(run_id)


async def _resume_one_run_post_claim(
    *,
    run_dir: Path,
    run_id: str,
    model: str,
    env_id: str,
    seed: int,
    image_digest: str,
    model_params: dict[str, Any],
    nudges: frozenset,
    mock_llm: bool,
    episode_timeout_s: int,
    budgets: Any,
    cfg: Any,
    on_replay_progress: Any,
) -> ResumeOutcome:
    """Body of resume_one_run after the row has been claimed and the
    heartbeat tracker (if any) has the run_id. Split out so the
    heartbeat lifecycle in the caller's try/finally is free to handle
    every exit path without indenting this body."""
    import asyncio
    from datetime import UTC, datetime
    from time import monotonic

    from qed_swe_bench.runner.budget import Budget
    from qed_swe_bench.runner.cost import compute_total_cost
    from qed_swe_bench.runner.env_manifest import (
        EnvManifestLoadError, load_env_manifest,
    )
    from qed_swe_bench.runner.llm.factory import build_client
    from qed_swe_bench.runner.loop import run_episode
    from qed_swe_bench.runner.mcp_client import McpDockerSession
    from qed_swe_bench.runner.replay import (
        read_tool_calls_jsonl, replay_tool_calls,
    )
    from qed_swe_bench.runner.run_dir import write_cost_json, write_score_json
    from qed_swe_bench.runner.runs_db import update_finished
    from qed_swe_bench.runner.transcript import TranscriptWriter

    log.info(
        "[%s] resume start: model=%s env=%s seed=%d image=%s",
        run_id, model, env_id, seed, image_digest[:24],
    )

    try:
        manifest = load_env_manifest(env_id)
    except EnvManifestLoadError as exc:
        return ResumeOutcome(
            run_id=run_id, status="infra_failed",
            exit_reason="manifest_load_failed",
            runtime_s=0.0, turns_total=0,
            error=str(exc),
        )
    excluded_tools = (
        frozenset(manifest.evaluation_tools) if manifest is not None else frozenset()
    )

    client = build_client(
        model, mock=mock_llm, config=cfg, params=model_params,
    )

    try:
        state = prepare_resume_state(
            run_dir, route=client.route, model_id=client.model,
            reasoning_replay=reasoning_replay,
        )
    except ValueError as exc:
        return ResumeOutcome(
            run_id=run_id, status="infra_failed",
            exit_reason="resume_state_unavailable",
            runtime_s=0.0, turns_total=0,
            error=str(exc),
        )

    budget = Budget(
        turn_budget=budgets.turn_budget,
        token_budget=budgets.token_budget,
        context_budget=budgets.context_budget,
    )

    started = monotonic()
    result = None
    try:
        async with McpDockerSession.start(
            image_digest, stderr_path=run_dir / "mcp_stderr.log",
        ) as session:
            tool_calls = read_tool_calls_jsonl(run_dir / "tool_calls.jsonl")
            replayed = await replay_tool_calls(
                session, tool_calls, skip={"grade"},
                on_progress=on_replay_progress,
            )
            n_errs = sum(1 for r in replayed if r.error is not None)
            log.info(
                "[%s] replayed %d tool calls (%d skipped, %d errors)",
                run_id,
                len(replayed),
                sum(1 for r in replayed if r.skipped),
                n_errs,
            )

            with TranscriptWriter(run_dir) as tw:
                tw.write_marker(
                    f"[resume] continuing episode at turn {state.prior_turns + 1}"
                )
                result = await asyncio.wait_for(
                    run_episode(
                        client=client,
                        mcp_session=session,
                        transcript=tw,
                        budget=budget,
                        init_prompt="",
                        seed=seed,
                        max_tokens=budgets.max_tokens,
                        nudges=nudges,
                        excluded_tools=excluded_tools,
                        safety_identifier=f"qed_swe_bench-run-{run_id}",
                        resume_state=state,
                    ),
                    timeout=episode_timeout_s,
                )
    except TimeoutError:
        new_runtime_s = monotonic() - started
        cumulative = _accumulate_and_persist_failure(
            run_id=run_id,
            new_runtime_s=new_runtime_s,
            exit_reason="crashed: TimeoutError",
            failure_reason=f"resume timed out after {episode_timeout_s}s",
            mock_llm=mock_llm,
        )
        return ResumeOutcome(
            run_id=run_id, status="infra_failed",
            exit_reason="crashed: TimeoutError",
            runtime_s=cumulative,
            turns_total=state.prior_turns,
            error=f"resume timed out after {episode_timeout_s}s",
        )
    except Exception as exc:  # noqa: BLE001
        if result is None:
            log.exception("[%s] resume crashed", run_id)
            new_runtime_s = monotonic() - started
            cumulative = _accumulate_and_persist_failure(
                run_id=run_id,
                new_runtime_s=new_runtime_s,
                exit_reason=f"error: {type(exc).__name__}",
                failure_reason=str(exc),
                mock_llm=mock_llm,
            )
            return ResumeOutcome(
                run_id=run_id, status="infra_failed",
                exit_reason=f"error: {type(exc).__name__}",
                runtime_s=cumulative,
                turns_total=state.prior_turns,
                error=str(exc),
            )
        log.warning("[%s] cleanup error after resume (result preserved): %s",
                    run_id, exc)

    # Aggregate kept for the tokens_* columns + write_cost_json. Cost
    # itself is computed per-call (not aggregate) so each call routes
    # to the correct long-context tier — see compute_total_cost.
    # `result.per_call_usages` merges the saved transcript's prior calls
    # (via ResumeState.prior_per_call_usages) with the new calls executed
    # during this resume, so a single pass yields the cumulative cost.
    usage_totals = NormalizedUsage(
        input_tokens=result.tokens_in,
        output_tokens=result.tokens_out,
        cache_read_tokens=result.tokens_cache_read,
        cache_creation_tokens=result.tokens_cache_creation,
        reasoning_tokens=result.reasoning_tokens,
    )
    cost_usd, cost_source = (
        (0.0, "mock") if mock_llm
        else compute_total_cost(model, result.per_call_usages)
    )

    if result.exit_reason.startswith("error:"):
        terminal_status = "model_failed"
    else:
        terminal_status = "succeeded"
    finished_at_iso = datetime.now(UTC).isoformat()

    # Read the prior row's runtime_s so the persisted value is cumulative
    # across the original death + this resume's wallclock. Other cumulative
    # axes (turns, tokens, weighted, peak) are recovered from transcript
    # by prepare_resume_state; runtime isn't on disk anywhere else.
    from qed_swe_bench.db.schema import transaction
    prior_runtime_s = 0.0
    with transaction() as con:
        row = con.execute(
            "SELECT runtime_s FROM runs WHERE run_id=?", (run_id,),
        ).fetchone()
        if row and row[0] is not None:
            prior_runtime_s = float(row[0])
    cumulative_runtime_s = prior_runtime_s + result.runtime_s

    score_value = compute_score(result.capabilities, DEFAULT_SCORING_POLICY)

    write_score_json(
        run_dir,
        capabilities=result.capabilities,
        score=score_value,
        exit_reason=result.exit_reason,
        finished_at=finished_at_iso,
        runtime_s=cumulative_runtime_s,
        turns_used=result.turns_used,
        failure_reason=result.failure_reason,
        status=terminal_status,
        weighted_tokens_used=result.weighted_tokens_used,
        peak_per_turn_context=result.peak_per_turn_context,
    )
    write_cost_json(
        run_dir,
        model=model,
        usage=usage_totals,
        cost_usd=cost_usd,
        cost_source=cost_source,
        weighted_tokens=result.weighted_tokens_used,
        served_model=result.served_model,
        llm_route=getattr(client, "route", None),
        api_base=getattr(client, "api_base", None),
    )
    update_finished(
        run_id=run_id,
        status=terminal_status,
        capabilities=result.capabilities,
        score=score_value,
        usage_totals=usage_totals,
        cost_usd=cost_usd,
        cost_source=cost_source,
        runtime_s=cumulative_runtime_s,
        turns_used=result.turns_used,
        exit_reason=result.exit_reason,
        llm_route=getattr(client, "route", "unknown"),
        api_base=getattr(client, "api_base", None),
        failure_reason=result.failure_reason,
        provenance="mock" if mock_llm else "native",
        weighted_tokens_used=result.weighted_tokens_used,
        peak_per_turn_context=result.peak_per_turn_context,
    )
    return ResumeOutcome(
        run_id=run_id, status=terminal_status,
        exit_reason=result.exit_reason,
        runtime_s=cumulative_runtime_s,
        turns_total=result.turns_used,
        cost_usd=cost_usd,
    )


async def run_resume_batch(
    bench: Any,
    *,
    mock_llm: bool = False,
    cfg: Any = None,
) -> dict[str, int]:
    """Resume every failed-but-resumable row for `bench.benchmark_id`.

    Runs `bench.max_parallel` resumes concurrently. Returns a status
    histogram for the CLI summary. Rows whose status doesn't pass
    `is_resumable` are not touched — re-run with `--retry-failed` to
    delete-and-retry those.
    """
    import asyncio
    from qed_swe_bench.config import Config
    from qed_swe_bench.runner.resilience import HeartbeatTracker

    from qed_swe_bench.runner.spend_tracker import SpendTracker

    cfg = cfg or Config.from_env()
    model_ids = [m.id for m in bench.models]
    env_ids = [e.id for e in bench.envs]
    seeds = list(bench.seeds)
    rows = select_resumable_run_dirs(
        benchmark_id=bench.benchmark_id,
        model_ids=model_ids or None,
        env_ids=env_ids or None,
        seeds=seeds or None,
    )
    if not rows:
        return {"no_resumable_rows": 0}

    sem = asyncio.Semaphore(bench.max_parallel)
    params_by_model = {m.id: m.params for m in bench.models}
    spend_tracker = SpendTracker(cap_usd=bench.cost_cap_usd)

    # Heartbeat tracker keeps last_heartbeat fresh on each in-flight
    # resumed row so the stale-queued sweep doesn't reap a slow resume.
    # Mirrors the orchestrator's run_benchmark plumbing.
    heartbeat = HeartbeatTracker()
    heartbeat.start()

    async def _one(run_id: str, run_dir: Path) -> ResumeOutcome:
        async with sem:
            # Cost-cap short-circuit: if a previously-resumed cell already
            # crossed the cap, skip without claiming the row so a future
            # `--resume-failed` (after the operator raises the cap) can
            # still pick it up. Mirrors run_benchmark's cap check but
            # leaves the row's prior status untouched instead of writing
            # cost_cap_exceeded over a real failure_reason.
            if await spend_tracker.cap_exceeded():
                total = await spend_tracker.total()
                log.warning(
                    "[%s] cost cap reached ($%.2f >= $%.2f); skipping resume",
                    run_id, total, spend_tracker.cap_usd,
                )
                return ResumeOutcome(
                    run_id=run_id, status="cost_cap_skipped",
                    exit_reason="cost_cap_exceeded",
                    runtime_s=0.0, turns_total=0,
                )
            try:
                job = json.loads((run_dir / "job.json").read_text())
                model = str(job["model"])
            except Exception:
                model = ""
            outcome = await resume_one_run(
                run_dir,
                budgets=bench.budgets,
                episode_timeout_s=bench.episode_timeout_s,
                model_params=params_by_model.get(model, {}),
                nudges=bench.nudges,
                mock_llm=mock_llm,
                cfg=cfg,
                heartbeat=heartbeat,
            )
            await spend_tracker.add(outcome.cost_usd)
            return outcome

    log.info(
        "resume batch: %d resumable rows for benchmark_id=%s",
        len(rows), bench.benchmark_id,
    )
    try:
        outcomes = await asyncio.gather(
            *[_one(rid, rd) for rid, rd in rows], return_exceptions=True,
        )
    finally:
        await heartbeat.stop()

    histogram: dict[str, int] = {}
    for o in outcomes:
        if isinstance(o, Exception):
            histogram["infra_failed"] = histogram.get("infra_failed", 0) + 1
            log.error("resume batch: unhandled exception: %s", o)
            continue
        histogram[o.status] = histogram.get(o.status, 0) + 1
    return histogram
