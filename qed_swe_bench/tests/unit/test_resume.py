"""Resume module: predicate + state reconstruction.

End-to-end `resume_one_run` is exercised against a stub MCP session
+ a stub LLM client so we don't need docker or live providers. Pure
helpers (`is_resumable`, `_rebuild_*`, `prime_budget`) are tested
directly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from qed_swe_bench.db.schema import init_db
from qed_swe_bench.runner.budget import GRADE_NUDGE_INTERVAL, Budget
from qed_swe_bench.runner.llm.base import NormalizedUsage
from qed_swe_bench.runner.orchestrator_config import EnvSpec
from qed_swe_bench.runner.resume import (
    ResumeState,
    _turns_since_last_grade,
    is_resumable,
    prepare_resume_state,
    prime_budget,
    select_resumable_run_dirs,
)
from qed_swe_bench.runner.runs_db import insert_queued, update_finished


# ---------------- is_resumable ----------------


@pytest.mark.parametrize(
    "status,exit_reason,failure_reason,expected",
    [
        # episode wallclock — was making progress, resume
        ("infra_failed", "crashed: TimeoutError", None, True),
        # orchestrator died — resume
        ("infra_failed", None, "stale_queued_recovery", True),
        # transient transport — resume
        ("model_failed", "error: Timeout", None, True),
        ("model_failed", "error: APIConnectionError", None, True),
        ("model_failed", "error: ServiceUnavailableError", None, True),
        ("model_failed", "error: InternalServerError", None, True),
        # Sustained rate limit (LiteLLM exhausted retries); waiting for the
        # quota window to reopen makes resume succeed.
        ("model_failed", "error: RateLimitError", None, True),
        # context overflow — same prompt → same 400, no point resuming
        ("model_failed", "error: BadRequestError", None, False),
        # silent reroute — integrity concern, re-run from scratch
        ("model_failed", "error: ModelMismatchError", None, False),
        # pre-episode infra failures — no transcript to resume from
        ("infra_failed", "image_unresolved", None, False),
        ("infra_failed", "manifest_load_failed", None, False),
        # cap trip is a budget decision; after operator raises the cap, the
        # cell resumes against the preserved transcript instead of re-running.
        ("infra_failed", "cost_cap_exceeded", None, True),
        # already finished
        ("succeeded", "budget: turn_budget (300 >= 300)", None, False),
        ("succeeded", "no_tool_calls", None, False),
        ("succeeded", "context_window_exceeded", None, False),
        # unknown / running
        ("running", None, None, False),
        ("queued", None, None, False),
    ],
)
def test_is_resumable_classification(status, exit_reason, failure_reason, expected):
    assert is_resumable(
        status=status,
        exit_reason=exit_reason,
        failure_reason=failure_reason,
    ) is expected


def test_is_resumable_handles_unknown_error_class_via_suffix() -> None:
    """A future SDK exception we haven't seen — if the class name ends
    in one of our known transient suffixes, treat as transient."""
    assert is_resumable(
        status="model_failed",
        exit_reason="error: SomeNewProviderTimeout",
        failure_reason=None,
    ) is True
    # Random noise class name — don't classify as transient.
    assert is_resumable(
        status="model_failed",
        exit_reason="error: KeyError",
        failure_reason=None,
    ) is False


# ---------------- failure_reason text-based promote/demote ----------------

# Verbatim sample failure_reason texts taken from the v8 sweep on
# 2026-05-08 (provider responses, slightly trimmed for line length).

_GEMINI_QUOTA_BADREQUEST_TEXT = (
    'litellm.BadRequestError: GeminiException BadRequestError - {"error": '
    '{"code": 429, "message": "You exceeded your current quota, please '
    'check your plan and billing details. ... Quota exceeded for metric: '
    'generate_content_paid_tier_2_input_token_count, limit: 5000000, '
    'model: gemini-3.1-pro\\nPlease retry in 58.584s.", "status": '
    '"RESOURCE_EXHAUSTED"}}'
)

_MINIMAX_OVERFLOW_APICONN_TEXT = (
    'litellm.APIConnectionError: MinimaxException - {"type":"error",'
    '"error":{"type":"bad_request_error","message":"invalid params, '
    'context window exceeds limit (2013)","http_code":"400"}}'
)

_GPT55_REQUEST_TOO_LARGE_TEXT = (
    'litellm.RateLimitError: OpenAIException - "Request too large for '
    'gpt-5.5 (for limit gpt-5.5-long-context) ... Limit 400000, Requested '
    '402827. The input or output tokens must be reduced..."'
)

_GPT55_RATE_LIMIT_REACHED_TEXT = (
    'litellm.RateLimitError: OpenAIException - "Rate limit reached for '
    'gpt-5.5 (for limit gpt-5.5-long-context) ... Limit 400000, Used '
    '276254, Requested 286507. Please try again in 24.414s."'
)


def test_is_resumable_promotes_wrapped_429_in_failure_reason() -> None:
    """LiteLLM wraps Gemini's 429 as BadRequestError; the class alone
    misclassifies the cell as terminal, but the message text reveals
    the underlying cause is a transient quota event."""
    assert is_resumable(
        status="model_failed",
        exit_reason="error: BadRequestError",
        failure_reason=_GEMINI_QUOTA_BADREQUEST_TEXT,
    ) is True


def test_is_resumable_demotes_overflow_wrapped_as_transient_class() -> None:
    """Minimax surfaces context-window 400s as APIConnectionError —
    the class would say resumable but the message reveals overflow."""
    assert is_resumable(
        status="model_failed",
        exit_reason="error: APIConnectionError",
        failure_reason=_MINIMAX_OVERFLOW_APICONN_TEXT,
    ) is False


def test_is_resumable_demotes_gpt55_request_too_large() -> None:
    """gpt-5.5 'Request too large' on the long-context tier — single
    request exceeds the 400K cap, can't fit in any minute window even
    at parallel=1. RateLimitError class hides this; text reveals it."""
    assert is_resumable(
        status="model_failed",
        exit_reason="error: RateLimitError",
        failure_reason=_GPT55_REQUEST_TOO_LARGE_TEXT,
    ) is False


def test_is_resumable_keeps_gpt55_rate_limit_reached_resumable() -> None:
    """Same exit_reason class as 'request too large' but the actual
    cause is accumulated TPM throttling — resumable at parallel=1 once
    the bucket refills."""
    assert is_resumable(
        status="model_failed",
        exit_reason="error: RateLimitError",
        failure_reason=_GPT55_RATE_LIMIT_REACHED_TEXT,
    ) is True


def test_is_resumable_no_promotion_for_random_badrequest_text() -> None:
    """A BadRequestError without quota / rate-limit text stays terminal
    — we don't accidentally promote a context-overflow row."""
    assert is_resumable(
        status="model_failed",
        exit_reason="error: BadRequestError",
        failure_reason="MoonshotException - exceeded model token limit: 262144",
    ) is False


# ---------------- select_resumable_run_dirs ----------------


@pytest.fixture
def tmp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db = tmp_path / "qed_swe_bench.sqlite"
    monkeypatch.setenv("QED_SWE_BENCH_DB", str(db))
    init_db(db)
    return db


def _seed_failed_row(
    *,
    run_id: str,
    benchmark_id: str,
    model: str,
    seed: int,
    run_dir: Path,
) -> None:
    insert_queued(
        run_id=run_id,
        benchmark_id=benchmark_id,
        model=model,
        env=EnvSpec(id="e", image="i"),
        image_digest="sha256:" + "0" * 64,
        seed=seed,
        run_dir=run_dir,
        nudges_used=False,
    )
    update_finished(
        run_id=run_id,
        status="model_failed",
        capabilities={},
        score=0.0,
        usage_totals=NormalizedUsage(),
        cost_usd=0.0,
        cost_source="mock",
        runtime_s=1.0,
        turns_used=1,
        exit_reason="error: APIConnectionError",
        llm_route="native",
        api_base=None,
        failure_reason=None,
    )


def test_select_resumable_run_dirs_filters_by_model_ids(tmp_db: Path, tmp_path: Path) -> None:
    """`--resume-failed --models X` must only claim rows for X. Without the
    filter it picks up rows for every model in the benchmark — corrupting
    other models' transcripts when the resume re-replays tools."""
    _seed_failed_row(run_id="r-a", benchmark_id="b1", model="m-A", seed=1, run_dir=tmp_path)
    _seed_failed_row(run_id="r-b", benchmark_id="b1", model="m-B", seed=1, run_dir=tmp_path)

    rows = select_resumable_run_dirs(benchmark_id="b1", model_ids=["m-A"])
    assert sorted(rid for rid, _ in rows) == ["r-a"]

    rows = select_resumable_run_dirs(benchmark_id="b1", model_ids=["m-A", "m-B"])
    assert sorted(rid for rid, _ in rows) == ["r-a", "r-b"]

    # No filter → all rows for the benchmark.
    rows = select_resumable_run_dirs(benchmark_id="b1")
    assert sorted(rid for rid, _ in rows) == ["r-a", "r-b"]


def test_select_resumable_run_dirs_filters_by_seeds(tmp_db: Path, tmp_path: Path) -> None:
    _seed_failed_row(run_id="r-1", benchmark_id="b1", model="m", seed=1, run_dir=tmp_path)
    _seed_failed_row(run_id="r-2", benchmark_id="b1", model="m", seed=2, run_dir=tmp_path)
    _seed_failed_row(run_id="r-3", benchmark_id="b1", model="m", seed=3, run_dir=tmp_path)

    rows = select_resumable_run_dirs(benchmark_id="b1", seeds=[1, 3])
    assert sorted(rid for rid, _ in rows) == ["r-1", "r-3"]


def test_select_resumable_run_dirs_filters_by_env_ids(
    tmp_db: Path, tmp_path: Path,
) -> None:
    """`--resume-failed --envs X` must only claim rows for env X — same
    invariant as the model and seed filters."""
    insert_queued(
        run_id="r-x", benchmark_id="b1", model="m",
        env=EnvSpec(id="env-X", image="i"),
        image_digest="sha256:" + "0" * 64, seed=1, run_dir=tmp_path,
        nudges_used=False,
    )
    insert_queued(
        run_id="r-y", benchmark_id="b1", model="m",
        env=EnvSpec(id="env-Y", image="i"),
        image_digest="sha256:" + "0" * 64, seed=1, run_dir=tmp_path,
        nudges_used=False,
    )
    for rid in ("r-x", "r-y"):
        update_finished(
            run_id=rid, status="model_failed", capabilities={}, score=0.0,
            usage_totals=NormalizedUsage(), cost_usd=0.0, cost_source="mock",
            runtime_s=1.0, turns_used=1,
            exit_reason="error: APIConnectionError",
            llm_route="native", api_base=None,
            failure_reason=None,
        )

    rows = select_resumable_run_dirs(benchmark_id="b1", env_ids=["env-X"])
    assert sorted(rid for rid, _ in rows) == ["r-x"]


def test_accumulate_and_persist_failure_adds_to_prior_runtime(
    tmp_db: Path, tmp_path: Path,
) -> None:
    """Failed resume's wall-clock is accumulated onto the prior row's
    runtime_s — not overwritten — and the row is left in infra_failed
    so the stale-queued sweep doesn't reap it.

    Regression: previously the TimeoutError/Exception paths in
    `_resume_one_run_post_claim` returned ResumeOutcome with the
    new-session-only runtime and never wrote the DB, so a failed
    resume's seconds were silently dropped (under-count).
    """
    from qed_swe_bench.db.schema import transaction
    from qed_swe_bench.runner.resume import _accumulate_and_persist_failure

    # Seed: a row that already failed once at 100s wall-clock.
    _seed_failed_row(run_id="r-1", benchmark_id="b", model="m", seed=1,
                     run_dir=tmp_path)
    with transaction() as con:
        con.execute("UPDATE runs SET runtime_s=100.0 WHERE run_id='r-1'")

    # First failed resume adds 50s.
    cum = _accumulate_and_persist_failure(
        run_id="r-1", new_runtime_s=50.0,
        exit_reason="crashed: TimeoutError",
        failure_reason="resume timed out", mock_llm=False,
    )
    assert cum == 150.0
    with transaction() as con:
        row = con.execute(
            "SELECT runtime_s, status FROM runs WHERE run_id='r-1'"
        ).fetchone()
    assert row[0] == 150.0
    assert row[1] == "infra_failed"

    # Second failed resume adds 30s on top — must read the latest
    # persisted value (150), not the original 100.
    cum2 = _accumulate_and_persist_failure(
        run_id="r-1", new_runtime_s=30.0,
        exit_reason="error: APIConnectionError",
        failure_reason="connection reset", mock_llm=False,
    )
    assert cum2 == 180.0
    with transaction() as con:
        row = con.execute(
            "SELECT runtime_s FROM runs WHERE run_id='r-1'"
        ).fetchone()
    assert row[0] == 180.0


def test_run_resume_batch_skips_remaining_when_cost_cap_trips(
    tmp_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--cost-cap-usd` must short-circuit subsequent resumes once tripped.
    Without this, resume invocations are unbounded by spend."""
    import asyncio
    import types

    from qed_swe_bench.runner import resume as resume_mod

    # Three resumable rows; each "completes" at $5 cost when resume_one_run
    # is invoked. With cap = $7, the first two run ($10 total trips it),
    # the third must short-circuit without invoking resume_one_run.
    for i in (1, 2, 3):
        run_dir = tmp_path / f"r-{i}"
        run_dir.mkdir()
        (run_dir / "job.json").write_text(
            json.dumps({"run_id": f"r-{i}", "model": "m"})
        )
        insert_queued(
            run_id=f"r-{i}", benchmark_id="b1", model="m",
            env=EnvSpec(id="e", image="i"),
            image_digest="sha256:" + "0" * 64, seed=i, run_dir=run_dir,
            nudges_used=False,
        )
        update_finished(
            run_id=f"r-{i}", status="model_failed", capabilities={}, score=0.0,
            usage_totals=NormalizedUsage(), cost_usd=0.0, cost_source="mock",
            runtime_s=1.0, turns_used=1,
            exit_reason="error: APIConnectionError",
            llm_route="native", api_base=None,
            failure_reason=None,
        )

    invoked: list[str] = []

    async def fake_resume_one_run(run_dir, **_kw):
        rid = json.loads((run_dir / "job.json").read_text())["run_id"]
        invoked.append(rid)
        return resume_mod.ResumeOutcome(
            run_id=rid, status="succeeded", exit_reason=None,
            runtime_s=1.0, turns_total=10, cost_usd=5.0,
        )

    monkeypatch.setattr(resume_mod, "resume_one_run", fake_resume_one_run)

    bench = types.SimpleNamespace(
        benchmark_id="b1",
        models=[types.SimpleNamespace(id="m", params={})],
        envs=[types.SimpleNamespace(id="e")],
        seeds=[1, 2, 3],
        max_parallel=1,
        cost_cap_usd=7.0,
        nudges=frozenset(),
        budgets=types.SimpleNamespace(
            max_tokens=16384, turn_budget=10,
            token_budget=None, context_budget=None,
        ),
        episode_timeout_s=60,
    )

    histogram = asyncio.run(resume_mod.run_resume_batch(bench))

    # First two cells run ($10 > $7 cap); third skipped.
    assert invoked == ["r-1", "r-2"]
    assert histogram.get("cost_cap_skipped") == 1
    assert histogram.get("succeeded") == 2


# ---------------- prepare_resume_state ----------------


def _write_run_dir(
    tmp_path: Path,
    *,
    transcript_lines: list[dict],
    grade_lines: list[dict] | None = None,
    image_digest: str = "sha256:abc",
) -> Path:
    (tmp_path / "job.json").write_text(json.dumps({
        "run_id": "rid-1",
        "model": "openai/gpt-5",
        "env_id": "v8-e00",
        "seed": 1,
        "image_digest": image_digest,
    }))
    (tmp_path / "transcript.jsonl").write_text(
        "\n".join(json.dumps(e) for e in transcript_lines) + "\n"
    )
    if grade_lines:
        (tmp_path / "grade_calls.jsonl").write_text(
            "\n".join(json.dumps(e) for e in grade_lines) + "\n"
        )
    return tmp_path


def test_prepare_resume_state_rebuilds_messages_and_caps(tmp_path: Path) -> None:
    """A 2-turn transcript: human → ai (tool_calls) → tool → ai (text).
    Plus one grade result. Prepared state should reflect both."""
    transcript = [
        {"role": "system", "content": "sys"},
        {"role": "human", "content": "find the bug"},
        {
            "role": "ai", "content": "",
            "stop_reason": "tool_calls",
            "served_model": "openai/gpt-5",
            "tool_calls": [{"id": "tc1", "name": "exec", "args": {"cmd": "ls"}}],
            "usage": {"input_tokens": 100, "output_tokens": 20, "reasoning_tokens": 5},
        },
        {"role": "tool", "tool_call_id": "tc1", "name": "exec", "content": "a\nb\n"},
        {
            "role": "ai", "content": "found it",
            "stop_reason": "stop",
            "served_model": "openai/gpt-5",
            "usage": {"input_tokens": 130, "output_tokens": 30},
        },
    ]
    grades = [
        {"path": "/p.js", "result": {"capabilities": {"cov_func": True, "diff": False}}},
    ]
    run_dir = _write_run_dir(tmp_path, transcript_lines=transcript, grade_lines=grades)

    state = prepare_resume_state(
        run_dir, route="litellm", model_id="openai/gpt-5",
    )
    assert state.image_digest == "sha256:abc"
    assert state.prior_turns == 2
    assert state.prior_input_tokens == 230
    assert state.prior_output_tokens == 50
    assert state.prior_reasoning_tokens == 5
    # Weighted tokens = sum of per-turn tick_ai_turn formula:
    #   turn 1: base_in=100 + cc=0 + cr*0.1=0 + 5*20 = 200
    #   turn 2: base_in=130 + 0 + 0 + 5*30           = 280
    #   total                                        = 480
    assert state.prior_weighted_tokens == 480
    assert state.served_model == "openai/gpt-5"
    # cumulative-OR caps
    assert state.best_caps == {"cov_func": True, "diff": False}
    # message list shape: 1 user (initial) + 2 assistant + 1 tool result
    msgs = state.messages.get()
    assert msgs[0]["role"] == "user" and msgs[0]["content"] == "find the bug"
    assert msgs[1]["role"] == "assistant"
    assert msgs[2]["role"] == "tool"
    assert msgs[3]["role"] == "assistant"


def test_prepare_resume_state_skips_marker_role(tmp_path: Path) -> None:
    """Role=`marker` entries (e.g. `[resume]` checkpoints) must NOT be
    replayed into the LLM message buffer — otherwise resume markers from
    earlier resume cycles bias the model toward summary/wrap-up behavior."""
    transcript = [
        {"role": "human", "content": "find the bug"},
        {
            "role": "ai", "content": "",
            "stop_reason": "tool_calls",
            "served_model": "openai/gpt-5",
            "tool_calls": [{"id": "tc1", "name": "exec", "args": {"cmd": "ls"}}],
            "usage": {"input_tokens": 100, "output_tokens": 20},
        },
        {"role": "tool", "tool_call_id": "tc1", "name": "exec", "content": "a\n"},
        {"role": "marker", "content": "[resume] continuing episode at turn 2"},
        {
            "role": "ai", "content": "ok",
            "stop_reason": "stop",
            "served_model": "openai/gpt-5",
            "usage": {"input_tokens": 130, "output_tokens": 5},
        },
    ]
    run_dir = _write_run_dir(tmp_path, transcript_lines=transcript)
    state = prepare_resume_state(run_dir, route="litellm", model_id="openai/gpt-5")
    msgs = state.messages.get()
    # Expected shape: user (initial) + assistant + tool + assistant. No
    # marker leaks in as a user/system turn.
    assert len(msgs) == 4
    assert msgs[0]["role"] == "user" and msgs[0]["content"] == "find the bug"
    assert msgs[1]["role"] == "assistant"
    assert msgs[2]["role"] == "tool"
    assert msgs[3]["role"] == "assistant"
    for m in msgs:
        content = m.get("content")
        if isinstance(content, str):
            assert "[resume]" not in content


def test_prepare_resume_state_skips_legacy_resume_human_marker(tmp_path: Path) -> None:
    """Legacy compat: pre-marker-role transcripts wrote `[resume] ...` as
    role=human. _rebuild_messages must skip them too, otherwise old runs
    keep re-feeding the marker on every subsequent resume."""
    transcript = [
        {"role": "human", "content": "find the bug"},
        {
            "role": "ai", "content": "",
            "stop_reason": "tool_calls",
            "tool_calls": [{"id": "tc1", "name": "exec", "args": {}}],
            "usage": {"input_tokens": 100, "output_tokens": 20},
        },
        {"role": "tool", "tool_call_id": "tc1", "name": "exec", "content": "a"},
        {"role": "human", "content": "[resume] continuing episode at turn 2"},
        {
            "role": "ai", "content": "ok",
            "stop_reason": "stop",
            "usage": {"input_tokens": 130, "output_tokens": 5},
        },
    ]
    run_dir = _write_run_dir(tmp_path, transcript_lines=transcript)
    state = prepare_resume_state(run_dir, route="litellm", model_id="openai/gpt-5")
    msgs = state.messages.get()
    assert len(msgs) == 4
    for m in msgs:
        content = m.get("content")
        if isinstance(content, str):
            assert "[resume]" not in content


def test_prepare_resume_state_falls_back_to_empty_when_transcript_minimal(
    tmp_path: Path,
) -> None:
    """A run that died before any turn produces an empty-but-valid state."""
    run_dir = _write_run_dir(tmp_path, transcript_lines=[
        {"role": "human", "content": "init"},
    ])
    state = prepare_resume_state(
        run_dir, route="litellm", model_id="openai/gpt-5",
    )
    assert state.prior_turns == 0
    assert state.best_caps == {}
    assert state.served_model is None


def test_prepare_resume_state_raises_when_image_digest_missing(tmp_path: Path) -> None:
    (tmp_path / "job.json").write_text(json.dumps({
        "run_id": "rid", "model": "x", "env_id": "y", "seed": 1,
    }))
    (tmp_path / "transcript.jsonl").write_text("")
    with pytest.raises(ValueError, match="image_digest"):
        prepare_resume_state(tmp_path, route="litellm", model_id="x")


def test_prepare_resume_state_weighted_tokens_with_cache(
    tmp_path: Path,
) -> None:
    """Weighted-token recovery exercises the full Budget.tick_ai_turn formula
    including cache_read (0.1× weight) and cache_creation (1× weight)."""
    transcript = [
        {"role": "human", "content": "go"},
        {
            "role": "ai", "content": "",
            "stop_reason": "tool_calls",
            "served_model": "anthropic/claude-haiku-4-5",
            "tool_calls": [{"id": "tc1", "name": "exec", "args": {}}],
            "usage": {
                "input_tokens": 1000,
                "output_tokens": 50,
                "cache_read": 600,
                "cache_creation": 200,
            },
        },
        {"role": "tool", "tool_call_id": "tc1", "name": "exec", "content": "ok"},
    ]
    state = prepare_resume_state(
        _write_run_dir(tmp_path, transcript_lines=transcript),
        route="anthropic_native", model_id="anthropic/claude-haiku-4-5",
    )
    # Hand-derive weighted using Budget.tick_ai_turn:
    #   base_in = max(0, 1000 - 600 - 200) = 200
    #   200 (base_in) + 200 (cc) + int(600*0.1)=60 + 5*50=250
    #   = 200 + 200 + 60 + 250 = 710
    assert state.prior_weighted_tokens == 710
    # Per-axis totals are sums of raw counts, untouched by the formula.
    assert state.prior_input_tokens == 1000
    assert state.prior_output_tokens == 50
    assert state.prior_cache_read_tokens == 600
    assert state.prior_cache_creation_tokens == 200


def test_prepare_resume_state_weighted_tokens_round_trip(
    tmp_path: Path,
) -> None:
    """Strongest correctness check: drive `Budget.tick_ai_turn` with the
    same per-turn usage values that get serialized into transcript.jsonl,
    and assert the live counter matches what `prepare_resume_state`
    reconstructs. Catches any drift between the two formulas."""
    from qed_swe_bench.runner.llm.base import NormalizedUsage

    usages = [
        NormalizedUsage(input_tokens=100, output_tokens=20),
        NormalizedUsage(input_tokens=1000, output_tokens=50,
                        cache_read_tokens=600, cache_creation_tokens=200),
        NormalizedUsage(input_tokens=200, output_tokens=10,
                        cache_read_tokens=150),
    ]
    live = Budget(turn_budget=None, token_budget=None, context_budget=None)
    for u in usages:
        live.tick_ai_turn(u)

    transcript = [{"role": "human", "content": "go"}]
    for u in usages:
        transcript.append({
            "role": "ai", "content": "", "stop_reason": "stop",
            "served_model": "x",
            "usage": {
                "input_tokens": u.input_tokens,
                "output_tokens": u.output_tokens,
                "cache_read": u.cache_read_tokens,
                "cache_creation": u.cache_creation_tokens,
            },
        })
    state = prepare_resume_state(
        _write_run_dir(tmp_path, transcript_lines=transcript),
        route="litellm", model_id="x",
    )
    assert state.prior_weighted_tokens == live.tokens_used
    assert state.prior_peak_per_turn_context == live.peak_per_turn_context


def test_prepare_resume_state_drops_dangling_ai_with_unmatched_tool_calls(
    tmp_path: Path,
) -> None:
    """Transcript ends with an ai turn whose tool_calls have no matching
    `tool` results (episode died mid-tool-call). The dangling turn must be
    dropped from both the message list and the prior-turn count, so the
    rebuilt history is well-formed for the next provider call."""
    transcript = [
        {"role": "human", "content": "go"},
        {
            "role": "ai", "content": "",
            "stop_reason": "tool_calls",
            "served_model": "x",
            "tool_calls": [{"id": "tc1", "name": "exec", "args": {"cmd": "ls"}}],
            "usage": {"input_tokens": 100, "output_tokens": 10},
        },
        {"role": "tool", "tool_call_id": "tc1", "name": "exec", "content": "ok"},
        {
            "role": "ai", "content": "",
            "stop_reason": "tool_calls",
            "served_model": "x",
            # dangling: no matching role=tool entry follows
            "tool_calls": [{"id": "tc2", "name": "grade", "args": {"path": "/p"}}],
            "usage": {"input_tokens": 200, "output_tokens": 5},
        },
    ]
    state = prepare_resume_state(
        _write_run_dir(tmp_path, transcript_lines=transcript),
        route="litellm", model_id="x",
    )
    assert state.prior_turns == 1
    assert state.prior_input_tokens == 100
    assert state.prior_output_tokens == 10
    msgs = state.messages.get()
    # 1 user (initial) + 1 assistant + 1 tool result; the dangling assistant
    # has been dropped.
    assert [m["role"] for m in msgs] == ["user", "assistant", "tool"]


def test_prepare_resume_state_persists_dangling_drop_to_disk(
    tmp_path: Path,
) -> None:
    """When a dangling AI turn is dropped, transcript.jsonl is rewritten so
    subsequent resume passes / audit / replay see a consistent log. The
    original is preserved at a sibling .pre-resume-trim-* backup."""
    transcript = [
        {"role": "human", "content": "go"},
        {
            "role": "ai", "content": "",
            "stop_reason": "tool_calls",
            "served_model": "x",
            "tool_calls": [{"id": "tc1", "name": "exec", "args": {"cmd": "ls"}}],
            "usage": {"input_tokens": 100, "output_tokens": 10},
        },
        {"role": "tool", "tool_call_id": "tc1", "name": "exec", "content": "ok"},
        {
            "role": "ai", "content": "",
            "stop_reason": "tool_calls",
            "served_model": "x",
            "tool_calls": [{"id": "tc2", "name": "grade", "args": {"path": "/p"}}],
            "usage": {"input_tokens": 200, "output_tokens": 5},
        },
    ]
    run_dir = _write_run_dir(tmp_path, transcript_lines=transcript)
    pre_lines = (run_dir / "transcript.jsonl").read_text().splitlines()
    assert len(pre_lines) == 4

    prepare_resume_state(run_dir, route="litellm", model_id="x")

    post_lines = (run_dir / "transcript.jsonl").read_text().splitlines()
    assert len(post_lines) == 3
    assert json.loads(post_lines[-1])["role"] == "tool"
    backups = list(run_dir.glob("transcript.jsonl.pre-resume-trim-*"))
    assert len(backups) == 1
    assert len(backups[0].read_text().splitlines()) == 4


def test_prepare_resume_state_no_disk_write_when_nothing_to_drop(
    tmp_path: Path,
) -> None:
    """A well-formed transcript must not trigger a backup or rewrite."""
    transcript = [
        {"role": "human", "content": "go"},
        {
            "role": "ai", "content": "done",
            "stop_reason": "stop",
            "served_model": "x",
            "usage": {"input_tokens": 50, "output_tokens": 5},
        },
    ]
    run_dir = _write_run_dir(tmp_path, transcript_lines=transcript)
    pre_bytes = (run_dir / "transcript.jsonl").read_bytes()

    prepare_resume_state(run_dir, route="litellm", model_id="x")

    assert (run_dir / "transcript.jsonl").read_bytes() == pre_bytes
    assert list(run_dir.glob("transcript.jsonl.pre-resume-trim-*")) == []


def test_prepare_resume_state_keeps_ai_with_all_tool_calls_resolved(
    tmp_path: Path,
) -> None:
    """A well-formed transcript ending with an ai turn whose tool_calls all
    have matching tool entries must NOT be touched by the dangling-drop."""
    transcript = [
        {"role": "human", "content": "go"},
        {
            "role": "ai", "content": "",
            "stop_reason": "tool_calls",
            "served_model": "x",
            "tool_calls": [
                {"id": "tc1", "name": "exec", "args": {"cmd": "a"}},
                {"id": "tc2", "name": "exec", "args": {"cmd": "b"}},
            ],
            "usage": {"input_tokens": 50, "output_tokens": 5},
        },
        {"role": "tool", "tool_call_id": "tc1", "name": "exec", "content": "x"},
        {"role": "tool", "tool_call_id": "tc2", "name": "exec", "content": "y"},
    ]
    state = prepare_resume_state(
        _write_run_dir(tmp_path, transcript_lines=transcript),
        route="litellm", model_id="x",
    )
    assert state.prior_turns == 1
    msgs = state.messages.get()
    assert [m["role"] for m in msgs] == ["user", "assistant", "tool", "tool"]


def test_prepare_resume_state_late_grade_overrides_earlier_false(
    tmp_path: Path,
) -> None:
    """Cumulative-OR: an earlier grade's `cap: False` should be lifted
    when a later grade returns `cap: True`."""
    transcript = [
        {"role": "human", "content": "go"},
        {"role": "ai", "content": "x", "stop_reason": "stop"},
    ]
    grades = [
        {"result": {"capabilities": {"diff": False}}},
        {"result": {"capabilities": {"diff": True}}},
    ]
    state = prepare_resume_state(
        _write_run_dir(tmp_path, transcript_lines=transcript, grade_lines=grades),
        route="litellm", model_id="openai/gpt-5",
    )
    assert state.best_caps["diff"] is True


# ---------------- prime_budget ----------------


def test_prime_budget_advances_counters_and_preserves_limits() -> None:
    state = ResumeState(
        messages=None,  # type: ignore[arg-type]  # not consumed by prime
        best_caps={},
        served_model=None,
        prior_turns=42,
        prior_input_tokens=1234,
        prior_output_tokens=567,
        prior_cache_read_tokens=89,
        prior_cache_creation_tokens=12,
        prior_reasoning_tokens=3,
        prior_peak_per_turn_context=0,
        prior_weighted_tokens=4321,
        prior_turns_since_grade=7,
        image_digest="sha256:x",
    )
    budget = Budget(turn_budget=300, token_budget=None, context_budget=None)
    prime_budget(budget, state)
    assert budget.turn == 42
    assert budget.total_input_tokens == 1234
    assert budget.total_output_tokens == 567
    assert budget.total_cache_read_tokens == 89
    assert budget.total_cache_creation_tokens == 12
    assert budget.tokens_used == 4321
    # Nudge state carried over: stuck counter restored, wrapup not yet due
    # (42 < 0.75 * 300 = 225).
    assert budget.turns_since_grade == 7
    assert budget.sent_wrapup is False
    # Limits untouched
    assert budget.turn_budget == 300
    # Resume must respect remaining-turn semantics: 42 turns done, 300
    # is the cap, so the loop has 258 turns left.
    assert not budget.exceeded()


def test_prime_budget_at_limit_marks_exceeded() -> None:
    """A run that died exactly at turn_budget shouldn't get more turns
    on resume."""
    state = ResumeState(
        messages=None,  # type: ignore[arg-type]
        best_caps={},
        served_model=None,
        prior_turns=300,
        prior_input_tokens=0,
        prior_output_tokens=0,
        prior_cache_read_tokens=0,
        prior_cache_creation_tokens=0,
        prior_reasoning_tokens=0,
        prior_peak_per_turn_context=0,
        prior_weighted_tokens=0,
        prior_turns_since_grade=0,
        image_digest="sha256:x",
    )
    budget = Budget(turn_budget=300, token_budget=None, context_budget=None)
    prime_budget(budget, state)
    assert budget.exceeded()
    # Resumed well past the 75% wrapup threshold -> latched, won't re-fire.
    assert budget.sent_wrapup is True


def test_turns_since_last_grade_counts_after_last_grade() -> None:
    entries = [
        {"role": "human", "content": "go"},
        {"role": "ai", "tool_calls": [{"name": "setup"}]},
        {"role": "ai", "tool_calls": [{"name": "grade"}]},  # reset to 0
        {"role": "tool", "content": "result"},
        {"role": "ai", "tool_calls": [{"name": "exec"}]},   # +1
        {"role": "ai", "content": "no tool calls"},          # +1
    ]
    assert _turns_since_last_grade(entries) == 2


def test_turns_since_last_grade_no_grade_counts_all_ai_turns() -> None:
    entries = [
        {"role": "ai", "tool_calls": [{"name": "exec"}]},
        {"role": "ai", "tool_calls": [{"name": "exec"}]},
    ]
    assert _turns_since_last_grade(entries) == 2


def test_turns_since_last_grade_simulates_stuck_nudge_reset() -> None:
    # A no-grade streak longer than the stuck-nudge interval: the live counter
    # resets when the stuck nudge fires, so the reconstruction wraps the same way.
    entries = [{"role": "ai", "content": "x"} for _ in range(GRADE_NUDGE_INTERVAL + 25)]
    assert _turns_since_last_grade(entries) == 25


# ---------------- claim + heartbeat lifecycle ------------------


@pytest.mark.asyncio
async def test_resume_one_run_skips_unclaimable_row(tmp_path: Path) -> None:
    """If claim_for_resume returns False (e.g., another sweep already
    claimed the row, or it's already terminal), resume_one_run returns
    early with status=skipped without touching anything else."""
    from unittest.mock import patch

    from qed_swe_bench.runner.resume import resume_one_run

    # Minimal job.json so the function gets past the file-existence check.
    (tmp_path / "job.json").write_text(json.dumps({
        "run_id": "rid-skip",
        "model": "openai/gpt-5",
        "env_id": "v8-e00",
        "seed": 1,
        "image_digest": "sha256:abc",
    }))

    class _RecordingTracker:
        def __init__(self):
            self.adds = []
            self.removes = []
        async def add(self, run_id):
            self.adds.append(run_id)
        async def remove(self, run_id):
            self.removes.append(run_id)

    tracker = _RecordingTracker()

    # claim_for_resume returns False → skipped path
    from qed_swe_bench.runner.orchestrator_config import Budgets
    with patch("qed_swe_bench.runner.runs_db.claim_for_resume", return_value=False):
        outcome = await resume_one_run(
            tmp_path,
            budgets=Budgets(),
            episode_timeout_s=1800,
            mock_llm=True,
            heartbeat=tracker,
        )
    assert outcome.status == "skipped"
    assert outcome.exit_reason == "already_claimed_or_terminal"
    # Tracker untouched: we never added the run because the claim failed.
    assert tracker.adds == []
    assert tracker.removes == []


# ---------------- run_episode resume_state integration ----------------


@pytest.mark.asyncio
async def test_run_episode_resumes_from_state_skips_init(tmp_path: Path) -> None:
    """When `resume_state` is provided, run_episode should NOT write a
    new init human prompt; it picks up from the rehydrated messages."""
    from unittest.mock import AsyncMock

    from qed_swe_bench.runner.llm.mock import MockClient
    from qed_swe_bench.runner.loop import run_episode
    from qed_swe_bench.runner.mcp_client import (
        McpDockerSession, ToolDef, ToolResult,
    )
    from qed_swe_bench.runner.transcript import TranscriptWriter

    tools = [
        ToolDef(name="setup", description="", input_schema={}),
        ToolDef(name="list_directory", description="", input_schema={}),
        ToolDef(name="write_file", description="", input_schema={}),
    ]

    async def fake_list_tools() -> list[ToolDef]:
        return tools

    async def fake_call_tool(name: str, _args: dict | None = None) -> ToolResult:
        return ToolResult(is_error=False, text=f"{name} ok", structured=None)

    sess = McpDockerSession(AsyncMock())
    sess.list_tools = fake_list_tools  # type: ignore[method-assign]
    sess.call_tool = fake_call_tool  # type: ignore[method-assign]

    # Build a resume state with one prior turn, prior tokens, and one
    # earned cap. The fresh tw points at tmp_path; run_episode runs
    # MockClient's small canned sequence on top.
    from qed_swe_bench.runner.llm.messages import build_messages_for
    msgs = build_messages_for("litellm", model_id="mock/test")
    msgs.initial("prior init")  # what would have been the first human msg
    state = ResumeState(
        messages=msgs,
        best_caps={"cov_func": True},
        served_model="mock/test",
        prior_turns=10,
        prior_input_tokens=500,
        prior_output_tokens=200,
        prior_cache_read_tokens=0,
        prior_cache_creation_tokens=0,
        prior_reasoning_tokens=0,
        prior_peak_per_turn_context=0,
        prior_weighted_tokens=1500,
        prior_turns_since_grade=0,
        image_digest="sha256:irrelevant",
    )

    budget = Budget(turn_budget=20, token_budget=None, context_budget=None)
    with TranscriptWriter(tmp_path) as tw:
        result = await run_episode(
            client=MockClient(model="mock/test"),
            mcp_session=sess,
            transcript=tw,
            budget=budget,
            init_prompt="should-not-be-written",
            seed=0,
            resume_state=state,
        )

    # Prior caps preserved.
    assert result.capabilities.get("cov_func") is True
    # Token counters include the prior totals.
    assert result.tokens_in >= 500
    # Turn count starts at prior_turns (10) and the loop took >=1 more.
    assert result.turns_used > 10
    # The init prompt "should-not-be-written" must not have been added
    # to the transcript as a human entry.
    transcript = (tmp_path / "transcript.jsonl").read_text()
    assert "should-not-be-written" not in transcript
