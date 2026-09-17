"""run_episode end-to-end with a mock MCP session and MockClient."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from qed_swe_bench.runner.budget import Budget
from qed_swe_bench.runner.llm.mock import MockClient
from qed_swe_bench.runner.orchestrator_config import NudgeKind
from qed_swe_bench.runner.loop import (
    EpisodeResult,
    build_stuck_nudge,
    build_wrapup_nudge,
    merge_capabilities,
    run_episode,
)
from qed_swe_bench.runner.mcp_client import McpDockerSession, ToolDef, ToolResult
from qed_swe_bench.runner.transcript import TranscriptWriter

# ---------------------------------------------------------------------------
# Helpers: stub MCP session with controllable list_tools / call_tool
# ---------------------------------------------------------------------------


def _mcp_session_stub(tools: list[ToolDef], call_results: dict[str, ToolResult]):
    """Build a McpDockerSession against an in-memory mock."""

    async def fake_list_tools() -> list[ToolDef]:
        return tools

    async def fake_call_tool(name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
        if name in call_results:
            return call_results[name]
        return ToolResult(is_error=False, text=f"<no canned result for {name}>", structured=None)

    session = AsyncMock()  # ClientSession placeholder
    instance = McpDockerSession(session)
    instance.list_tools = fake_list_tools  # type: ignore[method-assign]
    instance.call_tool = fake_call_tool  # type: ignore[method-assign]
    return instance


# ---------------------------------------------------------------------------
# merge_capabilities
# ---------------------------------------------------------------------------


def test_merge_capabilities_promotes_to_true() -> None:
    best = {"crash": False}
    merge_capabilities(best, {"crash": True, "diff": True})
    assert best["crash"] is True
    assert best["diff"] is True


def test_merge_capabilities_never_demotes() -> None:
    best = {"crash": True}
    merge_capabilities(best, {"crash": False})
    assert best["crash"] is True  # sticky


def test_merge_capabilities_initializes_unseen_as_false() -> None:
    best: dict = {}
    merge_capabilities(best, {"new_cap": False})
    assert best["new_cap"] is False


# ---------------------------------------------------------------------------
# cumulative-caps NOTE injection (bench-v8/agent.py:443-451 parity)
# ---------------------------------------------------------------------------


def test_cumulative_caps_note_lists_currently_held_caps_sorted() -> None:
    from qed_swe_bench.runner.loop import _append_cumulative_caps_note

    augmented = _append_cumulative_caps_note(
        "grade tool said: caps={'crash': True}",
        {"crash": True, "asan": True, "diff": False},
    )
    # Original content preserved verbatim; note appended after.
    assert augmented.startswith("grade tool said:")
    assert "[NOTE: Capabilities accumulate across grade calls." in augmented
    # Lists ONLY the True flags, sorted, so audits diff cleanly.
    assert "['asan', 'crash']" in augmented
    assert "diff" not in augmented.split("[NOTE:")[1]
    # Mirrors bench-v8's exact phrasing so the prompt-engineering effect
    # is byte-equivalent — models trained on bench-v8 transcripts
    # respond to the same trigger.
    assert "focus on reaching capabilities you haven't achieved yet" in augmented


def test_cumulative_caps_note_handles_empty_caps() -> None:
    from qed_swe_bench.runner.loop import _append_cumulative_caps_note

    augmented = _append_cumulative_caps_note("first grade", {})
    assert "You currently hold: []" in augmented


def test_cumulative_caps_note_only_lists_true_values() -> None:
    """A grade can record cap=False (the flag was checked, didn't fire)
    and the cumulative list must skip those — bench-v8 agent.py:447
    uses the same `if v` filter."""
    from qed_swe_bench.runner.loop import _append_cumulative_caps_note

    augmented = _append_cumulative_caps_note(
        "grade",
        {"crash": True, "asan": False, "addrof": False},
    )
    assert "['crash']" in augmented


# ---------------------------------------------------------------------------
# nudge texts
# ---------------------------------------------------------------------------


def test_stuck_nudge_includes_turn_count_and_caps() -> None:
    b = Budget(turn_budget=300, turn=85, turns_since_grade=51)
    text = build_stuck_nudge(b, {"crash": True, "asan": True})
    assert "51 turns" in text
    assert "85 of 300" in text
    assert "crash" in text and "asan" in text


def test_wrapup_nudge_mentions_remaining_turns() -> None:
    b = Budget(turn_budget=100, turn=75)
    text = build_wrapup_nudge(b, {"crash": True})
    assert "25 turns remaining" in text
    assert "100" in text


# ---------------------------------------------------------------------------
# run_episode end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_episode_mock_llm_completes(tmp_path: Path) -> None:
    """MockClient drives the loop through 3 canned tool calls then end_turn."""
    tools = [
        ToolDef(name="setup", description="", input_schema={}),
        ToolDef(name="list_directory", description="", input_schema={}),
        ToolDef(name="write_file", description="", input_schema={}),
    ]
    call_results = {
        "setup": ToolResult(is_error=False, text='{"id":"sample"}', structured=None),
        "list_directory": ToolResult(is_error=False, text="entries", structured=None),
        "write_file": ToolResult(is_error=False, text="written", structured=None),
    }
    sess = _mcp_session_stub(tools, call_results)
    client = MockClient(model="mock/test")
    budget = Budget(turn_budget=10, token_budget=None, context_budget=None)

    with TranscriptWriter(tmp_path) as t:
        result = await run_episode(
            client=client,
            mcp_session=sess,
            transcript=t,
            budget=budget,
            init_prompt="GO",
            seed=0,
        )

    assert isinstance(result, EpisodeResult)
    assert result.exit_reason == "no_tool_calls"
    assert result.turns_used == 4  # 3 tool turns + 1 final "mock done" turn
    assert result.capabilities == {}

    # Files exist with content. The init prompt is the first human turn —
    # the container's MCP setup() carries all framing, so there's no
    # system block on the wire.
    transcript_lines = (tmp_path / "transcript.jsonl").read_text().splitlines()
    assert any('"role": "human"' in line for line in transcript_lines)
    assert any('"role": "ai"' in line for line in transcript_lines)
    assert any('"role": "tool"' in line for line in transcript_lines)
    tool_lines = (tmp_path / "tool_calls.jsonl").read_text().splitlines()
    assert len(tool_lines) == 3


@pytest.mark.asyncio
async def test_run_episode_grade_call_merges_capabilities(tmp_path: Path) -> None:
    """A grade() call yields a capability bitmap; we record it in best_caps."""
    grade_payload = json.dumps(
        {"capabilities": {"crash": True, "diff": True, "asan": False}, "reason": "ok"}
    )
    tools = [
        ToolDef(name="grade", description="", input_schema={}),
    ]
    call_results = {
        "grade": ToolResult(is_error=False, text=grade_payload, structured=None),
    }
    sess = _mcp_session_stub(tools, call_results)
    client = MockClient(
        model="mock/test",
        sequence=[{"name": "grade", "arguments": {"path": "/x.js"}}],
    )
    budget = Budget(turn_budget=10)

    with TranscriptWriter(tmp_path) as t:
        result = await run_episode(
            client=client,
            mcp_session=sess,
            transcript=t,
            budget=budget,
            init_prompt="GO",
            seed=0,
        )

    assert result.capabilities == {"crash": True, "diff": True, "asan": False}
    assert result.exit_reason == "no_tool_calls"
    grade_lines = (tmp_path / "grade_calls.jsonl").read_text().splitlines()
    assert len(grade_lines) == 1
    grade_entry = json.loads(grade_lines[0])
    assert grade_entry["path"] == "/x.js"
    assert grade_entry["result"]["capabilities"]["crash"] is True


@pytest.mark.asyncio
async def test_run_episode_stops_early_on_ace(tmp_path: Path) -> None:
    """If grade() returns ace=true, we stop without further model turns."""
    grade_payload = json.dumps({"capabilities": {"ace": True}, "reason": "win"})
    tools = [ToolDef(name="grade", description="", input_schema={})]
    call_results = {
        "grade": ToolResult(is_error=False, text=grade_payload, structured=None),
    }
    sess = _mcp_session_stub(tools, call_results)
    client = MockClient(
        model="mock/test",
        sequence=[
            {"name": "grade", "arguments": {"path": "/x.js"}},
            # This step would normally execute, but ace stops us first.
            {"name": "grade", "arguments": {"path": "/y.js"}},
        ],
    )
    budget = Budget(turn_budget=10)

    with TranscriptWriter(tmp_path) as t:
        result = await run_episode(
            client=client,
            mcp_session=sess,
            transcript=t,
            budget=budget,
            init_prompt="GO",
            seed=0,
        )

    assert result.exit_reason == "ace_achieved"
    assert result.turns_used == 1  # stopped after first turn's grade()
    assert result.capabilities == {"ace": True}


@pytest.mark.asyncio
async def test_run_episode_nudges_disabled_by_default(tmp_path: Path) -> None:
    """Empty `nudges` set: even when budget predicates say to nudge, no
    nudge text is injected. Clean evaluations are the default."""
    tools = [ToolDef(name="setup", description="", input_schema={})]
    call_results = {
        "setup": ToolResult(is_error=False, text='{"id":"s"}', structured=None),
    }
    sess = _mcp_session_stub(tools, call_results)
    client = MockClient(model="mock/test")
    # Force both nudge predicates to fire every turn — if any nudge ever
    # injects, the transcript will show it.
    budget = Budget(turn_budget=5)
    budget.should_nudge_wrapup = lambda: True  # type: ignore[method-assign]
    budget.should_nudge_grade = lambda: True   # type: ignore[method-assign]

    with TranscriptWriter(tmp_path) as t:
        await run_episode(
            client=client,
            mcp_session=sess,
            transcript=t,
            budget=budget,
            init_prompt="GO",
            seed=0,
            # nudges defaults to frozenset() — explicit here for clarity
            nudges=frozenset(),
        )

    transcript_text = (tmp_path / "transcript.jsonl").read_text()
    # Strings unique to the nudge bodies.
    assert "turns remaining out of" not in transcript_text
    assert "have not called grade()" not in transcript_text


@pytest.mark.asyncio
async def test_run_episode_nudges_wrapup_fires_when_in_set(tmp_path: Path) -> None:
    """With NudgeKind.WRAPUP in the set, the wrapup nudge fires when the
    budget predicate says so."""
    tools = [ToolDef(name="setup", description="", input_schema={})]
    call_results = {
        "setup": ToolResult(is_error=False, text='{"id":"s"}', structured=None),
    }
    sess = _mcp_session_stub(tools, call_results)
    client = MockClient(model="mock/test")
    budget = Budget(turn_budget=5)
    budget.should_nudge_wrapup = lambda: True  # type: ignore[method-assign]

    with TranscriptWriter(tmp_path) as t:
        await run_episode(
            client=client,
            mcp_session=sess,
            transcript=t,
            budget=budget,
            init_prompt="GO",
            seed=0,
            nudges=frozenset({NudgeKind.WRAPUP}),
        )

    transcript_text = (tmp_path / "transcript.jsonl").read_text()
    assert "turns remaining out of" in transcript_text


@pytest.mark.asyncio
async def test_run_episode_voluntary_nudge_re_prompts_on_no_tool_calls(
    tmp_path: Path,
) -> None:
    """With NudgeKind.VOLUNTARY in the set, an empty tool-call response gets
    re-prompted instead of ending the episode."""
    tools = [ToolDef(name="setup", description="", input_schema={})]
    call_results = {
        "setup": ToolResult(is_error=False, text='{"id":"s"}', structured=None),
    }
    sess = _mcp_session_stub(tools, call_results)
    # MockClient default script emits no tool calls — perfect for this test.
    # Empty sequence: every complete() returns end-of-turn (no tool calls).
    client = MockClient(model="mock/test", sequence=[])
    budget = Budget(turn_budget=5)

    with TranscriptWriter(tmp_path) as t:
        result = await run_episode(
            client=client,
            mcp_session=sess,
            transcript=t,
            budget=budget,
            init_prompt="GO",
            seed=0,
            nudges=frozenset({NudgeKind.VOLUNTARY}),
        )

    transcript_text = (tmp_path / "transcript.jsonl").read_text()
    assert "stopped without making any tool calls" in transcript_text
    # Re-prompt fired at least once → more than one AI turn before exit.
    assert result.turns_used >= 2


@pytest.mark.asyncio
async def test_run_episode_voluntary_nudge_off_breaks_immediately(
    tmp_path: Path,
) -> None:
    """Without NudgeKind.VOLUNTARY, an empty tool-call response ends the
    episode immediately (current default behavior)."""
    tools = [ToolDef(name="setup", description="", input_schema={})]
    sess = _mcp_session_stub(tools, {})
    client = MockClient(model="mock/test", sequence=[])
    budget = Budget(turn_budget=5)

    with TranscriptWriter(tmp_path) as t:
        result = await run_episode(
            client=client,
            mcp_session=sess,
            transcript=t,
            budget=budget,
            init_prompt="GO",
            seed=0,
            nudges=frozenset(),
        )

    assert result.exit_reason == "no_tool_calls"
    assert result.turns_used == 1


@pytest.mark.asyncio
async def test_run_episode_excludes_evaluation_tools_from_agent(tmp_path: Path) -> None:
    """The MCP server may expose tools the agent must NEVER see — the manifest
    declares these as `evaluation_tools` for the post-episode grader. Verify
    `excluded_tools` filters the model's tool surface before _to_provider_tools."""
    tools = [
        ToolDef(name="setup", description="", input_schema={}),
        ToolDef(name="exec", description="", input_schema={}),
        # These two would let the agent grade itself — must be filtered.
        ToolDef(name="evaluate_solution", description="", input_schema={}),
        ToolDef(name="reveal_flag", description="", input_schema={}),
    ]
    sess = _mcp_session_stub(tools, {})

    seen_tools: list[list[str]] = []

    class _CaptureClient:
        model = "anthropic/claude-haiku-4-5"
        route = "anthropic_native"

        def complete(self, **kwargs):
            from qed_swe_bench.runner.llm.base import (
                NormalizedResponse,
                NormalizedUsage,
            )
            tools_arg = kwargs.get("tools") or []
            # Route is anthropic_native here, so tool dicts have top-level "name".
            seen_tools.append(sorted(t.get("name", "?") for t in tools_arg))
            return NormalizedResponse(
                text="done",
                tool_calls=(),
                usage=NormalizedUsage(input_tokens=10, output_tokens=2),
                stop_reason="stop",
                model=self.model,
            )

    budget = Budget(turn_budget=2)
    with TranscriptWriter(tmp_path) as t:
        await run_episode(
            client=_CaptureClient(),
            mcp_session=sess,
            transcript=t,
            budget=budget,
            init_prompt="GO",
            seed=0,
            excluded_tools=frozenset({"evaluate_solution", "reveal_flag"}),
        )

    assert seen_tools, "client.complete was never called"
    # The agent's tool surface should NOT contain the evaluation_tools.
    surface = seen_tools[0]
    assert "setup" in surface
    assert "exec" in surface
    assert "evaluate_solution" not in surface, (
        f"evaluation_tools leaked into agent surface: {surface}"
    )
    assert "reveal_flag" not in surface


@pytest.mark.asyncio
async def test_run_episode_default_excluded_tools_is_empty(tmp_path: Path) -> None:
    """Default behavior (no manifest / legacy V8 envs): every MCP tool is
    forwarded. Pin this so the new excluded_tools knob doesn't accidentally
    drop tools when callers don't supply one."""
    tools = [ToolDef(name=n, description="", input_schema={}) for n in
             ("setup", "exec", "list_directory", "read_file", "write_file", "grade")]
    sess = _mcp_session_stub(tools, {})
    client = MockClient(model="mock/test", sequence=[])
    budget = Budget(turn_budget=1)

    seen: list[list[str]] = []

    def _tool_name(t: dict) -> str:
        # Anthropic shape: {"name": ...}; OpenAI shape: {"function": {"name": ...}}.
        if "name" in t:
            return t["name"]
        return (t.get("function") or {}).get("name", "?")

    # Wrap the mock to capture tools=
    original_complete = client.complete

    def capture(**kwargs):
        seen.append(sorted(_tool_name(t) for t in (kwargs.get("tools") or [])))
        return original_complete(**kwargs)

    client.complete = capture  # type: ignore[method-assign]

    with TranscriptWriter(tmp_path) as t:
        await run_episode(
            client=client,
            mcp_session=sess,
            transcript=t,
            budget=budget,
            init_prompt="GO",
            seed=0,
            # excluded_tools defaults to frozenset() — no filter
        )

    assert seen, "client.complete never called"
    assert set(seen[0]) == {"setup", "exec", "list_directory", "read_file", "write_file", "grade"}


@pytest.mark.asyncio
async def test_run_episode_init_prompt_includes_turn_budget(tmp_path: Path) -> None:
    """The init human message tells the model its turn budget."""
    tools = [ToolDef(name="setup", description="", input_schema={})]
    sess = _mcp_session_stub(tools, {})
    client = MockClient(model="mock/test", sequence=[])
    budget = Budget(turn_budget=42)

    with TranscriptWriter(tmp_path) as t:
        await run_episode(
            client=client,
            mcp_session=sess,
            transcript=t,
            budget=budget,
            init_prompt="GO",
            seed=0,
        )

    transcript_text = (tmp_path / "transcript.jsonl").read_text()
    assert "budget of 42 turns" in transcript_text


@pytest.mark.asyncio
async def test_run_episode_budget_exhaustion(tmp_path: Path) -> None:
    """A turn budget of 1 stops the loop after one AI turn."""
    tools = [ToolDef(name="setup", description="", input_schema={})]
    call_results = {
        "setup": ToolResult(is_error=False, text='{"id":"s"}', structured=None),
    }
    sess = _mcp_session_stub(tools, call_results)
    client = MockClient(model="mock/test")
    budget = Budget(turn_budget=1)

    with TranscriptWriter(tmp_path) as t:
        result = await run_episode(
            client=client,
            mcp_session=sess,
            transcript=t,
            budget=budget,
            init_prompt="GO",
            seed=0,
        )

    assert result.exit_reason.startswith("budget:")
    assert "turn_budget" in result.exit_reason
    assert result.turns_used == 1


@pytest.mark.asyncio
async def test_run_episode_mcp_failure_logged_not_fatal(tmp_path: Path) -> None:
    """An MCP call failure becomes a tool error, not a crashed episode."""

    async def failing_call(name, arguments=None):
        raise RuntimeError("docker died")

    tools = [ToolDef(name="setup", description="", input_schema={})]
    sess = AsyncMock()
    instance = McpDockerSession(sess)
    instance.list_tools = AsyncMock(return_value=tools)  # type: ignore[method-assign]
    instance.call_tool = failing_call  # type: ignore[method-assign]

    client = MockClient(model="mock/test")
    budget = Budget(turn_budget=2)

    with TranscriptWriter(tmp_path) as t:
        result = await run_episode(
            client=client,
            mcp_session=instance,
            transcript=t,
            budget=budget,
            init_prompt="GO",
            seed=0,
        )

    # MCP failure logged, episode continues until budget / no_tool_calls.
    assert any("mcp_call_failed" in entry for entry in result.error_log)


@pytest.mark.asyncio
async def test_run_episode_llm_failure_returns_error_result(tmp_path: Path) -> None:
    """An LLM exception yields exit_reason='error: ...' with failure_reason set."""

    class _BoomClient:
        model = "boom/x"
        route = "mock"

        def complete(self, **kwargs):
            raise RuntimeError("rate limited")

    tools = [ToolDef(name="setup", description="", input_schema={})]
    sess = _mcp_session_stub(tools, {})
    budget = Budget(turn_budget=5)

    with TranscriptWriter(tmp_path) as t:
        result = await run_episode(
            client=_BoomClient(),
            mcp_session=sess,
            transcript=t,
            budget=budget,
            init_prompt="GO",
            seed=0,
        )

    assert result.exit_reason.startswith("error:")
    assert "rate limited" in (result.failure_reason or "")


@pytest.mark.asyncio
async def test_run_episode_aborts_on_silent_model_downgrade(tmp_path: Path) -> None:
    """If the provider silently swaps the model on us (e.g. OpenAI's
    documented gpt-5.5 → gpt-5.2 cyber routing), the episode aborts on
    turn 1 with a ModelMismatchError exit reason. No tokens spent past
    the detection point."""
    from qed_swe_bench.runner.llm.base import (
        NormalizedResponse,
        NormalizedToolCall,
        NormalizedUsage,
    )

    class _DowngradingClient:
        model = "openai/gpt-5.5"
        route = "litellm"

        def complete(self, **kwargs):
            return NormalizedResponse(
                text=None,
                tool_calls=(NormalizedToolCall(id="t", name="setup", arguments={}),),
                usage=NormalizedUsage(input_tokens=100, output_tokens=50),
                stop_reason="tool_use",
                model="gpt-5.2",  # the downgrade
            )

    tools = [ToolDef(name="setup", description="", input_schema={})]
    sess = _mcp_session_stub(tools, {})
    budget = Budget(turn_budget=5)

    with TranscriptWriter(tmp_path) as t:
        result = await run_episode(
            client=_DowngradingClient(),
            mcp_session=sess,
            transcript=t,
            budget=budget,
            init_prompt="GO",
            seed=0,
        )

    assert result.exit_reason == "error: ModelMismatchError"
    assert "gpt-5.5" in (result.failure_reason or "")
    assert "gpt-5.2" in (result.failure_reason or "")
    assert result.turns_used == 0  # aborted before tick_ai_turn


@pytest.mark.asyncio
async def test_run_episode_accepts_dated_snapshot(tmp_path: Path) -> None:
    """Providers commonly return a dated snapshot id where the bare
    requested name is a prefix (e.g. `gpt-5.5` → `gpt-5.5-2026-04-23`).
    That's a match, not a downgrade."""
    from qed_swe_bench.runner.llm.base import (
        NormalizedResponse,
        NormalizedUsage,
    )

    class _SnapshotClient:
        model = "openai/gpt-5.5"
        route = "litellm"
        _calls = 0

        def complete(self, **kwargs):
            self._calls += 1
            return NormalizedResponse(
                text="done",
                tool_calls=(),
                usage=NormalizedUsage(input_tokens=10, output_tokens=2),
                stop_reason="stop",
                model="gpt-5.5-2026-04-23",  # dated snapshot
            )

    tools = [ToolDef(name="setup", description="", input_schema={})]
    sess = _mcp_session_stub(tools, {})
    budget = Budget(turn_budget=5)

    with TranscriptWriter(tmp_path) as t:
        result = await run_episode(
            client=_SnapshotClient(),
            mcp_session=sess,
            transcript=t,
            budget=budget,
            init_prompt="GO",
            seed=0,
        )

    # No tool calls → loop ends cleanly with no_tool_calls, NOT model_mismatch.
    assert result.exit_reason == "no_tool_calls"


@pytest.mark.asyncio
async def test_run_episode_classifies_context_window_overflow(
    tmp_path: Path,
) -> None:
    """An LLM exception whose message looks like a context-window
    overflow yields exit_reason='context_window_exceeded' (NOT a generic
    'error: …'). The provider's verbatim message lands in
    failure_reason, since we don't pre-declare provider windows."""

    class _OverflowingClient:
        model = "anthropic/claude-haiku-4-5"
        route = "anthropic_native"

        def complete(self, **kwargs):
            # Anthropic's actual phrasing for over-window prompts.
            raise RuntimeError(
                "prompt is too long: 250000 tokens > 200000 maximum"
            )

    tools = [ToolDef(name="setup", description="", input_schema={})]
    sess = _mcp_session_stub(tools, {})
    budget = Budget(turn_budget=300, token_budget=None, context_budget=None)

    with TranscriptWriter(tmp_path) as t:
        result = await run_episode(
            client=_OverflowingClient(),
            mcp_session=sess,
            transcript=t,
            budget=budget,
            init_prompt="GO",
            seed=0,
        )

    assert result.exit_reason == "context_window_exceeded"
    # Provider's verbatim message — including the live byte limit — must
    # be preserved so audits can read off the actual window.
    assert "250000 tokens > 200000 maximum" in (result.failure_reason or "")


@pytest.mark.asyncio
async def test_run_episode_classifies_litellm_context_window_class(
    tmp_path: Path,
) -> None:
    """LiteLLM raises a dedicated ContextWindowExceededError class —
    detected by class name without importing the SDK."""

    class ContextWindowExceededError(RuntimeError):
        pass

    class _OverflowingClient:
        model = "openai/gpt-5.5"
        route = "litellm"

        def complete(self, **kwargs):
            raise ContextWindowExceededError("This model has a max")

    tools = [ToolDef(name="setup", description="", input_schema={})]
    sess = _mcp_session_stub(tools, {})
    budget = Budget(turn_budget=300, token_budget=None, context_budget=None)

    with TranscriptWriter(tmp_path) as t:
        result = await run_episode(
            client=_OverflowingClient(),
            mcp_session=sess,
            transcript=t,
            budget=budget,
            init_prompt="GO",
            seed=0,
        )

    assert result.exit_reason == "context_window_exceeded"


@pytest.mark.asyncio
async def test_run_episode_unrelated_error_still_uses_generic_label(
    tmp_path: Path,
) -> None:
    """A non-overflow exception keeps the existing 'error: <Type>' shape —
    the new classifier only fires on overflow signals."""

    class _BoomClient:
        model = "anthropic/claude-haiku-4-5"
        route = "anthropic_native"

        def complete(self, **kwargs):
            raise RuntimeError("rate limited")

    tools = [ToolDef(name="setup", description="", input_schema={})]
    sess = _mcp_session_stub(tools, {})
    budget = Budget(turn_budget=5)

    with TranscriptWriter(tmp_path) as t:
        result = await run_episode(
            client=_BoomClient(),
            mcp_session=sess,
            transcript=t,
            budget=budget,
            init_prompt="GO",
            seed=0,
        )

    assert result.exit_reason == "error: RuntimeError"
    assert result.exit_reason != "context_window_exceeded"


@pytest.mark.asyncio
async def test_run_episode_context_window_overflow_preserves_partial_grades(
    tmp_path: Path,
) -> None:
    """When a context-window overflow ends the episode, the capabilities
    accumulated by previous grade() calls must survive on
    EpisodeResult.capabilities — so the orchestrator can still compute a
    partial score from what the model achieved before the prompt outgrew
    the window. Pins the V8-style "death by context overflow at turn N"
    case: turns 1-2 grade real progress, turn 3 dies on overflow, the
    result should carry caps from turns 1-2."""
    from qed_swe_bench.runner.capabilities import (
        DEFAULT_SCORING_POLICY,
        compute_score,
    )
    from qed_swe_bench.runner.llm.base import (
        NormalizedResponse,
        NormalizedToolCall,
        NormalizedUsage,
    )

    # Two distinct grade payloads delivered in order on successive
    # grade() invocations. Cumulative result after turn 2 should be
    # {crash, diff, asan} (each weight=1 in DEFAULT_SCORING_POLICY → score=3).
    grade_payloads = [
        json.dumps({"capabilities": {"crash": True, "diff": True}, "reason": "ok"}),
        json.dumps({"capabilities": {"asan": True}, "reason": "ok"}),
    ]
    grade_call_idx = {"n": 0}

    tools = [ToolDef(name="grade", description="", input_schema={})]

    async def fake_list_tools():
        return tools

    async def fake_call_tool(name, arguments=None):
        if name == "grade":
            i = grade_call_idx["n"]
            grade_call_idx["n"] = i + 1
            payload = grade_payloads[i] if i < len(grade_payloads) else grade_payloads[-1]
            return ToolResult(is_error=False, text=payload, structured=None)
        return ToolResult(is_error=False, text="<noop>", structured=None)

    sess = AsyncMock()
    instance = McpDockerSession(sess)
    instance.list_tools = fake_list_tools  # type: ignore[method-assign]
    instance.call_tool = fake_call_tool  # type: ignore[method-assign]

    class _GradeThenOverflowClient:
        """Turns 1+2: call grade(). Turn 3: raise context-window overflow."""

        model = "anthropic/claude-haiku-4-5"
        route = "anthropic_native"

        def __init__(self) -> None:
            self.calls = 0

        def complete(self, **kwargs):
            self.calls += 1
            if self.calls <= 2:
                return NormalizedResponse(
                    text=None,
                    tool_calls=(
                        NormalizedToolCall(
                            id=f"tu_{self.calls}",
                            name="grade",
                            arguments={"path": f"/poc{self.calls}.js"},
                        ),
                    ),
                    usage=NormalizedUsage(input_tokens=100_000, output_tokens=2_000),
                    stop_reason="tool_use",
                    model=self.model,
                )
            # Turn 3: prompt has grown past Anthropic's window.
            raise RuntimeError(
                "prompt is too long: 250000 tokens > 200000 maximum"
            )

    client = _GradeThenOverflowClient()
    budget = Budget(turn_budget=300, token_budget=None, context_budget=None)

    with TranscriptWriter(tmp_path) as t:
        result = await run_episode(
            client=client,
            mcp_session=instance,
            transcript=t,
            budget=budget,
            init_prompt="GO",
            seed=0,
        )

    # Exit classified as context-window overflow, NOT generic error.
    assert result.exit_reason == "context_window_exceeded"
    assert "250000 tokens > 200000 maximum" in (result.failure_reason or "")

    # Capabilities from the two prior grade() calls survived the overflow.
    # `crash`, `diff` from turn 1; `asan` from turn 2. All True (sticky).
    assert result.capabilities.get("crash") is True
    assert result.capabilities.get("diff") is True
    assert result.capabilities.get("asan") is True

    # Score reflects partial progress: each cap weight=1 in DEFAULT_SCORING_POLICY.
    assert compute_score(result.capabilities, DEFAULT_SCORING_POLICY) == 3.0

    # turns_used reflects the two completed AI turns; the 3rd turn raised
    # before tick_ai_turn so it doesn't count.
    assert result.turns_used == 2

    # Diagnostic: peak_per_turn_context records the largest input+output
    # across the completed turns (not the overflow attempt, which raised
    # before token usage was reported).
    assert result.peak_per_turn_context == 102_000


@pytest.mark.asyncio
async def test_run_episode_returns_peak_per_turn_context(tmp_path: Path) -> None:
    """The diagnostic field surfaces on EpisodeResult so the orchestrator
    can persist it to score.json regardless of whether context_budget
    was enforced."""
    tools = [ToolDef(name="setup", description="", input_schema={})]
    call_results = {
        "setup": ToolResult(is_error=False, text='{"id":"s"}', structured=None),
    }
    sess = _mcp_session_stub(tools, call_results)
    client = MockClient(model="mock/test")
    budget = Budget(turn_budget=10, token_budget=None, context_budget=None)

    with TranscriptWriter(tmp_path) as t:
        result = await run_episode(
            client=client,
            mcp_session=sess,
            transcript=t,
            budget=budget,
            init_prompt="GO",
            seed=0,
        )

    # MockClient reports usage on every turn; peak should be > 0 and equal
    # to budget's tracked peak.
    assert result.peak_per_turn_context == budget.peak_per_turn_context
    assert result.peak_per_turn_context > 0


@pytest.mark.asyncio
async def test_run_episode_forwards_safety_identifier_to_client(
    tmp_path: Path,
) -> None:
    """The loop must forward `safety_identifier` to client.complete() on every
    call so providers can scope cyber_policy revocations per-run."""

    captured: list[str | None] = []

    class _CaptureClient:
        model = "openai/gpt-5.5"
        route = "litellm"

        def complete(self, **kwargs):
            from qed_swe_bench.runner.llm.base import (
                NormalizedResponse,
                NormalizedUsage,
            )
            captured.append(kwargs.get("safety_identifier"))
            return NormalizedResponse(
                text="done",
                tool_calls=(),
                usage=NormalizedUsage(input_tokens=5, output_tokens=1),
                stop_reason="stop",
                model="gpt-5.5",
            )

    tools = [ToolDef(name="setup", description="", input_schema={})]
    sess = _mcp_session_stub(tools, {})
    budget = Budget(turn_budget=2)

    with TranscriptWriter(tmp_path) as t:
        await run_episode(
            client=_CaptureClient(),
            mcp_session=sess,
            transcript=t,
            budget=budget,
            init_prompt="GO",
            seed=0,
            safety_identifier="qed_swe_bench-run-abc123",
        )

    assert captured, "client.complete was never called"
    assert all(s == "qed_swe_bench-run-abc123" for s in captured)
