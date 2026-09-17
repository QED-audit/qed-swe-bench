"""Verify run_episode's LLM-call dispatch doesn't starve the event loop
when multiple episodes are run concurrently.

The agent loop calls `client.complete(...)` directly inside an async
coroutine. `client.complete()` is sync (LiteLLMClient and AnthropicNative
both wrap blocking httpx). When N episodes share one event loop via
`asyncio.gather`, every sync LLM call blocks the entire loop until it
returns — which serializes all N episodes through their LLM calls back-
to-back. Empirical signature on the live EC2 run: zero turn timestamps
landing within 100ms of each other across 14 concurrent episodes; mean
inter-turn gap = per-episode turn time ÷ N.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

from qed_swe_bench.runner.budget import Budget
from qed_swe_bench.runner.llm.mock import MockClient
from qed_swe_bench.runner.loop import run_episode
from qed_swe_bench.runner.mcp_client import McpDockerSession, ToolDef, ToolResult
from qed_swe_bench.runner.transcript import TranscriptWriter

# Per-LLM-call simulated duration. Picked so a parallel run finishes well
# under a second while a serial run takes a clearly-different wallclock
# that won't flake on scheduling jitter.
LLM_CALL_S = 0.3
N_EPISODES = 5
# Default MockClient sequence is 3 tool-using turns + 1 end_turn = 4 calls.
N_LLM_CALLS_PER_EPISODE = 4


class _SlowMockClient(MockClient):
    """MockClient with a synthetic per-call delay to mimic the blocking
    sync httpx I/O LiteLLMClient/AnthropicNative do in production."""

    def __init__(self, model: str, sleep_s: float = LLM_CALL_S) -> None:
        super().__init__(model=model)
        self._sleep_s = sleep_s

    def complete(self, **kwargs: Any) -> Any:
        time.sleep(self._sleep_s)
        return super().complete(**kwargs)


def _mcp_session_stub(tools: list[ToolDef], call_results: dict[str, ToolResult]):
    """Mirror of test_loop.py:_mcp_session_stub — async-fake MCP session."""

    async def fake_list_tools() -> list[ToolDef]:
        return tools

    async def fake_call_tool(name: str, _arguments: dict[str, Any] | None = None) -> ToolResult:
        return call_results.get(
            name, ToolResult(is_error=False, text=f"<no canned result for {name}>", structured=None)
        )

    session = AsyncMock()
    instance = McpDockerSession(session)
    instance.list_tools = fake_list_tools  # type: ignore[method-assign]
    instance.call_tool = fake_call_tool  # type: ignore[method-assign]
    return instance


def _stub_setup() -> tuple[list[ToolDef], dict[str, ToolResult]]:
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
    return tools, call_results


async def _run_one_episode(tmp_path: Path, ep_id: int) -> float:
    """Run one episode end-to-end with a slow MockClient. Returns elapsed."""
    tools, call_results = _stub_setup()
    sess = _mcp_session_stub(tools, call_results)
    client = _SlowMockClient(model=f"mock/test-{ep_id}")
    budget = Budget(turn_budget=10, token_budget=None, context_budget=None)
    run_dir = tmp_path / f"ep{ep_id}"
    run_dir.mkdir()
    t0 = time.monotonic()
    with TranscriptWriter(run_dir) as t:
        await run_episode(
            client=client,
            mcp_session=sess,
            transcript=t,
            budget=budget,
            init_prompt="GO",
            seed=0,
        )
    return time.monotonic() - t0


async def test_single_episode_baseline(tmp_path: Path) -> None:
    """Single-episode wallclock baseline. With ~4 sync LLM calls and
    LLM_CALL_S=0.3, one episode takes ~1.2s end-to-end. Sanity check that
    the slow mock + MCP pipeline both work."""
    elapsed = await _run_one_episode(tmp_path, 0)
    expected = N_LLM_CALLS_PER_EPISODE * LLM_CALL_S
    assert elapsed >= expected * 0.9, (
        f"single episode took {elapsed:.2f}s; expected ≥ {expected * 0.9:.2f}s "
        f"({N_LLM_CALLS_PER_EPISODE} calls × {LLM_CALL_S}s). Mock is broken or "
        f"sleeps are not firing."
    )
    # Cap at 2× expected to catch obvious regressions (e.g. unintended retries).
    assert elapsed < expected * 2.5, (
        f"single episode took {elapsed:.2f}s; expected < {expected * 2.5:.2f}s. "
        f"Loop may be doing extra work per turn."
    )


async def test_concurrent_episodes_should_parallelize(tmp_path: Path) -> None:
    """The load-bearing test. Run N episodes concurrently and assert
    wallclock is closer to ONE episode's duration than to N episodes'.

    With sync `client.complete()` directly in the async loop (current
    bug), N episodes serialize and wallclock ≈ N × per-episode time.
    With proper threading (asyncio.to_thread), the event loop stays
    responsive and concurrency is real: wallclock ≈ per-episode time.

    Threshold picks: parallel-correct passes if wallclock < ½ of serial
    expectation. That's a generous floor — at N=5, fully-parallel would
    be ~1.2s and fully-serial would be ~6s; 3s is the midpoint and an
    obvious failure for either regime to land in.
    """
    per_ep_expected = N_LLM_CALLS_PER_EPISODE * LLM_CALL_S
    serial_expected = N_EPISODES * per_ep_expected

    t0 = time.monotonic()
    await asyncio.gather(*[_run_one_episode(tmp_path, i) for i in range(N_EPISODES)])
    wallclock = time.monotonic() - t0

    # Pass condition: wallclock < halfway between parallel and serial.
    threshold = (per_ep_expected + serial_expected) / 2
    assert wallclock < threshold, (
        f"{N_EPISODES} concurrent episodes took {wallclock:.2f}s "
        f"(parallel target ≈ {per_ep_expected:.2f}s, serial floor ≈ "
        f"{serial_expected:.2f}s, threshold {threshold:.2f}s). "
        f"\n\nThis means the event loop is being blocked during sync "
        f"client.complete() calls. Look at runner/loop.py — the LLM "
        f"call must be wrapped in `await asyncio.to_thread(...)` so it "
        f"runs on a worker thread and doesn't starve other coroutines."
    )
