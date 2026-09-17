"""Stub LLM client for `qed_swe_bench benchmark --mock-llm` mode.

Replays a canned tool-call sequence so the runner can be exercised without
real API calls (no spend, deterministic, runs in CI). Used in place of a
real provider client by `runner/llm/factory.build_client(..., mock=True)`.

Default sequence is intentionally short so it works against simple envs
like `bugs/sample-stack-bof`. Tests can pass a custom sequence to drive
specific code paths in the loop.
"""

from __future__ import annotations

from typing import Any

from qed_swe_bench.runner.llm.base import (
    NormalizedResponse,
    NormalizedToolCall,
    NormalizedUsage,
)

# Each step is {"name": tool_name, "arguments": dict}.
DEFAULT_SEQUENCE: list[dict[str, Any]] = [
    {"name": "setup", "arguments": {}},
    {
        "name": "list_directory",
        "arguments": {"path": "/rlenv/workspace"},
    },
    {
        "name": "write_file",
        "arguments": {
            "path": "/rlenv/workspace/qed_swe_bench_smoke.txt",
            "contents": "qed_swe_bench mock-llm smoke",
        },
    },
]


class MockClient:
    """Canned-response client for `--mock-llm`."""

    route = "mock"

    def __init__(
        self,
        model: str,
        sequence: list[dict[str, Any]] | None = None,
        *,
        end_text: str = "mock done",
    ) -> None:
        self.model = model
        self.sequence: list[dict[str, Any]] = list(sequence) if sequence is not None else list(
            DEFAULT_SEQUENCE
        )
        self._index = 0
        self._end_text = end_text

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int,
        seed: int | None = None,
        safety_identifier: str | None = None,
    ) -> NormalizedResponse:
        del messages, tools, max_tokens, seed, safety_identifier  # ignored

        if self._index >= len(self.sequence):
            return NormalizedResponse(
                text=self._end_text,
                tool_calls=(),
                usage=NormalizedUsage(input_tokens=10, output_tokens=2),
                stop_reason="end_turn",
                model=self.model,
            )

        step = self.sequence[self._index]
        self._index += 1

        tc = NormalizedToolCall(
            id=f"toolu_mock_{self._index:04d}",
            name=step["name"],
            arguments=dict(step.get("arguments", {})),
        )
        return NormalizedResponse(
            text=None,
            tool_calls=(tc,),
            usage=NormalizedUsage(input_tokens=10, output_tokens=5),
            stop_reason="tool_use",
            model=self.model,
        )
