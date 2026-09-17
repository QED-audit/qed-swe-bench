"""Feed recorded `(tool, args)` pairs back through an MCP session verbatim.

Used by `audit/reproduce.py` to re-grade PoCs on a fresh container, and
by `runner/resume.py` to rebuild filesystem state before continuing a
partial episode.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Container, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReplayedCall:
    """One replayed tool call, paired with its recorded counterpart."""

    tool: str
    args: dict[str, Any]
    is_error: bool
    text: str
    structured: dict[str, Any] | None
    duration_s: float
    # The original `result` string from tool_calls.jsonl, when present.
    recorded_text: str | None
    skipped: bool = False
    # Set when `session.call_tool` itself raised; distinct from
    # `is_error=True` which is a structured error returned by the server.
    error: str | None = None


def read_tool_calls_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load tool_calls.jsonl, skipping blank or malformed lines."""
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
            log.warning("skipping malformed line in %s", path)
            continue
    return out


async def replay_tool_calls(
    session: Any,
    tool_calls: Iterable[Mapping[str, Any]],
    *,
    skip: Container[str] = (),
    stop_after_index: int | None = None,
    on_progress: Callable[[int, int | None, str, float], None] | None = None,
) -> list[ReplayedCall]:
    """Replay each recorded tool call through `session.call_tool`.

    `session` is duck-typed against `McpDockerSession.call_tool` so tests
    can pass a stub. `skip` bypasses listed tool names (typically
    `{"grade"}` for state-reconstruction). `stop_after_index` (inclusive)
    truncates the sequence. Per-call exceptions are captured into
    `ReplayedCall.error` and the loop continues.

    `on_progress(index, total, tool_name, duration_s)` fires after every
    call (including skipped). `total` is None when `tool_calls` isn't a
    sized container.
    """
    seq = list(tool_calls) if not hasattr(tool_calls, "__len__") else tool_calls
    total = len(seq) if hasattr(seq, "__len__") else None
    results: list[ReplayedCall] = []
    for i, entry in enumerate(seq):
        if stop_after_index is not None and i > stop_after_index:
            break

        tool = entry.get("tool")
        args = entry.get("args") or {}
        if not isinstance(tool, str) or not isinstance(args, Mapping):
            log.warning("skipping malformed tool_call entry at index %d", i)
            continue
        recorded = entry.get("result")
        recorded_text = recorded if isinstance(recorded, str) else None

        # Emit before the call so the progress display reflects the
        # in-flight tool, not the previously-completed one.
        if on_progress is not None:
            on_progress(i, total, tool, 0.0)

        if tool in skip:
            results.append(
                ReplayedCall(
                    tool=tool, args=dict(args), is_error=False,
                    text="", structured=None, duration_s=0.0,
                    recorded_text=recorded_text, skipped=True,
                )
            )
            continue

        t0 = monotonic()
        try:
            tool_result = await session.call_tool(tool, dict(args))
        except Exception as exc:  # noqa: BLE001
            results.append(
                ReplayedCall(
                    tool=tool, args=dict(args), is_error=True,
                    text="", structured=None,
                    duration_s=monotonic() - t0,
                    recorded_text=recorded_text,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            log.warning("replay of %s failed at index %d: %s", tool, i, exc)
            continue

        results.append(
            ReplayedCall(
                tool=tool,
                args=dict(args),
                is_error=bool(tool_result.is_error),
                text=tool_result.text or "",
                structured=tool_result.structured,
                duration_s=monotonic() - t0,
                recorded_text=recorded_text,
            )
        )

    # Final flush so the bar lands at total once the loop exits.
    if on_progress is not None and total is not None:
        on_progress(total, total, "", 0.0)
    return results
