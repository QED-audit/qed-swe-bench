"""`audit --reproduce <run_id>` — re-fire each recorded grade() on a
fresh container and compare the resulting capability bitmap to what the
original grader recorded.

One MCP container per run (opened at the recorded `image_digest`); every
`tool_calls.jsonl` entry is replayed through `replay_tool_calls` so the
filesystem state the grader sees matches the live episode's. Each
`grade` call's structured result is compared via `ComparisonResult`.

Catches: PoCs that hardcode round-specific addresses (multi-round
shuffled-layout repro fails them), forged GRADER_RESULT_FD output
(we re-fire the real grader), and PoCs that depended on cumulative
state from earlier turns.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qed_swe_bench.runner.replay import read_tool_calls_jsonl, replay_tool_calls

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ComparisonResult:
    """Per-grade comparison of recorded vs reproduced capabilities."""

    grade_call_index: int
    poc_path: str
    recorded: dict[str, bool]
    reproduced: dict[str, bool]
    matches: bool                        # the True-set is identical
    only_recorded: tuple[str, ...] = ()  # caps in recorded but not reproduced
    only_reproduced: tuple[str, ...] = ()
    error: str | None = None              # set when the re-grade itself failed

    @classmethod
    def compare(
        cls,
        *,
        grade_call_index: int,
        poc_path: str,
        recorded: dict[str, bool],
        reproduced: dict[str, bool] | None,
        error: str | None = None,
    ) -> ComparisonResult:
        if error is not None or reproduced is None:
            return cls(
                grade_call_index=grade_call_index,
                poc_path=poc_path,
                recorded=dict(recorded),
                reproduced={},
                matches=False,
                error=error,
            )
        rec_true = {k for k, v in recorded.items() if v}
        rep_true = {k for k, v in reproduced.items() if v}
        return cls(
            grade_call_index=grade_call_index,
            poc_path=poc_path,
            recorded=dict(recorded),
            reproduced=dict(reproduced),
            matches=rec_true == rep_true,
            only_recorded=tuple(sorted(rec_true - rep_true)),
            only_reproduced=tuple(sorted(rep_true - rec_true)),
        )


@dataclass(frozen=True)
class ReproductionReport:
    """Per-run reproduction result."""

    run_id: str
    image_ref: str
    poc_count: int
    comparisons: tuple[ComparisonResult, ...]
    overall_match: bool
    error: str | None = None        # session-level error (e.g. couldn't start container)


def _read_image_ref(run_dir: Path) -> str | None:
    """Pull the image_digest (preferred) or image_ref from job.json."""
    job = run_dir / "job.json"
    if not job.is_file():
        return None
    try:
        data = json.loads(job.read_text())
    except json.JSONDecodeError:
        return None
    return data.get("image_digest") or data.get("image_ref")


def _grade_indices(tool_calls: list[dict[str, Any]]) -> list[int]:
    """Indices of `grade` calls within tool_calls (in order)."""
    return [i for i, e in enumerate(tool_calls) if e.get("tool") == "grade"]


def _recorded_capabilities(
    grade_calls: list[dict[str, Any]], index: int,
) -> dict[str, bool]:
    """Capabilities the original grader assigned to the i-th grade call."""
    if index >= len(grade_calls):
        return {}
    result = grade_calls[index].get("result") or {}
    caps = result.get("capabilities") or {}
    return {str(k): bool(v) for k, v in caps.items()}


def _parse_caps(replayed: Any) -> dict[str, bool] | None:
    """Capabilities from a replayed grade. Returns None on unparseable."""
    if replayed.structured:
        caps = replayed.structured.get("capabilities")
        if isinstance(caps, dict):
            return {str(k): bool(v) for k, v in caps.items()}
    if not replayed.text:
        return None
    try:
        payload = json.loads(replayed.text)
    except (ValueError, TypeError):
        return None
    if isinstance(payload, dict) and isinstance(payload.get("capabilities"), dict):
        return {str(k): bool(v) for k, v in payload["capabilities"].items()}
    return None


async def reproduce_run(
    run_dir: Path,
    *,
    image_ref: str | None = None,
    on_progress: Callable[[int, int | None, str, float], None] | None = None,
) -> ReproductionReport:
    """Replay each grade() call from run_dir against a fresh container.

    `image_ref` defaults to the digest in job.json so the reproduction
    targets the exact image bytes the original episode used, regardless
    of where registry tags have since moved.
    """
    from qed_swe_bench.runner.mcp_client import McpDockerSession
    from qed_swe_bench.runner.run_dir import parse_run_id_from_dir_name

    run_id = parse_run_id_from_dir_name(run_dir.name)

    if image_ref is None:
        image_ref = _read_image_ref(run_dir)
        if image_ref is None:
            return ReproductionReport(
                run_id=run_id, image_ref="", poc_count=0,
                comparisons=(), overall_match=False,
                error="job.json missing or has no image_digest/image_ref",
            )

    tool_calls = read_tool_calls_jsonl(run_dir / "tool_calls.jsonl")
    grade_calls = read_tool_calls_jsonl(run_dir / "grade_calls.jsonl")
    grade_idx_in_tool_calls = _grade_indices(tool_calls)

    if not grade_idx_in_tool_calls:
        return ReproductionReport(
            run_id=run_id, image_ref=image_ref, poc_count=0,
            comparisons=(), overall_match=True,  # nothing to verify
        )

    comparisons: list[ComparisonResult] = []
    try:
        async with McpDockerSession.start(image_ref) as session:
            replayed = await replay_tool_calls(
                session, tool_calls, on_progress=on_progress,
            )
    except Exception as exc:  # noqa: BLE001 — session-level fatal
        return ReproductionReport(
            run_id=run_id, image_ref=image_ref,
            poc_count=len(grade_idx_in_tool_calls),
            comparisons=(), overall_match=False,
            error=f"{type(exc).__name__}: {exc}",
        )

    for grade_position, idx in enumerate(grade_idx_in_tool_calls):
        recorded = _recorded_capabilities(grade_calls, grade_position)
        poc_path = str(tool_calls[idx].get("args", {}).get("path", ""))

        if idx >= len(replayed):
            comparisons.append(ComparisonResult.compare(
                grade_call_index=grade_position, poc_path=poc_path,
                recorded=recorded, reproduced=None,
                error="replay truncated before this grade call",
            ))
            continue
        rc = replayed[idx]
        if rc.error is not None:
            comparisons.append(ComparisonResult.compare(
                grade_call_index=grade_position, poc_path=poc_path,
                recorded=recorded, reproduced=None,
                error=f"grade replay raised: {rc.error}",
            ))
            continue
        caps = _parse_caps(rc)
        if caps is None:
            comparisons.append(ComparisonResult.compare(
                grade_call_index=grade_position, poc_path=poc_path,
                recorded=recorded, reproduced=None,
                error=f"grade returned unparseable result: {rc.text[:200]}",
            ))
            continue
        comparisons.append(ComparisonResult.compare(
            grade_call_index=grade_position, poc_path=poc_path,
            recorded=recorded, reproduced=caps,
        ))

    overall = bool(comparisons) and all(c.matches for c in comparisons)
    return ReproductionReport(
        run_id=run_id, image_ref=image_ref,
        poc_count=len(grade_idx_in_tool_calls),
        comparisons=tuple(comparisons), overall_match=overall,
    )


def reproduce_run_sync(
    run_dir: Path,
    *,
    image_ref: str | None = None,
    on_progress: Callable[[int, int | None, str, float], None] | None = None,
) -> ReproductionReport:
    """Sync wrapper for the CLI — `asyncio.run()` is fine for one-shot use."""
    import asyncio
    return asyncio.run(reproduce_run(
        run_dir, image_ref=image_ref, on_progress=on_progress,
    ))
