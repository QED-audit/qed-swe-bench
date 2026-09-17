"""Run an env's cli_oneshot grader and parse the result.

For envs with `manifest.evaluate.kind == 'cli_oneshot'` (e.g. rlenv-mcp's
detect / solve / patch tasks), grading happens *after* the episode
ends, in a fresh container that runs the manifest-declared `evaluate.command`
argv. This is the reward-hacking guard for detect/patch tasks: the
grading binary is never exposed to the agent during the episode.

Output contract: stdout SHOULD be a single JSON object containing a
`capabilities` map of `{flag: bool}`. We accept both `{capabilities:
{...}}` and a flat `{flag: bool, ...}` shape; downstream code only
cares about the `flag → bool` projection.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class GraderError(RuntimeError):
    """The cli_oneshot grader failed to produce a parseable result."""


@dataclass(frozen=True)
class GradeResult:
    capabilities: dict[str, bool]
    raw: dict[str, Any]
    stdout: str
    stderr: str
    exit_code: int
    metadata: dict[str, Any] = field(default_factory=dict)


def _extract_capabilities(raw: dict[str, Any]) -> dict[str, bool]:
    """Pull `{flag: bool}` from either {capabilities: {...}} or a flat dict.

    Non-bool values are ignored so the grader can include diagnostics
    (counts, timings, file paths) alongside flags without polluting the
    capability bitmap.
    """
    cap_section = raw.get("capabilities") if isinstance(raw.get("capabilities"), dict) else raw
    return {k: v for k, v in cap_section.items() if isinstance(v, bool)}


async def run_cli_oneshot_grader(
    *,
    image_ref: str,
    command: tuple[str, ...] | list[str],
    run_dir: Path,
    timeout_s: int = 120,
    extra_env: dict[str, str] | None = None,
) -> GradeResult:
    """Spawn `docker run --rm --network none <image> <command...>` and
    return the parsed grade.

    `run_dir` is mounted read-only at `/run` so the grader can read the
    transcript / artifacts the agent produced during the episode.

    Raises GraderError on a non-zero exit, JSON parse failure, or timeout.
    Raises ValueError if `image_ref` is shaped like a docker flag.
    """
    from qed_swe_bench.runner.image_ref import assert_safe_image_ref
    assert_safe_image_ref(image_ref)

    docker_cmd: list[str] = [
        "docker", "run", "--rm", "--network", "none",
        "--volume", f"{run_dir.resolve()}:/run:ro",
    ]
    for k, v in (extra_env or {}).items():
        docker_cmd += ["--env", f"{k}={v}"]
    docker_cmd += [image_ref, *list(command)]

    log.debug("cli_oneshot grader: %s", docker_cmd)
    proc = await asyncio.create_subprocess_exec(
        *docker_cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_b, stderr_b = await asyncio.wait_for(
            proc.communicate(), timeout=timeout_s
        )
    except TimeoutError:
        proc.kill()
        await proc.wait()
        raise GraderError(f"cli_oneshot grader timed out after {timeout_s}s") from None

    stdout = stdout_b.decode(errors="replace")
    stderr = stderr_b.decode(errors="replace")

    if proc.returncode != 0:
        raise GraderError(
            f"cli_oneshot grader exited {proc.returncode}: {stderr[:500]}"
        )

    try:
        raw = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise GraderError(
            f"cli_oneshot grader stdout is not JSON: {exc}; "
            f"first 200 chars: {stdout[:200]!r}"
        ) from exc

    if not isinstance(raw, dict):
        raise GraderError(
            f"cli_oneshot grader output is not a JSON object: {type(raw).__name__}"
        )

    capabilities = _extract_capabilities(raw)
    metadata = {k: v for k, v in raw.items() if k != "capabilities" and not isinstance(v, bool)}

    return GradeResult(
        capabilities=capabilities,
        raw=raw,
        stdout=stdout,
        stderr=stderr,
        exit_code=proc.returncode,
        metadata=metadata,
    )
