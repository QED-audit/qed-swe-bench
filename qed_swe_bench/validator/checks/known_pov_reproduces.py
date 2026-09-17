"""Check 4: documented PoV still reproduces the documented capabilities.

If the manifest's `metadata.known_pov` describes a recorded proof-of-vuln
(an `exec` command line + the capability flags it should set), we run it
inside a fresh container and grade. If the resulting capabilities at
least cover the documented set, PASS. If grading runs but flags drift,
FAIL. If no PoV is documented, UNVERIFIED (which the report rolls up to
'pass_unverified', not 'fail').

This is the load-bearing protection against silent env rot — image
rebuilds, library updates, or kernel changes can perturb a PoV's
behavior. We catch that here before the LLM benchmark wastes hours on a
broken target.

Format expected in `manifest.metadata.known_pov`:

    metadata:
      known_pov:
        # Run this as the PoV trigger, via the env's exec tool.
        exec: "/work/repro.sh"
        # After exec, call this grade tool (mcp_tool) or shell command
        # (cli_oneshot, falls back to evaluate.command).
        # Capability flags that MUST appear in the grader's output.
        expected_flags: ["crash", "asan"]
"""

from __future__ import annotations

import logging

from qed_swe_bench.contract.manifest import Manifest
from qed_swe_bench.runner.mcp_client import McpDockerSession
from qed_swe_bench.validator.runner import CheckResult, CheckStatus

log = logging.getLogger(__name__)
CHECK_NAME = "known_pov_reproduces"


def _expected(manifest: Manifest) -> tuple[str | None, list[str]]:
    pov = (manifest.metadata or {}).get("known_pov") or {}
    return pov.get("exec"), list(pov.get("expected_flags") or [])


async def check(manifest: Manifest, image_ref: str) -> CheckResult:
    pov_cmd, expected_flags = _expected(manifest)
    if not pov_cmd:
        return CheckResult(
            name=CHECK_NAME,
            status=CheckStatus.UNVERIFIED,
            message="manifest does not declare metadata.known_pov.exec",
        )

    if manifest.grader_kind != "mcp_tool":
        # cli_oneshot grading needs the runner to spawn a separate
        # container with --grade after the episode; out of scope for an
        # in-MCP smoke check. Mark UNVERIFIED rather than FAIL.
        return CheckResult(
            name=CHECK_NAME,
            status=CheckStatus.UNVERIFIED,
            message="cli_oneshot grader; PoV repro requires grade-container path (TODO)",
        )

    try:
        async with McpDockerSession.start(image_ref) as session:
            tools = {t.name for t in await session.list_tools()}

            if "exec" not in tools:
                return CheckResult(
                    name=CHECK_NAME,
                    status=CheckStatus.UNVERIFIED,
                    message="env has no exec tool; cannot run PoV from manifest",
                )
            if manifest.grader_tool not in tools:
                return CheckResult(
                    name=CHECK_NAME,
                    status=CheckStatus.FAIL,
                    message=f"grader tool '{manifest.grader_tool}' not exposed by env",
                )

            run_result = await session.call_tool("exec", {"command": pov_cmd})
            if run_result.is_error:
                return CheckResult(
                    name=CHECK_NAME,
                    status=CheckStatus.FAIL,
                    message=f"PoV exec failed: {run_result.text[:200]}",
                )

            grade_result = await session.call_tool(manifest.grader_tool, {})
            achieved = _flags_from_grade(grade_result.structured, grade_result.text)
            missing = [f for f in expected_flags if f not in achieved]
            if missing:
                return CheckResult(
                    name=CHECK_NAME,
                    status=CheckStatus.FAIL,
                    message=f"PoV ran but did not set expected flags: {missing}",
                    details={"achieved": sorted(achieved), "expected": expected_flags},
                )

            return CheckResult(
                name=CHECK_NAME,
                status=CheckStatus.PASS,
                message=f"PoV reproduces flags: {expected_flags}",
                details={"achieved": sorted(achieved)},
            )
    except Exception as e:  # noqa: BLE001
        log.warning("known_pov_reproduces: %s", e)
        return CheckResult(
            name=CHECK_NAME,
            status=CheckStatus.FAIL,
            message=f"container/MCP failed: {e}",
        )


def _flags_from_grade(structured: dict | None, text: str) -> set[str]:
    """Pull the set of true capability flags from a grade response.

    Accepts both the structuredContent path (preferred) and a fallback of
    parsing the text for `"flag": true` JSON markers — which matches
    bench-v8's grader output shape.
    """
    flags: set[str] = set()
    if structured:
        capabilities = structured.get("capabilities") or structured
        for k, v in (capabilities or {}).items():
            if v is True:
                flags.add(k)
        return flags
    # Best-effort text scan; any false-positives here only weaken the
    # check, not strengthen it, since we report PASS only when ALL
    # expected flags appear.
    import json
    import re

    for m in re.finditer(r"\{[^{}]+\}", text):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        for k, v in obj.items():
            if v is True:
                flags.add(k)
    return flags
