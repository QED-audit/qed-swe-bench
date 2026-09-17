"""Check 2: live MCP tool surface matches the manifest's episode_tools.

Spawns the container, opens an MCP session, lists tools, compares against
the manifest. Both directions are reported:
  - missing tools: declared in manifest but absent from the running server.
    HARD FAIL — the agent loop would crash mid-episode.
  - extra tools:   present in the server but undeclared in the manifest.
    HARD FAIL — the agent could call something the manifest didn't audit
    (reward-hacking surface). Detect-task envs especially must NOT expose
    grading internals as MCP tools.

For envs declaring `interface_flavor: rl_mcp_v1` we still verify the
default mcp serve surface; the `--validate` and `--grade` CLI commands
are out-of-band and not exercised by this check.
"""

from __future__ import annotations

import logging

from qed_swe_bench.contract.manifest import Manifest
from qed_swe_bench.runner.mcp_client import McpDockerSession
from qed_swe_bench.validator.runner import CheckResult, CheckStatus

log = logging.getLogger(__name__)
CHECK_NAME = "mcp_contract"


async def check(manifest: Manifest, image_ref: str) -> CheckResult:
    declared = set(manifest.episode_tools)
    if not declared:
        return CheckResult(
            name=CHECK_NAME,
            status=CheckStatus.UNVERIFIED,
            message="manifest declares no episode_tools; nothing to compare",
        )

    try:
        async with McpDockerSession.start(image_ref) as session:
            tools = await session.list_tools()
    except Exception as e:  # noqa: BLE001 — anything from docker/MCP is a fail
        log.warning("mcp_contract: failed to spawn or list tools: %s", e)
        return CheckResult(
            name=CHECK_NAME,
            status=CheckStatus.FAIL,
            message=f"could not list MCP tools: {e}",
        )

    actual = {t.name for t in tools}

    missing = sorted(declared - actual)
    extra = sorted(actual - declared - set(manifest.evaluation_tools))

    if missing or extra:
        msg_parts = []
        if missing:
            msg_parts.append(f"missing: {missing}")
        if extra:
            msg_parts.append(f"unexpected: {extra}")
        return CheckResult(
            name=CHECK_NAME,
            status=CheckStatus.FAIL,
            message="; ".join(msg_parts),
            details={"declared": sorted(declared), "actual": sorted(actual)},
        )

    return CheckResult(
        name=CHECK_NAME,
        status=CheckStatus.PASS,
        message=f"{len(actual)} tools match manifest",
        details={"tools": sorted(actual)},
    )
