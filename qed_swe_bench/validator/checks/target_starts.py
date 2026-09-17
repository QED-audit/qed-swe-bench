"""Check 3: target binary starts.

Calls the manifest-declared `setup` MCP tool (if present) and a 'noop' /
shell tool to confirm the env's basic plumbing works. We don't try to
*run* the target — that's expensive and lives in known_pov_reproduces.
What we want here is "the setup tool returns a non-error result", which
is the agent's first interaction in every episode.

If the env doesn't expose a `setup` tool (some custom interfaces won't),
we run `exec` with `echo ok` to verify the shell tool is alive.
"""

from __future__ import annotations

import logging

from qed_swe_bench.contract.manifest import Manifest
from qed_swe_bench.runner.mcp_client import McpDockerSession
from qed_swe_bench.validator.runner import CheckResult, CheckStatus

log = logging.getLogger(__name__)
CHECK_NAME = "target_starts"


async def check(manifest: Manifest, image_ref: str) -> CheckResult:
    try:
        async with McpDockerSession.start(image_ref) as session:
            tools = {t.name for t in await session.list_tools()}

            # Prefer a shell tool — universal smoke that doesn't need
            # task-specific arguments. bench-v8 envs use `exec`,
            # rlenv-mcp envs use `bash`. Both take {command: str}.
            for shell_name in ("exec", "bash"):
                if shell_name in tools:
                    result = await session.call_tool(shell_name, {"command": "echo ok"})
                    if result.is_error or "ok" not in result.text:
                        return CheckResult(
                            name=CHECK_NAME,
                            status=CheckStatus.FAIL,
                            message=f"{shell_name} smoke failed: {result.text[:200]}",
                        )
                    return CheckResult(
                        name=CHECK_NAME,
                        status=CheckStatus.PASS,
                        message=f"{shell_name} smoke succeeded",
                        details={"tool": shell_name},
                    )

            # No shell tool — fall back to a setup tool. bench-v8's
            # `setup` takes no args; rlenv-mcp's `setup_problem`
            # requires {problem_id: str} so we'd need to read
            # /rlenv/problem/id first. Without a shell tool we can't
            # do that cheaply; mark UNVERIFIED rather than guess.
            if "setup" in tools:
                result = await session.call_tool("setup", {})
                if result.is_error:
                    return CheckResult(
                        name=CHECK_NAME,
                        status=CheckStatus.FAIL,
                        message=f"setup returned is_error=True: {result.text[:200]}",
                    )
                return CheckResult(
                    name=CHECK_NAME,
                    status=CheckStatus.PASS,
                    message="setup tool succeeded",
                    details={"text_len": len(result.text), "tool": "setup"},
                )

            return CheckResult(
                name=CHECK_NAME,
                status=CheckStatus.UNVERIFIED,
                message="no exec/bash/setup tool to probe (setup_problem "
                        "needs a problem_id arg we don't infer here)",
            )
    except Exception as e:  # noqa: BLE001
        log.warning("target_starts: %s", e)
        return CheckResult(
            name=CHECK_NAME,
            status=CheckStatus.FAIL,
            message=f"container/MCP failed: {e}",
        )
