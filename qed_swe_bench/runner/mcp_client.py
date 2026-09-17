"""Async MCP stdio client wrapping `mcp` Python SDK.

Spawns `docker run --network none -i --rm <image_digest>` per episode and
connects to bench-v8's MCP server on stdio. Exposes a small, synchronous-
feeling API that the agent loop calls:

    async with McpDockerSession.start(image_digest) as session:
        tools = await session.list_tools()
        result = await session.call_tool("setup", {})

The docker run flags mirror `bench-v8/agent.py:48-52` — most importantly,
`--network none` keeps the model from reaching out from inside the env.
The one addition vs bench-v8 is `--rm` (bench-v8 leaves containers around;
across a 100+-tuple sweep we'd accumulate hundreds of stopped containers).
"""

from __future__ import annotations

import logging
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

log = logging.getLogger(__name__)


# Docker run flags. `--network none` blocks exfiltration. `--init` runs tini
# as PID 1 to reap the processes the exec tool's group-kill / UID-sweep leaves
# orphaned (a non-reaping PID-1 server would let them become zombies).
DEFAULT_DOCKER_RUN_ARGS: tuple[str, ...] = (
    "--init",
    "--network", "none",
    "-i", "--rm",
)


@dataclass(frozen=True)
class ToolDef:
    """Compact MCP tool description we hand to the LLM clients via tools.py."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    """Result of a single MCP tool call."""

    is_error: bool
    text: str                          # concatenation of all TextContent blocks
    structured: dict[str, Any] | None  # populated if server returned structuredContent


class McpDockerSession:
    """Holds an open MCP stdio session against a docker container."""

    def __init__(self, session: ClientSession) -> None:
        self._session = session

    async def list_tools(self) -> list[ToolDef]:
        """Return all tools the server exposes, normalized to ToolDef."""
        result = await self._session.list_tools()
        out: list[ToolDef] = []
        for t in result.tools:
            schema = t.inputSchema or {"type": "object", "properties": {}}
            out.append(
                ToolDef(
                    name=t.name,
                    description=t.description or "",
                    input_schema=schema,
                )
            )
        return out

    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> ToolResult:
        """Invoke a tool, normalizing the response."""
        result = await self._session.call_tool(name, arguments or {})

        text_parts: list[str] = []
        for block in result.content or []:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(block.text)
            else:
                # Image / resource / etc. — capture a placeholder so the loop
                # has *something* to feed back.
                text_parts.append(f"<{btype} content omitted>")

        return ToolResult(
            is_error=bool(result.isError),
            text="\n".join(text_parts),
            structured=result.structuredContent,
        )

    @classmethod
    @asynccontextmanager
    async def start(
        cls,
        image: str,
        *,
        docker_bin: str = "docker",
        extra_docker_args: tuple[str, ...] | list[str] = DEFAULT_DOCKER_RUN_ARGS,
        env: dict[str, str] | None = None,
        stderr_path: Path | None = None,
    ) -> AsyncIterator[McpDockerSession]:
        """Spawn a docker container and open an MCP session against it.

        `image` is whatever `docker run` accepts: a digest, tag, or local ref.
        For benchmark runs, prefer the resolved sha256 digest from
        runner/image_ref.py.

        The container is removed automatically (--rm) when the session closes.

        `stderr_path` (optional): when set, the MCP container's stderr is
        captured to that file (line-buffered, append-mode) instead of the
        orchestrator's stderr. Used by the orchestrator to land each
        episode's MCP server diagnostics at `run_dir/mcp_stderr.log` for
        post-hoc reward-hacking audits — repeated identical bash calls,
        unexpected MCP server warnings, etc.
        """
        # Defense-in-depth — refuse refs shaped like a docker-run flag.
        # In the normal flow `image` is `resolved.image_digest` from
        # `runner.image_ref.resolve`, which can't produce a flag-shaped
        # value. But this method is also reachable from tests, the
        # `audit --reproduce` path, and any future caller, so the same
        # guard `cli_oneshot_grader` and `validator/checks/integrity_posture`
        # apply belongs here too.
        from qed_swe_bench.runner.image_ref import assert_safe_image_ref
        assert_safe_image_ref(image)
        params = StdioServerParameters(
            command=docker_bin,
            args=["run", *extra_docker_args, image],
            env=env,
        )
        log.debug("spawning MCP container: %s run %s %s", docker_bin, list(extra_docker_args), image)

        # Open a file for the container's stderr if requested. Line-buffered
        # so partial output is readable mid-run; append mode in case a future
        # caller resumes against the same path.
        errlog_fh = None
        if stderr_path is not None:
            stderr_path.parent.mkdir(parents=True, exist_ok=True)
            errlog_fh = stderr_path.open("a", buffering=1, encoding="utf-8")
        try:
            errlog = errlog_fh if errlog_fh is not None else sys.stderr
            async with stdio_client(params, errlog=errlog) as (read, write), \
                    ClientSession(read, write) as session:
                await session.initialize()
                yield cls(session)
        finally:
            if errlog_fh is not None:
                errlog_fh.close()
