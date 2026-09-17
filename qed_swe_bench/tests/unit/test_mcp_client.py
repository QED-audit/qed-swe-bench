"""McpDockerSession normalization (mocked; no real Docker calls)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from qed_swe_bench.runner.mcp_client import (
    DEFAULT_DOCKER_RUN_ARGS,
    McpDockerSession,
    ToolDef,
    ToolResult,
)


def _fake_tool(name: str, description: str = "", schema: dict | None = None):
    return SimpleNamespace(name=name, description=description, inputSchema=schema)


def _fake_tools_result(tools):
    return SimpleNamespace(tools=tools)


def _fake_text_block(text: str):
    return SimpleNamespace(type="text", text=text)


def _fake_image_block():
    return SimpleNamespace(type="image", data="<bytes>")


def _fake_call_result(*, content, is_error: bool = False, structured=None):
    return SimpleNamespace(content=content, isError=is_error, structuredContent=structured)


@pytest.mark.asyncio
async def test_list_tools_returns_tooldef_with_schema_default() -> None:
    fake_session = SimpleNamespace(
        list_tools=AsyncMock(
            return_value=_fake_tools_result(
                [
                    _fake_tool("setup", "Initialize", {"type": "object"}),
                    _fake_tool("noschema", "no schema", None),  # missing inputSchema
                ]
            )
        )
    )
    sess = McpDockerSession(fake_session)
    tools = await sess.list_tools()
    assert len(tools) == 2
    assert isinstance(tools[0], ToolDef)
    assert tools[0].name == "setup"
    assert tools[0].description == "Initialize"
    assert tools[0].input_schema == {"type": "object"}
    # Missing schema gets a sane default.
    assert tools[1].input_schema == {"type": "object", "properties": {}}


@pytest.mark.asyncio
async def test_call_tool_concatenates_text_blocks() -> None:
    fake_session = SimpleNamespace(
        call_tool=AsyncMock(
            return_value=_fake_call_result(
                content=[_fake_text_block("line 1"), _fake_text_block("line 2")],
                structured={"k": "v"},
            )
        )
    )
    sess = McpDockerSession(fake_session)
    result = await sess.call_tool("exec", {"cmd": "ls"})
    assert isinstance(result, ToolResult)
    assert result.is_error is False
    assert result.text == "line 1\nline 2"
    assert result.structured == {"k": "v"}


@pytest.mark.asyncio
async def test_call_tool_handles_non_text_blocks() -> None:
    fake_session = SimpleNamespace(
        call_tool=AsyncMock(
            return_value=_fake_call_result(
                content=[_fake_text_block("ok"), _fake_image_block()],
            )
        )
    )
    sess = McpDockerSession(fake_session)
    result = await sess.call_tool("read_file", {"path": "/x"})
    assert "ok" in result.text
    assert "image content omitted" in result.text


@pytest.mark.asyncio
async def test_call_tool_propagates_error_flag() -> None:
    fake_session = SimpleNamespace(
        call_tool=AsyncMock(
            return_value=_fake_call_result(
                content=[_fake_text_block("ENOENT: no such file")],
                is_error=True,
            )
        )
    )
    sess = McpDockerSession(fake_session)
    result = await sess.call_tool("read_file", {"path": "/nonexistent"})
    assert result.is_error is True
    assert "ENOENT" in result.text


@pytest.mark.asyncio
async def test_call_tool_passes_default_args_dict() -> None:
    """When the loop calls without arguments, we pass an empty dict, not None."""
    mock_call = AsyncMock(return_value=_fake_call_result(content=[]))
    fake_session = SimpleNamespace(call_tool=mock_call)
    sess = McpDockerSession(fake_session)
    await sess.call_tool("setup")
    mock_call.assert_awaited_once_with("setup", {})


def test_default_docker_run_args_disable_network() -> None:
    """Network must be disabled by default — model can't reach out from env."""
    assert "--network" in DEFAULT_DOCKER_RUN_ARGS
    idx = DEFAULT_DOCKER_RUN_ARGS.index("--network")
    assert DEFAULT_DOCKER_RUN_ARGS[idx + 1] == "none"
    assert "-i" in DEFAULT_DOCKER_RUN_ARGS  # need stdin for MCP stdio
    assert "--rm" in DEFAULT_DOCKER_RUN_ARGS  # cleanup after exit


def test_default_docker_run_args_process_cleanup() -> None:
    """tini (--init) reaps the processes the exec cleanup orphans."""
    assert "--init" in DEFAULT_DOCKER_RUN_ARGS
    assert "--cap-add" not in DEFAULT_DOCKER_RUN_ARGS  # PID-ns path dropped
