"""Unit tests for runner/cli_oneshot_grader.py.

Subprocess-mocked: we never actually `docker run` here. Real-image coverage
lives in the integration tier (Phase B+, when the first authored
rl_mcp_v1 image lands).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from qed_swe_bench.runner.cli_oneshot_grader import (
    GraderError,
    _extract_capabilities,
    run_cli_oneshot_grader,
)


def _make_proc(*, stdout: bytes, stderr: bytes = b"", returncode: int = 0) -> AsyncMock:
    proc = AsyncMock()
    proc.communicate.return_value = (stdout, stderr)
    proc.returncode = returncode
    return proc


def test_extract_capabilities_from_nested_shape() -> None:
    raw = {"capabilities": {"crash": True, "asan": False}, "score": 0.5}
    assert _extract_capabilities(raw) == {"crash": True, "asan": False}


def test_extract_capabilities_from_flat_shape() -> None:
    raw = {"crash": True, "asan": True, "score": 0.6, "note": "extra"}
    assert _extract_capabilities(raw) == {"crash": True, "asan": True}


def test_extract_capabilities_skips_non_bool() -> None:
    raw = {"capabilities": {"crash": True, "count": 5, "label": "bad"}}
    assert _extract_capabilities(raw) == {"crash": True}


def test_refuses_flag_like_image_ref(tmp_path: Path) -> None:
    """Defense in depth — same guard as integrity_posture._compute_sha256."""
    import asyncio
    with pytest.raises(ValueError, match="flag-like image_ref"):
        asyncio.run(
            run_cli_oneshot_grader(
                image_ref="-v=/host:/inside",
                command=["/grader"],
                run_dir=tmp_path,
            )
        )


@pytest.mark.asyncio
async def test_happy_path_parses_capabilities(tmp_path: Path) -> None:
    out = json.dumps({"capabilities": {"crash": True, "asan": True}, "score": 0.7})
    with patch(
        "qed_swe_bench.runner.cli_oneshot_grader.asyncio.create_subprocess_exec",
        new=AsyncMock(return_value=_make_proc(stdout=out.encode())),
    ) as spawn:
        result = await run_cli_oneshot_grader(
            image_ref="local/sample:latest",
            command=("/rlenv/mcp/server", "--grade"),
            run_dir=tmp_path,
        )
    assert result.capabilities == {"crash": True, "asan": True}
    assert result.exit_code == 0
    assert "score" in result.metadata
    # docker_cmd is positional; make sure the image ref is present and
    # the manifest command appears after it as the container CMD args.
    args = spawn.call_args.args
    assert "docker" in args
    assert "local/sample:latest" in args
    assert "/rlenv/mcp/server" in args
    assert "--grade" in args


@pytest.mark.asyncio
async def test_non_zero_exit_raises_graderror(tmp_path: Path) -> None:
    proc = _make_proc(stdout=b"", stderr=b"build failed\n", returncode=2)
    with patch(
        "qed_swe_bench.runner.cli_oneshot_grader.asyncio.create_subprocess_exec",
        new=AsyncMock(return_value=proc),
    ), pytest.raises(GraderError, match="exited 2"):
        await run_cli_oneshot_grader(
            image_ref="local/sample:latest",
            command=("/grader",),
            run_dir=tmp_path,
        )


@pytest.mark.asyncio
async def test_unparseable_stdout_raises_graderror(tmp_path: Path) -> None:
    proc = _make_proc(stdout=b"not json at all")
    with patch(
        "qed_swe_bench.runner.cli_oneshot_grader.asyncio.create_subprocess_exec",
        new=AsyncMock(return_value=proc),
    ), pytest.raises(GraderError, match="not JSON"):
        await run_cli_oneshot_grader(
            image_ref="local/sample:latest",
            command=("/grader",),
            run_dir=tmp_path,
        )


@pytest.mark.asyncio
async def test_non_object_json_raises_graderror(tmp_path: Path) -> None:
    proc = _make_proc(stdout=b"[1,2,3]")
    with patch(
        "qed_swe_bench.runner.cli_oneshot_grader.asyncio.create_subprocess_exec",
        new=AsyncMock(return_value=proc),
    ), pytest.raises(GraderError, match="not a JSON object"):
        await run_cli_oneshot_grader(
            image_ref="local/sample:latest",
            command=("/grader",),
            run_dir=tmp_path,
        )


@pytest.mark.asyncio
async def test_empty_capabilities_yield_empty_bitmap(tmp_path: Path) -> None:
    proc = _make_proc(stdout=b'{"capabilities": {}}')
    with patch(
        "qed_swe_bench.runner.cli_oneshot_grader.asyncio.create_subprocess_exec",
        new=AsyncMock(return_value=proc),
    ):
        result = await run_cli_oneshot_grader(
            image_ref="local/sample:latest",
            command=("/grader",),
            run_dir=tmp_path,
        )
    assert result.capabilities == {}
    assert result.exit_code == 0


@pytest.mark.asyncio
async def test_extra_env_passed_through(tmp_path: Path) -> None:
    proc = _make_proc(stdout=b'{"capabilities": {"ok": true}}')
    with patch(
        "qed_swe_bench.runner.cli_oneshot_grader.asyncio.create_subprocess_exec",
        new=AsyncMock(return_value=proc),
    ) as spawn:
        await run_cli_oneshot_grader(
            image_ref="local/sample:latest",
            command=("/grader",),
            run_dir=tmp_path,
            extra_env={"RLENV_DEBUG": "1"},
        )
    args = spawn.call_args.args
    assert "RLENV_DEBUG=1" in args
