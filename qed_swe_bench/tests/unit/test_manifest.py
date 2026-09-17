"""Manifest schema + interface-consistency tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from qed_swe_bench.contract.manifest import (
    ManifestInterfaceError,
    ManifestSchemaError,
    from_dict,
    load,
    load_schema,
)

FIXTURES = Path(__file__).parent.parent / "golden" / "manifests"


def test_schema_loads_and_has_expected_top_level_keys() -> None:
    schema = load_schema()
    required = set(schema["required"])
    assert {"schema_version", "env_id", "interface", "image", "task_type", "evaluate"} <= required


def test_v8_manifest_loads_cleanly() -> None:
    m = load(FIXTURES / "v8_e25.yaml")
    assert m.env_id == "v8-e25"
    assert m.interface == "rl.mcp.v8_task.v1"
    assert m.interface_flavor == "bench_v8"
    assert m.grader_kind == "mcp_tool"
    assert m.grader_tool == "grade"
    assert m.expected_capabilities == (
        "cov_func", "cov_line", "crash", "asan", "addrof", "fakeobj",
    )
    assert "/rlenv/mcp/server" in m.integrity_baseline["grader_paths"]


def test_unknown_interface_rejected_by_consistency_check() -> None:
    with pytest.raises(ManifestInterfaceError) as exc:
        load(FIXTURES / "invalid_unknown_interface.yaml")
    assert "rl.mcp.totally_fake.v1" in str(exc.value)


def test_missing_required_fields_rejected_by_schema() -> None:
    with pytest.raises(ManifestSchemaError) as exc:
        load(FIXTURES / "invalid_missing_required.yaml")
    assert "evaluate" in str(exc.value)


def test_unknown_top_level_key_rejected() -> None:
    """additionalProperties=false at the top — typo guard."""
    raw = {
        "schema_version": "1",
        "env_id": "x",
        "interface": "rl.mcp.v8_task.v1",
        "task_type": "binary_task",
        "image": {"ref": "local/x:latest"},
        "evaluate": {"kind": "mcp_tool", "tool": "grade"},
        "mcp": {"episode_tools": ["grade"]},
        "i_meant_metadata": {},
    }
    with pytest.raises(ManifestSchemaError):
        from_dict(raw)


def test_episode_tool_not_in_interface_rejected() -> None:
    raw = {
        "schema_version": "1",
        "env_id": "x",
        "interface": "rl.mcp.v8_task.v1",
        "task_type": "binary_task",
        "image": {"ref": "local/x:latest"},
        "mcp": {"episode_tools": ["setup", "exec", "grade", "totally_made_up_tool"]},
        "evaluate": {"kind": "mcp_tool", "tool": "grade"},
    }
    with pytest.raises(ManifestInterfaceError) as exc:
        from_dict(raw)
    assert "totally_made_up_tool" in str(exc.value)


def test_expected_capability_outside_interface_bitmap_rejected() -> None:
    raw = {
        "schema_version": "1",
        "env_id": "x",
        "interface": "rl.mcp.v8_task.v1",
        "task_type": "binary_task",
        "image": {"ref": "local/x:latest"},
        "mcp": {"episode_tools": ["setup", "exec", "grade"]},
        "evaluate": {"kind": "mcp_tool", "tool": "grade"},
        "expected_capabilities": ["crash", "ace_super"],  # ace_super not declared
    }
    with pytest.raises(ManifestInterfaceError) as exc:
        from_dict(raw)
    assert "ace_super" in str(exc.value)


def test_image_ref_starting_with_dash_rejected_by_schema() -> None:
    """Defense: a manifest can't smuggle a docker-run flag as image.ref."""
    raw = {
        "schema_version": "1",
        "env_id": "x",
        "interface": "rl.mcp.v8_task.v1",
        "task_type": "binary_task",
        "image": {"ref": "-v=/host:/inside"},
        "mcp": {"episode_tools": ["setup", "exec", "grade"]},
        "evaluate": {"kind": "mcp_tool", "tool": "grade"},
    }
    with pytest.raises(ManifestSchemaError):
        from_dict(raw)


