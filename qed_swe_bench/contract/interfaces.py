"""RL env interface registry.

Each env declares an `interface` string from this registry. The interface
captures what an env IS (tools it must expose, submission shape it accepts,
how grading is computed, what capability flags are meaningful) without
constraining what target software it runs against.

Status today:
  - The interface name is stored on every `runs` row for audit / future
    enforcement.
  - The registry is pre-populated with the interfaces we expect long-term —
    so adding a new env type is "set interface: rl.mcp.X" in the config and
    the registry already knows the contract.
  - The actual-tool-surface validator (`qed_swe_bench validate-image`) checks
    a registered env against its declared interface; that catalog + manifest
    plumbing is in place. The interface registry itself is still primarily
    a typed namespace — call sites read from it, but enforcement is opt-in.

Naming convention `rl.mcp.<task>.<version>`:
  - `rl.mcp` — RL environment, MCP-served
  - `<task>` — what the env is FOR (binary_task, lifecycle, ...)
  - `<version>` — semver-ish; bump when the contract changes incompatibly
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

GraderKind = Literal["mcp_tool", "cli_oneshot"]
SubmissionKind = Literal["task", "lifecycle"]


@dataclass(frozen=True)
class Interface:
    """One interface contract."""

    name: str
    description: str
    base: str | None = None                   # parent interface, for shape inheritance / docs
    mcp_tools: tuple[str, ...] = ()           # tools the server is expected to expose
    capability_flags: tuple[str, ...] = ()    # bitmap keys the grader emits (if any)
    submission_kind: SubmissionKind | None = None
    grader_kind: GraderKind = "mcp_tool"
    grader_tool: str | None = "grade"         # MCP tool name when grader_kind=='mcp_tool'
    grader_command: tuple[str, ...] | None = None  # argv when grader_kind=='cli_oneshot'
    notes: str = ""


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

RL_MCP_BASE = Interface(
    name="rl.mcp.base.v1",
    description="Base contract: filesystem + shell exec. Not used directly; "
                "a parent class for task-specific interfaces.",
    mcp_tools=("setup", "exec", "list_directory", "read_file", "write_file"),
)


# ---------------------------------------------------------------------------
# Task family (V8 + binary-only variants)
# ---------------------------------------------------------------------------

RL_MCP_V8_TASK = Interface(
    name="rl.mcp.v8_task.v1",
    description="V8 JavaScript engine task. grade() returns the "
                "16-flag capability bitmap (cov_*, asan, crash, "
                "addrof/fakeobj/caged_*, infoleak_*, arb_*, pc_control, ace). "
                "The 'addrof' / 'fakeobj' / 'caged_*' flags are V8-engine-specific "
                "JS-runtime primitives, which is why this is its own interface "
                "and not the generic rl.mcp.binary_task.v1.",
    base="rl.mcp.base.v1",
    mcp_tools=("setup", "exec", "list_directory", "read_file", "write_file", "grade"),
    capability_flags=(
        "cov_func", "cov_line", "diff", "asan", "crash",
        "addrof", "fakeobj", "caged_read", "caged_write",
        "infoleak_binary", "infoleak_libc", "infoleak_stack",
        "arb_read", "arb_write", "pc_control", "ace",
    ),
    submission_kind="task",
    grader_kind="mcp_tool",
    grader_tool="grade",
)

# Generic binary-task interface, reserved for future variants
# CTF-shaped tasks. The grader for these typically reports just {crash, solve}
# (or similar smaller bitmap), not the V8 16-flag set. Different grader,
# different capability_flags, so it gets its own interface.
RL_MCP_BINARY_TASK = Interface(
    name="rl.mcp.binary_task.v1",
    description="Generic binary task (e.g., bountybench / CTF-shaped "
                "challenges). Reserved slot — capability_flags will be filled "
                "in when the first env of this kind lands. NOT used by V8 "
                "(see rl.mcp.v8_task.v1 instead).",
    base="rl.mcp.base.v1",
    mcp_tools=("setup", "exec", "list_directory", "read_file", "write_file", "grade"),
    capability_flags=(),  # placeholder; bountybench will fill these in
    submission_kind="task",
    grader_kind="mcp_tool",
    grader_tool="grade",
    notes="Reserved for bountybench / CTF-shaped envs.",
)






# ---------------------------------------------------------------------------
# Lifecycle (composite)
# ---------------------------------------------------------------------------

RL_MCP_LIFECYCLE = Interface(
    name="rl.mcp.lifecycle.v1",
    description="Full task lifecycle: detect → solve → patch in "
                "a single long-horizon episode with per-stage rewards.",
    base="rl.mcp.base.v1",
    mcp_tools=("setup", "exec", "list_files", "read_file", "edit_file",
               "reproduce", "verify", "submit_finding", "submit_patch",
               "submit_pov", "grade", "finish"),
    submission_kind="lifecycle",
    grader_kind="mcp_tool",
    grader_tool="grade",
    notes="Long-horizon.",
)



# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REGISTRY: dict[str, Interface] = {
    iface.name: iface
    for iface in (
        RL_MCP_BASE,
        RL_MCP_V8_TASK,
        RL_MCP_BINARY_TASK,
        RL_MCP_LIFECYCLE,
    )
}


def lookup(name: str) -> Interface | None:
    """Return the Interface for `name`, or None if unknown."""
    return REGISTRY.get(name)


def all_names() -> list[str]:
    """Sorted list of all known interface names."""
    return sorted(REGISTRY.keys())


def is_known(name: str) -> bool:
    return name in REGISTRY
