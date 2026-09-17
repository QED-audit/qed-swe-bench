"""Default init prompt for the agent loop.

The container's MCP `setup()` carries all bug-specific framing (target,
environment paths, capabilities, grader builtins, patch diff). The init
prompt is just the first user turn that points the agent at it.

Resolution order (handled by `run_benchmark`):
  1. `bench.init_prompt` / `bench.init_prompt_path` from YAML
  2. Caller override passed to `run_benchmark(init_prompt=...)`
  3. `default_init_prompt()` below

An optional `bench.init_prompt_hint` (or `init_prompt_hint_path`) is
appended after the resolved init prompt — the cheap experimental knob.
"""

from __future__ import annotations


def default_init_prompt() -> str:
    # `setup()` and `grade(...)` are qed_swe_bench's fixed contract with the
    # benchmark target — every env exposes them. Everything else (workspace
    # paths, target description, what to pass to grade, capability list) is
    # env-specific and lives in setup()'s response, so this prompt stays
    # benchmark-agnostic.
    return (
        "Use setup() to learn about the target. Then explore it, develop "
        "your solution, and call grade(...) to evaluate progress."
    )
