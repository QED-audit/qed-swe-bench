"""Validator suite for registered envs.

Five checks run against a registered env at registration time and again
before each benchmark batch:

  1. manifest_schema      — manifest.{yaml,json} validates and matches the
                            registered interface
  2. mcp_contract         — listed tools match the manifest's
                            episode_tools (running container)
  3. target_starts        — target binary inside the container is
                            executable / responds to a smoke command
  4. known_pov_reproduces — recorded PoV in the manifest still triggers
                            the documented capability flag(s)
  5. integrity_posture    — files listed under integrity_baseline.grader_paths
                            hash to the recorded sha256

Result types and the runner live here; per-check implementations live in
`qed_swe_bench/validator/checks/`.
"""

from qed_swe_bench.validator.runner import (
    CheckResult,
    CheckStatus,
    ValidationReport,
    run_all,
)

__all__ = ["CheckResult", "CheckStatus", "ValidationReport", "run_all"]
