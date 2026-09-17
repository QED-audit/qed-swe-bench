"""Validator runner: aggregate the five checks, summarize, persist status.

Each check is a callable returning a CheckResult. The runner sequences
them, short-circuits when an earlier check makes a later check
meaningless (e.g., manifest_schema FAIL → skip mcp_contract because we
have no manifest to compare against), and returns a ValidationReport.

The CLI translates ValidationReport → table output and updates
rlenv_images.validation_status.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

log = logging.getLogger(__name__)


class CheckStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"            # earlier check failed; skipping for clarity
    UNVERIFIED = "unverified"  # the check was attempted but couldn't conclude
                              # (e.g. manifest declares no PoV, so reproduction
                              # is "unverified" rather than fail)


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: CheckStatus
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationReport:
    env_id: str
    image_ref: str
    results: tuple[CheckResult, ...]

    @property
    def overall_status(self) -> str:
        """Roll-up status used as `rlenv_images.validation_status`.

        Mapping:
          - any FAIL                       → 'fail'
          - all SKIP                       → 'unvalidated'
          - any SKIP or UNVERIFIED present → 'pass_unverified'
                                              (manifest sound, but at least
                                              one downstream check did not
                                              actually confirm anything)
          - everything PASS                → 'pass_verified'
        """
        statuses = {r.status for r in self.results}
        if CheckStatus.FAIL in statuses:
            return "fail"
        if statuses == {CheckStatus.SKIP}:
            return "unvalidated"
        if CheckStatus.UNVERIFIED in statuses or CheckStatus.SKIP in statuses:
            return "pass_unverified"
        return "pass_verified"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

# Container-needing checks 2-5. Listed once so the SKIP fan-out paths
# (manifest_schema FAIL, --skip-container) don't duplicate the names.
_DOWNSTREAM_CHECKS = (
    "mcp_contract",
    "target_starts",
    "known_pov_reproduces",
    "integrity_posture",
)


def _skip_downstream(message: str) -> list[CheckResult]:
    return [
        CheckResult(name=n, status=CheckStatus.SKIP, message=message)
        for n in _DOWNSTREAM_CHECKS
    ]


async def run_all(
    manifest_path: str | None,
    env_id: str | None,
    image_ref: str | None,
    *,
    skip_container_checks: bool = False,
) -> ValidationReport:
    """Run all five checks against an env.

    `manifest_path` is the path to the manifest file. When None and
    `env_id` is given, the runner asks the catalog for the manifest path.

    `skip_container_checks` is for CI: skip checks 2-5 and only verify the
    static manifest. Useful for tier-1 unit testing without Docker.
    """
    # Local imports to avoid pulling Docker / MCP imports for unit tests
    # that only need check 1.
    from qed_swe_bench.validator.checks import (
        integrity_posture,
        known_pov_reproduces,
        manifest_schema,
        mcp_contract,
        target_starts,
    )

    results: list[CheckResult] = []

    # 1. manifest_schema (always runs; everything else needs the result)
    schema_result, manifest = manifest_schema.check(manifest_path)
    results.append(schema_result)

    if schema_result.status == CheckStatus.FAIL or manifest is None:
        # Cannot run remaining checks meaningfully.
        results.extend(_skip_downstream("manifest_schema failed; skipped"))
        return ValidationReport(
            env_id=env_id or (manifest.env_id if manifest else "?"),
            image_ref=image_ref or (manifest.image_ref if manifest else "?"),
            results=tuple(results),
        )

    resolved_image = image_ref or manifest.image_ref

    if skip_container_checks:
        results.extend(_skip_downstream("skipped (skip_container_checks=True)"))
    else:
        results.append(await mcp_contract.check(manifest, resolved_image))
        results.append(await target_starts.check(manifest, resolved_image))
        results.append(await known_pov_reproduces.check(manifest, resolved_image))
        results.append(await integrity_posture.check(manifest, resolved_image))

    return ValidationReport(
        env_id=manifest.env_id,
        image_ref=resolved_image,
        results=tuple(results),
    )
