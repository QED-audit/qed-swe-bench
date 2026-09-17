"""Check 1: manifest validates against the schema and the interface registry.

Pure / no Docker. Loads the manifest YAML/JSON, runs JSON Schema +
interface-consistency validation, returns the parsed Manifest on PASS.
"""

from __future__ import annotations

from pathlib import Path

from qed_swe_bench.contract.manifest import (
    Manifest,
    ManifestError,
    load,
)
from qed_swe_bench.validator.runner import CheckResult, CheckStatus

CHECK_NAME = "manifest_schema"


def check(manifest_path: str | Path | None) -> tuple[CheckResult, Manifest | None]:
    """Returns (result, manifest_or_None).

    The Manifest is returned alongside the result so the orchestrator can
    pass it to subsequent checks without re-loading.
    """
    if manifest_path is None:
        return (
            CheckResult(
                name=CHECK_NAME,
                status=CheckStatus.FAIL,
                message="no manifest path provided",
            ),
            None,
        )

    p = Path(manifest_path)
    if not p.exists():
        return (
            CheckResult(
                name=CHECK_NAME,
                status=CheckStatus.FAIL,
                message=f"manifest file not found: {p}",
            ),
            None,
        )

    try:
        m = load(p)
    except ManifestError as e:
        return (
            CheckResult(
                name=CHECK_NAME,
                status=CheckStatus.FAIL,
                message=str(e),
            ),
            None,
        )

    return (
        CheckResult(
            name=CHECK_NAME,
            status=CheckStatus.PASS,
            message=f"manifest valid; interface={m.interface}",
            details={"interface": m.interface, "task_type": m.task_type},
        ),
        m,
    )
