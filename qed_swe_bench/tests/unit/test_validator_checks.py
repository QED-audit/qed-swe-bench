"""Validator: per-check unit tests + report rollup.

The container-needing checks (mcp_contract, target_starts,
known_pov_reproduces, integrity_posture) get unit-tested only via the
SKIP path here. Their happy-path coverage is the integration tier.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from qed_swe_bench.validator import run_all
from qed_swe_bench.validator.checks import manifest_schema
from qed_swe_bench.validator.runner import CheckStatus

FIXTURES = Path(__file__).parent.parent / "golden" / "manifests"


def test_check_manifest_schema_passes_on_valid() -> None:
    result, manifest = manifest_schema.check(FIXTURES / "v8_e25.yaml")
    assert result.status == CheckStatus.PASS
    assert manifest is not None
    assert manifest.env_id == "v8-e25"


def test_check_manifest_schema_fails_on_missing_path() -> None:
    result, manifest = manifest_schema.check(FIXTURES / "does_not_exist.yaml")
    assert result.status == CheckStatus.FAIL
    assert manifest is None
    assert "not found" in result.message


def test_check_manifest_schema_fails_on_unknown_interface() -> None:
    result, manifest = manifest_schema.check(FIXTURES / "invalid_unknown_interface.yaml")
    assert result.status == CheckStatus.FAIL
    assert manifest is None


def test_check_manifest_schema_fails_on_missing_required_field() -> None:
    result, manifest = manifest_schema.check(FIXTURES / "invalid_missing_required.yaml")
    assert result.status == CheckStatus.FAIL
    assert manifest is None


@pytest.mark.asyncio
async def test_run_all_skip_container_only_runs_check_1() -> None:
    """skip_container_checks=True is the CI-friendly path."""
    report = await run_all(
        manifest_path=str(FIXTURES / "v8_e25.yaml"),
        env_id=None,
        image_ref=None,
        skip_container_checks=True,
    )
    assert report.env_id == "v8-e25"
    assert len(report.results) == 5
    by_name = {r.name: r for r in report.results}
    assert by_name["manifest_schema"].status == CheckStatus.PASS
    for skipped in ("mcp_contract", "target_starts",
                    "known_pov_reproduces", "integrity_posture"):
        assert by_name[skipped].status == CheckStatus.SKIP

    # 1 PASS + 4 SKIP → 'pass_unverified' (manifest is sound but downstream
    # checks didn't actually run, so we cannot promote to 'pass_verified').
    assert report.overall_status == "pass_unverified"


@pytest.mark.asyncio
async def test_run_all_short_circuits_on_schema_failure() -> None:
    """If check 1 fails, checks 2-5 should be SKIP and the rollup 'fail'."""
    report = await run_all(
        manifest_path=str(FIXTURES / "invalid_unknown_interface.yaml"),
        env_id="bogus-env",
        image_ref=None,
        skip_container_checks=False,
    )
    by_name = {r.name: r for r in report.results}
    assert by_name["manifest_schema"].status == CheckStatus.FAIL
    for downstream in ("mcp_contract", "target_starts",
                       "known_pov_reproduces", "integrity_posture"):
        assert by_name[downstream].status == CheckStatus.SKIP
    assert report.overall_status == "fail"


@pytest.mark.asyncio
async def test_integrity_posture_rejects_flag_like_image_ref() -> None:
    """Defense-in-depth: even if a caller bypasses the manifest schema and
    hands `_compute_sha256` a flag-shaped image_ref, the check refuses to
    invoke docker."""
    from qed_swe_bench.validator.checks.integrity_posture import _compute_sha256
    with pytest.raises(ValueError, match="flag-like image_ref"):
        await _compute_sha256("-v=/host:/inside", ["/some/path"])


@pytest.mark.asyncio
async def test_run_all_no_manifest_path_fails() -> None:
    report = await run_all(
        manifest_path=None,
        env_id="x",
        image_ref=None,
        skip_container_checks=True,
    )
    assert report.results[0].status == CheckStatus.FAIL
    assert "no manifest path" in report.results[0].message
