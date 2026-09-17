"""Check 5: integrity baseline of grading-critical files.

Each manifest may declare a list of container paths whose contents must
match a recorded sha256. This catches:
  - image rebuilds that silently change the grader binary;
  - in-container tampering by an agent (run before each episode in a
    longer-term hardened pipeline);
  - drift between adapter-generated manifests and the current image.

If `integrity_baseline.grader_paths` is empty, the check is UNVERIFIED
(by design — a manifest can opt into integrity later). If a path is
present in `grader_paths` but missing in the recorded `grader_hashes`
map, FAIL.

The hash compute runs *outside* the MCP loop: we exec
`docker run --rm <image> sha256sum <paths...>` directly. This avoids
spinning up the MCP server just to compute file hashes.
"""

from __future__ import annotations

import asyncio
import logging

from qed_swe_bench.contract.manifest import Manifest
from qed_swe_bench.validator.runner import CheckResult, CheckStatus

log = logging.getLogger(__name__)
CHECK_NAME = "integrity_posture"


async def check(manifest: Manifest, image_ref: str) -> CheckResult:
    baseline = manifest.integrity_baseline or {}
    paths = list(baseline.get("grader_paths") or [])
    hashes = dict(baseline.get("grader_hashes") or {})

    if not paths:
        return CheckResult(
            name=CHECK_NAME,
            status=CheckStatus.UNVERIFIED,
            message="manifest declares no integrity_baseline.grader_paths",
        )

    missing_hashes = [p for p in paths if p not in hashes]
    if missing_hashes:
        return CheckResult(
            name=CHECK_NAME,
            status=CheckStatus.FAIL,
            message=f"grader_paths missing recorded hash: {missing_hashes}",
        )

    try:
        actual = await _compute_sha256(image_ref, paths)
    except Exception as e:  # noqa: BLE001
        log.warning("integrity_posture: %s", e)
        return CheckResult(
            name=CHECK_NAME,
            status=CheckStatus.FAIL,
            message=f"could not compute hashes: {e}",
        )

    drift: dict[str, dict[str, str]] = {}
    for p, want in hashes.items():
        got = actual.get(p)
        if got is None:
            drift[p] = {"recorded": want, "actual": "<missing>"}
        elif got != want:
            drift[p] = {"recorded": want, "actual": got}

    if drift:
        return CheckResult(
            name=CHECK_NAME,
            status=CheckStatus.FAIL,
            message=f"{len(drift)} grader files drifted from baseline",
            details={"drift": drift},
        )

    return CheckResult(
        name=CHECK_NAME,
        status=CheckStatus.PASS,
        message=f"{len(paths)} grader files match baseline",
    )


async def _compute_sha256(image_ref: str, paths: list[str]) -> dict[str, str]:
    """Run sha256sum on `paths` inside a fresh container; return {path: sha256:...}."""
    # Defense in depth: a leading `-` in `image_ref` would be parsed by
    # docker as a flag. Manifest schema enforces this upstream, but the
    # validator can be invoked with `--image-ref` overrides too.
    from qed_swe_bench.runner.image_ref import assert_safe_image_ref
    assert_safe_image_ref(image_ref)
    cmd = [
        "docker", "run", "--rm", "--network", "none", "--entrypoint", "sha256sum",
        image_ref, "--", *paths,
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"sha256sum exit={proc.returncode}: {stderr.decode(errors='replace')[:200]}"
        )

    out: dict[str, str] = {}
    for line in stdout.decode().splitlines():
        # `sha256sum` format: "<hex>  <path>"
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2:
            continue
        digest, p = parts
        out[p] = f"sha256:{digest}"
    return out
