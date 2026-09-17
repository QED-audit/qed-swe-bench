"""Resolve a benchmark-config image reference to an immutable digest.

The config schema accepts three image-ref forms:

  1. Registry digest:  302524490333.dkr.ecr.us-east-1.amazonaws.com/swe-bench/v8-r2:e41@sha256:abc...
     → used as-is; we still inspect to confirm it's loaded locally after pull.

  2. Registry tag:     302524490333.dkr.ecr.us-east-1.amazonaws.com/swe-bench/v8-r2:e41:v1
     → docker pull, then docker inspect to get the canonical digest.

  3. Local image:      local/sample-stack-bof:latest  OR  sample-stack-bof:latest
     → docker inspect only. Errors if not loaded.

Any other shape (e.g. './path/to/Dockerfile') is rejected with a clear error
pointing the user at "build first, then reference local/...".

In every case we return both the original ref (for audit) and the resolved
sha256 digest (the durable, immutable identifier we record in runs.image_digest
and pass to `docker run`).
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
import re
import subprocess
from dataclasses import dataclass

log = logging.getLogger(__name__)

_SHA256_RE = re.compile(r"sha256:[a-f0-9]{64}")


_PULL_ATTEMPTS_DEFAULT = 5
_PULL_BACKOFF_BASE_S = 4.0
_PULL_BACKOFF_CAP_S = 60.0


def _pull_attempts() -> int:
    raw = os.environ.get("QED_SWE_BENCH_PULL_ATTEMPTS", "").strip()
    if not raw:
        return _PULL_ATTEMPTS_DEFAULT
    try:
        value = int(raw)
    except ValueError:
        return _PULL_ATTEMPTS_DEFAULT
    return max(1, value)


def _pull_backoff(attempt: int) -> float:
    window = min(_PULL_BACKOFF_CAP_S, _PULL_BACKOFF_BASE_S * (2 ** (attempt - 1)))
    return window / 2 + random.uniform(0, window / 2)


def rewrite_ref(ref: str) -> str:
    """Apply QED_SWE_BENCH_IMAGE_REWRITE, a comma-separated list of from=to prefixes."""
    spec = os.environ.get("QED_SWE_BENCH_IMAGE_REWRITE", "").strip()
    if not spec:
        return ref
    for rule in spec.split(","):
        rule = rule.strip()
        if not rule or "=" not in rule:
            continue
        old_prefix, new_prefix = rule.split("=", 1)
        old_prefix, new_prefix = old_prefix.strip(), new_prefix.strip()
        if old_prefix and ref.startswith(old_prefix):
            rewritten = new_prefix + ref[len(old_prefix):]
            log.info("image ref rewritten: %s -> %s", ref, rewritten)
            return rewritten
    return ref


def assert_safe_image_ref(image_ref: str) -> None:
    """Defense in depth — refuse refs shaped like a docker-run flag.

    Manifest schema's `image.ref` pattern enforces leading-alphanumeric,
    but several callers also accept image_refs from runtime overrides
    (e.g., `validate-image --image-ref`). Any docker-run-spawning code
    path should call this before subprocess'ing.
    """
    if not image_ref or image_ref.startswith("-"):
        raise ValueError(
            f"refusing to invoke docker with flag-like image_ref: {image_ref!r}"
        )


@dataclass(frozen=True)
class ResolvedImage:
    """Result of resolving a config image_ref to a usable docker handle."""

    image_ref: str          # original string from config
    image_digest: str       # sha256:... (always set on success)
    pulled: bool            # whether we ran `docker pull`


class ImageRefError(Exception):
    """The image ref couldn't be resolved into a runnable image."""


def is_digest(ref: str) -> bool:
    return "@sha256:" in ref


def is_local(ref: str) -> bool:
    """Local-only refs: anything without a registry hostname.

    Docker's hostname rule: the part before the first `/` is treated as a
    registry hostname iff it contains a `.` (DNS dot), a `:` (port), or is
    exactly `localhost`. Otherwise the first segment is a username/org and
    the registry defaults to docker.io (which we still treat as "local" for
    our purposes — i.e. don't auto-pull, expect the image to be loaded).

    Examples:
      local/sample-stack-bof:latest  → local (head 'local' has no '.' or ':')
      sample-stack-bof:latest        → local (no '/' at all; ':' is the tag)
      foo                            → local
      ghcr.io/x/y:v1                 → registry (head 'ghcr.io' has '.')
      registry.local:5000/foo:bar    → registry (head has both)
    """
    if is_digest(ref):
        return False
    if "/" not in ref:
        return True  # bare name(:tag) is local
    head = ref.split("/", 1)[0]
    return not ("." in head or ":" in head or head == "localhost")


def is_buildable_path(ref: str) -> bool:
    """References that look like a filesystem path (rejected — build first)."""
    return ref.startswith(("./", "../", "/")) or ref.endswith("/Dockerfile")


def resolve(ref: str, *, docker_bin: str = "docker") -> ResolvedImage:
    """Resolve `ref` to an immutable digest, pulling if needed.

    Raises ImageRefError on any failure (network, missing image, malformed ref).
    """

    ref = rewrite_ref(ref)
    if is_buildable_path(ref):
        raise ImageRefError(
            f"image ref {ref!r} looks like a filesystem path. "
            "Build the image first (docker build -t local/<name>:<tag> ...) "
            "and reference it as `local/<name>:<tag>` in the config."
        )

    if is_digest(ref):
        # `docker pull <ref>` pulls by digest deterministically; the registry
        # manifest digest in `ref` and the local image `Id` are different
        # SHAs, so after the pull we have to inspect the loaded image to get
        # the Id `docker run` accepts. Using the bare manifest digest from
        # the ref directly would 404 at run time (`docker run sha256:<MD>`
        # only resolves a local `Id`, not a registry manifest digest).
        pulled = False
        if not _image_present(ref, docker_bin):
            _docker_pull(ref, docker_bin)
            pulled = True
        digest = _inspect_digest(ref, docker_bin)
        return ResolvedImage(image_ref=ref, image_digest=digest, pulled=pulled)

    if is_local(ref):
        # No pull. Must already be loaded.
        digest = _inspect_digest(ref, docker_bin)
        return ResolvedImage(image_ref=ref, image_digest=digest, pulled=False)

    # Registry tag: pull then inspect. Skip the pull if the image is already
    # cached locally — the pinned-digest contract is satisfied by inspecting
    # whatever's loaded, and unconditionally pulling on every run breaks
    # offline / expired-token runs (ECR auth tokens lapse every ~12h).
    # `QED_SWE_BENCH_FORCE_PULL=1` opts back into the always-pull behavior
    # for the rare case where you want to verify a tag is current against
    # the registry.
    pulled = False
    if os.environ.get("QED_SWE_BENCH_FORCE_PULL") or not _image_present(ref, docker_bin):
        _docker_pull(ref, docker_bin)
        pulled = True
    digest = _inspect_digest(ref, docker_bin)
    return ResolvedImage(image_ref=ref, image_digest=digest, pulled=pulled)


# --------------------------------------------------------------------------
# subprocess wrappers (also imported by tests via monkeypatch)
# --------------------------------------------------------------------------


def _docker_pull(ref: str, docker_bin: str) -> None:
    attempts = _pull_attempts()
    detail = ""
    for attempt in range(1, attempts + 1):
        if log.isEnabledFor(logging.INFO):
            log.info("docker pull %s (attempt %d/%d)", ref, attempt, attempts)
            proc = subprocess.run([docker_bin, "pull", ref])
            detail = f"exit {proc.returncode}; see docker output above"
        else:
            proc = subprocess.run(
                [docker_bin, "pull", "--quiet", ref],
                capture_output=True,
                text=True,
            )
            detail = proc.stderr.strip()
        if proc.returncode == 0:
            return
        if attempt < attempts:
            delay = _pull_backoff(attempt)
            log.warning(
                "docker pull %s failed (attempt %d/%d), retrying in %.1fs: %s",
                ref,
                attempt,
                attempts,
                delay,
                detail,
            )
            time.sleep(delay)
    raise ImageRefError(
        f"docker pull {ref!r} failed after {attempts} attempts: {detail}"
    )


def _image_present(ref: str, docker_bin: str) -> bool:
    proc = subprocess.run(
        [docker_bin, "image", "inspect", ref],
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0


def _inspect_digest(ref: str, docker_bin: str) -> str:
    """Return the sha256 digest of the image identified by `ref`.

    Prefers `Id` (the local content hash, immutable). For images pulled from
    a registry, we could also use `RepoDigests[0]` to get the registry-side
    digest, but `Id` is sufficient for `docker run` and is always present.
    """
    proc = subprocess.run(
        [docker_bin, "image", "inspect", ref],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise ImageRefError(
            f"docker inspect {ref!r} failed: {proc.stderr.strip()}. "
            "Image not loaded? For a local ref, build with "
            "`docker build -t {ref} ...` first."
        )
    try:
        info = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise ImageRefError(f"docker inspect output not JSON: {exc}") from exc
    if not info or not isinstance(info, list):
        raise ImageRefError(f"docker inspect {ref!r} returned no info")

    image_id = info[0].get("Id", "")
    match = _SHA256_RE.search(image_id)
    if not match:
        raise ImageRefError(f"docker inspect {ref!r} missing valid Id sha256")
    return match.group(0)


def _extract_digest(ref: str) -> str:
    """Pull the sha256:... portion out of a digest-pinned ref."""
    match = _SHA256_RE.search(ref)
    if not match:
        raise ImageRefError(f"ref {ref!r} contains @sha256: but no valid digest")
    return match.group(0)
