"""Catalog → manifest lookup for the rl_mcp_v1 dispatch path.

The orchestrator asks `load_env_manifest(env_id)` once per tuple to decide
whether the post-episode `cli_oneshot` grader should run. Two distinct
return contracts:

  - **None** — no manifest is expected for this env. The orchestrator
    falls through to the legacy in-MCP grading path. Returned when:
      * the env isn't in the catalog (legacy V8 envs predate the
        manifest contract and grade via the in-MCP `grade` tool); or
      * the env has no `manifest_path` in its catalog metadata.

  - **Manifest** — manifest expected and loaded successfully.

  - **raises EnvManifestLoadError** — manifest expected (env is
    registered with a manifest_path) but loading failed. The
    orchestrator should treat this as an infra failure, NOT silently
    fall back to in-MCP grading. A malformed manifest changes the
    benchmark's grading contract; running anyway would record runs
    against a different grader than the catalog promised.

The local import of `qed_swe_bench.catalog` is load-bearing: the catalog
imports from `runner.image_ref`, which would create a circular import at
module-load time if we promoted this to a top-level import.
"""

from __future__ import annotations

import logging

from qed_swe_bench.contract.manifest import Manifest, ManifestError
from qed_swe_bench.contract.manifest import load as load_manifest

log = logging.getLogger(__name__)


class EnvManifestLoadError(RuntimeError):
    """Raised when a registered env's manifest can't be loaded.

    Distinct from `ManifestError` (which is the schema-level error from
    the contract module): this signals a catalog ↔ manifest-on-disk
    inconsistency that should fail the run rather than silently
    downgrade the grading contract.
    """


def load_env_manifest(env_id: str) -> Manifest | None:
    """Look up the env's manifest via the catalog.

    Returns None when no manifest is expected (env not in catalog, or
    catalog row has no manifest_path). Raises EnvManifestLoadError when
    a manifest_path IS declared but the file fails to load — see module
    docstring for the rationale (silent fallback would change the
    benchmark's grading contract).
    """
    # Local import to avoid a circular dep at module-import time
    # (catalog imports runner.image_ref → runner imports catalog).
    from qed_swe_bench.catalog import get_env

    row = get_env(env_id)
    if row is None:
        return None
    metadata = row.get("metadata") or {}
    if not isinstance(metadata, dict):
        return None
    manifest_path = metadata.get("manifest_path")
    if not manifest_path:
        return None
    try:
        return load_manifest(manifest_path)
    except (ManifestError, OSError) as exc:
        log.error(
            "env_id=%s has manifest_path=%s but load failed: %s",
            env_id, manifest_path, exc,
        )
        raise EnvManifestLoadError(
            f"env_id={env_id!r} is registered with manifest_path={manifest_path!r} "
            f"but the manifest failed to load: {exc}. "
            f"Refusing to silently fall back to in-MCP grading — that would "
            f"silently change the benchmark's grading contract."
        ) from exc
