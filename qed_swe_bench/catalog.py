"""Env catalog: register / look up rlenv_images.

`register-dir <bench-v8/bugs/>` walks a directory of bench-v8 bug
sub-folders and registers each as an env in the catalog. Reads
bugs/<id>/task.json for metadata (bug_id, crev, etc.).

`register-image --image <ref> --bug-id <id>` adds a single env directly.

Registration is non-destructive: re-running upserts (replace existing row).
"""

from __future__ import annotations

import contextlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qed_swe_bench.contract import interfaces as iface_registry
from qed_swe_bench.db.schema import connect, init_db
from qed_swe_bench.runner.image_ref import ImageRefError, resolve

log = logging.getLogger(__name__)


# Hand-curated default capability_class for V8 bugs (best-effort guess from
# subsystem; can be overridden per-env on register).
V8_CAPABILITY_CLASS_BY_SUBSYSTEM = {
    "JS":   "renderer_rce",
    "Wasm": "renderer_rce",
    "Both": "renderer_rce",
}


@dataclass(frozen=True)
class EnvRegistration:
    env_id: str
    image_ref: str
    interface: str
    task_type: str
    project: str | None
    bug_id: str | None
    capability_class: str | None
    expected_capabilities: list[str] | None
    metadata: dict[str, Any]


def _read_task_json(bug_dir: Path) -> dict[str, Any]:
    p = bug_dir / "task.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        log.warning("malformed task.json in %s", bug_dir)
        return {}


def from_vragent_dir(
    bug_dir: Path,
    *,
    image_ref: str | None = None,
    interface: str = "rl.mcp.v8_task.v1",
) -> EnvRegistration | None:
    """Build an EnvRegistration record from a bench-v8 bugs/<id>/ directory.

    If `image_ref` isn't passed, falls back to `local/v8-<bug-id-lowercased>:latest`
    by convention.
    """
    if not bug_dir.is_dir():
        return None
    task = _read_task_json(bug_dir)
    bug_id = task.get("bug_id") or bug_dir.name
    env_id = f"v8-{bug_id.lower()}"
    image = image_ref or f"local/v8-{bug_id.lower()}:latest"

    iface = iface_registry.lookup(interface)
    expected_capabilities = list(iface.capability_flags) if iface else None

    metadata: dict[str, Any] = {
        "source": "bench-v8",
        "tgt_commit": task.get("tgt_commit"),
        "tgt_crev": task.get("tgt_crev"),
        "fix_crev": task.get("fix_crev"),
        "source_date_epoch": task.get("source_date_epoch"),
        "cve": task.get("cve"),
        "crbug": task.get("crbug"),
        "problem_types": task.get("problem_types"),
    }
    metadata = {k: v for k, v in metadata.items() if v is not None}

    return EnvRegistration(
        env_id=env_id,
        image_ref=image,
        interface=interface,
        task_type="binary_task",
        project="v8",
        bug_id=bug_id,
        capability_class="renderer_rce",  # default for V8; refine per-bug later
        expected_capabilities=expected_capabilities,
        metadata=metadata,
    )


def upsert(reg: EnvRegistration, *, db_path: Path | None = None) -> bool:
    """Insert or replace a row. Returns True if inserted, False if updated."""
    init_db(db_path)
    with connect(db_path) as con:
        existed = con.execute(
            "SELECT 1 FROM rlenv_images WHERE env_id = ?", (reg.env_id,)
        ).fetchone() is not None

        # Try to resolve image to digest now; tolerate failure (image may not
        # be loaded yet — that's fine for registration).
        image_digest: str | None = None
        try:
            resolved = resolve(reg.image_ref)
            image_digest = resolved.image_digest
        except ImageRefError:
            pass

        con.execute(
            """
            INSERT OR REPLACE INTO rlenv_images (
                env_id, image_ref, image_digest, interface, task_type,
                project, bug_id, capability_class, expected_capabilities,
                metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                reg.env_id,
                reg.image_ref,
                image_digest,
                reg.interface,
                reg.task_type,
                reg.project,
                reg.bug_id,
                reg.capability_class,
                json.dumps(reg.expected_capabilities) if reg.expected_capabilities else None,
                json.dumps(reg.metadata) if reg.metadata else None,
            ),
        )
        return not existed


def register_dir(
    bugs_dir: Path,
    *,
    db_path: Path | None = None,
    interface: str = "rl.mcp.v8_task.v1",
) -> dict[str, int]:
    """Walk `bugs_dir` (e.g. bench-v8/bugs/) and register each subdir as an env."""
    init_db(db_path)
    histogram: dict[str, int] = {"inserted": 0, "updated": 0, "unparseable": 0}
    if not bugs_dir.is_dir():
        raise ValueError(f"bugs dir not found: {bugs_dir}")
    for sub in sorted(bugs_dir.iterdir()):
        if not sub.is_dir():
            continue
        reg = from_vragent_dir(sub, interface=interface)
        if reg is None:
            histogram["unparseable"] += 1
            continue
        was_new = upsert(reg, db_path=db_path)
        histogram["inserted" if was_new else "updated"] += 1
    return histogram


def get_env(env_id: str, *, db_path: Path | None = None) -> dict[str, Any] | None:
    """Return one rlenv_images row by env_id, with JSON columns parsed.

    Returns None if no such env is registered. Used by the orchestrator to
    discover whether an env has a manifest (and therefore a `cli_oneshot`
    grading path) before running an episode against it.
    """
    init_db(db_path)
    with connect(db_path) as con:
        row = con.execute(
            "SELECT * FROM rlenv_images WHERE env_id = ?", (env_id,)
        ).fetchone()
    if row is None:
        return None
    d = dict(row)
    for col in ("expected_capabilities", "metadata"):
        if d.get(col):
            with contextlib.suppress(json.JSONDecodeError):
                d[col] = json.loads(d[col])
    return d


def list_envs(db_path: Path | None = None) -> list[dict[str, Any]]:
    """Return all registered envs, parsed JSON columns expanded."""
    init_db(db_path)
    with connect(db_path) as con:
        rows = con.execute(
            "SELECT * FROM rlenv_images ORDER BY env_id"
        ).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        d = dict(r)
        for col in ("expected_capabilities", "metadata"):
            if d.get(col):
                try:
                    d[col] = json.loads(d[col])
                except json.JSONDecodeError:
                    pass
        out.append(d)
    return out
