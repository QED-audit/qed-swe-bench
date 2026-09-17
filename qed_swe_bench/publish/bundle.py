"""Write a publishable bundle to `dist/<repo>/<revision>/`.

Layout (flat — see docs/decisions.md D-15):

    dist/<repo>/<revision>/
      runs.parquet                                  # one row per cell
      transcripts/<model_slug>/<env_id>/seed_<N>.jsonl.zst
      tool_calls/<model_slug>/<env_id>/seed_<N>.jsonl.zst
      grade_calls/<model_slug>/<env_id>/seed_<N>.jsonl.zst
      config_snapshot.yaml                          # representative pin

`README.md`, `manifest.json`, `audit.json` are written by the CLI
orchestrator using `publish.card` after this module finishes.

`pyarrow` and `zstandard` are imported lazily — they live in the
`[publish]` optional extra so EC2 runners don't pay for them.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from qed_swe_bench.publish.selection import CellRecord, model_slug
from qed_swe_bench.runner.capabilities import CAPABILITY_FLAGS

_SIDECAR_KINDS = (
    # (source filename in run_dir, destination subdir in bundle)
    ("transcript.jsonl", "transcripts"),
    ("tool_calls.jsonl", "tool_calls"),
    ("grade_calls.jsonl", "grade_calls"),
)


@dataclass(frozen=True)
class BundleStats:
    """Result of writing a bundle. Used by the CLI dry-run / summary."""

    n_cells: int
    n_succeeded: int
    n_model_failed: int
    sidecar_bytes: int
    parquet_path: Path
    config_snapshot_sha: str | None


def _require_optional_deps() -> None:
    try:
        import pyarrow  # noqa: F401
        import zstandard  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "publish bundle requires the [publish] extras. Install with: "
            "pip install -e '.[publish]' (adds pyarrow + zstandard + "
            "huggingface_hub)."
        ) from exc


def _row_for_parquet(cell: CellRecord, sidecar_paths: dict[str, str | None]) -> dict:
    """Flatten a CellRecord into a dict of pyarrow-friendly columns."""
    row: dict[str, object | None] = {
        # Identity
        "model": cell.model,
        "env_id": cell.env_id,
        "seed": cell.seed,
        "run_id": cell.run_id,
        "benchmark_id": cell.benchmark_id,
        # Outcome
        "status": cell.status,
        "score": cell.score,
        "turns_used": cell.turns_used,
        "runtime_s": cell.runtime_s,
        "exit_reason": cell.exit_reason,
        # Cost
        "cost_usd": cell.cost_usd,
        "cost_source": cell.cost_source,
        "tokens_in": cell.tokens_in,
        "tokens_out": cell.tokens_out,
        "tokens_cache_read": cell.tokens_cache_read,
        "tokens_cache_creation": cell.tokens_cache_creation,
        "tokens_reasoning": cell.tokens_reasoning,
        # Provenance
        "image_ref": cell.image_ref,
        "image_digest": cell.image_digest,
        "git_sha": cell.git_sha,
        "served_model": cell.served_model,
        "weighted_tokens_used": cell.weighted_tokens_used,
        "peak_per_turn_context": cell.peak_per_turn_context,
        "started_at": cell.started_at,
        "finished_at": cell.finished_at,
        # Sidecar paths (relative to dataset root) — None if missing
        "transcript_path": sidecar_paths.get("transcript.jsonl"),
        "tool_calls_path": sidecar_paths.get("tool_calls.jsonl"),
        "grade_calls_path": sidecar_paths.get("grade_calls.jsonl"),
    }
    # Capability columns: explicit per-flag boolean. Unknown caps in the
    # cell's bitmap (e.g. future grader additions) are not emitted as
    # columns here — the parquet schema is pinned to the 16 known flags
    # so the dataset shape stays stable across revisions.
    for cap in CAPABILITY_FLAGS:
        row[f"caps_{cap}"] = bool(cell.capabilities.get(cap, False))
    return row


def _compress_sidecar(src: Path, dst: Path, *, level: int = 10) -> int:
    """Compress src JSONL → dst .jsonl.zst. Returns bytes written.

    Reads the full file into memory; sidecars are sub-megabyte each, so
    streaming would be premature optimization.
    """
    import zstandard as zstd  # type: ignore[import-not-found]

    dst.parent.mkdir(parents=True, exist_ok=True)
    cctx = zstd.ZstdCompressor(level=level)
    data = src.read_bytes()
    out = cctx.compress(data)
    dst.write_bytes(out)
    return len(out)


def _bundle_sidecars(cell: CellRecord, dest_root: Path) -> tuple[dict[str, str | None], int]:
    """Copy + compress one cell's sidecars; return ({src_name: rel_path}, bytes)."""
    slug = model_slug(cell.model)
    fname = f"seed_{cell.seed}.jsonl.zst"
    paths: dict[str, str | None] = {}
    total = 0
    for src_name, subdir in _SIDECAR_KINDS:
        src = cell.run_dir / src_name
        if not src.is_file():
            paths[src_name] = None
            continue
        rel = Path(subdir) / slug / cell.env_id / fname
        dst = dest_root / rel
        total += _compress_sidecar(src, dst)
        # POSIX-style for the parquet column so consumers on any OS can
        # join the relative path against the dataset root.
        paths[src_name] = rel.as_posix()
    return paths, total


def _write_parquet(rows: list[dict], dest: Path) -> None:
    import pyarrow as pa  # type: ignore[import-not-found]
    import pyarrow.parquet as pq  # type: ignore[import-not-found]

    table = pa.Table.from_pylist(rows)
    pq.write_table(table, dest, compression="zstd")


def _copy_config_snapshot(cells: Iterable[CellRecord], dest_root: Path) -> str | None:
    """Copy the first cell's config_snapshot.yaml and return its sha7.

    All cells in a bundle should share methodology (same v8.yaml at
    publish time); the first one's snapshot is representative. Returns
    None if no cell had a snapshot file (shouldn't happen for current
    runs but kept resilient).
    """
    import hashlib

    for cell in cells:
        src = cell.run_dir / "config_snapshot.yaml"
        if src.is_file():
            shutil.copy2(src, dest_root / "config_snapshot.yaml")
            return hashlib.sha256(src.read_bytes()).hexdigest()[:7]
    return None


def build(cells: list[CellRecord], dest_root: Path) -> BundleStats:
    """Materialize a flat bundle at `dest_root`.

    Wipes any prior contents at `dest_root` so the build is idempotent.
    Caller is responsible for `card.write_card` and `card.write_manifest`
    after this returns.
    """
    _require_optional_deps()

    if dest_root.exists():
        shutil.rmtree(dest_root)
    dest_root.mkdir(parents=True)

    parquet_rows: list[dict] = []
    total_sidecar_bytes = 0
    for cell in cells:
        sidecar_paths, n_bytes = _bundle_sidecars(cell, dest_root)
        total_sidecar_bytes += n_bytes
        parquet_rows.append(_row_for_parquet(cell, sidecar_paths))

    parquet_path = dest_root / "runs.parquet"
    _write_parquet(parquet_rows, parquet_path)

    config_sha = _copy_config_snapshot(cells, dest_root)

    n_succ = sum(1 for c in cells if c.status == "succeeded")
    n_fail = sum(1 for c in cells if c.status == "model_failed")
    return BundleStats(
        n_cells=len(cells),
        n_succeeded=n_succ,
        n_model_failed=n_fail,
        sidecar_bytes=total_sidecar_bytes,
        parquet_path=parquet_path,
        config_snapshot_sha=config_sha,
    )
