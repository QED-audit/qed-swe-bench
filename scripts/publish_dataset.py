#!/usr/bin/env python3
"""publish_dataset.py — bundle a canonical run bucket into a HuggingFace dataset.

Reads `runs/v8/<model>-<ip>/` (output of `scripts/curate_canonical_runs.py`),
produces an HF-shaped dataset under `dist/<repo-id>/<revision>/`, and
optionally pushes to HuggingFace.

Output layout:

    dist/<repo-id>/<revision>/
    ├── README.md                                  ← auto-generated dataset card
    ├── runs.parquet                               ← 1 row per (model, env, seed)
    ├── transcripts/<env_id>/seed_<N>.jsonl.zst
    ├── tool_calls/<env_id>/seed_<N>.jsonl.zst
    ├── grade_calls/<env_id>/seed_<N>.jsonl.zst
    ├── CANONICAL_MANIFEST.md                      ← copied from bucket
    ├── CANONICAL_MANIFEST.tsv                     ← copied
    ├── audit.json                                 ← qed_swe_bench audit --format json
    ├── config_snapshot.yaml                       ← representative pin (from one cell)
    └── manifest.json                              ← top-level versioning info

CLI:

    # Local-only (no network)
    python scripts/publish_dataset.py minimax-m2.7-44.211.48.34

    # All buckets in runs/v8/
    python scripts/publish_dataset.py --all

    # Override repo-id (default: qed_swe_bench/v8)
    python scripts/publish_dataset.py minimax-m2.7-44.211.48.34 \\
        --repo-id qed_swe_bench/v8

    # Push privately (smoke test)
    python scripts/publish_dataset.py minimax-m2.7-44.211.48.34 \\
        --push --private

    # Push public (refused without --license + --allow-public)
    python scripts/publish_dataset.py minimax-m2.7-44.211.48.34 \\
        --push --license cc-by-4.0 --allow-public
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_V8 = REPO_ROOT / "runs" / "v8"
DIST = REPO_ROOT / "dist"
DEFAULT_REPO_ID = "qed_swe_bench/v8"

# Capability columns mirror qed_swe_bench/runner/capabilities.py CAPABILITY_FLAGS.
CAPABILITY_FLAGS = [
    "cov_func", "cov_line",
    "diff", "asan", "crash",
    "addrof", "fakeobj", "caged_read", "caged_write",
    "infoleak_binary", "infoleak_libc", "infoleak_stack",
    "arb_read", "arb_write",
    "pc_control", "ace",
]


# ---------------------------------------------------------------------------
# Helpers (read job/score, compute revision)
# ---------------------------------------------------------------------------


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with path.open() as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def sha256_short(data: str | bytes, n: int = 7) -> str:
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data).hexdigest()[:n]


def compute_revision(bucket_dir: Path) -> str:
    """`v8-<v8.yaml-sha[:7]>-pt<prompt-template-sha[:7]>`.

    Reads `benchmarks/v8.yaml` and `benchmarks/bench-v8/prompt-template/v8.template`
    from the repo root (NOT from the run-dir; the cell's config_snapshot is a
    full snapshot, but for revision tagging we want the active repo state since
    that's what new buckets will be published under).
    """
    v8_yaml = REPO_ROOT / "benchmarks" / "v8.yaml"
    pt = REPO_ROOT / "benchmarks" / "bench-v8" / "prompt-template" / "v8.template"
    v8_sha = sha256_short(v8_yaml.read_bytes()) if v8_yaml.exists() else "noyaml"
    pt_sha = sha256_short(pt.read_bytes()) if pt.exists() else "nopt"
    return f"v8-{v8_sha}-pt{pt_sha}"


def discover_buckets() -> list[str]:
    """Return all <model>-<ip> bucket dir names under runs/v8/."""
    if not RUNS_V8.exists():
        return []
    out: list[str] = []
    for d in sorted(RUNS_V8.iterdir()):
        if not d.is_dir():
            continue
        if d.name.startswith("_"):  # skip _quarantine
            continue
        if (d / "CANONICAL_MANIFEST.tsv").exists():
            out.append(d.name)
    return out


# ---------------------------------------------------------------------------
# Walk a bucket, build the row + sidecar list
# ---------------------------------------------------------------------------


def collect_cells(bucket_dir: Path) -> list[dict]:
    """Walk bucket_dir/<dt>/<run_id>/ and return one row dict per cell.

    Each row is flattened ready for parquet write. Capability columns are
    boolean per-flag; transcript / tool_calls / grade_calls paths are
    relative paths into the bundle (computed at copy time).
    """
    rows: list[dict] = []
    for dt_dir in sorted(bucket_dir.iterdir()):
        if not dt_dir.is_dir() or dt_dir.name.startswith("_"):
            continue
        for run_dir in sorted(dt_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            job = read_json(run_dir / "job.json") or {}
            score = read_json(run_dir / "score.json") or {}
            cost = read_json(run_dir / "cost.json") or {}
            caps = score.get("capabilities") or {}
            row = {
                "model": job.get("model"),
                "env_id": job.get("env_id"),
                "seed": job.get("seed"),
                "run_id": job.get("run_id"),
                "benchmark_id": job.get("benchmark_id"),
                "host": run_dir.parent.parent.name.split("-")[-1] if run_dir.parent.parent.name else None,
                "started_at": job.get("started_at"),
                "finished_at": score.get("finished_at"),
                "status": score.get("status"),
                "score": score.get("score"),
                "turns_used": score.get("turns_used"),
                "runtime_s": score.get("runtime_s"),
                "exit_reason": score.get("exit_reason"),
                "image_ref": job.get("image_ref"),
                "image_digest": job.get("image_digest"),
                "git_sha": job.get("git_sha"),
                "weighted_tokens_used": score.get("weighted_tokens_used"),
                "peak_per_turn_context": score.get("peak_per_turn_context"),
                "cost_usd": cost.get("cost_usd"),
                "cost_source": cost.get("cost_source"),
                "tokens_in": cost.get("tokens_in"),
                "tokens_out": cost.get("tokens_out"),
                "tokens_cache_read": cost.get("tokens_cache_read"),
                "tokens_cache_creation": cost.get("tokens_cache_creation"),
                "tokens_reasoning": cost.get("tokens_reasoning"),
                "served_model": cost.get("served_model"),
                "_run_dir": run_dir,            # keep the source path for sidecar copy
                "_dt": dt_dir.name,             # for reconstructing relative paths
                "_run_id_dir": run_dir.name,
            }
            for cap in CAPABILITY_FLAGS:
                row[f"caps_{cap}"] = bool(caps.get(cap, False))
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Bundle a single cell's transcripts/tool_calls/grade_calls (zstd-compressed)
# ---------------------------------------------------------------------------


def compress_jsonl(src: Path, dst: Path) -> None:
    """Compress src JSONL → dst zstd. dst is created with parents."""
    import zstandard as zstd
    dst.parent.mkdir(parents=True, exist_ok=True)
    cctx = zstd.ZstdCompressor(level=10)
    with src.open("rb") as fin, dst.open("wb") as fout:
        fout.write(cctx.compress(fin.read()))


def bundle_cell_sidecars(row: dict, bundle_dir: Path) -> dict:
    """Copy + compress transcripts/tool_calls/grade_calls. Returns dict of
    relative paths to add back to the row."""
    run_dir: Path = row["_run_dir"]
    env_id = row["env_id"]
    seed = row["seed"]
    fname = f"seed_{seed}.jsonl.zst"
    out_paths = {}
    for kind in ("transcript", "tool_calls", "grade_calls"):
        src = run_dir / f"{kind}.jsonl"
        if not src.exists():
            out_paths[f"{kind}_path"] = None
            continue
        # transcripts/<env_id>/seed_<N>.jsonl.zst (singularize transcript→transcripts dir)
        subdir = "transcripts" if kind == "transcript" else kind
        rel = Path(subdir) / env_id / fname
        dst = bundle_dir / rel
        compress_jsonl(src, dst)
        out_paths[f"{kind}_path"] = str(rel)
    return out_paths


# ---------------------------------------------------------------------------
# Write parquet, manifest, audit, README
# ---------------------------------------------------------------------------


def write_runs_parquet(rows: list[dict], dest: Path) -> int:
    """Write rows to dest/runs.parquet. Returns row count."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    # Strip the underscore-prefixed bookkeeping fields before serializing.
    clean_rows = []
    for r in rows:
        cr = {k: v for k, v in r.items() if not k.startswith("_")}
        clean_rows.append(cr)
    table = pa.Table.from_pylist(clean_rows)
    pq.write_table(table, dest / "runs.parquet", compression="zstd")
    return len(clean_rows)


def write_audit_json(bucket_dir: Path, dest: Path) -> dict:
    """Run `qed_swe_bench audit --benchmark-id v8 --format json` and capture
    the per-bucket findings. Returns a counts dict for the manifest."""
    audit_path = dest / "audit.json"
    try:
        result = subprocess.run(
            [".venv/bin/qed-swe-bench", "audit",
             "--benchmark-id", "v8",
             "--format", "json"],
            cwd=str(REPO_ROOT),
            check=True, capture_output=True, text=True,
        )
        audit_data = json.loads(result.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError) as e:
        print(f"  ⚠ audit unavailable ({type(e).__name__}); writing empty audit.json",
              file=sys.stderr)
        audit_data = {"runs": [], "error": str(e)}
    # Filter to runs whose run_dir is under this bucket. Audit JSON emits
    # repo-relative paths like `runs/v8/<bucket>/...` — match by bucket name.
    bucket_name = bucket_dir.name
    audit_list = audit_data if isinstance(audit_data, list) else audit_data.get("runs", [])
    bucket_runs = [r for r in audit_list
                   if f"/{bucket_name}/" in str(r.get("run_dir", "")) + "/"
                   or str(r.get("run_dir", "")).endswith(f"/{bucket_name}")]
    counts = {"high": 0, "medium": 0, "info": 0}
    for r in bucket_runs:
        for f in r.get("findings", []):
            sev = f.get("severity", "").lower()
            if sev in counts:
                counts[sev] += 1
    audit_path.write_text(json.dumps({"runs": bucket_runs, "counts": counts}, indent=2))
    return counts


def copy_manifest_files(bucket_dir: Path, dest: Path) -> None:
    for name in ("CANONICAL_MANIFEST.md", "CANONICAL_MANIFEST.tsv"):
        src = bucket_dir / name
        if src.exists():
            shutil.copy2(src, dest / name)


def copy_config_snapshot(rows: list[dict], dest: Path) -> str | None:
    """Copy one cell's config_snapshot.yaml as the bundle's representative
    pin. They should all match (curate enforces same v8.yaml at curate time);
    the first cell's is fine."""
    if not rows:
        return None
    src = rows[0]["_run_dir"] / "config_snapshot.yaml"
    if src.exists():
        shutil.copy2(src, dest / "config_snapshot.yaml")
        return sha256_short(src.read_bytes())
    return None


def write_dataset_card(dest: Path, *, repo_id: str, revision: str,
                       bucket: str, rows: list[dict],
                       audit_counts: dict, license: str | None) -> None:
    """Auto-generate README.md (HF dataset card)."""
    n = len(rows)
    n_succ = sum(1 for r in rows if r["status"] == "succeeded")
    n_fail = sum(1 for r in rows if r["status"] == "model_failed")
    seeds = sorted({r["seed"] for r in rows if r["seed"] is not None})
    envs = sorted({r["env_id"] for r in rows if r["env_id"]})
    models = sorted({r["model"] for r in rows if r["model"]})

    yaml_header = "---\n"
    if license:
        yaml_header += f"license: {license}\n"
    yaml_header += "task_categories:\n  - reinforcement-learning\n"
    yaml_header += "tags:\n  - qed_swe_bench\n  - v8\n  - cybersecurity\n  - reasoning\n"
    yaml_header += "size_categories:\n  - n<1K\n"
    yaml_header += "---\n\n"

    lines = [
        yaml_header,
        f"# qed-swe-bench V8 — `{bucket}`\n",
        "",
        f"**Revision**: `{revision}`",
        "",
        f"Per-cell results from running **{', '.join(models)}** "
        f"against the V8 capability matrix.",
        "",
        "## Summary",
        "",
        f"- **Cells**: {n} ({n_succ} succeeded, {n_fail} model_failed)",
        f"- **Seeds**: {seeds}",
        f"- **Envs**: {len(envs)}",
        f"- **Audit findings**: {audit_counts.get('high', 0)} HIGH, "
        f"{audit_counts.get('medium', 0)} MEDIUM, {audit_counts.get('info', 0)} INFO",
        "",
        "## Loading",
        "",
        "```python",
        "from datasets import load_dataset",
        f'ds = load_dataset("{repo_id}", revision="{revision}")',
        "```",
        "",
        "## Schema",
        "",
        "`runs.parquet` has one row per `(model, env_id, seed)` cell with:",
        "",
        "- Identity: `model`, `env_id`, `seed`, `run_id`, `benchmark_id`",
        "- Outcome: `status`, `score`, `turns_used`, `runtime_s`, `exit_reason`",
        "- Cost: `cost_usd`, `tokens_in`, `tokens_out`, `tokens_cache_read`, "
        "`tokens_reasoning`",
        "- Capabilities: `caps_<cap>` boolean per flag in "
        "{cov_func, cov_line, diff, asan, crash, addrof, fakeobj, "
        "caged_read, caged_write, infoleak_*, arb_read, arb_write, "
        "pc_control, ace}",
        "- Provenance: `image_ref`, `image_digest`, `git_sha`, `served_model`",
        "- Sidecar paths: `transcript_path`, `tool_calls_path`, `grade_calls_path` "
        "(zstd-compressed JSONL relative to the dataset root)",
        "",
        "## Sidecars",
        "",
        "Per-cell logs live as zstd-compressed JSONL alongside `runs.parquet`:",
        "",
        "- `transcripts/<env_id>/seed_<N>.jsonl.zst` — full assistant + tool turns",
        "- `tool_calls/<env_id>/seed_<N>.jsonl.zst` — per-call timings",
        "- `grade_calls/<env_id>/seed_<N>.jsonl.zst` — per-grade results",
        "",
        "## Audit + canonical pick",
        "",
        "- `CANONICAL_MANIFEST.md` / `CANONICAL_MANIFEST.tsv` — explainability "
        "for which run-dir was picked per `(env, seed)` and what was rejected",
        "- `audit.json` — full C1-C11 transcript red-flag findings (see "
        "`qed_swe_bench/audit/transcripts.py`)",
        "",
        "## Reproducibility",
        "",
        "- `config_snapshot.yaml` — pinned `benchmarks/v8.yaml` for this revision",
        "- `image_digest` per row — re-pull via "
        "`docker pull <image_ref>@<image_digest>`",
        "- Re-run a cell: `qed_swe_bench rerun <run_id>` "
        "(see [qed_swe_bench](https://github.com/qed_swe_bench/qed_swe_bench))",
        "",
    ]
    if not license:
        lines.append("> **Note**: license unset on this revision. Set the "
                     "`license` field via `--license <spdx-id>` before "
                     "publishing publicly.")
        lines.append("")
    (dest / "README.md").write_text("\n".join(lines))


def write_top_manifest(dest: Path, *, repo_id: str, revision: str,
                       bucket: str, rows: list[dict],
                       audit_counts: dict, config_sha: str | None,
                       license: str | None) -> None:
    """manifest.json — top-level versioning + integrity info."""
    manifest = {
        "repo_id": repo_id,
        "revision": revision,
        "bucket": bucket,
        "license": license,
        "n_cells": len(rows),
        "n_succeeded": sum(1 for r in rows if r["status"] == "succeeded"),
        "n_model_failed": sum(1 for r in rows if r["status"] == "model_failed"),
        "seeds": sorted({r["seed"] for r in rows if r["seed"] is not None}),
        "envs": sorted({r["env_id"] for r in rows if r["env_id"]}),
        "models": sorted({r["model"] for r in rows if r["model"]}),
        "audit_counts": audit_counts,
        "config_snapshot_sha": config_sha,
        "files": {
            "runs": "runs.parquet",
            "audit": "audit.json",
            "manifest_md": "CANONICAL_MANIFEST.md",
            "manifest_tsv": "CANONICAL_MANIFEST.tsv",
            "config_snapshot": "config_snapshot.yaml",
            "transcripts_dir": "transcripts/",
            "tool_calls_dir": "tool_calls/",
            "grade_calls_dir": "grade_calls/",
        },
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2))


# ---------------------------------------------------------------------------
# Build one bucket
# ---------------------------------------------------------------------------


def build_bucket(bucket: str, repo_id: str, revision: str | None,
                 license: str | None) -> Path:
    bucket_dir = RUNS_V8 / bucket
    if not bucket_dir.is_dir():
        raise SystemExit(f"bucket dir not found: {bucket_dir}")
    rev = revision or compute_revision(bucket_dir)

    dest_root = DIST / repo_id.replace("/", "__") / rev
    dest = dest_root / bucket
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    print(f"[{bucket}] revision={rev}")
    print(f"  bundling → {dest.relative_to(REPO_ROOT)}/")

    rows = collect_cells(bucket_dir)
    if not rows:
        print(f"  ✗ no cells found in {bucket_dir}")
        return dest

    # Sidecars + collect their relative paths back into rows
    for row in rows:
        sidecar_paths = bundle_cell_sidecars(row, dest)
        row.update(sidecar_paths)

    # Parquet, manifests, audit
    n = write_runs_parquet(rows, dest)
    copy_manifest_files(bucket_dir, dest)
    config_sha = copy_config_snapshot(rows, dest)
    audit_counts = write_audit_json(bucket_dir, dest)

    write_dataset_card(dest, repo_id=repo_id, revision=rev, bucket=bucket,
                       rows=rows, audit_counts=audit_counts, license=license)
    write_top_manifest(dest, repo_id=repo_id, revision=rev, bucket=bucket,
                       rows=rows, audit_counts=audit_counts,
                       config_sha=config_sha, license=license)

    n_succ = sum(1 for r in rows if r["status"] == "succeeded")
    n_fail = sum(1 for r in rows if r["status"] == "model_failed")
    print(f"  ✓ {n} cells ({n_succ} succeeded, {n_fail} model_failed)")
    print(f"  ✓ audit: {audit_counts['high']} HIGH, {audit_counts['medium']} MEDIUM, "
          f"{audit_counts['info']} INFO")
    return dest


# ---------------------------------------------------------------------------
# HF push
# ---------------------------------------------------------------------------


def push_to_hf(bundle_dir: Path, repo_id: str, *, revision: str,
               private: bool, license: str | None,
               allow_public: bool) -> None:
    if not private and not allow_public:
        raise SystemExit("Refusing to push public without --allow-public flag.")
    if not private and not license:
        raise SystemExit("Refusing public push without --license <spdx-id>.")

    from huggingface_hub import HfApi, create_repo
    from huggingface_hub.errors import HfHubHTTPError
    api = HfApi()
    visibility = "private" if private else "public"
    print(f"  push → {repo_id} (visibility={visibility}, revision tag={revision})")

    # 1. Create or confirm the dataset repo
    try:
        create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    except HfHubHTTPError as e:
        raise SystemExit(
            f"create_repo failed: {e}. Token may lack 'Create repos' permission "
            f"for org `{repo_id.split('/')[0]}`. Workaround: pre-create the empty "
            f"dataset at https://huggingface.co/new-dataset (owner={repo_id.split('/')[0]}, "
            f"name={repo_id.split('/')[1]}, private={private}), then re-run."
        ) from e

    # 2. Upload to main branch. (The bundle_dir contains <bucket>/<files>, so
    # the dataset gets <bucket>/README.md, <bucket>/runs.parquet, etc.
    # Different model columns / sweep dates can publish alongside as
    # sibling subdirs.)
    api.upload_folder(
        folder_path=str(bundle_dir),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Publish bundle {bundle_dir.name} @ {revision}",
        create_pr=False,
    )
    print(f"  ✓ uploaded → https://huggingface.co/datasets/{repo_id} ({visibility})")

    # 3. Tag the push with the revision name (immutable snapshot for citing).
    try:
        api.create_tag(
            repo_id=repo_id,
            repo_type="dataset",
            tag=revision,
            tag_message=f"Methodology revision {revision}",
            exist_ok=True,
        )
        print(f"  ✓ tagged: revision={revision}")
    except HfHubHTTPError as e:
        # Non-fatal: upload succeeded, tag is the cherry on top.
        print(f"  ⚠ tag create failed (non-fatal): {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Bundle canonical runs/v8/<bucket>/ into HF dataset shape; optionally push.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("bucket", nargs="?",
                    help="<model>-<ip> bucket name under runs/v8/. Required unless --all.")
    ap.add_argument("--all", action="store_true",
                    help="Bundle every bucket discovered under runs/v8/.")
    ap.add_argument("--repo-id", default=DEFAULT_REPO_ID,
                    help=f"HF repo id (default: {DEFAULT_REPO_ID}).")
    ap.add_argument("--revision",
                    help="Override the auto-derived revision tag.")
    ap.add_argument("--push", action="store_true",
                    help="Push to HuggingFace after bundling.")
    ap.add_argument("--private", action="store_true",
                    help="Push as a private dataset (default if --push).")
    ap.add_argument("--license",
                    help="SPDX license id (required for public push).")
    ap.add_argument("--allow-public", action="store_true",
                    help="Acknowledge public push; required to override default-private.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    if not args.bucket and not args.all:
        print("Error: pass <bucket> or --all", file=sys.stderr)
        return 2

    buckets = discover_buckets() if args.all else [args.bucket]
    if not buckets:
        print("No buckets found under runs/v8/", file=sys.stderr)
        return 2

    revisions: dict[str, Path] = {}
    for b in buckets:
        try:
            dest = build_bucket(b, args.repo_id, args.revision, args.license)
            revisions[b] = dest
        except SystemExit as e:
            print(f"  ✗ {b}: {e}", file=sys.stderr)

    if args.push:
        # Push the parent dir of each bundle (contains all buckets at this revision).
        # All buckets share the same revision so we push once per (revision, repo).
        seen_roots = set()
        private = True if args.private or not args.allow_public else False
        for b, dest in revisions.items():
            root = dest.parent  # dist/<repo>/<rev>/
            if root in seen_roots:
                continue
            seen_roots.add(root)
            rev = root.name
            push_to_hf(root, args.repo_id, revision=rev,
                       private=private, license=args.license,
                       allow_public=args.allow_public)

    print()
    print("=== summary ===")
    for b, dest in revisions.items():
        print(f"  {b} → {dest.relative_to(REPO_ROOT)}/")
    if not args.push:
        print()
        print("Local-only build. To push: --push --private (smoke test).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
