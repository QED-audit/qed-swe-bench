"""Dataset card (`README.md`) and manifest writers for the publish bundle.

Two manifest variants exist:

  - **Local** (operator's `dist/...`): includes `excluded_models` so the
    operator has an audit trail of which models they kept private.
  - **Upload** (pushed to HuggingFace): omits `excluded_models` entirely.
    Private model names never enter any uploaded artifact.

`write_manifest(..., for_upload=True)` produces the upload variant.
The CLI writes both to disk under different filenames and only uploads
the `for_upload=True` one — see publish/cli.py and publish/hf.py.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from qed_swe_bench.publish.audit_gate import GateResult
from qed_swe_bench.publish.bundle import BundleStats
from qed_swe_bench.publish.selection import CellRecord


def _yaml_frontmatter(license_id: str | None) -> str:
    lines = ["---"]
    if license_id:
        lines.append(f"license: {license_id}")
    lines.append("task_categories:")
    lines.append("  - reinforcement-learning")
    lines.append("tags:")
    lines.append("  - qed_swe_bench")
    lines.append("  - v8")
    lines.append("  - cybersecurity")
    lines.append("  - reasoning")
    lines.append("size_categories:")
    lines.append("  - n<1K")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def _models_summary(cells: Iterable[CellRecord]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for cell in cells:
        slot = out.setdefault(
            cell.model, {"cells": 0, "succeeded": 0, "model_failed": 0}
        )
        slot["cells"] += 1
        if cell.status == "succeeded":
            slot["succeeded"] += 1
        elif cell.status == "model_failed":
            slot["model_failed"] += 1
    return dict(sorted(out.items()))


def write_card(
    dest_root: Path,
    *,
    repo_id: str,
    revision: str,
    cells: list[CellRecord],
    stats: BundleStats,
    gate: GateResult,
    license_id: str | None,
) -> Path:
    """Write `README.md` to dest_root and return its path."""
    models = _models_summary(cells)
    envs = sorted({c.env_id for c in cells})
    seeds = sorted({c.seed for c in cells})

    lines = [_yaml_frontmatter(license_id)]
    lines.append(f"# qed-swe-bench V8 — `{revision}`")
    lines.append("")
    lines.append(
        "Per-cell capability results from the V8 JavaScript engine "
        "benchmark, with full transcripts, tool-call logs, and capability "
        "grading. This dataset is the **academic record** for qed-swe-bench: "
        "succeeded runs and model-failed runs both ship, including cells "
        "where the model gamed the grader (see `audit.json`)."
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Cells**: {stats.n_cells} "
                 f"({stats.n_succeeded} succeeded, "
                 f"{stats.n_model_failed} model_failed)")
    lines.append(f"- **Models**: {len(models)}")
    lines.append(f"- **Envs**: {len(envs)}")
    lines.append(f"- **Seeds per cell**: {seeds}")
    lines.append(
        f"- **Audit**: {gate.counts.get('high', 0)} HIGH, "
        f"{gate.counts.get('medium', 0)} MEDIUM, "
        f"{gate.counts.get('info', 0)} INFO "
        f"(see `audit.json`)"
    )
    lines.append("")

    lines.append("## Models in this revision")
    lines.append("")
    lines.append("| Model | Cells | Succeeded | Model-failed |")
    lines.append("| --- | ---: | ---: | ---: |")
    for model, counts in models.items():
        lines.append(
            f"| `{model}` | {counts['cells']} | "
            f"{counts['succeeded']} | {counts['model_failed']} |"
        )
    lines.append("")

    lines.append("## Loading")
    lines.append("")
    lines.append("```python")
    lines.append("from datasets import load_dataset")
    lines.append(f'ds = load_dataset("{repo_id}", revision="{revision}")')
    lines.append("```")
    lines.append("")

    lines.append("## Schema")
    lines.append("")
    lines.append(
        "`runs.parquet` has one row per `(model, env_id, seed)` cell. "
        "Columns:"
    )
    lines.append("")
    lines.append(
        "- **Identity**: `model`, `env_id`, `seed`, `run_id`, `benchmark_id`"
    )
    lines.append(
        "- **Outcome**: `status` (`succeeded` | `model_failed`), `score`, "
        "`turns_used`, `runtime_s`, `exit_reason`"
    )
    lines.append(
        "- **Cost**: `cost_usd`, `tokens_in`, `tokens_out`, "
        "`tokens_cache_read`, `tokens_cache_creation`, `tokens_reasoning`"
    )
    lines.append(
        "- **Capabilities** (16 boolean columns, prefix `caps_`): "
        "`cov_func`, `cov_line`, `diff`, `asan`, `crash`, `addrof`, "
        "`fakeobj`, `caged_read`, `caged_write`, `infoleak_binary`, "
        "`infoleak_libc`, `infoleak_stack`, `arb_read`, `arb_write`, "
        "`pc_control`, `ace`"
    )
    lines.append(
        "- **Provenance**: `image_ref`, `image_digest`, `git_sha`, "
        "`served_model`"
    )
    lines.append(
        "- **Sidecar paths**: `transcript_path`, `tool_calls_path`, "
        "`grade_calls_path` (POSIX-style relative paths into the dataset)"
    )
    lines.append("")

    lines.append("## Sidecars")
    lines.append("")
    lines.append(
        "Per-cell logs are zstd-compressed JSONL alongside `runs.parquet`:"
    )
    lines.append("")
    lines.append(
        "- `transcripts/<model_slug>/<env_id>/seed_<N>.jsonl.zst` — "
        "full assistant + tool turns"
    )
    lines.append(
        "- `tool_calls/<model_slug>/<env_id>/seed_<N>.jsonl.zst` — "
        "per-call args, results, timings"
    )
    lines.append(
        "- `grade_calls/<model_slug>/<env_id>/seed_<N>.jsonl.zst` — "
        "per-grade capability bitmaps"
    )
    lines.append("")

    lines.append("## Audit")
    lines.append("")
    lines.append(
        "`audit.json` contains the C1-C11 transcript red-flag findings "
        "(see [`qed_swe_bench/audit/transcripts.py`](https://github.com/"
        "qed_swe_bench/qed_swe_bench/blob/main/qed_swe_bench/audit/"
        "transcripts.py))."
    )
    lines.append("")
    lines.append(
        "**The audit focuses manual review; it is not a definitive "
        "judgment of cheating.** Checks are grep-shaped substring scans "
        "on tool-call arguments — they are intentionally simple so that "
        "the audit is itself auditable, which means false positives are "
        "expected (especially in C1). A finding flags a run *for human "
        "inspection*. Treat HIGH/MEDIUM/INFO severity as \"how loudly to "
        "look,\" not \"how guilty.\" The publish pipeline blocks on HIGH "
        "to force human triage; once a human has confirmed each HIGH is "
        "benign or expected, the dataset ships with the findings "
        "preserved here for downstream readers to re-triage themselves."
    )
    lines.append("")

    lines.append("## Reproducibility")
    lines.append("")
    lines.append(
        "- `config_snapshot.yaml` — pinned `benchmarks/v8.yaml` for this "
        "revision."
    )
    lines.append(
        "- `image_digest` per row — re-pull the exact env via "
        "`docker pull <image_ref>@<image_digest>`."
    )
    lines.append(
        "- Re-run a single cell: `qed_swe_bench rerun <run_id>` "
        "(see the [qed_swe_bench](https://github.com/qed_swe_bench/"
        "qed_swe_bench) repo)."
    )
    lines.append("")

    if not license_id:
        lines.append(
            "> **Note**: license unset on this revision. Set the `license` "
            "field via `--license <spdx-id>` before publishing publicly."
        )
        lines.append("")

    out_path = dest_root / "README.md"
    out_path.write_text("\n".join(lines))
    return out_path


def write_manifest(
    dest_root: Path,
    *,
    repo_id: str,
    revision: str,
    cells: list[CellRecord],
    stats: BundleStats,
    gate: GateResult,
    license_id: str | None,
    excluded_models: list[str],
    for_upload: bool,
    filename: str = "manifest.json",
) -> Path:
    """Write a manifest JSON to dest_root.

    `for_upload=True` omits the `excluded_models` field — the upload
    variant must never name what isn't there. `for_upload=False` keeps
    it, for the operator's local audit trail.
    """
    manifest: dict[str, Any] = {
        "repo_id": repo_id,
        "revision": revision,
        "license": license_id,
        "n_cells": stats.n_cells,
        "n_succeeded": stats.n_succeeded,
        "n_model_failed": stats.n_model_failed,
        "models": sorted({c.model for c in cells}),
        "envs": sorted({c.env_id for c in cells}),
        "seeds": sorted({c.seed for c in cells}),
        "audit_counts": gate.counts,
        "config_snapshot_sha": stats.config_snapshot_sha,
        "files": {
            "runs": "runs.parquet",
            "audit": "audit.json",
            "card": "README.md",
            "config_snapshot": "config_snapshot.yaml",
            "transcripts_dir": "transcripts/",
            "tool_calls_dir": "tool_calls/",
            "grade_calls_dir": "grade_calls/",
        },
    }
    if not for_upload:
        manifest["excluded_models"] = sorted(excluded_models)

    out_path = dest_root / filename
    out_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return out_path


def write_audit_json(
    dest_root: Path,
    *,
    gate: GateResult,
) -> Path:
    """Write `audit.json` summarizing the gate's findings."""
    payload: dict[str, Any] = {
        "counts": gate.counts,
        "high_run_ids": list(gate.high_run_ids),
        "medium_run_ids": list(gate.medium_run_ids),
        "info_run_ids": list(gate.info_run_ids),
        "runs": [
            {
                "run_id": r.run_id,
                "run_dir": str(r.run_dir),
                "findings": [
                    {
                        "check_id": f.check_id,
                        "name": f.name,
                        "severity": f.severity.value,
                        "detail": f.detail,
                    }
                    for f in r.findings
                ],
            }
            for r in gate.reports
            if r.findings
        ],
    }
    out_path = dest_root / "audit.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    return out_path
