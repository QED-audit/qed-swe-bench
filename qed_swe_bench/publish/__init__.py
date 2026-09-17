"""HuggingFace dataset publication for qed_swe_bench runs.

Pipeline (one CLI invocation, `qed_swe_bench publish`):

  selection.select_canonical()  → pick best run per (model, env, seed) from
                                   the SQLite DB; apply --exclude-model
  audit_gate.gate()             → rerun audit; HIGH blocks unless overridden
  bundle.build()                → parquet + zstd JSONL sidecars under dist/
  card.write_card()             → README.md + manifest.json
  hf.push()                     → upload to HF (only if --push)

Two methodology rules drive the design (see docs/decisions.md D-15):

  1. HuggingFace is the academic record. Reward-hack cells, failures,
     and full transcripts all ship. Display-time filtering belongs to
     the website, not here.
  2. Private model names never enter a committed file. The only privacy
     mechanism is the ad-hoc `--exclude-model` CLI flag (repeatable).
"""

from qed_swe_bench.publish.selection import (
    CellRecord,
    model_slug,
    select_canonical,
)


class PublishError(Exception):
    """Raised by the publish pipeline for refused or invalid invocations."""


__all__ = [
    "CellRecord",
    "PublishError",
    "model_slug",
    "select_canonical",
]
