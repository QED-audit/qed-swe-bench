"""Revision-tag computation for the published dataset.

A revision encodes the methodology snapshot the dataset was produced
under: the benchmark YAML and the prompt template. Two artifacts means
two short SHAs joined into one tag, so a consumer can read the tag
and know exactly which methodology generated the data.

Format: `v8-<sha7(benchmarks/v8.yaml)>-pt<sha7(prompt-template)>`.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

DEFAULT_BENCHMARK_YAML = Path("benchmarks/v8.yaml")
DEFAULT_PROMPT_TEMPLATE = Path("benchmarks/bench-v8/prompt-template/v8.template")


def _sha7(path: Path) -> str:
    if not path.exists():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:7]


def compute_revision(
    repo_root: Path,
    *,
    benchmark_yaml: Path = DEFAULT_BENCHMARK_YAML,
    prompt_template: Path = DEFAULT_PROMPT_TEMPLATE,
    prefix: str = "v8",
) -> str:
    """Return the revision tag for the dataset built from the current repo.

    `repo_root` is the directory containing `benchmarks/`; the two file
    paths are interpreted relative to it. Both must exist for a "real"
    revision; missing files yield `missing` segments so the caller still
    gets a deterministic string (useful in tests).
    """
    yaml_sha = _sha7(repo_root / benchmark_yaml)
    pt_sha = _sha7(repo_root / prompt_template)
    return f"{prefix}-{yaml_sha}-pt{pt_sha}"
