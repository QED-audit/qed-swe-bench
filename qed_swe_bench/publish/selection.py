"""Canonical-cell selection for publication.

Reads the SQLite `runs` table (the canonical store) and returns one
`CellRecord` per (model, env_id, seed) tuple, picking the best
candidate per the rules below.

Pick rule per (model, env_id, seed) — same as the legacy
`scripts/curate_canonical_runs.py`:

  1. Filter to status in {succeeded, model_failed} (failures are data
     for the academic record; queued / infra-failed / unknown are not).
  2. Rank: succeeded > model_failed.
  3. Tie-break on most recent `started_at`.

Privacy is opt-out per invocation: `exclude_models` drops entire models
(any matching `model` string) before the pick. By design, no allowlist
file lives in the repo — see docs/decisions.md D-15.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

# Status rank: higher wins. Anything not here is ineligible.
_STATUS_RANK = {"succeeded": 2, "model_failed": 1}


@dataclass(frozen=True)
class CellRecord:
    """One canonical cell selected for publication.

    Carries the DB columns the bundle writer needs plus the on-disk
    `run_dir` (absolute path) so downstream IO is uniform.
    """

    run_id: str
    benchmark_id: str
    model: str
    env_id: str
    seed: int
    status: str
    score: float | None
    capabilities: dict[str, bool]
    run_dir: Path
    started_at: str | None
    finished_at: str | None
    runtime_s: float | None
    turns_used: int | None
    exit_reason: str | None
    image_ref: str | None
    image_digest: str | None
    git_sha: str | None
    weighted_tokens_used: int | None
    peak_per_turn_context: int | None
    cost_usd: float | None
    cost_source: str | None
    tokens_in: int | None
    tokens_out: int | None
    tokens_cache_read: int | None
    tokens_cache_creation: int | None
    tokens_reasoning: int | None
    served_model: str | None


def model_slug(model_id: str) -> str:
    """`anthropic/claude-opus-4-7` → `claude-opus-4-7`;
    `minimax/MiniMax-M2.7` → `minimax-m2.7`.

    Strips provider prefix, lowercases, preserves dots and digits (model
    versions like `M2.7` are common and dot-readable on disk),
    collapses other non-alnum runs to a single dash, strips leading /
    trailing dashes / dots.
    """
    if "/" in model_id:
        model_id = model_id.split("/", 1)[1]
    slug = re.sub(r"[^a-z0-9.]+", "-", model_id.lower()).strip("-.")
    return slug or "unknown-model"


def _parse_capabilities(blob: str | None) -> dict[str, bool]:
    if not blob:
        return {}
    try:
        parsed = json.loads(blob)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {k: bool(v) for k, v in parsed.items()}


def _read_cost_json_extras(run_dir: Path) -> tuple[int | None, str | None]:
    """Read fields that live in cost.json but not in the DB schema.

    `tokens_reasoning` and `served_model` are written to disk by
    `runner/run_dir.py` but are NOT mirrored in the SQLite schema (see
    `db/schema.py` — only the cost columns that all providers populate
    are stored). We pull them per-cell from the run_dir's cost.json so
    the published parquet stays informative even though the DB is leaner.
    """
    path = run_dir / "cost.json"
    if not path.is_file():
        return None, None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None, None
    if not isinstance(data, dict):
        return None, None
    tr = data.get("tokens_reasoning")
    sm = data.get("served_model")
    return (
        int(tr) if isinstance(tr, (int, float)) else None,
        str(sm) if isinstance(sm, str) else None,
    )


def _row_to_cell(row: sqlite3.Row) -> CellRecord:
    run_dir = Path(row["run_dir"])
    tokens_reasoning, served_model = _read_cost_json_extras(run_dir)
    return CellRecord(
        run_id=row["run_id"],
        benchmark_id=row["benchmark_id"],
        model=row["model"],
        env_id=row["env_id"],
        seed=row["seed"],
        status=row["status"],
        score=row["score"],
        capabilities=_parse_capabilities(row["capabilities"]),
        run_dir=run_dir,
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        runtime_s=row["runtime_s"],
        turns_used=row["turns_used"],
        exit_reason=row["exit_reason"],
        image_ref=row["image_ref"],
        image_digest=row["image_digest"],
        git_sha=row["git_sha"],
        weighted_tokens_used=row["weighted_tokens_used"],
        peak_per_turn_context=row["peak_per_turn_context"],
        cost_usd=row["cost_usd"],
        cost_source=row["cost_source"],
        tokens_in=row["tokens_in"],
        tokens_out=row["tokens_out"],
        tokens_cache_read=row["tokens_cache_read"],
        tokens_cache_creation=row["tokens_cache_creation"],
        tokens_reasoning=tokens_reasoning,
        served_model=served_model,
    )


def _started_at_key(cell: CellRecord) -> str:
    # ISO-8601 strings sort lexicographically. Empty / None sorts first
    # so a row missing started_at never wins a tie-break.
    return cell.started_at or ""


def select_canonical(
    db_path: Path,
    benchmark_id: str,
    *,
    exclude_models: Iterable[str] = (),
) -> list[CellRecord]:
    """Return one CellRecord per (model, env_id, seed) for `benchmark_id`.

    Rows whose `model` matches an entry in `exclude_models` are dropped
    before the canonical pick (so an excluded model can't even be
    counted in the dry-run cell totals).

    Rows missing `run_dir` are skipped with no error — they cannot be
    bundled. Callers should run `qed_swe_bench import runs/` to backfill.
    """
    excluded = set(exclude_models)
    sql = """
        SELECT run_id, benchmark_id, model, env_id, seed, status, score,
               capabilities, run_dir, started_at, finished_at, runtime_s,
               turns_used, exit_reason, image_ref, image_digest, git_sha,
               weighted_tokens_used, peak_per_turn_context,
               cost_usd, cost_source, tokens_in, tokens_out,
               tokens_cache_read, tokens_cache_creation
          FROM runs
         WHERE benchmark_id = ?
           AND status IN ('succeeded', 'model_failed')
           AND run_dir IS NOT NULL
    """
    with sqlite3.connect(str(db_path)) as con:
        con.row_factory = sqlite3.Row
        rows = list(con.execute(sql, (benchmark_id,)))

    candidates: dict[tuple[str, str, int], list[CellRecord]] = defaultdict(list)
    for row in rows:
        if row["model"] in excluded:
            continue
        cell = _row_to_cell(row)
        candidates[(cell.model, cell.env_id, cell.seed)].append(cell)

    selected: list[CellRecord] = []
    for cells in candidates.values():
        cells.sort(
            key=lambda c: (_STATUS_RANK.get(c.status, 0), _started_at_key(c)),
            reverse=True,
        )
        selected.append(cells[0])

    selected.sort(key=lambda c: (c.model, c.env_id, c.seed))
    return selected


def find_orphan_run_dirs(
    runs_root: Path,
    benchmark_id: str,
    cells: list[CellRecord],
) -> list[Path]:
    """Return `runs/<benchmark_id>/.../<run_id>/` dirs with no matching cell.

    Used by the CLI to warn when the on-disk tree is ahead of the DB
    (typically: someone synced new EC2 runs but didn't run
    `qed_swe_bench import runs/`). Walks shallowly: looks for any
    directory under `runs_root/<benchmark_id>/` that contains a
    `job.json` file but whose parent's name doesn't appear in `cells`.
    """
    bench_root = runs_root / benchmark_id
    if not bench_root.is_dir():
        return []
    selected_run_ids = {c.run_id for c in cells}
    selected_dirs = {c.run_dir.resolve() for c in cells}
    orphans: list[Path] = []
    for job_path in bench_root.rglob("job.json"):
        rd = job_path.parent
        if rd.name in selected_run_ids:
            continue
        if rd.resolve() in selected_dirs:
            continue
        orphans.append(rd)
    return orphans
