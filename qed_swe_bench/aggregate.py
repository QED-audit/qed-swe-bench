"""Emit results tables for a benchmark_id.

Three formats supported (-f/--format):
  markdown — model × env capability bitmap, one cell per (model, env), with
             flag glyphs and per-row aggregates (score, cost, turns).
  csv      — one CSV row per run (no aggregation), all columns.
  json     — nested {benchmark_id, models[], envs[], runs[]} object.
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import sqlite3
from pathlib import Path
from typing import Any

from qed_swe_bench.db.schema import connect
from qed_swe_bench.runner.capabilities import CAPABILITY_FLAGS


def _fetch_runs(
    db_path: Path | None,
    benchmark_id: str,
) -> list[sqlite3.Row]:
    # Aggregation must NOT include failed/queued runs — they have no caps,
    # zero score, and would drag every cell average down + inflate per-cell
    # seed counts. The leaderboard / summary endpoints already filter
    # status='succeeded'; aggregate is now consistent with that.
    with connect(db_path) as con:
        rows = con.execute(
            "SELECT * FROM runs WHERE benchmark_id = ? "
            "AND status = 'succeeded' "
            "ORDER BY model, env_id, seed",
            (benchmark_id,),
        ).fetchall()
    return rows


def _glyph(cap: str, achieved: bool) -> str:
    return "·" if not achieved else _CAP_GLYPHS.get(cap, "✓")


# Compact 1-2 char glyphs per capability for the matrix view.
_CAP_GLYPHS = {
    "cov_func": "F", "cov_line": "L",
    "diff": "D", "asan": "A", "crash": "C",
    "addrof": "ad", "fakeobj": "fk",
    "caged_read": "cR", "caged_write": "cW",
    "infoleak_binary": "iB", "infoleak_libc": "iL", "infoleak_stack": "iS",
    "arb_read": "rR", "arb_write": "rW",
    "pc_control": "PC", "ace": "★",
}


def _aggregate_capability_row(rows: list[dict[str, Any]]) -> dict[str, bool]:
    """Cumulative-OR over capabilities for multiple seeds of the same (model, env)."""
    from qed_swe_bench.runner.capabilities import merge_capability_bitmaps
    return merge_capability_bitmaps(r.get("capabilities") for r in rows)


def to_markdown(rows: list[sqlite3.Row], benchmark_id: str) -> str:
    """Group by (model, env_id), aggregate caps, render a 16-column matrix."""
    grouped: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        d = dict(r)
        grouped.setdefault((d["model"], d["env_id"]), []).append(d)

    out: list[str] = []
    out.append(f"# benchmark `{benchmark_id}`\n")
    out.append(f"_runs: {len(rows)}; (model, env) pairs: {len(grouped)}_\n")

    # Header
    headers = ["model", "env", "n", "score", "cost ($)", "turns"] + [
        _CAP_GLYPHS.get(c, c) for c in CAPABILITY_FLAGS
    ]
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join("---" for _ in headers) + " |")

    for (model, env_id), seeds in sorted(grouped.items()):
        caps_agg = _aggregate_capability_row(seeds)
        n = len(seeds)
        scores = [s["score"] for s in seeds if s["score"] is not None]
        costs = [s["cost_usd"] for s in seeds if s["cost_usd"] is not None]
        turns = [s["turns_used"] for s in seeds if s["turns_used"] is not None]
        avg_score = f"{sum(scores) / len(scores):.1f}" if scores else "-"
        sum_cost = f"{sum(costs):.4f}" if costs else "-"
        avg_turns = f"{int(sum(turns) / len(turns))}" if turns else "-"
        cells = [_glyph(c, caps_agg.get(c, False)) for c in CAPABILITY_FLAGS]
        row = [
            f"`{model}`",
            f"`{env_id}`",
            str(n),
            avg_score,
            sum_cost,
            avg_turns,
        ] + cells
        out.append("| " + " | ".join(row) + " |")

    # Legend
    out.append("\n**Legend:** " + ", ".join(
        f"{_CAP_GLYPHS[c]}={c}" for c in CAPABILITY_FLAGS
    ))
    return "\n".join(out) + "\n"


def to_csv(rows: list[sqlite3.Row]) -> str:
    if not rows:
        return ""
    buf = io.StringIO()
    fieldnames = list(rows[0].keys())
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        writer.writerow(dict(r))
    return buf.getvalue()


def to_json(rows: list[sqlite3.Row], benchmark_id: str) -> str:
    runs_json = []
    for r in rows:
        d = dict(r)
        # Parse JSON columns so the output is structured.
        for k in ("capabilities",):
            if d.get(k):
                with contextlib.suppress(json.JSONDecodeError):
                    d[k] = json.loads(d[k])
        runs_json.append(d)
    models = sorted({r["model"] for r in rows})
    envs = sorted({r["env_id"] for r in rows})
    return (
        json.dumps(
            {
                "benchmark_id": benchmark_id,
                "models": models,
                "envs": envs,
                "n_runs": len(rows),
                "runs": runs_json,
            },
            indent=2,
            default=str,
        )
        + "\n"
    )


def aggregate(
    benchmark_id: str,
    *,
    output_format: str = "markdown",
    output: Path | None = None,
    db_path: Path | None = None,
    compare_with: str | None = None,
) -> str:
    if compare_with is not None:
        return _aggregate_compare(
            primary_id=benchmark_id,
            secondary_id=compare_with,
            output_format=output_format,
            output=output,
            db_path=db_path,
        )

    rows = _fetch_runs(db_path, benchmark_id)
    if not rows:
        return f"# no runs for benchmark_id={benchmark_id!r}\n"

    if output_format == "markdown":
        text = to_markdown(rows, benchmark_id)
    elif output_format == "csv":
        text = to_csv(rows)
    elif output_format == "json":
        text = to_json(rows, benchmark_id)
    else:
        raise ValueError(f"unknown format: {output_format}")

    if output:
        output.write_text(text, encoding="utf-8")
    return text


def _aggregate_compare(
    *,
    primary_id: str,
    secondary_id: str,
    output_format: str,
    output: Path | None,
    db_path: Path | None,
) -> str:
    """Side-by-side per-cell comparison of two benchmark_ids.

    Intended use: scaffold-effect study (primary `v8` with nudges=false vs
    secondary `v8-nudged` with nudges=true) so the matrix view shows both
    regimes' scores and the delta in one table. Cells present in only one
    regime appear with `-` for the missing side.
    """
    primary = _fetch_runs(db_path, primary_id)
    secondary = _fetch_runs(db_path, secondary_id)
    if not primary and not secondary:
        return (
            f"# no runs for either benchmark_id "
            f"(primary={primary_id!r}, secondary={secondary_id!r})\n"
        )

    if output_format == "markdown":
        text = to_markdown_compare(primary, secondary, primary_id, secondary_id)
    elif output_format == "json":
        text = to_json_compare(primary, secondary, primary_id, secondary_id)
    elif output_format == "csv":
        # CSV compare is just both benchmarks' CSV concatenated with a
        # `regime` column prepended; less useful than markdown but keeps
        # the format set consistent.
        text = to_csv_compare(primary, secondary, primary_id, secondary_id)
    else:
        raise ValueError(f"unknown format: {output_format}")

    if output:
        output.write_text(text, encoding="utf-8")
    return text


def _group_by_cell(rows: list[sqlite3.Row]) -> dict[tuple[str, str], list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        d = dict(r)
        grouped.setdefault((d["model"], d["env_id"]), []).append(d)
    return grouped


def _cell_summary(seeds: list[dict]) -> tuple[float | None, dict[str, bool]]:
    """Return (mean_score, capability_bitmap) for a (model, env) cell."""
    scores = [s["score"] for s in seeds if s["score"] is not None]
    mean_score = sum(scores) / len(scores) if scores else None
    caps = _aggregate_capability_row(seeds)
    return mean_score, caps


def to_markdown_compare(
    primary_rows: list[sqlite3.Row],
    secondary_rows: list[sqlite3.Row],
    primary_id: str,
    secondary_id: str,
) -> str:
    """Render side-by-side comparison: per (model, env) cell, show primary
    mean score, secondary mean score, delta, and caps acquired only in
    the secondary regime."""
    p_grouped = _group_by_cell(primary_rows)
    s_grouped = _group_by_cell(secondary_rows)
    all_cells = sorted(set(p_grouped) | set(s_grouped))

    out: list[str] = []
    out.append(f"# `{primary_id}` vs `{secondary_id}` (regime comparison)\n")
    out.append(
        f"_primary runs: {len(primary_rows)}, secondary runs: {len(secondary_rows)}, "
        f"cells: {len(all_cells)}_\n"
    )
    out.append(
        "| model | env | n (1°/2°) | score 1° | score 2° | Δ | caps gained by 2° |"
    )
    out.append("| --- | --- | --- | --- | --- | --- | --- |")

    for (model, env_id) in all_cells:
        p_seeds = p_grouped.get((model, env_id), [])
        s_seeds = s_grouped.get((model, env_id), [])
        p_score, p_caps = _cell_summary(p_seeds)
        s_score, s_caps = _cell_summary(s_seeds)
        # Caps the secondary acquired that the primary didn't.
        gained = sorted(c for c in s_caps if s_caps[c] and not p_caps.get(c))
        gained_str = ", ".join(gained) if gained else "—"
        delta = (
            f"{(s_score - p_score):+.1f}"
            if p_score is not None and s_score is not None
            else "-"
        )
        out.append(
            "| "
            + " | ".join([
                f"`{model}`",
                f"`{env_id}`",
                f"{len(p_seeds)}/{len(s_seeds)}",
                f"{p_score:.1f}" if p_score is not None else "-",
                f"{s_score:.1f}" if s_score is not None else "-",
                delta,
                gained_str,
            ])
            + " |"
        )

    return "\n".join(out) + "\n"


def to_json_compare(
    primary_rows: list[sqlite3.Row],
    secondary_rows: list[sqlite3.Row],
    primary_id: str,
    secondary_id: str,
) -> str:
    """Per-cell comparison as a structured JSON object — for downstream
    plotting (matrix delta heatmap, scaffold-effect bar chart)."""
    p_grouped = _group_by_cell(primary_rows)
    s_grouped = _group_by_cell(secondary_rows)
    all_cells = sorted(set(p_grouped) | set(s_grouped))

    cells = []
    for (model, env_id) in all_cells:
        p_seeds = p_grouped.get((model, env_id), [])
        s_seeds = s_grouped.get((model, env_id), [])
        p_score, p_caps = _cell_summary(p_seeds)
        s_score, s_caps = _cell_summary(s_seeds)
        cells.append({
            "model": model,
            "env_id": env_id,
            "primary": {
                "n": len(p_seeds),
                "mean_score": p_score,
                "capabilities": p_caps,
            },
            "secondary": {
                "n": len(s_seeds),
                "mean_score": s_score,
                "capabilities": s_caps,
            },
            "delta_score": (
                s_score - p_score if p_score is not None and s_score is not None
                else None
            ),
            "caps_gained_by_secondary": sorted(
                c for c in s_caps if s_caps[c] and not p_caps.get(c)
            ),
            "caps_lost_by_secondary": sorted(
                c for c in p_caps if p_caps[c] and not s_caps.get(c)
            ),
        })
    return (
        json.dumps(
            {
                "primary_id": primary_id,
                "secondary_id": secondary_id,
                "cells": cells,
            },
            indent=2,
            default=str,
        )
        + "\n"
    )


def to_csv_compare(
    primary_rows: list[sqlite3.Row],
    secondary_rows: list[sqlite3.Row],
    primary_id: str,
    secondary_id: str,
) -> str:
    """Both benchmarks' CSVs concatenated with a leading `regime` column.
    Less useful than markdown for scanning but lets users pivot in a
    spreadsheet."""
    if not primary_rows and not secondary_rows:
        return ""
    sample = (primary_rows or secondary_rows)[0]
    fieldnames = ["regime"] + list(sample.keys())
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for r in primary_rows:
        row = {"regime": "primary", **dict(r)}
        writer.writerow(row)
    for r in secondary_rows:
        row = {"regime": "secondary", **dict(r)}
        writer.writerow(row)
    return buf.getvalue()
