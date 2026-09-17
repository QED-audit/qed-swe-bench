#!/usr/bin/env python3
"""publish_leaderboard.py — derive aggregate JSON from data/qed_swe_bench.sqlite.

Light, complementary to `publish_dataset.py`. Reads the imported DB,
groups by `(model, env_id)` across seeds, computes mean/std/se of score
plus per-capability hit rates, and writes one results.json per revision.

Output: `dist/leaderboard/<revision>/results.json`

Shape:
    {
      "revision": "v8-<sha>-pt<sha>",
      "n_models": 1,
      "n_envs": 14,
      "n_cells": 70,
      "models": {
        "minimax/MiniMax-M2.7": {
          "envs": {
            "v8-e01": {
              "n": 5,
              "score_mean": 2.2,
              "score_std": 0.748,
              "score_se": 0.335,
              "caps_hit_rate": {"cov_func": "5/5", "diff": "2/5", ...},
              "cost_total": 5.6749,
              "cost_mean": 1.135,
              "turns_mean": 176.0
            },
            ...
          },
          "overall": {
            "n_envs": 14,
            "score_mean": <mean across all envs>,
            "cost_total": <sum across all envs>
          }
        }
      }
    }

CLI:
    python scripts/publish_leaderboard.py
    python scripts/publish_leaderboard.py --benchmark-id v8 --revision <override>
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = REPO_ROOT / "data" / "qed_swe_bench.sqlite"
DIST = REPO_ROOT / "dist"

CAPABILITY_FLAGS = [
    "cov_func", "cov_line",
    "diff", "asan", "crash",
    "addrof", "fakeobj", "caged_read", "caged_write",
    "infoleak_binary", "infoleak_libc", "infoleak_stack",
    "arb_read", "arb_write",
    "pc_control", "ace",
]


def compute_revision() -> str:
    """`v8-<v8.yaml-sha[:7]>-pt<prompt-template-sha[:7]>` (matches publish_dataset.py)."""
    import hashlib
    v8_yaml = REPO_ROOT / "benchmarks" / "v8.yaml"
    pt = REPO_ROOT / "benchmarks" / "bench-v8" / "prompt-template" / "v8.template"
    def short(p: Path) -> str:
        if not p.exists():
            return "missing"
        return hashlib.sha256(p.read_bytes()).hexdigest()[:7]
    return f"v8-{short(v8_yaml)}-pt{short(pt)}"


def fetch_rows(conn: sqlite3.Connection, benchmark_id: str) -> list[sqlite3.Row]:
    conn.row_factory = sqlite3.Row
    return list(conn.execute("""
        SELECT model, env_id, seed, status, score, turns_used, cost_usd,
               capabilities
          FROM runs
         WHERE benchmark_id = ? AND status = 'succeeded'
    """, (benchmark_id,)))


def stats(values: list[float]) -> tuple[float, float, float]:
    """Return (mean, std (Bessel-corrected), stderr) for a list of floats."""
    n = len(values)
    if n == 0:
        return (0.0, 0.0, 0.0)
    mean = sum(values) / n
    if n < 2:
        return (mean, 0.0, 0.0)
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    std = math.sqrt(var)
    se = std / math.sqrt(n)
    return (mean, std, se)


def cap_hit_rate(rows_for_env: list[sqlite3.Row]) -> dict[str, str]:
    """Return e.g. {'cov_func': '5/5', 'diff': '2/5', ...} for each capability."""
    n = len(rows_for_env)
    out = {}
    for cap in CAPABILITY_FLAGS:
        hits = 0
        for r in rows_for_env:
            caps = json.loads(r["capabilities"]) if r["capabilities"] else {}
            if caps.get(cap):
                hits += 1
        out[cap] = f"{hits}/{n}"
    return out


def build_leaderboard(rows: list[sqlite3.Row], revision: str) -> dict:
    by_model: dict[str, dict[str, list[sqlite3.Row]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_model[r["model"]][r["env_id"]].append(r)

    out = {
        "revision": revision,
        "n_models": len(by_model),
        "n_envs": len({r["env_id"] for r in rows}),
        "n_cells": len(rows),
        "models": {},
    }

    for model, envs in sorted(by_model.items()):
        env_summaries = {}
        all_scores = []
        total_cost = 0.0
        for env_id, env_rows in sorted(envs.items()):
            scores = [r["score"] for r in env_rows if r["score"] is not None]
            costs = [r["cost_usd"] for r in env_rows if r["cost_usd"] is not None]
            turns = [r["turns_used"] for r in env_rows if r["turns_used"] is not None]
            mean, std, se = stats(scores)
            env_summaries[env_id] = {
                "n": len(env_rows),
                "score_mean": round(mean, 3),
                "score_std": round(std, 3),
                "score_se": round(se, 3),
                "caps_hit_rate": cap_hit_rate(env_rows),
                "cost_total": round(sum(costs), 4) if costs else 0.0,
                "cost_mean": round(sum(costs) / len(costs), 4) if costs else 0.0,
                "turns_mean": round(sum(turns) / len(turns), 1) if turns else None,
            }
            all_scores.extend(scores)
            total_cost += sum(costs)

        overall_mean, overall_std, overall_se = stats(all_scores)
        out["models"][model] = {
            "envs": env_summaries,
            "overall": {
                "n_envs": len(env_summaries),
                "n_cells": sum(s["n"] for s in env_summaries.values()),
                "score_mean": round(overall_mean, 3),
                "score_std": round(overall_std, 3),
                "score_se": round(overall_se, 3),
                "cost_total": round(total_cost, 4),
            },
        }

    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Derive aggregate leaderboard JSON from data/qed_swe_bench.sqlite.",
    )
    ap.add_argument("--benchmark-id", default="v8")
    ap.add_argument("--revision",
                    help="Override the auto-derived revision tag.")
    ap.add_argument("--db", default=str(DB_PATH),
                    help=f"SQLite path (default: {DB_PATH.relative_to(REPO_ROOT)}).")
    ap.add_argument("--output",
                    help="Write to this path instead of dist/leaderboard/<rev>/results.json.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    db_path = Path(args.db).resolve()
    if not db_path.exists():
        print(f"DB not found: {db_path}. Run `qed_swe_bench import runs/` first.",
              file=sys.stderr)
        return 2

    revision = args.revision or compute_revision()

    with sqlite3.connect(db_path) as conn:
        rows = fetch_rows(conn, args.benchmark_id)

    if not rows:
        print(f"No succeeded runs for benchmark_id={args.benchmark_id}",
              file=sys.stderr)
        return 1

    data = build_leaderboard(rows, revision)

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = DIST / "leaderboard" / revision / "results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2))

    print(f"Wrote {out_path.relative_to(REPO_ROOT) if out_path.is_relative_to(REPO_ROOT) else out_path}")
    print(f"  revision: {revision}")
    print(f"  models: {data['n_models']}")
    print(f"  envs: {data['n_envs']}")
    print(f"  cells: {data['n_cells']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
