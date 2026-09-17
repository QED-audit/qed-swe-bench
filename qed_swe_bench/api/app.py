"""FastAPI JSON backend for the qed_swe_bench Next.js webui.

This is a super-insecure server and should not be publicly exposed. 
It is made available for an internal dashboard to make it easier to 
explore results.  YOU HAVE BEEN WARNED!

The webui talks to this server over JSON. Auth happens at the webui layer
(better-auth gates `/dashboard/*`); this API trusts its caller and just
serves data from SQLite + run-dir files.

Run:
  qed_swe_bench api [--port 8000] [--reload]

Endpoints (all prefixed with /api):
  GET /health                                    — liveness
  GET /summary                                   — top-level dashboard stats
  GET /benchmarks                                — list of benchmark_ids with stats
  GET /benchmarks/{benchmark_id}                 — one benchmark's runs + agg
  GET /benchmarks/{benchmark_id}/matrix          — model × env capability matrix
  GET /runs                                      — list of all runs (paginated)
  GET /runs/{run_id}                             — one run's full record
  GET /runs/{run_id}/transcript                  — streamed transcript.jsonl
  GET /runs/{run_id}/tool_calls                  — streamed tool_calls.jsonl
  GET /runs/{run_id}/grade_calls                 — streamed grade_calls.jsonl
  GET /runs/{run_id}/mcp_stderr                  — streamed mcp_stderr.log
  GET /benchmarks/{benchmark_id}/bundle          — audit-bundle tarball download
  GET /leaderboard                               — model ranking
  GET /envs                                      — env catalog (distinct env_ids)
  GET /models                                    — model catalog (distinct models)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from qed_swe_bench.api.schemas import (
    BenchmarkDetail,
    BenchmarkMatrix,
    BenchmarkSummary,
    EnvSummary,
    LeaderboardEntry,
    ModelSummary,
    Run,
    RunsListResponse,
    SummaryResponse,
)
from qed_swe_bench.config import Config
from qed_swe_bench.db.schema import connect, init_db
from qed_swe_bench.runner.capabilities import CAPABILITY_FLAGS

app = FastAPI(
    title="qed_swe_bench API",
    description="JSON backend for the qed_swe_bench dashboard",
    version="0.1.0",
)

# CORS — Next.js dev server lives on :3000.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    init_db()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _row_to_dict(row) -> dict[str, Any]:
    d = dict(row)
    # Parse JSON columns so the wire format is structured.
    if d.get("capabilities"):
        try:
            d["capabilities"] = json.loads(d["capabilities"])
        except json.JSONDecodeError:
            pass
    return d


def _split_spend(rows: list[dict[str, Any]]) -> dict[str, float]:
    real = imputed = 0.0
    for r in rows:
        cost = r.get("cost_usd") or 0
        if (r.get("provenance") or "").startswith("imported_from_"):
            imputed += cost
        else:
            real += cost
    return {"real_fresh": round(real, 4), "imputed_historical": round(imputed, 4)}


# ---------------------------------------------------------------------------
# health + dashboard summary
# ---------------------------------------------------------------------------


@app.get("/api/health", operation_id="getHealth")
def health() -> dict[str, Any]:
    cfg = Config.from_env()
    return {
        "ok": True,
        "db_path": str(cfg.db_path),
        "runs_dir": str(cfg.runs_dir),
    }


@app.get("/api/summary", response_model=SummaryResponse, operation_id="getSummary")
def summary() -> dict[str, Any]:
    """Top-level stats for the dashboard home page."""
    with connect() as con:
        total = con.execute("SELECT count(*) AS n FROM runs").fetchone()["n"]
        by_status = [
            dict(r) for r in
            con.execute(
                "SELECT status, count(*) AS n FROM runs GROUP BY status ORDER BY n DESC"
            ).fetchall()
        ]
        by_model_top = [
            dict(r) for r in
            con.execute(
                "SELECT model, count(*) AS n, "
                "round(avg(score), 2) AS avg_score, "
                "round(sum(cost_usd), 4) AS total_cost "
                "FROM runs WHERE status='succeeded' "
                "GROUP BY model "
                "ORDER BY avg_score DESC NULLS LAST LIMIT 10"
            ).fetchall()
        ]
        all_costs = [
            dict(r) for r in
            con.execute(
                "SELECT cost_usd, provenance FROM runs WHERE cost_usd IS NOT NULL"
            ).fetchall()
        ]
        n_benchmarks = con.execute(
            "SELECT count(DISTINCT benchmark_id) AS n FROM runs"
        ).fetchone()["n"]
        n_envs = con.execute(
            "SELECT count(DISTINCT env_id) AS n FROM runs"
        ).fetchone()["n"]
        n_models = con.execute(
            "SELECT count(DISTINCT model) AS n FROM runs"
        ).fetchone()["n"]

    return {
        "total_runs": total,
        "n_benchmarks": n_benchmarks,
        "n_envs": n_envs,
        "n_models": n_models,
        "by_status": by_status,
        "by_model_top": by_model_top,
        "spend": _split_spend(all_costs),
    }


# ---------------------------------------------------------------------------
# benchmarks
# ---------------------------------------------------------------------------


@app.get("/api/benchmarks", response_model=list[BenchmarkSummary], operation_id="listBenchmarks")
def list_benchmarks() -> list[dict[str, Any]]:
    """Each benchmark with summary stats."""
    with connect() as con:
        rows = con.execute(
            """
            SELECT benchmark_id,
                   count(*) AS n_runs,
                   count(DISTINCT model) AS n_models,
                   count(DISTINCT env_id) AS n_envs,
                   count(DISTINCT seed) AS n_seeds,
                   round(avg(score), 2) AS avg_score,
                   round(sum(cost_usd), 4) AS total_cost,
                   max(provenance) AS provenance,
                   json_group_array(DISTINCT agent) AS agents_json,
                   min(started_at) AS first_started,
                   max(finished_at) AS last_finished
            FROM runs
            GROUP BY benchmark_id
            ORDER BY first_started DESC
            """
        ).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        d = dict(r)
        try:
            d["agents"] = sorted(json.loads(d.pop("agents_json") or "[]"))
        except (json.JSONDecodeError, TypeError):
            d["agents"] = []
            d.pop("agents_json", None)
        out.append(d)
    return out


@app.get("/api/benchmarks/{benchmark_id}", response_model=BenchmarkDetail, operation_id="getBenchmark")
def get_benchmark(benchmark_id: str) -> dict[str, Any]:
    with connect() as con:
        runs = [
            _row_to_dict(r) for r in
            con.execute(
                "SELECT * FROM runs WHERE benchmark_id = ? "
                "ORDER BY model, env_id, seed",
                (benchmark_id,),
            ).fetchall()
        ]
    if not runs:
        raise HTTPException(status_code=404, detail=f"benchmark {benchmark_id!r} not found")
    return {
        "benchmark_id": benchmark_id,
        "n_runs": len(runs),
        "models": sorted({r["model"] for r in runs}),
        "envs": sorted({r["env_id"] for r in runs}),
        "seeds": sorted({r["seed"] for r in runs}),
        "spend": _split_spend(runs),
        "runs": runs,
    }


@app.get("/api/benchmarks/{benchmark_id}/matrix", response_model=BenchmarkMatrix, operation_id="getBenchmarkMatrix")
def benchmark_matrix(benchmark_id: str) -> dict[str, Any]:
    """Model × env grid; each cell = cumulative-OR capability bitmap +
    avg score across seeds. Filters to status='succeeded' so failed /
    queued / cost-cap-skipped runs don't drag averages to zero or
    inflate per-cell seed counts; matches the filter used by /summary
    and /leaderboard."""
    with connect() as con:
        rows = [
            _row_to_dict(r) for r in
            con.execute(
                "SELECT * FROM runs WHERE benchmark_id = ? "
                "AND status = 'succeeded'",
                (benchmark_id,),
            ).fetchall()
        ]
    if not rows:
        raise HTTPException(status_code=404, detail=f"benchmark {benchmark_id!r} not found")

    # Aggregate by (model, env_id)
    cells: dict[tuple[str, str], dict[str, Any]] = {}
    for r in rows:
        key = (r["model"], r["env_id"])
        cell = cells.setdefault(key, {
            "model": r["model"],
            "env_id": r["env_id"],
            "n_seeds": 0,
            "avg_score": 0.0,
            "total_cost": 0.0,
            "capabilities": {},
        })
        cell["n_seeds"] += 1
        cell["avg_score"] += r.get("score") or 0
        cell["total_cost"] += r.get("cost_usd") or 0
        caps = r.get("capabilities") or {}
        if isinstance(caps, dict):
            for k, v in caps.items():
                if v:
                    cell["capabilities"][k] = True
                elif k not in cell["capabilities"]:
                    cell["capabilities"][k] = False

    # Finalize averages
    for cell in cells.values():
        if cell["n_seeds"]:
            cell["avg_score"] = round(cell["avg_score"] / cell["n_seeds"], 2)
            cell["total_cost"] = round(cell["total_cost"], 4)

    return {
        "benchmark_id": benchmark_id,
        "models": sorted({r["model"] for r in rows}),
        "envs": sorted({r["env_id"] for r in rows}),
        "capability_flags": list(CAPABILITY_FLAGS),
        "cells": list(cells.values()),
    }


# ---------------------------------------------------------------------------
# runs
# ---------------------------------------------------------------------------


@app.get("/api/runs", response_model=RunsListResponse, operation_id="listRuns")
def list_runs(
    benchmark_id: str | None = None,
    model: str | None = None,
    env_id: str | None = None,
    status: str | None = None,
    nudges_used: bool | None = None,
    limit: int = 200,
    offset: int = 0,
) -> dict[str, Any]:
    where: list[str] = []
    params: list[Any] = []
    if benchmark_id:
        where.append("benchmark_id = ?")
        params.append(benchmark_id)
    if model:
        where.append("model = ?")
        params.append(model)
    if env_id:
        where.append("env_id = ?")
        params.append(env_id)
    if status:
        where.append("status = ?")
        params.append(status)
    if nudges_used is not None:
        where.append("nudges_used = ?")
        params.append(1 if nudges_used else 0)

    where_sql = ("WHERE " + " AND ".join(where)) if where else ""
    with connect() as con:
        total = con.execute(
            f"SELECT count(*) AS n FROM runs {where_sql}", params
        ).fetchone()["n"]
        rows = [
            _row_to_dict(r) for r in
            con.execute(
                f"SELECT * FROM runs {where_sql} "
                f"ORDER BY started_at DESC NULLS LAST "
                f"LIMIT ? OFFSET ?",
                [*params, limit, offset],
            ).fetchall()
        ]
    return {"total": total, "limit": limit, "offset": offset, "runs": rows}


@app.get("/api/runs/{run_id}", response_model=Run, operation_id="getRun")
def get_run(run_id: str) -> dict[str, Any]:
    with connect() as con:
        row = con.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"run {run_id!r} not found")
    return _row_to_dict(row)


def _run_dir_for(run_id: str) -> Path:
    """Locate a run's directory, both for native runs and import_from_eval.

    Path-confinement: the runner writes `run_dir = cfg.runs_dir / benchmark_id
    / run_id` for native runs, and `import-eval` likewise records paths
    that should sit under `cfg.runs_dir`. We re-validate here so a buggy
    importer or a hand-edited DB row can't turn the file-streaming
    endpoints into arbitrary file read by run_id. Imported-from-eval
    rows that legitimately point outside `cfg.runs_dir` (e.g. the
    historical bench-v8 eval/ tree before it was copied in) are
    permitted via the explicit allowlist below.
    """
    with connect() as con:
        row = con.execute(
            "SELECT run_dir, provenance FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
    if row is None or not row["run_dir"]:
        raise HTTPException(status_code=404, detail=f"run {run_id!r} not found")
    p = Path(row["run_dir"]).resolve()
    if not p.is_dir():
        raise HTTPException(
            status_code=404,
            detail=f"run directory missing on disk: {p}",
        )
    cfg = Config.from_env()
    runs_root = cfg.runs_dir.resolve()
    if not p.is_relative_to(runs_root) and not (
        row["provenance"] or ""
    ).startswith("imported_from_"):
        # Defense-in-depth: the runner controls run_dir on every native
        # run, but a malformed import / DB tamper / future writer could
        # land a path outside runs_root. Refuse to serve files from
        # there even though the row exists in the DB.
        raise HTTPException(
            status_code=404,
            detail=f"run_dir is outside runs_root for native run {run_id!r}",
        )
    return p


def _stream_jsonl(path: Path):
    def gen():
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                yield line
    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.get("/api/runs/{run_id}/transcript", operation_id="getTranscript")
def get_transcript(run_id: str):
    p = _run_dir_for(run_id) / "transcript.jsonl"
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"no transcript for {run_id}")
    return _stream_jsonl(p)


@app.get("/api/runs/{run_id}/tool_calls", operation_id="getToolCalls")
def get_tool_calls(run_id: str):
    p = _run_dir_for(run_id) / "tool_calls.jsonl"
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"no tool_calls for {run_id}")
    return _stream_jsonl(p)


@app.get("/api/runs/{run_id}/grade_calls", operation_id="getGradeCalls")
def get_grade_calls(run_id: str):
    p = _run_dir_for(run_id) / "grade_calls.jsonl"
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"no grade_calls for {run_id}")
    return _stream_jsonl(p)


@app.get("/api/runs/{run_id}/mcp_stderr", operation_id="getMcpStderr")
def get_mcp_stderr(run_id: str):
    """Plain-text stream of the MCP container's stderr for one run.

    Captured by `McpDockerSession.start(..., stderr_path=...)`. The agent
    never sees this stream; it's the server-side audit channel.
    """
    p = _run_dir_for(run_id) / "mcp_stderr.log"
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"no mcp_stderr for {run_id}")

    def gen():
        with p.open("r", encoding="utf-8") as fh:
            yield from fh
    return StreamingResponse(gen(), media_type="text/plain; charset=utf-8")


@app.get("/api/benchmarks/{benchmark_id}/bundle", operation_id="getAuditBundle")
def get_audit_bundle(benchmark_id: str):
    """Build (synchronously) and stream an audit-bundle tarball for one benchmark.

    Shells out to `scripts/build_audit_bundle.sh`. The bundle includes
    every per-episode artifact + a sha256 manifest + a portable summary
    JSON; receivers verify integrity with `sha256sum -c MANIFEST.sha256`.

    Note: this builds on demand — for large benchmarks the request may
    take a few seconds. The script writes to `audit-bundles/` and we
    stream that file back; the file persists for re-download.
    """
    import re
    import shlex
    import subprocess

    # Strict allow-list: matches the canonical YAML benchmark_id shape and
    # leaves no room for SQL/shell metacharacters to reach the build script
    # (which itself enforces the same regex as defense-in-depth — see
    # scripts/build_audit_bundle.sh). Single-quote / semicolon used to slip
    # through the looser path-shaped check, enabling SQL injection in the
    # bash script's `sqlite3 ... WHERE benchmark_id = '$BENCHMARK_ID'`.
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", benchmark_id or ""):
        raise HTTPException(status_code=400, detail="invalid benchmark_id")

    cfg = Config.from_env()
    repo_root = cfg.runs_dir.parent  # data/ and audit-bundles/ are siblings
    script = repo_root / "scripts" / "build_audit_bundle.sh"
    if not script.exists():
        raise HTTPException(status_code=500, detail="audit-bundle script missing")

    # Run the script. It validates the benchmark and writes
    # audit-bundles/<id>-<utc-ts>.tar.gz.
    proc = subprocess.run(
        ["bash", str(script), benchmark_id],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        # Script's stderr already explains (missing benchmark, etc.).
        raise HTTPException(
            status_code=404 if "no runs for benchmark_id" in proc.stderr else 500,
            detail=proc.stderr.strip() or "audit-bundle build failed",
        )

    # Parse the tarball path out of the script's stdout. The "wrote <path>"
    # convention is stable; if it ever changes, glob the audit-bundles dir
    # for the most recent matching prefix as a fallback.
    tarball: Path | None = None
    for line in proc.stdout.splitlines():
        if line.startswith("wrote "):
            tarball = Path(shlex.split(line)[1])
            break
    if tarball is None or not tarball.exists():
        raise HTTPException(status_code=500, detail="bundle path not found in script output")

    def gen():
        with tarball.open("rb") as fh:
            while True:
                chunk = fh.read(64 * 1024)
                if not chunk:
                    break
                yield chunk
    return StreamingResponse(
        gen(),
        media_type="application/gzip",
        headers={"Content-Disposition": f'attachment; filename="{tarball.name}"'},
    )


# ---------------------------------------------------------------------------
# leaderboard / envs / models
# ---------------------------------------------------------------------------


@app.get("/api/leaderboard", response_model=list[LeaderboardEntry], operation_id="getLeaderboard")
def leaderboard(
    include_imported: bool = True,
) -> list[dict[str, Any]]:
    where = ["status = 'succeeded'"]
    if not include_imported:
        where.append("provenance NOT LIKE 'imported_from_%'")
    where_sql = " AND ".join(where)
    with connect() as con:
        rows = con.execute(
            f"""
            SELECT model,
                   count(*) AS n_runs,
                   count(DISTINCT env_id) AS n_envs,
                   round(sum(score), 2) AS total_score,
                   round(avg(score), 2) AS avg_score,
                   round(sum(cost_usd), 4) AS total_cost,
                   round(avg(turns_used), 1) AS avg_turns
            FROM runs WHERE {where_sql}
            GROUP BY model
            ORDER BY total_score DESC NULLS LAST
            """,
        ).fetchall()
    return [dict(r) for r in rows]


@app.get("/api/envs", response_model=list[EnvSummary], operation_id="listEnvs")
def list_envs() -> list[dict[str, Any]]:
    """Combined view: registered envs (rlenv_images) UNIONed with envs that
    have runs but aren't registered. Each row carries catalog metadata when
    available + run aggregates."""
    with connect() as con:
        rows = con.execute(
            """
            WITH run_agg AS (
                SELECT env_id,
                       count(*) AS n_runs,
                       count(DISTINCT model) AS n_models,
                       round(max(score), 2) AS best_score
                FROM runs GROUP BY env_id
            )
            SELECT
                COALESCE(c.env_id, r.env_id) AS env_id,
                COALESCE(c.image_ref, '') AS image_ref,
                c.image_digest,
                c.interface,
                COALESCE(c.task_type, 'binary_task') AS task_type,
                c.project,
                c.bug_id,
                c.capability_class,
                c.expected_capabilities,
                c.metadata,
                COALESCE(c.validation_status, 'unregistered') AS validation_status,
                COALESCE(r.n_runs, 0) AS n_runs,
                COALESCE(r.n_models, 0) AS n_models,
                r.best_score,
                CASE WHEN c.env_id IS NULL THEN 0 ELSE 1 END AS registered
            FROM rlenv_images c
            LEFT JOIN run_agg r ON r.env_id = c.env_id
            UNION
            SELECT
                r.env_id,
                '' AS image_ref,
                NULL AS image_digest,
                NULL AS interface,
                'binary_task' AS task_type,
                NULL AS project,
                NULL AS bug_id,
                NULL AS capability_class,
                NULL AS expected_capabilities,
                NULL AS metadata,
                'unregistered' AS validation_status,
                r.n_runs,
                r.n_models,
                r.best_score,
                0 AS registered
            FROM run_agg r
            WHERE r.env_id NOT IN (SELECT env_id FROM rlenv_images)
            ORDER BY env_id
            """
        ).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        d = dict(r)
        for k in ("expected_capabilities", "metadata"):
            if d.get(k):
                try:
                    d[k] = json.loads(d[k])
                except json.JSONDecodeError:
                    pass
        out.append(d)
    return out


@app.get("/api/models", response_model=list[ModelSummary], operation_id="listModels")
def list_models() -> list[dict[str, Any]]:
    with connect() as con:
        rows = con.execute(
            """
            SELECT model,
                   max(llm_route) AS llm_route,
                   count(*) AS n_runs,
                   count(DISTINCT env_id) AS n_envs,
                   round(avg(score), 2) AS avg_score,
                   round(sum(cost_usd), 4) AS total_cost
            FROM runs GROUP BY model ORDER BY model
            """
        ).fetchall()
    return [dict(r) for r in rows]
