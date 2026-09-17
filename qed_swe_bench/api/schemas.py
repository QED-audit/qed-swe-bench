"""Pydantic response models for the dashboard-consumed API routes.

Used as FastAPI `response_model=` annotations so the auto-generated
`/openapi.json` carries proper types — the webui's
`@hey-api/openapi-ts` codegen reads that spec to produce a typed
fetch client at build time.

Scoping (per the API-typing audit):
  - Each model below corresponds to one route on `api/app.py`.
  - Streaming endpoints (transcript / tool_calls / grade_calls /
    mcp_stderr) and the binary audit-bundle endpoint stay untyped —
    they don't return JSON shapes the OpenAPI codegen can use.
  - `/health` is unannotated (single boolean, not worth a model).

Convention: every field is non-strict by default. SQLite columns can
be NULL where the DB schema permits it; we surface those as Optional
so the codegen produces `string | null` instead of `string`. This
matches what the wire actually carries today.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------


class Spend(BaseModel):
    """Cost breakdown by provenance — any `imported_from_*` row
    (vr-agent, codex, etc.) has imputed cost (computed post-hoc from
    token counts), fresh runs have real cost from the provider's
    response."""

    real_fresh: float
    imputed_historical: float


class StatusCount(BaseModel):
    """One row of GROUP BY status / count(*)."""

    status: str
    n: int


class ModelTopEntry(BaseModel):
    """One row of the top-models table on the dashboard summary card."""

    model: str
    n: int
    avg_score: float | None = None
    total_cost: float | None = None


class Run(BaseModel):
    """A single row from the `runs` SQLite table.

    Capabilities are parsed JSON (a dict of flag → bool) by the time
    they reach this shape; the DB stores them as a JSON-encoded TEXT.
    Most numeric / timing columns can be NULL until a run finishes.
    """

    # Permit extras so a future SQL-schema additive change doesn't
    # break the wire contract until the model catches up.
    model_config = ConfigDict(extra="allow")

    run_id: str
    benchmark_id: str
    model: str
    env_id: str
    image_ref: str
    image_digest: str
    task_type: str
    interface: str | None = None
    seed: int
    status: str
    capabilities: dict[str, bool] | None = None
    score: float | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    tokens_cache_read: int | None = None
    tokens_cache_creation: int | None = None
    cost_usd: float | None = None
    cost_source: str | None = None
    runtime_s: float | None = None
    turns_used: int | None = None
    exit_reason: str | None = None
    run_dir: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    provenance: str = "native"
    llm_route: str | None = None
    api_base: str | None = None
    failure_reason: str | None = None
    nudges_used: bool = False
    agent: str = "qed_swe_bench"


# ---------------------------------------------------------------------------
# /api/summary
# ---------------------------------------------------------------------------


class SummaryResponse(BaseModel):
    total_runs: int
    n_benchmarks: int
    n_envs: int
    n_models: int
    by_status: list[StatusCount]
    by_model_top: list[ModelTopEntry]
    spend: Spend


# ---------------------------------------------------------------------------
# /api/benchmarks
# ---------------------------------------------------------------------------


class BenchmarkSummary(BaseModel):
    """One row of the benchmarks-list table.

    Fields are derived from `GROUP BY benchmark_id` aggregates; some
    columns can be NULL for benchmarks where every run failed before
    a score was recorded.
    """

    benchmark_id: str
    n_runs: int
    n_models: int
    n_envs: int
    n_seeds: int
    avg_score: float | None = None
    total_cost: float | None = None
    provenance: str | None = None
    agents: list[str] = []
    first_started: str | None = None
    last_finished: str | None = None


class BenchmarkDetail(BaseModel):
    """`/api/benchmarks/{id}` — one benchmark with all its runs.

    `models` / `envs` / `seeds` are sorted lists of distinct values
    across the benchmark's runs (cheap to compute server-side; saves
    the dashboard from doing the same dedup).
    """

    benchmark_id: str
    n_runs: int
    models: list[str]
    envs: list[str]
    seeds: list[int]
    spend: Spend
    runs: list[Run]


# ---------------------------------------------------------------------------
# /api/benchmarks/{id}/matrix
# ---------------------------------------------------------------------------


class MatrixCell(BaseModel):
    """One (model, env_id) cell of the capability matrix.

    `capabilities` is a cumulative-OR over the cell's seeds — a
    capability is True if any seed achieved it.
    """

    model: str
    env_id: str
    n_seeds: int
    avg_score: float
    total_cost: float
    capabilities: dict[str, bool]


class BenchmarkMatrix(BaseModel):
    benchmark_id: str
    models: list[str]
    envs: list[str]
    capability_flags: list[str]
    cells: list[MatrixCell]


# ---------------------------------------------------------------------------
# /api/runs
# ---------------------------------------------------------------------------


class RunsListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    runs: list[Run]


# ---------------------------------------------------------------------------
# /api/leaderboard
# ---------------------------------------------------------------------------


class LeaderboardEntry(BaseModel):
    model: str
    n_runs: int
    n_envs: int
    total_score: float | None = None
    avg_score: float | None = None
    total_cost: float | None = None
    avg_turns: float | None = None


# ---------------------------------------------------------------------------
# /api/envs and /api/models
# ---------------------------------------------------------------------------


class EnvSummary(BaseModel):
    """Combined view: registered envs (rlenv_images catalog) UNIONed
    with envs that have runs but aren't in the catalog."""

    model_config = ConfigDict(extra="allow")

    env_id: str
    image_ref: str = ""
    image_digest: str | None = None
    interface: str | None = None
    task_type: str = "binary_task"
    project: str | None = None
    bug_id: str | None = None
    capability_class: str | None = None
    expected_capabilities: list[Any] | None = None
    metadata: dict[str, Any] | None = None
    validation_status: str = "unregistered"
    n_runs: int = 0
    n_models: int = 0
    best_score: float | None = None
    registered: int = 0


class ModelSummary(BaseModel):
    model: str
    llm_route: str | None = None
    n_runs: int
    n_envs: int
    avg_score: float | None = None
    total_cost: float | None = None
