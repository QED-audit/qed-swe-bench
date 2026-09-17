"""SQLite schema for qed_swe_bench.

Two tables today:
  - `runs`         — one row per (benchmark_id, model, env_id, seed) episode.
  - `rlenv_images` — catalog of registered envs (lookup target for env_id).

Migration strategy: `CREATE TABLE IF NOT EXISTS` plus idempotent
`ADD COLUMN` for fields added after schema v1 (see LATE_COLUMNS). When the
schema grows past what idempotent DDL can handle, switch to Alembic.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

from qed_swe_bench.config import Config

# Bumping this requires a migration story for any existing DB. Today we
# rely on CREATE TABLE IF NOT EXISTS + idempotent ADD COLUMN for late
# additions; a future move to Alembic should replace that scheme.
SCHEMA_VERSION = 7

CREATE_RUNS_TABLE = """
CREATE TABLE IF NOT EXISTS runs (
    run_id                  TEXT PRIMARY KEY,
    benchmark_id            TEXT NOT NULL,
    model                   TEXT NOT NULL,
    env_id                  TEXT NOT NULL,
    image_ref               TEXT NOT NULL,
    image_digest            TEXT NOT NULL,
    task_type               TEXT NOT NULL,
    interface               TEXT,
    seed                    INTEGER NOT NULL,
    status                  TEXT NOT NULL,
    capabilities            TEXT,
    score                   REAL,
    tokens_in               INTEGER,
    tokens_out              INTEGER,
    tokens_cache_read       INTEGER,
    tokens_cache_creation   INTEGER,
    cost_usd                REAL,
    cost_source             TEXT,
    runtime_s               REAL,
    turns_used              INTEGER,
    exit_reason             TEXT,
    run_dir                 TEXT,
    started_at              TEXT,
    finished_at             TEXT,
    provenance              TEXT NOT NULL DEFAULT 'native',
    llm_route               TEXT,
    api_base                TEXT,
    failure_reason          TEXT,
    UNIQUE(benchmark_id, model, env_id, seed)
);
"""

# Columns added after schema v1. Idempotent via PRAGMA table_info check on init.
LATE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("interface", "TEXT"),
    # Reproducibility provenance (added 2026-05). git_sha records which
    # version of the runner code ran; repro_cmd is a copy-pasteable
    # single-tuple `qed_swe_bench benchmark` invocation that re-runs the
    # exact (model, env, seed) cell. Both nullable for backfill of
    # pre-existing rows; populated forward by `insert_queued`.
    ("git_sha", "TEXT"),
    ("repro_cmd", "TEXT"),
    # Heartbeat for stale-queued recovery. Each running orchestrator
    # periodically updates last_heartbeat on its own queued rows; the
    # resilience sweep treats a row as stale only if its heartbeat is
    # absent OR older than STALE_HEARTBEAT_AFTER. Replaces the prior
    # pure-timestamp scheme that mis-reaped sibling processes' live
    # rows. Nullable so legacy rows (pre-heartbeat) aren't marked
    # stale on first sweep — the sweep falls back to the old
    # started_at-based cutoff for those.
    ("last_heartbeat", "TEXT"),
    # Always-on diagnostics (added 2026-05 with turn-as-effort
    # methodology, see docs/decisions.md). Tracked by `Budget`
    # regardless of whether the corresponding budget is enforced, so
    # the same cell looks identical across "budget on" and "budget off"
    # runs except for the (optional) early termination. Nullable so
    # pre-existing rows (and partial / infra-failed rows) stay
    # representable.
    ("weighted_tokens_used", "INTEGER"),
    ("peak_per_turn_context", "INTEGER"),
    # Full benchmark YAML the row was produced under (raw text — comments
    # preserved). Stored verbatim from the source file (the same bytes
    # that land in `<run_dir>/config_snapshot.yaml` per D-10). Lets
    # `qed_swe_bench rerun <run_id>` re-parse and replay without the
    # source YAML being on disk. See docs/decisions.md D-13. Nullable so
    # legacy rows (pre-this column) stay representable; rerun falls
    # back to reading `<run_dir>/config_snapshot.yaml` for those.
    ("config_snapshot", "TEXT"),
    # Whether mid-episode scaffolding nudges were enabled for this run
    # (build_stuck_nudge / build_wrapup_nudge / build_voluntary_exit_nudge
    # in runner/loop.py). NOT NULL so every row carries a definite
    # answer; DEFAULT 0 covers existing native rows at migration time.
    # Rows from the older `import_eval_dir` path (provenance=
    # 'imported_from_eval') are backfilled to 1 by the one-shot
    # migration in init_db; bench-v8 generated those with nudges ON.
    # Newer importers (`import_legacy_vr_agent_dir`,
    # `scripts/import_codex_bench.py`) set nudges_used explicitly per
    # row, so the backfill stays narrow.
    ("nudges_used", "INTEGER NOT NULL DEFAULT 0"),
    # Identity of the agent harness that produced this row. Distinct
    # from `model` (which LLM) and `provenance` (data lineage):
    # `qed_swe_bench` runs talk to the LLM via the chat-completions loop
    # in runner/loop.py; `codex` runs talk through codex CLI's stateful
    # Responses API; `vr-agent` runs are historical imports from
    # Anthropic's research harness. Leaderboards group by
    # (model, agent) so the same model under different harnesses gets
    # distinct rows. NOT NULL DEFAULT 'qed_swe_bench' so existing rows
    # keep their semantics and new native rows don't need to specify
    # explicitly.
    ("agent", "TEXT NOT NULL DEFAULT 'qed_swe_bench'"),
)

CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS runs_bench ON runs(benchmark_id);",
    "CREATE INDEX IF NOT EXISTS runs_model ON runs(model);",
    "CREATE INDEX IF NOT EXISTS runs_env   ON runs(env_id);",
]


# ---------------------------------------------------------------------------
# rlenv_images: registered envs.
#
# An env enters the catalog via `qed_swe_bench register-image` or
# `register-dir`. Once in the catalog, an env can be referenced by env_id
# in benchmark configs without an explicit image_ref (looked up here).
# Validation status is recorded but not enforced.
# ---------------------------------------------------------------------------

CREATE_RLENV_IMAGES_TABLE = """
CREATE TABLE IF NOT EXISTS rlenv_images (
    env_id              TEXT PRIMARY KEY,
    image_ref           TEXT NOT NULL,
    image_digest        TEXT,
    interface           TEXT NOT NULL,
    task_type           TEXT NOT NULL,
    project             TEXT,                  -- 'v8' | 'bountybench' | ...
    bug_id              TEXT,                  -- canonical id (CVE-XXXX, crbug-XXXXX)
    capability_class    TEXT,                  -- 'sandbox_escape'|'renderer_rce'|'info_leak'|'other'
    expected_capabilities TEXT,                -- JSON array of flag names (subset of grader's capability_flags)
    metadata            TEXT,                  -- JSON: free-form (e.g., crev, fix_crev, subsystem, bug_class)
    validation_status   TEXT NOT NULL DEFAULT 'unvalidated',  -- 'unvalidated' | 'pass_verified' | 'pass_unverified' | 'fail' | 'quarantined'
    last_validated_at   TEXT,
    registered_at       TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_RLENV_IMAGES_INDEXES = [
    "CREATE INDEX IF NOT EXISTS rlenv_project ON rlenv_images(project);",
    "CREATE INDEX IF NOT EXISTS rlenv_iface ON rlenv_images(interface);",
    "CREATE INDEX IF NOT EXISTS rlenv_status ON rlenv_images(validation_status);",
]

CREATE_META_TABLE = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    cfg = Config.from_env()
    path = db_path or cfg.db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path), isolation_level=None)  # autocommit
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA foreign_keys=ON;")
    return con


def _existing_columns(con: sqlite3.Connection, table: str) -> set[str]:
    rows = con.execute(f"PRAGMA table_info({table})").fetchall()
    return {r["name"] for r in rows}


def init_db(db_path: Path | None = None) -> Path:
    """Create the schema if missing; idempotently add late columns."""
    cfg = Config.from_env()
    path = db_path or cfg.db_path
    with connect(path) as con:
        # runs table + late columns
        con.execute(CREATE_RUNS_TABLE)
        existing = _existing_columns(con, "runs")
        newly_added: set[str] = set()
        for col_name, col_type in LATE_COLUMNS:
            if col_name not in existing:
                con.execute(f"ALTER TABLE runs ADD COLUMN {col_name} {col_type}")
                newly_added.add(col_name)
        # One-shot backfill: when nudges_used is first added, flip rows
        # from the older `import_eval_dir` path (provenance=
        # 'imported_from_eval') to 1. bench-v8 ran nudges ON, so the
        # source-benchmark policy substitutes for missing per-eval-dir
        # metadata. Runs once per DB; subsequent connects find the
        # column already present and skip this branch. Newer importers
        # (`import_legacy_vr_agent_dir`, `scripts/import_codex_bench.py`)
        # set nudges_used explicitly per row, so the backfill stays
        # narrow.
        if "nudges_used" in newly_added:
            con.execute(
                "UPDATE runs SET nudges_used = 1 "
                "WHERE provenance = 'imported_from_eval'"
            )
        # One-shot backfill: when `agent` is first added, retro-label
        # the import provenances. `imported_from_legacy` and
        # `imported_from_eval` are both vr-agent (the upstream agent
        # was renamed from `bench-v8` to `vr-agent`; the two provenance
        # values map to two on-disk archive shapes from successive
        # cycles, not two distinct agents). `imported_from_codex` rows
        # came from `scripts/import_codex_bench.py`. Forward importers
        # also set `agent` explicitly so this only affects rows that
        # existed before the column did.
        if "agent" in newly_added:
            con.execute(
                "UPDATE runs SET agent = 'vr-agent' "
                "WHERE provenance IN ('imported_from_legacy', 'imported_from_eval')"
            )
            con.execute(
                "UPDATE runs SET agent = 'codex' "
                "WHERE provenance = 'imported_from_codex'"
            )
        for stmt in CREATE_INDEXES:
            con.execute(stmt)
        # rlenv_images table + indexes
        con.execute(CREATE_RLENV_IMAGES_TABLE)
        for stmt in CREATE_RLENV_IMAGES_INDEXES:
            con.execute(stmt)
        # meta
        con.execute(CREATE_META_TABLE)
        con.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?);",
            (str(SCHEMA_VERSION),),
        )
    return path


@contextmanager
def transaction(db_path: Path | None = None):
    """Context manager yielding a connection in a transaction."""
    con = connect(db_path)
    try:
        con.execute("BEGIN;")
        yield con
        con.execute("COMMIT;")
    except Exception:
        con.execute("ROLLBACK;")
        raise
    finally:
        con.close()
