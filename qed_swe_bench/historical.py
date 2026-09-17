"""Ingest bench-v8's eval/<bug>_runN/ directories as historical runs rows.

These are the ~70 Claude Opus 4.6 runs the user has already produced. They
land in our SQLite with provenance='imported_from_eval' so the aggregate /
leaderboard tables include the existing baseline alongside any new
qed_swe_bench runs.

For each eval directory we read:
  config.json        — model id, bench-v8 image (registry-qualified)
  grade_calls.jsonl  — capabilities (via best_caps_from_grade_log)
  transcript.jsonl   — token sums (input/output/cache_read/cache_creation)

Schema gaps for historical rows:
  image_digest    — bench-v8 recorded a registry tag, not a digest. Stored
                    as 'imported:<bug>' so the column stays non-null and
                    queryable but we don't pretend we know the real sha256.
  task_type       — assumed 'binary_task' (V8 only in eval/).
  llm_route       — 'imported' so it's distinguishable in the UI.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from qed_swe_bench.db.schema import init_db, transaction
from qed_swe_bench.runner.capabilities import (
    DEFAULT_SCORING_POLICY,
    best_caps_from_grade_log,
    compute_score,
)
from qed_swe_bench.runner.cost import compute_total_cost
from qed_swe_bench.runner.llm.base import NormalizedUsage

log = logging.getLogger(__name__)


_RUN_DIR_PAT = re.compile(r"^(?P<bug>.+)_run(?P<seed>\d+)$")


@dataclass(frozen=True)
class ImportedRun:
    bug: str
    seed: int
    model: str
    image_ref: str
    capabilities: dict[str, bool]
    score: float
    usage: NormalizedUsage
    runtime_s: float | None
    turns_used: int
    grade_count: int
    # Per-call NormalizedUsage records reconstructed from the saved
    # transcript ai entries. Lets cost compute_total_cost route each call
    # to the correct long-context tier rather than over-pricing the
    # aggregate.
    per_call_usages: tuple[NormalizedUsage, ...] = ()


def _parse_dir_name(name: str) -> tuple[str, int] | None:
    m = _RUN_DIR_PAT.match(name)
    if not m:
        return None
    return m.group("bug"), int(m.group("seed"))


def _read_config(eval_dir: Path) -> dict:
    cfg_path = eval_dir / "config.json"
    if cfg_path.exists():
        try:
            return json.loads(cfg_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log.warning("config.json malformed in %s", eval_dir)
    return {}


def _sum_usage(
    transcript: Path,
    *,
    model_id: str | None = None,
) -> tuple[NormalizedUsage, int, tuple[NormalizedUsage, ...]]:
    """Sum per-turn `usage` blocks recorded in transcript.jsonl and
    return the per-turn list alongside the aggregate.

    Returns (totals, ai_turn_count, per_call_usages). The aggregate is
    kept for the existing tokens_* columns; the per-call tuple feeds
    cost.compute_total_cost so each call is priced at its own tier.

    Cache semantics: historical LiteLLM-written transcripts (everything
    routed through the LiteLLM client before 2026-05-13) recorded
    input_tokens INCLUSIVE of cache_read (matching the OpenAI/LiteLLM
    convention). Anthropic-native transcripts (and post-fix LiteLLM)
    record input_tokens DISJOINT from cache_read.

    compute_cost now expects DISJOINT input_tokens for all providers, so
    when ``model_id`` is supplied and identifies a LiteLLM-routed model
    (i.e. NOT ``anthropic/*``), we subtract cache_read + cache_creation
    here to convert legacy inclusive entries into the disjoint shape.
    Calls without ``model_id`` get the raw transcript values (legacy
    behavior) for callers that don't want the conversion.
    """
    should_convert_inclusive = (
        isinstance(model_id, str)
        and bool(model_id)
        and not model_id.startswith("anthropic/")
    )

    in_t = out_t = cr = cc = 0
    turns = 0
    per_call: list[NormalizedUsage] = []
    if not transcript.exists():
        return NormalizedUsage(), 0, ()
    with transcript.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("role") != "ai":
                continue
            turns += 1
            u = d.get("usage") or {}
            raw_in = int(u.get("input_tokens", 0) or 0)
            call_out = int(u.get("output_tokens", 0) or 0)
            call_cr = int(u.get("cache_read", 0) or 0)
            call_cc = int(u.get("cache_creation", 0) or 0)
            if should_convert_inclusive:
                # LiteLLM legacy inclusive → disjoint
                call_in = max(0, raw_in - call_cr - call_cc)
            else:
                call_in = raw_in
            in_t += call_in
            out_t += call_out
            cr += call_cr
            cc += call_cc
            per_call.append(NormalizedUsage(
                input_tokens=call_in,
                output_tokens=call_out,
                cache_read_tokens=call_cr,
                cache_creation_tokens=call_cc,
            ))
    return (
        NormalizedUsage(
            input_tokens=in_t,
            output_tokens=out_t,
            cache_read_tokens=cr,
            cache_creation_tokens=cc,
        ),
        turns,
        tuple(per_call),
    )


def _grade_count(grade_log: Path) -> int:
    if not grade_log.exists():
        return 0
    n = 0
    with grade_log.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _runtime_s_from_transcript(transcript: Path) -> float | None:
    """Approximate runtime from first → last ts in transcript.jsonl."""
    first = last = None
    if not transcript.exists():
        return None
    with transcript.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = d.get("ts")
            if not ts:
                continue
            if first is None:
                first = ts
            last = ts
    if not first or not last:
        return None
    try:
        a = datetime.fromisoformat(first)
        b = datetime.fromisoformat(last)
    except ValueError:
        return None
    return max(0.0, (b - a).total_seconds())


def parse_eval_run(
    eval_dir: Path, *, default_model: str = "anthropic/claude-opus-4-6"
) -> ImportedRun | None:
    """Read a single bench-v8 eval/<bug>_runN/ dir into an ImportedRun."""
    parsed = _parse_dir_name(eval_dir.name)
    if parsed is None:
        return None
    bug, seed = parsed

    config = _read_config(eval_dir)
    model = config.get("model") or default_model
    image_ref = config.get("image") or f"imported://bench-v8/{bug}"

    caps = best_caps_from_grade_log(eval_dir / "grade_calls.jsonl")
    score = compute_score(caps, DEFAULT_SCORING_POLICY)
    usage, turns, per_call = _sum_usage(eval_dir / "transcript.jsonl")
    runtime_s = _runtime_s_from_transcript(eval_dir / "transcript.jsonl")
    grade_count = _grade_count(eval_dir / "grade_calls.jsonl")

    return ImportedRun(
        bug=bug,
        seed=seed,
        model=model,
        image_ref=image_ref,
        capabilities=caps,
        score=score,
        usage=usage,
        runtime_s=runtime_s,
        turns_used=turns,
        grade_count=grade_count,
        per_call_usages=per_call,
    )


def import_eval_dir(
    eval_root: Path,
    *,
    nudges_used: bool,
    benchmark_id: str = "imported-opus",
    db_path: Path | None = None,
    default_model: str = "anthropic/claude-opus-4-6",
) -> dict[str, int]:
    """Walk eval_root for <bug>_runN/ subdirs and insert each as a runs row.

    Returns a status histogram suitable for CLI display.

    Idempotent: thanks to UNIQUE(benchmark_id, model, env_id, seed), re-running
    against the same eval dir is a no-op for already-imported rows.

    `nudges_used` is required and explicit: vr-agent eval dirs do not
    record nudge state in `config.json`, and the importer cannot infer
    it from the transcript reliably. The caller must state what the
    source benchmark ran with — typically `True` for the imported-opus
    baseline (bench-v8 ran nudges ON; see `benchmarks/v8.yaml:225-232`).
    """
    init_db(db_path)
    histogram: dict[str, int] = {"imported": 0, "skipped_duplicate": 0, "unparseable": 0}
    candidates = sorted(p for p in eval_root.iterdir() if p.is_dir())
    for ed in candidates:
        run = parse_eval_run(ed, default_model=default_model)
        if run is None:
            histogram["unparseable"] += 1
            continue
        # Cost: try our pricing table; on miss, leave None. Per-call
        # so tier-priced models route each call to its own bracket.
        cost_usd, cost_source = compute_total_cost(run.model, list(run.per_call_usages))
        run_id = uuid.uuid4().hex[:16]
        try:
            with transaction(db_path) as con:
                con.execute(
                    """
                    INSERT INTO runs (
                        run_id, benchmark_id, model, env_id, image_ref, image_digest,
                        task_type, seed, status, capabilities, score,
                        tokens_in, tokens_out, tokens_cache_read, tokens_cache_creation,
                        cost_usd, cost_source,
                        runtime_s, turns_used, exit_reason, run_dir,
                        started_at, finished_at, provenance, llm_route, nudges_used, agent
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'succeeded', ?, ?,
                              ?, ?, ?, ?, ?, ?, ?, ?, 'imported', ?, NULL, NULL, 'imported_from_eval', 'imported', ?, 'vr-agent')
                    """,
                    (
                        run_id,
                        benchmark_id,
                        run.model,
                        run.bug,
                        run.image_ref,
                        f"imported:{run.bug}",
                        "binary_task",
                        run.seed,
                        json.dumps(run.capabilities),
                        run.score,
                        run.usage.input_tokens,
                        run.usage.output_tokens,
                        run.usage.cache_read_tokens,
                        run.usage.cache_creation_tokens,
                        cost_usd,
                        cost_source,
                        run.runtime_s,
                        run.turns_used,
                        str(ed),
                        1 if nudges_used else 0,
                    ),
                )
            histogram["imported"] += 1
        except sqlite3.IntegrityError:
            histogram["skipped_duplicate"] += 1
    return histogram


# ---------------------------------------------------------------------------
# Native run-dir import (D-10 bijection)
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> dict:
    """Read a JSON file; return empty dict if missing or malformed.

    Used by import_native_run_dir for optional artifacts (score.json /
    cost.json may be absent on a partially-written run-dir, e.g. one
    that crashed mid-episode). job.json is required and not loaded
    through this helper.
    """
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def import_native_run_dir(run_dir: Path) -> dict:
    """Read a run-dir and produce a dict suitable for `runs` table insert.

    Reads job.json (required) + score.json + cost.json (both optional;
    a partial run-dir without them is allowed and produces a row in
    'queued' status with NULL telemetry columns). The bijection (D-10)
    requires every DB column have a flat-text source; this function
    materializes that mapping.

    Raises FileNotFoundError if job.json is missing — that's the one
    artifact every run-dir must have.
    """
    job_path = run_dir / "job.json"
    if not job_path.is_file():
        raise FileNotFoundError(f"no job.json in {run_dir}")
    job = json.loads(job_path.read_text(encoding="utf-8"))
    score = _read_json(run_dir / "score.json")
    cost = _read_json(run_dir / "cost.json")
    # config_snapshot.yaml: verbatim source YAML the run was produced
    # under (D-13). Optional — a partial run-dir or a pre-D-13 run
    # may not have it; in that case the column stays NULL and
    # `qed_swe_bench rerun` rejects with a clear error.
    config_snapshot_path = run_dir / "config_snapshot.yaml"
    config_snapshot = (
        config_snapshot_path.read_text(encoding="utf-8")
        if config_snapshot_path.is_file() else None
    )

    capabilities_json = (
        json.dumps(score.get("capabilities", {}))
        if score.get("capabilities") is not None else None
    )

    # Mid-episode nudges (build_stuck_nudge / build_wrapup_nudge /
    # build_voluntary_exit_nudge). The DB column is NOT NULL so we
    # always produce a definite value via this fallback chain:
    #   1. job.json["nudges_used"] — authoritative; written by the
    #      runner since the column landed.
    #   2. config_snapshot.yaml's `nudges:` field — for run-dirs that
    #      predate the job.json field but recorded the source YAML
    #      verbatim under D-13. parse_nudges accepts bool/list/null
    #      (matches the orchestrator-time parser exactly), so any
    #      YAML the runner could have produced round-trips here.
    #   3. False with warning — for very old run-dirs without either.
    if "nudges_used" in job:
        nudges_used = bool(job["nudges_used"])
    elif config_snapshot is not None:
        try:
            import yaml
            from qed_swe_bench.runner.orchestrator_config import parse_nudges
            yaml_obj = yaml.safe_load(config_snapshot) or {}
            nudges_used = bool(parse_nudges(yaml_obj.get("nudges")))
        except (yaml.YAMLError, ValueError) as exc:
            log.warning(
                "%s: config_snapshot.yaml unparseable for nudges (%s); "
                "defaulting to False",
                run_dir, exc,
            )
            nudges_used = False
    else:
        log.warning(
            "%s: no nudges_used in job.json and no config_snapshot.yaml; "
            "defaulting to False. If this run was actually nudged, fix "
            "it manually.",
            run_dir,
        )
        nudges_used = False

    # status precedence: explicit value in score.json (preferred) →
    # inferred from presence of score data → queued (no score yet)
    if score.get("status"):
        status = score["status"]
        # Defensive: `running` is a runtime-only state. score.json is only
        # written at episode end, so a `running` value here means the
        # source DB was scraped mid-flight (e.g. qed_swe_bench export run
        # while a sibling orchestrator was still working). Treat as
        # queued so a re-import doesn't permanently pin the row to a
        # state no live process owns.
        if status == "running":
            status = "queued"
    elif score:
        # score.json exists but no explicit status — treat as succeeded.
        # This handles legacy run-dirs written before the status field
        # was added.
        status = "succeeded"
    else:
        status = "queued"

    return {
        "run_id": job["run_id"],
        "benchmark_id": job["benchmark_id"],
        "model": job["model"],
        "env_id": job["env_id"],
        "image_ref": job["image_ref"],
        "image_digest": job["image_digest"],
        "task_type": job["task_type"],
        "interface": job.get("interface"),
        "seed": job["seed"],
        "status": status,
        "capabilities": capabilities_json,
        "score": score.get("score"),
        "tokens_in": cost.get("tokens_in"),
        "tokens_out": cost.get("tokens_out"),
        "tokens_cache_read": cost.get("tokens_cache_read"),
        "tokens_cache_creation": cost.get("tokens_cache_creation"),
        "cost_usd": cost.get("cost_usd"),
        "cost_source": cost.get("cost_source"),
        "runtime_s": score.get("runtime_s"),
        "turns_used": score.get("turns_used"),
        "exit_reason": score.get("exit_reason"),
        "run_dir": str(run_dir),
        "started_at": job["started_at"],
        "finished_at": score.get("finished_at"),
        # Provenance default 'native' covers (a) old job.json files
        # written before the field existed and (b) imported_from_eval
        # rows that don't go through this code path. Mock and any
        # future provenances round-trip via job.json.
        "provenance": job.get("provenance", "native"),
        "llm_route": cost.get("llm_route"),
        "api_base": cost.get("api_base"),
        "failure_reason": score.get("failure_reason"),
        "git_sha": job.get("git_sha"),
        "repro_cmd": job.get("repro_cmd"),
        "last_heartbeat": None,  # heartbeats are runtime-only
        # Always-on diagnostics (turn-as-effort, decisions.md). Nullable
        # for run-dirs from before the field existed.
        "weighted_tokens_used": score.get("weighted_tokens_used"),
        "peak_per_turn_context": score.get("peak_per_turn_context"),
        # Source YAML the row was produced under — D-13. Nullable for
        # run-dirs that predate the column.
        "config_snapshot": config_snapshot,
        # Mid-episode nudges. NOT NULL on the DB column; coerced to 0/1
        # here so the value lands as INTEGER not Python bool.
        "nudges_used": 1 if nudges_used else 0,
        # Agent harness identity. Defaults to 'qed_swe_bench' for
        # pre-column job.json files. Importers that produce non-native
        # rows (codex, vr-agent) write the right value into job.json so
        # the round-trip lands here.
        "agent": job.get("agent", "qed_swe_bench"),
    }


def import_native_runs(
    root: Path, *, db_path: Path | None = None,
) -> dict[str, int]:
    """Walk `root` recursively, find every job.json, and ingest its
    run-dir into the runs DB.

    Idempotent via the existing UNIQUE(benchmark_id, model, env_id,
    seed) constraint: re-imports of the same logical tuple are
    silently skipped (counted as `skipped_duplicate`).

    Histogram keys: imported, skipped_duplicate, unparseable.
    """
    init_db(db_path)
    histogram: dict[str, int] = {
        "imported": 0,
        "skipped_duplicate": 0,
        "unparseable": 0,
    }
    for job_path in root.rglob("job.json"):
        run_dir = job_path.parent
        try:
            row = import_native_run_dir(run_dir)
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as exc:
            log.warning("skip %s: %s", run_dir, exc)
            histogram["unparseable"] += 1
            continue
        cols = ", ".join(row.keys())
        placeholders = ", ".join("?" for _ in row)
        try:
            with transaction(db_path) as con:
                con.execute(
                    f"INSERT OR IGNORE INTO runs ({cols}) "
                    f"VALUES ({placeholders})",
                    tuple(row.values()),
                )
                # rowcount on INSERT OR IGNORE: 1 if inserted, 0 if dup
                if con.execute(
                    "SELECT changes()"
                ).fetchone()[0] == 0:
                    histogram["skipped_duplicate"] += 1
                else:
                    histogram["imported"] += 1
        except sqlite3.Error as exc:
            log.warning("DB error inserting %s: %s", row["run_id"], exc)
            histogram["unparseable"] += 1
    return histogram


# ---------------------------------------------------------------------------
# Path migration: legacy run-dir shape → D-10 layered layout
# ---------------------------------------------------------------------------


def migrate_runs_to_layered_layout(
    runs_root: Path,
    *,
    legacy_host: str = "legacy",
    dry_run: bool = False,
    db_path: Path | None = None,
) -> dict[str, int]:
    """One-shot: move legacy-shaped run-dirs into the D-10 layered
    layout `runs/<benchmark_id>/<host>/<datetime>/<run_id>/`.

    Algorithm:
      1. Walk runs.run_dir from the DB (canonical pointer).
      2. For each row whose path is the legacy shape and exists on
         disk, compute the new path using job.json's `started_at` for
         <datetime> and `legacy_host` for <host> (we don't know the
         original host post-hoc).
      3. Two-phase move: shutil.copytree old → new; update DB
         `runs.run_dir` to absolute new path; shutil.rmtree old.
      4. Idempotent: if the new path already exists, skip.

    `dry_run=True` reports what would be moved without touching disk
    or DB.

    Refuses to run if any DB row is `status='queued'` with a recent
    `last_heartbeat` (within QED_SWE_BENCH_STALE_HEARTBEAT_MIN, default
    5 min) — there's an active orchestrator and migrating mid-flight
    would race.

    Histogram keys: migrated, skipped_already_new, missing_on_disk,
    error, dry_run_only.
    """
    import shutil
    from datetime import UTC, timedelta
    from qed_swe_bench.runner.run_dir import construct_run_dir_path

    init_db(db_path)
    histogram: dict[str, int] = {
        "migrated": 0,
        "skipped_already_new": 0,
        "missing_on_disk": 0,
        "error": 0,
        "dry_run_only": 0,
    }

    runs_root = runs_root.resolve()

    # Liveness check: refuse if a sibling orchestrator is still working.
    with transaction(db_path) as con:
        live = con.execute(
            "SELECT COUNT(*) FROM runs WHERE status='queued' "
            "AND last_heartbeat IS NOT NULL "
            "AND last_heartbeat > datetime('now', '-5 minutes')"
        ).fetchone()[0]
    if live > 0:
        raise RuntimeError(
            f"refusing to migrate while {live} queued row(s) have "
            f"recent heartbeats — orchestrators are still running. "
            f"Wait for them to finish, then re-run."
        )

    with transaction(db_path) as con:
        rows = con.execute(
            "SELECT run_id, benchmark_id, run_dir, started_at FROM runs "
            "WHERE run_dir IS NOT NULL"
        ).fetchall()

    for r in rows:
        old = Path(r["run_dir"])
        if not old.is_absolute():
            old = (runs_root.parent / old).resolve()
        # Recognize the new shape by depth: runs_root/<bid>/<host>/<dt>/<rid>/
        # has 4 segments after runs_root. Legacy is 2.
        try:
            rel_parts = old.relative_to(runs_root).parts
        except ValueError:
            # Path doesn't live under runs_root (e.g. tier2/, imported eval/).
            # Skip — out of scope for the layered layout.
            continue

        if len(rel_parts) == 4 and rel_parts[-1] == r["run_id"]:
            histogram["skipped_already_new"] += 1
            continue
        if not old.is_dir():
            log.warning("missing on disk: %s (run_id=%s)", old, r["run_id"])
            histogram["missing_on_disk"] += 1
            continue

        # Compute the new path. Use started_at from the DB; fall back
        # to file mtime if absent.
        from datetime import datetime as _dt
        try:
            ts = _dt.fromisoformat(r["started_at"]) if r["started_at"] else _dt.fromtimestamp(old.stat().st_mtime, tz=UTC)
        except Exception:  # noqa: BLE001
            ts = _dt.fromtimestamp(old.stat().st_mtime, tz=UTC)

        new = construct_run_dir_path(
            runs_root,
            benchmark_id=r["benchmark_id"],
            run_id=r["run_id"],
            ts=ts,
            host_override=legacy_host,
        )
        if new.exists():
            histogram["skipped_already_new"] += 1
            continue

        if dry_run:
            log.info("DRY RUN: would move %s → %s", old, new)
            histogram["dry_run_only"] += 1
            continue

        try:
            new.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(old, new)
        except Exception as exc:  # noqa: BLE001
            log.error("copy failed for %s → %s: %s", old, new, exc)
            histogram["error"] += 1
            # Clean up partial copy if it exists
            if new.exists():
                shutil.rmtree(new, ignore_errors=True)
            continue

        # Update DB to point at the new location BEFORE deleting old.
        try:
            with transaction(db_path) as con:
                con.execute(
                    "UPDATE runs SET run_dir = ? WHERE run_id = ?",
                    (str(new), r["run_id"]),
                )
        except sqlite3.Error as exc:
            log.error("DB update failed for %s: %s", r["run_id"], exc)
            shutil.rmtree(new, ignore_errors=True)
            histogram["error"] += 1
            continue

        # Now safe to remove the old location.
        shutil.rmtree(old, ignore_errors=True)
        histogram["migrated"] += 1

    return histogram


# ---------------------------------------------------------------------------
# Legacy vr-agent import (sensitive — source kept under tmp/, never committed)
#
# Source layout (the dataset shipped from the older vr-agent codebase):
#   <root>/scores/manifest.csv        — flat row per (model, cve, rep)
#   <root>/scores/per-cell.csv        — same rows, more columns
#   <root>/transcripts/<model>/<bug>/rep<N>/{result.json, config.json,
#                                            transcript.jsonl, tool_calls.jsonl,
#                                            grade_calls.jsonl, agent-run.log}
# `result.json` is authoritative for per-run summary fields (cost_usd,
# agent_turns, agent_duration_s, grade_bitmap, token_totals); manifest.csv
# is the only place the `excluded` flag lives.
#
# Neither `import_eval_dir` (bench-v8 eval/<bug>_runN/ shape) nor
# `import_native_runs` (D-10 layered runs/<bench>/<host>/<dt>/<run_id>/)
# matches this layout, hence a third importer.
# ---------------------------------------------------------------------------


# vr-agent uses bare slugs like "claude-opus-4-6"; qed_swe_bench uses the
# qualified provider/model form. Override via the importer's `model_map=`
# kwarg if the dataset ever contains a slug not in this default table.
_DEFAULT_VR_AGENT_MODEL_MAP: dict[str, str] = {
    "claude-opus-4-6": "anthropic/claude-opus-4-6",
    "claude-sonnet-4-6": "anthropic/claude-sonnet-4-6",
    "claude-sonnet-4-5": "anthropic/claude-sonnet-4-5",
}


_ENV_ID_ALIASES = {
    "v8-cve-2024-1939": "v8-e01",
    "v8-cve-2024-6100": "v8-e02",
    "v8-cve-2024-10231": "v8-e03",
    "v8-crbug-378779897": "v8-e04",
    "v8-cve-2024-10230": "v8-e05",
    "v8-cve-2024-12053": "v8-e06",
    "v8-cve-2024-2887": "v8-e07",
    "v8-cve-2024-7971": "v8-e08",
    "v8-cve-2024-8194": "v8-e09",
    "v8-cve-2024-9122": "v8-e10",
    "v8-cve-2024-9602": "v8-e11",
    "v8-cve-2024-9859": "v8-e12",
    "v8-cve-2025-0291": "v8-e13",
    "v8-cve-2025-0995": "v8-e14",
    "v8-cve-2025-13226": "v8-e15",
    "v8-cve-2025-5959": "v8-e16",
    "v8-cve-2026-2649": "v8-e17",
    "v8-cve-2023-6702": "v8-e18",
    "v8-cve-2024-0517": "v8-e19",
    "v8-cve-2024-0519": "v8-e20",
    "v8-cve-2024-3159": "v8-e21",
    "v8-cve-2024-4947": "v8-e22",
    "v8-crbug-339064932": "v8-e23",
    "v8-crbug-386565144": "v8-e24",
    "v8-crbug-1509576": "v8-e25",
    "v8-cve-2024-5274": "v8-e26",
    "v8-cve-2024-7965": "v8-e27",
    "v8-cve-2025-10891": "v8-e28",
    "v8-cve-2025-12727": "v8-e29",
    "v8-cve-2025-13223": "v8-e30",
    "v8-cve-2025-1920": "v8-e31",
    "v8-cve-2025-2135": "v8-e32",
    "v8-cve-2025-5419": "v8-e33",
    "v8-cve-2025-6554": "v8-e34",
    "v8-cve-2025-8010": "v8-e35",
    "v8-cve-2025-9132": "v8-e36",
    "v8-cve-2026-3910": "v8-e37",
    "v8-cve-2026-4447": "v8-e38",
    "v8-crbug-339736513": "v8-e39",
    "v8-crbug-403364367": "v8-e40",
    "v8-cve-2024-4761": "v8-e41",
}


def _normalize_vr_agent_env_id(bug: str) -> str:
    """Legacy bug id to the current environment-id scheme.

    Historical imports keep the old CVE/crbug dirnames as parsed input and
    surface the neutral env id the rest of the pipeline uses.
    """
    low = bug.lower()
    if not low.startswith("v8-"):
        low = f"v8-{low}"
    return _ENV_ID_ALIASES.get(low, low)


def parse_legacy_vr_agent_run(rep_dir: Path) -> dict | None:
    """Read one transcripts/<model>/<bug>/rep<N>/ dir into a column-shaped dict.

    Returns None if the directory doesn't contain a parseable result.json
    (the only required artifact). Caller is responsible for combining this
    with the manifest.csv `excluded` flag and inserting into the runs
    table.

    Why a dict rather than ImportedRun: the legacy dataset carries fields
    (`cost_usd` directly, `error`, `agent_run_ok`) that don't fit the
    bench-v8 ImportedRun shape, and the importer needs to set status
    based on agent_run_ok. Returning a flat dict makes the insert call
    site read like `import_native_run_dir`, which is the closer
    structural relative.
    """
    result_path = rep_dir / "result.json"
    if not result_path.is_file():
        return None
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        log.warning("malformed result.json in %s", rep_dir)
        return None

    bug = result.get("cve")
    seed = result.get("run")
    raw_model = result.get("model")
    if not bug or seed is None or not raw_model:
        log.warning("result.json missing identity fields in %s", rep_dir)
        return None

    grade_bitmap = result.get("grade_bitmap") or {}
    n_caps = result.get("n_caps")
    score = float(n_caps) if n_caps is not None else float(sum(
        1 for v in grade_bitmap.values() if v
    ))

    token_totals = result.get("token_totals") or {}
    error_msg = result.get("error") or ""
    agent_run_ok = bool(result.get("agent_run_ok"))

    # status: succeeded if the agent finished cleanly, model_failed if not
    # (timeout, error mid-episode). The infrastructure_failed status is
    # set later from manifest.csv's `excluded` column.
    status = "succeeded" if agent_run_ok else "model_failed"
    exit_reason = error_msg.strip() or ("imported_legacy" if agent_run_ok
                                        else "agent_run_not_ok")

    started_at = None
    finished_at = None
    try:
        mtime = result_path.stat().st_mtime
        from datetime import datetime as _dt, UTC as _UTC
        finished_at_dt = _dt.fromtimestamp(mtime, tz=_UTC)
        finished_at = finished_at_dt.isoformat()
        agent_duration = result.get("agent_duration_s")
        if agent_duration is not None:
            from datetime import timedelta as _td
            started_at = (
                finished_at_dt - _td(seconds=float(agent_duration))
            ).isoformat()
    except OSError:
        pass

    config_path = rep_dir / "config.json"
    image_ref = f"imported://vr-agent/{bug}"
    if config_path.is_file():
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            image_ref = cfg.get("image") or cfg.get("image_ref") or image_ref
        except json.JSONDecodeError:
            pass

    return {
        # identity
        "raw_bug": bug,
        "raw_model": raw_model,
        "seed": int(seed),
        # outcome
        "status": status,
        "capabilities_json": json.dumps(grade_bitmap),
        "score": score,
        # telemetry
        "tokens_in": token_totals.get("in"),
        "tokens_out": token_totals.get("out"),
        "tokens_cache_read": token_totals.get("cache_read"),
        "tokens_cache_creation": token_totals.get("cache_creat"),
        "cost_usd": result.get("cost_usd"),
        "runtime_s": result.get("agent_duration_s"),
        "turns_used": result.get("agent_turns"),
        "exit_reason": exit_reason,
        # paths + provenance
        "image_ref": image_ref,
        "image_digest": f"imported:{bug}",
        "run_dir": str(rep_dir),
        "started_at": started_at,
        "finished_at": finished_at,
    }


def _read_excluded_index(manifest_path: Path) -> dict[tuple[str, str, int], str]:
    """Load scores/manifest.csv into an index keyed by (model, cve, rep).

    Returns a map where the value is the `excluded` reason string (or
    "" if the row was kept). Only rows whose `excluded` column is
    non-empty matter for the importer's status assignment, but we
    materialize the full index so callers can also detect rows that
    have a result.json but were dropped from the published set.
    """
    import csv
    index: dict[tuple[str, str, int], str] = {}
    if not manifest_path.is_file():
        return index
    with manifest_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rep = int(row["rep"])
            except (KeyError, ValueError):
                continue
            key = (row.get("model", ""), row.get("cve", ""), rep)
            index[key] = (row.get("excluded") or "").strip()
    return index


def import_legacy_vr_agent_dir(
    root: Path,
    *,
    benchmark_id: str = "imported-vr-agent",
    model_map: dict[str, str] | None = None,
    db_path: Path | None = None,
) -> dict[str, int]:
    """Walk `<root>/transcripts/<model>/<bug>/rep<N>/` and ingest each
    rep dir as one runs row.

    `<root>/scores/manifest.csv` is consulted for the `excluded` flag;
    excluded rows are inserted with status='infrastructure_failed' (so
    they're still counted as "attempted" but distinguishable from the
    real successes / failures).

    Idempotent: thanks to UNIQUE(benchmark_id, model, env_id, seed),
    re-runs against the same root are a no-op for already-imported rows.

    Histogram keys: imported, skipped_duplicate, unparseable, excluded.
    """
    init_db(db_path)
    histogram: dict[str, int] = {
        "imported": 0,
        "skipped_duplicate": 0,
        "unparseable": 0,
        "excluded": 0,
    }
    map_ = dict(_DEFAULT_VR_AGENT_MODEL_MAP)
    if model_map:
        map_.update(model_map)

    transcripts_root = root / "transcripts"
    if not transcripts_root.is_dir():
        log.warning("no transcripts/ subdir in %s", root)
        return histogram

    excluded_index = _read_excluded_index(root / "scores" / "manifest.csv")

    rep_dirs: list[Path] = []
    for model_dir in sorted(transcripts_root.iterdir()):
        if not model_dir.is_dir():
            continue
        for bug_dir in sorted(model_dir.iterdir()):
            if not bug_dir.is_dir():
                continue
            for rep_dir in sorted(bug_dir.iterdir()):
                if rep_dir.is_dir() and rep_dir.name.startswith("rep"):
                    rep_dirs.append(rep_dir)

    for rep_dir in rep_dirs:
        parsed = parse_legacy_vr_agent_run(rep_dir)
        if parsed is None:
            histogram["unparseable"] += 1
            continue

        # Apply manifest-exclusion override AFTER parsing so we still
        # count the row in `excluded` rather than dropping it silently.
        excluded_reason = excluded_index.get(
            (parsed["raw_model"], parsed["raw_bug"], parsed["seed"]),
            "",
        )
        if excluded_reason:
            parsed["status"] = "infrastructure_failed"
            parsed["exit_reason"] = f"excluded: {excluded_reason}"
            histogram["excluded"] += 1
            # still inserted — visible in the DB for accountability,
            # but won't be aggregated into the leaderboard's
            # succeeded/model_failed cohort.

        model = map_.get(parsed["raw_model"], parsed["raw_model"])
        env_id = _normalize_vr_agent_env_id(parsed["raw_bug"])
        run_id = uuid.uuid4().hex[:16]

        try:
            with transaction(db_path) as con:
                # The vr-agent archive was always run with nudges enabled
                # (per PR #54's docstring: "vr-agent imports = 1"). The
                # archive itself doesn't record that fact per-run, so we
                # hardcode nudges_used=1 here. Without this, rows fall
                # back to the column default 0 and the public snapshot's
                # nudged-view filter excludes the entire Anthropic
                # historical column.
                con.execute(
                    """
                    INSERT INTO runs (
                        run_id, benchmark_id, model, env_id, image_ref,
                        image_digest, task_type, interface, seed, status,
                        capabilities, score,
                        tokens_in, tokens_out, tokens_cache_read,
                        tokens_cache_creation, cost_usd, cost_source,
                        runtime_s, turns_used, exit_reason, run_dir,
                        started_at, finished_at, provenance, llm_route,
                        nudges_used, agent
                    ) VALUES (
                        ?, ?, ?, ?, ?,
                        ?, 'binary_task', 'rl.mcp.v8_task.v1', ?, ?,
                        ?, ?,
                        ?, ?, ?,
                        ?, ?, 'imported',
                        ?, ?, ?, ?,
                        ?, ?, 'imported_from_legacy', 'imported',
                        1, 'vr-agent'
                    )
                    """,
                    (
                        run_id,
                        benchmark_id,
                        model,
                        env_id,
                        parsed["image_ref"],
                        parsed["image_digest"],
                        parsed["seed"],
                        parsed["status"],
                        parsed["capabilities_json"],
                        parsed["score"],
                        parsed["tokens_in"],
                        parsed["tokens_out"],
                        parsed["tokens_cache_read"],
                        parsed["tokens_cache_creation"],
                        parsed["cost_usd"],
                        parsed["runtime_s"],
                        parsed["turns_used"],
                        parsed["exit_reason"],
                        parsed["run_dir"],
                        parsed["started_at"],
                        parsed["finished_at"],
                    ),
                )
            histogram["imported"] += 1
        except sqlite3.IntegrityError:
            histogram["skipped_duplicate"] += 1
            if excluded_reason:
                # we counted it as excluded above; the duplicate-on-rerun
                # path shouldn't double-count it as imported. Decrement
                # excluded count to keep the histogram honest.
                histogram["excluded"] -= 1
    return histogram


def export_native_run(row: dict, target_dir: Path) -> None:
    """Write flat-text artifacts (job.json + score.json + cost.json) for
    one runs-table row at `target_dir`. The reverse of
    `import_native_run_dir` — together they form the D-10 bijection.

    `row` is a dict-shaped runs-table row (works on both raw sqlite3.Row
    via dict() and on the dict produced by import_native_run_dir).
    `target_dir` is created if missing; existing files are overwritten.
    """
    target_dir.mkdir(parents=True, exist_ok=True)

    # job.json — identity + provenance
    job = {
        "run_id": row["run_id"],
        "benchmark_id": row["benchmark_id"],
        "model": row["model"],
        "env_id": row["env_id"],
        "image_ref": row["image_ref"],
        "image_digest": row["image_digest"],
        "image_pulled": False,  # we don't know post-hoc
        "task_type": row["task_type"],
        "interface": row.get("interface"),
        "seed": row["seed"],
        "nudges_used": bool(row.get("nudges_used", 0)),
        "started_at": row["started_at"],
        "provenance": row.get("provenance", "native"),
        "agent": row.get("agent", "qed_swe_bench"),
    }
    if row.get("git_sha"):
        job["git_sha"] = row["git_sha"]
    if row.get("repro_cmd"):
        job["repro_cmd"] = row["repro_cmd"]
    (target_dir / "job.json").write_text(
        json.dumps(job, indent=2) + "\n", encoding="utf-8",
    )

    # config_snapshot.yaml: verbatim YAML round-tripped from the column.
    # D-13 / D-10: the same bytes that were stored on the runs row are
    # written back so import → export → import is byte-identical on the
    # snapshot. NULL column → no file (legacy / partial rows).
    snapshot = row.get("config_snapshot")
    if snapshot:
        (target_dir / "config_snapshot.yaml").write_text(
            snapshot, encoding="utf-8",
        )

    # score.json — outcome (only if we have outcome data)
    if row.get("score") is not None or row.get("status") not in (None, "queued"):
        capabilities = json.loads(row["capabilities"]) if row.get("capabilities") else {}
        score_obj: dict = {
            "capabilities": capabilities,
            "score": row.get("score"),
            "exit_reason": row.get("exit_reason"),
            "status": row.get("status"),
        }
        for k in ("finished_at", "runtime_s", "turns_used",
                  "failure_reason",
                  "weighted_tokens_used", "peak_per_turn_context"):
            if row.get(k) is not None:
                score_obj[k] = row[k]
        (target_dir / "score.json").write_text(
            json.dumps(score_obj, indent=2) + "\n", encoding="utf-8",
        )

    # cost.json — telemetry (only if we have token data)
    if row.get("tokens_in") is not None:
        cost_obj: dict = {
            "model": row["model"],
            "tokens_in": row.get("tokens_in"),
            "tokens_out": row.get("tokens_out"),
            "tokens_cache_read": row.get("tokens_cache_read"),
            "tokens_cache_creation": row.get("tokens_cache_creation"),
            "cost_usd": row.get("cost_usd"),
            "cost_source": row.get("cost_source"),
        }
        # Optional / nullable fields:
        for k in ("llm_route", "api_base"):
            if row.get(k) is not None:
                cost_obj[k] = row[k]
        # NOTE: `or_reconciliation` (added 2026-05) lives only in
        # cost.json today — no corresponding DB column. Export from a
        # row therefore can't restore it; if the DB row was hydrated
        # from a cost.json that had reconciliation data, that data is
        # already preserved on disk. Add a `or_reconciliation TEXT`
        # column via LATE_COLUMNS if/when we want it queryable.
        (target_dir / "cost.json").write_text(
            json.dumps(cost_obj, indent=2) + "\n", encoding="utf-8",
        )


def export_native_runs(
    target_root: Path,
    *,
    benchmark_id: str | None = None,
    only_missing_run_dir: bool = True,
    db_path: Path | None = None,
) -> dict[str, int]:
    """Write run-dir artifacts for DB rows whose disk presence is
    incomplete. By default skips rows that already have a populated
    run-dir on disk (the common case — most rows have their FS
    counterpart from the runner that wrote them).

    Layout: `<target_root>/<benchmark_id>/exported/<run_id>/`. Picked
    `exported` as the host-level segment so re-imports don't conflict
    with original runs (which use `<host>` from `QED_SWE_BENCH_HOST`).

    Use cases:
      - Imported-opus rows whose original eval/ tree was deleted
      - Recovery from a deleted run-dir (DB still has the row)
      - Producing a flat-text artifact set for a benchmark you only
        have in the DB

    Histogram keys: exported, skipped_has_dir, skipped_no_data.
    """
    init_db(db_path)
    histogram: dict[str, int] = {
        "exported": 0,
        "skipped_has_dir": 0,
        "skipped_no_data": 0,
    }
    where = ["1=1"]
    params: list = []
    if benchmark_id is not None:
        where.append("benchmark_id = ?")
        params.append(benchmark_id)
    sql = f"SELECT * FROM runs WHERE {' AND '.join(where)}"
    with transaction(db_path) as con:
        rows = con.execute(sql, tuple(params)).fetchall()

    for sql_row in rows:
        row = dict(sql_row)
        if only_missing_run_dir and row.get("run_dir"):
            existing = Path(row["run_dir"])
            if (existing / "job.json").is_file():
                histogram["skipped_has_dir"] += 1
                continue
        # Need at least the identity fields to export
        if not row.get("run_id") or not row.get("benchmark_id"):
            histogram["skipped_no_data"] += 1
            continue
        target = target_root / row["benchmark_id"] / "exported" / row["run_id"]
        export_native_run(row, target)
        histogram["exported"] += 1
    return histogram
