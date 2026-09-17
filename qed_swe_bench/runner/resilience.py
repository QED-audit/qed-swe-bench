"""Orchestrator resilience helpers: stale-queued recovery + retry-failed.

Two related concerns, both about making `--resume` actually do the right
thing across orchestrator restarts and partial failures:

- **Stale-queued recovery.** A row enters `runs` with `status='queued'`
  the moment `_insert_queued` succeeds. If the orchestrator dies hard
  (kill -9, OOM, container daemon crash) before the matching
  `_update_finished`, the row is stuck `queued` forever. The
  `UNIQUE(benchmark_id, model, env_id, seed)` constraint then blocks
  any retry of that tuple. `mark_stale_queued_as_failed` runs at the
  start of `run_benchmark` and converts orphaned `queued` rows to
  `infra_failed` so the next concern can pick them up.

  Heartbeat-based liveness (added 2026-05): each running orchestrator
  periodically calls `heartbeat_run_ids` to refresh `last_heartbeat`
  on its own queued rows. The sweep treats a row as stale only when
  its heartbeat is missing OR older than `STALE_HEARTBEAT_AFTER`.
  This replaces the prior pure-`started_at` scheme that wrongly
  reaped sibling-process live rows when long V8 episodes ran past
  the 30-minute cutoff.

- **Retry-failed.** `--resume` skips any row already in the DB,
  regardless of status — that's correct for `succeeded` runs (don't
  re-pay) but blocks retry of `infra_failed` / `model_failed` rows
  even when the cause was transient. `delete_failed_rows` removes
  failed rows for one benchmark_id so the next sweep re-inserts and
  retries them. Run-dir artifacts on disk are left in place (cheap;
  the new run gets a new run_id and its own dir).

All helpers are pure SQL with explicit scoping. No side effects
beyond the database.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta

from qed_swe_bench.db.schema import transaction

log = logging.getLogger(__name__)


def _env_minutes(name: str, default_min: int) -> timedelta:
    """Parse a positive-int minutes env override; fall back to default."""
    raw = os.environ.get(name)
    if raw is None:
        return timedelta(minutes=default_min)
    try:
        v = int(raw)
        if v <= 0:
            raise ValueError
        return timedelta(minutes=v)
    except ValueError:
        log.warning(
            "%s=%r is not a positive int; using default %d min", name, raw, default_min,
        )
        return timedelta(minutes=default_min)


def _env_seconds(name: str, default_s: int) -> timedelta:
    """Parse a positive-int seconds env override; fall back to default."""
    raw = os.environ.get(name)
    if raw is None:
        return timedelta(seconds=default_s)
    try:
        v = int(raw)
        if v <= 0:
            raise ValueError
        return timedelta(seconds=v)
    except ValueError:
        log.warning(
            "%s=%r is not a positive int; using default %ds", name, raw, default_s,
        )
        return timedelta(seconds=default_s)


def _stale_queued_after() -> timedelta:
    """Legacy cutoff for queued rows with NO heartbeat at all.

    Covers two cases: (1) rows inserted before the heartbeat code
    landed, and (2) rows from a process that died between insert and
    first heartbeat. Override with QED_SWE_BENCH_STALE_QUEUED_MIN.
    """
    return _env_minutes("QED_SWE_BENCH_STALE_QUEUED_MIN", 30)


def _stale_heartbeat_after() -> timedelta:
    """Cutoff for rows with a heartbeat: if last heartbeat is older
    than this, the owner is dead.

    Should be > heartbeat_interval × ~3 so a brief stall doesn't reap
    a live owner. Default 5 min; override with
    QED_SWE_BENCH_STALE_HEARTBEAT_MIN.
    """
    return _env_minutes("QED_SWE_BENCH_STALE_HEARTBEAT_MIN", 5)


def heartbeat_interval() -> timedelta:
    """How often a live orchestrator should refresh its rows'
    `last_heartbeat`. Default 60s; override with
    QED_SWE_BENCH_HEARTBEAT_INTERVAL_S.
    """
    return _env_seconds("QED_SWE_BENCH_HEARTBEAT_INTERVAL_S", 60)


# Public alias kept for backward compat with tests + RUNBOOK references.
STALE_QUEUED_AFTER = timedelta(minutes=30)


def heartbeat_run_ids(
    run_ids: Iterable[str], now: datetime | None = None,
) -> int:
    """Refresh `last_heartbeat` for the given run_ids. Returns rows updated.

    Touches rows in `queued` OR `running` — both are in-flight states
    the orchestrator owns. Once update_finished has flipped status to
    a terminal value the row is no longer the orchestrator's concern.
    Skipping the WHERE clause filter would cause a brief race where a
    finished row's last_heartbeat keeps getting bumped.

    Empty `run_ids` is a no-op (returns 0).
    """
    ids = list(run_ids)
    if not ids:
        return 0
    ts = (now or datetime.now(UTC)).isoformat()
    placeholders = ",".join("?" for _ in ids)
    with transaction() as con:
        cur = con.execute(
            f"UPDATE runs SET last_heartbeat = ? "
            f"WHERE run_id IN ({placeholders}) "
            f"AND status IN ('queued', 'running')",
            (ts, *ids),
        )
        return cur.rowcount


def mark_stale_queued_as_failed(now: datetime | None = None) -> int:
    """Convert orphaned `queued`/`running` rows to `infra_failed`. Returns count.

    Called once at the start of every `run_benchmark` so a previous
    orchestrator's crash doesn't permanently jam the (benchmark_id,
    model, env_id, seed) UNIQUE slot.

    Two reaping conditions (a row matching either is reaped):
      1. Has heartbeat AND heartbeat is older than STALE_HEARTBEAT_AFTER.
         (Owner went silent → process is dead → reap.) Applies to both
         `queued` (waiting for a slot) and `running` (mid-episode) rows
         — once we issued the row a heartbeat, staleness is staleness.
      2. Has NO heartbeat AND status='queued' AND started_at is older
         than STALE_QUEUED_AFTER. (Legacy rows from before heartbeat
         landed; or rows whose orchestrator died before first
         heartbeat. Generous cutoff so a sibling process's brand-new
         pre-heartbeat insert isn't reaped. Doesn't apply to
         `running` since `mark_running` always seeds last_heartbeat.)

    `now` is injectable for testing; defaults to UTC wallclock.
    """
    now = now or datetime.now(UTC)
    cutoff_heartbeat = (now - _stale_heartbeat_after()).isoformat()
    cutoff_started = (now - _stale_queued_after()).isoformat()
    with transaction() as con:
        cur = con.execute(
            """
            UPDATE runs
               SET status='infra_failed',
                   failure_reason=COALESCE(failure_reason,
                                            'stale_queued_recovery'),
                   finished_at=?
             WHERE status IN ('queued', 'running')
               AND finished_at IS NULL
               AND (
                 -- Has a heartbeat that's gone stale (queued or running)
                 (last_heartbeat IS NOT NULL AND last_heartbeat < ?)
                 -- Or queued + no heartbeat AND started_at past legacy cutoff
                 OR (status='queued'
                     AND last_heartbeat IS NULL
                     AND started_at < ?)
               )
            """,
            (now.isoformat(), cutoff_heartbeat, cutoff_started),
        )
        n = cur.rowcount
    if n:
        log.warning(
            "stale-queued recovery: converted %d row(s) to infra_failed "
            "(heartbeat-cutoff=%s started_at-cutoff=%s)",
            n, cutoff_heartbeat, cutoff_started,
        )
    return n


def delete_failed_rows(benchmark_id: str) -> int:
    """Delete `infra_failed` / `model_failed` rows for one benchmark.

    Triggered by `qed_swe_bench benchmark --retry-failed`. The next
    sweep's `_insert_queued` will create fresh queued rows in their
    place (UNIQUE no longer blocks). On-disk run-dir artifacts from
    the failed runs are NOT cleaned up — they're cheap and the new
    run_id gets a fresh directory anyway.

    Returns the number of rows deleted.
    """
    with transaction() as con:
        cur = con.execute(
            """
            DELETE FROM runs
             WHERE benchmark_id = ?
               AND status IN ('infra_failed', 'model_failed')
            """,
            (benchmark_id,),
        )
        n = cur.rowcount
    log.info("retry-failed: deleted %d failed row(s) for benchmark_id=%s", n, benchmark_id)
    return n


class HeartbeatTracker:
    """Async background-task that periodically refreshes `last_heartbeat`
    on this orchestrator's own queued rows.

    Lifecycle:
      tracker = HeartbeatTracker()
      tracker.start()                # spawns the background loop
      await tracker.add(run_id)      # claim a row; bump immediately
      ... run the episode ...
      await tracker.remove(run_id)   # release after update_finished
      await tracker.stop()           # cancel the background loop

    Adding a run_id triggers an immediate heartbeat write so the row
    has a non-NULL last_heartbeat the moment it's tracked, before the
    next periodic tick. This narrows the window where a sibling
    process's startup sweep would see "no heartbeat + new started_at"
    and rely on the legacy cutoff.

    Heartbeat writes are off-thread (`asyncio.to_thread`) since the
    underlying DB driver is sync.
    """

    def __init__(self, interval: timedelta | None = None) -> None:
        self._interval = interval or heartbeat_interval()
        self._lock = asyncio.Lock()
        self._run_ids: set[str] = set()
        self._task: asyncio.Task[None] | None = None

    async def add(self, run_id: str) -> None:
        async with self._lock:
            self._run_ids.add(run_id)
        # Immediate single-row heartbeat so the row leaves NULL state
        # before the next periodic tick. Off-thread since DB is sync.
        await asyncio.to_thread(heartbeat_run_ids, [run_id])

    async def remove(self, run_id: str) -> None:
        async with self._lock:
            self._run_ids.discard(run_id)

    async def _loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._interval.total_seconds())
                async with self._lock:
                    ids = list(self._run_ids)
                if ids:
                    try:
                        await asyncio.to_thread(heartbeat_run_ids, ids)
                    except Exception:  # noqa: BLE001
                        # Heartbeat failure shouldn't kill the orchestrator;
                        # the sweep cutoff will eventually reap the row if
                        # this becomes persistent.
                        log.exception("heartbeat write failed")
        except asyncio.CancelledError:
            return

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
