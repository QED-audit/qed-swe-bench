"""Running-cost tracker with an optional global cap.

Each (model, env, seed) tuple reports its `cost_usd` after the episode
ends. When a `cost_cap_usd` is configured on the BenchmarkConfig, the
orchestrator checks the cap before scheduling each tuple — once the
running total crosses the cap, remaining tuples short-circuit with
status `infra_failed` (failure_reason="cost_cap_exceeded").

Concurrent-safe via `asyncio.Lock` because tuples run under
`asyncio.gather` with `Semaphore(max_parallel)` parallelism.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class SpendTracker:
    """Tracks running USD spend across an in-progress benchmark sweep.

    Construct once per `run_benchmark` invocation; share the instance
    across all bound tuple coroutines. `cap_usd=None` disables the
    cap (the tracker still records the running total for logging).
    """

    cap_usd: float | None = None
    _total: float = 0.0

    def __post_init__(self) -> None:
        # asyncio.Lock() construction is loop-free on Python ≥3.10
        # (the repo's floor; pyproject targets modern interpreters).
        self._lock = asyncio.Lock()

    async def add(self, cost_usd: float | None) -> None:
        """Add a tuple's spend to the running total. Tolerates None."""
        if not cost_usd:
            return
        async with self._lock:
            self._total += float(cost_usd)
            log.debug("spend: +$%.4f → $%.2f total", cost_usd, self._total)

    async def total(self) -> float:
        async with self._lock:
            return self._total

    async def cap_exceeded(self) -> bool:
        """True iff a cap is set AND the running total is >= the cap.

        Checked before scheduling each new tuple. Once True, all later
        tuples short-circuit; they don't actually run, just record an
        `infra_failed` row so `--retry-failed` can revisit them later.
        """
        if self.cap_usd is None:
            return False
        async with self._lock:
            return self._total >= self.cap_usd
