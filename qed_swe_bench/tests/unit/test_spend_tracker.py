"""Unit tests for runner/spend_tracker.py."""

from __future__ import annotations

import asyncio

import pytest

from qed_swe_bench.runner.spend_tracker import SpendTracker


@pytest.mark.asyncio
async def test_add_accumulates_cost() -> None:
    t = SpendTracker()
    await t.add(0.05)
    await t.add(0.10)
    assert await t.total() == pytest.approx(0.15)


@pytest.mark.asyncio
async def test_add_tolerates_none_and_zero() -> None:
    t = SpendTracker()
    await t.add(None)
    await t.add(0)
    await t.add(0.5)
    assert await t.total() == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_no_cap_means_never_exceeded() -> None:
    t = SpendTracker(cap_usd=None)
    for _ in range(100):
        await t.add(1_000_000)
    assert await t.cap_exceeded() is False


@pytest.mark.asyncio
async def test_cap_triggers_at_threshold() -> None:
    t = SpendTracker(cap_usd=1.00)
    await t.add(0.50)
    assert await t.cap_exceeded() is False
    await t.add(0.50)
    # Equal to cap should trigger — the contract is "stop spending more".
    assert await t.cap_exceeded() is True


@pytest.mark.asyncio
async def test_concurrent_adds_are_consistent() -> None:
    """Many concurrent adds must produce the right total — no race."""
    t = SpendTracker()
    n_tasks = 200
    await asyncio.gather(*(t.add(0.01) for _ in range(n_tasks)))
    assert await t.total() == pytest.approx(n_tasks * 0.01)
