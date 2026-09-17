"""Budget arithmetic for the weighted-token formula."""

from __future__ import annotations

import pytest

from qed_swe_bench.runner.budget import (
    CACHE_READ_WEIGHT,
    GRADE_NUDGE_INTERVAL,
    OUTPUT_WEIGHT,
    WRAPUP_FRACTION,
    Budget,
)
from qed_swe_bench.runner.llm.base import NormalizedUsage


def test_constants() -> None:
    assert CACHE_READ_WEIGHT == 0.1
    assert OUTPUT_WEIGHT == 5
    assert GRADE_NUDGE_INTERVAL == 50
    assert WRAPUP_FRACTION == 0.75


def test_tick_increments_turn_and_counters() -> None:
    b = Budget()
    b.tick_ai_turn(NormalizedUsage(input_tokens=100, output_tokens=20))
    assert b.turn == 1
    assert b.turns_since_grade == 1
    assert b.total_input_tokens == 100
    assert b.total_output_tokens == 20
    # base_in=100, no cache, output*5 = 100 + 100 = 200
    assert b.tokens_used == 200
    assert b.last_turn_context == 120


def test_tick_weighted_formula_with_caching() -> None:
    """Anthropic-shaped turn: input includes cache_read + cache_creation."""
    b = Budget()
    b.tick_ai_turn(
        NormalizedUsage(
            input_tokens=10_000,
            output_tokens=500,
            cache_read_tokens=8_000,
            cache_creation_tokens=300,
        )
    )
    # base_in = 10000 - 8000 - 300 = 1700
    expected = 1700 + 300 + int(8000 * 0.1) + 5 * 500
    assert b.tokens_used == expected
    assert b.total_cache_read_tokens == 8000
    assert b.total_cache_creation_tokens == 300


def test_tick_accumulates_across_turns() -> None:
    b = Budget()
    for _ in range(3):
        b.tick_ai_turn(NormalizedUsage(input_tokens=10, output_tokens=5))
    assert b.turn == 3
    # per turn: base_in=10 + 5*5 = 35; 3 turns = 105
    assert b.tokens_used == 105
    assert b.turns_since_grade == 3


def test_grade_resets_stuck_counter() -> None:
    b = Budget()
    for _ in range(10):
        b.tick_ai_turn(NormalizedUsage(input_tokens=1, output_tokens=1))
    assert b.turns_since_grade == 10
    b.note_grade_called()
    assert b.turns_since_grade == 0
    # Tick again: counter resumes from 1.
    b.tick_ai_turn(NormalizedUsage(input_tokens=1, output_tokens=1))
    assert b.turns_since_grade == 1


def test_turn_budget_breach() -> None:
    b = Budget(turn_budget=3, token_budget=None, context_budget=None)
    for _ in range(2):
        b.tick_ai_turn(NormalizedUsage(input_tokens=1, output_tokens=1))
    assert not b.exceeded()
    b.tick_ai_turn(NormalizedUsage(input_tokens=1, output_tokens=1))
    assert b.turns_exceeded()
    assert b.exceeded()
    reasons = b.exhaustion_reasons()
    assert any("turn_budget" in r for r in reasons)


def test_token_budget_breach() -> None:
    b = Budget(turn_budget=None, token_budget=10, context_budget=None)
    b.tick_ai_turn(NormalizedUsage(input_tokens=0, output_tokens=10))
    assert b.tokens_exceeded()
    reasons = b.exhaustion_reasons()
    assert any("token_budget" in r for r in reasons)


def test_context_budget_breach() -> None:
    b = Budget(turn_budget=None, token_budget=None, context_budget=50)
    b.tick_ai_turn(NormalizedUsage(input_tokens=40, output_tokens=20))  # ctx=60
    assert b.context_exceeded()
    reasons = b.exhaustion_reasons()
    assert any("context_size" in r for r in reasons)


def test_zero_budget_is_unbounded() -> None:
    b = Budget(turn_budget=0, token_budget=0, context_budget=0)
    for _ in range(100):
        b.tick_ai_turn(NormalizedUsage(input_tokens=1000, output_tokens=1000))
    # No limits → never exceeded
    assert not b.exceeded()


def test_token_budget_none_disables_enforcement_but_still_tracks() -> None:
    """Turn-as-effort: token_budget=None means no per-episode limit, but
    `tokens_used` keeps incrementing so score.json still reports it
    (`weighted_tokens_used` diagnostic)."""
    b = Budget(turn_budget=300, token_budget=None, context_budget=None)
    for _ in range(50):
        b.tick_ai_turn(NormalizedUsage(input_tokens=10_000, output_tokens=2_000))
    assert not b.tokens_exceeded()
    assert not b.exceeded()
    # Per turn: base_in=10000 + 5*2000 = 20000; 50 turns = 1_000_000.
    assert b.tokens_used == 1_000_000


def test_context_budget_none_disables_enforcement_but_tracks_peak() -> None:
    """Turn-as-effort: context_budget=None means no per-turn limit, but
    `peak_per_turn_context` is still updated so score.json can report
    what the peak would have hit."""
    b = Budget(turn_budget=300, token_budget=None, context_budget=None)
    b.tick_ai_turn(NormalizedUsage(input_tokens=40_000, output_tokens=10_000))
    b.tick_ai_turn(NormalizedUsage(input_tokens=200_000, output_tokens=5_000))  # peak
    b.tick_ai_turn(NormalizedUsage(input_tokens=30_000, output_tokens=1_000))
    assert not b.context_exceeded()
    assert b.peak_per_turn_context == 205_000


def test_peak_per_turn_context_tracked_when_enforcement_active() -> None:
    """Even when context_budget enforces a cap, peak_per_turn_context
    still tracks the max — they're independent (one is a circuit breaker,
    the other is a diagnostic)."""
    b = Budget(turn_budget=300, token_budget=None, context_budget=180_000)
    b.tick_ai_turn(NormalizedUsage(input_tokens=50_000, output_tokens=1_000))
    assert b.peak_per_turn_context == 51_000
    b.tick_ai_turn(NormalizedUsage(input_tokens=120_000, output_tokens=2_000))
    assert b.peak_per_turn_context == 122_000


def test_peak_per_turn_context_starts_at_zero_and_only_grows() -> None:
    """A smaller turn after a larger one must NOT shrink the peak."""
    b = Budget()
    assert b.peak_per_turn_context == 0
    b.tick_ai_turn(NormalizedUsage(input_tokens=100_000, output_tokens=10_000))
    assert b.peak_per_turn_context == 110_000
    b.tick_ai_turn(NormalizedUsage(input_tokens=1_000, output_tokens=100))
    assert b.peak_per_turn_context == 110_000  # unchanged


def test_grade_nudge_at_50_turns() -> None:
    b = Budget(turn_budget=300)
    for _ in range(GRADE_NUDGE_INTERVAL - 1):
        b.tick_ai_turn(NormalizedUsage(input_tokens=1, output_tokens=1))
    assert not b.should_nudge_grade()
    b.tick_ai_turn(NormalizedUsage(input_tokens=1, output_tokens=1))
    assert b.should_nudge_grade()
    # Reset on grade()
    b.note_grade_called()
    assert not b.should_nudge_grade()


def test_wrapup_nudge_at_75_pct() -> None:
    b = Budget(turn_budget=100)
    for _ in range(74):
        b.tick_ai_turn(NormalizedUsage(input_tokens=1, output_tokens=1))
    assert not b.should_nudge_wrapup()
    b.tick_ai_turn(NormalizedUsage(input_tokens=1, output_tokens=1))  # turn=75
    assert b.should_nudge_wrapup()
    # One-shot — after marking sent, no more nudges.
    b.mark_wrapup_sent()
    assert not b.should_nudge_wrapup()


def test_wrapup_disabled_without_turn_budget() -> None:
    b = Budget(turn_budget=None)
    for _ in range(1000):
        b.tick_ai_turn(NormalizedUsage(input_tokens=1, output_tokens=1))
    assert not b.should_nudge_wrapup()


@pytest.mark.parametrize(
    "in_t,out_t,cr,cc,expected",
    [
        (0, 100, 0, 0, 500),
        (0, 100, 0, 50, 50 + 500),
        (0, 0, 1000, 0, 100),
        (5000, 200, 4000, 100, 900 + 100 + 400 + 1000),
        # No caching (e.g. non-Anthropic): all input counts as base_in.
        (50_000, 1000, 0, 0, 50_000 + 5000),
        # base_in clamps to 0 when cr+cc > in_t.
        (1000, 100, 800, 500, 0 + 500 + 80 + 500),
    ],
)
def test_tokens_used_formula(in_t, out_t, cr, cc, expected) -> None:
    b = Budget()
    b.tick_ai_turn(
        NormalizedUsage(
            input_tokens=in_t,
            output_tokens=out_t,
            cache_read_tokens=cr,
            cache_creation_tokens=cc,
        )
    )
    assert b.tokens_used == expected
