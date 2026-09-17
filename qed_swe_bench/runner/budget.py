"""Per-episode budget tracker.

`tokens_used` is a weighted cost-equivalent in input-token units. Used as
a per-episode circuit-breaker without per-model pricing lookups:

    base_input = max(0, input - cache_read - cache_creation)
    tokens_used += base_input
                 + cache_creation
                 + int(cache_read * CACHE_READ_WEIGHT)
                 + OUTPUT_WEIGHT * output

OUTPUT_WEIGHT = 5 matches Anthropic's published output:input ratio
($5/$25 Opus, $3/$15 Sonnet, $1/$5 Haiku); for OpenAI and Gemini it sits
in the 3-8× range, so the bound is exact for Anthropic and approximate
elsewhere.

Three independent limits can stop the episode. Per the turn-as-effort
methodology (decisions.md), `turn_budget` is the only fairness anchor for
the v8 matrix; `token_budget` and `context_budget` are *optional with
always-report* — pass None to disable enforcement and the runner still
tracks `weighted_tokens_used` and `peak_per_turn_context` for diagnostics.

  - turn_budget    max number of AI turns
  - token_budget   weighted cumulative tokens (above formula); None = unbounded
  - context_budget max input+output of a single turn; None = unbounded
"""

from __future__ import annotations

from dataclasses import dataclass

from qed_swe_bench.runner.llm.base import NormalizedUsage

CACHE_READ_WEIGHT = 0.1
OUTPUT_WEIGHT = 5

# Default nudge cadence — 50 turns without grade triggers a stuck nudge.
GRADE_NUDGE_INTERVAL = 50

# When `turn` >= turn_budget * WRAPUP_FRACTION, send a wrapup nudge once.
WRAPUP_FRACTION = 0.75


@dataclass
class Budget:
    """Mutable counters + immutable limits for one episode.

    Limits of 0 / None are treated as "unbounded" for that axis. The
    diagnostic counters (`tokens_used`, `peak_per_turn_context`) are
    tracked regardless of whether the corresponding budget is active —
    callers always read them off `Budget` to write into `score.json`.
    """

    turn_budget: int | None = 300
    token_budget: int | None = 2_500_000
    context_budget: int | None = 180_000

    # Counters
    turn: int = 0                  # AI turn count (incremented on each AI message)
    tokens_used: int = 0           # weighted cumulative
    last_turn_context: int = 0     # input + output of the most recent AI turn
    peak_per_turn_context: int = 0  # max input+output across all AI turns
    turns_since_grade: int = 0     # incremented per AI message, reset on grade()
    sent_wrapup: bool = False      # one-shot flag for the wrapup nudge

    # Per-axis cumulative token counts (cost.py reads these).
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_cache_creation_tokens: int = 0

    def tick_ai_turn(self, usage: NormalizedUsage) -> None:
        """Account for one AI turn's token usage. Call ONCE per AI message.

        Increments `turn`, `turns_since_grade`, and the cumulative counters.
        Updates `last_turn_context` and `peak_per_turn_context` for the
        per-turn check / always-on diagnostic.
        """
        self.turn += 1
        self.turns_since_grade += 1

        in_t = usage.input_tokens
        out_t = usage.output_tokens
        cr = usage.cache_read_tokens
        cc = usage.cache_creation_tokens

        self.total_input_tokens += in_t
        self.total_output_tokens += out_t
        self.total_cache_read_tokens += cr
        self.total_cache_creation_tokens += cc

        base_in = max(0, in_t - cr - cc)
        self.tokens_used += (
            base_in + cc + int(cr * CACHE_READ_WEIGHT) + OUTPUT_WEIGHT * out_t
        )
        self.last_turn_context = in_t + out_t
        if self.last_turn_context > self.peak_per_turn_context:
            self.peak_per_turn_context = self.last_turn_context

    def note_grade_called(self) -> None:
        """Reset the stuck-nudge counter."""
        self.turns_since_grade = 0

    # ---------------- limit checks ----------------

    def context_exceeded(self) -> bool:
        return bool(self.context_budget) and self.last_turn_context >= self.context_budget

    def tokens_exceeded(self) -> bool:
        return bool(self.token_budget) and self.tokens_used >= self.token_budget

    def turns_exceeded(self) -> bool:
        return bool(self.turn_budget) and self.turn >= self.turn_budget

    def exceeded(self) -> bool:
        return self.context_exceeded() or self.tokens_exceeded() or self.turns_exceeded()

    def exhaustion_reasons(self) -> list[str]:
        """Human-readable list of which limits tripped (for `exit_reason`)."""
        reasons: list[str] = []
        if self.context_exceeded():
            reasons.append(
                f"context_size ({self.last_turn_context} >= {self.context_budget})"
            )
        if self.tokens_exceeded():
            reasons.append(f"token_budget ({self.tokens_used} >= {self.token_budget})")
        if self.turns_exceeded():
            reasons.append(f"turn_budget ({self.turn} >= {self.turn_budget})")
        return reasons

    # ---------------- nudge policy ----------------

    def should_nudge_grade(self) -> bool:
        """Stuck nudge: too many AI turns since last grade()."""
        return self.turns_since_grade >= GRADE_NUDGE_INTERVAL

    def should_nudge_wrapup(self) -> bool:
        """Wrapup nudge: at >= 75% of turn budget, once. Latches via sent_wrapup."""
        if self.sent_wrapup or not self.turn_budget:
            return False
        return self.turn >= int(self.turn_budget * WRAPUP_FRACTION)

    def mark_wrapup_sent(self) -> None:
        self.sent_wrapup = True
