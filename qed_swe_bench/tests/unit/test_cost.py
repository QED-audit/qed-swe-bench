"""Pricing-table lookup + cost computation."""

from __future__ import annotations

import pytest

from qed_swe_bench.runner.cost import (
    PRICING_TABLE,
    ModelPricing,
    compute_cost,
    compute_total_cost,
    lookup_pricing,
)
from qed_swe_bench.runner.llm.base import NormalizedUsage


def test_lookup_pricing_exact_match() -> None:
    p = lookup_pricing("claude-haiku-4-5")
    assert isinstance(p, ModelPricing)
    assert p.input == 1.0


def test_lookup_pricing_strips_provider_prefix() -> None:
    assert lookup_pricing("anthropic/claude-haiku-4-5") == PRICING_TABLE["claude-haiku-4-5"]
    assert lookup_pricing("openai/gpt-5-mini") == PRICING_TABLE["gpt-5-mini"]
    assert lookup_pricing("gemini/gemini-2.5-flash") == PRICING_TABLE["gemini-2.5-flash"]


def test_lookup_pricing_strips_openrouter_double_prefix() -> None:
    assert lookup_pricing("openrouter/anthropic/claude-haiku-4-5") == PRICING_TABLE[
        "claude-haiku-4-5"
    ]


def test_lookup_pricing_unknown_returns_none() -> None:
    # Genuinely unregistered ids stay None.
    assert lookup_pricing("openai/llama-3.3-70b") is None
    assert lookup_pricing("totally-fake-model") is None


def test_compute_cost_anthropic_haiku_no_cache() -> None:
    """Pure base-input + output, no caching."""
    usage = NormalizedUsage(input_tokens=1_000_000, output_tokens=100_000)
    cost, source = compute_cost("anthropic/claude-haiku-4-5", usage)
    # 1M base @ $1 + 0.1M output @ $5 = $1 + $0.50 = $1.50
    assert cost == pytest.approx(1.5, rel=1e-6)
    assert source == "pricing_table"


def test_compute_cost_anthropic_with_cache() -> None:
    """The three input categories (input_tokens / cache_read / cache_creation)
    are DISJOINT — verified against live Anthropic API 2026-05-13 where a
    cached follow-up returned input_tokens=13 (just the user msg) with
    cache_read=3603 (the cached prefix). Sum them at their respective rates;
    no subtraction. LiteLLM-populated rows match the same convention because
    `litellm_client.py` subtracts cache_read from prompt_tokens at populate
    time."""
    usage = NormalizedUsage(
        input_tokens=30_000,              # base (uncached new) input only
        output_tokens=10_000,
        cache_read_tokens=150_000,        # cached prefix served
        cache_creation_tokens=20_000,     # prefix that wrote to cache
    )
    # cost = 30k/1M * 1 + 10k/1M * 5 + 150k/1M * 0.10 + 20k/1M * 1.25
    #      = 0.03 + 0.05 + 0.015 + 0.025 = 0.12
    cost, source = compute_cost("anthropic/claude-haiku-4-5", usage)
    assert cost == pytest.approx(0.12, rel=1e-4)
    assert source == "pricing_table"


def test_compute_cost_unknown_model_returns_none() -> None:
    usage = NormalizedUsage(input_tokens=10_000, output_tokens=1_000)
    cost, source = compute_cost("openai/llama-3.3-70b", usage)
    assert cost is None
    assert source == "unknown"


def test_compute_cost_zero_usage() -> None:
    usage = NormalizedUsage()
    cost, source = compute_cost("anthropic/claude-haiku-4-5", usage)
    assert cost == 0.0
    assert source == "pricing_table"


def test_compute_cost_disjoint_input_and_cache() -> None:
    """The disjoint-convention invariant: cache_read > input_tokens is
    expected for high-cache-hit cells (the live Anthropic probe showed
    input_tokens=13, cache_read=3603). Each category gets billed
    independently — no clamping or subtraction in compute_cost."""
    usage = NormalizedUsage(
        input_tokens=100,         # base only — small, fresh new tokens
        output_tokens=10,
        cache_read_tokens=200,    # cached prefix served
    )
    # cost = 100/1M*1 + 10/1M*5 + 200/1M*0.1
    #      = 0.0001 + 0.00005 + 0.00002 = 0.00017
    cost, source = compute_cost("anthropic/claude-haiku-4-5", usage)
    assert cost == pytest.approx(0.00017, rel=1e-4)
    assert source == "pricing_table"


# ---------- long-context tier bucketing ----------


def test_for_input_size_no_long_tier_returns_self() -> None:
    """Entries without long_context_threshold return themselves unchanged."""
    p = PRICING_TABLE["claude-haiku-4-5"]
    assert p.for_input_size(0) is p
    assert p.for_input_size(10_000_000) is p


def test_gpt55_short_context_below_threshold() -> None:
    """gpt-5.5 with input <= 272K uses short-context rates: $5 / $0.50 / $30."""
    p = lookup_pricing("openai/gpt-5.5", input_tokens=100_000)
    assert p is not None
    assert p.input == 5.0
    assert p.cache_read == 0.50
    assert p.output == 30.0


def test_gpt55_long_context_above_threshold() -> None:
    """gpt-5.5 with input > 272K uses long-context rates: $10 / $1.00 / $45."""
    p = lookup_pricing("openai/gpt-5.5", input_tokens=300_000)
    assert p is not None
    assert p.input == 10.0
    assert p.cache_read == 1.0
    assert p.output == 45.0


def test_gpt55_at_threshold_is_short() -> None:
    """Inclusive boundary: input == 272K stays in short tier per the
    OpenAI pricing page ('<272K context length' for short). Anything
    strictly above flips to long."""
    p_short = lookup_pricing("openai/gpt-5.5", input_tokens=272_000)
    p_long = lookup_pricing("openai/gpt-5.5", input_tokens=272_001)
    assert p_short is not None and p_long is not None
    assert p_short.input == 5.0
    assert p_long.input == 10.0


def test_compute_cost_gpt55_long_context() -> None:
    """End-to-end: a 300K-input call bills at the long-context tier."""
    usage = NormalizedUsage(input_tokens=300_000, output_tokens=10_000)
    cost, source = compute_cost("openai/gpt-5.5", usage)
    # long tier: 300k/1M * 10 + 10k/1M * 45 = 3.0 + 0.45 = 3.45
    assert cost == pytest.approx(3.45, rel=1e-6)
    assert source == "pricing_table"


def test_gemini_31_pro_short_context() -> None:
    """gemini-3.1-pro-preview short tier: $2 / $0.20 / $12."""
    p = lookup_pricing("gemini/gemini-3.1-pro-preview", input_tokens=100_000)
    assert p is not None
    assert p.input == 2.0
    assert p.cache_read == 0.20
    assert p.output == 12.0


def test_gemini_31_pro_long_context() -> None:
    """gemini-3.1-pro-preview long tier (input > 200K): $4 / $0.40 / $18."""
    p = lookup_pricing("gemini/gemini-3.1-pro-preview", input_tokens=250_000)
    assert p is not None
    assert p.input == 4.0
    assert p.cache_read == 0.40
    assert p.output == 18.0


def test_gemini_31_pro_at_threshold_is_short() -> None:
    """Gemini's docs: '<= 200K' is short; '> 200K' is long. 200K stays short."""
    p_short = lookup_pricing("gemini/gemini-3.1-pro-preview", input_tokens=200_000)
    p_long = lookup_pricing("gemini/gemini-3.1-pro-preview", input_tokens=200_001)
    assert p_short is not None and p_long is not None
    assert p_short.input == 2.0
    assert p_long.input == 4.0


def test_gpt54_long_context() -> None:
    """gpt-5.4 long tier (input > 272K): $5 / $0.50 / $22.50."""
    p = lookup_pricing("openai/gpt-5.4", input_tokens=400_000)
    assert p is not None
    assert p.input == 5.0
    assert p.cache_read == 0.50
    assert p.output == 22.50


def test_gpt54_mini_no_long_tier() -> None:
    """gpt-5.4-mini is flat-priced — no long-context tier."""
    p_small = lookup_pricing("openai/gpt-5.4-mini", input_tokens=10_000)
    p_huge = lookup_pricing("openai/gpt-5.4-mini", input_tokens=10_000_000)
    assert p_small is not None and p_huge is not None
    assert p_small.input == p_huge.input == 0.75
    assert p_small.output == p_huge.output == 4.50


def test_lookup_pricing_default_input_size_is_short_tier() -> None:
    """Callers that don't pass input_tokens get short tier (sane for tests/
    inspection that don't care about bucketing)."""
    p = lookup_pricing("openai/gpt-5.5")  # defaults to input_tokens=0
    assert p is not None
    assert p.input == 5.0  # short rate


# ---------- compute_total_cost: per-call tier bucketing ----------


def test_compute_total_cost_mixed_short_and_long_calls_gpt55() -> None:
    """The whole point of compute_total_cost: an episode with some calls
    below 272K and some above gets priced correctly per-call. Passing
    the aggregate to compute_cost would incorrectly price every call at
    the long-context tier."""
    calls = [
        NormalizedUsage(input_tokens=100_000, output_tokens=1_000),  # short
        NormalizedUsage(input_tokens=200_000, output_tokens=2_000),  # short
        NormalizedUsage(input_tokens=400_000, output_tokens=5_000),  # long
    ]
    cost, source = compute_total_cost("openai/gpt-5.5", calls)
    # Short tier ($5/$30): 100k*5 + 1k*30 = 0.5 + 0.03 = 0.53
    #                      200k*5 + 2k*30 = 1.0 + 0.06 = 1.06
    # Long tier ($10/$45): 400k*10 + 5k*45 = 4.0 + 0.225 = 4.225
    # Total: 0.53 + 1.06 + 4.225 = 5.815
    assert cost == pytest.approx(5.815, rel=1e-4)
    assert source == "pricing_table"


def test_compute_total_cost_vs_aggregate_overprices_long_for_gpt55() -> None:
    """Confirms the bug-shape: aggregating then single-call costs trips
    long-context tier for every call. compute_total_cost avoids this."""
    calls = [NormalizedUsage(input_tokens=100_000, output_tokens=1_000)] * 3
    aggregate = NormalizedUsage(input_tokens=300_000, output_tokens=3_000)
    correct_cost, _ = compute_total_cost("openai/gpt-5.5", calls)
    buggy_cost, _ = compute_cost("openai/gpt-5.5", aggregate)
    # Correct: 3 × (100k*5 + 1k*30) / 1M = 3 × 0.53 = 1.59
    # Buggy:   300k > 272K so long-tier: 300k*10 + 3k*45 = 3.135
    assert correct_cost == pytest.approx(1.59, rel=1e-4)
    assert buggy_cost == pytest.approx(3.135, rel=1e-4)
    assert correct_cost < buggy_cost  # aggregate-then-price overcharges


def test_compute_total_cost_empty_list() -> None:
    cost, source = compute_total_cost("openai/gpt-5.5", [])
    assert cost == 0.0
    assert source == "pricing_table"


def test_compute_total_cost_unknown_model() -> None:
    calls = [NormalizedUsage(input_tokens=10_000, output_tokens=1_000)]
    cost, source = compute_total_cost("totally-fake-model", calls)
    assert cost is None
    assert source == "unknown"
