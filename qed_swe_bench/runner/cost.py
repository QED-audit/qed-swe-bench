"""Cost capture per episode.

Strategy:
  1. We always *capture* the four token counts (input, output, cache_read,
     cache_creation) into runs.tokens_*. Those are durable.
  2. Cost is a derived view computed at episode end from a local pricing table
     (PRICING_TABLE below). When a model isn't in the table — most likely an
     OSS model behind a gateway — we still write the row but flag
     cost_source='estimated' (best-effort using the cheapest known tier as a
     placeholder) or, if nothing applies at all, leave cost_usd as None and
     cost_source='unknown'.

  3. For Anthropic we always have the cache-token split from the native SDK
     response. For LiteLLM-backed providers, cache_read may come from
     prompt_tokens_details.cached_tokens; cache_creation is rare outside
     Anthropic.

Pricing values are in USD per million tokens, picked to match bench-v8's
compute-token-costs.py reference for Anthropic Opus 4.6/4.7 and
otherwise updated to public Apr-2026 list prices. They are deliberately
versioned in code (not pulled live) so historical runs can be re-priced
deterministically by changing the table and re-running aggregate.

Some entries (currently OpenAI gpt-5.5 and Gemini 3.1 Pro Preview) have an
optional long-context tier — requests where input_tokens exceeds the
entry's ``long_context_threshold`` are priced at higher per-token rates.
``compute_cost`` passes ``usage.input_tokens`` to ``lookup_pricing``,
which calls ``ModelPricing.for_input_size`` to pick the right tier.

Entries not verified against a live published rate are AI-generated
placeholders. The Anthropic / gpt-5.5 / gemini-3.1-pro blocks were
verified against the provider's pricing page on 2026-05-11. Other rows
may be stale — verify before using ``cost_usd`` as ground truth at scale.
"""

from __future__ import annotations

from dataclasses import dataclass

from qed_swe_bench.runner.llm.base import NormalizedUsage


@dataclass(frozen=True)
class ModelPricing:
    """USD per 1,000,000 tokens for each token class.

    Some providers (OpenAI gpt-5.5, Gemini 3.x Pro) bill at a higher rate
    once input crosses a per-model threshold. Set the optional
    ``long_context_*`` fields to enable that bucketing; ``for_input_size``
    returns the effective pricing for a given input size.
    """

    input: float
    output: float
    cache_read: float = 0.0
    # 5-minute TTL cache write rate. Anthropic also has a 1h rate; we don't
    # distinguish today because the response doesn't tell us which TTL the
    # creation block used.
    cache_creation: float = 0.0
    # Optional long-context tier: when input_tokens > long_context_threshold,
    # the long_context_* overrides replace the short-context rates. Fields
    # left None fall back to their short-context value.
    long_context_threshold: int | None = None
    long_context_input: float | None = None
    long_context_output: float | None = None
    long_context_cache_read: float | None = None
    long_context_cache_creation: float | None = None
    # Flag for non-public rates (preview/confidential SKUs). When True,
    # `compute_cost` returns cost_source='estimated' instead of
    # 'pricing_table' so downstream consumers (snapshot, leaderboards,
    # audit) can distinguish provider-reported from price-sheet-estimated
    # numbers.
    estimated: bool = False

    def for_input_size(self, input_tokens: int) -> ModelPricing:
        """Return effective pricing for a request of ``input_tokens`` size.

        If no long-context tier is configured, returns ``self`` unchanged.
        Otherwise, returns a fresh ModelPricing with the long_context_*
        overrides applied (or the short-context value where the override
        is None).
        """
        if (
            self.long_context_threshold is None
            or input_tokens <= self.long_context_threshold
        ):
            return self
        return ModelPricing(
            input=self.long_context_input if self.long_context_input is not None else self.input,
            output=self.long_context_output if self.long_context_output is not None else self.output,
            cache_read=(
                self.long_context_cache_read
                if self.long_context_cache_read is not None
                else self.cache_read
            ),
            cache_creation=(
                self.long_context_cache_creation
                if self.long_context_cache_creation is not None
                else self.cache_creation
            ),
            estimated=self.estimated,
        )


# May be outdated. Use and update with care. Per-provider header comment
# names the source page + the date the entries were last verified there.
# Entries marked UNVERIFIED are placeholders kept so lookups don't return
# None for historical data — don't trust their cost_usd at scale.
PRICING_TABLE: dict[str, ModelPricing] = {
    # ---- Anthropic — verified anthropic.com/api/pricing 2026-05-11 ----
    # cache_creation = 5m-TTL write rate (1h-TTL is 2× input, not tracked).
    "claude-opus-4-7": ModelPricing(input=5.0, output=25.0, cache_read=0.50, cache_creation=6.25),
    "claude-opus-4-6": ModelPricing(input=5.0, output=25.0, cache_read=0.50, cache_creation=6.25),
    "claude-sonnet-4-6": ModelPricing(input=3.0, output=15.0, cache_read=0.30, cache_creation=3.75),
    "claude-sonnet-4-5": ModelPricing(input=3.0, output=15.0, cache_read=0.30, cache_creation=3.75),
    "claude-haiku-4-5": ModelPricing(input=1.0, output=5.0, cache_read=0.10, cache_creation=1.25),
    # Preview SKU; rates estimated, so cost_source='estimated' (estimated=True).
    "claude-mythos-preview": ModelPricing(
        input=25.0, output=125.0, cache_read=2.50, cache_creation=31.25,
        estimated=True,
    ),

    # ---- OpenAI — verified platform.openai.com/api/pricing 2026-05-11 ----
    # gpt-5.4 / gpt-5.4-pro / gpt-5.5 / gpt-5.5-pro have a long-context
    # tier (input > 272K). "-pro" SKUs don't offer cached input.
    "gpt-5": ModelPricing(input=1.25, output=10.0, cache_read=0.125),
    "gpt-5-mini": ModelPricing(input=0.25, output=2.0, cache_read=0.025),
    "gpt-5-nano": ModelPricing(input=0.05, output=0.40, cache_read=0.005),
    "gpt-5-pro": ModelPricing(input=15.0, output=120.0),
    "gpt-5.1": ModelPricing(input=1.25, output=10.0, cache_read=0.125),
    "gpt-5.2": ModelPricing(input=1.75, output=14.0, cache_read=0.175),
    "gpt-5.2-pro": ModelPricing(input=21.0, output=168.0),
    "gpt-5.4": ModelPricing(
        input=2.50, output=15.0, cache_read=0.25,
        long_context_threshold=272_000,
        long_context_input=5.0, long_context_output=22.50, long_context_cache_read=0.50,
    ),
    "gpt-5.4-mini": ModelPricing(input=0.75, output=4.50, cache_read=0.075),
    "gpt-5.4-nano": ModelPricing(input=0.20, output=1.25, cache_read=0.02),
    "gpt-5.4-pro": ModelPricing(
        input=30.0, output=180.0,
        long_context_threshold=272_000,
        long_context_input=60.0, long_context_output=270.0,
    ),
    "gpt-5.5": ModelPricing(
        input=5.0, output=30.0, cache_read=0.50,
        long_context_threshold=272_000,
        long_context_input=10.0, long_context_output=45.0, long_context_cache_read=1.0,
    ),
    "gpt-5.5-pro": ModelPricing(
        input=30.0, output=180.0,
        long_context_threshold=272_000,
        long_context_input=60.0, long_context_output=270.0,
    ),
    "gpt-4.1": ModelPricing(input=2.0, output=8.0, cache_read=0.50),
    "gpt-4.1-mini": ModelPricing(input=0.40, output=1.60, cache_read=0.10),
    "gpt-4.1-nano": ModelPricing(input=0.10, output=0.40, cache_read=0.025),
    "gpt-4o": ModelPricing(input=2.50, output=10.0, cache_read=1.25),
    "gpt-4o-mini": ModelPricing(input=0.15, output=0.60, cache_read=0.075),

    # ---- Gemini — verified ai.google.dev/gemini-api/docs/pricing 2026-05-11 ----
    # gemini-3.1-pro* / gemini-3-pro* / gemini-2.5-pro carry a long-context
    # tier (input > 200K). Aliases registered for routing-path robustness.
    "gemini-3.1-pro-preview": ModelPricing(
        input=2.0, output=12.0, cache_read=0.20,
        long_context_threshold=200_000,
        long_context_input=4.0, long_context_output=18.0, long_context_cache_read=0.40,
    ),
    "gemini-3.1-pro": ModelPricing(
        input=2.0, output=12.0, cache_read=0.20,
        long_context_threshold=200_000,
        long_context_input=4.0, long_context_output=18.0, long_context_cache_read=0.40,
    ),
    "gemini-3-pro": ModelPricing(
        input=2.0, output=12.0, cache_read=0.20,
        long_context_threshold=200_000,
        long_context_input=4.0, long_context_output=18.0, long_context_cache_read=0.40,
    ),
    "gemini-3-pro-preview": ModelPricing(
        input=2.0, output=12.0, cache_read=0.20,
        long_context_threshold=200_000,
        long_context_input=4.0, long_context_output=18.0, long_context_cache_read=0.40,
    ),
    "gemini-2.5-pro": ModelPricing(
        input=1.25, output=10.0, cache_read=0.125,
        long_context_threshold=200_000,
        long_context_input=2.50, long_context_output=15.0, long_context_cache_read=0.25,
    ),
    "gemini-2.5-flash": ModelPricing(input=0.30, output=2.50, cache_read=0.03),
    "gemini-2.5-flash-lite": ModelPricing(input=0.10, output=0.40, cache_read=0.01),

    # ---- GLM (Z.ai direct) — verified docs.z.ai/guides/overview/pricing 2026-05-11 ----
    # `zai/` prefix → https://api.z.ai/api/paas/v4. Cache storage is
    # Limited-time Free through Apr 2026 (not in read/write rates).
    # OpenRouter pass-through prices differ; lookup currently resolves
    # both prefixes to these direct rates.
    "glm-5.2": ModelPricing(input=1.40, output=4.40, cache_read=0.26),
    "glm-5.1": ModelPricing(input=1.40, output=4.40, cache_read=0.26),
    "glm-5": ModelPricing(input=1.0, output=3.20, cache_read=0.20),
    "glm-5-turbo": ModelPricing(input=1.20, output=4.0, cache_read=0.24),
    "glm-4.7": ModelPricing(input=0.60, output=2.20, cache_read=0.11),
    "glm-4.7-flashx": ModelPricing(input=0.07, output=0.40, cache_read=0.01),
    "glm-4.7-flash": ModelPricing(input=0.0, output=0.0, cache_read=0.0),  # free tier
    "glm-4.5-flash": ModelPricing(input=0.0, output=0.0, cache_read=0.0),  # free tier
    "glm-4.6": ModelPricing(input=0.60, output=2.20, cache_read=0.11),
    "glm-4.5": ModelPricing(input=0.60, output=2.20, cache_read=0.11),
    "glm-4.5-x": ModelPricing(input=2.20, output=8.90, cache_read=0.45),
    "glm-4.5-air": ModelPricing(input=0.20, output=1.10, cache_read=0.03),
    "glm-4.5-airx": ModelPricing(input=1.10, output=4.50, cache_read=0.22),

    # ---- MiniMax — verified platform.minimax.io/docs/guides/pricing-paygo 2026-05-11 ----
    # Flat-priced across the 262K context window. cache_creation = $0.375/M
    # for M2.x. LiteLLM passes the upstream id through unchanged → CamelCase
    # keys are live; lowercase aliases below kept for legacy code paths.
    "MiniMax-M3": ModelPricing(
        input=0.30, output=1.20, cache_read=0.06,
        long_context_threshold=512_000,
        long_context_input=0.60, long_context_output=2.40, long_context_cache_read=0.12,
    ),
    "MiniMax-M2.7": ModelPricing(input=0.30, output=1.20, cache_read=0.06, cache_creation=0.375),
    "MiniMax-M2.7-highspeed": ModelPricing(input=0.60, output=2.40, cache_read=0.06, cache_creation=0.375),
    "MiniMax-M2.5": ModelPricing(input=0.30, output=1.20, cache_read=0.03, cache_creation=0.375),
    "MiniMax-M2.5-highspeed": ModelPricing(input=0.60, output=2.40, cache_read=0.03, cache_creation=0.375),
    "MiniMax-M2.1": ModelPricing(input=0.30, output=1.20, cache_read=0.03, cache_creation=0.375),
    "MiniMax-M2.1-highspeed": ModelPricing(input=0.60, output=2.40, cache_read=0.03, cache_creation=0.375),
    "MiniMax-M2": ModelPricing(input=0.30, output=1.20, cache_read=0.03, cache_creation=0.375),
    "M2-her": ModelPricing(input=0.30, output=1.20),  # caching not offered
    "minimax-m2.7": ModelPricing(input=0.30, output=1.20, cache_read=0.06, cache_creation=0.375),
    "minimax-m2.5": ModelPricing(input=0.30, output=1.20, cache_read=0.03, cache_creation=0.375),
    "minimax-m2": ModelPricing(input=0.30, output=1.20, cache_read=0.03, cache_creation=0.375),

    # ---- Moonshot Kimi — verified platform.moonshot.ai 2026-05-11 ----
    # cache_read = cache-hit input; base input = cache-miss. No long-context
    # tier in K2.x. -turbo SKUs bill ~1.9× input / ~3.2× output of the base.
    "kimi-k2.6": ModelPricing(input=0.95, output=4.0, cache_read=0.16),
    "kimi-k2.5": ModelPricing(input=0.60, output=3.0, cache_read=0.10),
    "kimi-k2": ModelPricing(input=0.60, output=2.50, cache_read=0.15),  # alias for 0711
    "kimi-k2-0711-preview": ModelPricing(input=0.60, output=2.50, cache_read=0.15),
    "kimi-k2-0905-preview": ModelPricing(input=0.60, output=2.50, cache_read=0.15),
    "kimi-k2-thinking": ModelPricing(input=0.60, output=2.50, cache_read=0.15),
    "kimi-k2-turbo-preview": ModelPricing(input=1.15, output=8.0, cache_read=0.15),
    "kimi-k2-thinking-turbo": ModelPricing(input=1.15, output=8.0, cache_read=0.15),

    # ---- DeepSeek (api-docs.deepseek.com/quick_start/pricing) ----
    "deepseek-v4-pro": ModelPricing(input=0.435, output=0.87, cache_read=0.003625),

    # Other models: add with verified rates from the provider's page
    # (direct provider preferred over OpenRouter pass-through when both
    # exist). Lookup returns None for unregistered ids → cost_source='unknown'.
}


def lookup_pricing(model_id: str, input_tokens: int = 0) -> ModelPricing | None:
    """Find pricing for a model_id; tries exact match, then strips prefixes.

    When the resolved entry has a long-context tier configured, ``input_tokens``
    decides which tier applies (see ``ModelPricing.for_input_size``). The
    default ``input_tokens=0`` always returns short-context pricing, which is
    the right answer for callers that don't have a usage record (e.g. unit
    tests of lookup wiring).

    Examples that hit:
      "anthropic/claude-haiku-4-5"
      "claude-haiku-4-5"
      "openrouter/anthropic/claude-haiku-4-5"
      "openai/gpt-5-mini"
    Examples that miss (return None):
      "openrouter/qwen/qwen3-coder"
      "openai/llama-3.3-70b" (gateway-served OSS)
    """
    entry: ModelPricing | None = None
    if model_id in PRICING_TABLE:
        entry = PRICING_TABLE[model_id]
    else:
        parts = model_id.split("/")
        for i in range(len(parts)):
            candidate = "/".join(parts[i:])
            if candidate in PRICING_TABLE:
                entry = PRICING_TABLE[candidate]
                break
    if entry is None:
        return None
    return entry.for_input_size(input_tokens)


def compute_cost(
    model_id: str, usage: NormalizedUsage
) -> tuple[float | None, str]:
    """Return (cost_usd, cost_source) for a usage record.

    cost_source:
      'pricing_table' — we have prices for this model; cost is exact
                        modulo per-token rounding.
      'estimated'     — model in table but flagged estimated=True.
      'unknown'       — model not in table; cost is None.

    Computation: the three input categories are DISJOINT (verified live
    against the Anthropic API 2026-05-13 — input_tokens reports base only,
    cache_read_input_tokens and cache_creation_input_tokens report their
    own counts independently). LiteLLM-backed providers (OpenAI, Gemini)
    surface ``prompt_tokens`` which IS inclusive — `litellm_client.py`
    subtracts cache_read from it at population time so NormalizedUsage
    stays disjoint regardless of provider.
    """
    # Tier selection uses the *total* prompt size (= base + cached read +
    # cache creation), not just the base, because that's the actual context
    # the model processed. The cached portion still occupies the context
    # window — caching only changes how it's billed, not whether it's
    # included in the tier-threshold calculation. For OpenAI gpt-5.4/5.5
    # the docs say "input > 272K" trips long context; that "input" means
    # the request's full prompt as the model sees it.
    total_prompt = (
        usage.input_tokens + usage.cache_read_tokens + usage.cache_creation_tokens
    )
    p = lookup_pricing(model_id, input_tokens=total_prompt)
    if p is None:
        return None, "unknown"

    cost = (
        usage.input_tokens / 1_000_000 * p.input
        + usage.output_tokens / 1_000_000 * p.output
        + usage.cache_read_tokens / 1_000_000 * p.cache_read
        + usage.cache_creation_tokens / 1_000_000 * p.cache_creation
    )
    # Round to 6 decimal places — sub-cent precision is meaningful at scale.
    return round(cost, 6), ("estimated" if p.estimated else "pricing_table")


def compute_total_cost(
    model_id: str, per_call_usages: list[NormalizedUsage],
) -> tuple[float | None, str]:
    """Sum per-call costs for an episode, applying long-context tiering
    per call so each call routes to the correct price bracket.

    Aggregating tokens and calling ``compute_cost`` once is incorrect for
    tier-priced models: a 5M-token episode aggregate trips the long-context
    threshold even if every individual call was below it, overpricing all
    the short-context calls. Callers should pass the per-call usages.

    Returns ``(None, "unknown")`` if the model isn't in PRICING_TABLE.
    Empty list → ``(0.0, "pricing_table")`` (no calls cost $0).
    """
    total = 0.0
    for u in per_call_usages:
        call_cost, _ = compute_cost(model_id, u)
        if call_cost is None:
            return None, "unknown"
        total += call_cost
    return round(total, 6), "pricing_table"
