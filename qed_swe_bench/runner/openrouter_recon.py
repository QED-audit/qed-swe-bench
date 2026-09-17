"""Post-episode cost reconciliation for OpenRouter-routed cells.

Our `cost.json` field `cost_usd` is *inferred* — computed by
`runner/cost.py:compute_cost` from token counts × the static
`PRICING_TABLE`. For OpenRouter-routed cells that's a best-effort
estimate: the table assumes `cache_read = input_price` (no discount) on
the principle that OR's cache pass-through is provider-dependent and
historically unreliable (see docs/FINDINGS.md F-2 / F-5).

This module closes the loop by hitting OpenRouter's
`GET /api/v1/generation?id=<gen_id>` endpoint per OR turn (using the
`provider_response_id` captured in `transcript.jsonl`) and aggregating
the authoritative `total_cost`, `cache_discount`, and
`upstream_inference_cost` fields into a reconciliation summary that's
written into `cost.json` alongside the original inferred `cost_usd`.

Both numbers are kept so post-hoc audits can quantify the
inferred-vs-authoritative drift per cell — that's the F-N-worthy data
point this enables.

Best-effort by design: any HTTP error / missing id / malformed
response logs and skips the affected turn. The reconciliation summary
records `incomplete=True` if any turn was skipped so downstream
tooling can flag it.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

log = logging.getLogger(__name__)


# OpenRouter API base. The generation lookup endpoint is documented at
# https://openrouter.ai/docs/api-reference/generations/get-generation —
# `GET /api/v1/generation?id=<gen_id>` returning per-generation cost,
# cache_discount, upstream_inference_cost, etc. Override base via
# OPENROUTER_API_BASE for a custom proxy.
DEFAULT_API_BASE = "https://openrouter.ai/api/v1"

# Per-request HTTP timeout for the lookup. Each call is a small JSON
# fetch; 10s is generous and prevents a wedged endpoint from holding
# up an episode finish indefinitely. Override with
# OPENROUTER_RECON_TIMEOUT_S.
DEFAULT_TIMEOUT_S = 10


def _api_base() -> str:
    return os.environ.get("OPENROUTER_API_BASE", DEFAULT_API_BASE).rstrip("/")


def _timeout_s() -> float:
    raw = os.environ.get("OPENROUTER_RECON_TIMEOUT_S")
    if raw is None:
        return DEFAULT_TIMEOUT_S
    try:
        v = float(raw)
        if v <= 0:
            raise ValueError
        return v
    except ValueError:
        log.warning("OPENROUTER_RECON_TIMEOUT_S=%r is not a positive number; "
                    "using default %ds", raw, DEFAULT_TIMEOUT_S)
        return DEFAULT_TIMEOUT_S


@dataclass(frozen=True)
class GenerationCost:
    """Per-generation authoritative cost view from OR's lookup endpoint."""

    generation_id: str
    total_cost: float | None
    cache_discount: float | None
    upstream_inference_cost: float | None
    provider_name: str | None
    native_tokens_cached: int | None


@dataclass
class ReconciliationSummary:
    """Aggregated reconciliation for one episode."""

    n_turns_recorded: int = 0       # OR turns we attempted to look up
    n_turns_resolved: int = 0       # successfully fetched
    n_turns_skipped: int = 0        # missing id / 4xx / 5xx / network err
    cost_usd_authoritative: float = 0.0  # sum of total_cost
    cache_discount_total: float = 0.0    # sum of cache_discount (may be 0/None)
    upstream_cost_total: float = 0.0     # sum of upstream_inference_cost
    providers_seen: list[str] = field(default_factory=list)
    incomplete: bool = False        # True if any turn was skipped

    def to_dict(self) -> dict:
        return {
            "n_turns_recorded": self.n_turns_recorded,
            "n_turns_resolved": self.n_turns_resolved,
            "n_turns_skipped": self.n_turns_skipped,
            "cost_usd_authoritative": round(self.cost_usd_authoritative, 6),
            "cache_discount_total": round(self.cache_discount_total, 6),
            "upstream_cost_total": round(self.upstream_cost_total, 6),
            "providers_seen": sorted(set(self.providers_seen)),
            "incomplete": self.incomplete,
        }


def fetch_generation_cost(
    generation_id: str,
    *,
    api_key: str,
    api_base: str | None = None,
    timeout_s: float | None = None,
) -> GenerationCost | None:
    """Look up one generation's cost via OR's API. Returns None on any error."""
    base = (api_base or _api_base()).rstrip("/")
    url = f"{base}/generation"
    try:
        r = httpx.get(
            url,
            params={"id": generation_id},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout_s or _timeout_s(),
        )
        r.raise_for_status()
        body = r.json()
    except (httpx.HTTPError, json.JSONDecodeError, ValueError) as exc:
        log.warning("OR generation lookup failed for %s: %s: %s",
                    generation_id, type(exc).__name__, exc)
        return None
    # OR wraps the payload in a `data` field per their spec.
    data = body.get("data") or body
    if not isinstance(data, dict):
        log.warning("OR generation lookup returned non-dict payload for %s",
                    generation_id)
        return None
    return GenerationCost(
        generation_id=generation_id,
        total_cost=data.get("total_cost"),
        cache_discount=data.get("cache_discount"),
        upstream_inference_cost=data.get("upstream_inference_cost"),
        provider_name=data.get("provider_name"),
        native_tokens_cached=data.get("native_tokens_cached"),
    )


def collect_or_generation_ids(transcript_path: Path) -> list[str]:
    """Read transcript.jsonl and return the OR generation_ids per AI turn.

    Filters to ids matching OR's `gen-...` prefix so direct-provider
    response ids (e.g. Anthropic msg_..., OpenAI chatcmpl_...) don't get
    sent to OR's lookup endpoint by mistake.
    """
    out: list[str] = []
    if not transcript_path.is_file():
        return out
    with transcript_path.open(encoding="utf-8") as f:
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
            gid = d.get("provider_response_id")
            if isinstance(gid, str) and gid.startswith("gen-"):
                out.append(gid)
    return out


def reconcile_run_dir(
    run_dir: Path,
    *,
    api_key: str | None = None,
    api_base: str | None = None,
    timeout_s: float | None = None,
    inter_request_sleep_s: float = 0.0,
) -> ReconciliationSummary | None:
    """Reconcile costs for one episode's run-dir.

    Returns None when the run is not OR-routed (no `gen-...` ids found
    in transcript.jsonl) — caller can short-circuit and skip the
    cost.json write entirely. Returns a populated `ReconciliationSummary`
    otherwise, even if some turns failed (`incomplete=True` flags it).

    `api_key` defaults to `OPENROUTER_API_KEY`. `inter_request_sleep_s`
    inserts a small pause between calls to be polite with OR's API
    rate limits — default 0 (no sleep) since the lookup endpoint is
    light.
    """
    transcript = run_dir / "transcript.jsonl"
    gen_ids = collect_or_generation_ids(transcript)
    if not gen_ids:
        return None

    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        log.warning("OPENROUTER_API_KEY not set; skipping OR cost reconciliation")
        s = ReconciliationSummary()
        s.n_turns_recorded = len(gen_ids)
        s.n_turns_skipped = len(gen_ids)
        s.incomplete = True
        return s

    summary = ReconciliationSummary(n_turns_recorded=len(gen_ids))
    for i, gid in enumerate(gen_ids):
        gc = fetch_generation_cost(
            gid, api_key=key, api_base=api_base, timeout_s=timeout_s,
        )
        if gc is None:
            summary.n_turns_skipped += 1
            summary.incomplete = True
        else:
            summary.n_turns_resolved += 1
            if gc.total_cost is not None:
                summary.cost_usd_authoritative += float(gc.total_cost)
            if gc.cache_discount is not None:
                summary.cache_discount_total += float(gc.cache_discount)
            if gc.upstream_inference_cost is not None:
                summary.upstream_cost_total += float(gc.upstream_inference_cost)
            if gc.provider_name:
                summary.providers_seen.append(gc.provider_name)
        if inter_request_sleep_s > 0 and i < len(gen_ids) - 1:
            time.sleep(inter_request_sleep_s)
    return summary
