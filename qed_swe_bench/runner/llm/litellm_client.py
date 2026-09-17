"""LiteLLM-backed client for non-Anthropic providers and OpenAI-compatible gateways.

- For openai/* model-ids, honors OPENAI_API_BASE so users can route through any
  OpenAI-compatible gateway (LiteLLM proxy / vLLM / Ollama / in-house).
  This is a primary deployment path for evaluating OSS models without hosting
  their weights ourselves.

- LiteLLM normalizes provider responses to the OpenAI Chat Completions shape;
  we just translate that into NormalizedResponse.

- Tool-call arguments arrive as a JSON-encoded string; some smaller / OSS models
  emit malformed JSON. We log the raw text and proceed with empty arguments
  rather than crashing the loop. The loop downstream can decide whether to
  abort the episode (likely for any tool requiring non-trivial args).
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from typing import Any

import litellm

from qed_swe_bench.runner.llm.base import (
    NormalizedResponse,
    NormalizedToolCall,
    NormalizedUsage,
)

log = logging.getLogger(__name__)


# Per-request HTTP timeout (seconds). Generous default — multi-turn
# OpenRouter calls with 5+ turns of tool_use/tool_result history can
# legitimately take 60-120s to stream a complete response. The first
# soak's `TimeoutError() (via ExceptionGroup)` failures were
# downstream HTTP timeouts firing well below the per-tuple
# `episode_timeout_s` budget.
#
# Override per deployment with LITELLM_REQUEST_TIMEOUT_S; can be set
# either in `.env` or as a one-shot before `qed_swe_bench benchmark`.
DEFAULT_REQUEST_TIMEOUT_S = 300

# Retries litellm does internally on RateLimitError / APIError /
# ServiceUnavailableError / Timeout, with exponential backoff and
# Retry-After header awareness. Matches AnthropicNative's
# DEFAULT_RATE_LIMIT_MAX_RETRIES so behavior is consistent across the
# routing layer; OpenAI's 500k-TPM cap typically returns
# Retry-After ~6s, so 5 attempts means up to ~30s on the worst call.
DEFAULT_RATE_LIMIT_MAX_RETRIES = 5


def _is_gpt5_family(model_id: str) -> bool:
    """Match `openai/gpt-5`, `openai/gpt-5.1`, `openai/gpt-5.5`, etc.

    The gpt-5 family rejects `temperature=0` (only temperature=1 is supported
    for reasoning-capable variants); we use this to drop the default before
    the request goes out.
    """
    if not model_id.startswith("openai/"):
        return False
    rest = model_id[len("openai/"):]
    return rest == "gpt-5" or rest.startswith("gpt-5.") or rest.startswith("gpt-5-")


def _is_gemini3_family(model_id: str) -> bool:
    """Match Gemini 3.x models regardless of routing prefix.

    LiteLLM warns explicitly: `temperature < 1.0 for Gemini 3 models can
    cause infinite loops, degraded reasoning performance, and failure on
    complex tasks`. We've observed both — empty `output_tokens=0` returns
    after 100s+ stalls, and HTTP-level header-stuck hangs that exhaust the
    request timeout. Match on the substring `gemini-3` so direct
    (`gemini/gemini-3.1-pro-preview`), OpenRouter (`openrouter/google/...`),
    LiteLLM-proxy (`litellm_proxy/...`), Vertex (`vertex_ai/...`), and any
    gateway-prefixed routes are all covered.
    """
    return "gemini-3" in model_id.lower()


def _request_timeout_s() -> int:
    """Resolve the per-request HTTP timeout. Env override → default."""
    raw = os.environ.get("LITELLM_REQUEST_TIMEOUT_S")
    if raw is None:
        return DEFAULT_REQUEST_TIMEOUT_S
    try:
        v = int(raw)
        if v <= 0:
            raise ValueError
        return v
    except ValueError:
        log.warning(
            "LITELLM_REQUEST_TIMEOUT_S=%r is not a positive int; "
            "using default %ds", raw, DEFAULT_REQUEST_TIMEOUT_S,
        )
        return DEFAULT_REQUEST_TIMEOUT_S


def _rate_limit_max_retries() -> int:
    """Resolve litellm's internal retry count for rate-limit / 5xx
    responses. Env override → default. Returns 0 to disable retries."""
    raw = os.environ.get("QED_SWE_BENCH_RATE_LIMIT_MAX_RETRIES")
    if raw is None:
        return DEFAULT_RATE_LIMIT_MAX_RETRIES
    try:
        v = int(raw)
        if v < 0:
            raise ValueError
        return v
    except ValueError:
        log.warning(
            "QED_SWE_BENCH_RATE_LIMIT_MAX_RETRIES=%r is not a non-negative int; "
            "using default %d", raw, DEFAULT_RATE_LIMIT_MAX_RETRIES,
        )
        return DEFAULT_RATE_LIMIT_MAX_RETRIES


# OpenAI's TPM-cap rate-limit messages embed the actual time-to-refill in
# plain English: "Please try again in 5.247s" / "in 921ms". Tenacity's
# default exponential backoff (1, 2, 4s) is too short — at 500k TPM, a
# 85k-token request needs ~10s of bucket refill, so a 2s wait still fails.
# Parse the provider-supplied delay and sleep that exactly.
_RETRY_AFTER_SECONDS_RE = re.compile(r"in\s+([\d.]+)\s*s\b", re.IGNORECASE)
_RETRY_AFTER_MS_RE = re.compile(r"in\s+([\d.]+)\s*ms\b", re.IGNORECASE)


def _parse_retry_after(message: str) -> float | None:
    """Extract the seconds value from a rate-limit error message.

    Recognizes OpenAI's two formats: '...try again in 5.247s' and
    '...try again in 921ms'. Returns None if neither is present.
    """
    m = _RETRY_AFTER_MS_RE.search(message)
    if m:
        return float(m.group(1)) / 1000.0
    m = _RETRY_AFTER_SECONDS_RE.search(message)
    if m:
        return float(m.group(1))
    return None


def _is_rate_limit(exc: BaseException) -> bool:
    """Duck-type check: is this exception a rate-limit error?

    We don't import litellm.exceptions.RateLimitError because LiteLLM
    occasionally wraps the same condition in different classes across
    versions. Match on the class name and on the message body, both of
    which are stable across LiteLLM releases.
    """
    cls_name = type(exc).__name__
    msg = str(exc).lower()
    return (
        "ratelimit" in cls_name.lower()
        or "rate_limit_exceeded" in msg
        or "rate limit reached" in msg
    )


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read `key` from obj uniformly, whether obj is a dict or an attribute holder.

    LiteLLM responses may be pydantic models, plain dicts, or namespaces depending
    on the provider — this hides the difference.
    """
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class LiteLLMClient:
    """LLM client backed by `litellm.completion`."""

    def __init__(
        self,
        model: str,
        *,
        api_base: str | None = None,
        api_key: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> None:
        # litellm expects model-ids with provider prefix verbatim (e.g.
        # "openai/gpt-5", "gemini/gemini-2.5-pro", "openrouter/...").
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.params = dict(params or {})
        self.route = "litellm_gateway" if api_base else "litellm"

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int,
        seed: int | None = None,
        safety_identifier: str | None = None,
    ) -> NormalizedResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "max_tokens": max_tokens,
            "temperature": 0,
            "timeout": _request_timeout_s(),
            "num_retries": _rate_limit_max_retries(),
        }
        # Some providers reject `seed` (LiteLLM 400s with UnsupportedParamsError
        # unless drop_params is set). Known so far: Gemini, Z.ai. OpenAI supports
        # it; Anthropic's native client ignores it. Gate forwarding so a
        # multi-provider config doesn't trip those cells.
        seed_unsupported = self.model.startswith("gemini/") or self.model.startswith("zai/")
        if seed is not None and not seed_unsupported:
            kwargs["seed"] = seed
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.api_key:
            kwargs["api_key"] = self.api_key
        # OpenAI's documented lever for limiting cyber_policy revocation
        # blast radius — without this, a single bad-looking request can
        # revoke access org-wide; with it, revocations apply per identifier.
        # See https://developers.openai.com/api/docs/guides/safety-checks/cybersecurity.
        # Only forwarded for openai/* model ids; other providers ignore it
        # (or 400 on the unknown kwarg via litellm), so we gate it.
        if safety_identifier and self.model.startswith("openai/"):
            kwargs["safety_identifier"] = safety_identifier

        # OpenAI gpt-5 family rejects temperature=0 (only temperature=1 is
        # accepted; gpt-5.5 + xhigh reasoning is the same family). Drop the
        # default rather than 400 the user out. Determinism is lost but
        # the call works; pair this with `seed` (above) for what little
        # reproducibility OpenAI offers on reasoning models.
        # Moonshot's Kimi K2.x family has the same constraint (verified
        # 2026-05 on kimi-k2.6: "MoonshotException - invalid temperature:
        # only 1 is allowed for this model").
        # Gemini 3.x family: same drop, different failure mode — temperature
        # < 1.0 causes infinite loops / empty `output_tokens=0` returns and
        # in some cases HTTP-level header-stuck hangs (LiteLLM emits an
        # explicit warning when temperature is < 1.0 on Gemini 3).
        if (
            _is_gpt5_family(self.model)
            or self.model.startswith("moonshot/")
            or _is_gemini3_family(self.model)
        ):
            kwargs.pop("temperature", None)

        # Per-model param overrides from `ModelSpec.params` (e.g.
        # `reasoning_effort: xhigh`). A None value pops the kwarg, which
        # lets a config explicitly suppress a default we'd otherwise add.
        for k, v in self.params.items():
            if v is None:
                kwargs.pop(k, None)
            else:
                kwargs[k] = v

        # Rate-limit-aware retry on top of litellm's own num_retries. The
        # SDK retry path uses tenacity's default exponential backoff, which
        # is too short for OpenAI's TPM-cap pattern (85k tokens needs ~10s
        # of bucket refill at 500k TPM, but tenacity waits 1-4s). Parse the
        # provider-supplied retry-after from the error message and sleep
        # that exactly; fall back to exponential if absent.
        max_retries = _rate_limit_max_retries()
        for attempt in range(max_retries + 1):
            try:
                resp = litellm.completion(**kwargs)
                return self._to_normalized(resp)
            except Exception as exc:
                if not _is_rate_limit(exc):
                    log.warning(
                        "non-retriable %s on %s (attempt %d): body=%s",
                        type(exc).__name__, self.model, attempt + 1,
                        str(exc)[:400],
                    )
                    raise
                if attempt >= max_retries:
                    log.error(
                        "rate limit on %s, exhausted %d retries — giving up. "
                        "body=%s",
                        self.model, max_retries, str(exc)[:400],
                    )
                    raise
                retry_after = _parse_retry_after(str(exc))
                if retry_after is None:
                    # Exponential fallback: 1, 2, 4, 8, 16 capped at 30s.
                    retry_after = min(2.0 ** attempt, 30.0)
                # Add 0-500ms jitter so concurrent tuples don't all wake at
                # the same instant and trip the bucket again.
                wait = retry_after + random.uniform(0.0, 0.5)
                log.warning(
                    "rate limit on %s (attempt %d/%d), sleeping %.2fs. "
                    "body=%s",
                    self.model, attempt + 1, max_retries + 1, wait,
                    str(exc)[:200],
                )
                time.sleep(wait)
        # Unreachable — the loop either returns or raises.
        raise RuntimeError("retry loop exited without returning")

    @staticmethod
    def _to_normalized(resp: Any) -> NormalizedResponse:
        choice = resp.choices[0]
        msg = choice.message

        tool_calls: list[NormalizedToolCall] = []
        raw_tool_calls = _get(msg, "tool_calls") or []
        for tc in raw_tool_calls:
            fn = _get(tc, "function")
            if fn is None:
                continue
            name = _get(fn, "name") or ""
            raw_args = _get(fn, "arguments")
            try:
                args = json.loads(raw_args) if raw_args else {}
                if not isinstance(args, dict):
                    args = {}
            except (json.JSONDecodeError, TypeError):
                log.warning(
                    "model %s emitted malformed tool-call arguments; using empty dict. raw=%r",
                    _get(resp, "model", "?"),
                    raw_args,
                )
                args = {}
            tc_id = _get(tc, "id") or ""
            tool_calls.append(NormalizedToolCall(id=tc_id, name=name, arguments=args))

        usage = _get(resp, "usage")
        # LiteLLM's `prompt_tokens` is INCLUSIVE of cached_tokens (the
        # OpenAI/LiteLLM convention). NormalizedUsage uses the DISJOINT
        # convention (input_tokens = base only, cache_read separate, cache
        # creation separate) so `compute_cost` can treat the three as
        # distinct token classes regardless of upstream provider. Subtract
        # cache_read here so the value stored downstream is the base
        # portion only. Verified live against Anthropic + OpenAI APIs
        # 2026-05-13 (Anthropic native already disjoint, OpenAI inclusive).
        raw_prompt = int(_get(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(_get(usage, "completion_tokens", 0) or 0)

        cache_read = 0
        details = _get(usage, "prompt_tokens_details")
        if details is not None:
            cache_read = int(_get(details, "cached_tokens", 0) or 0)
        input_tokens = max(0, raw_prompt - cache_read)

        reasoning_tokens = 0
        ct_details = _get(usage, "completion_tokens_details")
        if ct_details is not None:
            reasoning_tokens = int(_get(ct_details, "reasoning_tokens", 0) or 0)

        # OpenAI-style: content is null for pure tool-call responses.
        content = _get(msg, "content")
        text = content if content else None

        try:
            raw = resp.model_dump()  # pydantic v2
        except Exception:
            raw = dict(resp) if isinstance(resp, dict) else {}

        # Capture the provider's response id. For OpenRouter routes this is
        # the OR `gen-XXX...` identifier — reused later by openrouter_recon
        # to fetch the authoritative billed cost via GET /api/v1/generation.
        # For direct providers (OpenAI, Gemini, Moonshot, etc.) the id is
        # provider-specific and currently unused downstream.
        response_id = _get(resp, "id") or None

        # Moonshot-style reasoning trace. Moonshot's API requires this
        # field be replayed verbatim in subsequent assistant messages
        # (see messages.py); without it LiteLLM injects a placeholder
        # and warns, but the model effectively loses access to its
        # prior thinking trace turn-over-turn. Capturing it here lets
        # the message builder replay it. Other LiteLLM providers either
        # don't surface a separate reasoning trace via this field
        # (gpt-5/Gemini reasoning is server-side; GLM/MiniMax may use
        # different fields) or don't require client-side replay.
        reasoning_content = _get(msg, "reasoning_content") or None

        return NormalizedResponse(
            text=text,
            tool_calls=tuple(tool_calls),
            usage=NormalizedUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read,
                reasoning_tokens=reasoning_tokens,
            ),
            stop_reason=_get(choice, "finish_reason") or "unknown",
            model=_get(resp, "model", "") or "",
            raw=raw,
            provider_response_id=response_id,
            reasoning_content=reasoning_content,
        )
