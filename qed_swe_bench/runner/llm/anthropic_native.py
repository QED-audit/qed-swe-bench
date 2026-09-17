"""Anthropic native SDK client with prompt-cache support.

We use the native `anthropic` Python SDK rather than LiteLLM's Anthropic
passthrough because LiteLLM's cache_control handling has historically been
fragile (bench-v8's TODO-litellm-benchmark.md T2). Native SDK gives us
direct cache_creation_input_tokens / cache_read_input_tokens in the
response, which we surface as NormalizedUsage.cache_creation_tokens /
cache_read_tokens.
"""

from __future__ import annotations

import copy
import logging
import os
from typing import Any

import anthropic

from qed_swe_bench.runner.llm.base import (
    NormalizedResponse,
    NormalizedToolCall,
    NormalizedUsage,
)

log = logging.getLogger(__name__)


# Per-request HTTP timeout for the Anthropic SDK (seconds). The
# anthropic Python client's default is 600s for non-streaming, but
# observed cap-demo runs hit a wrapped TimeoutError() at <=600s on
# multi-turn calls — likely a tighter timeout somewhere in the
# httpx → anyio → asyncio.TaskGroup stack the SDK uses internally.
# Pass an explicit timeout so the value is auditable from the
# AnthropicNative client surface and overridable per deployment via
# `ANTHROPIC_REQUEST_TIMEOUT_S`.
DEFAULT_REQUEST_TIMEOUT_S = 300

# Retries the anthropic SDK does internally on 429 / 5xx, with exponential
# backoff and Retry-After header awareness. 5 attempts at OpenAI-shaped
# rate limits (Retry-After ~6s) means up to ~30s wait on the worst call —
# tolerable inside the per-tuple `episode_timeout_s` (default 1800s).
DEFAULT_RATE_LIMIT_MAX_RETRIES = 5


def _request_timeout_s() -> int:
    """Resolve the per-request HTTP timeout. Env override → default."""
    raw = os.environ.get("ANTHROPIC_REQUEST_TIMEOUT_S")
    if raw is None:
        return DEFAULT_REQUEST_TIMEOUT_S
    try:
        v = int(raw)
        if v <= 0:
            raise ValueError
        return v
    except ValueError:
        log.warning(
            "ANTHROPIC_REQUEST_TIMEOUT_S=%r is not a positive int; "
            "using default %ds", raw, DEFAULT_REQUEST_TIMEOUT_S,
        )
        return DEFAULT_REQUEST_TIMEOUT_S


def _rate_limit_max_retries() -> int:
    """Resolve the SDK's internal retry count for rate-limit / 5xx
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


class AnthropicNative:
    """Direct Anthropic API client with cache_control on tools + last message.

    Cache strategy (Anthropic permits up to 4 cache breakpoints per request;
    we use 2):

    - The *last* tool definition in the tools list is marked
      ``cache_control: {type: "ephemeral"}``. Anthropic caches the cumulative
      prefix up through any breakpoint, so marking the last tool
      effectively caches all tool definitions.
    - The last block of the last message is also marked, giving a rolling
      cache breakpoint that walks forward as the conversation grows.

    On a second consecutive call with the same tools + matching message
    prefix, we expect ``usage.cache_read_input_tokens > 0`` and
    `cache_creation_input_tokens` to drop close to zero. The anthropic-cache
    regression test asserts this.
    """

    route = "anthropic_native"

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        *,
        params: dict[str, Any] | None = None,
    ) -> None:
        # Accept "anthropic/claude-sonnet-4-5" or bare "claude-sonnet-4-5";
        # Anthropic API expects the bare form.
        self.model = model.removeprefix("anthropic/")
        self.params = dict(params or {})
        self.client = anthropic.Anthropic(
            api_key=api_key,
            timeout=_request_timeout_s(),
            max_retries=_rate_limit_max_retries(),
        )

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int,
        seed: int | None = None,
        safety_identifier: str | None = None,
    ) -> NormalizedResponse:
        # `seed` is recorded by the orchestrator but Anthropic's API doesn't
        # accept a seed parameter. We deliberately ignore it here; reproducibility
        # at the model level is provider-dependent.
        # `safety_identifier` is OpenAI-only; ignored here.
        del seed, safety_identifier

        cached_tools = self._mark_last_cache(tools)
        cached_messages = self._mark_last_message_cache(messages)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "tools": cached_tools,
            "messages": cached_messages,
            "max_tokens": max_tokens,
        }

        # Per-model param overrides from `ModelSpec.params` (e.g.
        # `thinking: {type: enabled, budget_tokens: 10000}`,
        # `temperature: 1.0`). A None value pops the kwarg, matching
        # the LiteLLMClient convention.
        for k, v in self.params.items():
            if v is None:
                kwargs.pop(k, None)
            else:
                kwargs[k] = v

        resp = self.client.messages.create(**kwargs)

        return self._to_normalized(resp)

    # ------------------------------------------------------------------
    # Helpers (also used by tests)
    # ------------------------------------------------------------------

    @staticmethod
    def _mark_last_cache(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return a copy of `tools` with cache_control on the final entry.

        Anthropic caches the cumulative prefix; one breakpoint at the end of
        the tools list caches all tool definitions plus the system block.
        """
        if not tools:
            return tools
        new = [copy.deepcopy(t) for t in tools]
        new[-1]["cache_control"] = {"type": "ephemeral"}
        return new

    @staticmethod
    def _mark_last_message_cache(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Return a copy of `messages` with cache_control on the last block of
        the last message — the rolling-cache breakpoint.

        Without this, only the static tools prefix is cached; the growing
        messages history pays full input price every turn. With it, turn N
        writes a cache up to messages[-1], and turn N+1 reads that prefix
        back (since N's last message is N+1's second-to-last).
        """
        if not messages:
            return messages
        new = [dict(m) for m in messages]
        last = new[-1]
        content = last.get("content")
        if isinstance(content, str):
            # Wrap string content as a single text block with cache_control.
            last["content"] = [
                {
                    "type": "text",
                    "text": content,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        elif isinstance(content, list) and content:
            new_blocks = [dict(b) for b in content]
            new_blocks[-1] = {**new_blocks[-1], "cache_control": {"type": "ephemeral"}}
            last["content"] = new_blocks
        return new

    @staticmethod
    def _to_normalized(resp: Any) -> NormalizedResponse:
        """Translate an anthropic.types.Message to NormalizedResponse.

        We capture every content block verbatim into `content_blocks` so the
        Anthropic message builder can replay the assistant turn in original
        order on the next request — required for thinking-block signature
        validation when extended thinking + tool use is enabled. `text` and
        `tool_calls` are derived convenience views for the transcript and
        loop control flow.
        """
        text_parts: list[str] = []
        tool_calls: list[NormalizedToolCall] = []
        content_blocks: list[dict[str, Any]] = []

        for block in resp.content:
            content_blocks.append(_block_to_dict(block))

            btype = _block_get(block, "type")
            if btype == "text":
                text_parts.append(_block_get(block, "text") or "")
            elif btype == "tool_use":
                raw_input = _block_get(block, "input")
                tool_calls.append(
                    NormalizedToolCall(
                        id=_block_get(block, "id") or "",
                        name=_block_get(block, "name") or "",
                        # anthropic SDK gives us a parsed dict; defensive copy.
                        arguments=dict(raw_input) if raw_input else {},
                    )
                )
            # thinking / redacted_thinking blocks are kept only in
            # content_blocks (they round-trip but don't surface as `text`).

        usage = resp.usage
        return NormalizedResponse(
            text="\n".join(text_parts) if text_parts else None,
            tool_calls=tuple(tool_calls),
            usage=NormalizedUsage(
                input_tokens=getattr(usage, "input_tokens", 0) or 0,
                output_tokens=getattr(usage, "output_tokens", 0) or 0,
                cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
                cache_creation_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            ),
            stop_reason=resp.stop_reason or "unknown",
            model=resp.model,
            raw=resp.model_dump() if hasattr(resp, "model_dump") else {},
            content_blocks=tuple(content_blocks),
        )


def _block_get(block: Any, key: str, default: Any = None) -> Any:
    """Read a field from an Anthropic content block uniformly.

    Real SDK responses are pydantic models (attr access); test fixtures
    sometimes use dicts. Hide the difference.
    """
    if isinstance(block, dict):
        return block.get(key, default)
    return getattr(block, key, default)


def _block_to_dict(block: Any) -> dict[str, Any]:
    """Serialize an Anthropic content block to a plain dict.

    Round-trip is load-bearing for `thinking` / `redacted_thinking`: the
    `signature` (or `data`) field must reach the next request unchanged or
    Anthropic rejects the message with a signature error.
    """
    if isinstance(block, dict):
        return dict(block)
    if hasattr(block, "model_dump"):
        # Pydantic v2: drop None-valued fields the SDK may inject (e.g.
        # cache_control=None on a tool_use we re-send) so the JSON shape
        # matches what we'd hand-build.
        return {k: v for k, v in block.model_dump().items() if v is not None}
    # Last-resort fallback for unfamiliar block types — copy known fields.
    btype = getattr(block, "type", None)
    out: dict[str, Any] = {"type": btype} if btype else {}
    for attr in ("text", "thinking", "signature", "data", "id", "name", "input"):
        if hasattr(block, attr):
            val = getattr(block, attr)
            if val is not None:
                out[attr] = val
    return out
