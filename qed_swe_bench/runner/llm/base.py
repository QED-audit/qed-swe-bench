"""Provider-agnostic response shapes for the agent loop.

The agent loop talks to AnthropicNative / LiteLLMClient / MockClient through
this contract. Provider-specific formatting (Anthropic blocks, OpenAI tool
schema, etc.) lives in the individual clients; the loop only sees these
NormalizedX types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class ModelMismatchError(RuntimeError):
    """Raised when a provider returns a response from a different model than the
    one we requested. Used by the loop to abort an episode immediately rather
    than burn tokens on a silently-rerouted (downgraded) model — for example,
    OpenAI's documented routing of high-risk cyber traffic from gpt-5.5 to
    gpt-5.2.
    """

    def __init__(self, requested: str, served: str) -> None:
        self.requested = requested
        self.served = served
        super().__init__(
            f"served model {served!r} does not match requested {requested!r}"
        )


def _bare_model_name(model_id: str) -> str:
    """Strip the provider prefix from a model id.

    `openai/gpt-5.5` → `gpt-5.5`
    `anthropic/claude-haiku-4-5` → `claude-haiku-4-5`
    `openrouter/openai/gpt-5.5` → `gpt-5.5` (strip the deepest prefix)
    `gpt-5.5` → `gpt-5.5` (no prefix)
    """
    return model_id.rsplit("/", 1)[-1]


def served_matches_requested(*, requested: str, served: str) -> bool:
    """Return True if the provider's `served` model is consistent with what we
    `requested`. Provider-agnostic.

    Both ids are reduced to their bare model name (provider prefix stripped).
    A served id matches when it equals the requested id or extends it on a
    hyphen boundary — providers commonly return a dated snapshot
    (`gpt-5.5` → `gpt-5.5-2026-04-23`, `claude-haiku-4-5` →
    `claude-haiku-4-5-20250101`). The hyphen requirement prevents false-
    positive matches like `gpt-5` ↔ `gpt-5.5` (different model families that
    happen to share a string prefix).

    An empty `served` string is treated as a match — some test stubs and
    older gateway responses don't echo the model field, and we don't want
    to flag those absences as downgrades.
    """
    if not served:
        return True
    bare_req = _bare_model_name(requested)
    bare_served = _bare_model_name(served)
    return bare_served == bare_req or bare_served.startswith(bare_req + "-")


@dataclass(frozen=True)
class NormalizedToolCall:
    """A model's request to invoke a tool.

    `id` is the provider's tool-use identifier; the loop must echo it back
    when delivering the tool result so providers can correlate results to
    calls (Anthropic uses tool_use_id; OpenAI uses tool_call_id).
    """

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class NormalizedUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    # OpenAI reasoning models (gpt-5, gpt-5.5, o-series) bill visible
    # reasoning tokens as part of completion_tokens but break them out under
    # `completion_tokens_details.reasoning_tokens`. We surface this so the
    # transcript can prove `reasoning_effort` actually fired (a turn at xhigh
    # with reasoning_tokens=0 means the request was silently downgraded).
    # Anthropic uses thinking blocks instead and leaves this at 0.
    reasoning_tokens: int = 0


@dataclass(frozen=True)
class NormalizedResponse:
    """One model turn's worth of output."""

    text: str | None
    tool_calls: tuple[NormalizedToolCall, ...]
    usage: NormalizedUsage
    stop_reason: str
    model: str  # model id as reported by the provider (may differ from request)
    raw: dict[str, Any] = field(default_factory=dict)  # untouched provider response
    # Anthropic-only: every content block from `resp.content` captured verbatim
    # in original order (text, thinking, redacted_thinking, tool_use). When
    # extended thinking + tool use is enabled the model interleaves thinking
    # and tool_use blocks, and Anthropic validates each thinking block's
    # signature against its position in the next turn's assistant message —
    # any reordering 400s the request. Replaying this list unmodified is the
    # only safe option. Empty tuple for non-Anthropic providers and for
    # Anthropic calls without thinking enabled (the loop falls back to
    # rebuilding content from `text` + `tool_calls`).
    content_blocks: tuple[dict[str, Any], ...] = ()
    # Moonshot-style reasoning trace, captured verbatim. Moonshot's API
    # requires this field be replayed in subsequent assistant messages
    # or the model loses access to its prior thinking. Other LiteLLM-routed
    # providers leave this None (their reasoning is either server-side
    # only — gpt-5, Gemini, GLM — or surfaced via Anthropic-style
    # content_blocks instead). See messages.py: only included in
    # outgoing messages when the model is moonshot/* to avoid 400ing
    # providers that reject unknown fields.
    reasoning_content: str | None = None
    # Provider-side response identifier, captured for post-hoc cost
    # reconciliation. For OpenRouter routes this is the OR
    # `gen-XXX...` id which can be looked up via
    # GET /api/v1/generation?id=<id> to retrieve the authoritative
    # billed cost (incl. cache_discount, upstream_inference_cost) —
    # see runner/openrouter_recon.py. For other providers the id is
    # provider-specific (Anthropic's message_id, OpenAI's chat
    # completion id) and is not currently consumed downstream;
    # captured anyway so the transcript carries provenance.
    provider_response_id: str | None = None


class LLMClient(Protocol):
    """Contract every model client implements."""

    model: str  # model id, e.g. "claude-sonnet-4-5" (provider-specific)
    route: str  # 'anthropic_native' | 'litellm' | 'litellm_gateway' | 'mock'

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int,
        seed: int | None = None,
        safety_identifier: str | None = None,
    ) -> NormalizedResponse:
        """Single non-streaming completion.

        `messages` is a list of provider-format messages. The loop is
        responsible for keeping these in the format the client expects (the
        loop and client are paired through `tools.py`). The first message
        carries all framing (init prompt + optional hint); there is no
        separate system block — the container's MCP setup() supplies all
        bug-specific context.

        `tools` is the list of tool definitions in the provider's expected
        format (Anthropic schema vs OpenAI tools schema). The loop converts
        from MCP via `runner/llm/tools.py`.

        `safety_identifier` is forwarded to providers that support per-user
        safety scoping (OpenAI). It limits the blast radius of a cyber-policy
        revocation to a single identifier instead of the whole org. Clients
        that don't support it ignore the kwarg.
        """
        ...
