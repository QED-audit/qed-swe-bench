"""Build an LLMClient for a given model_id.

Routing rules:
  mock          → MockClient (canned responses, --mock-llm)
  anthropic/*   → AnthropicNative (native SDK + cache_control)
  everything    → LiteLLMClient (litellm.completion)
                  - openai/* picks up OPENAI_API_BASE env var transparently
                    so OSS-via-gateway is the primary multi-model path.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from qed_swe_bench.config import Config
from qed_swe_bench.runner.llm.anthropic_native import AnthropicNative
from qed_swe_bench.runner.llm.base import LLMClient
from qed_swe_bench.runner.llm.litellm_client import LiteLLMClient
from qed_swe_bench.runner.llm.mock import MockClient


def build_client(
    model_id: str,
    *,
    mock: bool = False,
    config: Config | None = None,
    params: Mapping[str, Any] | None = None,
) -> LLMClient:
    from swebench_runner.trajectory import record_client

    return record_client(
        _build_client(model_id, mock=mock, config=config, params=params),
        model_id=model_id,
    )


def _build_client(
    model_id: str,
    *,
    mock: bool = False,
    config: Config | None = None,
    params: Mapping[str, Any] | None = None,
) -> LLMClient:
    """Return the right client for `model_id`.

    `mock=True` overrides routing entirely and returns a MockClient.
    `params` is a per-model dict of provider completion overrides
    (e.g. `{"reasoning_effort": "xhigh"}` for OpenAI gpt-5*).
    """
    if mock:
        return MockClient(model=model_id)

    cfg = config or Config.from_env()
    params = dict(params or {})

    if model_id.startswith("litellm_proxy/oauth-"):
        from swebench_runner.oauth import build_qed_oauth_client
        return build_qed_oauth_client(
            model_id, api_base=cfg.litellm_proxy_api_base,
            api_key=cfg.litellm_proxy_api_key, params=params,
        )

    if model_id.startswith("anthropic/"):
        return AnthropicNative(
            model=model_id, api_key=cfg.anthropic_api_key, params=params,
        )

    # Provider-key + gateway routing for LiteLLM.
    api_base: str | None = None
    api_key: str | None = None

    if model_id.startswith("openai/"):
        api_key = cfg.openai_api_key
        # OPENAI_API_BASE is the single switch for all openai/* model-ids.
        # Set this when running against an OpenAI-compatible gateway.
        api_base = cfg.openai_api_base
    elif model_id.startswith("litellm_proxy/"):
        # LiteLLM's `litellm_proxy/<model>` prefix is for routing through
        # an external LiteLLM proxy that preserves the rest of the model
        # name verbatim in the request body. Use case: a gateway whose
        # model_list/whitelist is keyed by the upstream's full model name
        # (e.g. `gemini/gemini-3.1-pro-preview`) and that applies its own
        # provider-specific param mappings (e.g. reasoning_effort →
        # thinkingConfig). Kept separate from OPENAI_API_BASE so a proxy
        # routing third-party models doesn't silently capture direct
        # openai/* calls from the same .env.
        api_key = cfg.litellm_proxy_api_key
        api_base = cfg.litellm_proxy_api_base
    elif model_id.startswith("gemini/"):
        api_key = cfg.gemini_api_key
    elif model_id.startswith("openrouter/"):
        api_key = cfg.openrouter_api_key
    elif model_id.startswith("zai/"):
        # LiteLLM has a built-in `zai/` provider (PR #17307) that resolves
        # api_base internally to https://api.z.ai/api/paas/v4. We just need
        # to plumb the key. Direct path preferred over `openrouter/z-ai/*`
        # for the steeper cache-hit discount (81% vs OR's 50%).
        api_key = cfg.zai_api_key
    elif model_id.startswith("moonshot/"):
        # LiteLLM has a built-in `moonshot/` provider that resolves api_base
        # to https://api.moonshot.ai/v1. Direct path required for prompt
        # caching: openrouter/moonshotai/* shows 0% cache hit empirically
        # (multi-provider routing on OR breaks Moonshot's prefix cache;
        # documented Moonshot quirk + OR provider switching). Direct gets
        # the 75% cache discount ($0.15/M cached vs $0.60/M base).
        api_key = cfg.moonshot_api_key
    # else: let LiteLLM pick from its built-in env-var conventions
    # (e.g. AZURE_API_KEY, BEDROCK_*, etc.).

    return LiteLLMClient(
        model=model_id, api_base=api_base, api_key=api_key, params=params,
    )
