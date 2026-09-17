"""LLM client factory routes by prefix and respects gateway env."""

from __future__ import annotations

from qed_swe_bench.config import Config
from qed_swe_bench.runner.llm.anthropic_native import AnthropicNative
from qed_swe_bench.runner.llm.factory import build_client
from qed_swe_bench.runner.llm.litellm_client import LiteLLMClient
from qed_swe_bench.runner.llm.mock import MockClient


def _empty_config() -> Config:
    return Config(
        db_path=__import__("pathlib").Path("/tmp/x.db"),
        runs_dir=__import__("pathlib").Path("/tmp/runs"),
        bench_v8_bugs=__import__("pathlib").Path("/tmp/bugs"),
        anthropic_api_key="sk-ant-x",
        openai_api_key="sk-x",
        gemini_api_key="g-x",
        openrouter_api_key="r-x",
        zai_api_key="z-x",
        moonshot_api_key="m-x",
        openai_api_base=None,
        litellm_proxy_api_base=None,
        litellm_proxy_api_key=None,
    )


def test_mock_overrides_routing() -> None:
    client = build_client("anthropic/claude-sonnet-4-5", mock=True)
    assert isinstance(client, MockClient)
    assert client.route == "mock"


def test_anthropic_routes_to_native() -> None:
    cfg = _empty_config()
    client = build_client("anthropic/claude-sonnet-4-5", config=cfg)
    assert isinstance(client, AnthropicNative)
    assert client.route == "anthropic_native"
    # Prefix is stripped from the API-side model id.
    assert client.model == "claude-sonnet-4-5"


def test_anthropic_params_plumbed_to_native() -> None:
    """Per-model `params:` from the YAML must reach AnthropicNative so
    extended thinking / temperature / etc. can be configured."""
    cfg = _empty_config()
    params = {"thinking": {"type": "enabled", "budget_tokens": 8000}}
    client = build_client("anthropic/claude-opus-4-6", config=cfg, params=params)
    assert isinstance(client, AnthropicNative)
    assert client.params == params


def test_openai_routes_to_litellm_no_gateway() -> None:
    cfg = _empty_config()
    client = build_client("openai/gpt-5", config=cfg)
    assert isinstance(client, LiteLLMClient)
    assert client.route == "litellm"
    assert client.api_base is None
    assert client.api_key == "sk-x"


def test_openai_with_gateway_routes_to_litellm_gateway() -> None:
    cfg = Config(
        **{**_empty_config().__dict__, "openai_api_base": "https://gw.example.com/v1"}
    )
    client = build_client("openai/llama-3.3-70b", config=cfg)
    assert isinstance(client, LiteLLMClient)
    assert client.route == "litellm_gateway"
    assert client.api_base == "https://gw.example.com/v1"


def test_litellm_proxy_routes_through_proxy_env_not_openai() -> None:
    """`litellm_proxy/<model>` uses LITELLM_PROXY_API_BASE/_KEY, not the
    OPENAI_* pair, so a gateway routing third-party models doesn't
    silently capture direct openai/* calls in the same .env."""
    cfg = Config(
        **{
            **_empty_config().__dict__,
            "openai_api_base": "https://api.openai.com/v1",  # untouched
            "litellm_proxy_api_base": "https://gateway.example.com",
            "litellm_proxy_api_key": "sk-proxy",
        }
    )
    client = build_client(
        "litellm_proxy/gemini/gemini-3.1-pro-preview", config=cfg,
    )
    assert isinstance(client, LiteLLMClient)
    assert client.api_base == "https://gateway.example.com"
    assert client.api_key == "sk-proxy"
    assert client.route == "litellm_gateway"

    # Direct openai/* call from the same cfg goes to OpenAI's base, not the proxy.
    direct = build_client("openai/gpt-5.5", config=cfg)
    assert direct.api_base == "https://api.openai.com/v1"
    assert direct.api_key == "sk-x"


def test_gemini_routes_to_litellm() -> None:
    cfg = _empty_config()
    client = build_client("gemini/gemini-2.5-pro", config=cfg)
    assert isinstance(client, LiteLLMClient)
    assert client.api_base is None
    assert client.api_key == "g-x"


def test_openrouter_routes_to_litellm() -> None:
    cfg = _empty_config()
    client = build_client("openrouter/z-ai/glm-4.6", config=cfg)
    assert isinstance(client, LiteLLMClient)
    assert client.api_key == "r-x"


def test_zai_routes_to_litellm() -> None:
    """Direct z.ai via LiteLLM's native `zai/` provider — api_base resolved
    inside LiteLLM (not overridden here), key plumbed through."""
    cfg = _empty_config()
    client = build_client("zai/glm-5.1", config=cfg)
    assert isinstance(client, LiteLLMClient)
    assert client.api_base is None
    assert client.api_key == "z-x"


def test_moonshot_routes_to_litellm() -> None:
    """Direct Moonshot via LiteLLM's native `moonshot/` provider for the
    Kimi family. Resolves to api.moonshot.ai/v1 internally; we just plumb
    the key. Bypasses OR's multi-upstream routing that broke caching."""
    cfg = _empty_config()
    client = build_client("moonshot/kimi-k2.6", config=cfg)
    assert isinstance(client, LiteLLMClient)
    assert client.api_base is None
    assert client.api_key == "m-x"


def test_unknown_prefix_falls_through_to_litellm() -> None:
    cfg = _empty_config()
    client = build_client("bedrock/anthropic.claude-3", config=cfg)
    assert isinstance(client, LiteLLMClient)
    assert client.api_key is None  # let litellm read its own env vars
