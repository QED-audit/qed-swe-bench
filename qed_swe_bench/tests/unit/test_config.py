"""Config env-var roundtrips."""

from __future__ import annotations

import pytest

from qed_swe_bench.config import Config


def test_from_env_provider_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-xxx")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-xxx")
    monkeypatch.setenv("ZAI_API_KEY", "zai-xxx")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    cfg = Config.from_env()
    assert cfg.anthropic_api_key == "sk-ant-xxx"
    assert cfg.openai_api_key == "sk-xxx"
    assert cfg.zai_api_key == "zai-xxx"
    assert cfg.gemini_api_key is None


def test_provider_key_for_routes_by_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    monkeypatch.setenv("OPENAI_API_KEY", "o")
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("OPENROUTER_API_KEY", "r")
    monkeypatch.setenv("ZAI_API_KEY", "z")
    monkeypatch.setenv("MOONSHOT_API_KEY", "m")
    cfg = Config.from_env()
    assert cfg.provider_key_for("anthropic/claude-sonnet-4-5") == "a"
    assert cfg.provider_key_for("openai/gpt-5") == "o"
    assert cfg.provider_key_for("gemini/gemini-2.5-pro") == "g"
    assert cfg.provider_key_for("openrouter/z-ai/glm-4.6") == "r"
    assert cfg.provider_key_for("zai/glm-5.1") == "z"
    assert cfg.provider_key_for("moonshot/kimi-k2.6") == "m"
    assert cfg.provider_key_for("unknown/model") is None


def test_openai_api_base_optional(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    cfg = Config.from_env()
    assert cfg.openai_api_base is None

    monkeypatch.setenv("OPENAI_API_BASE", "https://gw.example.com/v1")
    cfg = Config.from_env()
    assert cfg.openai_api_base == "https://gw.example.com/v1"
