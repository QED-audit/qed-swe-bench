"""Runtime configuration. Reads env vars with sensible defaults.

Loads `.env` from the repo root (or any parent of CWD) on import via
python-dotenv. Values already set in the environment win over .env entries
(this matches dotenv's default).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env once at import time. Searches for .env starting at CWD and
# walking up; falls back to noop if none is found.
load_dotenv()


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = REPO_ROOT / "data" / "qed_swe_bench.sqlite"
DEFAULT_RUNS_DIR = REPO_ROOT / "runs"
# bench-v8 submodule still hosts the per-bug bugs/ directory used by
# historical-import paths. Per-bug framing reaches the model via the
# container's MCP setup(); no host-side prompt template is needed.
_BENCH_V8 = REPO_ROOT / "benchmarks" / "bench-v8"
DEFAULT_BENCH_V8_BUGS = _BENCH_V8 / "bugs"


@dataclass(frozen=True)
class Config:
    db_path: Path
    runs_dir: Path
    bench_v8_bugs: Path

    # Provider keys (presence-checked by `doctor`; not required to instantiate).
    anthropic_api_key: str | None
    openai_api_key: str | None
    gemini_api_key: str | None
    openrouter_api_key: str | None
    zai_api_key: str | None
    moonshot_api_key: str | None

    # Gateway: if set, all openai/* model-ids route through this URL.
    openai_api_base: str | None

    # LiteLLM-proxy gateway: if set, all litellm_proxy/* model-ids route
    # through this URL. Kept separate from OPENAI_API_BASE so a proxy
    # routing gemini-via-gateway doesn't silently capture direct openai/*
    # calls in the same .env.
    litellm_proxy_api_base: str | None
    litellm_proxy_api_key: str | None

    @classmethod
    def from_env(cls) -> Config:
        return cls(
            db_path=Path(os.environ.get("QED_SWE_BENCH_DB", str(DEFAULT_DB_PATH))),
            runs_dir=Path(os.environ.get("QED_SWE_BENCH_RUNS_DIR", str(DEFAULT_RUNS_DIR))),
            bench_v8_bugs=Path(
                os.environ.get("BENCH_V8_BUGS_PATH", str(DEFAULT_BENCH_V8_BUGS))
            ),
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            gemini_api_key=os.environ.get("GEMINI_API_KEY"),
            openrouter_api_key=os.environ.get("OPENROUTER_API_KEY"),
            zai_api_key=os.environ.get("ZAI_API_KEY"),
            moonshot_api_key=os.environ.get("MOONSHOT_API_KEY"),
            openai_api_base=os.environ.get("OPENAI_API_BASE"),
            litellm_proxy_api_base=os.environ.get("LITELLM_PROXY_API_BASE"),
            litellm_proxy_api_key=os.environ.get("LITELLM_PROXY_API_KEY"),
        )

    def provider_key_for(self, model_id: str) -> str | None:
        """Return the API key relevant for a model_id prefix, or None if not set."""
        if model_id.startswith("anthropic/"):
            return self.anthropic_api_key
        if model_id.startswith("openai/"):
            return self.openai_api_key
        if model_id.startswith("litellm_proxy/"):
            return self.litellm_proxy_api_key
        if model_id.startswith("gemini/"):
            return self.gemini_api_key
        if model_id.startswith("openrouter/"):
            return self.openrouter_api_key
        if model_id.startswith("zai/"):
            return self.zai_api_key
        if model_id.startswith("moonshot/"):
            return self.moonshot_api_key
        return None
