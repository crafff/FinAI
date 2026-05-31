"""
Unified configuration for FinAI.

All secrets and environment-specific settings live in one place. Values
are read from environment variables (optionally seeded from a .env file
at the repo root), so that anyone who checks out the code can copy
.env.example to .env, fill in their own keys, and run everything.

Design principles:

    - Nothing is required at import / load time. `load_settings()` always
      succeeds, returning a Settings whose fields may be None. A value is
      only *required* when a code path actually needs it, via the
      `require_*` accessors, which raise a clear MissingConfigError that
      names the missing variable and how to set it.

      This keeps the offline test suite green with no keys configured,
      while giving a precise error the moment a real network/LLM call is
      attempted without its credential.

    - Two backends: "anthropic" (the hosted Claude API) and an
      OpenAI-compatible one. LLM_BACKEND in {local, openai, deepseek,
      vllm, openai-compatible} all route through the OpenAI SDK at
      LOCAL_MODEL_BASE_URL, so the endpoint may be a local vLLM or a
      remote API (OpenAI, DeepSeek). Credentials are validated lazily
      per backend.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_FILE = REPO_ROOT / ".env"

LLMBackend = Literal["anthropic", "local"]

# Accepted LLM_BACKEND values, normalized to the two canonical backends
# above. Anything in OPENAI_COMPATIBLE_BACKENDS routes through the OpenAI
# SDK pointed at LOCAL_MODEL_BASE_URL, so the endpoint may be a local vLLM
# or a remote OpenAI-compatible API (OpenAI, DeepSeek, ...).
ANTHROPIC_BACKENDS = {"anthropic", "claude"}
OPENAI_COMPATIBLE_BACKENDS = {
    "local", "openai", "deepseek", "vllm", "openai-compatible",
}

DEFAULT_ANTHROPIC_MODEL = "claude-opus-4-8"
DEFAULT_LOCAL_MODEL = "llama3.1"
# vLLM's default OpenAI-compatible endpoint.
DEFAULT_LOCAL_BASE_URL = "http://localhost:8000/v1"


class MissingConfigError(RuntimeError):
    """Raised when a required configuration value is absent."""


def _clean(value: str | None) -> str | None:
    """Strip whitespace and treat the empty string as missing (None)."""
    if value is None:
        return None

    value = value.strip()

    return value or None


def _env(name: str) -> str | None:
    return _clean(os.environ.get(name))


def _require(value: str | None, env_name: str, hint: str = "") -> str:
    if value:
        return value

    message = (
        f"Missing required configuration {env_name!r}. "
        f"Copy .env.example to .env and set it, or export {env_name}."
    )

    if hint:
        message += f" {hint}"

    raise MissingConfigError(message)


@dataclass(frozen=True)
class LLMConfig:
    """
    Backbone-model configuration shared by all agents.

    `backend` chooses between the hosted Anthropic API and a local
    OpenAI-compatible server. `model` is the model id to call. The
    credential accessors validate only the fields the chosen backend
    needs.
    """

    backend: LLMBackend
    model: str
    anthropic_api_key: str | None = None
    local_base_url: str | None = None
    local_api_key: str | None = None

    def require_api_key(self) -> str:
        if self.backend == "anthropic":
            return _require(self.anthropic_api_key, "ANTHROPIC_API_KEY")

        return _require(
            self.local_api_key,
            "LOCAL_MODEL_API_KEY",
            "Local OpenAI-compatible servers usually accept any non-empty "
            "value (e.g. 'ollama').",
        )

    def require_base_url(self) -> str:
        if self.backend == "local":
            return _require(self.local_base_url, "LOCAL_MODEL_BASE_URL")

        # The Anthropic SDK uses its own default base URL.
        return "https://api.anthropic.com"


@dataclass(frozen=True)
class Settings:
    """
    Top-level configuration. Data-tool credentials plus the LLM config.

    Fields may be None (unset). Use the `require_*` accessors at the
    point of use to get a validated value or a clear error.
    """

    sec_user_agent: str | None
    finnhub_api_key: str | None
    fmp_api_key: str | None
    reddit_client_id: str | None
    reddit_client_secret: str | None
    reddit_user_agent: str | None
    llm: LLMConfig
    # Base directory for per-run agent artifacts (transcripts / reports).
    runs_dir: str = "runs"

    # -- data-tool accessors -------------------------------------------------

    def require_sec_user_agent(self) -> str:
        return _require(
            self.sec_user_agent,
            "SEC_USER_AGENT",
            "SEC requires a descriptive User-Agent like 'Name email'.",
        )

    def require_finnhub_api_key(self) -> str:
        return _require(self.finnhub_api_key, "FINNHUB_API_KEY")

    def require_fmp_api_key(self) -> str:
        return _require(self.fmp_api_key, "FMP_API_KEY")

    def require_reddit(self) -> tuple[str, str, str]:
        """Return (client_id, client_secret, user_agent) or raise."""
        return (
            _require(self.reddit_client_id, "REDDIT_CLIENT_ID"),
            _require(self.reddit_client_secret, "REDDIT_CLIENT_SECRET"),
            _require(self.reddit_user_agent, "REDDIT_USER_AGENT"),
        )

    # -- bulk pre-flight check ----------------------------------------------

    def missing(self, *services: str) -> list[str]:
        """
        Return the env-var names still unset for the requested services.

        Lets a runner validate everything up front and print one friendly
        message instead of failing midway through a 30-ticker loop.

        Known services: "sec", "finnhub", "fmp", "reddit", "llm".
        """
        checks: dict[str, list[tuple[str, str | None]]] = {
            "sec": [("SEC_USER_AGENT", self.sec_user_agent)],
            "finnhub": [("FINNHUB_API_KEY", self.finnhub_api_key)],
            "fmp": [("FMP_API_KEY", self.fmp_api_key)],
            "reddit": [
                ("REDDIT_CLIENT_ID", self.reddit_client_id),
                ("REDDIT_CLIENT_SECRET", self.reddit_client_secret),
                ("REDDIT_USER_AGENT", self.reddit_user_agent),
            ],
            "llm": self._llm_checks(),
        }

        result: list[str] = []

        for service in services:
            if service not in checks:
                raise ValueError(
                    f"Unknown service {service!r}. "
                    f"Known: {sorted(checks)}."
                )

            for env_name, value in checks[service]:
                if not value:
                    result.append(env_name)

        return result

    def _llm_checks(self) -> list[tuple[str, str | None]]:
        if self.llm.backend == "anthropic":
            return [("ANTHROPIC_API_KEY", self.llm.anthropic_api_key)]

        return [
            ("LOCAL_MODEL_BASE_URL", self.llm.local_base_url),
            ("LOCAL_MODEL_API_KEY", self.llm.local_api_key),
        ]


def _load_dotenv(env_file: Path | str) -> bool:
    """
    Load a .env file into os.environ if python-dotenv is installed and the
    file exists. Real environment variables take precedence (override=False)
    so CI / shell exports always win over the file.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False

    env_path = Path(env_file)

    if not env_path.exists():
        return False

    load_dotenv(env_path, override=False)
    return True


def load_settings(
    env_file: Path | str = DEFAULT_ENV_FILE,
    use_dotenv: bool = True,
) -> Settings:
    """
    Build a Settings from the environment, optionally seeding from a .env
    file first. Always succeeds; missing values surface later via the
    require_* accessors.
    """
    if use_dotenv:
        _load_dotenv(env_file)

    backend_raw = (_env("LLM_BACKEND") or "anthropic").lower()

    if backend_raw in ANTHROPIC_BACKENDS:
        backend: LLMBackend = "anthropic"
    elif backend_raw in OPENAI_COMPATIBLE_BACKENDS:
        backend = "local"
    else:
        allowed = sorted(ANTHROPIC_BACKENDS | OPENAI_COMPATIBLE_BACKENDS)
        raise MissingConfigError(
            f"LLM_BACKEND must be one of {allowed}, got {backend_raw!r}."
        )

    default_model = (
        DEFAULT_ANTHROPIC_MODEL if backend == "anthropic" else DEFAULT_LOCAL_MODEL
    )

    llm = LLMConfig(
        backend=backend,
        model=_env("LLM_MODEL") or default_model,
        anthropic_api_key=_env("ANTHROPIC_API_KEY"),
        local_base_url=_env("LOCAL_MODEL_BASE_URL") or (
            DEFAULT_LOCAL_BASE_URL if backend == "local" else None
        ),
        local_api_key=_env("LOCAL_MODEL_API_KEY"),
    )

    return Settings(
        sec_user_agent=_env("SEC_USER_AGENT"),
        finnhub_api_key=_env("FINNHUB_API_KEY"),
        fmp_api_key=_env("FMP_API_KEY"),
        reddit_client_id=_env("REDDIT_CLIENT_ID"),
        reddit_client_secret=_env("REDDIT_CLIENT_SECRET"),
        reddit_user_agent=_env("REDDIT_USER_AGENT"),
        llm=llm,
        runs_dir=_env("RUNS_DIR") or "runs",
    )
