import pytest

from settings import (
    LLMConfig,
    MissingConfigError,
    Settings,
    load_settings,
)


# Every test loads with use_dotenv=False so a developer's real .env never
# leaks into the assertions; the environment is set explicitly via
# monkeypatch.


ALL_VARS = [
    "SEC_USER_AGENT", "FMP_API_KEY", "FINNHUB_API_KEY",
    "REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "REDDIT_USER_AGENT",
    "LLM_BACKEND", "LLM_MODEL", "ANTHROPIC_API_KEY",
    "LOCAL_MODEL_BASE_URL", "LOCAL_MODEL_API_KEY",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ALL_VARS:
        monkeypatch.delenv(var, raising=False)


def test_load_settings_succeeds_with_empty_env():
    # Nothing required at load time -> always succeeds.
    settings = load_settings(use_dotenv=False)

    assert settings.finnhub_api_key is None
    assert settings.llm.backend == "anthropic"
    assert settings.llm.model == "claude-opus-4-8"


def test_require_returns_value_when_set(monkeypatch):
    monkeypatch.setenv("FINNHUB_API_KEY", "fh_key")
    monkeypatch.setenv("FMP_API_KEY", "fmp_key")

    settings = load_settings(use_dotenv=False)

    assert settings.require_finnhub_api_key() == "fh_key"
    assert settings.require_fmp_api_key() == "fmp_key"


def test_require_raises_when_unset():
    settings = load_settings(use_dotenv=False)

    with pytest.raises(MissingConfigError) as excinfo:
        settings.require_fmp_api_key()

    # Error names the missing variable.
    assert "FMP_API_KEY" in str(excinfo.value)


def test_empty_and_whitespace_treated_as_missing(monkeypatch):
    monkeypatch.setenv("FINNHUB_API_KEY", "   ")

    settings = load_settings(use_dotenv=False)

    assert settings.finnhub_api_key is None

    with pytest.raises(MissingConfigError):
        settings.require_finnhub_api_key()


def test_require_reddit_returns_triple(monkeypatch):
    monkeypatch.setenv("REDDIT_CLIENT_ID", "id")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "secret")
    monkeypatch.setenv("REDDIT_USER_AGENT", "ua")

    settings = load_settings(use_dotenv=False)

    assert settings.require_reddit() == ("id", "secret", "ua")


def test_require_reddit_raises_on_partial(monkeypatch):
    monkeypatch.setenv("REDDIT_CLIENT_ID", "id")
    # secret + user agent missing

    settings = load_settings(use_dotenv=False)

    with pytest.raises(MissingConfigError):
        settings.require_reddit()


def test_missing_reports_unset_vars(monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "fmp_key")

    settings = load_settings(use_dotenv=False)

    assert settings.missing("fmp") == []
    assert set(settings.missing("reddit")) == {
        "REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "REDDIT_USER_AGENT",
    }


def test_missing_unknown_service_raises():
    settings = load_settings(use_dotenv=False)

    with pytest.raises(ValueError):
        settings.missing("does-not-exist")


def test_llm_anthropic_backend(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-123")

    settings = load_settings(use_dotenv=False)

    assert settings.llm.backend == "anthropic"
    assert settings.llm.require_api_key() == "sk-ant-123"
    assert settings.llm.require_base_url() == "https://api.anthropic.com"
    assert settings.missing("llm") == []


def test_llm_local_backend_defaults(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "local")
    monkeypatch.setenv("LOCAL_MODEL_API_KEY", "ollama")

    settings = load_settings(use_dotenv=False)

    assert settings.llm.backend == "local"
    assert settings.llm.model == "llama3.1"
    # base URL defaults to the vLLM endpoint when backend is local.
    assert settings.llm.require_base_url() == "http://localhost:8000/v1"
    assert settings.llm.require_api_key() == "ollama"


def test_llm_local_backend_missing_key_raises(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "local")

    settings = load_settings(use_dotenv=False)

    with pytest.raises(MissingConfigError):
        settings.llm.require_api_key()

    assert "LOCAL_MODEL_API_KEY" in settings.missing("llm")


def test_invalid_backend_raises(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "gemini")

    with pytest.raises(MissingConfigError):
        load_settings(use_dotenv=False)


def test_openai_compatible_backend_aliases(monkeypatch):
    # openai / deepseek / vllm are convenience aliases that all route
    # through the OpenAI-compatible ("local") path.
    for alias in ("openai", "deepseek", "vllm", "local", "OpenAI"):
        monkeypatch.setenv("LLM_BACKEND", alias)

        settings = load_settings(use_dotenv=False)

        assert settings.llm.backend == "local"


def test_openai_alias_with_remote_endpoint(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "deepseek")
    monkeypatch.setenv("LLM_MODEL", "deepseek-chat")
    monkeypatch.setenv("LOCAL_MODEL_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setenv("LOCAL_MODEL_API_KEY", "sk-deepseek")

    settings = load_settings(use_dotenv=False)

    assert settings.llm.model == "deepseek-chat"
    assert settings.llm.require_base_url() == "https://api.deepseek.com/v1"
    assert settings.llm.require_api_key() == "sk-deepseek"


def test_llm_model_override(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "claude-sonnet-4-6")

    settings = load_settings(use_dotenv=False)

    assert settings.llm.model == "claude-sonnet-4-6"


def test_llm_config_is_constructible_directly():
    # Agents can also build an LLMConfig without the env, e.g. in tests.
    cfg = LLMConfig(backend="anthropic", model="m", anthropic_api_key="k")

    assert cfg.require_api_key() == "k"
    assert isinstance(load_settings(use_dotenv=False), Settings)
