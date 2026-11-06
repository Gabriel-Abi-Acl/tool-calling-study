# Configuração por ambiente e import tardio dos SDKs. Nenhum teste chama a API.
# A conversão de mensagens fica coberta com objetos simples, sem o pacote do provedor.

import sys
from types import SimpleNamespace

import pytest

from tool_calling_study.config import Settings, load_settings
from tool_calling_study.errors import (
    ConfigError,
    MissingCredential,
    ParseError,
    ProviderNotInstalled,
)
from tool_calling_study.llm.base import LLMMessage
from tool_calling_study.llm.factory import build_llm
from tool_calling_study.llm.fake import FakeLLM, FakeLLMExhausted
from tool_calling_study.llm.providers import (
    _anthropic_response_payload as anthropic_response_payload,
)
from tool_calling_study.llm.providers import (
    _openai_message_payload as openai_message_payload,
)
from tool_calling_study.llm.providers import (
    to_anthropic_messages,
    to_openai_messages,
)
from tool_calling_study.strategies.native import parse_native_payload
from tool_calling_study.tools.schema import ToolCall

_ENV = {
    "LLM_PROVIDER": "fake",
    "OPENAI_API_KEY": "",
    "ANTHROPIC_API_KEY": "",
    "OPENAI_MODEL": "gpt-4o-mini",
    "ANTHROPIC_MODEL": "claude-3-5-haiku-latest",
    "LLM_TIMEOUT_S": "30",
}


def test_load_settings_defaults_to_fake_provider():
    settings = load_settings(_ENV)
    assert settings.llm_provider == "fake"
    assert settings.openai_api_key is None
    assert settings.anthropic_api_key is None
    assert settings.llm_timeout_s == 30


def test_load_settings_rejects_unknown_provider_and_bad_timeout():
    with pytest.raises(ConfigError):
        load_settings({**_ENV, "LLM_PROVIDER": "local-http"})
    with pytest.raises(ConfigError):
        load_settings({**_ENV, "LLM_TIMEOUT_S": "0"})


def test_fake_client_refuses_to_invent_a_completion():
    client = build_llm(load_settings(_ENV))
    assert isinstance(client, FakeLLM)
    with pytest.raises(FakeLLMExhausted):
        client.complete([LLMMessage(role="user", content="oi")])


def test_openai_client_raises_before_import_when_the_key_is_missing():
    sys.modules.pop("openai", None)
    settings = _settings(provider="openai", openai_api_key=None)
    from tool_calling_study.llm.providers import OpenAIClient

    with pytest.raises(MissingCredential):
        OpenAIClient(settings).complete([LLMMessage(role="user", content="oi")])
    assert "openai" not in sys.modules


def test_openai_sdk_is_imported_on_the_first_live_call(monkeypatch):
    sys.modules.pop("openai", None)
    monkeypatch.setitem(sys.modules, "openai", None)
    from tool_calling_study.llm.providers import OpenAIClient

    client = OpenAIClient(_settings(provider="openai", openai_api_key="test-key"))
    with pytest.raises(ProviderNotInstalled):
        client.complete([LLMMessage(role="user", content="oi")])


def test_anthropic_sdk_is_imported_on_the_first_live_call(monkeypatch):
    sys.modules.pop("anthropic", None)
    monkeypatch.setitem(sys.modules, "anthropic", None)
    from tool_calling_study.llm.providers import AnthropicClient

    client = AnthropicClient(_settings(provider="anthropic", anthropic_api_key="test-key"))
    with pytest.raises(ProviderNotInstalled):
        client.complete([LLMMessage(role="user", content="oi")])


def test_building_a_live_client_does_not_import_the_sdk():
    sys.modules.pop("openai", None)
    sys.modules.pop("anthropic", None)
    build_llm(_settings(provider="openai", openai_api_key="test-key"))
    build_llm(_settings(provider="anthropic", anthropic_api_key="test-key"))
    assert "openai" not in sys.modules
    assert "anthropic" not in sys.modules


def test_openai_message_projection_and_response_parse():
    call = ToolCall("lookup", {"query": "capital of France"}, call_id="call_1")
    messages = [
        LLMMessage(role="user", content="Qual é a capital da França?"),
        LLMMessage(role="assistant", content="", tool_calls=(call,)),
        LLMMessage(role="tool", content="Paris", tool_call_id="call_1"),
    ]
    payload = to_openai_messages(messages)
    assert payload[1]["content"] is None
    assert payload[1]["tool_calls"][0]["function"]["name"] == "lookup"
    assert payload[2]["role"] == "tool"
    message = SimpleNamespace(
        content=None,
        tool_calls=[
            SimpleNamespace(
                id="call_1",
                function=SimpleNamespace(name="lookup", arguments='{"query": "capital of France"}'),
            )
        ],
    )
    parsed = parse_native_payload(openai_message_payload(message))
    assert parsed.tool_calls[0].name == "lookup"
    assert dict(parsed.tool_calls[0].arguments) == {"query": "capital of France"}


def test_anthropic_message_projection_groups_tool_results():
    call = ToolCall("calculator", {"expression": "17 * 23"}, call_id="toolu_1")
    system, messages = to_anthropic_messages(
        [
            LLMMessage(role="system", content="seja literal"),
            LLMMessage(role="user", content="Quanto é 17 * 23?"),
            LLMMessage(role="assistant", content="", tool_calls=(call,)),
            LLMMessage(role="tool", content="391", tool_call_id="toolu_1"),
        ]
    )
    assert system == "seja literal"
    assert messages[1]["content"][0]["type"] == "tool_use"
    assert messages[2]["content"][0]["type"] == "tool_result"
    assert messages[2]["content"][0]["tool_use_id"] == "toolu_1"
    response = SimpleNamespace(
        content=[
            SimpleNamespace(
                type="tool_use",
                id="toolu_1",
                name="calculator",
                input={"expression": "1+1"},
            ),
            SimpleNamespace(type="text", text=""),
        ]
    )
    parsed = parse_native_payload(anthropic_response_payload(response))
    assert parsed.tool_calls[0].name == "calculator"
    assert dict(parsed.tool_calls[0].arguments) == {"expression": "1+1"}


def test_provider_projection_requires_tool_call_id():
    with pytest.raises(ParseError):
        to_openai_messages([LLMMessage(role="tool", content="391")])


def _settings(
    *,
    provider: str,
    openai_api_key: str | None = None,
    anthropic_api_key: str | None = None,
) -> Settings:
    return Settings(
        llm_provider=provider,
        openai_api_key=openai_api_key,
        anthropic_api_key=anthropic_api_key,
        openai_model="gpt-4o-mini",
        anthropic_model="claude-3-5-haiku-latest",
        llm_timeout_s=30,
    )
