# Escolhe o cliente a partir da configuração. SDKs reais não são importados aqui.
# O import do provedor fica dentro do `complete` da classe correspondente.

from __future__ import annotations

from tool_calling_study.config import Settings, load_settings
from tool_calling_study.llm.base import LLMClient
from tool_calling_study.llm.fake import FakeLLM


def build_llm(settings: Settings | None = None) -> LLMClient:
    """Monta o cliente do provedor configurado. `fake` devolve um roteiro vazio."""
    # Import local: o módulo dos provedores puxa o parser nativo, e o SDK continua fora daqui.
    # Assim `import tool_calling_study` não toca em openai nem em anthropic.
    resolved = load_settings() if settings is None else settings
    if resolved.llm_provider == "fake":
        return FakeLLM()
    from tool_calling_study.llm.providers import AnthropicClient, OpenAIClient

    if resolved.llm_provider == "openai":
        return OpenAIClient(resolved)
    return AnthropicClient(resolved)
