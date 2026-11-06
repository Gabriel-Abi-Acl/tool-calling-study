# Configuração só por variável de ambiente. O repositório versiona o exemplo, não o .env.
# O padrão é o provedor falso para a suíte rodar sem chave e sem rede.

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from tool_calling_study.errors import ConfigError

_PROVIDERS = ("fake", "openai", "anthropic")


@dataclass(frozen=True)
class Settings:
    """Recorte de configuração que o runner precisa para escolher o cliente de LLM."""

    llm_provider: str
    openai_api_key: str | None
    anthropic_api_key: str | None
    openai_model: str
    anthropic_model: str
    llm_timeout_s: float


def load_settings(environ: Mapping[str, str] | None = None) -> Settings:
    # Lê o ambiente explícito nos testes. Em produção usa os.environ, sem arquivo de segredo.
    # Provedor desconhecido falha aqui, antes de qualquer import de SDK.
    env = os.environ if environ is None else environ
    provider = env.get("LLM_PROVIDER", "fake").strip().lower()
    if provider not in _PROVIDERS:
        raise ConfigError(f"LLM_PROVIDER inválido: {provider}")
    timeout_raw = env.get("LLM_TIMEOUT_S", "30").strip()
    try:
        timeout_s = float(timeout_raw)
    except ValueError as exc:
        raise ConfigError("LLM_TIMEOUT_S precisa ser numérico") from exc
    if timeout_s <= 0:
        raise ConfigError("LLM_TIMEOUT_S precisa ser positivo")
    return Settings(
        llm_provider=provider,
        openai_api_key=_blank_to_none(env.get("OPENAI_API_KEY")),
        anthropic_api_key=_blank_to_none(env.get("ANTHROPIC_API_KEY")),
        openai_model=env.get("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini",
        anthropic_model=env.get("ANTHROPIC_MODEL", "claude-3-5-haiku-latest").strip()
        or "claude-3-5-haiku-latest",
        llm_timeout_s=timeout_s,
    )


def _blank_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None
