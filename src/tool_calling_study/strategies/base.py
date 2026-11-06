# Decisão comum às três estratégias: ou há chamadas para o loop executar, ou há resposta final.
# O histórico devolvido ao modelo é por estratégia; o loop só guarda o que já foi executado.

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from tool_calling_study.errors import ConfigError
from tool_calling_study.llm.base import LLMClient
from tool_calling_study.tools.schema import ToolCall, ToolSpec


@dataclass(frozen=True)
class ExecutedCall:
    """Uma chamada que o loop já rodou, com o texto devolvido pela ferramenta."""

    call: ToolCall
    output: str
    ok: bool


@dataclass(frozen=True)
class PriorTurn:
    """Turno anterior. A estratégia reapresenta isso no formato do protocolo dela."""

    assistant_text: str
    calls: tuple[ExecutedCall, ...]


@dataclass(frozen=True)
class Decision:
    """Saída de um turno. Chamadas e resposta final não vêm juntas."""

    tool_calls: tuple[ToolCall, ...]
    final_answer: str | None
    assistant_text: str


class ToolStrategy(Protocol):
    """Uma forma de pedir e ler a próxima ação do modelo."""

    name: str

    def decide(
        self,
        task_prompt: str,
        turns: Sequence[PriorTurn],
        tools: Sequence[ToolSpec],
    ) -> Decision:
        """Pede a próxima decisão. Pode levantar `ParseError` se o formato vier quebrado."""


def build_strategy(name: str, llm: LLMClient) -> ToolStrategy:
    """Instancia a estratégia pelo nome estável usado no runner."""
    from tool_calling_study.strategies.json_plan import JsonPlanStrategy
    from tool_calling_study.strategies.native import NativeToolStrategy
    from tool_calling_study.strategies.react import ReactToolStrategy

    strategies = {
        "react": ReactToolStrategy,
        "native": NativeToolStrategy,
        "json_plan": JsonPlanStrategy,
    }
    try:
        selected = strategies[name]
    except KeyError as exc:
        raise ConfigError(f"estratégia desconhecida: {name}") from exc
    return selected(llm)


def render_tool_catalog(tools: Sequence[ToolSpec]) -> str:
    return "\n".join(tool.render_line() for tool in tools)
