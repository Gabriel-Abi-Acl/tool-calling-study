# Mensagens e cliente no formato interno do estudo.
# Provedores traduzem este formato na borda; o restante do código não vê o SDK.

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from tool_calling_study.tools.schema import ToolCall, ToolSpec


@dataclass(frozen=True)
class LLMMessage:
    """Um item do histórico no formato comum às três estratégias."""

    role: str
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    name: str | None = None
    tool_call_id: str | None = None


@dataclass(frozen=True)
class LLMResponse:
    """Resposta já normalizada: texto e, quando houver, chamadas estruturadas."""

    content: str
    tool_calls: tuple[ToolCall, ...] = ()


class LLMClient(Protocol):
    """Contrato mínimo. `tools=None` pede texto; a lista pede function calling nativo."""

    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[ToolSpec] | None = None,
    ) -> LLMResponse:
        """Devolve a próxima resposta do modelo para o histórico informado."""
