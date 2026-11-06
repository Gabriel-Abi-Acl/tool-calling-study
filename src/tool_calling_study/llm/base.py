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
    """Resposta já normalizada: texto e, quando houver, chamadas estruturadas.

    Token e custo ficam vazios até o cliente informar. O harness não estima preço.
    """

    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    estimated_cost_usd: float | None = None


class LLMClient(Protocol):
    """Contrato mínimo. `tools=None` pede texto; a lista pede function calling nativo."""

    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[ToolSpec] | None = None,
    ) -> LLMResponse:
        """Devolve a próxima resposta do modelo para o histórico informado."""
