# Cliente falso com roteiro. Testes e o CI passam sem chave e sem rede.
# Sem resposta scriptada o cliente falha: ele não inventa completion.

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from tool_calling_study.errors import StudyError
from tool_calling_study.llm.base import LLMMessage, LLMResponse
from tool_calling_study.tools.schema import ToolSpec


class FakeLLMExhausted(StudyError):
    """O roteiro acabou. Quem chama precisa fornecer a próxima resposta."""


@dataclass(frozen=True)
class RecordedLLMCall:
    """O que a estratégia enviou. Testes conferem se houve schema nativo ou só texto."""

    messages: tuple[LLMMessage, ...]
    tool_names: tuple[str, ...] | None


class FakeLLM:
    """Entrega `LLMResponse` na ordem do roteiro e guarda cada chamada."""

    def __init__(self, responses: Sequence[LLMResponse] = ()) -> None:
        self._pending = list(responses)
        self.calls: list[RecordedLLMCall] = []

    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[ToolSpec] | None = None,
    ) -> LLMResponse:
        tool_names = None if tools is None else tuple(tool.name for tool in tools)
        self.calls.append(RecordedLLMCall(messages=tuple(messages), tool_names=tool_names))
        if not self._pending:
            raise FakeLLMExhausted("FakeLLM esgotou o roteiro")
        return self._pending.pop(0)
