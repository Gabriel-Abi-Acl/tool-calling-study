# Function calling nativo: o payload já traz nome e argumentos, como string JSON ou objeto.
# O parser também aceita o envelope `choices[0].message` para o mesmo código ler resposta bruta.

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from tool_calling_study.errors import ParseError
from tool_calling_study.llm.base import LLMClient, LLMMessage, LLMResponse
from tool_calling_study.strategies.base import Decision, PriorTurn
from tool_calling_study.strategies.json_util import (
    coerce_arguments,
    loads_json_document,
    require_object,
)
from tool_calling_study.tools.schema import ToolCall, ToolSpec

_SYSTEM = (
    "You may call the provided tools. "
    "If no tool is needed, answer in plain text and do not call any tool."
)


@dataclass(frozen=True)
class NativeMessage:
    """Mensagem nativa já validada. `content` guarda o texto quando não há chamada."""

    tool_calls: tuple[ToolCall, ...]
    content: str

    def as_response(self) -> LLMResponse:
        return LLMResponse(content=self.content, tool_calls=self.tool_calls)


class NativeToolStrategy:
    """Envia os schemas ao provedor e lê `tool_calls`. Texto puro fecha o turno."""

    name = "native"

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def decide(
        self,
        task_prompt: str,
        turns: Sequence[PriorTurn],
        tools: Sequence[ToolSpec],
    ) -> Decision:
        response = self._llm.complete(_messages(task_prompt, turns), tools=tools)
        if response.tool_calls:
            return Decision(
                tool_calls=response.tool_calls,
                final_answer=None,
                assistant_text=response.content,
            )
        text = response.content.strip()
        if text.startswith("{") or text.startswith("```"):
            parsed = parse_native_payload(text)
            if parsed.tool_calls:
                return Decision(
                    tool_calls=parsed.tool_calls,
                    final_answer=None,
                    assistant_text=text,
                )
            if parsed.content.strip():
                return Decision(
                    tool_calls=(),
                    final_answer=parsed.content.strip(),
                    assistant_text=text,
                )
        if not text:
            raise ParseError("resposta nativa vazia")
        return Decision(tool_calls=(), final_answer=text, assistant_text=text)


def parse_native_payload(payload: dict[str, Any] | str) -> NativeMessage:
    """Lê uma mensagem de function calling. Argumentos inválidos levantam `ParseError`."""
    if isinstance(payload, str):
        document = require_object(loads_json_document(payload), message="payload nativo inválido")
    elif isinstance(payload, dict):
        document = payload
    else:
        raise ParseError("payload nativo inválido")
    message = _message_dict(document)
    content = message.get("content") or ""
    if not isinstance(content, str):
        raise ParseError("content nativo inválido")
    raw_calls = message.get("tool_calls") or []
    if not isinstance(raw_calls, list):
        raise ParseError("tool_calls precisa ser uma lista")
    calls = tuple(_parse_call(item, index) for index, item in enumerate(raw_calls, start=1))
    return NativeMessage(tool_calls=calls, content=content)


def _messages(task_prompt: str, turns: Sequence[PriorTurn]) -> list[LLMMessage]:
    messages = [
        LLMMessage(role="system", content=_SYSTEM),
        LLMMessage(role="user", content=task_prompt),
    ]
    for turn in turns:
        calls = tuple(executed.call for executed in turn.calls)
        messages.append(LLMMessage(role="assistant", content=turn.assistant_text, tool_calls=calls))
        for executed in turn.calls:
            messages.append(
                LLMMessage(
                    role="tool",
                    content=executed.output,
                    name=executed.call.name,
                    tool_call_id=executed.call.call_id,
                )
            )
    return messages


def _message_dict(document: dict[str, Any]) -> dict[str, Any]:
    if "choices" not in document:
        return document
    choices = document["choices"]
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ParseError("choices vazio")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ParseError("message ausente")
    return message


def _parse_call(item: Any, index: int) -> ToolCall:
    if not isinstance(item, dict):
        raise ParseError(f"tool call {index} inválida")
    function = item.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        arguments = function.get("arguments")
    else:
        name = item.get("name")
        arguments = item.get("arguments")
    if not isinstance(name, str) or not name.strip():
        raise ParseError(f"tool call {index} sem nome")
    if arguments is None:
        raise ParseError(f"tool call {index} sem arguments")
    call_id = item.get("id")
    if call_id is not None and not isinstance(call_id, str):
        raise ParseError(f"tool call {index} com id inválido")
    return ToolCall(
        name=name.strip(),
        arguments=coerce_arguments(arguments),
        call_id=call_id or f"call_{index}",
    )
