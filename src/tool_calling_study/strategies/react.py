# ReAct em texto: um `Action` por turno, ou `Final Answer`. O parser é estrito de propósito.
# Formato quebrado vira `ParseError` para a métrica de formato não ser mascarada por um chute.

from __future__ import annotations

import re
from collections.abc import Sequence

from tool_calling_study.errors import ParseError
from tool_calling_study.llm.base import LLMClient, LLMMessage
from tool_calling_study.strategies.base import Decision, PriorTurn, render_tool_catalog
from tool_calling_study.strategies.json_util import coerce_arguments
from tool_calling_study.tools.schema import ToolCall, ToolSpec

_TOOL_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_LABELS = ("Action Input:", "Final Answer:", "Thought:", "Action:", "Observation:")

_SYSTEM = """You answer questions that may require tools. Use this format exactly.

To call one tool:
Thought: why a tool is needed
Action: tool_name
Action Input: {"param": "value"}

When you know the answer, or when no tool should be called:
Thought: why you are done
Final Answer: the answer

Call at most one tool per turn. Do not call a tool the question does not need.
Available tools:
"""


class ReactToolStrategy:
    """Pede um traço ReAct em texto e lê no máximo uma chamada por turno."""

    name = "react"

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def decide(
        self,
        task_prompt: str,
        turns: Sequence[PriorTurn],
        tools: Sequence[ToolSpec],
    ) -> Decision:
        # Sem `tools` no cliente: o protocolo deste braço é o texto, não o function calling.
        # Passar o schema nativo mudaria o que o modelo devolve e misturaria as estratégias.
        response = self._llm.complete(
            _messages(task_prompt, turns, tools),
            tools=None,
        )
        return parse_react_trace(response.content)


def parse_react_trace(text: str) -> Decision:
    """Converte um turno ReAct em chamada ou resposta final."""
    if not text or not text.strip():
        raise ParseError("traço ReAct vazio")
    blocks = _split_blocks(text)
    actions = [body for label, body in blocks if label == "Action"]
    inputs = [body for label, body in blocks if label == "Action Input"]
    finals = [body for label, body in blocks if label == "Final Answer"]
    if len(actions) > 1:
        raise ParseError("ReAct aceita uma Action por turno")
    if len(finals) > 1:
        raise ParseError("ReAct aceita um Final Answer por turno")
    if actions and finals:
        raise ParseError("turno ReAct não pode misturar Action e Final Answer")
    if actions:
        if len(inputs) != 1:
            raise ParseError("Action exige exatamente um Action Input")
        name = actions[0].strip()
        if not _TOOL_NAME.fullmatch(name):
            raise ParseError("nome de ferramenta inválido")
        call = ToolCall(name=name, arguments=coerce_arguments(inputs[0]), call_id=None)
        return Decision(tool_calls=(call,), final_answer=None, assistant_text=text.strip())
    if finals:
        answer = finals[0].strip()
        if not answer:
            raise ParseError("Final Answer vazio")
        return Decision(tool_calls=(), final_answer=answer, assistant_text=text.strip())
    raise ParseError("traço ReAct sem Action nem Final Answer")


def _messages(
    task_prompt: str,
    turns: Sequence[PriorTurn],
    tools: Sequence[ToolSpec],
) -> list[LLMMessage]:
    messages = [
        LLMMessage(role="system", content=_SYSTEM + render_tool_catalog(tools)),
        LLMMessage(role="user", content=task_prompt),
    ]
    for turn in turns:
        messages.append(LLMMessage(role="assistant", content=turn.assistant_text))
        for executed in turn.calls:
            messages.append(LLMMessage(role="user", content=f"Observation: {executed.output}"))
    return messages


def _split_blocks(text: str) -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []
    label: str | None = None
    lines: list[str] = []

    def flush() -> None:
        if label is not None:
            blocks.append((label, "\n".join(lines).strip()))

    for line in text.splitlines():
        matched = _match_label(line)
        if matched is None:
            if label is not None:
                lines.append(line)
            continue
        flush()
        label, rest = matched
        lines = [rest] if rest else []
    flush()
    return blocks


def _match_label(line: str) -> tuple[str, str] | None:
    for prefix in _LABELS:
        if line.startswith(prefix):
            return prefix[:-1], line[len(prefix) :].lstrip()
    return None
