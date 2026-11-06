# Plano JSON: o modelo lista os passos antes da execução. O loop roda a lista inteira.
# Resposta final no mesmo objeto dos passos é ignorada pela estratégia até o turno seguinte.

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

from tool_calling_study.errors import ParseError
from tool_calling_study.llm.base import LLMClient, LLMMessage
from tool_calling_study.strategies.base import Decision, PriorTurn, render_tool_catalog
from tool_calling_study.strategies.json_util import (
    coerce_arguments,
    loads_json_document,
    require_object,
)
from tool_calling_study.tools.schema import ToolCall, ToolSpec

_SYSTEM = """Reply with one JSON object and nothing else.
{"steps": [{"tool": "tool_name", "arguments": {"param": "value"}}], "final_answer": "optional"}

Put every tool you need in steps before any final answer.
If no tool should be called, return {"steps": [], "final_answer": "the answer"}.
After you receive tool results, return {"steps": [], "final_answer": "the answer"}.
Do not call a tool the question does not need.
Available tools:
"""


@dataclass(frozen=True)
class JsonPlan:
    """Plano validado. `steps` vazio exige `final_answer`."""

    steps: tuple[ToolCall, ...]
    final_answer: str | None


class JsonPlanStrategy:
    """Pede um plano JSON em texto. Os schemas não vão como function calling."""

    name = "json_plan"

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def decide(
        self,
        task_prompt: str,
        turns: Sequence[PriorTurn],
        tools: Sequence[ToolSpec],
    ) -> Decision:
        response = self._llm.complete(_messages(task_prompt, turns, tools), tools=None)
        plan = parse_json_plan(response.content)
        if plan.steps:
            return Decision(
                tool_calls=plan.steps,
                final_answer=None,
                assistant_text=response.content,
            )
        return Decision(
            tool_calls=(),
            final_answer=plan.final_answer,
            assistant_text=response.content,
        )


def parse_json_plan(text: str) -> JsonPlan:
    """Lê o objeto de plano. Cada passo precisa de `tool` e de `arguments` objeto."""
    data = require_object(loads_json_document(text), message="plano JSON precisa ser um objeto")
    if "steps" not in data:
        raise ParseError("plano sem steps")
    raw_steps = data["steps"]
    if not isinstance(raw_steps, list):
        raise ParseError("steps precisa ser uma lista")
    steps = tuple(_parse_step(step, index) for index, step in enumerate(raw_steps, start=1))
    final_answer = data.get("final_answer")
    if final_answer is not None and not isinstance(final_answer, str):
        raise ParseError("final_answer precisa ser string")
    cleaned = final_answer.strip() if isinstance(final_answer, str) else None
    if not steps and not cleaned:
        raise ParseError("plano sem passos exige final_answer")
    return JsonPlan(steps=steps, final_answer=cleaned or None)


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
        # O plano emitido volta como assistant para o turno final ver a própria lista.
        # Sem isso o modelo recebe só os retornos e perde a ordem que pediu.
        messages.append(LLMMessage(role="assistant", content=turn.assistant_text))
        messages.append(LLMMessage(role="user", content=_observations((turn,))))
    return messages


def _observations(turns: Sequence[PriorTurn]) -> str:
    lines = ["Tool results:"]
    for turn in turns:
        for executed in turn.calls:
            arguments = json.dumps(
                dict(executed.call.arguments),
                ensure_ascii=False,
                sort_keys=True,
            )
            lines.append(f"{executed.call.name}({arguments}) -> {executed.output}")
    lines.append('Reply with {"steps": [], "final_answer": "..."} only.')
    return "\n".join(lines)


def _parse_step(step: object, index: int) -> ToolCall:
    if not isinstance(step, dict):
        raise ParseError(f"passo {index} inválido")
    tool = step.get("tool")
    if not isinstance(tool, str) or not tool.strip():
        raise ParseError(f"passo {index} sem tool")
    if "arguments" not in step:
        raise ParseError(f"passo {index} sem arguments")
    return ToolCall(
        name=tool.strip(),
        arguments=coerce_arguments(step["arguments"]),
        call_id=f"plan_{index}",
    )
