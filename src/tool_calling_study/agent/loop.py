# Loop único para as três estratégias. A diferença fica no `decide`, não num executor por protocolo.
# Parse quebrado encerra a tarefa com `stopped_reason` para a bateria continuar nas outras.

from __future__ import annotations

from dataclasses import dataclass

from tool_calling_study.checks import find_unnecessary_calls
from tool_calling_study.errors import ParseError
from tool_calling_study.strategies.base import ExecutedCall, PriorTurn, ToolStrategy
from tool_calling_study.tasks.fixtures import Task
from tool_calling_study.tools.registry import ToolRegistry

_DEFAULT_MAX_TURNS = 4


@dataclass(frozen=True)
class AgentRun:
    """Rastro de uma tarefa: ferramentas, argumentos, resposta e chamadas fora do gabarito."""

    task_id: str
    strategy: str
    calls: tuple[ExecutedCall, ...]
    final_answer: str | None
    unnecessary_calls: tuple[str, ...]
    stopped_reason: str

    @property
    def selected_tools(self) -> tuple[str, ...]:
        return tuple(item.call.name for item in self.calls)


def run_task(
    task: Task,
    strategy: ToolStrategy,
    registry: ToolRegistry,
    *,
    max_turns: int = _DEFAULT_MAX_TURNS,
) -> AgentRun:
    """Roda a tarefa até a resposta final, o limite de turnos ou um erro de parse."""
    if max_turns < 1:
        raise ValueError("max_turns precisa ser pelo menos 1")
    turns: list[PriorTurn] = []
    recorded: list[ExecutedCall] = []
    final_answer: str | None = None
    stopped_reason = "max_turns"
    for _ in range(max_turns):
        try:
            decision = strategy.decide(task.prompt, turns, registry.specs())
        except ParseError:
            stopped_reason = "parse_error"
            break
        if decision.tool_calls:
            executed: list[ExecutedCall] = []
            for call in decision.tool_calls:
                result = registry.execute(call)
                item = ExecutedCall(call=call, output=result.output, ok=result.ok)
                executed.append(item)
                recorded.append(item)
            turns.append(PriorTurn(assistant_text=decision.assistant_text, calls=tuple(executed)))
            continue
        final_answer = decision.final_answer
        stopped_reason = "final"
        break
    invoked = [item.call.name for item in recorded]
    return AgentRun(
        task_id=task.id,
        strategy=strategy.name,
        calls=tuple(recorded),
        final_answer=final_answer,
        unnecessary_calls=tuple(find_unnecessary_calls(invoked, task)),
        stopped_reason=stopped_reason,
    )
