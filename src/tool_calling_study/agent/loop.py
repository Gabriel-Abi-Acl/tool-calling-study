# Loop único para as três estratégias. A diferença fica no `decide`, não num executor por protocolo.
# Parse quebrado entra na nota. Falha de transporte encerra a tarefa e o harness não inventa nota.

from __future__ import annotations

from dataclasses import dataclass

from tool_calling_study.checks import find_unnecessary_calls
from tool_calling_study.errors import ParseError, is_transport_failure
from tool_calling_study.strategies.base import ExecutedCall, PriorTurn, ToolStrategy
from tool_calling_study.tasks.fixtures import Task
from tool_calling_study.tools.registry import ToolRegistry

_DEFAULT_MAX_TURNS = 4


@dataclass(frozen=True)
class AgentRun:
    """Rastro de uma tarefa. `error` só é preenchido quando o modelo não chegou a responder."""

    task_id: str
    strategy: str
    calls: tuple[ExecutedCall, ...]
    final_answer: str | None
    unnecessary_calls: tuple[str, ...]
    stopped_reason: str
    error: str | None = None

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
    """Roda até a resposta final, o limite de turnos, erro de parse ou falha de transporte."""
    if max_turns < 1:
        raise ValueError("max_turns precisa ser pelo menos 1")
    turns: list[PriorTurn] = []
    recorded: list[ExecutedCall] = []
    final_answer: str | None = None
    stopped_reason = "max_turns"
    error: str | None = None
    for _ in range(max_turns):
        try:
            decision = strategy.decide(task.prompt, turns, registry.specs())
        except ParseError:
            stopped_reason = "parse_error"
            break
        except Exception as exc:
            # Chamadas já feitas permanecem no rastro. A nota fica por conta de quem avalia.
            if not is_transport_failure(exc):
                raise
            stopped_reason = "transport_error"
            error = _transport_detail(exc)
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
        error=error,
    )


def _transport_detail(exc: BaseException) -> str:
    # O JSONL guarda o tipo e a mensagem, sem traceback. Resposta crua de SDK não entra inteira.
    detail = f"{type(exc).__name__}: {exc}"
    if len(detail) > 300:
        return detail[:300] + "..."
    return detail
