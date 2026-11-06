# Chamada desnecessária no loop: ferramenta fora do gabarito ou listada como proibida.
# Repetir uma ferramenta esperada não entra aqui. O harness conta essa repetição como extra_call.

from __future__ import annotations

from collections.abc import Sequence

from tool_calling_study.tasks.fixtures import Task


def find_unnecessary_calls(invoked_tools: Sequence[str], task: Task) -> list[str]:
    """Devolve as chamadas que a tarefa não pedia, na ordem em que ocorreram."""
    allowed = set(task.expected_tools)
    forbidden = set(task.forbidden_tools)
    return [name for name in invoked_tools if name in forbidden or name not in allowed]
