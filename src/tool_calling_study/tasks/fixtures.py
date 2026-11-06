# Fixture provisória: uma ferramenta, duas ferramentas, ou abstenção.
# O carregador recusa tarefa fora desse contrato para o conjunto não mudar de forma silenciosa.

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from tool_calling_study.errors import StudyError

_KINDS = ("single", "multi", "abstain")
_KNOWN_TOOLS = ("lookup", "calculator", "weather")
_MIN_TASKS = 8
_MAX_TASKS = 12


class FixtureError(StudyError):
    """A fixture sintética quebrou o contrato do PR1."""


@dataclass(frozen=True)
class ExpectedCall:
    """Chamada do gabarito. Argumentos são os canônicos, não um log de modelo."""

    tool: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Task:
    """Tarefa sintética. `reference_answer` é o texto da última ferramenta."""

    id: str
    kind: str
    prompt: str
    expected_calls: tuple[ExpectedCall, ...]
    forbidden_tools: tuple[str, ...]
    reference_answer: str | None

    @property
    def expected_tools(self) -> tuple[str, ...]:
        return tuple(call.tool for call in self.expected_calls)


def load_tasks() -> tuple[Task, ...]:
    """Lê e valida `fixtures.json`. O arquivo é dado de estudo, não configuração secreta."""
    return _load_cached()


def get_task(task_id: str) -> Task:
    for task in load_tasks():
        if task.id == task_id:
            return task
    raise FixtureError(f"tarefa ausente: {task_id}")


@lru_cache(maxsize=1)
def _load_cached() -> tuple[Task, ...]:
    path = Path(__file__).with_name("fixtures.json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise FixtureError("fixture precisa ser uma lista")
    tasks = tuple(_parse_task(item) for item in raw)
    _validate_set(tasks)
    return tasks


def _parse_task(item: object) -> Task:
    if not isinstance(item, dict):
        raise FixtureError("tarefa inválida")
    task_id = _require_str(item, "id")
    kind = _require_str(item, "kind")
    prompt = _require_str(item, "prompt")
    expected_raw = item.get("expected_calls")
    forbidden_raw = item.get("forbidden_tools")
    if not isinstance(expected_raw, list) or not isinstance(forbidden_raw, list):
        raise FixtureError(f"{task_id}: listas de chamadas inválidas")
    expected = tuple(_parse_call(task_id, call) for call in expected_raw)
    forbidden = tuple(_require_tool_name(task_id, name) for name in forbidden_raw)
    reference = item.get("reference_answer")
    if reference is not None and not isinstance(reference, str):
        raise FixtureError(f"{task_id}: reference_answer inválido")
    return Task(
        id=task_id,
        kind=kind,
        prompt=prompt,
        expected_calls=expected,
        forbidden_tools=forbidden,
        reference_answer=reference,
    )


def _parse_call(task_id: str, call: object) -> ExpectedCall:
    if not isinstance(call, dict):
        raise FixtureError(f"{task_id}: chamada inválida")
    tool = _require_tool_name(task_id, call.get("tool"))
    arguments = call.get("arguments")
    if not isinstance(arguments, dict) or not arguments:
        raise FixtureError(f"{task_id}: arguments inválidos")
    if not all(isinstance(key, str) for key in arguments):
        raise FixtureError(f"{task_id}: chave de argumento inválida")
    return ExpectedCall(tool=tool, arguments=dict(arguments))


def _validate_set(tasks: tuple[Task, ...]) -> None:
    if not _MIN_TASKS <= len(tasks) <= _MAX_TASKS:
        raise FixtureError("a fixture do PR1 precisa ter entre 8 e 12 tarefas")
    ids = [task.id for task in tasks]
    if len(ids) != len(set(ids)):
        raise FixtureError("id de tarefa duplicado")
    for task in tasks:
        _validate_task(task)


def _validate_task(task: Task) -> None:
    if task.kind not in _KINDS:
        raise FixtureError(f"{task.id}: kind inválido")
    expected = task.expected_tools
    forbidden = set(task.forbidden_tools)
    if set(expected) & forbidden:
        raise FixtureError(f"{task.id}: ferramenta esperada e proibida ao mesmo tempo")
    if task.kind == "single" and len(expected) != 1:
        raise FixtureError(f"{task.id}: tarefa single exige uma chamada")
    if task.kind == "multi" and len(expected) != 2:
        raise FixtureError(f"{task.id}: tarefa multi exige duas chamadas")
    if task.kind == "abstain" and expected:
        raise FixtureError(f"{task.id}: abstenção não espera ferramenta")
    if task.kind == "abstain" and task.reference_answer is not None:
        raise FixtureError(f"{task.id}: abstenção não tem reference_answer neste PR")
    if task.kind != "abstain" and not task.reference_answer:
        raise FixtureError(f"{task.id}: reference_answer vazio")


def _require_str(item: dict[str, Any], field: str) -> str:
    value = item.get(field)
    if not isinstance(value, str) or not value.strip():
        raise FixtureError(f"campo {field} inválido")
    return value.strip()


def _require_tool_name(task_id: str, name: object) -> str:
    if not isinstance(name, str) or name not in _KNOWN_TOOLS:
        raise FixtureError(f"{task_id}: ferramenta desconhecida")
    return name
