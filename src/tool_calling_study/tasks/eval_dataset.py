# Dataset v1 de avaliação: uma ferramenta, cadeia de duas, ou abstenção.
# Linha fora do schema ou do contrato das ferramentas falsas é recusada.

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tool_calling_study.errors import StudyError
from tool_calling_study.tools.fake_tools import build_fake_registry
from tool_calling_study.tools.registry import ToolRegistry
from tool_calling_study.tools.schema import ToolCall

_CATEGORIES = ("single", "multi", "abstain")
_SEQUENCE_TOOLS = ("lookup", "calculator")
_TASK_FIELDS = {
    "id",
    "category",
    "prompt",
    "allowed_tools",
    "tool_call_allowed",
    "expected_sequence",
    "reference_answer",
}
_STEP_FIELDS = {"name", "argument_constraints"}
_ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_DATASET = Path("data") / "eval" / "tasks.v1.jsonl"


class EvalDatasetError(StudyError):
    """Linha do dataset v1 quebrou o schema ou o gabarito das ferramentas falsas."""


@dataclass(frozen=True)
class ExpectedStep:
    """Passo do gabarito. No v1, a restrição de cada argumento é o valor exato."""

    name: str
    argument_constraints: dict[str, Any]


@dataclass(frozen=True)
class EvalTask:
    """Tarefa do dataset. `reference_answer` é nulo só quando a resposta correta é recusar."""

    id: str
    category: str
    prompt: str
    allowed_tools: tuple[str, ...]
    tool_call_allowed: bool
    expected_sequence: tuple[ExpectedStep, ...]
    reference_answer: str | None


def load_eval_tasks(path: Path | None = None) -> tuple[EvalTask, ...]:
    """Lê o JSONL e recusa linha malformada. Sem `path`, abre `data/eval/tasks.v1.jsonl`."""
    source = _default_dataset_path() if path is None else Path(path)
    if not source.is_file():
        raise EvalDatasetError(f"dataset ausente: {source}")
    registry = build_fake_registry()
    known = {spec.name for spec in registry.specs()}
    tasks = _parse_document(source.read_text(encoding="utf-8"), registry, known)
    if not tasks:
        raise EvalDatasetError("dataset vazio")
    return tuple(tasks)


def _default_dataset_path() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / _DATASET
        if (parent / "pyproject.toml").is_file() and candidate.is_file():
            return candidate
    raise EvalDatasetError("data/eval/tasks.v1.jsonl não encontrado")


def _parse_document(text: str, registry: ToolRegistry, known: set[str]) -> list[EvalTask]:
    tasks: list[EvalTask] = []
    seen: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvalDatasetError(f"linha {line_number}: JSON inválido") from exc
        if not isinstance(item, dict):
            raise EvalDatasetError(f"linha {line_number}: tarefa inválida")
        task = _parse_task(item, registry, known)
        if task.id in seen:
            raise EvalDatasetError(f"id duplicado: {task.id}")
        seen.add(task.id)
        tasks.append(task)
    return tasks


def _parse_task(item: dict[str, Any], registry: ToolRegistry, known: set[str]) -> EvalTask:
    extra = set(item) - _TASK_FIELDS
    if extra:
        raise EvalDatasetError(f"campo inesperado: {sorted(extra)[0]}")
    missing = _TASK_FIELDS - set(item)
    if missing:
        raise EvalDatasetError(f"campo ausente: {sorted(missing)[0]}")
    task_id = _require_id(item["id"])
    category = item["category"]
    if category not in _CATEGORIES:
        raise EvalDatasetError(f"{task_id}: category inválida")
    prompt = _require_text(task_id, item["prompt"], "prompt")
    allowed = _parse_allowed(task_id, item["allowed_tools"], known)
    tool_call_allowed = item["tool_call_allowed"]
    if not isinstance(tool_call_allowed, bool):
        raise EvalDatasetError(f"{task_id}: tool_call_allowed inválido")
    sequence = _parse_sequence(task_id, item["expected_sequence"], allowed)
    reference = _parse_reference(task_id, item["reference_answer"])
    _check_shape(task_id, category, tool_call_allowed, sequence, reference)
    if sequence:
        _check_executable(task_id, sequence, reference, registry)
    return EvalTask(
        id=task_id,
        category=category,
        prompt=prompt,
        allowed_tools=allowed,
        tool_call_allowed=tool_call_allowed,
        expected_sequence=sequence,
        reference_answer=reference,
    )


def _check_shape(
    task_id: str,
    category: str,
    tool_call_allowed: bool,
    sequence: tuple[ExpectedStep, ...],
    reference: str | None,
) -> None:
    # A categoria tem de bater com o booleano e com o tamanho da sequência.
    # Abstenção pode ter texto fechado ou nulo; tarefa com ferramenta não pode ter nulo.
    if category == "single" and (len(sequence) != 1 or not tool_call_allowed):
        raise EvalDatasetError(f"{task_id}: tarefa single exige uma chamada")
    if category == "multi" and (len(sequence) != 2 or not tool_call_allowed):
        raise EvalDatasetError(f"{task_id}: tarefa multi exige duas chamadas")
    if category == "abstain" and (tool_call_allowed or sequence):
        raise EvalDatasetError(f"{task_id}: abstenção não chama ferramenta")
    if category != "abstain" and reference is None:
        raise EvalDatasetError(f"{task_id}: reference_answer vazio")


def _check_executable(
    task_id: str,
    sequence: tuple[ExpectedStep, ...],
    reference: str | None,
    registry: ToolRegistry,
) -> None:
    # O gabarito tem de rodar na ferramenta falsa. Na cadeia, o passo seguinte
    # carrega a saída anterior em algum argumento, senão os passos são soltos.
    previous: str | None = None
    last = ""
    for step in sequence:
        if previous is not None and not _uses_previous(step, previous):
            raise EvalDatasetError(f"{task_id}: cadeia não usa a saída anterior")
        result = registry.execute(ToolCall(step.name, step.argument_constraints))
        if not result.ok:
            raise EvalDatasetError(f"{task_id}: argumentos fora do schema ({result.output})")
        if result.output == "not_found":
            raise EvalDatasetError(f"{task_id}: consulta fora da base sintética")
        previous = result.output
        last = result.output
    if reference != last:
        raise EvalDatasetError(f"{task_id}: reference_answer difere da última ferramenta ({last})")


def _uses_previous(step: ExpectedStep, previous: str) -> bool:
    return any(
        isinstance(value, str) and previous in value for value in step.argument_constraints.values()
    )


def _parse_sequence(
    task_id: str,
    raw: object,
    allowed: tuple[str, ...],
) -> tuple[ExpectedStep, ...]:
    if not isinstance(raw, list):
        raise EvalDatasetError(f"{task_id}: expected_sequence inválida")
    return tuple(_parse_step(task_id, step, allowed) for step in raw)


def _parse_step(task_id: str, raw: object, allowed: tuple[str, ...]) -> ExpectedStep:
    if not isinstance(raw, dict):
        raise EvalDatasetError(f"{task_id}: passo inválido")
    if set(raw) - _STEP_FIELDS:
        raise EvalDatasetError(f"{task_id}: campo inesperado no passo")
    if not _STEP_FIELDS <= set(raw):
        raise EvalDatasetError(f"{task_id}: passo incompleto")
    name = raw["name"]
    if not isinstance(name, str) or name not in allowed:
        raise EvalDatasetError(f"{task_id}: ferramenta fora de allowed_tools")
    if name not in _SEQUENCE_TOOLS:
        raise EvalDatasetError(f"{task_id}: sequência espera lookup ou calculator")
    constraints = raw["argument_constraints"]
    if not isinstance(constraints, dict) or not constraints:
        raise EvalDatasetError(f"{task_id}: argument_constraints inválido")
    if not all(isinstance(key, str) for key in constraints):
        raise EvalDatasetError(f"{task_id}: argument_constraints inválido")
    for value in constraints.values():
        if isinstance(value, str) and (not value or value != value.strip()):
            raise EvalDatasetError(f"{task_id}: argumento vazio ou com espaço nas bordas")
    return ExpectedStep(name=name, argument_constraints=dict(constraints))


def _parse_allowed(task_id: str, raw: object, known: set[str]) -> tuple[str, ...]:
    if not isinstance(raw, list) or not raw:
        raise EvalDatasetError(f"{task_id}: allowed_tools inválido")
    names: list[str] = []
    for name in raw:
        if not isinstance(name, str) or name not in known:
            raise EvalDatasetError(f"{task_id}: ferramenta desconhecida")
        names.append(name)
    if len(names) != len(set(names)):
        raise EvalDatasetError(f"{task_id}: allowed_tools duplicado")
    return tuple(names)


def _parse_reference(task_id: str, value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise EvalDatasetError(f"{task_id}: reference_answer inválido")
    return value


def _require_id(value: object) -> str:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise EvalDatasetError("id inválido")
    return value


def _require_text(task_id: str, value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvalDatasetError(f"{task_id}: {field} inválido")
    return value.strip()
