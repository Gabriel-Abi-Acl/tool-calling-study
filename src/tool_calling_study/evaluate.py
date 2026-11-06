# Harness offline do dataset v1. Mede a corrida; não publica acurácia de modelo.
# Erro de transporte deixa a nota nula. Os números do estudo continuam no PR4.

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tool_calling_study.agent.loop import AgentRun, run_task
from tool_calling_study.config import load_settings
from tool_calling_study.llm.base import LLMClient, LLMResponse
from tool_calling_study.llm.factory import build_llm
from tool_calling_study.strategies.base import build_strategy
from tool_calling_study.tasks.eval_dataset import EvalDatasetError, EvalTask, load_eval_tasks
from tool_calling_study.tasks.fixtures import ExpectedCall, Task
from tool_calling_study.tools.fake_tools import build_fake_registry

STRATEGY_NAMES = ("react", "native", "json_plan")
FAILURE_MODES = (
    "wrong_tool",
    "bad_arguments",
    "extra_call",
    "missed_call",
    "retrieved_distractor",
    "failed_abstention",
)
_DISTRACTOR = "weather"


@dataclass(frozen=True)
class SelectedCall:
    """Uma chamada executada, na ordem do loop."""

    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class TaskRecord:
    """Uma linha do JSONL. Notas nulas significam que a corrida não foi julgada."""

    task_id: str
    strategy: str
    category: str
    selected_sequence: tuple[SelectedCall, ...]
    sequence_match: bool | None
    arguments_correct: bool | None
    unnecessary_calls: tuple[str, ...]
    abstained: bool | None
    final_answer: str | None
    answer_correct: bool | None
    correct: bool | None
    failure_modes: tuple[str, ...]
    latency_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None
    estimated_cost_usd: float | None
    stopped_reason: str
    error: str | None

    def as_json(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "strategy": self.strategy,
            "category": self.category,
            "selected_sequence": [
                {"name": call.name, "arguments": call.arguments} for call in self.selected_sequence
            ],
            "sequence_match": self.sequence_match,
            "arguments_correct": self.arguments_correct,
            "unnecessary_calls": list(self.unnecessary_calls),
            "abstained": self.abstained,
            "final_answer": self.final_answer,
            "answer_correct": self.answer_correct,
            "correct": self.correct,
            "failure_modes": list(self.failure_modes),
            "latency_ms": self.latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
            "stopped_reason": self.stopped_reason,
            "error": self.error,
        }


def evaluate_tasks(
    tasks: Sequence[EvalTask],
    strategy_name: str,
    llm_for_task: Callable[[EvalTask], LLMClient],
) -> tuple[TaskRecord, ...]:
    """Avalia cada tarefa com um cliente novo. O roteiro de uma não vaza para a próxima."""
    return tuple(evaluate_task(task, strategy_name, llm_for_task(task)) for task in tasks)


def evaluate_task(task: EvalTask, strategy_name: str, llm: LLMClient) -> TaskRecord:
    """Roda uma tarefa nas ferramentas falsas e devolve o registro."""
    meter = _Meter(llm)
    strategy = build_strategy(strategy_name, meter)
    started = time.perf_counter()
    run = run_task(_as_loop_task(task), strategy, build_fake_registry(task.allowed_tools))
    latency_ms = round((time.perf_counter() - started) * 1000, 3)
    return score_run(
        task,
        run,
        latency_ms=latency_ms,
        prompt_tokens=meter.prompt_tokens,
        completion_tokens=meter.completion_tokens,
        estimated_cost_usd=meter.estimated_cost_usd,
    )


def score_run(
    task: EvalTask,
    run: AgentRun,
    *,
    latency_ms: float = 0.0,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    estimated_cost_usd: float | None = None,
) -> TaskRecord:
    """Compara o rastro com o gabarito. Transporte não recebe nota nem rótulo de falha."""
    selected = tuple(
        SelectedCall(name=item.call.name, arguments=dict(item.call.arguments)) for item in run.calls
    )
    # Transporte: a sequência já executada fica no registro, mas as notas não são inventadas.
    scored = run.stopped_reason != "transport_error"
    unnecessary = _unnecessary_calls(task, selected)
    modes = _failure_modes(task, selected) if scored else ()
    sequence_match = _sequence_match(task, selected) if scored else None
    arguments_correct = _arguments_correct(task, selected) if scored else None
    return TaskRecord(
        task_id=task.id,
        strategy=run.strategy,
        category=task.category,
        selected_sequence=selected,
        sequence_match=sequence_match,
        arguments_correct=arguments_correct,
        unnecessary_calls=unnecessary,
        abstained=None if not scored else len(selected) == 0,
        final_answer=run.final_answer,
        answer_correct=_answer_correct(task, run.final_answer) if scored else None,
        correct=_correct(task, selected, run.final_answer) if scored else None,
        failure_modes=modes,
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        estimated_cost_usd=estimated_cost_usd,
        stopped_reason=run.stopped_reason,
        error=run.error,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI. JSONL na saída; o resumo no stderr não traz acurácia."""
    parser = _parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    blocked = _live_block_message(allow_live=args.allow_live)
    if blocked is not None:
        print(blocked, file=sys.stderr)
        return 2
    try:
        tasks = load_eval_tasks(args.dataset)
    except EvalDatasetError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    names = STRATEGY_NAMES if args.strategy == "all" else (args.strategy,)
    records: list[TaskRecord] = []
    for name in names:
        records.extend(evaluate_tasks(tasks, name, lambda _task: build_llm()))
    try:
        _emit(records, args.output)
    except OSError as exc:
        print(f"não foi possível gravar a saída: {exc}", file=sys.stderr)
        return 1
    print(f"registros={len(records)} estrategia={args.strategy}", file=sys.stderr)
    return 0


def cli() -> None:
    """Entrada do console script. O código de saída é o de `main`."""
    raise SystemExit(main())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate-tools",
        description=(
            "Percorre o dataset v1 com ferramentas falsas. Não publica acurácia de modelo."
        ),
    )
    parser.add_argument(
        "--strategy",
        required=True,
        choices=(*STRATEGY_NAMES, "all"),
        help="react, native, json_plan ou all",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="JSONL das tarefas. O padrão é data/eval/tasks.v1.jsonl.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Arquivo JSONL. Sem este argumento, a saída vai para stdout.",
    )
    parser.add_argument(
        "--allow-live",
        action="store_true",
        help="Permite LLM_PROVIDER openai ou anthropic. A medição com modelo real é o PR4.",
    )
    return parser


def _live_block_message(*, allow_live: bool) -> str | None:
    # Sem a flag, openai e anthropic nem começam. Evita corrida acidental com chave no ambiente.
    settings = load_settings()
    if settings.llm_provider == "fake" or allow_live:
        return None
    return (
        f"provedor {settings.llm_provider} bloqueado: este harness não mede modelo real. "
        "Deixe LLM_PROVIDER=fake ou passe --allow-live só no experimento do PR4."
    )


def _emit(records: Sequence[TaskRecord], output: Path | None) -> None:
    text = "".join(json.dumps(record.as_json(), ensure_ascii=False) + "\n" for record in records)
    if output is None:
        sys.stdout.write(text)
        return
    output.write_text(text, encoding="utf-8")


def _as_loop_task(task: EvalTask) -> Task:
    expected = tuple(
        ExpectedCall(tool=step.name, arguments=dict(step.argument_constraints))
        for step in task.expected_sequence
    )
    expected_names = {step.name for step in task.expected_sequence}
    forbidden = tuple(name for name in task.allowed_tools if name not in expected_names)
    return Task(
        id=task.id,
        kind=task.category,
        prompt=task.prompt,
        expected_calls=expected,
        forbidden_tools=forbidden,
        reference_answer=task.reference_answer,
    )


def _sequence_match(task: EvalTask, calls: Sequence[SelectedCall]) -> bool:
    expected = tuple(step.name for step in task.expected_sequence)
    return tuple(call.name for call in calls) == expected


def _arguments_correct(task: EvalTask, calls: Sequence[SelectedCall]) -> bool:
    # Argumento certo exige a sequência de nomes inteira. Prefixo bom com chamada a mais não conta.
    if not _sequence_match(task, calls):
        return False
    return all(
        dict(call.arguments) == dict(step.argument_constraints)
        for call, step in zip(calls, task.expected_sequence, strict=True)
    )


def _answer_correct(task: EvalTask, final_answer: str | None) -> bool | None:
    # Referência nula não tem texto para comparar. Na abstenção, não chamar ferramenta é a nota.
    if task.reference_answer is None:
        return None
    return final_answer == task.reference_answer


def _correct(task: EvalTask, calls: Sequence[SelectedCall], final_answer: str | None) -> bool:
    if task.category == "abstain":
        if calls:
            return False
        if task.reference_answer is None:
            return True
        return final_answer == task.reference_answer
    if not _sequence_match(task, calls) or not _arguments_correct(task, calls):
        return False
    if task.reference_answer is None:
        return False
    return final_answer == task.reference_answer


def _unnecessary_calls(task: EvalTask, calls: Sequence[SelectedCall]) -> tuple[str, ...]:
    expected = task.expected_sequence
    names: list[str] = []
    for index in range(min(len(expected), len(calls))):
        if calls[index].name != expected[index].name:
            names.append(calls[index].name)
    names.extend(call.name for call in calls[len(expected) :])
    return tuple(names)


def _failure_modes(task: EvalTask, calls: Sequence[SelectedCall]) -> tuple[str, ...]:
    expected = task.expected_sequence
    expected_names = {step.name for step in expected}
    found: set[str] = set()
    paired = min(len(expected), len(calls))
    for index in range(paired):
        call = calls[index]
        step = expected[index]
        if call.name != step.name:
            found.add("wrong_tool")
        elif dict(call.arguments) != dict(step.argument_constraints):
            found.add("bad_arguments")
    if len(calls) > len(expected):
        found.add("extra_call")
        for call in calls[len(expected) :]:
            if call.name not in expected_names:
                found.add("wrong_tool")
    if len(expected) > len(calls):
        found.add("missed_call")
    if any(call.name == _DISTRACTOR for call in calls) and _DISTRACTOR not in expected_names:
        found.add("retrieved_distractor")
    if task.category == "abstain" and calls:
        found.add("failed_abstention")
    return tuple(mode for mode in FAILURE_MODES if mode in found)


class _Meter:
    """Soma token e custo só quando todas as respostas da tarefa informaram o campo."""

    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner
        self._prompt: list[int | None] = []
        self._completion: list[int | None] = []
        self._cost: list[float | None] = []

    def complete(
        self,
        messages: Sequence[Any],
        *,
        tools: Sequence[Any] | None = None,
    ) -> LLMResponse:
        response = self._inner.complete(messages, tools=tools)
        self._prompt.append(response.prompt_tokens)
        self._completion.append(response.completion_tokens)
        self._cost.append(response.estimated_cost_usd)
        return response

    @property
    def prompt_tokens(self) -> int | None:
        return _sum_tokens(self._prompt)

    @property
    def completion_tokens(self) -> int | None:
        return _sum_tokens(self._completion)

    @property
    def estimated_cost_usd(self) -> float | None:
        if not self._cost or any(value is None for value in self._cost):
            return None
        return float(sum(value for value in self._cost if value is not None))


def _sum_tokens(values: Sequence[int | None]) -> int | None:
    if not values or any(value is None for value in values):
        return None
    total = 0
    for value in values:
        if value is None:
            return None
        total += value
    return total


if __name__ == "__main__":
    cli()
