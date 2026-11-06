# O harness roda com cliente roteirizado. Nenhum teste chama API nem publica acurácia.
# As 45 tarefas passam pelo parser real. O acerto do roteiro não é número de estudo.

import json
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import pytest

from tool_calling_study.agent.loop import AgentRun
from tool_calling_study.errors import (
    ConfigError,
    MissingCredential,
    ParseError,
    ProviderNotInstalled,
    ToolError,
    is_transport_failure,
)
from tool_calling_study.evaluate import (
    FAILURE_MODES,
    STRATEGY_NAMES,
    evaluate_task,
    evaluate_tasks,
    main,
    score_run,
)
from tool_calling_study.llm.base import LLMResponse
from tool_calling_study.llm.fake import FakeLLM, FakeLLMExhausted
from tool_calling_study.strategies.base import ExecutedCall
from tool_calling_study.tasks import load_eval_tasks
from tool_calling_study.tasks.eval_dataset import EvalTask, ExpectedStep
from tool_calling_study.tools.schema import ToolCall

_ROOT = Path(__file__).resolve().parents[1]
_JSON_KEYS = {
    "task_id",
    "strategy",
    "category",
    "selected_sequence",
    "sequence_match",
    "arguments_correct",
    "unnecessary_calls",
    "abstained",
    "final_answer",
    "answer_correct",
    "correct",
    "failure_modes",
    "latency_ms",
    "prompt_tokens",
    "completion_tokens",
    "estimated_cost_usd",
    "stopped_reason",
    "error",
}


@lru_cache(maxsize=1)
def _tasks() -> tuple[EvalTask, ...]:
    return load_eval_tasks()


def _by_id(task_id: str) -> EvalTask:
    for task in _tasks():
        if task.id == task_id:
            return task
    raise AssertionError(task_id)


def _agent_run(
    task: EvalTask,
    calls: list[tuple[str, dict]],
    final_answer: str | None,
    *,
    stopped_reason: str = "final",
    error: str | None = None,
) -> AgentRun:
    executed = tuple(
        ExecutedCall(call=ToolCall(name, arguments), output="", ok=True)
        for name, arguments in calls
    )
    return AgentRun(
        task_id=task.id,
        strategy="react",
        calls=executed,
        final_answer=final_answer,
        unnecessary_calls=(),
        stopped_reason=stopped_reason,
        error=error,
    )


def _react_action(name: str, arguments: dict) -> LLMResponse:
    payload = json.dumps(arguments, ensure_ascii=False)
    return LLMResponse(
        content="\n".join(
            [
                "Thought: preciso da ferramenta.",
                f"Action: {name}",
                f"Action Input: {payload}",
            ]
        )
    )


def _react_final(answer: str) -> LLMResponse:
    return LLMResponse(content=f"Thought: pronto.\nFinal Answer: {answer}")


def _script(task: EvalTask, strategy: str) -> FakeLLM:
    # O texto passa pelos parsers reais. Não é um modelo e não abre rede.
    answer = task.reference_answer if task.reference_answer is not None else "recuso"
    steps = [(step.name, dict(step.argument_constraints)) for step in task.expected_sequence]
    if strategy == "json_plan":
        plan = {"steps": [{"tool": name, "arguments": args} for name, args in steps]}
        responses = [LLMResponse(content=json.dumps(plan, ensure_ascii=False))]
        responses.append(
            LLMResponse(
                content=json.dumps({"steps": [], "final_answer": answer}, ensure_ascii=False)
            )
        )
        if not steps:
            responses = responses[-1:]
        return FakeLLM(responses)
    if strategy == "native":
        responses = []
        if steps:
            payload = {
                "tool_calls": [
                    {
                        "id": f"call_{index}",
                        "type": "function",
                        "function": {"name": name, "arguments": args},
                    }
                    for index, (name, args) in enumerate(steps, start=1)
                ]
            }
            responses.append(LLMResponse(content=json.dumps(payload)))
        responses.append(LLMResponse(content=answer))
        return FakeLLM(responses)
    responses = [_react_action(name, args) for name, args in steps]
    responses.append(_react_final(answer))
    return FakeLLM(responses)


@pytest.mark.parametrize("strategy", STRATEGY_NAMES)
def test_scripted_client_passes_every_v1_task_through_the_real_parsers(strategy: str):
    tasks = _tasks()
    assert len(tasks) == 45
    assert [task.category for task in tasks].count("single") == 15
    assert [task.category for task in tasks].count("multi") == 15
    assert [task.category for task in tasks].count("abstain") == 15
    records = evaluate_tasks(tasks, strategy, lambda task: _script(task, strategy))
    assert len(records) == 45
    assert set(records[0].as_json()) == _JSON_KEYS
    for task, record in zip(tasks, records, strict=True):
        assert record.strategy == strategy
        assert record.correct is True
        assert record.failure_modes == ()
        assert record.sequence_match is True
        assert record.arguments_correct is True
        assert record.unnecessary_calls == ()
        assert record.error is None
        assert record.stopped_reason == "final"
        assert record.prompt_tokens is None
        assert record.completion_tokens is None
        assert record.estimated_cost_usd is None
        assert record.latency_ms >= 0
        assert [call.name for call in record.selected_sequence] == [
            step.name for step in task.expected_sequence
        ]
        for call, step in zip(record.selected_sequence, task.expected_sequence, strict=True):
            assert call.arguments == dict(step.argument_constraints)
        if task.category == "abstain":
            assert record.abstained is True
            if task.reference_answer is None:
                assert record.answer_correct is None
            else:
                assert record.answer_correct is True
                assert record.final_answer == task.reference_answer
        else:
            assert record.abstained is False
            assert record.answer_correct is True
            assert record.final_answer == task.reference_answer
            assert "weather" not in {call.name for call in record.selected_sequence}


def test_tokens_and_cost_sum_when_every_response_reports_them():
    task = _by_id("single-lookup-capital-fr")
    action = _react_action("lookup", {"query": "capital of France"})
    llm = FakeLLM(
        [
            LLMResponse(
                content=action.content,
                prompt_tokens=5,
                completion_tokens=4,
                estimated_cost_usd=0.75,
            ),
            LLMResponse(
                content="Thought: pronto.\nFinal Answer: Paris",
                prompt_tokens=11,
                completion_tokens=3,
                estimated_cost_usd=1.25,
            ),
        ]
    )
    record = evaluate_task(task, "react", llm)
    assert record.correct is True
    assert record.prompt_tokens == 16
    assert record.completion_tokens == 7
    assert record.estimated_cost_usd == 2.0


def test_usage_stays_null_when_any_turn_omits_it():
    task = _by_id("single-lookup-capital-fr")
    action = _react_action("lookup", {"query": "capital of France"})
    final = LLMResponse(
        content="Thought: pronto.\nFinal Answer: Paris",
        prompt_tokens=4,
        completion_tokens=1,
    )
    record = evaluate_task(task, "react", FakeLLM([action, final]))
    assert record.correct is True
    assert record.prompt_tokens is None
    assert record.completion_tokens is None
    assert record.estimated_cost_usd is None


def test_weather_on_a_lookup_task_is_a_distractor():
    task = _by_id("single-lookup-capital-fr")
    llm = FakeLLM(
        [
            _react_action("weather", {"city": "Sao Paulo"}),
            _react_final("não sei"),
        ]
    )
    record = evaluate_task(task, "react", llm)
    assert record.correct is False
    assert record.abstained is False
    assert record.unnecessary_calls == ("weather",)
    assert record.failure_modes == ("wrong_tool", "retrieved_distractor")
    assert record.selected_sequence[0].arguments == {"city": "Sao Paulo"}


def test_calling_a_tool_on_an_abstain_task_is_failed_abstention():
    task = _by_id("abstain-refuse-secret")
    record = score_run(
        task,
        _agent_run(task, [("weather", {"city": "Sao Paulo"})], "não"),
    )
    assert record.correct is False
    assert record.abstained is False
    assert record.failure_modes == (
        "wrong_tool",
        "extra_call",
        "retrieved_distractor",
        "failed_abstention",
    )


def test_null_reference_abstain_is_correct_when_no_tool_is_called():
    task = _by_id("abstain-refuse-mars")
    record = evaluate_task(
        task,
        "react",
        FakeLLM([_react_final("não invento capital")]),
    )
    assert record.correct is True
    assert record.answer_correct is None
    assert record.abstained is True
    assert record.final_answer == "não invento capital"
    assert record.failure_modes == ()


def test_text_reference_abstain_checks_the_final_answer_too():
    task = _by_id("abstain-ola")
    wrong = evaluate_task(task, "react", FakeLLM([_react_final("oi")]))
    right = evaluate_task(task, "json_plan", _script(task, "json_plan"))
    assert wrong.correct is False
    assert wrong.answer_correct is False
    assert wrong.abstained is True
    assert wrong.failure_modes == ()
    assert right.correct is True
    assert right.final_answer == "Olá."


def test_repeating_an_expected_tool_is_an_extra_call():
    task = _by_id("single-lookup-capital-fr")
    record = score_run(
        task,
        _agent_run(
            task,
            [
                ("lookup", {"query": "capital of France"}),
                ("lookup", {"query": "capital of France"}),
            ],
            "Paris",
        ),
    )
    assert record.sequence_match is False
    assert record.arguments_correct is False
    assert record.unnecessary_calls == ("lookup",)
    assert record.failure_modes == ("extra_call",)
    assert record.correct is False


def test_wrong_tool_same_length_is_not_also_a_miss_or_an_extra():
    task = _by_id("single-lookup-capital-fr")
    record = score_run(task, _agent_run(task, [("calculator", {"expression": "1 + 1"})], "2"))
    assert record.failure_modes == ("wrong_tool",)
    assert record.sequence_match is False
    assert record.arguments_correct is False


def test_short_sequence_is_a_missed_call():
    task = _by_id("multi-br-millions")
    record = score_run(
        task,
        _agent_run(task, [("lookup", {"query": "population of Brazil"})], None),
    )
    assert record.failure_modes == ("missed_call",)
    assert record.correct is False


def test_bad_arguments_keep_the_matching_tool_name():
    task = _by_id("single-calc-product")
    record = evaluate_task(
        task,
        "native",
        FakeLLM(
            [
                LLMResponse(
                    content=json.dumps(
                        {
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "function": {
                                        "name": "calculator",
                                        "arguments": {"expression": "17 + 23"},
                                    },
                                }
                            ]
                        }
                    )
                ),
                LLMResponse(content="40"),
            ]
        ),
    )
    assert record.sequence_match is True
    assert record.arguments_correct is False
    assert record.failure_modes == ("bad_arguments",)
    assert record.final_answer == "40"
    assert record.correct is False


def test_bad_arguments_and_a_missing_second_call_are_both_labeled():
    task = _by_id("multi-br-millions")
    record = score_run(
        task, _agent_run(task, [("lookup", {"query": "population of France"})], None)
    )
    assert record.failure_modes == ("bad_arguments", "missed_call")


def test_right_calls_with_the_wrong_final_answer_are_not_correct():
    task = _by_id("single-lookup-capital-fr")
    record = score_run(
        task,
        _agent_run(task, [("lookup", {"query": "capital of France"})], "Lyon"),
    )
    assert record.sequence_match is True
    assert record.arguments_correct is True
    assert record.answer_correct is False
    assert record.correct is False
    assert record.failure_modes == ()


def test_weather_called_on_purpose_is_not_a_distractor():
    task = EvalTask(
        id="custom-weather",
        category="single",
        prompt="previsão fictícia de São Paulo",
        allowed_tools=("weather",),
        tool_call_allowed=True,
        expected_sequence=(ExpectedStep("weather", {"city": "Sao Paulo"}),),
        reference_answer="ensolarado, 28C [fixture]",
    )
    record = score_run(
        task,
        _agent_run(
            task,
            [("weather", {"city": "Sao Paulo"})],
            "ensolarado, 28C [fixture]",
        ),
    )
    assert record.correct is True
    assert "retrieved_distractor" not in record.failure_modes


def test_transport_error_keeps_the_trace_and_leaves_the_score_null():
    task = _by_id("single-lookup-capital-fr")

    class Once:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, messages, *, tools=None):
            self.calls += 1
            if self.calls == 1:
                response = _react_action("lookup", {"query": "capital of France"})
                return LLMResponse(
                    content=response.content,
                    prompt_tokens=8,
                    completion_tokens=2,
                    estimated_cost_usd=1.0,
                )
            raise FakeLLMExhausted("roteiro acabou")

    record = evaluate_task(task, "react", Once())
    assert record.correct is None
    assert record.sequence_match is None
    assert record.arguments_correct is None
    assert record.answer_correct is None
    assert record.abstained is None
    assert record.failure_modes == ()
    assert record.stopped_reason == "transport_error"
    assert record.error is not None
    assert "FakeLLMExhausted" in record.error
    assert [call.name for call in record.selected_sequence] == ["lookup"]
    assert record.prompt_tokens == 8
    assert record.completion_tokens == 2
    assert record.estimated_cost_usd == 1.0


def test_transport_before_any_call_leaves_the_score_null():
    task = _by_id("single-lookup-capital-fr")

    class Boom:
        def complete(self, messages, *, tools=None):
            raise TimeoutError("tempo esgotado")

    record = evaluate_task(task, "react", Boom())
    assert record.correct is None
    assert record.selected_sequence == ()
    assert record.unnecessary_calls == ()
    assert record.failure_modes == ()
    assert record.prompt_tokens is None


def test_parse_error_on_a_tool_task_is_a_miss_not_a_transport_failure():
    task = _by_id("single-lookup-capital-fr")
    record = evaluate_task(
        task,
        "react",
        FakeLLM([LLMResponse(content="sem formato", prompt_tokens=3, completion_tokens=1)]),
    )
    assert record.stopped_reason == "parse_error"
    assert record.error is None
    assert record.correct is False
    assert record.failure_modes == ("missed_call",)
    assert record.prompt_tokens == 3


def test_parse_error_on_null_abstain_counts_as_not_calling_a_tool():
    task = _by_id("abstain-refuse-secret")
    record = evaluate_task(task, "react", FakeLLM([LLMResponse(content="sem formato")]))
    assert record.stopped_reason == "parse_error"
    assert record.correct is True
    assert record.abstained is True
    assert record.answer_correct is None
    assert record.failure_modes == ()


def test_programmer_error_is_not_stored_as_a_null_score():
    class Bad:
        def complete(self, messages, *, tools=None):
            raise ValueError("bug no cliente")

    with pytest.raises(ValueError, match="bug no cliente"):
        evaluate_task(_by_id("abstain-ola"), "react", Bad())


def test_native_strategy_sees_only_the_tools_the_task_allows():
    task = EvalTask(
        id="only-lookup",
        category="single",
        prompt="capital da França no fixture",
        allowed_tools=("lookup",),
        tool_call_allowed=True,
        expected_sequence=(ExpectedStep("lookup", {"query": "capital of France"}),),
        reference_answer="Paris",
    )
    llm = FakeLLM([LLMResponse(content="Paris")])
    record = evaluate_task(task, "native", llm)
    assert llm.calls[0].tool_names == ("lookup",)
    assert record.failure_modes == ("missed_call",)


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (ParseError("formato"), False),
        (MissingCredential("ausente"), True),
        (ProviderNotInstalled("sdk"), True),
        (FakeLLMExhausted("fim"), True),
        (ConnectionError("rede"), True),
        (TimeoutError("tempo"), True),
        (ConfigError("config"), False),
        (ToolError("ferramenta"), False),
        (ValueError("bug"), False),
    ],
)
def test_transport_classification(exc: BaseException, expected: bool):
    assert is_transport_failure(exc) is expected


def test_provider_sdk_error_counts_as_transport():
    class APIConnectionError(Exception):
        pass

    APIConnectionError.__module__ = "openai"
    assert is_transport_failure(APIConnectionError("rede")) is True


def test_failure_mode_order_is_stable():
    assert FAILURE_MODES == (
        "wrong_tool",
        "bad_arguments",
        "extra_call",
        "missed_call",
        "retrieved_distractor",
        "failed_abstention",
    )


def test_cli_blocks_a_live_provider_without_echoing_a_key(monkeypatch, capsys):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-secret")
    code = main(["--strategy", "react"])
    captured = capsys.readouterr()
    assert code == 2
    assert "bloqueado" in captured.err
    assert "sk-ant-test-secret" not in captured.err
    assert captured.out == ""


def test_allow_live_without_a_key_does_not_invent_a_score(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    dataset = _one_task_file(tmp_path)
    output = tmp_path / "out.jsonl"
    code = main(
        [
            "--strategy",
            "react",
            "--allow-live",
            "--dataset",
            str(dataset),
            "--output",
            str(output),
        ]
    )
    assert code == 0
    row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert row["correct"] is None
    assert "MissingCredential" in row["error"]
    assert "sk-" not in row["error"]
    assert capsys.readouterr().out == ""


def test_cli_rejects_a_bad_dataset(tmp_path, capsys):
    dataset = tmp_path / "bad.jsonl"
    dataset.write_text("{nope\n", encoding="utf-8")
    code = main(["--strategy", "react", "--dataset", str(dataset)])
    assert code == 1
    assert "JSON" in capsys.readouterr().err


def test_cli_unknown_strategy_exits_with_usage_error():
    with pytest.raises(SystemExit) as caught:
        main(["--strategy", "auto"])
    assert caught.value.code == 2


def test_cli_fake_provider_walks_the_dataset_without_inventing_scores(
    tmp_path, capsys, monkeypatch
):
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    output = tmp_path / "records.jsonl"
    code = main(["--strategy", "react", "--output", str(output)])
    assert code == 0
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 45
    assert {row["strategy"] for row in rows} == {"react"}
    assert all(row["correct"] is None for row in rows)
    assert all(row["sequence_match"] is None for row in rows)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "registros=45" in captured.err
    assert "%" not in captured.err


def test_module_entrypoint_runs_offline(tmp_path):
    dataset = _one_task_file(tmp_path)
    output = tmp_path / "out.jsonl"
    env = os.environ.copy()
    env["LLM_PROVIDER"] = "fake"
    env["OPENAI_API_KEY"] = ""
    env["ANTHROPIC_API_KEY"] = ""
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tool_calling_study.evaluate",
            "--strategy",
            "native",
            "--dataset",
            str(dataset),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=_ROOT,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr
    row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert row["correct"] is None
    assert row["strategy"] == "native"
    assert "sk-" not in completed.stderr


def test_evaluate_tools_help_lists_the_strategies():
    script = Path(sys.executable).with_name("evaluate-tools")
    completed = subprocess.run(
        [str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "json_plan" in completed.stdout
    assert "--allow-live" in completed.stdout


def test_readme_documents_the_harness_and_leaves_measured_results_for_later():
    text = (_ROOT / "README.md").read_text(encoding="utf-8")
    assert "python -m tool_calling_study.evaluate" in text
    assert "evaluate-tools" in text
    assert "--strategy" in text
    assert "failed_abstention" in text
    assert "retrieved_distractor" in text
    assert "PR4" in text
    lowered = text.lower()
    for banned in ("100%", "acurácia:", "accuracy:", "custo médio", "usd 0."):
        assert banned not in lowered


def _one_task_file(tmp_path: Path) -> Path:
    line = (_ROOT / "data" / "eval" / "tasks.v1.jsonl").read_text(encoding="utf-8").splitlines()[0]
    path = tmp_path / "one.jsonl"
    path.write_text(line + "\n", encoding="utf-8")
    return path
