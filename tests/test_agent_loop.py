# O loop aplica a estratégia, executa o catálogo falso e grava ferramenta, argumentos e resposta.
# O cliente é roteirizado: estes testes não medem um modelo e não abrem rede.

import json

from tool_calling_study.agent import run_task
from tool_calling_study.llm.base import LLMResponse
from tool_calling_study.llm.fake import FakeLLM
from tool_calling_study.strategies import (
    JsonPlanStrategy,
    NativeToolStrategy,
    ReactToolStrategy,
    build_strategy,
)
from tool_calling_study.tasks import get_task
from tool_calling_study.tools.fake_tools import build_fake_registry
from tool_calling_study.tools.schema import ToolCall


def test_react_loop_records_tool_arguments_and_final_answer():
    task = get_task("single-capital-fr")
    llm = FakeLLM(
        [
            LLMResponse(
                content="\n".join(
                    [
                        "Thought: consultar a base.",
                        "Action: lookup",
                        'Action Input: {"query": "capital of France"}',
                    ]
                )
            ),
            LLMResponse(content="Thought: a base respondeu.\nFinal Answer: Paris"),
        ]
    )
    run = run_task(task, ReactToolStrategy(llm), build_fake_registry())
    assert run.selected_tools == ("lookup",)
    assert dict(run.calls[0].call.arguments) == {"query": "capital of France"}
    assert run.calls[0].output == "Paris"
    assert run.calls[0].ok is True
    assert run.final_answer == "Paris"
    assert run.unnecessary_calls == ()
    assert run.stopped_reason == "final"
    assert llm.calls[0].tool_names is None
    assert any(message.content == "Observation: Paris" for message in llm.calls[1].messages)


def test_json_plan_loop_runs_two_tools_then_asks_for_the_answer():
    task = get_task("multi-br-millions")
    plan = {
        "steps": [
            {"tool": "lookup", "arguments": {"query": "population of Brazil"}},
            {"tool": "calculator", "arguments": {"expression": "203000000 / 1000000"}},
        ]
    }
    final = {"steps": [], "final_answer": "203"}
    llm = FakeLLM(
        [
            LLMResponse(content=json.dumps(plan)),
            LLMResponse(content=json.dumps(final)),
        ]
    )
    run = run_task(task, JsonPlanStrategy(llm), build_fake_registry())
    assert run.selected_tools == ("lookup", "calculator")
    assert run.calls[0].output == "203000000"
    assert dict(run.calls[1].call.arguments) == {"expression": "203000000 / 1000000"}
    assert run.calls[1].output == "203"
    assert run.final_answer == "203"
    assert run.unnecessary_calls == ()
    assert llm.calls[0].tool_names is None
    assert "population of Brazil" in llm.calls[1].messages[2].content
    assert "203000000" in llm.calls[1].messages[-1].content


def test_native_loop_executes_structured_tool_call():
    task = get_task("single-product")
    llm = FakeLLM(
        [
            LLMResponse(
                content="",
                tool_calls=(ToolCall("calculator", {"expression": "17 * 23"}, call_id="call_1"),),
            ),
            LLMResponse(content="391"),
        ]
    )
    run = run_task(task, NativeToolStrategy(llm), build_fake_registry())
    assert run.selected_tools == ("calculator",)
    assert dict(run.calls[0].call.arguments) == {"expression": "17 * 23"}
    assert run.calls[0].output == "391"
    assert run.final_answer == "391"
    assert run.unnecessary_calls == ()
    assert llm.calls[0].tool_names == ("lookup", "calculator", "weather")


def test_native_strategy_reads_tool_payload_from_message_content():
    task = get_task("single-product")
    payload = {
        "tool_calls": [
            {
                "id": "call_1",
                "function": {
                    "name": "calculator",
                    "arguments": json.dumps({"expression": "17 * 23"}),
                },
            }
        ]
    }
    llm = FakeLLM([LLMResponse(content=json.dumps(payload)), LLMResponse(content="391")])
    run = run_task(task, NativeToolStrategy(llm), build_fake_registry())
    assert run.selected_tools == ("calculator",)
    assert run.final_answer == "391"


def test_abstain_loop_records_no_tool_call():
    task = get_task("abstain-greeting")
    llm = FakeLLM([LLMResponse(content="Thought: saudação.\nFinal Answer: Olá.")])
    run = run_task(task, ReactToolStrategy(llm), build_fake_registry())
    assert run.calls == ()
    assert run.final_answer == "Olá."
    assert run.unnecessary_calls == ()
    assert run.stopped_reason == "final"


def test_loop_marks_weather_call_as_unnecessary_on_a_lookup_task():
    task = get_task("single-capital-fr")
    llm = FakeLLM(
        [
            LLMResponse(
                content="\n".join(
                    [
                        "Thought: vou olhar o clima.",
                        "Action: weather",
                        'Action Input: {"city": "Sao Paulo"}',
                    ]
                )
            ),
            LLMResponse(content="Thought: isso não responde a capital.\nFinal Answer: não sei"),
        ]
    )
    run = run_task(task, ReactToolStrategy(llm), build_fake_registry())
    assert run.selected_tools == ("weather",)
    assert run.calls[0].output == "ensolarado, 28C [fixture]"
    assert run.unnecessary_calls == ("weather",)
    assert run.final_answer == "não sei"


def test_loop_stops_on_parse_error_without_raising():
    task = get_task("abstain-repeat")
    llm = FakeLLM([LLMResponse(content="resposta sem o formato pedido")])
    run = run_task(task, ReactToolStrategy(llm), build_fake_registry())
    assert run.stopped_reason == "parse_error"
    assert run.calls == ()
    assert run.final_answer is None


def test_loop_stops_at_max_turns_when_the_model_keeps_calling_tools():
    task = get_task("abstain-preference")
    call = LLMResponse(
        content="\n".join(
            [
                "Thought: de novo.",
                "Action: weather",
                'Action Input: {"city": "Curitiba"}',
            ]
        )
    )
    llm = FakeLLM([call, call])
    run = run_task(task, ReactToolStrategy(llm), build_fake_registry(), max_turns=2)
    assert run.stopped_reason == "max_turns"
    assert run.selected_tools == ("weather", "weather")
    assert run.unnecessary_calls == ("weather", "weather")
    assert run.final_answer is None


def test_build_strategy_wires_the_three_names():
    llm = FakeLLM([])
    assert build_strategy("react", llm).name == "react"
    assert build_strategy("native", llm).name == "native"
    assert build_strategy("json_plan", llm).name == "json_plan"
