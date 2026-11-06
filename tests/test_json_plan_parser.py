# Contrato do plano JSON: lista de passos com argumentos objeto, ou abstenção com resposta.
# Cerca markdown é aceita; plano sem steps e sem resposta é erro.

import json

import pytest

from tool_calling_study.errors import ParseError
from tool_calling_study.llm.base import LLMResponse
from tool_calling_study.llm.fake import FakeLLM
from tool_calling_study.strategies.json_plan import JsonPlanStrategy, parse_json_plan
from tool_calling_study.tools.fake_tools import build_fake_registry


def test_parse_json_plan_with_two_steps():
    text = json.dumps(
        {
            "steps": [
                {"tool": "lookup", "arguments": {"query": "population of Brazil"}},
                {"tool": "calculator", "arguments": {"expression": "203000000 / 1000000"}},
            ]
        }
    )
    plan = parse_json_plan(text)
    assert [step.name for step in plan.steps] == ["lookup", "calculator"]
    assert dict(plan.steps[1].arguments) == {"expression": "203000000 / 1000000"}
    assert plan.final_answer is None


def test_parse_json_plan_inside_fence():
    text = """```json
{"steps": [], "final_answer": "Olá."}
```"""
    plan = parse_json_plan(text)
    assert plan.steps == ()
    assert plan.final_answer == "Olá."


def test_parse_json_plan_abstain_requires_final_answer():
    with pytest.raises(ParseError):
        parse_json_plan('{"steps": []}')


def test_parse_json_plan_rejects_step_without_arguments():
    with pytest.raises(ParseError):
        parse_json_plan('{"steps": [{"tool": "lookup"}]}')


def test_json_plan_strategy_holds_final_answer_until_steps_run():
    payload = {
        "steps": [{"tool": "lookup", "arguments": {"query": "capital of France"}}],
        "final_answer": "não usar esta resposta ainda",
    }
    llm = FakeLLM([LLMResponse(content=json.dumps(payload))])
    decision = JsonPlanStrategy(llm).decide(
        "Qual é a capital da França?",
        [],
        build_fake_registry().specs(),
    )
    assert decision.final_answer is None
    assert decision.tool_calls[0].name == "lookup"
    assert llm.calls[0].tool_names is None
