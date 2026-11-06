# Ferramentas falsas: a base e o clima são tabelas fixas; a calculadora só avalia aritmética.
# O gabarito da fixture tem de bater com o retorno real dessas funções.

import pytest

from tool_calling_study.errors import ToolError
from tool_calling_study.tasks import load_tasks
from tool_calling_study.tools.arithmetic import evaluate_arithmetic
from tool_calling_study.tools.fake_tools import build_fake_registry
from tool_calling_study.tools.schema import ToolCall


def test_calculator_and_lookup_return_fixture_values():
    registry = build_fake_registry()
    capital = registry.execute(ToolCall("lookup", {"query": "capital of France"}))
    brasil = registry.execute(ToolCall("lookup", {"query": "capital do Brasil"}))
    product = registry.execute(ToolCall("calculator", {"expression": "17 * 23"}))
    forecast = registry.execute(ToolCall("weather", {"city": "São Paulo"}))
    assert capital.ok and capital.output == "Paris"
    assert brasil.ok and brasil.output == "Brasília"
    assert product.ok and product.output == "391"
    assert forecast.ok and forecast.output == "ensolarado, 28C [fixture]"


def test_unknown_query_and_unknown_tool_do_not_invent_a_value():
    registry = build_fake_registry()
    missing = registry.execute(ToolCall("lookup", {"query": "capital of Mars"}))
    unknown = registry.execute(ToolCall("telescope", {"query": "x"}))
    assert missing.ok and missing.output == "not_found"
    assert unknown.ok is False
    assert unknown.output == "unknown_tool:telescope"


def test_registry_rejects_missing_and_unexpected_arguments():
    registry = build_fake_registry()
    missing = registry.execute(ToolCall("lookup", {}))
    extra = registry.execute(ToolCall("lookup", {"query": "capital of France", "extra": "1"}))
    assert missing.ok is False
    assert extra.ok is False


def test_calculator_rejects_names_calls_and_large_powers():
    with pytest.raises(ToolError):
        evaluate_arithmetic("__import__('os')")
    with pytest.raises(ToolError):
        evaluate_arithmetic("city")
    with pytest.raises(ToolError):
        evaluate_arithmetic("2 ** 20")
    with pytest.raises(ToolError):
        evaluate_arithmetic("1 / 0")


def test_reference_answers_match_the_fake_tool_chain():
    registry = build_fake_registry()
    for task in load_tasks():
        if task.kind == "abstain":
            assert task.reference_answer is None
            continue
        last_output = None
        for index, expected in enumerate(task.expected_calls):
            result = registry.execute(ToolCall(expected.tool, expected.arguments))
            assert result.ok, task.id
            last_output = result.output
            if index == 0 and len(task.expected_calls) == 2:
                expression = task.expected_calls[1].arguments["expression"]
                assert isinstance(expression, str)
                assert result.output in expression
        assert last_output == task.reference_answer
