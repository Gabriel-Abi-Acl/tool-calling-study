# Contrato do parser ReAct: um Action com objeto JSON, ou um Final Answer.
# Formato misto ou argumento solto é erro, para a métrica de formato não virar chute.

import pytest

from tool_calling_study.errors import ParseError
from tool_calling_study.strategies.react import parse_react_trace


def test_parse_react_action_with_json_object():
    decision = parse_react_trace(
        "\n".join(
            [
                "Thought: preciso da capital.",
                "Action: lookup",
                'Action Input: {"query": "capital of France"}',
            ]
        )
    )
    assert decision.final_answer is None
    assert len(decision.tool_calls) == 1
    assert decision.tool_calls[0].name == "lookup"
    assert dict(decision.tool_calls[0].arguments) == {"query": "capital of France"}


def test_parse_react_action_input_over_multiple_lines():
    decision = parse_react_trace(
        "\n".join(
            [
                "Thought: consultar.",
                "Action: lookup",
                "Action Input: {",
                '  "query": "capital of France"',
                "}",
            ]
        )
    )
    assert dict(decision.tool_calls[0].arguments) == {"query": "capital of France"}


def test_parse_react_final_answer():
    decision = parse_react_trace("Thought: já sei.\nFinal Answer: Paris")
    assert decision.tool_calls == ()
    assert decision.final_answer == "Paris"


def test_parse_react_rejects_action_without_input():
    with pytest.raises(ParseError):
        parse_react_trace("Thought: vou chamar.\nAction: lookup")


def test_parse_react_rejects_mixed_action_and_final_answer():
    text = "\n".join(
        [
            "Action: lookup",
            'Action Input: {"query": "capital of France"}',
            "Final Answer: Paris",
        ]
    )
    with pytest.raises(ParseError):
        parse_react_trace(text)


def test_parse_react_rejects_non_object_arguments():
    with pytest.raises(ParseError):
        parse_react_trace("Action: calculator\nAction Input: 17 * 23")
