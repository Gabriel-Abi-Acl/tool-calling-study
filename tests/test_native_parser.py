# Contrato do payload nativo: arguments em string JSON ou objeto, e o envelope de choices.
# Sem tool_calls o texto em content é a resposta; arguments quebrado não vira objeto vazio.

import json

import pytest

from tool_calling_study.errors import ParseError
from tool_calling_study.strategies.native import parse_native_payload


def test_parse_native_payload_with_string_arguments():
    payload = {
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "calculator",
                    "arguments": json.dumps({"expression": "17 * 23"}),
                },
            }
        ]
    }
    parsed = parse_native_payload(payload)
    assert parsed.content == ""
    assert parsed.tool_calls[0].name == "calculator"
    assert parsed.tool_calls[0].call_id == "call_1"
    assert dict(parsed.tool_calls[0].arguments) == {"expression": "17 * 23"}


def test_parse_native_payload_with_object_arguments():
    payload = {
        "tool_calls": [
            {"name": "lookup", "arguments": {"query": "capital of Brazil"}},
        ]
    }
    parsed = parse_native_payload(json.dumps(payload))
    assert parsed.tool_calls[0].call_id == "call_1"
    assert dict(parsed.tool_calls[0].arguments) == {"query": "capital of Brazil"}


def test_parse_native_payload_reads_openai_choices_envelope():
    payload = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_9",
                            "function": {
                                "name": "weather",
                                "arguments": {"city": "Sao Paulo"},
                            },
                        }
                    ],
                }
            }
        ]
    }
    parsed = parse_native_payload(payload)
    assert parsed.tool_calls[0].name == "weather"
    assert parsed.tool_calls[0].call_id == "call_9"


def test_parse_native_payload_without_tool_calls_keeps_content():
    parsed = parse_native_payload({"role": "assistant", "content": "391", "tool_calls": []})
    assert parsed.tool_calls == ()
    assert parsed.content == "391"


def test_parse_native_payload_rejects_invalid_argument_json():
    payload = {
        "tool_calls": [
            {"id": "call_1", "function": {"name": "calculator", "arguments": "{not-json"}},
        ]
    }
    with pytest.raises(ParseError):
        parse_native_payload(payload)


def test_parse_native_payload_rejects_argument_list():
    payload = {"tool_calls": [{"name": "lookup", "arguments": ["capital of France"]}]}
    with pytest.raises(ParseError):
        parse_native_payload(payload)
