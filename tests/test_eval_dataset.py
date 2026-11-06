# O dataset v1 fecha o gabarito: schema, contagem por categoria e sequência executável.
# Estes testes não medem modelo e não publicam acurácia.

import json
from collections import Counter
from pathlib import Path

import pytest

from tool_calling_study.tasks import EvalDatasetError, load_eval_tasks
from tool_calling_study.tools.fake_tools import build_fake_registry
from tool_calling_study.tools.schema import ToolCall

_ALLOWED = ["lookup", "calculator", "weather"]


def test_loader_accepts_one_fixture_per_category(tmp_path):
    tasks = load_eval_tasks(
        _write(
            tmp_path,
            _single_lookup(),
            _single_calculator(),
            _multi(),
            _abstain_text(),
            _abstain_refuse(),
        )
    )
    by_category = {task.category for task in tasks}
    by_id = {task.id: task for task in tasks}
    assert by_category == {"single", "multi", "abstain"}
    assert by_id["single-lookup-capital-fr"].expected_sequence[0].name == "lookup"
    assert by_id["single-lookup-capital-fr"].expected_sequence[0].argument_constraints == {
        "query": "capital of France"
    }
    assert by_id["single-calc-product"].expected_sequence[0].name == "calculator"
    assert by_id["single-calc-product"].reference_answer == "391"
    assert by_id["multi-br-millions"].tool_call_allowed is True
    assert by_id["multi-br-millions"].reference_answer == "203"
    assert by_id["abstain-ola"].tool_call_allowed is False
    assert by_id["abstain-ola"].expected_sequence == ()
    assert by_id["abstain-ola"].reference_answer == "Olá."
    assert by_id["abstain-refuse-secret"].reference_answer is None
    assert "weather" in by_id["abstain-refuse-secret"].allowed_tools


def test_loader_rejects_invalid_json_empty_file_and_non_object(tmp_path):
    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text("{nope\n", encoding="utf-8")
    array = tmp_path / "array.jsonl"
    array.write_text("[]\n", encoding="utf-8")
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n\n", encoding="utf-8")
    missing = tmp_path / "missing.jsonl"
    for path in (invalid, array, empty, missing):
        with pytest.raises(EvalDatasetError):
            load_eval_tasks(path)


def test_loader_rejects_missing_fields_unknown_keys_and_duplicate_ids(tmp_path):
    missing = _single_lookup()
    del missing["prompt"]
    extra = _single_lookup()
    extra["accuracy"] = 1
    blank = _single_lookup()
    blank["prompt"] = "   "
    bad_id = _single_lookup()
    bad_id["id"] = "Capital FR"
    bad_flag = _single_lookup()
    bad_flag["tool_call_allowed"] = "true"
    for row in (missing, extra, blank, bad_id, bad_flag):
        with pytest.raises(EvalDatasetError):
            load_eval_tasks(_write(tmp_path, row))
    with pytest.raises(EvalDatasetError):
        load_eval_tasks(_write(tmp_path, _single_lookup(), _single_lookup()))


def test_loader_rejects_arguments_outside_the_fake_schema(tmp_path):
    extra_arg = _single_lookup()
    extra_arg["expected_sequence"] = [
        {
            "name": "lookup",
            "argument_constraints": {"query": "capital of France", "extra": "1"},
        }
    ]
    wrong_key = _single_lookup()
    wrong_key["expected_sequence"] = [
        {"name": "lookup", "argument_constraints": {"city": "Sao Paulo"}}
    ]
    bad_type = _single_lookup()
    bad_type["expected_sequence"] = [{"name": "lookup", "argument_constraints": {"query": 12}}]
    weather = _single_lookup()
    weather["expected_sequence"] = [
        {"name": "weather", "argument_constraints": {"city": "Sao Paulo"}}
    ]
    unknown = _single_lookup()
    unknown["allowed_tools"] = ["lookup", "telescope"]
    duplicate_tools = _single_lookup()
    duplicate_tools["allowed_tools"] = ["lookup", "lookup"]
    mars = _single_lookup()
    mars["expected_sequence"] = [
        {"name": "lookup", "argument_constraints": {"query": "capital of Mars"}}
    ]
    mars["reference_answer"] = "not_found"
    outside = _single_lookup()
    outside["allowed_tools"] = ["calculator", "weather"]
    for row in (
        extra_arg,
        wrong_key,
        bad_type,
        weather,
        unknown,
        duplicate_tools,
        mars,
        outside,
    ):
        with pytest.raises(EvalDatasetError):
            load_eval_tasks(_write(tmp_path, row))


def test_loader_rejects_bad_category_shape_and_reference(tmp_path):
    two_steps = _single_lookup()
    two_steps["expected_sequence"] = [
        *_single_lookup()["expected_sequence"],
        {"name": "calculator", "argument_constraints": {"expression": "1 + 1"}},
    ]
    calls_forbidden = _single_lookup()
    calls_forbidden["tool_call_allowed"] = False
    null_reference = _single_lookup()
    null_reference["reference_answer"] = None
    wrong_reference = _single_lookup()
    wrong_reference["reference_answer"] = "Lyon"
    bad_category = _single_lookup()
    bad_category["category"] = "both"
    broken_chain = _multi()
    broken_chain["expected_sequence"][1] = {
        "name": "calculator",
        "argument_constraints": {"expression": "1 + 1"},
    }
    abstain_calls = _abstain_text()
    abstain_calls["expected_sequence"] = _single_lookup()["expected_sequence"]
    for row in (
        two_steps,
        calls_forbidden,
        null_reference,
        wrong_reference,
        bad_category,
        broken_chain,
        abstain_calls,
    ):
        with pytest.raises(EvalDatasetError):
            load_eval_tasks(_write(tmp_path, row))


def test_shipped_dataset_is_balanced_across_categories():
    tasks = load_eval_tasks()
    counts = Counter(task.category for task in tasks)
    assert 40 <= len(tasks) <= 50
    assert set(counts) == {"single", "multi", "abstain"}
    assert all(count >= 12 for count in counts.values())
    assert max(counts.values()) - min(counts.values()) <= 1
    assert len({task.id for task in tasks}) == len(tasks)
    assert len({task.prompt for task in tasks}) == len(tasks)
    single_tools = Counter(
        task.expected_sequence[0].name for task in tasks if task.category == "single"
    )
    assert set(single_tools) <= {"lookup", "calculator"}
    assert single_tools["lookup"] >= 5
    assert single_tools["calculator"] >= 5
    assert abs(single_tools["lookup"] - single_tools["calculator"]) <= 2


def test_each_shipped_category_has_a_task_and_weather_is_only_a_distractor():
    tasks = load_eval_tasks()
    abstain = [task for task in tasks if task.category == "abstain"]
    assert any(task.category == "single" for task in tasks)
    assert any(task.category == "multi" for task in tasks)
    assert abstain
    assert any(task.reference_answer is None for task in abstain)
    assert any(task.reference_answer for task in abstain)
    for task in tasks:
        assert task.allowed_tools == ("lookup", "calculator", "weather")
        assert all(step.name != "weather" for step in task.expected_sequence)
    for task in abstain:
        assert task.tool_call_allowed is False
        assert task.expected_sequence == ()


def test_expected_sequences_match_fake_tool_schemas():
    registry = build_fake_registry()
    specs = {spec.name: spec for spec in registry.specs()}
    for task in load_eval_tasks():
        previous: str | None = None
        for step in task.expected_sequence:
            schema = specs[step.name].json_schema()
            assert set(step.argument_constraints) == set(schema["properties"])
            assert set(schema["required"]) <= set(step.argument_constraints)
            for key, value in step.argument_constraints.items():
                assert _matches_json_type(value, schema["properties"][key]["type"]), task.id
            if previous is not None:
                assert any(
                    isinstance(value, str) and previous in value
                    for value in step.argument_constraints.values()
                ), task.id
            result = registry.execute(ToolCall(step.name, step.argument_constraints))
            assert result.ok, task.id
            assert result.output != "not_found"
            previous = result.output
        if task.tool_call_allowed:
            assert task.reference_answer == previous
        else:
            assert previous is None


def test_eval_file_is_versioned_jsonl_and_stays_synthetic():
    path = Path("data/eval/tasks.v1.jsonl")
    text = path.read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if line.strip()]
    assert path.name == "tasks.v1.jsonl"
    assert lines
    assert text.endswith("\n")
    for line in lines:
        assert isinstance(json.loads(line), dict)
    lowered = text.lower()
    for token in ("outsmart", "api_key", "sk-", "faturamento", "accuracy", "acuracia"):
        assert token not in lowered


def _single_lookup() -> dict:
    return {
        "id": "single-lookup-capital-fr",
        "category": "single",
        "prompt": "Consulte a base sintética: qual é a capital da França?",
        "allowed_tools": list(_ALLOWED),
        "tool_call_allowed": True,
        "expected_sequence": [
            {"name": "lookup", "argument_constraints": {"query": "capital of France"}}
        ],
        "reference_answer": "Paris",
    }


def _single_calculator() -> dict:
    return {
        "id": "single-calc-product",
        "category": "single",
        "prompt": "Quanto é 17 * 23? Use a calculadora.",
        "allowed_tools": list(_ALLOWED),
        "tool_call_allowed": True,
        "expected_sequence": [
            {"name": "calculator", "argument_constraints": {"expression": "17 * 23"}}
        ],
        "reference_answer": "391",
    }


def _multi() -> dict:
    return {
        "id": "multi-br-millions",
        "category": "multi",
        "prompt": (
            "Qual é a população do Brasil no fixture, em milhões? "
            "Consulte a base e converta com a calculadora."
        ),
        "allowed_tools": list(_ALLOWED),
        "tool_call_allowed": True,
        "expected_sequence": [
            {"name": "lookup", "argument_constraints": {"query": "population of Brazil"}},
            {
                "name": "calculator",
                "argument_constraints": {"expression": "203000000 / 1000000"},
            },
        ],
        "reference_answer": "203",
    }


def _abstain_text() -> dict:
    return {
        "id": "abstain-ola",
        "category": "abstain",
        "prompt": "Sem usar ferramentas, responda apenas: Olá.",
        "allowed_tools": list(_ALLOWED),
        "tool_call_allowed": False,
        "expected_sequence": [],
        "reference_answer": "Olá.",
    }


def _abstain_refuse() -> dict:
    return {
        "id": "abstain-refuse-secret",
        "category": "abstain",
        "prompt": "Não chame ferramentas e não invente segredo. Recuse o pedido de uma senha.",
        "allowed_tools": list(_ALLOWED),
        "tool_call_allowed": False,
        "expected_sequence": [],
        "reference_answer": None,
    }


def _write(tmp_path: Path, *rows: dict) -> Path:
    path = tmp_path / "tasks.v1.jsonl"
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _matches_json_type(value: object, json_type: str) -> bool:
    if json_type == "string":
        return isinstance(value, str)
    if json_type == "boolean":
        return isinstance(value, bool)
    if json_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if json_type == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    return False
