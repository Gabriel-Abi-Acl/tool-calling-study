# Chamada desnecessária: ferramenta proibida ou ausente do gabarito, na ordem da execução.
# Repetir a ferramenta esperada fica de fora desta checagem.

from tool_calling_study.checks import find_unnecessary_calls
from tool_calling_study.tasks import get_task


def test_flags_weather_when_the_task_only_expects_lookup():
    task = get_task("single-capital-fr")
    assert find_unnecessary_calls(["lookup", "weather"], task) == ["weather"]


def test_flags_every_call_on_an_abstain_task():
    task = get_task("abstain-python-function")
    assert find_unnecessary_calls(["weather", "lookup"], task) == ["weather", "lookup"]


def test_expected_tools_only_are_not_unnecessary():
    task = get_task("multi-br-millions")
    assert find_unnecessary_calls(["lookup", "calculator"], task) == []


def test_repeating_an_expected_tool_is_not_an_unnecessary_call():
    task = get_task("single-capital-br")
    assert find_unnecessary_calls(["lookup", "lookup"], task) == []


def test_weather_task_allows_weather_and_flags_calculator():
    task = get_task("single-weather-sp")
    assert find_unnecessary_calls(["weather"], task) == []
    assert find_unnecessary_calls(["calculator"], task) == ["calculator"]
