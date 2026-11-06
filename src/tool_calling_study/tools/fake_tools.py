# Três ferramentas fixas: consulta, cálculo e clima fictício.
# O clima entra nas tarefas de abstenção como distração; a base não inventa fato ausente.

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

from tool_calling_study.errors import ToolError
from tool_calling_study.tools.arithmetic import evaluate_arithmetic, format_number
from tool_calling_study.tools.registry import RegisteredTool, ToolRegistry
from tool_calling_study.tools.schema import ToolParameter, ToolSpec

# Constantes redondas de fixture. Não são censo nem previsão real.
_KNOWLEDGE = {
    "capital of france": "Paris",
    "capital da franca": "Paris",
    "capital of brazil": "Brasília",
    "capital do brasil": "Brasília",
    "capital of japan": "Tokyo",
    "capital do japao": "Tokyo",
    "population of brazil": "203000000",
    "populacao do brasil": "203000000",
    "population of portugal": "10000000",
    "populacao de portugal": "10000000",
    "population of japan": "124000000",
    "populacao do japao": "124000000",
    # Capitais e populações do dataset v1. O número é constante redonda do estudo, não censo.
    "capital of germany": "Berlin",
    "capital da alemanha": "Berlin",
    "capital of italy": "Rome",
    "capital da italia": "Rome",
    "capital of spain": "Madrid",
    "capital da espanha": "Madrid",
    "population of germany": "80000000",
    "populacao da alemanha": "80000000",
    "population of italy": "60000000",
    "populacao da italia": "60000000",
    "population of spain": "50000000",
    "populacao da espanha": "50000000",
    "population of canada": "40000000",
    "populacao do canada": "40000000",
    "population of mexico": "120000000",
    "populacao do mexico": "120000000",
    "population of egypt": "100000000",
    "populacao do egito": "100000000",
    "population of argentina": "45000000",
    "populacao da argentina": "45000000",
    "population of france": "70000000",
    "populacao da franca": "70000000",
}

_FORECASTS = {
    "sao paulo": "ensolarado, 28C [fixture]",
    "curitiba": "nublado, 18C [fixture]",
}


def build_fake_registry(names: Sequence[str] | None = None) -> ToolRegistry:
    """Catálogo falso. Com `names`, só essas ferramentas aparecem, na ordem pedida."""
    tools = (lookup_tool(), calculator_tool(), weather_tool())
    if names is None:
        return ToolRegistry(tools)
    by_name = {tool.spec.name: tool for tool in tools}
    selected = []
    for name in names:
        tool = by_name.get(name)
        if tool is None:
            raise ToolError(f"ferramenta desconhecida: {name}")
        selected.append(tool)
    return ToolRegistry(tuple(selected))


def lookup_tool() -> RegisteredTool:
    spec = ToolSpec(
        name="lookup",
        description=(
            "Look up a fact in the fixed synthetic knowledge base. "
            "Unknown queries return not_found. Do not use for opinions or the weather."
        ),
        parameters=(
            ToolParameter(
                name="query",
                type="string",
                description="Canonical fact key, for example 'capital of France'.",
            ),
        ),
    )
    return RegisteredTool(spec, _lookup)


def calculator_tool() -> RegisteredTool:
    spec = ToolSpec(
        name="calculator",
        description="Evaluate one safe arithmetic expression using +, -, *, /, //, % and **.",
        parameters=(
            ToolParameter(
                name="expression",
                type="string",
                description="Arithmetic expression. Names and function calls are rejected.",
            ),
        ),
    )
    return RegisteredTool(spec, _calculate)


def weather_tool() -> RegisteredTool:
    spec = ToolSpec(
        name="weather",
        description=(
            "Return the fictional fixed forecast for a city in this study. "
            "Do not call it unless the question asks for that forecast."
        ),
        parameters=(
            ToolParameter(
                name="city",
                type="string",
                description="City name, for example 'Sao Paulo'.",
            ),
        ),
    )
    return RegisteredTool(spec, _weather)


def _lookup(arguments: dict[str, object]) -> str:
    query = arguments["query"]
    if not isinstance(query, str):
        raise ToolError("bad_type:query")
    return _KNOWLEDGE.get(_normalize(query), "not_found")


def _calculate(arguments: dict[str, object]) -> str:
    expression = arguments["expression"]
    if not isinstance(expression, str):
        raise ToolError("bad_type:expression")
    return format_number(evaluate_arithmetic(expression))


def _weather(arguments: dict[str, object]) -> str:
    city = arguments["city"]
    if not isinstance(city, str):
        raise ToolError("bad_type:city")
    return _FORECASTS.get(_normalize(city), "not_found")


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.strip().lower())
    without_accents = "".join(char for char in decomposed if not unicodedata.combining(char))
    cleaned = re.sub(r"[^a-z0-9]+", " ", without_accents)
    return re.sub(r"\s+", " ", cleaned).strip()
