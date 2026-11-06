# Executa uma chamada contra o catálogo falso. Ferramenta desconhecida vira resultado, não exceção.
# O loop precisa do registro mesmo quando a seleção está errada, para contar a chamada.

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from tool_calling_study.errors import ToolError
from tool_calling_study.tools.schema import ToolCall, ToolResult, ToolSpec

ToolHandler = Callable[[Mapping[str, Any]], str]


class RegisteredTool:
    """Uma ferramenta do catálogo: esquema mais a função pura que produz o texto."""

    def __init__(self, spec: ToolSpec, handler: ToolHandler) -> None:
        self.spec = spec
        self._handler = handler

    def run(self, arguments: Mapping[str, Any]) -> str:
        return self._handler(arguments)


class ToolRegistry:
    """Catálogo usado pelo loop. A ordem de `specs` é a ordem de registro."""

    def __init__(self, tools: Sequence[RegisteredTool]) -> None:
        self._tools = {tool.spec.name: tool for tool in tools}

    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(tool.spec for tool in self._tools.values())

    def execute(self, call: ToolCall) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(name=call.name, output=f"unknown_tool:{call.name}", ok=False)
        error = _validate(tool.spec, call.arguments)
        if error is not None:
            return ToolResult(name=call.name, output=error, ok=False)
        try:
            output = tool.run(call.arguments)
        except ToolError as exc:
            return ToolResult(name=call.name, output=str(exc), ok=False)
        return ToolResult(name=call.name, output=output, ok=True)


def _validate(spec: ToolSpec, arguments: Mapping[str, Any]) -> str | None:
    if not isinstance(arguments, Mapping):
        return "arguments_not_object"
    known = {parameter.name: parameter for parameter in spec.parameters}
    missing = [
        name for name, parameter in known.items() if parameter.required and name not in arguments
    ]
    if missing:
        return "missing_argument:" + ",".join(missing)
    unexpected = sorted(set(arguments) - set(known))
    if unexpected:
        return "unexpected_argument:" + ",".join(unexpected)
    for name, parameter in known.items():
        if name not in arguments:
            continue
        if not _type_matches(arguments[name], parameter.type):
            return f"bad_type:{name}"
    return None


def _type_matches(value: Any, json_type: str) -> bool:
    if json_type == "string":
        return isinstance(value, str)
    if json_type == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if json_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if json_type == "boolean":
        return isinstance(value, bool)
    return False
