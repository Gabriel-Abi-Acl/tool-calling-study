# Estratégias de tool use. Cada uma fala com o mesmo `LLMClient` e devolve um `Decision`.

from tool_calling_study.strategies.base import Decision, PriorTurn, ToolStrategy, build_strategy
from tool_calling_study.strategies.json_plan import JsonPlanStrategy, parse_json_plan
from tool_calling_study.strategies.native import NativeToolStrategy, parse_native_payload
from tool_calling_study.strategies.react import ReactToolStrategy, parse_react_trace

__all__ = [
    "Decision",
    "JsonPlanStrategy",
    "NativeToolStrategy",
    "PriorTurn",
    "ReactToolStrategy",
    "ToolStrategy",
    "build_strategy",
    "parse_json_plan",
    "parse_native_payload",
    "parse_react_trace",
]
