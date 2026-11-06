# Esquemas tipados das ferramentas. O estudo mede argumento contra este contrato.
# O schema também vira o payload de function calling, sem um segundo formato paralelo.

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolParameter:
    """Um campo do objeto de argumentos. `type` segue o subconjunto JSON Schema usado aqui."""

    name: str
    type: str
    description: str
    required: bool = True


@dataclass(frozen=True)
class ToolSpec:
    """Contrato público de uma ferramenta: nome estável, descrição e parâmetros."""

    name: str
    description: str
    parameters: tuple[ToolParameter, ...]

    def json_schema(self) -> dict[str, Any]:
        properties: dict[str, Any] = {}
        required: list[str] = []
        for parameter in self.parameters:
            properties[parameter.name] = {
                "type": parameter.type,
                "description": parameter.description,
            }
            if parameter.required:
                required.append(parameter.name)
        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }

    def openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.json_schema(),
            },
        }

    def anthropic_tool(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.json_schema(),
        }

    def render_line(self) -> str:
        params = ", ".join(f"{parameter.name}: {parameter.type}" for parameter in self.parameters)
        return f"- {self.name}: {self.description} Parameters: {params}."


@dataclass(frozen=True)
class ToolCall:
    """Chamada já separada do texto: nome, argumentos e id quando o protocolo exige."""

    name: str
    arguments: Mapping[str, Any]
    call_id: str | None = None


@dataclass(frozen=True)
class ToolResult:
    """Retorno da execução. `ok` falso ainda entra no registro do loop."""

    name: str
    output: str
    ok: bool
