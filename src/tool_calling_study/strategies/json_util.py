# Extração de um documento JSON, com ou sem cerca markdown.
# Os parsers param no primeiro erro de formato; não tentam adivinhar o objeto.

from __future__ import annotations

import json
import re
from typing import Any

from tool_calling_study.errors import ParseError

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def loads_json_document(text: str) -> Any:
    """Lê um JSON object/array. Aceita cerca ```json e um objeto cercado por texto curto."""
    if not text or not text.strip():
        raise ParseError("JSON ausente")
    candidate = text.strip()
    fenced = _FENCE.search(candidate)
    if fenced:
        candidate = fenced.group(1).strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end <= start:
            raise ParseError("JSON ausente") from None
        try:
            return json.loads(candidate[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ParseError("JSON inválido") from exc


def require_object(value: Any, *, message: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ParseError(message)
    return value


def coerce_arguments(value: Any) -> dict[str, Any]:
    """Aceita objeto ou string JSON de objeto. Lista e escalar são erro de argumento."""
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        loaded = loads_json_document(value)
        return require_object(loaded, message="argumentos precisam ser um objeto JSON")
    raise ParseError("argumentos precisam ser um objeto JSON")
