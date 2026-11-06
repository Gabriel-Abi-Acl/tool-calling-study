# Calculadora por AST. Só entram constantes e operadores aritméticos.
# Nome, chamada e atributo são recusados para a ferramenta não virar eval arbitrário.

from __future__ import annotations

import ast
import math
import operator

from tool_calling_study.errors import ToolError

_MAX_EXPRESSION_LENGTH = 100
_MAX_NODES = 40
_MAX_POWER = 8

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}


def evaluate_arithmetic(expression: str) -> int | float:
    """Avalia uma expressão aritmética curta. Fora do subconjunto seguro, levanta `ToolError`."""
    if not isinstance(expression, str) or not expression.strip():
        raise ToolError("empty_expression")
    if len(expression) > _MAX_EXPRESSION_LENGTH:
        raise ToolError("expression_too_long")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolError("invalid_expression") from exc
    if sum(1 for _ in ast.walk(tree)) > _MAX_NODES:
        raise ToolError("expression_too_complex")
    return _eval(tree.body)


def format_number(value: int | float) -> str:
    """Inteiro quando o resultado é integral, para o gabarito não oscilar entre `2` e `2.0`."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _eval(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int | float):
            raise ToolError("invalid_expression")
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd | ast.USub):
        value = _eval(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        left = _eval(node.left)
        right = _eval(node.right)
        if isinstance(node.op, ast.Pow):
            _reject_large_power(right)
        try:
            result = _BINOPS[type(node.op)](left, right)
        except ZeroDivisionError as exc:
            raise ToolError("division_by_zero") from exc
        if isinstance(result, float) and math.isnan(result):
            raise ToolError("invalid_expression")
        return result
    raise ToolError("invalid_expression")


def _reject_large_power(exponent: int | float) -> None:
    if isinstance(exponent, bool) or not isinstance(exponent, int | float):
        raise ToolError("invalid_expression")
    if isinstance(exponent, float) and not exponent.is_integer():
        raise ToolError("invalid_expression")
    if not 0 <= int(exponent) <= _MAX_POWER:
        raise ToolError("exponent_out_of_range")
