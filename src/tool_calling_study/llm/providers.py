# Adaptadores OpenAI e Anthropic. O SDK entra só na primeira chamada ao vivo.
# A suíte cobre a conversão de mensagens e a falha sem chave ou sem pacote; não chama a API.

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from tool_calling_study.config import Settings
from tool_calling_study.errors import MissingCredential, ParseError, ProviderNotInstalled
from tool_calling_study.llm.base import LLMMessage, LLMResponse
from tool_calling_study.strategies.native import parse_native_payload
from tool_calling_study.tools.schema import ToolCall, ToolSpec

_MAX_TOKENS = 1024


class OpenAIClient:
    """Chat Completions com function calling. A chave vazia não chega a importar o SDK."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[ToolSpec] | None = None,
    ) -> LLMResponse:
        if not self._settings.openai_api_key:
            raise MissingCredential("OPENAI_API_KEY ausente")
        openai = _import_module("openai")
        client = openai.OpenAI(
            api_key=self._settings.openai_api_key,
            timeout=self._settings.llm_timeout_s,
        )
        kwargs: dict[str, Any] = {
            "model": self._settings.openai_model,
            "messages": to_openai_messages(messages),
        }
        if tools is not None:
            kwargs["tools"] = [spec.openai_tool() for spec in tools]
        raw = client.chat.completions.create(**kwargs)
        choices = getattr(raw, "choices", None) or []
        if not choices:
            raise ParseError("resposta OpenAI sem choices")
        return parse_native_payload(_openai_message_payload(choices[0].message)).as_response()


class AnthropicClient:
    """Messages API com tool use. Mesma regra de import tardio e chave obrigatória."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[ToolSpec] | None = None,
    ) -> LLMResponse:
        if not self._settings.anthropic_api_key:
            raise MissingCredential("ANTHROPIC_API_KEY ausente")
        anthropic = _import_module("anthropic")
        client = anthropic.Anthropic(
            api_key=self._settings.anthropic_api_key,
            timeout=self._settings.llm_timeout_s,
        )
        system, payload = to_anthropic_messages(messages)
        kwargs: dict[str, Any] = {
            "model": self._settings.anthropic_model,
            "max_tokens": _MAX_TOKENS,
            "messages": payload,
        }
        if system:
            kwargs["system"] = system
        if tools is not None:
            kwargs["tools"] = [spec.anthropic_tool() for spec in tools]
        raw = client.messages.create(**kwargs)
        if getattr(raw, "content", None) is None:
            raise ParseError("resposta Anthropic sem content")
        return parse_native_payload(_anthropic_response_payload(raw)).as_response()


def to_openai_messages(messages: Sequence[LLMMessage]) -> list[dict[str, Any]]:
    """Projeta o histórico interno no formato de Chat Completions."""
    payload: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "tool":
            if not message.tool_call_id:
                raise ParseError("mensagem tool sem tool_call_id")
            payload.append(
                {
                    "role": "tool",
                    "tool_call_id": message.tool_call_id,
                    "content": message.content,
                }
            )
            continue
        item: dict[str, Any] = {"role": message.role, "content": message.content}
        if message.tool_calls:
            item["tool_calls"] = [_openai_tool_call(call) for call in message.tool_calls]
            if not message.content:
                item["content"] = None
        payload.append(item)
    return payload


def to_anthropic_messages(messages: Sequence[LLMMessage]) -> tuple[str, list[dict[str, Any]]]:
    """Separa o system e agrupa tool results num único turno de usuário, como a API exige."""
    system_parts = [message.content for message in messages if message.role == "system"]
    payload: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "system":
            continue
        if message.role == "tool":
            block = _anthropic_tool_result(message)
            if payload and payload[-1]["role"] == "user" and _is_tool_result_message(payload[-1]):
                payload[-1]["content"].append(block)
            else:
                payload.append({"role": "user", "content": [block]})
            continue
        if message.role == "assistant":
            payload.append({"role": "assistant", "content": _anthropic_assistant_content(message)})
            continue
        payload.append({"role": "user", "content": message.content})
    return "\n".join(part for part in system_parts if part), payload


def _import_module(name: str) -> Any:
    try:
        return __import__(name)
    except ImportError as exc:
        raise ProviderNotInstalled(
            f"Pacote '{name}' ausente. Instale o extra do provedor só no ambiente de experimento."
        ) from exc


def _openai_tool_call(call: ToolCall) -> dict[str, Any]:
    if not call.call_id:
        raise ParseError("chamada OpenAI sem id")
    return {
        "id": call.call_id,
        "type": "function",
        "function": {
            "name": call.name,
            "arguments": json.dumps(call.arguments, ensure_ascii=False),
        },
    }


def _openai_message_payload(message: Any) -> dict[str, Any]:
    tool_calls = []
    for item in message.tool_calls or []:
        tool_calls.append(
            {
                "id": item.id,
                "type": "function",
                "function": {
                    "name": item.function.name,
                    "arguments": item.function.arguments,
                },
            }
        )
    return {"content": message.content or "", "tool_calls": tool_calls}


def _anthropic_response_payload(response: Any) -> dict[str, Any]:
    text_parts: list[str] = []
    tool_calls = []
    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append(
                {
                    "id": block.id,
                    "type": "function",
                    "function": {"name": block.name, "arguments": block.input},
                }
            )
    return {"content": "".join(text_parts), "tool_calls": tool_calls}


def _anthropic_assistant_content(message: LLMMessage) -> list[dict[str, Any]] | str:
    if not message.tool_calls:
        return message.content
    blocks: list[dict[str, Any]] = []
    if message.content:
        blocks.append({"type": "text", "text": message.content})
    for call in message.tool_calls:
        if not call.call_id:
            raise ParseError("chamada Anthropic sem id")
        blocks.append(
            {
                "type": "tool_use",
                "id": call.call_id,
                "name": call.name,
                "input": call.arguments,
            }
        )
    return blocks


def _anthropic_tool_result(message: LLMMessage) -> dict[str, Any]:
    if not message.tool_call_id:
        raise ParseError("tool result Anthropic sem tool_use_id")
    return {
        "type": "tool_result",
        "tool_use_id": message.tool_call_id,
        "content": message.content,
    }


def _is_tool_result_message(message: dict[str, Any]) -> bool:
    content = message.get("content")
    return isinstance(content, list) and all(
        isinstance(block, dict) and block.get("type") == "tool_result" for block in content
    )
