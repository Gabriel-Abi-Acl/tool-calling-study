# Interface de LLM e fábrica. Estratégias dependem disto, não de um SDK específico.

from tool_calling_study.llm.base import LLMClient, LLMMessage, LLMResponse
from tool_calling_study.llm.factory import build_llm
from tool_calling_study.llm.fake import FakeLLM

__all__ = [
    "FakeLLM",
    "LLMClient",
    "LLMMessage",
    "LLMResponse",
    "build_llm",
]
