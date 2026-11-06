# Ferramentas falsas do estudo e o registro que o loop executa.

from tool_calling_study.tools.fake_tools import build_fake_registry
from tool_calling_study.tools.registry import ToolRegistry
from tool_calling_study.tools.schema import ToolCall, ToolSpec

__all__ = ["ToolCall", "ToolRegistry", "ToolSpec", "build_fake_registry"]
