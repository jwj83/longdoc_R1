"""Base abstractions for environment tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ToolResult:
    """Structured tool output for consistent agent observations."""

    tool_name: str
    success: bool
    input_payload: Dict[str, Any]
    output_payload: Dict[str, Any]
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "input": self.input_payload,
            "output": self.output_payload,
            "error": self.error,
        }


class BaseTool(ABC):
    """Abstract base class for all tools."""

    name: str

    @abstractmethod
    def run(self, env: Any, **kwargs: Any) -> ToolResult:
        """Execute the tool in the provided document environment."""
