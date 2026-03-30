"""Read tool implementation for hierarchical navigation."""

from __future__ import annotations

from typing import Any

from tools.base import BaseTool, ToolResult


class ReadTool(BaseTool):
    """Return node summary, children previews, and span metadata."""

    name = "read"

    def run(self, env: Any, **kwargs: Any) -> ToolResult:
        node_id = str(kwargs.get("node_id", ""))
        payload = {"node_id": node_id}
        if not node_id:
            return ToolResult(self.name, False, payload, {}, "Missing required field: node_id")

        data = env.read_node(node_id)
        if data is None:
            return ToolResult(self.name, False, payload, {}, f"Node not found: {node_id}")
        return ToolResult(self.name, True, payload, data)
