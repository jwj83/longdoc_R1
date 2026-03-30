"""QA tool implementation."""

from __future__ import annotations

from typing import Any

from tools.base import BaseTool, ToolResult


class QATool(BaseTool):
    """Answer question using full text from one node via QA model."""

    name = "qa"

    def run(self, env: Any, **kwargs: Any) -> ToolResult:
        node_id = str(kwargs.get("node_id", ""))
        question = str(kwargs.get("question", ""))
        payload = {"node_id": node_id, "question": question}

        if not node_id:
            return ToolResult(self.name, False, payload, {}, "Missing required field: node_id")
        if not question:
            return ToolResult(self.name, False, payload, {}, "Missing required field: question")

        data = env.qa_from_node(node_id=node_id, question=question)
        if data is None:
            return ToolResult(self.name, False, payload, {}, f"Node not found: {node_id}")

        return ToolResult(self.name, True, payload, data)
