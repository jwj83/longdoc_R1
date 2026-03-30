"""Tree node definitions for hierarchical document representation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TreeNode:
    """A node in the document hierarchy tree."""

    node_id: str
    parent_id: Optional[str]
    children_ids: List[str] = field(default_factory=list)
    level: int = 0
    text: str = ""
    summary: str = ""
    start_word: int = 0
    end_word: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize node to JSON-compatible dict."""
        return {
            "node_id": self.node_id,
            "parent_id": self.parent_id,
            "children_ids": list(self.children_ids),
            "level": self.level,
            "text": self.text,
            "summary": self.summary,
            "start_word": self.start_word,
            "end_word": self.end_word,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TreeNode":
        """Create node from serialized dict."""
        return cls(
            node_id=str(data.get("node_id", "")),
            parent_id=data.get("parent_id"),
            children_ids=list(data.get("children_ids", [])),
            level=int(data.get("level", 0)),
            text=str(data.get("text", "")),
            summary=str(data.get("summary", "")),
            start_word=int(data.get("start_word", 0)),
            end_word=int(data.get("end_word", 0)),
        )
