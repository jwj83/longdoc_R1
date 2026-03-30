"""Parser for ReAct-style model outputs."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ParsedStep:
    """Parsed content of one model turn."""

    kind: str
    thought: str
    action: Optional[str] = None
    action_input: Optional[Dict[str, Any]] = None
    final_answer: Optional[str] = None
    error: Optional[str] = None
    raw: str = ""


def _extract_field(text: str, field: str) -> str:
    """Extract one labeled field from model output."""
    pattern = rf"{field}:\s*(.*)"
    m = re.search(pattern, text)
    if not m:
        return ""
    return m.group(1).strip()


def _parse_action_input(raw_input: str) -> Dict[str, Any]:
    """Parse JSON-like action input with graceful fallback."""
    if not raw_input:
        return {}
    try:
        return json.loads(raw_input)
    except Exception:
        try:
            obj = ast.literal_eval(raw_input)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return {}


def parse_react_output(text: str) -> ParsedStep:
    """Parse model text into read/qa action or final answer structure."""
    raw = text.strip()
    thought = _extract_field(raw, "Thought")

    tagged_final = re.search(r"<answer>\s*(.*?)\s*</answer>", raw, flags=re.DOTALL | re.IGNORECASE)
    if tagged_final:
        return ParsedStep(kind="final", thought=thought, final_answer=tagged_final.group(1).strip(), raw=raw)

    tagged_tool = re.search(r"<tool>\s*(.*?)\s*</tool>", raw, flags=re.DOTALL | re.IGNORECASE)
    if tagged_tool:
        tool_text = tagged_tool.group(1).strip()
        read_match = re.match(r"read\s*\(\s*\((.*?)\)\s*\)\s*$", tool_text, flags=re.IGNORECASE)
        if read_match:
            nums = [p.strip() for p in read_match.group(1).split(",") if p.strip()]
            try:
                indices = [int(x) for x in nums]
            except Exception:
                indices = []
            if len(indices) in {1, 2, 3}:
                return ParsedStep(kind="action", thought=thought, action="read", action_input={"index_tuple": indices}, raw=raw)
            return ParsedStep(kind="invalid", thought=thought, raw=raw, error="read tool requires index tuple of length 1, 2, or 3")

        qa_match = re.match(r"qa\s*\(\s*\((.*?)\)\s*,\s*(.*?)\s*\)\s*$", tool_text, flags=re.IGNORECASE | re.DOTALL)
        if qa_match:
            nums = [p.strip() for p in qa_match.group(1).split(",") if p.strip()]
            query_raw = qa_match.group(2).strip()
            if (query_raw.startswith('"') and query_raw.endswith('"')) or (query_raw.startswith("'") and query_raw.endswith("'")):
                query = query_raw[1:-1].strip()
            else:
                query = query_raw
            try:
                indices = [int(x) for x in nums]
            except Exception:
                indices = []
            if len(indices) != 3:
                return ParsedStep(kind="invalid", thought=thought, raw=raw, error="qa tool requires index tuple of length 3")
            if not query:
                return ParsedStep(kind="invalid", thought=thought, raw=raw, error="qa tool requires non-empty query")
            return ParsedStep(
                kind="action",
                thought=thought,
                action="qa",
                action_input={"index_tuple": indices, "question": query},
                raw=raw,
            )
        return ParsedStep(kind="invalid", thought=thought, raw=raw, error="Unsupported <tool> format")

    final_field = _extract_field(raw, "Final Answer")
    if final_field:
        return ParsedStep(kind="final", thought=thought, final_answer=final_field, raw=raw)

    action = _extract_field(raw, "Action")
    action_input_raw = _extract_field(raw, "Action Input")
    action_input = _parse_action_input(action_input_raw)

    if action not in {"read", "qa"}:
        return ParsedStep(
            kind="invalid",
            thought=thought,
            raw=raw,
            error="Action must be one of: read, qa",
        )

    if not isinstance(action_input, dict) or not action_input:
        return ParsedStep(
            kind="invalid",
            thought=thought,
            raw=raw,
            error="Action Input must be a non-empty JSON object",
        )

    if action == "read":
        if set(action_input.keys()) not in ({"node_id"}, {"index_tuple"}):
            return ParsedStep(
                kind="invalid",
                thought=thought,
                raw=raw,
                error="read Action Input must be {\"node_id\": \"...\"} or {\"index_tuple\": [..]}",
            )
        if "node_id" in action_input and not str(action_input.get("node_id", "")).strip():
            return ParsedStep(
                kind="invalid",
                thought=thought,
                raw=raw,
                error="read requires non-empty node_id",
            )
        if "index_tuple" in action_input:
            iv = action_input.get("index_tuple")
            if not isinstance(iv, list) or len(iv) not in {1, 2, 3}:
                return ParsedStep(
                    kind="invalid",
                    thought=thought,
                    raw=raw,
                    error="read index_tuple must be a list with length 1, 2, or 3",
                )

    if action == "qa":
        valid_keys = ({"node_id", "question"}, {"index_tuple", "question"})
        if set(action_input.keys()) not in valid_keys:
            return ParsedStep(
                kind="invalid",
                thought=thought,
                raw=raw,
                error='qa Action Input must be {"node_id": "...", "question": "..."} or {"index_tuple": [..], "question": "..."}',
            )
        if not str(action_input.get("question", "")).strip():
            return ParsedStep(
                kind="invalid",
                thought=thought,
                raw=raw,
                error="qa requires non-empty question",
            )
        if "node_id" in action_input and not str(action_input.get("node_id", "")).strip():
            return ParsedStep(
                kind="invalid",
                thought=thought,
                raw=raw,
                error="qa requires non-empty node_id when node_id is used",
            )
        if "index_tuple" in action_input:
            iv = action_input.get("index_tuple")
            if not isinstance(iv, list) or len(iv) != 3:
                return ParsedStep(
                    kind="invalid",
                    thought=thought,
                    raw=raw,
                    error="qa index_tuple must be a list with length 3",
                )

    return ParsedStep(
        kind="action",
        thought=thought,
        action=action,
        action_input=action_input,
        raw=raw,
    )
