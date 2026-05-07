"""Minimal ReAct-style hierarchical tool-use agent."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Set

from agent.parser import parse_react_output
from agent.prompt import build_initial_messages
from env.document_env import DocumentEnv
from llm.client import OpenAICompatibleClient


class HierarchicalReActAgent:
    """Iterative tool-use QA agent over a document environment."""

    def __init__(self, llm_client: OpenAICompatibleClient, max_steps: int = 8) -> None:
        self.llm = llm_client
        self.max_steps = max_steps

    @staticmethod
    def _tree_shape_info(env: DocumentEnv) -> str:
        """Describe valid tree index ranges for the decision model."""
        root = env.get_node(env.root_id)
        if root is None:
            return "Tree shape unavailable."
        lines = [f"- High-level segments: 1..{len(root.children_ids)}"]
        for h, l1_id in enumerate(root.children_ids, start=1):
            l1 = env.get_node(l1_id)
            if l1 is None:
                continue
            lines.append(f"- Segment ({h}) has medium segments: 1..{len(l1.children_ids)}")
            for m, l2_id in enumerate(l1.children_ids, start=1):
                l2 = env.get_node(l2_id)
                if l2 is None:
                    continue
                lines.append(f"- Segment ({h},{m}) has low-level leaves: 1..{len(l2.children_ids)}")
        return "\n".join(lines)

    def answer(self, question: str, env: DocumentEnv, max_tokens: int = 256) -> Dict[str, Any]:
        """Run tool-use loop where decision model outputs the final answer."""
        messages = build_initial_messages(
            question=question,
            root_id=env.root_id,
            tree_shape_info=self._tree_shape_info(env),
        )
        trajectory: List[Dict[str, Any]] = []
        used_tools: Set[str] = set()
        final_answer = ""
        qa_used = False
        last_read_node_id = env.root_id
        read_nodes: Set[str] = set()

        for step in range(1, self.max_steps + 1):
            model_output = self.llm.generate(messages, temperature=0.0, max_tokens=max_tokens)
            parsed = parse_react_output(model_output)

            if parsed.kind == "final":
                final_answer = parsed.final_answer or ""
                trajectory.append(
                    {
                        "step": step,
                        "thought": parsed.thought,
                        "final_answer": final_answer,
                        "raw": parsed.raw,
                    }
                )
                break

            if parsed.kind == "invalid":
                trajectory.append(
                    {
                        "step": step,
                        "thought": parsed.thought,
                        "invalid": True,
                        "error": parsed.error or "Invalid action format.",
                        "raw": parsed.raw,
                    }
                )
                messages.append({"role": "assistant", "content": parsed.raw})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Observation: Invalid action format. "
                            "Use exactly one of:\n"
                            "Action: read with Action Input JSON {\"node_id\": \"...\"}\n"
                            "or\n"
                            "Action: qa with Action Input JSON "
                            "{\"node_id\": \"...\", \"question\": \"...\"}."
                            "\nOr output final answer with <answer>...</answer>."
                        ),
                    }
                )
                continue

            action_name = parsed.action or ""
            action_input = parsed.action_input or {}

            if "index_tuple" in action_input:
                idx_raw = action_input.get("index_tuple", [])
                idx: List[int] = []
                if isinstance(idx_raw, list):
                    for x in idx_raw:
                        try:
                            idx.append(int(x))
                        except Exception:
                            idx = []
                            break
                node_id = env.resolve_node_id(idx)
                if not node_id:
                    trajectory.append(
                        {
                            "step": step,
                            "thought": parsed.thought,
                            "invalid": True,
                            "error": f"Invalid index_tuple path: {idx_raw}",
                            "raw": parsed.raw,
                        }
                    )
                    messages.append({"role": "assistant", "content": parsed.raw})
                    messages.append(
                        {
                            "role": "user",
                            "content": "Observation: index_tuple cannot be resolved to a valid node. Choose a valid branch.",
                        }
                    )
                    continue
                action_input = dict(action_input)
                action_input.pop("index_tuple", None)
                action_input["node_id"] = node_id

            if action_name == "qa" and qa_used:
                trajectory.append(
                    {
                        "step": step,
                        "thought": parsed.thought,
                        "invalid": True,
                        "error": "qa has already been used",
                        "raw": parsed.raw,
                    }
                )
                messages.append({"role": "assistant", "content": parsed.raw})
                messages.append(
                    {
                        "role": "user",
                        "content": "Observation: qa tool can be called at most once. Decide with existing observations or output <answer>...</answer>.",
                    }
                )
                continue

            if action_name == "qa":
                node_id = str(action_input.get("node_id", ""))
                node = env.get_node(node_id)
                if node is None:
                    trajectory.append(
                        {
                            "step": step,
                            "thought": parsed.thought,
                            "invalid": True,
                            "error": f"qa target node not found: {node_id}",
                            "raw": parsed.raw,
                        }
                    )
                    messages.append({"role": "assistant", "content": parsed.raw})
                    messages.append(
                        {
                            "role": "user",
                            "content": "Observation: qa target node does not exist. Choose a valid low-level node.",
                        }
                    )
                    continue
                if node.level != 3:
                    trajectory.append(
                        {
                            "step": step,
                            "thought": parsed.thought,
                            "invalid": True,
                            "error": f"qa requires a low-level node (level=3), got level={node.level}",
                            "raw": parsed.raw,
                        }
                    )
                    messages.append({"role": "assistant", "content": parsed.raw})
                    messages.append(
                        {
                            "role": "user",
                            "content": "Observation: qa can only be called on a low-level node (level=3).",
                        }
                    )
                    continue
                if node_id not in read_nodes:
                    trajectory.append(
                        {
                            "step": step,
                            "thought": parsed.thought,
                            "invalid": True,
                            "error": "qa requires prior read on the same low-level node",
                            "raw": parsed.raw,
                        }
                    )
                    messages.append({"role": "assistant", "content": parsed.raw})
                    messages.append(
                        {
                            "role": "user",
                            "content": "Observation: call read on this low-level node before qa.",
                        }
                    )
                    continue

            tool_result = env.execute_tool(action_name, action_input)
            used_tools.add(action_name)

            step_record = {
                "step": step,
                "thought": parsed.thought,
                "action": action_name,
                "action_input": action_input,
                "observation": tool_result.to_dict(),
                "raw": parsed.raw,
            }
            trajectory.append(step_record)

            if action_name == "read" and tool_result.success:
                node_id = action_input.get("node_id", "")
                if isinstance(node_id, str) and node_id:
                    last_read_node_id = node_id
                    read_nodes.add(node_id)

            if action_name == "qa":
                qa_used = True

            messages.append({"role": "assistant", "content": parsed.raw})
            messages.append(
                {
                    "role": "user",
                    "content": "Observation: " + json.dumps(tool_result.to_dict(), ensure_ascii=False),
                }
            )

        if not final_answer:
            if not qa_used:
                fallback = env.execute_tool("qa", {"node_id": last_read_node_id, "question": question})
                used_tools.add("qa")
                trajectory.append(
                    {
                        "step": len(trajectory) + 1,
                        "thought": "Fallback qa due to missing final answer.",
                        "action": "qa",
                        "action_input": {"node_id": last_read_node_id, "question": question},
                        "observation": fallback.to_dict(),
                        "raw": "",
                        "forced": True,
                    }
                )
                if fallback.success:
                    final_answer = str(fallback.output_payload.get("qa_answer", ""))
            else:
                # If model forgot to emit final answer text, use the last qa result.
                for item in reversed(trajectory):
                    obs = item.get("observation", {})
                    if isinstance(obs, dict) and obs.get("tool_name") == "qa" and obs.get("success"):
                        final_answer = str(obs.get("output", {}).get("qa_answer", ""))
                        break

        return {
            "final_answer": final_answer,
            "trajectory": trajectory,
            "used_tools": sorted(list(used_tools)),
            "step_count": len(trajectory),
        }
