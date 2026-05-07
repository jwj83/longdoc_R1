"""verl agent loop for LongDoc-R1 XML read/qa tool use."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Set
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.workers.rollout.replica import TokenOutput

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.parser import parse_react_output
from data.sft_data_generation_longdoc import build_global_overview
from env.document_env import DocumentEnv
from llm.client import OpenAICompatibleClient
from tree.build_tree import build_hierarchical_tree


def _client(prefix: str, fallback: str | None = None) -> OpenAICompatibleClient:
    prefix = prefix.upper()
    fallback = (fallback or prefix).upper()
    return OpenAICompatibleClient(
        base_url=os.getenv(f"{prefix}_BASE_URL") or os.getenv(f"{fallback}_BASE_URL") or os.getenv("BASE_URL", ""),
        api_key=os.getenv(f"{prefix}_API_KEY") or os.getenv(f"{fallback}_API_KEY") or os.getenv("API_KEY", ""),
        model_name=os.getenv(f"{prefix}_MODEL_NAME") or os.getenv(f"{fallback}_MODEL_NAME") or os.getenv("MODEL_NAME", ""),
    )


def _observation(text: str) -> Dict[str, str]:
    return {"role": "user", "content": "Observation: " + text}


def _to_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


@register("longdoc_xml_agent")
class LongDocXMLAgentLoop(AgentLoopBase):
    """A multi-turn agent loop that executes the project's XML read/qa tool calls."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.max_assistant_turns = self.rollout_config.multi_turn.max_assistant_turns or 8
        self.max_user_turns = self.rollout_config.multi_turn.max_user_turns or 8
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        self.read_client = _client("READ")
        self.qa_client = _client("QA", fallback="ANSWER")

    async def _generate(self, request_id: str, prompt_ids: List[int], sampling_params: Dict[str, Any]) -> TokenOutput:
        return await self.server_manager.generate(
            request_id=request_id,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
        )

    async def _encode_observation(self, message: Dict[str, str]) -> List[int]:
        return await self.apply_chat_template([message], remove_system_prompt=True)

    async def run(self, sampling_params: Dict[str, Any], **kwargs: Any) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        extra_info = kwargs.get("extra_info", {}) or {}
        sample_id = str(extra_info.get("sample_id") or extra_info.get("index") or "longdoc")
        document = str(extra_info.get("document", "") or "")
        if not document:
            messages.append(_observation("Missing document in extra_info; cannot execute tools."))

        root_id, nodes = build_hierarchical_tree(document_text=document, doc_id=sample_id)
        env = DocumentEnv(nodes=nodes, root_id=root_id, read_llm_client=self.read_client, qa_llm_client=self.qa_client)

        overview_chars = int(extra_info.get("overview_max_summary_chars", 0) or 0)
        if overview_chars > 0:
            overview = build_global_overview(env, max_summary_chars=overview_chars)
            messages.append(_observation(overview))

        target_tuple = list(extra_info.get("target_index_tuple", []) or [])
        clue_level = int(extra_info.get("clue_level", 0) or 0)
        if target_tuple and clue_level > 0:
            clue_tuple = tuple(target_tuple[: max(1, min(clue_level, 3))])
            messages.append(
                _observation(
                    "Retrieval prior: relevant evidence is likely under "
                    f"branch {clue_tuple}. Use this only as a weak navigation prior; "
                    "still verify with read and qa before answering."
                )
            )

        request_id = uuid4().hex
        prompt_ids = await self.apply_chat_template(messages)
        initial_prompt_len = len(prompt_ids)
        response_mask: List[int] = []
        response_logprobs: List[float] = []
        trajectory: List[Dict[str, Any]] = []
        read_nodes: Set[str] = set()
        qa_success_count = 0
        invalid_count = 0
        final_answer = ""
        metrics: Dict[str, Any] = {}

        for turn in range(1, self.max_assistant_turns + 1):
            output = await self._generate(request_id, prompt_ids, sampling_params)
            assistant_ids = output.token_ids
            assistant_text = await self.loop.run_in_executor(None, self.tokenizer.decode, assistant_ids)
            prompt_ids += assistant_ids
            response_mask += [1] * len(assistant_ids)
            if output.log_probs:
                response_logprobs += output.log_probs
            messages.append({"role": "assistant", "content": assistant_text})

            parsed = parse_react_output(assistant_text)
            if parsed.kind == "final":
                final_answer = str(parsed.final_answer or "")
                trajectory.append({"turn": turn, "kind": "final", "final_answer": final_answer, "raw": parsed.raw})
                break

            if parsed.kind == "invalid" or not parsed.action or not parsed.action_input:
                invalid_count += 1
                obs = _observation(
                    "Invalid format. Use exactly one XML action: "
                    "<tool>read((h,m,l))</tool>, <tool>qa((h,m,l), \"question\")</tool>, "
                    "or <answer>...</answer>."
                )
                trajectory.append({"turn": turn, "kind": "invalid", "raw": parsed.raw, "error": parsed.error})
                messages.append(obs)
                obs_ids = await self._encode_observation(obs)
                prompt_ids += obs_ids
                response_mask += [0] * len(obs_ids)
                response_logprobs += [0.0] * len(obs_ids) if response_logprobs else []
                continue

            action_input = dict(parsed.action_input)
            if "index_tuple" in action_input:
                node_id = env.resolve_node_id(action_input["index_tuple"])
                if not node_id:
                    invalid_count += 1
                    obs = _observation(f"Invalid node index: {action_input['index_tuple']}")
                    trajectory.append({"turn": turn, "kind": "invalid_action", "action": parsed.action, "raw": parsed.raw})
                    messages.append(obs)
                    obs_ids = await self._encode_observation(obs)
                    prompt_ids += obs_ids
                    response_mask += [0] * len(obs_ids)
                    response_logprobs += [0.0] * len(obs_ids) if response_logprobs else []
                    continue
                action_input["node_id"] = node_id
                action_input.pop("index_tuple", None)

            if parsed.action == "qa":
                node_id = str(action_input.get("node_id", ""))
                if node_id not in read_nodes:
                    invalid_count += 1
                    obs = _observation("qa may only be called after read on the same low-level node.")
                    trajectory.append({"turn": turn, "kind": "invalid_action", "action": "qa", "raw": parsed.raw})
                    messages.append(obs)
                    obs_ids = await self._encode_observation(obs)
                    prompt_ids += obs_ids
                    response_mask += [0] * len(obs_ids)
                    response_logprobs += [0.0] * len(obs_ids) if response_logprobs else []
                    continue

            result = env.execute_tool(parsed.action, action_input).to_dict()
            if parsed.action == "read" and result.get("success"):
                read_nodes.add(str(action_input.get("node_id", "")))
            if parsed.action == "qa" and result.get("success"):
                qa_success_count += 1
            trajectory.append({"turn": turn, "kind": "action", "action": parsed.action, "result": result, "raw": parsed.raw})
            obs = _observation(_to_json(result))
            messages.append(obs)
            obs_ids = await self._encode_observation(obs)
            prompt_ids += obs_ids
            response_mask += [0] * len(obs_ids)
            response_logprobs += [0.0] * len(obs_ids) if response_logprobs else []
            if len(response_mask) >= self.response_length or len(prompt_ids) >= self.prompt_length + self.response_length:
                break

        response_ids = prompt_ids[initial_prompt_len:]
        extra_fields = {
            "final_answer": final_answer,
            "trajectory": trajectory,
            "qa_success_count": qa_success_count,
            "invalid_count": invalid_count,
            "target_leaf_node_id": extra_info.get("target_leaf_node_id", ""),
        }
        metrics.update(
            {
                "longdoc/qa_success": float(qa_success_count > 0),
                "longdoc/invalid_count": float(invalid_count),
                "longdoc/turns": float(len(trajectory)),
            }
        )
        return AgentLoopOutput(
            prompt_ids=prompt_ids[:initial_prompt_len],
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
            num_turns=len(trajectory),
            metrics=metrics,
            extra_fields=extra_fields,
        )
