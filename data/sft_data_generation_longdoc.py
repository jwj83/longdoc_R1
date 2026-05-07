#!/usr/bin/env python3
"""Generate LongDoc SFT trajectories via dynamic multi-round tool interaction.

This implements a LongVideo-R1-style online trajectory generation loop:
- model decides each step (<think> + <tool> / <answer>)
- tool call is validated and executed
- observation is fed back to model
- answer is judged against gold (semantic proxy), with hint retry

Accepted samples are saved to *_accepted.jsonl and rejected ones to *_rejected.jsonl.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from agent.parser import ParsedStep, parse_react_output
from agent.prompt import SYSTEM_PROMPT, build_initial_messages
from data.load_narrativeqa import load_narrativeqa_subset
from env.document_env import DocumentEnv
from llm.client import OpenAICompatibleClient
from tree.build_tree import build_hierarchical_tree


def normalize_text(text: str) -> str:
    """Normalize text for robust lexical matching."""
    text = text or ""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^0-9a-z\u4e00-\u9fff ]", "", text)
    return text.strip()


def semantic_match(pred: str, gold: str, mode: str = "contain") -> bool:
    """Compare prediction against gold by configurable lexical proxy."""
    p = normalize_text(pred)
    g = normalize_text(gold)
    if not p or not g:
        return False
    if mode == "strict":
        return p == g
    return (p in g) or (g in p)


def parse_judge_json(raw: str) -> Tuple[Optional[bool], str]:
    """Parse judge model output JSON with markdown/noise tolerance."""
    text = str(raw or "").strip()
    if not text:
        return None, "empty judge output"

    def _normalize_jsonish(s: str) -> str:
        s = s.strip()
        s = s.replace("\ufeff", "")
        s = s.replace("\u200b", "")
        s = s.replace("\u200c", "")
        s = s.replace("\u200d", "")
        s = s.replace("\u201c", '"').replace("\u201d", '"')
        s = s.replace("\u2018", "'").replace("\u2019", "'")
        s = re.sub(r"\bTrue\b", "true", s)
        s = re.sub(r"\bFalse\b", "false", s)
        s = re.sub(r"\bNone\b", "null", s)
        s = re.sub(r",\s*([}\]])", r"\1", s)
        return s

    def _parse_obj(s: str) -> Optional[Dict[str, Any]]:
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
        return None

    candidates: List[str] = []

    for m in re.finditer(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, flags=re.DOTALL | re.IGNORECASE):
        block = m.group(1).strip()
        if block:
            candidates.append(block)

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start : end + 1])

    candidates.append(text)

    seen: Set[str] = set()
    uniq_candidates: List[str] = []
    for c in candidates:
        c2 = _normalize_jsonish(c)
        if c2 and c2 not in seen:
            seen.add(c2)
            uniq_candidates.append(c2)

    for c in uniq_candidates:
        obj = _parse_obj(c)
        if obj is None:
            continue
        val = obj.get("consistent")
        if isinstance(val, bool):
            return val, str(obj.get("reason", ""))
        if isinstance(val, str):
            v = val.strip().lower()
            if v in {"true", "false"}:
                return (v == "true"), str(obj.get("reason", ""))

    text_norm = _normalize_jsonish(text)
    bool_match = re.search(r'"?consistent"?\s*[:=]\s*(true|false)', text_norm, flags=re.IGNORECASE)
    if bool_match:
        v = bool_match.group(1).lower() == "true"
        reason_match = re.search(r'"?reason"?\s*[:=]\s*"([^\"]*)"', text_norm, flags=re.IGNORECASE)
        reason = reason_match.group(1).strip() if reason_match else ""
        return v, reason

    zh_true = re.search(r"\b(?:语义一致|一致|正确|匹配)\b", text)
    zh_false = re.search(r"\b(?:不一致|不正确|不匹配|错误)\b", text)
    if zh_true and not zh_false:
        return True, "fallback from textual consistency cue"
    if zh_false and not zh_true:
        return False, "fallback from textual inconsistency cue"

    return None, "judge parse error: cannot extract boolean field 'consistent'"


def judge_answer_with_llm(
    question: str,
    gold: str,
    pred: str,
    judge_client: OpenAICompatibleClient,
    timeout: int,
) -> Dict[str, Any]:
    """Judge semantic consistency by a separate small model."""
    prompt = (
        "You are an answer consistency judge. Determine whether prediction is semantically consistent with gold answer.\n"
        "Rules:\n"
        "1) Same meaning with paraphrase => consistent=true.\n"
        "2) Contradiction, missing key fact, or unrelated => consistent=false.\n"
        "3) If prediction says unknown/insufficient but gold is specific => consistent=false.\n"
        "4) Output strict JSON only: {\"consistent\": true/false, \"reason\": \"...\"}.\n\n"
        f"Question: {question}\n"
        f"Gold: {gold}\n"
        f"Prediction: {pred}\n"
    )
    try:
        raw = judge_client.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=192,
            timeout=timeout,
        )
        consistent, reason = parse_judge_json(raw)
        if consistent is None:
            repair_prompt = (
                "Extract a strict JSON object from the following text. Output JSON only with keys "
                "consistent (boolean) and reason (string).\n\n"
                f"Text:\n{raw}\n"
            )
            repaired_raw = judge_client.generate(
                messages=[{"role": "user", "content": repair_prompt}],
                temperature=0.0,
                max_tokens=128,
                timeout=timeout,
            )
            consistent2, reason2 = parse_judge_json(repaired_raw)
            if consistent2 is None:
                return {"pass": False, "reason": reason, "raw": raw, "error": True}
            return {"pass": bool(consistent2), "reason": reason2, "raw": repaired_raw, "error": False}
        return {"pass": bool(consistent), "reason": reason, "raw": raw, "error": False}
    except Exception as e:
        return {"pass": False, "reason": f"judge request failed: {e}", "raw": "", "error": True}


def load_local_judge_model(model_name: str, device: str) -> Tuple[Any, Any, str]:
    """Load local HF causal LM for semantic judge."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto" if device == "auto" else None,
        trust_remote_code=True,
    )
    if device not in {"auto", ""}:
        model = model.to(device)
    runtime_device = "auto" if device == "auto" else device
    return tokenizer, model, runtime_device


def judge_answer_with_local_model(
    question: str,
    gold: str,
    pred: str,
    tokenizer: Any,
    model: Any,
    max_new_tokens: int,
) -> Dict[str, Any]:
    """Judge semantic consistency by local HF generate."""
    prompt = (
        "You are an answer consistency judge. Determine whether prediction is semantically consistent with gold answer.\n"
        "Rules:\n"
        "1) Same meaning with paraphrase => consistent=true.\n"
        "2) Contradiction, missing key fact, or unrelated => consistent=false.\n"
        "3) If prediction says unknown/insufficient but gold is specific => consistent=false.\n"
        "4) Output strict JSON only: {\"consistent\": true/false, \"reason\": \"...\"}.\n\n"
        f"Question: {question}\n"
        f"Gold: {gold}\n"
        f"Prediction: {pred}\n"
    )
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        if hasattr(model, "device"):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )
        generated = output_ids[0][inputs["input_ids"].shape[1] :]
        raw = tokenizer.decode(generated, skip_special_tokens=True).strip()
        consistent, reason = parse_judge_json(raw)
        if consistent is None:
            return {"pass": False, "reason": reason, "raw": raw, "error": True}
        return {"pass": bool(consistent), "reason": reason, "raw": raw, "error": False}
    except Exception as e:
        return {"pass": False, "reason": f"local judge generate failed: {e}", "raw": "", "error": True}


def judge_answer(
    question: str,
    gold: str,
    pred: str,
    timeout: int,
    judge_client: Optional[OpenAICompatibleClient],
    local_judge: Optional[Tuple[Any, Any]],
    local_max_new_tokens: int,
) -> Dict[str, Any]:
    """Route answer judging to remote API or local HF model."""
    if local_judge is not None:
        tokenizer, model = local_judge
        return judge_answer_with_local_model(
            question=question,
            gold=gold,
            pred=pred,
            tokenizer=tokenizer,
            model=model,
            max_new_tokens=local_max_new_tokens,
        )
    if judge_client is None:
        return {"pass": False, "reason": "judge backend not configured", "raw": "", "error": True}
    return judge_answer_with_llm(
        question=question,
        gold=gold,
        pred=pred,
        judge_client=judge_client,
        timeout=timeout,
    )


def parse_sample_id(sample_id: str) -> Tuple[str, int]:
    """Parse sample id like train_10 or validation_5."""
    m = re.match(r"^(train|validation)_(\d+)$", sample_id or "")
    if not m:
        raise ValueError(f"unsupported sample id format: {sample_id}")
    return m.group(1), int(m.group(2))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load records from a JSONL file."""
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_samples_for_records(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Load enough NarrativeQA train/validation samples for all record ids."""
    split_to_max: Dict[str, int] = {}
    for rec in records:
        split, idx = parse_sample_id(str(rec.get("id", "")))
        split_to_max[split] = max(split_to_max.get(split, -1), idx)

    train_size = split_to_max.get("train", -1) + 1 if "train" in split_to_max else 1
    validation_size = split_to_max.get("validation", -1) + 1 if "validation" in split_to_max else 1
    return load_narrativeqa_subset(train_size=train_size, validation_size=validation_size)


def first_selected_chunk_id(rec: Dict[str, Any]) -> Optional[int]:
    """Choose selected chunk id by model1 priority, then model2."""
    m1 = rec.get("model_outputs", {}).get("model1", {})
    m2 = rec.get("model_outputs", {}).get("model2", {})
    c1 = m1.get("selected_chunk_ids", []) or []
    c2 = m2.get("selected_chunk_ids", []) or []
    if c1 and isinstance(c1[0], int):
        return c1[0]
    if c2 and isinstance(c2[0], int):
        return c2[0]
    return None


def leaf_nodes_in_order(nodes: Dict[str, Any]) -> List[Any]:
    """Return level-3 nodes sorted by global span."""
    leaves = [n for n in nodes.values() if int(getattr(n, "level", -1)) == 3]
    leaves.sort(key=lambda x: (int(x.start_word), int(x.end_word), str(x.node_id)))
    return leaves


def map_chunk_to_leaf_node_id(
    chunk_id: int,
    chunk_size: int,
    total_words: int,
    leaves: List[Any],
) -> Optional[str]:
    """Map flat chunk index to best-overlap leaf node id."""
    if chunk_id < 0:
        return None
    c_start = chunk_id * chunk_size
    c_end = min(total_words, (chunk_id + 1) * chunk_size)
    if c_start >= c_end:
        return None

    best_id: Optional[str] = None
    best_overlap = -1
    for node in leaves:
        ov = max(0, min(c_end, int(node.end_word)) - max(c_start, int(node.start_word)))
        if ov > best_overlap:
            best_overlap = ov
            best_id = str(node.node_id)
    return best_id


def node_id_to_index_tuple(node_id: str) -> Optional[List[int]]:
    """Convert node id suffix to 1-based (h,m,l)."""
    m = re.search(r"_l1_(\d+)_l2_(\d+)_l3_(\d+)$", node_id or "")
    if not m:
        return None
    return [int(m.group(1)) + 1, int(m.group(2)) + 1, int(m.group(3)) + 1]


def build_tree_shape_info(env: DocumentEnv) -> str:
    """Describe valid 1-based index ranges for the current tree."""
    root = env.get_node(env.root_id)
    if root is None:
        return "Tree shape unavailable."
    lines = [f"- High-level segments: 1..{len(root.children_ids)}"]
    medium_counts: List[int] = []
    low_counts: List[int] = []
    for h, l1_id in enumerate(root.children_ids, start=1):
        l1 = env.get_node(l1_id)
        if l1 is None:
            continue
        medium_counts.append(len(l1.children_ids))
        lines.append(f"- Segment ({h}) has medium segments: 1..{len(l1.children_ids)}")
        for m, l2_id in enumerate(l1.children_ids, start=1):
            l2 = env.get_node(l2_id)
            if l2 is None:
                continue
            low_counts.append(len(l2.children_ids))
            lines.append(f"- Segment ({h},{m}) has low-level leaves: 1..{len(l2.children_ids)}")
    if medium_counts:
        lines.insert(1, f"- Medium count range per high segment: {min(medium_counts)}..{max(medium_counts)}")
    if low_counts:
        lines.insert(2, f"- Low-level leaf count range per medium segment: {min(low_counts)}..{max(low_counts)}")
    return "\n".join(lines)


def first_non_empty_answer(rec: Dict[str, Any], sample: Dict[str, Any]) -> str:
    """Prefer model1 answer, then model2, then sample gold."""
    m1 = rec.get("model_outputs", {}).get("model1", {})
    m2 = rec.get("model_outputs", {}).get("model2", {})
    a1 = str(m1.get("answer", "") or "").strip()
    if a1:
        return a1
    a2 = str(m2.get("answer", "") or "").strip()
    if a2:
        return a2
    return str(rec.get("gold", sample.get("answer", "")) or "").strip()


def build_global_overview(env: DocumentEnv, max_summary_chars: int = 800) -> str:
    """Build first-round level-1 global overview, aligned with LongVideo high_data.

    This provides all high-level child segments in one observation before local drilling.
    """
    root = env.get_node(env.root_id)
    if root is None:
        return "High-level overview unavailable: root missing."

    lines: List[str] = [
        "High-level overview (all level-1 segments):",
    ]
    for i, l1_id in enumerate(root.children_ids, start=1):
        data = env.read_node(l1_id)
        if data is None:
            lines.append(f"- L1-{i}: [read failed]")
            continue
        summary = str(data.get("summary", "") or "").replace("\n", " ").strip()
        summary = summary[:max_summary_chars]
        span = data.get("span", {}) or {}
        lines.append(
            f"- L1-{i} node={l1_id} span=[{span.get('start_word', '?')},{span.get('end_word', '?')}) summary={summary}"
        )
    return "\n".join(lines)


def resolve_action_input_with_env(parsed: ParsedStep, env: DocumentEnv) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Resolve index_tuple to node_id and validate basic input shape."""
    action = parsed.action or ""
    action_input = dict(parsed.action_input or {})

    if "index_tuple" in action_input:
        idx_raw = action_input.get("index_tuple")
        if not isinstance(idx_raw, list):
            return None, "index_tuple must be a list"
        idx: List[int] = []
        for x in idx_raw:
            try:
                idx.append(int(x))
            except Exception:
                return None, f"invalid index_tuple element: {x}"
        node_id = env.resolve_node_id(idx)
        if not node_id:
            return None, f"index_tuple cannot be resolved: {idx_raw}"
        action_input.pop("index_tuple", None)
        action_input["node_id"] = node_id

    if action == "read":
        node_id = str(action_input.get("node_id", "")).strip()
        if not node_id:
            return None, "read requires node_id"
        return action_input, None

    if action == "qa":
        node_id = str(action_input.get("node_id", "")).strip()
        question = str(action_input.get("question", "")).strip()
        if not node_id:
            return None, "qa requires node_id"
        if not question:
            return None, "qa requires question"
        return action_input, None

    return None, f"unsupported action: {action}"


def build_progressive_clue_hint(
    fail_count: int,
    wrong_answer: str,
    judge_reason: str,
    recent_read_nodes: List[str],
    target_index_tuple: Optional[List[int]],
) -> str:
    """Build progressively stronger clue-guided hints.

    Level-1 hint: high-level segment only.
    Level-2 hint: high+medium segment.
    Level-3 hint: full low-level segment and explicit qa requirement.
    """
    level = min(fail_count, 3)
    recent = ", ".join(recent_read_nodes[-2:]) if recent_read_nodes else "none"

    if not target_index_tuple or len(target_index_tuple) != 3:
        if level <= 1:
            return (
                "Observation: Judge says your answer is not semantically consistent. "
                f"Reason: {judge_reason}. Please verify core entity/relation before answering again."
            )
        if level == 2:
            return (
                "Observation: Still incorrect. Use read/qa to gather new evidence instead of paraphrasing previous answer. "
                f"Recent read nodes: {recent}. Judge reason: {judge_reason}."
            )
        return (
            "Observation: Final warning before stop. You must call a tool to fetch additional evidence, "
            "then provide one concise answer grounded in that evidence only. "
            f"Previous wrong answer: {wrong_answer}. Judge reason: {judge_reason}."
        )

    h, m, l = target_index_tuple

    if level <= 1:
        return (
            "Observation: Incorrect answer. Clue-level-1: relevant evidence is likely under "
            f"high segment ({h}). Read this high branch before answering again. Judge reason: {judge_reason}."
        )
    if level == 2:
        return (
            "Observation: Still incorrect. Clue-level-2: focus on branch "
            f"({h},{m}). Read this medium segment and nearby low segments. "
            f"Recent read nodes: {recent}. Judge reason: {judge_reason}."
        )
    return (
        "Observation: Final warning before stop. Clue-level-3: key evidence is near "
        f"({h},{m},{l}). You must call qa on a relevant low-level node before final answer. "
        f"Previous wrong answer: {wrong_answer}. Judge reason: {judge_reason}."
    )


def run_multistep_longdoc(
    question: str,
    gold: str,
    env: DocumentEnv,
    decision_client: OpenAICompatibleClient,
    judge_client: Optional[OpenAICompatibleClient],
    local_judge: Optional[Tuple[Any, Any]],
    semantic_mode: str,
    max_rounds: int,
    model_max_tokens: int,
    request_timeout: int,
    judge_max_retry: int,
    local_judge_max_new_tokens: int,
    target_index_tuple: Optional[List[int]],
    overview_max_summary_chars: int,
    initial_clue_level: int,
) -> Dict[str, Any]:
    """Run dynamic multi-round model->tool->observation loop for one sample."""
    messages = build_initial_messages(
        question=question,
        root_id=env.root_id,
        tree_shape_info=build_tree_shape_info(env),
    )
    trajectory: List[Dict[str, Any]] = []
    used_tools: Set[str] = set()
    read_nodes: Set[str] = set()
    qa_used = False
    qa_success_count = 0
    final_answer = ""
    stop_reason = "max_rounds"
    semantic_ok = False
    answer_fail_count = 0
    judge_error_count = 0
    invalid_final_no_qa_count = 0
    judge_trace: List[Dict[str, Any]] = []
    recent_read_nodes: List[str] = []

    # LongVideo-style global first pass: provide all level-1 summaries up front.
    global_overview = build_global_overview(env, max_summary_chars=overview_max_summary_chars)
    messages.append({"role": "user", "content": "Observation: " + global_overview})
    trajectory.append(
        {
            "step": 0,
            "kind": "global_overview",
            "observation": global_overview,
        }
    )

    if target_index_tuple and initial_clue_level > 0:
        clue_level = max(1, min(initial_clue_level, len(target_index_tuple), 3))
        clue_tuple = tuple(target_index_tuple[:clue_level])
        if clue_level == 1:
            clue_text = (
                "Observation: Retrieval prior: relevant evidence is likely under "
                f"high-level segment {clue_tuple}. Use this only as a weak navigation prior; "
                "still verify with read and qa before answering."
            )
        elif clue_level == 2:
            clue_text = (
                "Observation: Retrieval prior: relevant evidence is likely under "
                f"branch {clue_tuple}. Use this only as a weak navigation prior; "
                "still verify with read and qa before answering."
            )
        else:
            clue_text = (
                "Observation: Retrieval prior: relevant evidence is likely near "
                f"low-level leaf {clue_tuple}. You must verify with read and qa before answering."
            )
        messages.append({"role": "user", "content": clue_text})
        trajectory.append(
            {
                "step": 0,
                "kind": "initial_clue",
                "clue_level": clue_level,
                "target_index_tuple": target_index_tuple,
                "observation": clue_text.replace("Observation: ", "", 1),
            }
        )

    for step in range(1, max_rounds + 1):
        raw = decision_client.generate(
            messages=messages,
            temperature=0.0,
            max_tokens=model_max_tokens,
            timeout=request_timeout,
        )
        parsed = parse_react_output(raw)

        if parsed.kind == "invalid":
            trajectory.append(
                {
                    "step": step,
                    "kind": "invalid",
                    "thought": parsed.thought,
                    "error": parsed.error or "invalid format",
                    "raw": parsed.raw,
                }
            )
            messages.append({"role": "assistant", "content": parsed.raw})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Observation: Invalid format. Use exactly one:\n"
                        "<think>...</think>\\n<tool>read((h,m,l))</tool>\n"
                        "or\n"
                        "<think>...</think>\\n<tool>qa((h,m,l), \"question\")</tool>\n"
                        "or\n"
                        "<think>...</think>\\n<answer>...</answer>"
                    ),
                }
            )
            continue

        if parsed.kind == "final":
            candidate = str(parsed.final_answer or "").strip()

            if qa_success_count == 0:
                invalid_final_no_qa_count += 1
                trajectory.append(
                    {
                        "step": step,
                        "kind": "invalid_final_no_qa",
                        "thought": parsed.thought,
                        "final_answer": candidate,
                        "error": "Must call qa before final answer.",
                        "raw": parsed.raw,
                    }
                )
                messages.append({"role": "assistant", "content": parsed.raw})
                messages.append(
                    {
                        "role": "user",
                        "content": "Observation: Must call qa before final answer.",
                    }
                )
                continue

            judge = judge_answer(
                question=question,
                gold=gold,
                pred=candidate,
                timeout=request_timeout,
                judge_client=judge_client,
                local_judge=local_judge,
                local_max_new_tokens=local_judge_max_new_tokens,
            )
            semantic_ok = bool(judge.get("pass", False))
            judge_trace.append(
                {
                    "step": step,
                    "candidate": candidate,
                    "pass": semantic_ok,
                    "reason": str(judge.get("reason", "")),
                    "error": bool(judge.get("error", False)),
                }
            )
            trajectory.append(
                {
                    "step": step,
                    "kind": "final",
                    "thought": parsed.thought,
                    "final_answer": candidate,
                    "semantic_ok": semantic_ok,
                    "judge_reason": str(judge.get("reason", "")),
                    "raw": parsed.raw,
                }
            )

            if semantic_ok:
                final_answer = candidate
                stop_reason = "semantic_pass"
                break

            if bool(judge.get("error", False)):
                judge_error_count += 1
                messages.append({"role": "assistant", "content": parsed.raw})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Observation: Judge formatting failed while scoring your answer. "
                            "Do not change facts. Provide one concise final answer again in "
                            "<answer>...</answer>."
                        ),
                    }
                )
                continue

            answer_fail_count += 1
            if answer_fail_count >= judge_max_retry:
                stop_reason = "judge_max_retry"
                break

            # failed final -> progressive hint and continue
            messages.append({"role": "assistant", "content": parsed.raw})
            messages.append(
                {
                    "role": "user",
                    "content": build_progressive_clue_hint(
                        fail_count=answer_fail_count,
                        wrong_answer=candidate,
                        judge_reason=str(judge.get("reason", "")),
                        recent_read_nodes=recent_read_nodes,
                        target_index_tuple=target_index_tuple,
                    ),
                }
            )
            continue

        # parsed.kind == "action"
        action = parsed.action or ""
        action_input, input_error = resolve_action_input_with_env(parsed=parsed, env=env)
        if input_error is not None or action_input is None:
            trajectory.append(
                {
                    "step": step,
                    "kind": "invalid_action",
                    "thought": parsed.thought,
                    "action": action,
                    "error": input_error,
                    "raw": parsed.raw,
                }
            )
            messages.append({"role": "assistant", "content": parsed.raw})
            messages.append({"role": "user", "content": f"Observation: {input_error}"})
            continue

        # Runtime constraints mirroring agent behavior
        if action == "qa" and qa_used:
            trajectory.append(
                {
                    "step": step,
                    "kind": "invalid_action",
                    "thought": parsed.thought,
                    "action": action,
                    "action_input": action_input,
                    "error": "qa can be called at most once",
                    "raw": parsed.raw,
                }
            )
            messages.append({"role": "assistant", "content": parsed.raw})
            messages.append(
                {
                    "role": "user",
                    "content": "Observation: qa can be called at most once. Use existing evidence or output final answer.",
                }
            )
            continue

        if action == "qa":
            node_id = str(action_input.get("node_id", ""))
            node = env.get_node(node_id)
            if node is None:
                trajectory.append(
                    {
                        "step": step,
                        "kind": "invalid_action",
                        "thought": parsed.thought,
                        "action": action,
                        "action_input": action_input,
                        "error": f"qa target node not found: {node_id}",
                        "raw": parsed.raw,
                    }
                )
                messages.append({"role": "assistant", "content": parsed.raw})
                messages.append({"role": "user", "content": "Observation: qa target node not found."})
                continue
            if node.level != 3:
                trajectory.append(
                    {
                        "step": step,
                        "kind": "invalid_action",
                        "thought": parsed.thought,
                        "action": action,
                        "action_input": action_input,
                        "error": f"qa requires level=3 node, got level={node.level}",
                        "raw": parsed.raw,
                    }
                )
                messages.append({"role": "assistant", "content": parsed.raw})
                messages.append(
                    {
                        "role": "user",
                        "content": "Observation: qa can only be called on low-level node (level=3).",
                    }
                )
                continue
            if node_id not in read_nodes:
                trajectory.append(
                    {
                        "step": step,
                        "kind": "invalid_action",
                        "thought": parsed.thought,
                        "action": action,
                        "action_input": action_input,
                        "error": "qa requires prior read on same node",
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

        tool_result = env.execute_tool(action, action_input)
        used_tools.add(action)
        if action == "read" and tool_result.success:
            node_id = str(action_input.get("node_id", ""))
            if node_id:
                read_nodes.add(node_id)
                recent_read_nodes.append(node_id)
        if action == "qa":
            qa_used = True
            if tool_result.success:
                qa_success_count += 1

        trajectory.append(
            {
                "step": step,
                "kind": "action",
                "thought": parsed.thought,
                "action": action,
                "action_input": action_input,
                "observation": tool_result.to_dict(),
                "raw": parsed.raw,
            }
        )

        messages.append({"role": "assistant", "content": parsed.raw})
        messages.append(
            {
                "role": "user",
                "content": "Observation: " + json.dumps(tool_result.to_dict(), ensure_ascii=False),
            }
        )

    return {
        "final_answer": final_answer,
        "trajectory": trajectory,
        "messages": messages,
        "used_tools": sorted(list(used_tools)),
        "step_count": len(trajectory),
        "semantic_ok": semantic_ok,
        "replay_ok": bool(final_answer),
        "stop_reason": stop_reason,
        "answer_fail_count": answer_fail_count,
        "judge_error_count": judge_error_count,
        "qa_success_count": qa_success_count,
        "invalid_final_no_qa_count": invalid_final_no_qa_count,
        "judge_trace": judge_trace,
    }


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate dynamic LongDoc SFT trajectories")
    parser.add_argument("--agreement_file", type=str, required=True)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--decision_model", type=str, default="")
    parser.add_argument("--decision_base_url", type=str, default="")
    parser.add_argument("--decision_api_key", type=str, default="")
    parser.add_argument("--read_model", type=str, default="")
    parser.add_argument("--read_base_url", type=str, default="")
    parser.add_argument("--read_api_key", type=str, default="")
    parser.add_argument("--qa_model", type=str, default="")
    parser.add_argument("--qa_base_url", type=str, default="")
    parser.add_argument("--qa_api_key", type=str, default="")
    parser.add_argument("--judge_model", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--judge_base_url", type=str, default="")
    parser.add_argument("--judge_api_key", type=str, default="")
    parser.add_argument("--judge_local", action="store_true", default=False)
    parser.add_argument("--judge_local_device", type=str, default="cuda")
    parser.add_argument("--judge_local_max_new_tokens", type=int, default=192)
    parser.add_argument("--chunk_size", type=int, default=300)
    parser.add_argument("--max_rounds", type=int, default=8)
    parser.add_argument("--model_max_tokens", type=int, default=512)
    parser.add_argument("--request_timeout", type=int, default=120)
    parser.add_argument("--judge_max_retry", type=int, default=3)
    parser.add_argument("--semantic_mode", type=str, choices=["contain", "strict"], default="contain")
    parser.add_argument("--overview_max_summary_chars", type=int, default=800)
    parser.add_argument("--initial_clue_level", type=int, default=0)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--resume", action="store_true", default=True)
    return parser.parse_args()


def build_client(role: str, model: str, base_url: str, api_key: str) -> OpenAICompatibleClient:
    """Create one role-specific OpenAI-compatible client."""
    return OpenAICompatibleClient(base_url=base_url, api_key=api_key, model_name=model)


def main() -> int:
    """Main entry for dynamic SFT data generation."""
    args = parse_args()

    in_path = Path(args.agreement_file)
    if not in_path.exists():
        print(f"ERROR: agreement_file not found: {in_path}")
        return 2
    if args.chunk_size <= 0 or args.max_rounds <= 0 or args.save_every <= 0:
        print("ERROR: chunk_size/max_rounds/save_every must be > 0")
        return 2
    if args.judge_max_retry <= 0:
        print("ERROR: judge_max_retry must be > 0")
        return 2
    if args.overview_max_summary_chars <= 0:
        print("ERROR: overview_max_summary_chars must be > 0")
        return 2
    if args.initial_clue_level < 0 or args.initial_clue_level > 3:
        print("ERROR: initial_clue_level must be in [0, 3]")
        return 2
    if args.start_index < 0 or args.end_index < 0:
        print("ERROR: start_index/end_index must be >= 0")
        return 2
    if args.end_index and args.end_index <= args.start_index:
        print("ERROR: end_index must be > start_index when set")
        return 2

    if not args.decision_model or not args.decision_base_url or not args.decision_api_key:
        print("ERROR: decision model/base_url/api_key are required")
        return 2
    read_model = args.read_model or args.decision_model
    read_base_url = args.read_base_url or args.decision_base_url
    read_api_key = args.read_api_key or args.decision_api_key
    if not read_model or not read_base_url or not read_api_key:
        print("ERROR: read model/base_url/api_key are required (or inherit from decision)")
        return 2

    qa_model = args.qa_model or args.decision_model
    qa_base_url = args.qa_base_url or args.decision_base_url
    qa_api_key = args.qa_api_key or args.decision_api_key
    if not qa_model or not qa_base_url or not qa_api_key:
        print("ERROR: qa model/base_url/api_key are required (or inherit from decision)")
        return 2

    judge_model = args.judge_model or args.decision_model
    judge_base_url = args.judge_base_url or args.decision_base_url
    judge_api_key = args.judge_api_key or args.decision_api_key
    if not judge_model:
        print("ERROR: judge model is required")
        return 2
    if not args.judge_local and (not judge_base_url or not judge_api_key):
        print("ERROR: judge base_url/api_key are required when --judge_local is not set")
        return 2

    out_base = Path(args.output) if args.output else in_path.with_name(f"{in_path.stem}_sft_dynamic.jsonl")
    accepted_path = out_base.with_name(f"{out_base.stem}_accepted{out_base.suffix}")
    rejected_path = out_base.with_name(f"{out_base.stem}_rejected{out_base.suffix}")
    accepted_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_path.parent.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(in_path)
    if args.max_samples > 0:
        records = records[: args.max_samples]
    if args.start_index or args.end_index:
        records = records[args.start_index : args.end_index or None]
    loaded = load_samples_for_records(records)

    done_ids: Set[str] = set()
    if args.resume and accepted_path.exists():
        for row in load_jsonl(accepted_path):
            rid = str(row.get("id", ""))
            if rid:
                done_ids.add(rid)
    if args.resume and rejected_path.exists():
        for row in load_jsonl(rejected_path):
            rid = str(row.get("id", ""))
            if rid:
                done_ids.add(rid)

    decision_client = build_client("decision", args.decision_model, args.decision_base_url, args.decision_api_key)
    read_client = build_client("read", read_model, read_base_url, read_api_key)
    qa_client = build_client("qa", qa_model, qa_base_url, qa_api_key)
    local_judge: Optional[Tuple[Any, Any]] = None
    judge_client: Optional[OpenAICompatibleClient] = None
    if args.judge_local:
        tokenizer, model, runtime_device = load_local_judge_model(judge_model, args.judge_local_device)
        local_judge = (tokenizer, model)
        print(f"Loaded local judge model: {judge_model} on device={runtime_device}")
    else:
        judge_client = build_client("judge", judge_model, judge_base_url, judge_api_key)

    print(
        "Configured models: "
                        f"decision={args.decision_model}, read={read_model}, qa={qa_model}, judge={judge_model}"
    )

    processed = 0
    accepted = 0
    rejected = 0
    errors = 0

    mode = "a" if args.resume else "w"
    with accepted_path.open(mode, encoding="utf-8") as fa, rejected_path.open(mode, encoding="utf-8") as fr:
        for rec in records:
            rid = str(rec.get("id", ""))
            if not rid:
                errors += 1
                continue
            if rid in done_ids:
                continue

            try:
                split, idx = parse_sample_id(rid)
                samples = loaded.get(split, [])
                if idx < 0 or idx >= len(samples):
                    raise ValueError(f"sample index out of range: {rid}")
                sample = samples[idx]

                question = str(rec.get("question", sample.get("question", "")) or "").strip()
                gold = str(rec.get("gold", sample.get("answer", "")) or "").strip()
                if not question:
                    raise ValueError("empty question")

                document = str(sample.get("document", "") or "")
                if not document.strip():
                    raise ValueError("empty document")

                root_id, nodes = build_hierarchical_tree(document_text=document, doc_id=rid, min_leaf_words=args.chunk_size)
                env = DocumentEnv(
                    nodes=nodes,
                    root_id=root_id,
                    read_llm_client=read_client,
                    qa_llm_client=qa_client,
                )

                target_index_tuple: Optional[List[int]] = None
                selected_chunk_id = first_selected_chunk_id(rec)
                if selected_chunk_id is not None:
                    leaves = leaf_nodes_in_order(nodes)
                    total_words = len(document.split())
                    leaf_node_id = map_chunk_to_leaf_node_id(
                        chunk_id=selected_chunk_id,
                        chunk_size=args.chunk_size,
                        total_words=total_words,
                        leaves=leaves,
                    )
                    if leaf_node_id:
                        target_index_tuple = node_id_to_index_tuple(leaf_node_id)

                result = run_multistep_longdoc(
                    question=question,
                    gold=gold,
                    env=env,
                    decision_client=decision_client,
                    judge_client=judge_client,
                    local_judge=local_judge,
                    semantic_mode=args.semantic_mode,
                    max_rounds=args.max_rounds,
                    model_max_tokens=args.model_max_tokens,
                    request_timeout=args.request_timeout,
                    judge_max_retry=args.judge_max_retry,
                    local_judge_max_new_tokens=args.judge_local_max_new_tokens,
                    target_index_tuple=target_index_tuple,
                    overview_max_summary_chars=args.overview_max_summary_chars,
                    initial_clue_level=args.initial_clue_level,
                )

                out = {
                    "id": rid,
                    "split": split,
                    "question": question,
                    "gold": gold,
                    "final_answer": result.get("final_answer", ""),
                    "messages": result.get("messages", []),
                    "trajectory": result.get("trajectory", []),
                    "meta": {
                        "source": "dynamic_longdoc_generator",
                        "semantic_mode": args.semantic_mode,
                        "semantic_ok": bool(result.get("semantic_ok", False)),
                        "replay_ok": bool(result.get("replay_ok", False)),
                        "stop_reason": result.get("stop_reason", ""),
                        "step_count": int(result.get("step_count", 0)),
                        "used_tools": result.get("used_tools", []),
                        "answer_fail_count": int(result.get("answer_fail_count", 0)),
                        "judge_error_count": int(result.get("judge_error_count", 0)),
                        "qa_success_count": int(result.get("qa_success_count", 0)),
                        "invalid_final_no_qa_count": int(result.get("invalid_final_no_qa_count", 0)),
                        "overview_max_summary_chars": int(args.overview_max_summary_chars),
                        "initial_clue_level": int(args.initial_clue_level),
                        "judge_trace": result.get("judge_trace", []),
                    },
                }

                if out["meta"]["semantic_ok"] and out["meta"]["replay_ok"] and out["meta"]["qa_success_count"] >= 1:
                    fa.write(json.dumps(out, ensure_ascii=False) + "\n")
                    accepted += 1
                else:
                    fr.write(json.dumps(out, ensure_ascii=False) + "\n")
                    rejected += 1

            except Exception as e:
                errors += 1
                err_row = {
                    "id": rid,
                    "meta": {
                        "source": "dynamic_longdoc_generator",
                        "error": str(e),
                    },
                }
                fr.write(json.dumps(err_row, ensure_ascii=False) + "\n")

            processed += 1
            if processed % args.save_every == 0:
                fa.flush()
                fr.flush()
                print(
                    f"[{processed}/{len(records)}] accepted={accepted} rejected={rejected} errors={errors} "
                    f"accepted_file={accepted_path}"
                )

    print("\n===== Dynamic SFT Generation Summary =====")
    print(f"Input:     {in_path}")
    print(f"Accepted:  {accepted_path}")
    print(f"Rejected:  {rejected_path}")
    print(f"Processed: {processed}")
    print(f"Accepted:  {accepted}")
    print(f"Rejected:  {rejected}")
    print(f"Errors:    {errors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
