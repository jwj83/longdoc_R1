#!/usr/bin/env python3
"""Clean and score LongDoc-R1 tool-use SFT trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from agent.parser import parse_react_output
from data.load_narrativeqa import load_narrativeqa_subset
from tree.build_tree import build_hierarchical_tree


PROJECT_ROOT = Path(__file__).resolve().parents[1]


UNKNOWN_PATTERNS = [
    "i don't know",
    "i do not know",
    "cannot determine",
    "not enough information",
    "insufficient information",
    "无法判断",
    "无法从给定文本中判断",
    "不知道",
]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file."""
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write rows to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(text: str) -> str:
    """Normalize text for simple lexical checks."""
    text = (text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^0-9a-z\u4e00-\u9fff ]", "", text)
    return text.strip()


def is_unknown_answer(text: str) -> bool:
    """Return true if a QA answer says the answer is unavailable."""
    norm = normalize_text(text)
    return any(normalize_text(pattern) in norm for pattern in UNKNOWN_PATTERNS)


def parse_sample_id(sample_id: str) -> Tuple[str, int]:
    """Parse ids such as train_10 and validation_5."""
    m = re.match(r"^(train|validation)_(\d+)$", sample_id or "")
    if not m:
        raise ValueError(f"unsupported sample id format: {sample_id}")
    return m.group(1), int(m.group(2))


def load_saved_split(split: str, min_size: int) -> Optional[List[Dict[str, Any]]]:
    """Load a saved NarrativeQA split from local outputs when available."""
    candidates = [
        PROJECT_ROOT / "outputs" / "data" / f"narrativeqa_{split}.jsonl",
        PROJECT_ROOT / "outputs" / "20260330_172948" / "data" / f"narrativeqa_{split}.jsonl",
        PROJECT_ROOT / "outputs" / "20260330_165640" / "data" / f"narrativeqa_{split}.jsonl",
    ]
    for path in candidates:
        if not path.exists():
            continue
        rows = load_jsonl(path)
        if len(rows) >= min_size:
            return rows
    return None


def load_samples_for_ids(sample_ids: Set[str]) -> Dict[str, List[Dict[str, Any]]]:
    """Load enough NarrativeQA samples for the requested ids."""
    split_to_max: Dict[str, int] = {}
    for sample_id in sample_ids:
        split, idx = parse_sample_id(sample_id)
        split_to_max[split] = max(split_to_max.get(split, -1), idx)
    train_size = split_to_max.get("train", -1) + 1 if "train" in split_to_max else 1
    validation_size = split_to_max.get("validation", -1) + 1 if "validation" in split_to_max else 1
    local_train = load_saved_split("train", train_size)
    local_validation = load_saved_split("validation", validation_size)
    if local_train is not None and local_validation is not None:
        return {"train": local_train[:train_size], "validation": local_validation[:validation_size]}

    loaded = load_narrativeqa_subset(train_size=train_size, validation_size=validation_size)
    if local_train is not None:
        loaded["train"] = local_train[:train_size]
    if local_validation is not None:
        loaded["validation"] = local_validation[:validation_size]
    return loaded


def first_selected_chunk_id(label: Dict[str, Any]) -> Optional[int]:
    """Return the preferred selected chunk id from a high-quality label row."""
    outputs = label.get("model_outputs", {}) or {}
    for model_name in ("model1", "model2"):
        selected = (outputs.get(model_name, {}) or {}).get("selected_chunk_ids", []) or []
        if selected and isinstance(selected[0], int):
            return selected[0]
    return None


def leaf_nodes_in_order(nodes: Dict[str, Any]) -> List[Any]:
    """Return level-3 nodes sorted by word span."""
    leaves = [node for node in nodes.values() if int(getattr(node, "level", -1)) == 3]
    leaves.sort(key=lambda node: (int(node.start_word), int(node.end_word), str(node.node_id)))
    return leaves


def map_chunk_to_leaf_node_id(
    chunk_id: int,
    chunk_size: int,
    total_words: int,
    leaves: List[Any],
) -> Tuple[Optional[str], int]:
    """Map a flat chunk id to the leaf with maximum word-span overlap."""
    if chunk_id < 0:
        return None, 0
    chunk_start = chunk_id * chunk_size
    chunk_end = min(total_words, (chunk_id + 1) * chunk_size)
    if chunk_start >= chunk_end:
        return None, 0

    best_id: Optional[str] = None
    best_overlap = 0
    for node in leaves:
        overlap = max(0, min(chunk_end, int(node.end_word)) - max(chunk_start, int(node.start_word)))
        if overlap > best_overlap:
            best_overlap = overlap
            best_id = str(node.node_id)
    return best_id, best_overlap


def load_label_map(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load high-quality labels keyed by sample id."""
    return {str(row.get("id", "")): row for row in load_jsonl(path) if row.get("id")}


def successful_actions(trajectory: List[Dict[str, Any]], action: str) -> List[Dict[str, Any]]:
    """Return successful action steps for an action name."""
    rows: List[Dict[str, Any]] = []
    for step in trajectory:
        if step.get("kind") != "action" or step.get("action") != action:
            continue
        observation = step.get("observation", {}) or {}
        if bool(observation.get("success", False)):
            rows.append(step)
    return rows


def action_node_id(step: Optional[Dict[str, Any]]) -> str:
    """Extract a node id from a trajectory action."""
    if not step:
        return ""
    action_input = step.get("action_input", {}) or {}
    if action_input.get("node_id"):
        return str(action_input.get("node_id"))
    observation = step.get("observation", {}) or {}
    obs_input = observation.get("input", {}) or {}
    if obs_input.get("node_id"):
        return str(obs_input.get("node_id"))
    output = observation.get("output", {}) or {}
    return str(output.get("node_id", ""))


def qa_answer(step: Optional[Dict[str, Any]]) -> str:
    """Extract the qa answer text from a qa action step."""
    if not step:
        return ""
    observation = step.get("observation", {}) or {}
    output = observation.get("output", {}) or {}
    return str(output.get("qa_answer", ""))


def qa_question(step: Dict[str, Any]) -> str:
    """Extract the question passed to qa."""
    action_input = step.get("action_input", {}) or {}
    if action_input.get("question"):
        return str(action_input.get("question"))
    observation = step.get("observation", {}) or {}
    obs_input = observation.get("input", {}) or {}
    return str(obs_input.get("question", ""))


def index_tuple_from_node_id(node_id: str) -> Optional[List[int]]:
    """Convert a node id suffix to 1-based hierarchical indices."""
    m3 = re.search(r"_l1_(\d+)_l2_(\d+)_l3_(\d+)$", node_id or "")
    if m3:
        return [int(m3.group(1)) + 1, int(m3.group(2)) + 1, int(m3.group(3)) + 1]
    m2 = re.search(r"_l1_(\d+)_l2_(\d+)$", node_id or "")
    if m2:
        return [int(m2.group(1)) + 1, int(m2.group(2)) + 1]
    m1 = re.search(r"_l1_(\d+)$", node_id or "")
    if m1:
        return [int(m1.group(1)) + 1]
    return None


def tool_call_from_step(step: Dict[str, Any]) -> str:
    """Build a canonical tool call for a cleaned assistant message."""
    action = str(step.get("action", ""))
    raw = str(step.get("raw", ""))
    parsed = parse_react_output(raw)
    if parsed.kind == "action" and parsed.action == action:
        tagged_tool = re.search(r"<tool>\s*(.*?)\s*</tool>", raw, flags=re.DOTALL | re.IGNORECASE)
        if tagged_tool:
            return tagged_tool.group(1).strip()

    idx = index_tuple_from_node_id(action_node_id(step))
    if action == "read" and idx:
        return f"read(({','.join(str(x) for x in idx)}))"
    if action == "qa" and idx:
        question = qa_question(step).replace('"', '\\"')
        return f'qa(({",".join(str(x) for x in idx)}), "{question}")'
    return raw


def initial_clean_messages(row: Dict[str, Any]) -> List[Dict[str, str]]:
    """Keep system/question/global-overview messages from the original conversation."""
    messages = row.get("messages", []) or []
    cleaned: List[Dict[str, str]] = []
    for msg in messages[:3]:
        role = str(msg.get("role", ""))
        content = str(msg.get("content", ""))
        if role in {"system", "user"} and content:
            cleaned.append({"role": role, "content": content})
    return cleaned


def build_clean_messages(row: Dict[str, Any], action_steps: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Build clean SFT messages from successful actions and final answer."""
    messages = initial_clean_messages(row)
    for step in action_steps:
        thought = str(step.get("thought", "") or "").strip()
        if not thought:
            thought = "I will use the available evidence to continue the hierarchical search."
        messages.append(
            {
                "role": "assistant",
                "content": f"<think>{thought}</think>\n<tool>{tool_call_from_step(step)}</tool>",
            }
        )
        messages.append(
            {
                "role": "user",
                "content": "Observation: " + json.dumps(step.get("observation", {}), ensure_ascii=False),
            }
        )
    final_answer = str(row.get("final_answer", "") or "")
    messages.append(
        {
            "role": "assistant",
            "content": (
                "<think>The evidence is sufficient to answer the question.</think>\n"
                f"<answer>{final_answer}</answer>"
            ),
        }
    )
    return messages


def build_evidence_map(
    labels: Dict[str, Dict[str, Any]],
    chunk_size: int,
) -> Dict[str, Dict[str, Any]]:
    """Build sample id -> evidence leaf metadata."""
    loaded = load_samples_for_ids(set(labels.keys()))
    evidence: Dict[str, Dict[str, Any]] = {}

    for sample_id, label in labels.items():
        split, idx = parse_sample_id(sample_id)
        samples = loaded.get(split, [])
        if idx < 0 or idx >= len(samples):
            continue
        sample = samples[idx]
        document = str(sample.get("document", "") or "")
        chunk_id = first_selected_chunk_id(label)
        if chunk_id is None or not document.strip():
            continue
        _, nodes = build_hierarchical_tree(
            document_text=document,
            doc_id=sample_id,
            min_leaf_words=chunk_size,
        )
        leaf_id, overlap = map_chunk_to_leaf_node_id(
            chunk_id=chunk_id,
            chunk_size=chunk_size,
            total_words=len(document.split()),
            leaves=leaf_nodes_in_order(nodes),
        )
        evidence[sample_id] = {
            "selected_chunk_id": chunk_id,
            "target_leaf_node_id": leaf_id,
            "target_overlap_words": overlap,
        }
    return evidence


def count_kind(trajectory: List[Dict[str, Any]], kind: str) -> int:
    """Count trajectory records by kind."""
    return sum(1 for step in trajectory if step.get("kind") == kind)


def analyze_row(
    row: Dict[str, Any],
    evidence: Dict[str, Any],
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Analyze one raw SFT row and return report, strict row, soft row."""
    sample_id = str(row.get("id", ""))
    trajectory = row.get("trajectory", []) or []
    meta = row.get("meta", {}) or {}

    read_steps = successful_actions(trajectory, "read")
    qa_steps = successful_actions(trajectory, "qa")
    action_steps = read_steps + qa_steps
    action_steps.sort(key=lambda step: int(step.get("step", 0)))

    last_qa = qa_steps[-1] if qa_steps else None
    last_qa_step = int(last_qa.get("step", 0)) if last_qa else 0
    terminal_action_steps = [
        step for step in action_steps if not last_qa or int(step.get("step", 0)) <= last_qa_step
    ]
    action_after_qa_count = sum(
        1 for step in action_steps if last_qa and int(step.get("step", 0)) > last_qa_step
    )
    last_qa_node = action_node_id(last_qa)
    read_node_ids = {action_node_id(step) for step in read_steps}
    target_leaf = str((evidence or {}).get("target_leaf_node_id", "") or "")

    invalid_action_count = count_kind(trajectory, "invalid_action") + count_kind(trajectory, "invalid")
    invalid_final_count = count_kind(trajectory, "invalid_final_no_qa")
    semantic_ok = bool(meta.get("semantic_ok", False))
    replay_ok = bool(meta.get("replay_ok", False))
    qa_after_read = bool(last_qa_node and last_qa_node in read_node_ids)
    last_qa_text = qa_answer(last_qa)
    gold = str(row.get("gold", "") or "").strip()
    qa_unknown = is_unknown_answer(last_qa_text)
    evidence_hit = bool(last_qa_node and target_leaf and last_qa_node == target_leaf)
    qa_gold_support = bool(gold and normalize_text(gold) in normalize_text(last_qa_text))
    soft_evidence_ok = evidence_hit or qa_gold_support
    final_answer = str(row.get("final_answer", "") or "").strip()

    strict_keep = all(
        [
            semantic_ok,
            replay_ok,
            bool(final_answer),
            bool(qa_steps),
            invalid_action_count == 0,
            invalid_final_count == 0,
            qa_after_read,
            not qa_unknown,
            evidence_hit,
        ]
    )
    soft_keep = all(
        [
            semantic_ok,
            replay_ok,
            bool(final_answer),
            bool(qa_steps),
            qa_after_read,
            not qa_unknown,
            soft_evidence_ok,
        ]
    )

    report = {
        "id": sample_id,
        "semantic_ok": semantic_ok,
        "replay_ok": replay_ok,
        "read_steps": len(read_steps),
        "qa_success_count": len(qa_steps),
        "invalid_action_count": invalid_action_count,
        "invalid_final_no_qa_count": invalid_final_count,
        "qa_after_read": qa_after_read,
        "qa_unknown": qa_unknown,
        "last_qa_node_id": last_qa_node,
        "target_leaf_node_id": target_leaf,
        "evidence_hit": evidence_hit,
        "qa_gold_support": qa_gold_support,
        "soft_evidence_ok": soft_evidence_ok,
        "action_after_qa_count": action_after_qa_count,
        "total_steps": len(trajectory),
        "strict_keep": strict_keep,
        "soft_keep": soft_keep,
    }

    strict_row = None
    soft_row = None
    if strict_keep:
        strict_row = {**row, "messages": build_clean_messages(row, terminal_action_steps)}
        strict_row["clean_meta"] = {**report, "clean_level": "strict"}
    if soft_keep:
        soft_row = {**row, "messages": build_clean_messages(row, terminal_action_steps)}
        soft_row["clean_meta"] = {**report, "clean_level": "soft"}
    return report, strict_row, soft_row


def summarize_reports(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate quality metrics."""
    total = len(reports)

    def rate(key: str) -> float:
        return sum(1 for row in reports if bool(row.get(key, False))) / max(total, 1)

    def avg(key: str) -> float:
        return sum(float(row.get(key, 0) or 0) for row in reports) / max(total, 1)

    return {
        "raw_generated": total,
        "strict_clean": sum(1 for row in reports if row.get("strict_keep")),
        "soft_clean": sum(1 for row in reports if row.get("soft_keep")),
        "strict_keep_rate": rate("strict_keep"),
        "soft_keep_rate": rate("soft_keep"),
        "semantic_ok_rate": rate("semantic_ok"),
        "replay_ok_rate": rate("replay_ok"),
        "invalid_action_rate": sum(1 for row in reports if int(row.get("invalid_action_count", 0)) > 0) / max(total, 1),
        "invalid_final_rate": sum(1 for row in reports if int(row.get("invalid_final_no_qa_count", 0)) > 0) / max(total, 1),
        "qa_unknown_rate": rate("qa_unknown"),
        "evidence_hit_rate": rate("evidence_hit"),
        "qa_gold_support_rate": rate("qa_gold_support"),
        "soft_evidence_ok_rate": rate("soft_evidence_ok"),
        "action_after_qa_rate": sum(1 for row in reports if int(row.get("action_after_qa_count", 0)) > 0)
        / max(total, 1),
        "avg_read_steps": avg("read_steps"),
        "avg_total_steps": avg("total_steps"),
    }


def write_report_csv(path: Path, reports: List[Dict[str, Any]]) -> None:
    """Write per-sample quality report."""
    fieldnames = [
        "id",
        "semantic_ok",
        "replay_ok",
        "read_steps",
        "qa_success_count",
        "invalid_action_count",
        "invalid_final_no_qa_count",
        "qa_after_read",
        "qa_unknown",
        "last_qa_node_id",
        "target_leaf_node_id",
        "evidence_hit",
        "qa_gold_support",
        "soft_evidence_ok",
        "action_after_qa_count",
        "total_steps",
        "strict_keep",
        "soft_keep",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in reports:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Clean LongDoc-R1 SFT trajectories")
    parser.add_argument("--sft_file", type=str, required=True)
    parser.add_argument("--label_file", type=str, required=True)
    parser.add_argument("--output_prefix", type=str, required=True)
    parser.add_argument("--chunk_size", type=int, default=300)
    return parser.parse_args()


def main() -> int:
    """Run cleaner."""
    args = parse_args()
    rows = load_jsonl(Path(args.sft_file))
    labels = load_label_map(Path(args.label_file))
    needed_labels = {
        str(row.get("id", "")): labels[str(row.get("id", ""))]
        for row in rows
        if str(row.get("id", "")) in labels
    }
    evidence_map = build_evidence_map(needed_labels, chunk_size=args.chunk_size)

    reports: List[Dict[str, Any]] = []
    strict_rows: List[Dict[str, Any]] = []
    soft_rows: List[Dict[str, Any]] = []
    for row in rows:
        sample_id = str(row.get("id", ""))
        report, strict_row, soft_row = analyze_row(row, evidence_map.get(sample_id, {}))
        reports.append(report)
        if strict_row is not None:
            strict_rows.append(strict_row)
        if soft_row is not None:
            soft_rows.append(soft_row)

    output_prefix = Path(args.output_prefix)
    strict_path = output_prefix.with_name(f"{output_prefix.name}_strict_clean.jsonl")
    soft_path = output_prefix.with_name(f"{output_prefix.name}_soft_clean.jsonl")
    report_path = output_prefix.with_name(f"{output_prefix.name}_quality_report.csv")
    summary_path = output_prefix.with_name(f"{output_prefix.name}_quality_summary.json")

    write_jsonl(strict_path, strict_rows)
    write_jsonl(soft_path, soft_rows)
    write_report_csv(report_path, reports)
    summary = summarize_reports(reports)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"strict: {strict_path}")
    print(f"soft:   {soft_path}")
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
