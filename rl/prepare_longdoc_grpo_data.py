#!/usr/bin/env python3
"""Prepare LongDoc-R1 trajectory seeds as verl GRPO parquet tasks."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.prompt import build_initial_messages
from data.clean_sft_trajectories import load_samples_for_ids
from data.sft_data_generation_longdoc import build_tree_shape_info
from tree.build_tree import build_hierarchical_tree


class EnvLike:
    """Tiny adapter for prompt helpers that only need nodes and root_id."""

    def __init__(self, nodes: Dict[str, Any], root_id: str) -> None:
        self.nodes = nodes
        self.root_id = root_id

    def get_node(self, node_id: str) -> Any:
        """Return one tree node."""
        return self.nodes.get(node_id)


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """Yield JSON objects from a JSONL file."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="outputs/train_COT/train_sft_v2_clue2_184_plus_clue3_30_soft_clean.jsonl",
        type=Path,
    )
    parser.add_argument("--output_dir", default="outputs/rl_data/longdoc_grpo_214", type=Path)
    parser.add_argument("--train_size", default=200, type=int)
    parser.add_argument("--max_rows", default=0, type=int)
    parser.add_argument("--clue_level", default=2, type=int)
    parser.add_argument("--overview_max_summary_chars", default=0, type=int)
    return parser.parse_args()


def main() -> int:
    """Create train/test parquet files for verl."""
    args = parse_args()
    rows = list(read_jsonl(args.input))
    if args.max_rows > 0:
        rows = rows[: args.max_rows]

    sample_ids: Set[str] = {str(row["id"]) for row in rows}
    loaded = load_samples_for_ids(sample_ids)

    out_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        sample_id = str(row["id"])
        split, sample_index_raw = sample_id.split("_", 1)
        sample_index = int(sample_index_raw)
        sample = loaded[split][sample_index]
        document = str(sample.get("document", "") or "")
        question = str(row.get("question", sample.get("question", "")) or "").strip()
        gold = str(row.get("gold", sample.get("answer", "")) or "").strip()
        root_id, nodes = build_hierarchical_tree(document_text=document, doc_id=sample_id)
        prompt = build_initial_messages(
            question=question,
            root_id=root_id,
            tree_shape_info=build_tree_shape_info(EnvLike(nodes=nodes, root_id=root_id)),
        )

        clean_meta = row.get("clean_meta", {}) or {}
        target_index_tuple = []
        for step in row.get("trajectory", []) or []:
            if step.get("kind") == "initial_clue":
                target_index_tuple = list(step.get("target_index_tuple", []) or [])
                break

        ground_truth = {
            "gold": gold,
            "target_leaf_node_id": clean_meta.get("target_leaf_node_id", ""),
            "target_index_tuple": target_index_tuple,
        }
        extra_info = {
            "split": split,
            "index": idx,
            "sample_id": sample_id,
            "question": question,
            "gold": gold,
            "document": document,
            "root_id": root_id,
            "target_leaf_node_id": clean_meta.get("target_leaf_node_id", ""),
            "target_index_tuple": target_index_tuple,
            "clue_level": int(args.clue_level),
            "overview_max_summary_chars": int(args.overview_max_summary_chars),
        }
        out_rows.append(
            {
                "data_source": "longdoc_r1",
                "agent_name": "longdoc_xml_agent",
                "prompt": prompt,
                "ability": "longdoc_qa",
                "reward_model": {"style": "rule", "ground_truth": json.dumps(ground_truth, ensure_ascii=False)},
                "extra_info": extra_info,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = out_rows[: args.train_size]
    test_rows = out_rows[args.train_size :] or out_rows[: min(14, len(out_rows))]
    pd.DataFrame(train_rows).to_parquet(args.output_dir / "train.parquet", index=False)
    pd.DataFrame(test_rows).to_parquet(args.output_dir / "test.parquet", index=False)

    summary = {
        "input": str(args.input),
        "output_dir": str(args.output_dir),
        "total": len(out_rows),
        "train": len(train_rows),
        "test": len(test_rows),
        "clue_level": args.clue_level,
        "overview_max_summary_chars": args.overview_max_summary_chars,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
