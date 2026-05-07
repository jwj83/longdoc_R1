#!/usr/bin/env python3
"""Create a high-quality label subset from ids found in SFT trajectory files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Set


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL rows."""
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def category(row: Dict[str, Any]) -> str:
    """Classify a rejected SFT row for recovery."""
    meta = row.get("meta", {}) or {}
    if meta.get("error"):
        return "api_error"
    qa_success = int(meta.get("qa_success_count", 0) or 0)
    answer_fail = int(meta.get("answer_fail_count", 0) or 0)
    if qa_success == 0:
        return "no_qa"
    if answer_fail > 0:
        return "has_qa_wrong_final"
    return "has_qa_no_final"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Filter label rows by ids from an SFT file")
    parser.add_argument("--label_file", type=str, required=True)
    parser.add_argument("--sft_file", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument(
        "--category",
        type=str,
        choices=["all", "api_error", "no_qa", "has_qa_no_final", "has_qa_wrong_final"],
        default="all",
    )
    return parser.parse_args()


def main() -> int:
    """Write filtered labels."""
    args = parse_args()
    sft_rows = load_jsonl(Path(args.sft_file))
    ids: Set[str] = set()
    for row in sft_rows:
        if args.category != "all" and category(row) != args.category:
            continue
        sample_id = str(row.get("id", ""))
        if sample_id:
            ids.add(sample_id)

    label_rows = load_jsonl(Path(args.label_file))
    filtered = [row for row in label_rows if str(row.get("id", "")) in ids]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in filtered:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"ids: {len(ids)}")
    print(f"written: {len(filtered)}")
    print(f"output: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
