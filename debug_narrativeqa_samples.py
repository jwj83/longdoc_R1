"""Inspect raw NarrativeQA examples and normalized extraction outputs.

This script helps verify whether the loader extracts full stories or summaries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from datasets import load_dataset

from data.load_narrativeqa import _extract_answers, _extract_document, _extract_question


def _to_preview(value: Any, limit: int = 180) -> str:
    """Return a short one-line preview for debug printing."""
    if value is None:
        return "<None>"
    text = str(value).replace("\n", " ")
    return text[:limit] + ("..." if len(text) > limit else "")


def _text_len(value: Any) -> int:
    """Return word length of a value after string conversion."""
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value.split())
    return len(str(value).split())


def _describe_document_field(example: Dict[str, Any]) -> Dict[str, Any]:
    """Show detailed structure under the raw `document` field."""
    doc = example.get("document")
    result: Dict[str, Any] = {"document_type": type(doc).__name__}

    if isinstance(doc, dict):
        result["document_keys"] = sorted(doc.keys())
        for key in ["story", "summary", "text", "document"]:
            if key in doc:
                result[f"document.{key}.words"] = _text_len(doc.get(key))
                result[f"document.{key}.preview"] = _to_preview(doc.get(key))
    else:
        result["document.words"] = _text_len(doc)
        result["document.preview"] = _to_preview(doc)

    return result


def inspect_split(
    split_name: str,
    num_samples: int,
    cache_dir: str | None = None,
    output_jsonl: str | None = None,
) -> None:
    """Print raw and normalized views and optionally save to JSONL."""
    ds_dict = load_dataset("narrativeqa", cache_dir=cache_dir)
    if split_name not in ds_dict:
        raise ValueError(f"Split '{split_name}' not found. Available: {list(ds_dict.keys())}")

    ds = ds_dict[split_name]
    n = min(num_samples, len(ds))
    print(f"Loaded split={split_name}, total={len(ds)}, inspect_first_n={n}")
    records: List[Dict[str, Any]] = []

    for i in range(n):
        ex = ds[i]
        extracted_document = _extract_document(ex)
        extracted_question = _extract_question(ex)
        extracted_answers = _extract_answers(ex)

        print("\n" + "=" * 88)
        print(f"index={i}")
        print(f"raw_keys={sorted(ex.keys())}")

        doc_desc = _describe_document_field(ex)
        normalized = {
            "document_words": len(extracted_document.split()),
            "document_preview": _to_preview(extracted_document),
            "question": extracted_question,
            "answers": extracted_answers,
        }

        record = {
            "index": i,
            "split": split_name,
            "raw_keys": sorted(ex.keys()),
            "raw_document_info": doc_desc,
            "normalized_extraction": normalized,
        }
        records.append(record)

        print("raw_document_info=")
        print(json.dumps(doc_desc, ensure_ascii=False, indent=2))

        print("normalized_extraction=")
        print(json.dumps(normalized, ensure_ascii=False, indent=2))

    if output_jsonl:
        out_path = Path(output_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for row in records:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\nSaved debug records to: {out_path}")


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description="Inspect NarrativeQA raw fields vs loader extraction.")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default="long_doc_agent/outputs/data/narrativeqa_debug_samples.jsonl",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    inspect_split(
        split_name=args.split,
        num_samples=args.num_samples,
        cache_dir=args.cache_dir,
        output_jsonl=args.output_jsonl,
    )
