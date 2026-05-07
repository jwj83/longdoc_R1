#!/usr/bin/env python3
"""Sample NarrativeQA data: pick N documents, take 10 questions from each."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from datasets import load_dataset


def safe_text(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, dict):
        for k in ("text", "story", "document", "summary", "answer"):
            if k in val and val[k]:
                return safe_text(val[k])
        return ""
    if isinstance(val, list):
        for item in val:
            t = safe_text(item)
            if t:
                return t
    return str(val).strip()


def sample_data(
    train_docs: int = 30,
    test_docs: int = 20,
    questions_per_doc: int = 10,
    output_dir: str = "outputs/data",
    seed: int = 42,
) -> None:
    random.seed(seed)

    print("Loading NarrativeQA...")
    ds = load_dataset("narrativeqa")

    # Group questions by document ID
    doc_map: Dict[str, Dict[str, Any]] = {}
    for split_name in ["train", "validation", "test"]:
        if split_name not in ds:
            continue
        for ex in ds[split_name]:
            doc = ex.get("document", {})
            if not isinstance(doc, dict):
                continue
            doc_id = doc.get("id", "")
            if not doc_id:
                continue
            doc_text = safe_text(doc.get("text"))
            if not doc_text:
                continue
            if doc_id not in doc_map:
                doc_map[doc_id] = {"text": doc_text, "split": split_name, "questions": []}
            q = safe_text(ex.get("question"))
            answers = []
            for field in [ex.get("answers"), ex.get("answer")]:
                if isinstance(field, list):
                    for item in field:
                        t = safe_text(item)
                        if t:
                            answers.append(t)
                elif isinstance(field, str) and field.strip():
                    answers.append(field.strip())
            for key in ("answer1", "answer2"):
                t = safe_text(ex.get(key))
                if t:
                    answers.append(t)
            seen = set()
            deduped = [a for a in answers if a and a not in seen and not seen.add(a)]
            if q and deduped:
                doc_map[doc_id]["questions"].append({
                    "question": q,
                    "answer": deduped[0],
                    "answers": deduped,
                })

    print(f"Total unique documents with text: {len(doc_map)}")

    # Filter: only keep docs with >= questions_per_doc questions
    eligible = {k: v for k, v in doc_map.items() if len(v["questions"]) >= questions_per_doc}
    print(f"Documents with >= {questions_per_doc} questions: {len(eligible)}")

    # Split by original split
    train_docs_map = {k: v for k, v in eligible.items() if v["split"] == "train"}
    other_docs_map = {k: v for k, v in eligible.items() if v["split"] in ("validation", "test")}

    print(f"Train-eligible docs: {len(train_docs_map)}")
    print(f"Val/Test-eligible docs: {len(other_docs_map)}")

    # Sample documents
    train_doc_ids = random.sample(
        list(train_docs_map.keys()),
        min(train_docs, len(train_docs_map)),
    )
    test_doc_ids = random.sample(
        list(other_docs_map.keys()),
        min(test_docs, len(other_docs_map)),
    )

    print(f"\nSampled {len(train_doc_ids)} train docs, {len(test_doc_ids)} test docs")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for split_name, doc_ids in [("train", train_doc_ids), ("test", test_doc_ids)]:
        samples = []
        for doc_id in doc_ids:
            doc_info = doc_map[doc_id]
            selected_qs = random.sample(doc_info["questions"], questions_per_doc)
            for i, qa in enumerate(selected_qs):
                samples.append({
                    "id": f"{split_name}_{doc_id}_{i}",
                    "doc_id": doc_id,
                    "split": split_name,
                    "document": doc_info["text"],
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "answers": qa["answers"],
                })

        out_file = out_path / f"narrativeqa_{split_name}_sampled.jsonl"
        with out_file.open("w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        print(f"Saved {len(samples)} samples to {out_file}")

    # Save doc mapping for reference
    doc_info_file = out_path / "sampled_doc_ids.json"
    with doc_info_file.open("w") as f:
        json.dump({
            "train": train_doc_ids,
            "test": test_doc_ids,
        }, f, indent=2)
    print(f"Saved doc IDs to {doc_info_file}")


if __name__ == "__main__":
    sample_data()
