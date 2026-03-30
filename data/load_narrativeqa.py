"""Utilities for loading and normalizing a small NarrativeQA subset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import Dataset, load_dataset


def _safe_get_text(value: Any) -> str:
    """Convert nested values to a cleaned string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("text", "summary", "story", "document", "value", "answer"):
            if key in value and value[key]:
                return _safe_get_text(value[key])
        return ""
    if isinstance(value, list):
        for item in value:
            text = _safe_get_text(item)
            if text:
                return text
        return ""
    return str(value).strip()


def _extract_document(example: Dict[str, Any]) -> str:
    """Extract document/story text with compatibility fallbacks."""
    candidates = [
        example.get("document"),
        example.get("story"),
        example.get("context"),
        example.get("article"),
        example.get("passage"),
    ]
    for cand in candidates:
        if isinstance(cand, dict):
            # Prefer full document content before short summaries.
            for key in ("text", "story", "document", "summary"):
                text = _safe_get_text(cand.get(key))
                if text:
                    return text
        text = _safe_get_text(cand)
        if text:
            return text
    return ""


def _extract_question(example: Dict[str, Any]) -> str:
    """Extract question text with compatibility fallbacks."""
    candidates = [example.get("question"), example.get("query"), example.get("input")]
    for cand in candidates:
        text = _safe_get_text(cand)
        if text:
            return text
    return ""


def _extract_answers(example: Dict[str, Any]) -> List[str]:
    """Extract reference answers and return non-empty list."""
    answer_fields = [
        example.get("answers"),
        example.get("answer"),
        example.get("target"),
        example.get("reference"),
    ]
    answers: List[str] = []

    for field in answer_fields:
        if field is None:
            continue
        if isinstance(field, list):
            for item in field:
                text = _safe_get_text(item)
                if text:
                    answers.append(text)
        else:
            text = _safe_get_text(field)
            if text:
                answers.append(text)

    # NarrativeQA variants sometimes store "answer1" / "answer2".
    for key in ("answer1", "answer2"):
        text = _safe_get_text(example.get(key))
        if text:
            answers.append(text)

    deduped: List[str] = []
    seen = set()
    for ans in answers:
        norm = ans.strip()
        if norm and norm not in seen:
            deduped.append(norm)
            seen.add(norm)
    return deduped


def _iter_first_n(ds: Dataset, n: int):
    """Yield at most n examples from a HuggingFace dataset split."""
    limit = min(n, len(ds))
    for idx in range(limit):
        yield idx, ds[idx]


def load_narrativeqa_subset(
    train_size: int = 50,
    validation_size: int = 50,
    cache_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Load and normalize a small NarrativeQA subset.

    Output schema of each sample:
    - id: str
    - split: str
    - document: str
    - question: str
    - answer: str (first reference)
    - answers: List[str]
    """
    dataset = load_dataset("narrativeqa", cache_dir=cache_dir)

    split_aliases = {
        "train": ["train"],
        "validation": ["validation", "valid", "dev"],
    }

    output: Dict[str, List[Dict[str, Any]]] = {"train": [], "validation": []}
    target_sizes = {"train": train_size, "validation": validation_size}

    for out_split, aliases in split_aliases.items():
        ds = None
        for alias in aliases:
            if alias in dataset:
                ds = dataset[alias]
                break
        if ds is None:
            continue

        for idx, example in _iter_first_n(ds, target_sizes[out_split]):
            document = _extract_document(example)
            question = _extract_question(example)
            answers = _extract_answers(example)
            if not document or not question or not answers:
                continue

            sample_id = str(example.get("id", f"{out_split}_{idx}"))
            row = {
                "id": sample_id,
                "split": out_split,
                "document": document,
                "question": question,
                "answer": answers[0],
                "answers": answers,
            }
            output[out_split].append(row)

    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        for split_name, rows in output.items():
            out_file = save_path / f"narrativeqa_{split_name}.jsonl"
            with out_file.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return output
