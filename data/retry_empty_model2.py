#!/usr/bin/env python3
"""Retry model2 generation for empty/failed agreement records."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from data.load_narrativeqa import load_narrativeqa_subset
from data.model_agreement_narrativeqa import (
    chunk_document,
    compute_agreement,
    compute_evidence_agreement,
    format_numbered_chunks_by_ids,
    get_answer,
    parse_structured_response,
    retrieve_candidate_chunk_ids,
)
from llm.client import OpenAICompatibleClient


def parse_sample_id(sample_id: str) -> Tuple[str, int]:
    """Parse sample id like train_123 or validation_45."""
    m = re.match(r"^(train|validation)_(\d+)$", sample_id or "")
    if not m:
        raise ValueError(f"unsupported sample id format: {sample_id}")
    split = m.group(1)
    idx = int(m.group(2))
    return split, idx


def should_retry_model2(model2_output: Dict[str, Any]) -> bool:
    """Return True if model2 output should be regenerated."""
    answer = str(model2_output.get("answer", "") or "").strip()
    parse_error = bool(model2_output.get("parse_error", False))
    return parse_error or (answer == "")


def load_split_samples(split: str, max_index: int) -> List[Dict[str, Any]]:
    """Load enough NarrativeQA samples so index lookup is valid."""
    max_n = max_index + 1
    loaded = load_narrativeqa_subset(train_size=max_n, validation_size=max_n)
    if split not in loaded:
        raise ValueError(f"split not found in loaded data: {split}")
    samples = loaded[split]
    if len(samples) <= max_index:
        raise ValueError(
            f"split={split} only has {len(samples)} samples, need index {max_index}"
        )
    return samples


def _filter_valid_chunk_ids(ids: Any, chunks_len: int) -> List[int]:
    """Keep only valid integer chunk ids."""
    if not isinstance(ids, list):
        return []
    out: List[int] = []
    for cid in ids:
        if isinstance(cid, int) and 0 <= cid < chunks_len:
            out.append(cid)
    return sorted(set(out))


def main() -> int:
    parser = argparse.ArgumentParser(description="Retry empty/failed model2 records")
    parser.add_argument("--agreement_file", type=str, required=True)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--model2", type=str, default=os.getenv("MODEL_2", "MiniMax-M2.7"))
    parser.add_argument("--base_url", type=str, default=os.getenv("BASE_URL", ""))
    parser.add_argument("--api_key", type=str, default=os.getenv("API_KEY", ""))
    parser.add_argument("--chunk_size", type=int, default=300)
    parser.add_argument("--bm25_top_k", type=int, default=5)
    parser.add_argument("--dense_top_k", type=int, default=5)
    parser.add_argument("--embedding_model", type=str, default="all-mpnet-base-v2")
    parser.add_argument("--retry_times", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument(
        "--reuse_existing_candidates",
        action="store_true",
        default=True,
        help="Reuse existing *_candidate_chunk_ids in agreement file when available",
    )
    args = parser.parse_args()

    if not args.base_url or not args.api_key:
        print("ERROR: --base_url and --api_key are required")
        return 2
    if args.retry_times < 1:
        print("ERROR: --retry_times must be >= 1")
        return 2
    if args.save_every < 1:
        print("ERROR: --save_every must be >= 1")
        return 2

    in_path = Path(args.agreement_file)
    if not in_path.exists():
        print(f"ERROR: file not found: {in_path}")
        return 2

    records: List[Dict[str, Any]] = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    retry_indices: List[int] = []
    split_to_max_index: Dict[str, int] = {}
    for i, rec in enumerate(records):
        model2_output = rec.get("model_outputs", {}).get("model2", {})
        if should_retry_model2(model2_output):
            retry_indices.append(i)
            split, idx = parse_sample_id(rec.get("id", ""))
            split_to_max_index[split] = max(split_to_max_index.get(split, -1), idx)

    print(f"Loaded records: {len(records)}")
    print(f"Need retry: {len(retry_indices)}")
    out_path = Path(args.output) if args.output else in_path.with_name(f"{in_path.stem}_retry.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not retry_indices:
        with out_path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"No retries needed. Copied to: {out_path}")
        return 0

    split_cache: Dict[str, List[Dict[str, Any]]] = {}
    for split, max_idx in split_to_max_index.items():
        print(f"Loading split={split} up to index={max_idx}...")
        split_cache[split] = load_split_samples(split=split, max_index=max_idx)

    client2 = OpenAICompatibleClient(
        base_url=args.base_url,
        api_key=args.api_key,
        model_name=args.model2,
    )

    success = 0
    failed = 0
    dense_model: Optional[Any] = None

    for n, rec_idx in enumerate(retry_indices, start=1):
        rec = records[rec_idx]
        sample_id = rec.get("id", "")
        split, idx = parse_sample_id(sample_id)
        sample = split_cache[split][idx]

        doc = sample["document"]
        question = rec.get("question", sample.get("question", ""))
        gold = rec.get("gold", sample.get("answer", ""))

        bm25_k = int(rec.get("bm25_top_k", args.bm25_top_k))
        dense_k = int(rec.get("dense_top_k", args.dense_top_k))

        chunks = chunk_document(doc, chunk_size=args.chunk_size)

        candidate_ids = []
        retrieval: Dict[str, List[int]] = {
            "bm25_ids": [],
            "dense_ids": [],
            "fused_ids": [],
        }

        if args.reuse_existing_candidates:
            existing_fused = _filter_valid_chunk_ids(rec.get("fused_candidate_chunk_ids", []), len(chunks))
            if existing_fused:
                candidate_ids = existing_fused
                retrieval = {
                    "bm25_ids": _filter_valid_chunk_ids(rec.get("bm25_candidate_chunk_ids", []), len(chunks)),
                    "dense_ids": _filter_valid_chunk_ids(rec.get("dense_candidate_chunk_ids", []), len(chunks)),
                    "fused_ids": candidate_ids,
                }

        if not candidate_ids:
            if dense_model is None:
                try:
                    from sentence_transformers import SentenceTransformer

                    print(f"Loading embedding model once: {args.embedding_model}")
                    dense_model = SentenceTransformer(args.embedding_model)
                except Exception as e:
                    print(f"WARNING: failed to load embedding model, dense retrieval disabled: {e}")
                    dense_model = False

            retrieval = retrieve_candidate_chunk_ids(
                chunks=chunks,
                question=question,
                gold=gold,
                bm25_k=bm25_k,
                dense_k=dense_k,
                dense_model=(None if dense_model is False else dense_model),
            )
            candidate_ids = retrieval["fused_ids"]

        numbered_chunks = format_numbered_chunks_by_ids(chunks, candidate_ids)

        last_raw = ""
        parsed: Dict[str, Any] = {
            "selected_chunk_ids": [],
            "answer": "",
            "quoted_evidence": [],
            "parse_error": True,
        }
        for _ in range(args.retry_times):
            try:
                last_raw = get_answer(client2, numbered_chunks, question)
                parsed = parse_structured_response(last_raw, selected_chunk_ids=None, chunks=chunks)
                answer = str(parsed.get("answer", "") or "").strip()
                if (not parsed.get("parse_error", False)) and answer:
                    break
            except Exception as e:
                last_raw = f"<ERROR: {e}>"
                parsed = {
                    "selected_chunk_ids": [],
                    "answer": "",
                    "quoted_evidence": [],
                    "parse_error": True,
                }

        selected_chunk_ids = parsed.get("selected_chunk_ids", [])
        selected_chunk_texts = [
            chunks[cid] for cid in selected_chunk_ids if isinstance(cid, int) and 0 <= cid < len(chunks)
        ]
        answer = str(parsed.get("answer", "") or "")
        quoted_evidence = parsed.get("quoted_evidence", [])
        evidence_verified = bool(parsed.get("evidence_verified", False))
        parse_error = bool(parsed.get("parse_error", False))

        rec["chunks_count"] = len(chunks)
        rec["bm25_top_k"] = bm25_k
        rec["dense_top_k"] = dense_k
        rec["bm25_candidate_chunk_ids"] = retrieval["bm25_ids"]
        rec["dense_candidate_chunk_ids"] = retrieval["dense_ids"]
        rec["fused_candidate_chunk_ids"] = retrieval["fused_ids"]
        rec["fused_candidate_count"] = len(retrieval["fused_ids"])

        rec["model_outputs"]["model2"] = {
            "raw": last_raw,
            "selected_chunk_ids": selected_chunk_ids,
            "selected_chunk_texts": selected_chunk_texts,
            "answer": answer,
            "quoted_evidence": quoted_evidence,
            "evidence_verified": evidence_verified,
            "parse_error": parse_error,
        }

        m1_answer = rec["model_outputs"]["model1"].get("answer", "")
        m2_answer = rec["model_outputs"]["model2"].get("answer", "")
        rec["answer_agreement"] = compute_agreement([m1_answer, m2_answer])

        m1_chunks = rec["model_outputs"]["model1"].get("selected_chunk_ids", [])
        m2_chunks = rec["model_outputs"]["model2"].get("selected_chunk_ids", [])
        evidence_agree = compute_evidence_agreement([m1_chunks, m2_chunks])
        rec["evidence_agreement"] = evidence_agree
        rec["evidence_votes"] = {
            "chunk_vote_counts": evidence_agree["chunk_vote_counts"],
            "core_evidence_chunk_ids": evidence_agree["core_evidence_chunk_ids"],
            "union_evidence_chunk_ids": evidence_agree["union_evidence_chunk_ids"],
            "intersection_evidence_chunk_ids": evidence_agree["intersection_evidence_chunk_ids"],
        }

        if parse_error or not answer.strip():
            failed += 1
            status = "FAIL"
        else:
            success += 1
            status = "OK"

        if n % 10 == 0 or n == len(retry_indices):
            print(f"[{n}/{len(retry_indices)}] {status} success={success} failed={failed}")

        if n % args.save_every == 0 or n == len(retry_indices):
            with out_path.open("w", encoding="utf-8") as f:
                for item in records:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"Checkpoint saved at {n}/{len(retry_indices)} -> {out_path}")

    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("\n===== Retry Summary =====")
    print(f"Input: {in_path}")
    print(f"Output: {out_path}")
    print(f"Retried: {len(retry_indices)}")
    print(f"Recovered: {success}")
    print(f"Still failed/empty: {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
