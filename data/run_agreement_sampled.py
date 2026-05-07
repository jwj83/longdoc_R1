#!/usr/bin/env python3
"""Run 2-model agreement on sampled NarrativeQA data for CoTwT training data generation."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from sentence_transformers import SentenceTransformer, util

from llm.client import OpenAICompatibleClient


def normalize_text(text: str) -> str:
    text = text or ""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^0-9a-z\u4e00-\u9fff ]", "", text)
    return text.strip()


def chunk_document(document: str, chunk_size: int = 300) -> List[str]:
    words = document.strip().split()
    if not words:
        return []
    chunks: List[str] = []
    for start in range(0, len(words), chunk_size):
        end = start + chunk_size
        chunk_words = words[start:end]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        if end >= len(words):
            break
    return chunks


def format_numbered_chunks_by_ids(chunks: List[str], chunk_ids: List[int]) -> str:
    blocks = []
    for idx in chunk_ids:
        if 0 <= idx < len(chunks):
            blocks.append(f"[Chunk {idx}]\n{chunks[idx]}")
    return "\n\n".join(blocks)


def tokenize_for_bm25(text: str) -> List[str]:
    text = (text or "").lower()
    return re.findall(r"[a-z0-9]+", text)


def bm25_top_k(chunks: List[str], query: str, top_k: int = 5) -> List[int]:
    if not chunks or top_k <= 0:
        return []
    tokenized_docs = [tokenize_for_bm25(c) for c in chunks]
    query_tokens = tokenize_for_bm25(query)
    if not query_tokens:
        return list(range(min(top_k, len(chunks))))

    doc_count = len(tokenized_docs)
    avgdl = sum(len(doc) for doc in tokenized_docs) / max(doc_count, 1)
    k1, b = 1.5, 0.75

    df: Dict[str, int] = {}
    for doc in tokenized_docs:
        for t in set(doc):
            df[t] = df.get(t, 0) + 1

    idf: Dict[str, float] = {}
    for t, n_qi in df.items():
        idf[t] = math.log(1.0 + (doc_count - n_qi + 0.5) / (n_qi + 0.5))

    scores: List[float] = []
    for doc in tokenized_docs:
        doc_len = len(doc)
        tf_counter = Counter(doc)
        score = 0.0
        for t in query_tokens:
            if t not in tf_counter:
                continue
            tf = tf_counter[t]
            numerator = tf * (k1 + 1.0)
            denominator = tf + k1 * (1.0 - b + b * (doc_len / max(avgdl, 1e-9)))
            score += idf.get(t, 0.0) * (numerator / max(denominator, 1e-9))
        scores.append(score)

    ranked_ids = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    return ranked_ids[: min(top_k, len(ranked_ids))]


def dense_top_k(
    chunks: List[str],
    query: str,
    top_k: int = 5,
    model: Optional[SentenceTransformer] = None,
) -> List[int]:
    if not chunks or top_k <= 0:
        return []
    if model is None:
        model = SentenceTransformer("all-mpnet-base-v2")
    chunk_embeddings = model.encode(chunks, convert_to_tensor=True, show_progress_bar=False)
    query_embedding = model.encode(query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, chunk_embeddings)[0]
    top_results = torch.topk(cos_scores, k=min(top_k, len(chunks)))
    return top_results.indices.cpu().tolist()


def retrieve_candidate_chunk_ids(
    chunks: List[str],
    question: str,
    gold: str,
    bm25_k: int = 5,
    dense_k: int = 5,
    dense_model: Optional[SentenceTransformer] = None,
) -> Dict[str, List[int]]:
    top_q = bm25_top_k(chunks, question, top_k=bm25_k)
    q_plus_gold = f"{question} {gold}".strip()
    top_qg = bm25_top_k(chunks, q_plus_gold, top_k=bm25_k) if gold else []
    bm25_ids = sorted(set(top_q) | set(top_qg))
    dense_ids = dense_top_k(chunks, question, top_k=dense_k, model=dense_model)
    fused_ids = sorted(set(bm25_ids) | set(dense_ids))
    if not fused_ids:
        fused_ids = list(range(len(chunks)))
    return {"bm25_ids": bm25_ids, "dense_ids": dense_ids, "fused_ids": fused_ids}


def prompt_for_qa(numbered_chunks: str, question: str) -> str:
    return f"""你是一个证据驱动的问答助手。你会得到 NarrativeQA 文档的完整分块列表（已编号）。

任务要求：
1. 必须优先只选择 1 个最相关的 chunk 编号；仅在绝对必要时可选择 2 个。
2. 只能根据你选择的 chunk 回答问题。
3. 如果证据不足，答案必须且只能是：无法从给定文本中判断
4. 必须从你选中的 chunk 中逐字引用包含答案证据的原文句子（每个选中 chunk 至少引用 1 句）。
5. 引用内容必须与原文完全一致，不得编造或修改。
6. 只输出严格 JSON，禁止任何额外文本、解释、Markdown。

输出 JSON 格式（严格遵守）：
{{
  "selected_chunk_ids": [12, 13],
  "answer": "...",
  "quoted_evidence": [
    "从 chunk 12 中逐字引用的原文句子",
    "从 chunk 13 中逐字引用的原文句子"
  ]
}}

全文分块如下：
{numbered_chunks}

问题：
{question}
"""


def validate_quoted_evidence(
    quoted_evidence: List[str],
    selected_chunk_ids: List[int],
    chunks: List[str],
) -> Tuple[bool, List[bool]]:
    if not quoted_evidence or not selected_chunk_ids:
        return False, []
    verified_per_quote = []
    for quote in quoted_evidence:
        found = False
        quote_normalized = normalize_text(quote)
        for cid in selected_chunk_ids:
            if 0 <= cid < len(chunks):
                chunk_normalized = normalize_text(chunks[cid])
                if quote_normalized in chunk_normalized:
                    found = True
                    break
        verified_per_quote.append(found)
    return all(verified_per_quote), verified_per_quote


def parse_structured_response(
    raw: str,
    chunks: Optional[List[str]] = None,
) -> Dict[str, Any]:
    raw = raw.strip()
    if not raw:
        return {"selected_chunk_ids": [], "answer": "", "quoted_evidence": [], "parse_error": True}
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("response is not a JSON object")

        selected_chunk_ids_parsed = data.get("selected_chunk_ids")
        answer = data.get("answer")
        quoted_evidence = data.get("quoted_evidence", [])

        if not isinstance(selected_chunk_ids_parsed, list):
            raise ValueError("selected_chunk_ids must be a list")
        if not (1 <= len(selected_chunk_ids_parsed) <= 2):
            raise ValueError("selected_chunk_ids length must be between 1 and 2")
        if not all(isinstance(item, int) for item in selected_chunk_ids_parsed):
            raise ValueError("selected_chunk_ids must contain only integers")
        if not isinstance(answer, str):
            raise ValueError("answer must be a string")
        if not isinstance(quoted_evidence, list):
            raise ValueError("quoted_evidence must be a list")

        all_verified = False
        verified_per_quote = []
        if chunks is not None and quoted_evidence:
            all_verified, verified_per_quote = validate_quoted_evidence(
                quoted_evidence, selected_chunk_ids_parsed, chunks
            )

        return {
            "selected_chunk_ids": selected_chunk_ids_parsed,
            "answer": answer,
            "quoted_evidence": quoted_evidence,
            "evidence_verified": all_verified,
            "verified_per_quote": verified_per_quote,
            "parse_error": False,
        }
    except (json.JSONDecodeError, ValueError, TypeError):
        return {"selected_chunk_ids": [], "answer": "", "quoted_evidence": [], "parse_error": True}


def compute_agreement(answer_texts: List[str]) -> Dict[str, Any]:
    normalized = [normalize_text(t) for t in answer_texts]
    uniq = set(normalized)
    all_same = len(uniq) == 1
    c = Counter(normalized)
    pairwise_agree = sum(1 for k, v in c.items() if v > 1)
    return {
        "all_same": all_same,
        "unique_count": len(uniq),
        "majority_candidate": c.most_common(1)[0][0] if c else "",
        "pairwise_agree_groups": pairwise_agree,
    }


def compute_evidence_agreement(selected_chunk_sets: List[List[int]]) -> Dict[str, Any]:
    normalized_sets = [set(x) for x in selected_chunk_sets]
    normalized = [tuple(sorted(s)) for s in normalized_sets]
    uniq = set(normalized)
    all_same = len(uniq) == 1
    if normalized_sets:
        intersection = set(normalized_sets[0])
        for s in normalized_sets[1:]:
            intersection &= s
    else:
        intersection = set()
    union = set().union(*normalized_sets) if normalized_sets else set()
    vote_counter = Counter()
    for s in normalized_sets:
        for cid in s:
            vote_counter[cid] += 1
    chunk_vote_counts = {str(k): v for k, v in sorted(vote_counter.items())}
    core_evidence_chunk_ids = sorted([cid for cid, v in vote_counter.items() if v >= 2])
    return {
        "all_same": all_same,
        "unique_count": len(uniq),
        "intersection_evidence_chunk_ids": sorted(list(intersection)),
        "union_evidence_chunk_ids": sorted(list(union)),
        "core_evidence_chunk_ids": core_evidence_chunk_ids,
        "chunk_vote_counts": chunk_vote_counts,
    }


def load_sampled_samples(input_file: str) -> List[Dict[str, Any]]:
    samples = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description="NarrativeQA 2-model agreement on sampled data")
    parser.add_argument("--input", type=str, required=True, help="Path to sampled JSONL file")
    parser.add_argument("--output", type=str, default="outputs/model_agreement_500.jsonl")
    parser.add_argument("--model1", type=str, default=os.getenv("MODEL_1", "gpt-4.1"))
    parser.add_argument("--model2", type=str, default=os.getenv("MODEL_2", "gpt-4o-mini"))
    parser.add_argument("--base_url", type=str, default=os.getenv("BASE_URL", ""))
    parser.add_argument("--api_key", type=str, default=os.getenv("API_KEY", ""))
    parser.add_argument("--chunk_size", type=int, default=300)
    parser.add_argument("--bm25_top_k", type=int, default=5)
    parser.add_argument("--dense_top_k", type=int, default=5)
    parser.add_argument("--embedding_model", type=str, default="all-mpnet-base-v2")
    parser.add_argument("--resume_from", type=int, default=0, help="Resume from sample index")

    args = parser.parse_args()

    if not args.base_url or not args.api_key:
        print("ERROR: --base_url and --api_key are required (or set BASE_URL/API_KEY env)")
        return 2

    print(f"Loading embedding model: {args.embedding_model}...")
    dense_model = SentenceTransformer(args.embedding_model)

    clients = {
        "model1": OpenAICompatibleClient(base_url=args.base_url, api_key=args.api_key, model_name=args.model1),
        "model2": OpenAICompatibleClient(base_url=args.base_url, api_key=args.api_key, model_name=args.model2),
    }

    print(f"Loading sampled data from {args.input}...")
    samples = load_sampled_samples(args.input)
    print(f"Loaded {len(samples)} samples")

    # Resume support
    existing_records = []
    if args.resume_from > 0 and Path(args.output).exists():
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    existing_records.append(json.loads(line))
        print(f"Resuming from {len(existing_records)} existing records")
        samples = samples[len(existing_records):]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    summary = Counter()
    out_records = existing_records

    for i, sample in enumerate(samples, start=1):
        doc = sample["document"]
        q = sample["question"]
        gold = sample.get("answer", "")

        chunks = chunk_document(doc, chunk_size=args.chunk_size)
        retrieval_results = retrieve_candidate_chunk_ids(
            chunks=chunks,
            question=q,
            gold=gold,
            bm25_k=args.bm25_top_k,
            dense_k=args.dense_top_k,
            dense_model=dense_model,
        )
        candidate_chunk_ids = retrieval_results["fused_ids"]
        numbered_chunks = format_numbered_chunks_by_ids(chunks, candidate_chunk_ids)

        model_results = {}
        answers = []
        selected_chunk_ids_list = []

        for name, client in clients.items():
            messages = [
                {"role": "system", "content": "你是一个严格 JSON 输出助手。"},
                {"role": "user", "content": prompt_for_qa(numbered_chunks, q)},
            ]
            try:
                raw = client.generate(messages=messages, temperature=0.0, max_tokens=1024)
                parsed = parse_structured_response(raw, chunks=chunks)

                selected_chunk_ids = parsed.get("selected_chunk_ids", [])
                selected_chunk_texts = [
                    chunks[idx] for idx in selected_chunk_ids if isinstance(idx, int) and 0 <= idx < len(chunks)
                ]

                answer = parsed.get("answer", "")
                quoted_evidence = parsed.get("quoted_evidence", [])
                evidence_verified = parsed.get("evidence_verified", False)
                parse_error = bool(parsed.get("parse_error", False))
            except Exception as e:
                raw = f"<ERROR: {e}>"
                selected_chunk_ids = []
                selected_chunk_texts = []
                answer = ""
                quoted_evidence = []
                evidence_verified = False
                parse_error = True

            model_results[name] = {
                "raw": raw,
                "selected_chunk_ids": selected_chunk_ids,
                "selected_chunk_texts": selected_chunk_texts,
                "answer": answer,
                "quoted_evidence": quoted_evidence,
                "evidence_verified": evidence_verified,
                "parse_error": parse_error,
            }
            answers.append(answer)
            selected_chunk_ids_list.append(selected_chunk_ids)

            if parse_error:
                summary["parse_error_count"] += 1

        answer_agree = compute_agreement(answers)
        evidence_agree = compute_evidence_agreement(selected_chunk_ids_list)

        if answer_agree["all_same"]:
            summary["answer_all_same"] += 1
        summary[f"answer_unique_{answer_agree['unique_count']}"] += 1

        if evidence_agree["all_same"]:
            summary["evidence_all_same"] += 1
        summary[f"evidence_unique_{evidence_agree['unique_count']}"] += 1

        rec = {
            "id": sample["id"],
            "doc_id": sample.get("doc_id", ""),
            "split": sample.get("split", ""),
            "question": q,
            "gold": gold,
            "chunks_count": len(chunks),
            "bm25_top_k": args.bm25_top_k,
            "dense_top_k": args.dense_top_k,
            "bm25_candidate_chunk_ids": retrieval_results["bm25_ids"],
            "dense_candidate_chunk_ids": retrieval_results["dense_ids"],
            "fused_candidate_chunk_ids": retrieval_results["fused_ids"],
            "fused_candidate_count": len(retrieval_results["fused_ids"]),
            "model_outputs": model_results,
            "answer_agreement": answer_agree,
            "evidence_agreement": evidence_agree,
            "evidence_votes": {
                "chunk_vote_counts": evidence_agree["chunk_vote_counts"],
                "core_evidence_chunk_ids": evidence_agree["core_evidence_chunk_ids"],
                "union_evidence_chunk_ids": evidence_agree["union_evidence_chunk_ids"],
                "intersection_evidence_chunk_ids": evidence_agree["intersection_evidence_chunk_ids"],
            },
        }
        out_records.append(rec)

        # Write incrementally
        with out_path.open("w", encoding="utf-8") as f:
            for r in out_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        if i % 10 == 0:
            print(
                f"[{len(out_records)}/{len(existing_records) + len(samples)}] "
                f"answer_all_same={summary['answer_all_same']}, "
                f"evidence_all_same={summary['evidence_all_same']}, "
                f"parse_errors={summary['parse_error_count']}"
            )

    print("\n===== 统计结果 =====")
    print(f"总样本数: {len(out_records)}")
    print(f"answer_all_same: {summary['answer_all_same']}")
    print(f"evidence_all_same: {summary['evidence_all_same']}")
    print(f"parse_error_count: {summary['parse_error_count']}")
    print(f"answer_unique_count: {dict({k: v for k, v in summary.items() if k.startswith('answer_unique_')})}")
    print(f"evidence_unique_count: {dict({k: v for k, v in summary.items() if k.startswith('evidence_unique_')})}")
    print(f"输出路径: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
