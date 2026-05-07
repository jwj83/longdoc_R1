#!/usr/bin/env python3
"""LLM-based semantic consistency checker.

For samples where both models selected the same chunk(s),
send question + gold_answer + model_answers to an LLM to judge
whether the model answers are semantically consistent with the gold answer.

Output: high_quality_labels.jsonl with only samples that pass the check.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List

from llm.client import OpenAICompatibleClient


def normalize_text(text: str) -> str:
    text = text or ""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^0-9a-z\u4e00-\u9fff ]", "", text)
    return text.strip()


def prompt_for_judge(question: str, gold: str, model1_answer: str, model2_answer: str) -> str:
    return f"""你是一个问答答案一致性判断器。请分别独立判断两个模型答案是否与标准答案在语义上一致。

判断标准：
1. 如果模型答案与标准答案表达相同的意思，即使措辞不同，也判定为一致。
2. 如果模型答案包含标准答案的关键信息，判定为一致。
3. 如果模型答案与标准答案矛盾或完全无关，判定为不一致。
4. 如果模型回答"无法从给定文本中判断"，但标准答案有明确内容，判定为不一致。
5. 如果标准答案和模型答案都表示"无法判断"，判定为一致。
6. 必须对模型1和模型2分别判断，不得因为另一个模型的答案影响当前判断。

问题：{question}
标准答案：{gold}
模型1答案：{model1_answer}
模型2答案：{model2_answer}

请只输出严格JSON：
{{
  "model1_consistent": true/false,
  "model1_reason": "简要说明判断理由",
  "model2_consistent": true/false,
  "model2_reason": "简要说明判断理由"
}}
"""


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def judge_consistency(
    client: OpenAICompatibleClient,
    question: str,
    gold: str,
    model1_answer: str,
    model2_answer: str,
) -> Dict[str, Any]:
    """Judge whether model1/model2 answers are semantically consistent with gold."""
    judge_prompt = prompt_for_judge(question, gold, model1_answer, model2_answer)
    raw = client.generate(
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0.0,
        max_tokens=256,
    )

    cleaned = raw.strip()
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", cleaned, re.DOTALL)
    if m:
        cleaned = m.group(1).strip()

    if not cleaned.startswith("{"):
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start : end + 1]

    judge_result = json.loads(cleaned)
    return {
        "model1_consistent": bool(judge_result.get("model1_consistent", False)),
        "model1_reason": str(judge_result.get("model1_reason", "")),
        "model2_consistent": bool(judge_result.get("model2_consistent", False)),
        "model2_reason": str(judge_result.get("model2_reason", "")),
        "raw_judge": cleaned,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM semantic consistency checker")
    parser.add_argument("--agreement_file", type=str, required=True, help="Path to model_agreement_*_fixed.jsonl")
    parser.add_argument("--output", type=str, default=None, help="Output path for high quality labels")
    parser.add_argument(
        "--consistency_mode",
        type=str,
        choices=["both", "any", "both_any"],
        default="both_any",
        help="both: intersection only; any: union only; both_any: write both in one run",
    )
    parser.add_argument("--model", type=str, default=os.getenv("JUDGE_MODEL", "qwen3.5-plus-2026-02-15"))
    parser.add_argument("--base_url", type=str, default=os.getenv("BASE_URL", ""))
    parser.add_argument("--api_key", type=str, default=os.getenv("API_KEY", ""))

    args = parser.parse_args()

    if not args.base_url or not args.api_key:
        print("ERROR: --base_url and --api_key are required")
        return 2

    records = load_jsonl(args.agreement_file)
    print(f"Loaded {len(records)} records from {args.agreement_file}")

    # Filter: both models parse OK and selected same chunk(s)
    same_chunk_records = []
    for rec in records:
        m1 = rec["model_outputs"]["model1"]
        m2 = rec["model_outputs"]["model2"]
        if not m1.get("parse_error") and not m2.get("parse_error"):
            c1 = set(m1.get("selected_chunk_ids", []))
            c2 = set(m2.get("selected_chunk_ids", []))
            if c1 == c2 and len(c1) > 0:
                same_chunk_records.append(rec)

    print(f"Same-chunk samples: {len(same_chunk_records)}")
    print(f"Consistency mode: {args.consistency_mode}")

    client = OpenAICompatibleClient(
        base_url=args.base_url, api_key=args.api_key, model_name=args.model
    )

    high_quality_both: List[Dict[str, Any]] = []
    high_quality_any: List[Dict[str, Any]] = []
    consistent_count_both = 0
    consistent_count_any = 0
    error_count = 0

    total = len(same_chunk_records)
    for i, rec in enumerate(same_chunk_records, start=1):
        question = rec["question"]
        gold = rec.get("gold", "")
        m1_answer = rec["model_outputs"]["model1"].get("answer", "")
        m2_answer = rec["model_outputs"]["model2"].get("answer", "")

        try:
            judge_result = judge_consistency(client, question, gold, m1_answer, m2_answer)

            m1_consistent = bool(judge_result["model1_consistent"])
            m2_consistent = bool(judge_result["model2_consistent"])

            if args.consistency_mode == "both":
                is_consistent = m1_consistent and m2_consistent
            elif args.consistency_mode == "any":
                is_consistent = m1_consistent or m2_consistent
            else:
                is_consistent = (m1_consistent and m2_consistent) or (m1_consistent or m2_consistent)

            keep_both = m1_consistent and m2_consistent
            keep_any = m1_consistent or m2_consistent

            rec["judge_result"] = {
                "consistent": is_consistent,
                "consistency_mode": args.consistency_mode,
                "judge_model": args.model,
                "model1_answer": m1_answer,
                "model2_answer": m2_answer,
                "model1_consistent": m1_consistent,
                "model1_reason": judge_result["model1_reason"],
                "model2_consistent": m2_consistent,
                "model2_reason": judge_result["model2_reason"],
                "raw_judge": judge_result["raw_judge"],
                "keep_both": keep_both,
                "keep_any": keep_any,
            }

            if keep_both:
                consistent_count_both += 1
                high_quality_both.append(rec)
            if keep_any:
                consistent_count_any += 1
                high_quality_any.append(rec)

            status = "PASS" if is_consistent else "FAIL"

        except Exception as e:
            error_count += 1
            status = "ERROR"
            rec["judge_result"] = {"consistent": False, "reason": str(e), "error": True}

        if i % 10 == 0 or i == total:
            pct = i / total * 100
            print(
                f"[{i}/{total}] ({pct:.1f}%) BOTH={consistent_count_both} "
                f"ANY={consistent_count_any} ERR={error_count}"
            )

        # Rate limiting
        time.sleep(0.3)

    input_path = Path(args.agreement_file)
    if args.consistency_mode == "both_any":
        out_path_both = (
            Path(args.output)
            if args.output is not None
            else input_path.parent / f"high_quality_labels_both_{input_path.stem}.jsonl"
        )
        if args.output is not None:
            out_path_any = out_path_both.with_name(f"{out_path_both.stem}_any{out_path_both.suffix}")
        else:
            out_path_any = input_path.parent / f"high_quality_labels_any_{input_path.stem}.jsonl"

        out_path_both.parent.mkdir(parents=True, exist_ok=True)
        out_path_any.parent.mkdir(parents=True, exist_ok=True)

        with out_path_both.open("w", encoding="utf-8") as f:
            for rec in high_quality_both:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        with out_path_any.open("w", encoding="utf-8") as f:
            for rec in high_quality_any:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    else:
        if args.output is None:
            out_path = input_path.parent / f"high_quality_labels_{args.consistency_mode}_{input_path.stem}.jsonl"
        else:
            out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        selected = high_quality_both if args.consistency_mode == "both" else high_quality_any
        with out_path.open("w", encoding="utf-8") as f:
            for rec in selected:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n===== Results =====")
    print(f"Same-chunk samples: {total}")
    denom = max(total, 1)
    print(f"Both-consistent(intersection): {consistent_count_both} ({consistent_count_both/denom*100:.1f}%)")
    print(f"Any-consistent(union): {consistent_count_any} ({consistent_count_any/denom*100:.1f}%)")
    print(f"Errors: {error_count} ({error_count/denom*100:.1f}%)")
    if args.consistency_mode == "both_any":
        print(f"High quality BOTH saved to: {out_path_both}")
        print(f"High quality ANY  saved to: {out_path_any}")
    else:
        print(f"High quality labels saved to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
