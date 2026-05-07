#!/usr/bin/env python3
"""Score prediction files with an LLM semantic judge."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from evaluation.metrics import evaluate_predictions
from llm.client import OpenAICompatibleClient


class LocalHFChatClient:
    """Local Transformers chat client with the same generate interface."""

    def __init__(
        self,
        model_name_or_path: str,
        device_map: str = "auto",
        torch_dtype: str = "bfloat16",
        trust_remote_code: bool = True,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            raise RuntimeError("LocalHFChatClient requires torch and transformers.") from exc

        dtype_map = {
            "auto": "auto",
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        dtype = dtype_map.get(torch_dtype.lower(), torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout: int = 120,
    ) -> str:
        """Generate one chat completion locally."""
        del timeout
        import torch

        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        inputs = self.tokenizer([prompt], return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        do_sample = temperature > 0
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL rows."""
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_json_object(raw: str) -> Optional[Dict[str, Any]]:
    """Parse a JSON object from noisy model output."""
    text = str(raw or "").strip()
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        text = m.group(1).strip()
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
    try:
        obj = json.loads(text)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def judge_prompt(question: str, gold: str, prediction: str) -> str:
    """Build a strict semantic consistency prompt."""
    return (
        "You are a QA semantic consistency judge. Determine whether the prediction is semantically "
        "consistent with the gold answer for the question.\n\n"
        "Rules:\n"
        "1. Mark consistent=true if the prediction has the same meaning as the gold answer or includes the key gold information.\n"
        "2. Mark consistent=false if the prediction contradicts, omits the key fact, or answers a different question.\n"
        "3. Extra context is acceptable only if it does not introduce a wrong fact.\n"
        "4. Output strict JSON only with keys: consistent, reason, missing_key_info, extra_wrong_info.\n\n"
        f"Question: {question}\n"
        f"Gold answer: {gold}\n"
        f"Prediction: {prediction}\n"
    )


def references_from_row(row: Dict[str, Any]) -> List[str]:
    """Extract references from a result row."""
    answers = row.get("answers", [])
    if isinstance(answers, list):
        return [str(x) for x in answers]
    if isinstance(answers, str) and answers:
        return [answers]
    gold = str(row.get("gold", "") or "")
    return [gold] if gold else []


def judge_one(client: Any, row: Dict[str, Any], timeout: int, max_tokens: int) -> Dict[str, Any]:
    """Judge one prediction row."""
    question = str(row.get("question", "") or "")
    prediction = str(row.get("prediction", "") or "")
    refs = references_from_row(row)
    gold = refs[0] if refs else ""
    raw = client.generate(
        messages=[{"role": "user", "content": judge_prompt(question, gold, prediction)}],
        temperature=0.0,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    parsed = parse_json_object(raw) or {}
    return {
        "consistent": bool(parsed.get("consistent", False)),
        "reason": str(parsed.get("reason", "")),
        "missing_key_info": str(parsed.get("missing_key_info", "")),
        "extra_wrong_info": str(parsed.get("extra_wrong_info", "")),
        "raw_judge": raw,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Add LLM semantic accuracy to experiment results")
    parser.add_argument("--results_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--summary_json", type=str, default="")
    parser.add_argument("--backend", choices=["api", "local"], default="api")
    parser.add_argument("--model", type=str, default=os.getenv("JUDGE_MODEL", os.getenv("MODEL_NAME", "")))
    parser.add_argument("--base_url", type=str, default=os.getenv("JUDGE_BASE_URL", os.getenv("BASE_URL", "")))
    parser.add_argument("--api_key", type=str, default=os.getenv("JUDGE_API_KEY", os.getenv("API_KEY", "")))
    parser.add_argument("--local_model_path", type=str, default=os.getenv("JUDGE_LOCAL_MODEL_PATH", ""))
    parser.add_argument("--local_device_map", type=str, default=os.getenv("JUDGE_LOCAL_DEVICE_MAP", "auto"))
    parser.add_argument("--local_torch_dtype", type=str, default=os.getenv("JUDGE_LOCAL_TORCH_DTYPE", "bfloat16"))
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--timeout", type=int, default=120)
    return parser.parse_args()


def main() -> int:
    """Run semantic judge over a result JSONL."""
    args = parse_args()
    if args.backend == "local":
        model_path = args.local_model_path or args.model
        if not model_path:
            raise ValueError("--local_model_path or --model is required for local judge backend.")
        client = LocalHFChatClient(
            model_name_or_path=model_path,
            device_map=args.local_device_map,
            torch_dtype=args.local_torch_dtype,
        )
    else:
        client = OpenAICompatibleClient(
            base_url=args.base_url,
            api_key=args.api_key,
            model_name=args.model,
        )
    rows = load_jsonl(Path(args.results_jsonl))
    judged_rows: List[Dict[str, Any]] = []
    for i, row in enumerate(rows, start=1):
        judge = judge_one(client, row, timeout=args.timeout, max_tokens=args.max_tokens)
        judged_rows.append({**row, "semantic_judge": judge, "semantic_acc": float(judge["consistent"])})
        if i % 10 == 0 or i == len(rows):
            print(f"[{i}/{len(rows)}] judged")

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in judged_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_method: Dict[str, List[Dict[str, Any]]] = {}
    for row in judged_rows:
        by_method.setdefault(str(row.get("method", "")), []).append(row)

    summary: Dict[str, Any] = {}
    for method, method_rows in by_method.items():
        preds = [str(row.get("prediction", "") or "") for row in method_rows]
        refs = [references_from_row(row) for row in method_rows]
        metrics = evaluate_predictions(preds, refs)
        semantic_acc = sum(float(row.get("semantic_acc", 0.0) or 0.0) for row in method_rows) / max(len(method_rows), 1)
        summary[method] = {
            "num_samples": len(method_rows),
            "EM": metrics["em"],
            "F1": metrics["f1"],
            "BLEU": metrics["bleu"],
            "ROUGE_L": metrics["rouge_l"],
            "semantic_acc": semantic_acc,
        }

    summary_path = Path(args.summary_json) if args.summary_json else out_path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
