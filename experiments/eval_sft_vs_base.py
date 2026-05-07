"""Evaluate prompt-only and SFT decision agents on held-out NarrativeQA samples."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.react_agent import HierarchicalReActAgent
from data.load_narrativeqa import load_narrativeqa_subset
from env.document_env import DocumentEnv
from evaluation.metrics import (
    bleu_score,
    exact_match_score,
    rouge_l_score,
    token_f1_score,
)
from llm.client import OpenAICompatibleClient
from tree.build_tree import build_hierarchical_tree


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """Yield JSON objects from a JSONL file."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl_row(path: Path, row: Dict[str, Any]) -> None:
    """Append one JSONL row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_seen_ids(paths: Sequence[str]) -> Set[str]:
    """Load sample ids that should be excluded from held-out eval."""
    seen: Set[str] = set()
    for raw_path in paths:
        if not raw_path:
            continue
        path = Path(raw_path)
        if not path.exists():
            continue
        for row in read_jsonl(path):
            sample_id = str(row.get("id") or row.get("sample_id") or row.get("source_id") or "")
            if sample_id:
                seen.add(sample_id)
    return seen


def resolve_env(name: str, fallback: Optional[str] = None) -> Optional[str]:
    """Return an environment variable with optional fallback."""
    value = os.getenv(name)
    if value:
        return value
    if fallback:
        return os.getenv(fallback)
    return None


def first_env(names: Sequence[str]) -> Optional[str]:
    """Return the first non-empty environment value among names."""
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def build_client(
    prefix: str,
    default_model: str | None = None,
    fallback_prefixes: Sequence[str] = (),
) -> OpenAICompatibleClient:
    """Build an OpenAI-compatible client from PREFIX_* environment variables."""
    prefix = prefix.upper()
    prefixes = [prefix, *[p.upper() for p in fallback_prefixes]]
    return OpenAICompatibleClient(
        base_url=first_env([*(f"{p}_BASE_URL" for p in prefixes), "BASE_URL"]),
        api_key=first_env([*(f"{p}_API_KEY" for p in prefixes), "API_KEY"]),
        model_name=first_env([*(f"{p}_MODEL_NAME" for p in prefixes), "MODEL_NAME"]) or default_model,
    )


class LocalHFChatClient:
    """Local Transformers chat client with the same generate interface."""

    def __init__(
        self,
        model_name_or_path: str,
        adapter_name_or_path: str = "",
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
        if adapter_name_or_path:
            try:
                from peft import PeftModel
            except Exception as exc:
                raise RuntimeError("Loading a LoRA adapter requires peft.") from exc
            self.model = PeftModel.from_pretrained(self.model, adapter_name_or_path)

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
                enable_thinking=True,
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


def build_decision_client(
    method: str,
    backend: str,
    default_api_prefix: str,
    default_model_env: str,
    local_model_path: str,
    local_adapter_path: str,
    local_device_map: str,
    local_dtype: str,
) -> Any:
    """Build either an API or local decision client."""
    if backend == "api":
        return build_client(
            default_api_prefix,
            default_model=resolve_env(default_model_env),
            fallback_prefixes=["DECISION"] if method == "base" else (),
        )
    if backend == "local":
        if not local_model_path:
            raise ValueError(f"--{method}_model_path is required when {method} backend is local.")
        return LocalHFChatClient(
            model_name_or_path=local_model_path,
            adapter_name_or_path=local_adapter_path,
            device_map=local_device_map,
            torch_dtype=local_dtype,
        )
    raise ValueError(f"Unsupported decision backend: {backend}")


def load_heldout_samples(
    split: str,
    max_samples: int,
    load_samples: int,
    exclude_ids: Set[str],
    cache_dir: str | None,
    save_dir: str | None,
) -> List[Dict[str, Any]]:
    """Load held-out NarrativeQA samples, excluding SFT sample ids."""
    data = load_narrativeqa_subset(
        train_size=load_samples if split == "train" else 1,
        validation_size=load_samples if split == "validation" else 1,
        cache_dir=cache_dir,
        save_dir=save_dir,
    )
    rows = []
    for sample in data.get(split, []):
        sample_id = str(sample.get("id", ""))
        if sample_id in exclude_ids:
            continue
        rows.append(sample)
        if len(rows) >= max_samples:
            break
    return rows


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    """Load a fixed sample manifest."""
    if not path.exists():
        return []
    return list(read_jsonl(path))


def save_manifest(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    """Save selected samples for reproducible evaluation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_or_create_samples(
    split: str,
    max_samples: int,
    load_samples: int,
    exclude_ids: Set[str],
    cache_dir: str | None,
    save_dir: str | None,
    sample_manifest: str,
    generate_manifest_if_missing: bool,
    seed: int,
) -> List[Dict[str, Any]]:
    """Load samples from a manifest or create one by random held-out QA sampling."""
    manifest_path = Path(sample_manifest) if sample_manifest else None
    if manifest_path is not None and manifest_path.exists():
        rows = load_manifest(manifest_path)
        return rows[:max_samples]

    if manifest_path is not None and not generate_manifest_if_missing:
        raise FileNotFoundError(
            f"Sample manifest does not exist: {manifest_path}. "
            "Pass --generate_manifest_if_missing to create it."
        )

    pool = load_heldout_samples(
        split=split,
        max_samples=max(load_samples, max_samples),
        load_samples=max(load_samples, max_samples),
        exclude_ids=exclude_ids,
        cache_dir=cache_dir,
        save_dir=save_dir,
    )
    rng = random.Random(seed)
    rng.shuffle(pool)
    rows = pool[:max_samples]
    if len(rows) < max_samples:
        print(f"WARNING: requested {max_samples} samples but only selected {len(rows)} after exclusion.")
    if manifest_path is not None:
        save_manifest(manifest_path, rows)
        print(f"Saved sample manifest: {manifest_path}")
    return rows


def safe_run_agent(
    agent: HierarchicalReActAgent,
    sample: Dict[str, Any],
    nodes: Dict[str, Any],
    root_id: str,
    read_client: OpenAICompatibleClient,
    qa_client: OpenAICompatibleClient,
    decision_max_tokens: int,
) -> Dict[str, Any]:
    """Run one agent call and normalize exceptions."""
    env = DocumentEnv(
        nodes=copy.deepcopy(nodes),
        root_id=root_id,
        read_llm_client=read_client,
        qa_llm_client=qa_client,
    )
    try:
        output = agent.answer(
            question=str(sample["question"]),
            env=env,
            max_tokens=decision_max_tokens,
        )
        return {"output": output, "error": ""}
    except Exception as exc:
        return {"output": {"final_answer": "", "trajectory": [], "used_tools": [], "step_count": 0}, "error": str(exc)}


def tool_stats(raw_output: Dict[str, Any]) -> Dict[str, Any]:
    """Compute tool behavior statistics from an agent trajectory."""
    trajectory = raw_output.get("trajectory", []) or []
    read_calls = 0
    read_success = 0
    qa_calls = 0
    qa_success = 0
    invalid_steps = 0
    qa_before_read_errors = 0
    forced_steps = 0
    has_final_turn = False
    final_before_qa = False
    seen_qa = False

    for step in trajectory:
        if step.get("forced"):
            forced_steps += 1
        if step.get("invalid"):
            invalid_steps += 1
            error = str(step.get("error", "")).lower()
            if "prior read" in error or "before qa" in error:
                qa_before_read_errors += 1
        if "final_answer" in step and "action" not in step:
            has_final_turn = True
            if not seen_qa:
                final_before_qa = True

        action = str(step.get("action", "") or "")
        obs = step.get("observation", {}) or {}
        if action == "read":
            read_calls += 1
            if bool(obs.get("success", False)):
                read_success += 1
        if action == "qa":
            qa_calls += 1
            seen_qa = True
            if bool(obs.get("success", False)):
                qa_success += 1

    final_answer = str(raw_output.get("final_answer", "") or "")
    return {
        "step_count": int(raw_output.get("step_count", len(trajectory)) or len(trajectory)),
        "read_calls": read_calls,
        "read_success": read_success,
        "qa_calls": qa_calls,
        "qa_success": qa_success,
        "invalid_steps": invalid_steps,
        "qa_before_read_errors": qa_before_read_errors,
        "forced_steps": forced_steps,
        "has_final_turn": has_final_turn,
        "final_before_qa": final_before_qa,
        "empty_answer": not bool(final_answer.strip()),
    }


def metric_row(prediction: str, answers: List[str]) -> Dict[str, float]:
    """Compute answer metrics for one prediction."""
    return {
        "em": exact_match_score(prediction, answers),
        "f1": token_f1_score(prediction, answers),
        "bleu": bleu_score(prediction, answers),
        "rouge_l": rouge_l_score(prediction, answers),
    }


def existing_keys(path: Path, rerun_errors: bool = False) -> Set[str]:
    """Return method/sample keys already present in a result file."""
    if not path.exists():
        return set()
    keys = set()
    for row in read_jsonl(path):
        if rerun_errors and row.get("error"):
            continue
        keys.add(f"{row.get('method')}::{row.get('sample_id')}")
    return keys


def summarize(results_path: Path, summary_path: Path) -> List[Dict[str, Any]]:
    """Aggregate JSONL results into a CSV summary."""
    raw_rows = list(read_jsonl(results_path)) if results_path.exists() else []
    by_key: Dict[str, Dict[str, Any]] = {}
    for row in raw_rows:
        key = f"{row.get('method')}::{row.get('sample_id')}"
        by_key[key] = row
    rows = list(by_key.values())
    methods = sorted({str(row.get("method", "")) for row in rows if row.get("method")})
    summary_rows: List[Dict[str, Any]] = []
    for method in methods:
        sub = [row for row in rows if row.get("method") == method]
        n = max(len(sub), 1)

        def avg(key: str) -> float:
            return sum(float(row.get(key, 0.0) or 0.0) for row in sub) / n

        summary_rows.append(
            {
                "method": method,
                "num_samples": len(sub),
                "em": avg("em"),
                "f1": avg("f1"),
                "bleu": avg("bleu"),
                "rouge_l": avg("rouge_l"),
                "error_rate": sum(bool(row.get("error")) for row in sub) / n,
                "empty_answer_rate": avg("empty_answer"),
                "invalid_step_rate": sum(int(row.get("invalid_steps", 0) or 0) > 0 for row in sub) / n,
                "qa_before_read_error_rate": sum(
                    int(row.get("qa_before_read_errors", 0) or 0) > 0 for row in sub
                )
                / n,
                "premature_final_rate": avg("final_before_qa"),
                "qa_success_rate": sum(int(row.get("qa_success", 0) or 0) > 0 for row in sub) / n,
                "avg_steps": avg("step_count"),
                "avg_read_calls": avg("read_calls"),
                "avg_qa_calls": avg("qa_calls"),
                "avg_forced_steps": avg("forced_steps"),
            }
        )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()) if summary_rows else ["method"])
        writer.writeheader()
        writer.writerows(summary_rows)
    return summary_rows


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "validation"], default="validation")
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--load_samples", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_manifest", type=str, default="")
    parser.add_argument("--generate_manifest_if_missing", action="store_true")
    parser.add_argument("--rerun_errors", action="store_true")
    parser.add_argument("--max_steps", type=int, default=8)
    parser.add_argument("--decision_max_tokens", type=int, default=256)
    parser.add_argument("--methods", type=str, default="base,sft", help="Comma-separated: base,sft")
    parser.add_argument("--output_dir", type=str, default="outputs/eval_sft_vs_base")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--cache_dir", type=str, default="")
    parser.add_argument("--save_loaded_data", action="store_true")
    parser.add_argument("--base_decision_backend", choices=["api", "local"], default="api")
    parser.add_argument("--sft_decision_backend", choices=["api", "local"], default="api")
    parser.add_argument("--base_model_path", type=str, default="")
    parser.add_argument("--sft_model_path", type=str, default="")
    parser.add_argument("--sft_adapter_path", type=str, default="")
    parser.add_argument("--base_device_map", type=str, default="auto")
    parser.add_argument("--sft_device_map", type=str, default="auto")
    parser.add_argument("--local_torch_dtype", type=str, default="bfloat16")
    parser.add_argument(
        "--exclude_sft_file",
        action="append",
        default=[],
        help="JSONL file whose ids are excluded from held-out evaluation. Can be repeated.",
    )
    return parser.parse_args()


def main() -> int:
    """Run held-out SFT-vs-base evaluation."""
    args = parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = set(methods) - {"base", "sft"}
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / run_name
    result_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.csv"
    loaded_data_dir = output_dir / "data" if args.save_loaded_data else None
    output_dir.mkdir(parents=True, exist_ok=True)

    default_excludes = [
        "outputs/train_COT/train_sft_gpt51_clue2_plus_val184_terminalqa_notrunc_soft_clean.jsonl"
    ]
    exclude_ids = load_seen_ids([*default_excludes, *args.exclude_sft_file])
    samples = load_or_create_samples(
        split=args.split,
        max_samples=args.max_samples,
        load_samples=args.load_samples,
        exclude_ids=exclude_ids,
        cache_dir=args.cache_dir or None,
        save_dir=str(loaded_data_dir) if loaded_data_dir is not None else None,
        sample_manifest=args.sample_manifest,
        generate_manifest_if_missing=args.generate_manifest_if_missing,
        seed=args.seed,
    )
    if len(samples) < args.max_samples:
        print(f"WARNING: requested {args.max_samples} samples but only loaded {len(samples)} after exclusion.")

    print(f"Run: {run_name}")
    print(f"Output: {output_dir}")
    print(f"Split: {args.split}, samples: {len(samples)}, excluded ids: {len(exclude_ids)}")
    print(f"Methods: {methods}")

    read_client = build_client("READ")
    qa_client = build_client("QA", default_model=resolve_env("ANSWER_MODEL_NAME"), fallback_prefixes=["ANSWER"])
    decision_clients: Dict[str, OpenAICompatibleClient] = {}
    if "base" in methods:
        decision_clients["base"] = build_decision_client(
            method="base",
            backend=args.base_decision_backend,
            default_api_prefix="BASE_DECISION",
            default_model_env="DECISION_MODEL_NAME",
            local_model_path=args.base_model_path,
            local_adapter_path="",
            local_device_map=args.base_device_map,
            local_dtype=args.local_torch_dtype,
        )
    if "sft" in methods:
        decision_clients["sft"] = build_decision_client(
            method="sft",
            backend=args.sft_decision_backend,
            default_api_prefix="SFT_DECISION",
            default_model_env="SFT_MODEL_NAME",
            local_model_path=args.sft_model_path,
            local_adapter_path=args.sft_adapter_path,
            local_device_map=args.sft_device_map,
            local_dtype=args.local_torch_dtype,
        )

    agents = {
        method: HierarchicalReActAgent(llm_client=client, max_steps=args.max_steps)
        for method, client in decision_clients.items()
    }
    done = existing_keys(result_path, rerun_errors=args.rerun_errors)

    for sample in tqdm(samples, desc="held-out samples"):
        sample_id = str(sample["id"])
        root_id, nodes = build_hierarchical_tree(document_text=str(sample["document"]), doc_id=sample_id)
        answers = list(sample.get("answers", []) or [sample.get("answer", "")])

        for method in methods:
            key = f"{method}::{sample_id}"
            if key in done:
                continue
            result = safe_run_agent(
                agent=agents[method],
                sample=sample,
                nodes=nodes,
                root_id=root_id,
                read_client=read_client,
                qa_client=qa_client,
                decision_max_tokens=args.decision_max_tokens,
            )
            raw_output = result["output"]
            prediction = str(raw_output.get("final_answer", "") or "")
            stats = tool_stats(raw_output)
            metrics = metric_row(prediction, answers)
            row = {
                "method": method,
                "sample_id": sample_id,
                "split": args.split,
                "question": sample.get("question", ""),
                "answers": answers,
                "prediction": prediction,
                "error": result["error"],
                **metrics,
                **stats,
                "raw_output": raw_output,
            }
            write_jsonl_row(result_path, row)
            done.add(key)

        summarize(result_path, summary_path)

    summary_rows = summarize(result_path, summary_path)
    print("\n=== Summary ===")
    for row in summary_rows:
        print(json.dumps(row, ensure_ascii=False))
    print(f"\nResults: {result_path}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
