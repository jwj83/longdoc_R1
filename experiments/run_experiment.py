"""Run end-to-end experiments for flat RAG, long-context, and hierarchical agent."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm

# Ensure long_doc_agent root is in path when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.react_agent import HierarchicalReActAgent
from baselines.flat_rag import FlatRAGBaseline
from baselines.long_context import DirectLongContextBaseline
from data.load_narrativeqa import load_narrativeqa_subset
from env.document_env import DocumentEnv
from evaluation.metrics import evaluate_predictions
from llm.client import OpenAICompatibleClient
from tree.build_tree import build_hierarchical_tree, save_tree_json


def _run_single_method_safe(method_name: str, fn, sample: Dict[str, Any]) -> Dict[str, Any]:
    """Execute one method safely and return a normalized result row."""
    try:
        output = fn()
        if method_name == "hierarchical_agent":
            prediction = output.get("final_answer", "")
        else:
            prediction = output.get("prediction", "")
        return {
            "method": method_name,
            "sample_id": sample["id"],
            "question": sample["question"],
            "answers": sample["answers"],
            "prediction": prediction,
            "raw_output": output,
            "error": "",
        }
    except Exception as e:
        return {
            "method": method_name,
            "sample_id": sample["id"],
            "question": sample["question"],
            "answers": sample["answers"],
            "prediction": "",
            "raw_output": {},
            "error": str(e),
        }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run long-document QA experiment prototype.")
    parser.add_argument("--train_samples", type=int, default=5)
    parser.add_argument("--validation_samples", type=int, default=5)
    parser.add_argument("--eval_split", type=str, default="validation", choices=["train", "validation"])
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=12)
    parser.add_argument("--embedding_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--decision_model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--read_model", type=str, default="Qwen/Qwen2.5-72B-Instruct")
    parser.add_argument("--qa_model", type=str, default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "outputs"))
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional run folder name under output_dir. Defaults to current timestamp.",
    )
    return parser.parse_args()


def _resolve_model_name(cli_value: str | None, role_env: str) -> str | None:
    """Resolve model name by priority: CLI > role env > MODEL_NAME."""
    if cli_value:
        return cli_value
    return os.getenv(role_env) or os.getenv("MODEL_NAME")


def _build_role_client(role: str, role_model_name: str | None) -> OpenAICompatibleClient:
    """Create one client with role-specific optional endpoint/key overrides."""
    role = role.upper()
    role_base_url = os.getenv(f"{role}_BASE_URL")
    role_api_key = os.getenv(f"{role}_API_KEY")
    return OpenAICompatibleClient(
        base_url=role_base_url or os.getenv("BASE_URL"),
        api_key=role_api_key or os.getenv("API_KEY"),
        model_name=role_model_name,
    )


def main() -> None:
    """Main entry for running experiment pipeline."""
    args = parse_args()

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / run_name
    tree_dir = output_dir / "trees"
    result_dir = output_dir / "results"
    data_dir = output_dir / "data"
    tree_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run name: {run_name}")
    print(f"Output root: {output_dir}")

    print("Loading NarrativeQA subset...")
    data = load_narrativeqa_subset(
        train_size=args.train_samples,
        validation_size=args.validation_samples,
        save_dir=str(data_dir),
    )
    samples = data.get(args.eval_split, [])
    if not samples:
        raise RuntimeError(f"No samples loaded for split: {args.eval_split}")

    decision_model = _resolve_model_name(args.decision_model, "DECISION_MODEL_NAME")
    read_model = _resolve_model_name(args.read_model, "READ_MODEL_NAME")
    qa_model = _resolve_model_name(args.qa_model, "QA_MODEL_NAME") or _resolve_model_name(None, "ANSWER_MODEL_NAME")

    decision_client = _build_role_client("DECISION", decision_model)
    read_client = _build_role_client("READ", read_model)
    qa_client = _build_role_client("QA", qa_model)

    # Keep baselines on decision model by default for comparable cost profile.
    flat_rag = FlatRAGBaseline(llm_client=decision_client, embedding_model_name=args.embedding_model)
    long_context = DirectLongContextBaseline(llm_client=decision_client)
    agent = HierarchicalReActAgent(llm_client=decision_client, max_steps=args.max_steps)

    rows: List[Dict[str, Any]] = []

    for sample in tqdm(samples, desc=f"Running {args.eval_split} samples"):
        sample_id = str(sample["id"])
        doc = sample["document"]
        question = sample["question"]

        root_id, nodes = build_hierarchical_tree(document_text=doc, doc_id=sample_id)
        tree_path = tree_dir / f"{args.eval_split}_{sample_id}.json"
        save_tree_json(root_id=root_id, nodes=nodes, path=str(tree_path))

        env = DocumentEnv(
            nodes=nodes,
            root_id=root_id,
            read_llm_client=read_client,
            qa_llm_client=qa_client,
        )

        flat_row = _run_single_method_safe(
            "flat_rag",
            lambda: flat_rag.answer(question=question, document=doc, top_k=args.top_k),
            sample,
        )
        rows.append(flat_row)

        lc_row = _run_single_method_safe(
            "long_context",
            lambda: long_context.answer(question=question, document=doc, max_context_words=4000),
            sample,
        )
        rows.append(lc_row)

        agent_row = _run_single_method_safe(
            "hierarchical_agent",
            lambda: agent.answer(question=question, env=env),
            sample,
        )
        rows.append(agent_row)

    results_jsonl = result_dir / "results.jsonl"
    with results_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            row_to_save = dict(row)
            row_to_save["raw_output"] = json.dumps(row_to_save["raw_output"], ensure_ascii=False)
            f.write(json.dumps(row_to_save, ensure_ascii=False) + "\n")

    df = pd.DataFrame(rows)
    df["raw_output"] = df["raw_output"].apply(lambda x: json.dumps(x, ensure_ascii=False))
    results_csv = result_dir / "results.csv"
    df.to_csv(results_csv, index=False)

    summary_rows = []
    for method in sorted(df["method"].unique().tolist()):
        sub = df[df["method"] == method]
        preds = sub["prediction"].tolist()
        refs = sub["answers"].tolist()
        metric = evaluate_predictions(preds, refs)
        summary_rows.append(
            {
                "method": method,
                "num_samples": len(sub),
                "EM": metric["em"],
                "F1": metric["f1"],
                "BLEU": metric["bleu"],
                "ROUGE_L": metric["rouge_l"],
                "num_errors": int((sub["error"] != "").sum()),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(by=["F1", "ROUGE_L", "EM"], ascending=False)
    summary_csv = result_dir / "summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print("\n=== Experiment Summary ===")
    print(summary_df.to_string(index=False))
    print("\nSaved files:")
    print(f"- {results_jsonl}")
    print(f"- {results_csv}")
    print(f"- {summary_csv}")


if __name__ == "__main__":
    main()
