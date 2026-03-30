<<<<<<< HEAD
# Long Document Hierarchical Tool-Use Agent (MVP)

This repository is a **research-oriented prototype** for comparing:

- Flat RAG baseline
- Direct long-context baseline
- Hierarchical tool-use agent (ReAct style)

on long-document QA, using a small NarrativeQA subset for quick validation.

## 1. Project Overview

The project provides a clean and modular framework for experiments, not product deployment.

Core ideas:

- Build a two-level hierarchical tree for each long document.
- Expose only two hierarchical tools:
  - `read(node_id)`
  - `answer(node_id, question)`
- Run a minimal ReAct agent with strict action schema:
  - multiple `read` steps
  - one final `answer` step
- Compare against two baselines and evaluate with EM/F1.

## 2. Directory Structure

```text
long_doc_agent/
├── data/
│   └── load_narrativeqa.py
├── tree/
│   ├── node.py
│   └── build_tree.py
├── tools/
│   ├── answer.py
│   ├── base.py
│   ├── export_predictions_json.py
│   └── read.py
├── env/
│   └── document_env.py
├── llm/
│   └── client.py
├── baselines/
│   ├── flat_rag.py
│   └── long_context.py
├── agent/
│   ├── prompt.py
│   ├── parser.py
│   └── react_agent.py
├── evaluation/
│   └── metrics.py
├── experiments/
│   └── run_experiment.py
├── requirements.txt
└── README.md
```

## 3. Installation

```bash
cd long_doc_agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 4. Environment Variables

Set OpenAI-compatible API settings:

```bash
export BASE_URL="https://your-openai-compatible-endpoint/v1"
export API_KEY="your_api_key"
export MODEL_NAME="gpt-4o-mini"

# Optional: role-specific model split (fallback to MODEL_NAME if omitted)
export DECISION_MODEL_NAME="qwen-8b-instruct"
export READ_MODEL_NAME="qwen-72b-instruct"
export ANSWER_MODEL_NAME="qwen-8b-instruct"

# Optional: role-specific endpoint/key (fallback to BASE_URL/API_KEY)
# export DECISION_BASE_URL="..."
# export DECISION_API_KEY="..."
# export READ_BASE_URL="..."
# export READ_API_KEY="..."
# export ANSWER_BASE_URL="..."
# export ANSWER_API_KEY="..."
```

Notes:

- `BASE_URL` should be the API root containing `/chat/completions`.
- `MODEL_NAME` is used as fallback when role-specific model names are not set.

## 5. Run Experiment

From workspace root (the parent of `long_doc_agent`):

```bash
python long_doc_agent/experiments/run_experiment.py \
  --train_samples 50 \
  --validation_samples 50 \
  --eval_split validation \
  --top_k 5 \
  --max_steps 8 \
  --decision_model qwen-8b-instruct \
  --read_model qwen-72b-instruct \
  --answer_model qwen-8b-instruct
```

Main outputs are saved to:

- `long_doc_agent/outputs/trees/` (serialized trees)
- `long_doc_agent/outputs/results/results.jsonl`
- `long_doc_agent/outputs/results/results.csv`
- `long_doc_agent/outputs/results/summary.csv`

## 6. What Is Implemented

- Robust NarrativeQA subset loader with schema compatibility fallback.
- Hierarchical document tree builder and JSON I/O.
- Tool system with strict two-tool interface (`read`/`answer`).
- Document environment with trajectory recording and separate read/answer model hooks.
- Document environment with separate read-model summary generation and answer-model response generation.
- Lightweight OpenAI-compatible LLM client (`requests`-based).
- Baselines:
  - Flat RAG
  - Direct long context
- ReAct-style hierarchical tool-use agent with parser.
- Evaluation metrics: normalize, EM, token-level F1, batch evaluation.
- One-click experiment pipeline with result export and summary table.

## 7. Suggested Extensions

- Add more datasets (Qasper, GovReport QA, LongBench).
- Add deeper tree levels and learned splitting policy.
- Replace extractive summary with LLM/encoder summarizers.
- Add tool budgets / cost-aware planning.
- Add trajectory quality metrics (tool efficiency, depth usage).
- Add async batching and caching for embeddings.
=======
# longdoc_R1
>>>>>>> 7f5ed4c2a156e2a911276fb3359b75264095ca6d
