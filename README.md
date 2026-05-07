# LongDoc-R1: Hierarchical Tool-Use Reasoning for Long-Document QA

LongDoc-R1 is a research prototype for training a small decision model to reason over
long documents with hierarchical tools. It adapts the broad idea of LongVideo-R1-style
global-to-local navigation to text: build a document tree, inspect useful nodes, query a
leaf evidence segment, and answer only after enough evidence has been collected.

The current target task is NarrativeQA long-document question answering.

## Motivation

Prompt-only ReAct agents often fail at long-document tool use: they call tools with
invalid arguments, answer before collecting evidence, or query the wrong segment.
LongDoc-R1 treats tool-use behavior as something to train directly. The decision model
does not memorize document facts; it learns when and where to use evidence tools.

## Method Overview

Each document is converted into a four-level tree:

1. Root: the full document.
2. Level 1: high-level segments.
3. Level 2: medium-level segments.
4. Level 3: low-level leaf segments.

The agent receives high-level summaries at the beginning. It can use two tools:

- `read((h))`, `read((h,m))`, or `read((h,m,l))`: obtain a summary and child previews for
  a tree node.
- `qa((h,m,l), "query")`: answer a focused query using only one low-level leaf segment.

The decision model emits exactly one step at a time:

```text
<think>...</think>
<tool>read((h,m))</tool>
```

```text
<think>...</think>
<tool>qa((h,m,l), "question")</tool>
```

```text
<think>...</think>
<answer>...</answer>
```

## High-Quality Label Construction

The current evidence labels are weak-supervised, not manually authored gold rationales.
They are built as follows:

1. Split each NarrativeQA document into 300-word chunks.
2. Retrieve candidate chunks using BM25 and dense retrieval.
3. Ask two models to select evidence chunks, answer the question, and quote support.
4. Keep samples where both models selected the same non-empty chunk set.
5. Use an LLM judge to check whether model answers are semantically consistent with the
   gold answer.

Two filtered label sets are used:

- `any`: at least one model answer is judged consistent with the gold answer.
- `both`: both model answers are judged consistent; this is the stricter subset.

Current available labels:

| Split | any | both |
| --- | ---: | ---: |
| train | 585 | 512 |
| validation | 54 | 43 |

## Tool-Use SFT Data

Generate raw dynamic trajectories:

```bash
python -m data.sft_data_generation_longdoc \
  --agreement_file outputs/label/high_quality_labels_any_model_agreement_train_1800_retry2.jsonl \
  --output outputs/label/sft/sft_train_dynamic.jsonl \
  --decision_model "$DECISION_MODEL_NAME" \
  --decision_base_url "$DECISION_BASE_URL" \
  --decision_api_key "$DECISION_API_KEY" \
  --read_model "$READ_MODEL_NAME" \
  --read_base_url "$READ_BASE_URL" \
  --read_api_key "$READ_API_KEY" \
  --qa_model "$QA_MODEL_NAME" \
  --qa_base_url "$QA_BASE_URL" \
  --qa_api_key "$QA_API_KEY" \
  --judge_model "$JUDGE_MODEL" \
  --judge_base_url "$JUDGE_BASE_URL" \
  --judge_api_key "$JUDGE_API_KEY" \
  --max_samples 120 \
  --max_rounds 8 \
  --chunk_size 300
```

Accepted raw trajectories are not automatically training-ready. Some accepted samples can
contain invalid intermediate actions that were later corrected. Training uses cleaned
messages only.

Clean trajectories:

```bash
python -m data.clean_sft_trajectories \
  --sft_file outputs/label/sft/sft_train_dynamic_accepted.jsonl \
  --label_file outputs/label/high_quality_labels_any_model_agreement_train_1800_retry2.jsonl \
  --output_prefix outputs/label/sft/train_sft \
  --chunk_size 300
```

This writes:

- `outputs/label/sft/train_sft_strict_clean.jsonl`
- `outputs/label/sft/train_sft_soft_clean.jsonl`
- `outputs/label/sft/train_sft_quality_report.csv`
- `outputs/label/sft/train_sft_quality_summary.json`

## Clean Data Criteria

`strict-clean` keeps trajectories that satisfy all core constraints:

- final answer is semantically correct according to generator metadata;
- at least one successful `qa` call exists;
- no invalid action appears in the raw trajectory;
- no premature final answer appears before `qa`;
- `qa` is called on a leaf after reading that same leaf;
- the last successful `qa` answer is not an unknown/insufficient-evidence answer;
- the `qa` leaf overlaps the evidence chunk selected in the high-quality label.

`soft-clean` deletes invalid turns and keeps a reconstructed successful path when the final
answer is correct, there is a valid read/qa chain, the `qa` answer is usable, and the final
`qa` node hits the labeled evidence.

## Experiments

Primary methods:

- `Flat RAG`
- `Direct Long Context`
- `Prompt-only Qwen3-8B Agent`
- `SFT Qwen3-8B Agent`
- `Strong Tool Agent`
- optional `Oracle Evidence QA`

Answer-quality metrics:

- EM
- token-level F1
- BLEU
- ROUGE-L
- LLM-judge semantic accuracy

Tool-use metrics:

- invalid tool rate
- qa-before-read error rate
- premature final rate
- evidence hit rate
- average read calls
- average qa calls
- average total steps

Useful recovery metric:

```text
(SFT_8B - Prompt_8B) / (Strong_Agent - Prompt_8B)
```

## Run Baseline Experiment

```bash
python experiments/run_experiment.py \
  --train_samples 50 \
  --validation_samples 50 \
  --eval_split validation \
  --top_k 5 \
  --max_steps 8 \
  --decision_model qwen-8b-instruct \
  --read_model qwen-72b-instruct \
  --qa_model qwen-8b-instruct
```

## One-Day Execution Plan

1. Fix project packaging and clean trajectory validation.
2. Generate the first 120 train trajectories from the 585 `any` labels.
3. Clean accepted trajectories into strict and soft SFT files.
4. LoRA-SFT Qwen3-8B with the clean soft set first; use strict as an ablation.
5. Evaluate 30-50 validation samples across Flat RAG, direct long context, prompt-only
   agent, SFT agent, and strong tool agent.
6. Produce a main result table, a data quality table, success/failure cases, and a short
   next-step RL reward design.

## Environment

```bash
export BASE_URL="https://your-openai-compatible-endpoint/v1"
export API_KEY="your_api_key"
export MODEL_NAME="qwen-8b-instruct"

export DECISION_MODEL_NAME="qwen-8b-instruct"
export READ_MODEL_NAME="qwen-72b-instruct"
export QA_MODEL_NAME="qwen-8b-instruct"
export JUDGE_MODEL="qwen-72b-instruct"
```

Role-specific endpoints are supported by the scripts:

```bash
export DECISION_BASE_URL="$BASE_URL"
export DECISION_API_KEY="$API_KEY"
export READ_BASE_URL="$BASE_URL"
export READ_API_KEY="$API_KEY"
export QA_BASE_URL="$BASE_URL"
export QA_API_KEY="$API_KEY"
export JUDGE_BASE_URL="$BASE_URL"
export JUDGE_API_KEY="$API_KEY"
```

## Repository Layout

```text
agent/          ReAct decision loop, prompt, parser
baselines/      Flat RAG and direct long-context baselines
data/           NarrativeQA loading, label construction, SFT generation, SFT cleaning
env/            Document environment and tool execution
evaluation/     EM/F1/BLEU/ROUGE-L and judge helpers
experiments/    End-to-end experiment runner
llm/            OpenAI-compatible client
tools/          read and qa tools
tree/           Four-level document tree
outputs/        Generated labels, trajectories, trees, and results
```
