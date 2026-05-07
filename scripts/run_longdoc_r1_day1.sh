#!/usr/bin/env bash
set -euo pipefail

# Day-1 LongDoc-R1 pipeline:
# 1) generate a small batch of dynamic tool-use trajectories;
# 2) clean accepted trajectories into strict/soft SFT files;
# 3) optionally run the baseline experiment and LLM semantic judge.

TRAIN_LABEL_FILE="${TRAIN_LABEL_FILE:-outputs/label/high_quality_labels_any_model_agreement_train_1800_retry2.jsonl}"
VAL_LABEL_FILE="${VAL_LABEL_FILE:-outputs/label/high_quality_labels_any_model_agreement_validation_200_retry4.jsonl}"
SFT_DIR="${SFT_DIR:-outputs/train_COT}"
TRAIN_OUTPUT_BASE="${TRAIN_OUTPUT_BASE:-${SFT_DIR}/sft_train_day1.jsonl}"
VAL_OUTPUT_BASE="${VAL_OUTPUT_BASE:-${SFT_DIR}/sft_validation_day1.jsonl}"
TRAIN_CLEAN_PREFIX="${TRAIN_CLEAN_PREFIX:-${SFT_DIR}/train_sft_day1}"
VAL_CLEAN_PREFIX="${VAL_CLEAN_PREFIX:-${SFT_DIR}/validation_sft_day1}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-120}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-50}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_VAL="${SKIP_VAL:-0}"
CHUNK_SIZE="${CHUNK_SIZE:-300}"
MAX_ROUNDS="${MAX_ROUNDS:-8}"
SAVE_EVERY="${SAVE_EVERY:-20}"
OVERVIEW_MAX_SUMMARY_CHARS="${OVERVIEW_MAX_SUMMARY_CHARS:-800}"
INITIAL_CLUE_LEVEL="${INITIAL_CLUE_LEVEL:-1}"
JUDGE_LOCAL="${JUDGE_LOCAL:-1}"
JUDGE_LOCAL_DEVICE="${JUDGE_LOCAL_DEVICE:-cuda}"
JUDGE_LOCAL_MAX_NEW_TOKENS="${JUDGE_LOCAL_MAX_NEW_TOKENS:-192}"

mkdir -p "${SFT_DIR}"

required_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "ERROR: ${name} is required" >&2
    exit 2
  fi
}

required_env DECISION_MODEL_NAME
required_env READ_MODEL_NAME
required_env JUDGE_MODEL

DECISION_BASE_URL="${DECISION_BASE_URL:-${BASE_URL:-}}"
DECISION_API_KEY="${DECISION_API_KEY:-${API_KEY:-}}"
READ_BASE_URL="${READ_BASE_URL:-${BASE_URL:-}}"
READ_API_KEY="${READ_API_KEY:-${API_KEY:-}}"
QA_MODEL_NAME="${QA_MODEL_NAME:-${ANSWER_MODEL_NAME:-}}"
QA_BASE_URL="${QA_BASE_URL:-${ANSWER_BASE_URL:-${BASE_URL:-}}}"
QA_API_KEY="${QA_API_KEY:-${ANSWER_API_KEY:-${API_KEY:-}}}"

required_env DECISION_BASE_URL
required_env DECISION_API_KEY
required_env READ_BASE_URL
required_env READ_API_KEY
required_env QA_MODEL_NAME
required_env QA_BASE_URL
required_env QA_API_KEY

JUDGE_ARGS=(--judge_model "${JUDGE_MODEL}")
if [[ "${JUDGE_LOCAL}" == "1" || "${JUDGE_LOCAL}" == "true" || "${JUDGE_LOCAL}" == "yes" ]]; then
  JUDGE_ARGS+=(--judge_local --judge_local_device "${JUDGE_LOCAL_DEVICE}")
  JUDGE_ARGS+=(--judge_local_max_new_tokens "${JUDGE_LOCAL_MAX_NEW_TOKENS}")
else
  required_env JUDGE_BASE_URL
  required_env JUDGE_API_KEY
  JUDGE_ARGS+=(--judge_base_url "${JUDGE_BASE_URL}" --judge_api_key "${JUDGE_API_KEY}")
fi

if [[ "${SKIP_TRAIN}" != "1" && "${SKIP_TRAIN}" != "true" && "${SKIP_TRAIN}" != "yes" ]]; then
  echo "== Generate train trajectories =="
  python -m data.sft_data_generation_longdoc \
  --agreement_file "${TRAIN_LABEL_FILE}" \
  --output "${TRAIN_OUTPUT_BASE}" \
  --decision_model "${DECISION_MODEL_NAME}" \
  --decision_base_url "${DECISION_BASE_URL}" \
  --decision_api_key "${DECISION_API_KEY}" \
  --read_model "${READ_MODEL_NAME}" \
  --read_base_url "${READ_BASE_URL}" \
  --read_api_key "${READ_API_KEY}" \
  --qa_model "${QA_MODEL_NAME}" \
  --qa_base_url "${QA_BASE_URL}" \
  --qa_api_key "${QA_API_KEY}" \
  "${JUDGE_ARGS[@]}" \
  --max_samples "${TRAIN_MAX_SAMPLES}" \
  --start_index "${START_INDEX}" \
  --end_index "${END_INDEX}" \
  --max_rounds "${MAX_ROUNDS}" \
  --chunk_size "${CHUNK_SIZE}" \
  --overview_max_summary_chars "${OVERVIEW_MAX_SUMMARY_CHARS}" \
  --initial_clue_level "${INITIAL_CLUE_LEVEL}" \
  --save_every "${SAVE_EVERY}"

  echo "== Clean train trajectories =="
  python -m data.clean_sft_trajectories \
  --sft_file "${TRAIN_OUTPUT_BASE%.jsonl}_accepted.jsonl" \
  --label_file "${TRAIN_LABEL_FILE}" \
  --output_prefix "${TRAIN_CLEAN_PREFIX}" \
  --chunk_size "${CHUNK_SIZE}"
fi

if [[ "${SKIP_VAL}" != "1" && "${SKIP_VAL}" != "true" && "${SKIP_VAL}" != "yes" ]]; then
  echo "== Generate validation trajectories =="
  python -m data.sft_data_generation_longdoc \
  --agreement_file "${VAL_LABEL_FILE}" \
  --output "${VAL_OUTPUT_BASE}" \
  --decision_model "${DECISION_MODEL_NAME}" \
  --decision_base_url "${DECISION_BASE_URL}" \
  --decision_api_key "${DECISION_API_KEY}" \
  --read_model "${READ_MODEL_NAME}" \
  --read_base_url "${READ_BASE_URL}" \
  --read_api_key "${READ_API_KEY}" \
  --qa_model "${QA_MODEL_NAME}" \
  --qa_base_url "${QA_BASE_URL}" \
  --qa_api_key "${QA_API_KEY}" \
  "${JUDGE_ARGS[@]}" \
  --max_samples "${VAL_MAX_SAMPLES}" \
  --max_rounds "${MAX_ROUNDS}" \
  --chunk_size "${CHUNK_SIZE}" \
  --overview_max_summary_chars "${OVERVIEW_MAX_SUMMARY_CHARS}" \
  --initial_clue_level "${INITIAL_CLUE_LEVEL}" \
  --save_every "${SAVE_EVERY}"

  echo "== Clean validation trajectories =="
  python -m data.clean_sft_trajectories \
  --sft_file "${VAL_OUTPUT_BASE%.jsonl}_accepted.jsonl" \
  --label_file "${VAL_LABEL_FILE}" \
  --output_prefix "${VAL_CLEAN_PREFIX}" \
  --chunk_size "${CHUNK_SIZE}"
fi

echo "Done."
echo "Train soft clean: ${TRAIN_CLEAN_PREFIX}_soft_clean.jsonl"
echo "Train strict clean: ${TRAIN_CLEAN_PREFIX}_strict_clean.jsonl"
echo "Validation soft clean: ${VAL_CLEAN_PREFIX}_soft_clean.jsonl"
echo "Validation strict clean: ${VAL_CLEAN_PREFIX}_strict_clean.jsonl"
