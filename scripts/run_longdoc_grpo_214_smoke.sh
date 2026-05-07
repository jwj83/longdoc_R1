#!/usr/bin/env bash
set -xeuo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/jiangwenjun25/Longdoc_R1}
MODEL_PATH=${MODEL_PATH:-/home/jiangwenjun25/cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}
LORA_ADAPTER_PATH=${LORA_ADAPTER_PATH:-${PROJECT_ROOT}/outputs/checkpoints/longdoc_r1_qwen3_8b_lora_sft_184}
DATA_DIR=${DATA_DIR:-${PROJECT_ROOT}/outputs/rl_data/longdoc_grpo_214}

NGPUS_PER_NODE=${NGPUS_PER_NODE:-3}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
ROLLOUT_N=${ROLLOUT_N:-4}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
SAVE_FREQ=${SAVE_FREQ:-5}
TEST_FREQ=${TEST_FREQ:-5}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-12288}
ACTOR_LR=${ACTOR_LR:-1e-6}
AGENT_NUM_WORKERS=${AGENT_NUM_WORKERS:-1}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.25}
ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-8192}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-8}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-4096}

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

cd "${PROJECT_ROOT}"

conda run -n doc ray stop --force || true

conda run -n doc python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="${DATA_DIR}/train.parquet" \
  data.val_files="${DATA_DIR}/test.parquet" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.lora_adapter_path="${LORA_ADAPTER_PATH}" \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.max_model_len="${ROLLOUT_MAX_MODEL_LEN}" \
  actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}" \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.temperature=0.7 \
  actor_rollout_ref.rollout.top_p=0.9 \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.layered_summon=True \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
  actor_rollout_ref.rollout.agent.num_workers="${AGENT_NUM_WORKERS}" \
  actor_rollout_ref.rollout.agent.default_agent_loop=longdoc_xml_agent \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="${PROJECT_ROOT}/configs/verl_longdoc_agent_loop.yaml" \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=8 \
  actor_rollout_ref.rollout.multi_turn.max_user_turns=8 \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  custom_reward_function.path="${PROJECT_ROOT}/rl/longdoc_reward.py" \
  custom_reward_function.name=compute_score \
  trainer.balance_batch=True \
  trainer.logger='["console"]' \
  trainer.project_name=longdoc_r1 \
  trainer.experiment_name=grpo_sft184_on_214_smoke \
  trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
  trainer.nnodes=1 \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  +ray_kwargs.ray_init.num_gpus="${NGPUS_PER_NODE}" \
  "$@"
