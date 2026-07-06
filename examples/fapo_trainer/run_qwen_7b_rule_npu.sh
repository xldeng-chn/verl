#!/usr/bin/env bash
# Degraded fallback | Qwen2.5-Math-7B | Ascend NPU | rule-based reward (NO GenRM)
#
# This is a stripped variant of run_qwen_7b_rm_colocate_npu.sh that DROPS the
# reward model (FAPO-GenRM-4B) and the reward_loop HTTP load balancer entirely,
# and scores rollouts with the rule-based ground_truth already baked into the
# dataset (reward_model.ground_truth, via the dapo reward manager's
# default_compute_score).
#
# Purpose: prove the base GRPO/PPO loop runs end-to-end on NPU *under this
# container's pids-limit (10000)*, which the full colocate GenRM config exceeds.
# It is NOT FAPO (no learned reward) — just a sanity check of the NPU pipeline.

set -xeuo pipefail

PYTHON=${PYTHON:-python}

########################### NPU environment ###########################
ulimit -s unlimited 2>/dev/null || true
_pids=$(cat /sys/fs/cgroup/pids/pids.max 2>/dev/null || echo max)
if [ "${_pids}" != "max" ] && [ "${_pids}" -lt 50000 ] 2>/dev/null; then
    echo "WARNING: container pids-limit is ${_pids} (/sys/fs/cgroup/pids/pids.max)." >&2
fi
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-1500}
export HCCL_HOST_SOCKET_PORT_RANGE=${HCCL_HOST_SOCKET_PORT_RANGE:-60000-60500}
export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-61000-61500}
export HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-AIV}
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
# Pin specific NPUs (torch_npu equivalent of CUDA_VISIBLE_DEVICES). This container
# exposes physical NPUs 3,4,5,6, so set ASCEND_RT_VISIBLE_DEVICES=3,4,5,6.
if [ -n "${ASCEND_RT_VISIBLE_DEVICES:-}" ]; then export ASCEND_RT_VISIBLE_DEVICES; fi
# Cap thread-hungry libraries to stay under tight container pids-limits. The HF
# tokenizer's Rust rayon pool (encode_batch) otherwise bursts to num_cpus threads
# per AgentLoopWorker and trips pthread_create EAGAIN around ~10k threads.
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-1}

DEVICE=${DEVICE:-$($PYTHON -c 'import torch_npu' 2>/dev/null && echo npu || echo gpu)}

MODEL_PATH=${MODEL_PATH:-/workspace/user_data/models/Qwen/Qwen2.5-Math-7B}
TRAIN_FILE=${TRAIN_FILE:-/workspace/user_data/data/fapo-train-boxed.parquet}
TEST_FILE=${TEST_FILE:-/workspace/user_data/data/fapo-test-full-boxed.parquet}
CKPTS_DIR=${CKPTS_DIR:-/workspace/user_data/ckpts/fapo-rule-npu}

NNODES=${NNODES:-1}
NPU_PER_NODE=${NPU_PER_NODE:-4}

train_batch_size=${TRAIN_BATCH_SIZE:-8}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-8}
n_resp_per_prompt=${ROLLOUT_N:-2}
max_prompt_length=${MAX_PROMPT_LENGTH:-1024}
max_response_length=${MAX_RESPONSE_LENGTH:-2048}
rollout_tp=${ROLLOUT_TP:-1}
sp_size=${SP_SIZE:-1}
max_model_len=${MAX_MODEL_LEN:-4096}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.3}
overlong_buffer_len=${OVERLONG_BUFFER_LEN:-$((max_response_length / 2))}

total_training_steps=${TOTAL_TRAINING_STEPS:-2}
# test_freq=0 disables ALL validation (incl. the forced last-step one at
# ray_trainer.py:1686). The test set holds gpqa-diamond which default_compute_score
# can't score -> NotImplementedError at end of training. Keep 0 for smoke tests.
test_freq=${TEST_FREQ:-0}
save_freq=${SAVE_FREQ:--1}

PROJECT_NAME=${PROJECT_NAME:-FAPO-NPU-rule}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-FAPO-7B-rule-npu-$(date +%Y%m%d_%H%M)}

########################### launch (rule-based reward, no reward model) ###########################
$PYTHON -m verl.trainer.main_ppo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='error' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_batch_size} \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_model_len=${max_model_len} \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    reward.reward_model.enable=False \
    reward.reward_manager.name=dapo \
    +reward.reward_kwargs.overlong_buffer_cfg.enable=True \
    +reward.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward.reward_kwargs.overlong_buffer_cfg.log=True \
    +reward.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.logger='["console"]' \
    trainer.balance_batch=True \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node=${NPU_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.val_before_train=False \
    trainer.test_freq=${test_freq} \
    trainer.save_freq=${save_freq} \
    trainer.total_training_steps=${total_training_steps} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.log_val_generations=0 \
    "$@"
