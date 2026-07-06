#!/usr/bin/env bash
# FAPO colocate (GenRM) | Qwen2.5-Math-7B policy + FAPO-GenRM-4B reward | Ascend NPU
#
# Adapted from run_qwen_7b_rm_colocate.sh for a single node of Ascend 910B NPU.
# The original GPU script is left untouched; this is a separate NPU entry point.
#
# This ships a SMOKE-TEST-scale config (small batch/len, 2 steps, console-only log)
# to validate the full colocate reward-loop pipeline on NPU. To do a real run,
# scale up the knobs below, e.g.:
#   TRAIN_BATCH_SIZE=64 ROLLOUT_N=8 MAX_RESPONSE_LENGTH=8192 TOTAL_TRAINING_STEPS=200 \
#     bash examples/fapo_trainer/run_qwen_7b_rm_colocate_npu.sh
#
# Differences vs the original GPU script:
#   - no `ray job submit` wrapper; verl's main_ppo calls ray.init() itself.
#   - NPU env vars (HCCL_*, RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES).
#   - conservative NPU settings: tp/sp=1, micro_batch=1, gpu_mem_util=0.3,
#     param/optimizer offload, torch_compile disabled.
#   - colocate reward model also uses low gpu_memory_utilization.

set -xeuo pipefail

# Interpreter: this box has python=3.11 (with torch_npu/vllm/verl) but python3=3.10
# (system, no deps). Default to `python`; override with PYTHON=/path/to/python.
PYTHON=${PYTHON:-python}

########################### NPU environment ###########################
# NOTE: this container hard-caps memlock at 64MB (ulimit -l, soft==hard), which is
# below what HCCL wants and cannot be raised without CAP_SYS_RESOURCE. The classic
# Ascend requirement is `ulimit -l unlimited` set at container creation time
# (`--ulimit memlock=-1:-1`). If vLLM workers still fail HCCL init with
# "Resource temporarily unavailable", that is the root cause to fix at the host.
ulimit -s unlimited 2>/dev/null || true
# Colocate spawns many vLLM engines whose multiproc workers + HCCL create a lot of
# threads (well over 10k). A low container pids-limit makes pthread_create fail with
# "Resource temporarily unavailable" across HCCL / boost::asio / gRPC. The cgroup is
# mounted read-only inside the container, so it must be raised at container creation
# time (`docker run --pids-limit 200000`). Warn here if it is too low.
_pids=$(cat /sys/fs/cgroup/pids/pids.max 2>/dev/null || echo max)
if [ "${_pids}" != "max" ] && [ "${_pids}" -lt 50000 ] 2>/dev/null; then
    echo "WARNING: container pids-limit is ${_pids} (/sys/fs/cgroup/pids/pids.max)." >&2
    echo "         FAPO colocate needs >10k threads. Recreate the container with" >&2
    echo "         --pids-limit 200000 (or unlimited); pthread_create will otherwise fail." >&2
fi
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-1500}
# Widen far beyond the default 50 ports: colocate spawns many vLLM engines
# (rollout + GRM, data-parallel across NPUs), each opening its own HCCL sockets.
export HCCL_HOST_SOCKET_PORT_RANGE=${HCCL_HOST_SOCKET_PORT_RANGE:-60000-60500}
export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-61000-61500}
export HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-AIV}
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
# Pin specific NPUs on a shared / multi-card box. This is the torch_npu equivalent of
# CUDA_VISIBLE_DEVICES. E.g. this container currently exposes physical NPUs 3,4,5,6
# (npu-smi), so set ASCEND_RT_VISIBLE_DEVICES=3,4,5,6. When unset, all NPUs visible to
# the container are used (re-numbered as logical 0..N-1) -- but if physical IDs don't
# start at 0, pin explicitly or device init can address the wrong/absent card.
export ASCEND_RT_VISIBLE_DEVICES=3,4,5,6
if [ -n "${ASCEND_RT_VISIBLE_DEVICES:-}" ]; then export ASCEND_RT_VISIBLE_DEVICES; fi
# Cap thread-hungry libraries to stay under tight container pids-limits. The HF
# tokenizer's Rust rayon pool (encode_batch) otherwise bursts to num_cpus threads
# per AgentLoopWorker and trips pthread_create EAGAIN around ~10k threads.
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-1}

# DEVICE is auto-detected by probing torch_npu; override only for special cases.
DEVICE=${DEVICE:-$($PYTHON -c 'import torch_npu' 2>/dev/null && echo npu || echo gpu)}

########################### paths ###########################
MODEL_PATH=${MODEL_PATH:-/workspace/user_data/models/Qwen/Qwen2.5-Math-7B}
GRM_PATH=${GRM_PATH:-/workspace/user_data/models/FAPO-GenRM-4B}
TRAIN_FILE=${TRAIN_FILE:-/workspace/user_data/data/fapo-train-boxed.parquet}
TEST_FILE=${TEST_FILE:-/workspace/user_data/data/fapo-test-full-boxed.parquet}
CKPTS_DIR=${CKPTS_DIR:-/workspace/user_data/ckpts/fapo-npu}
REWARD_FN=${REWARD_FN:-examples/fapo_trainer/reward_fn.py}

########################### topology ###########################
NNODES=${NNODES:-1}
NPU_PER_NODE=${NPU_PER_NODE:-4}

########################### smoke-test knobs (override to scale up) ###########################
train_batch_size=${TRAIN_BATCH_SIZE:-8}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-8}
n_resp_per_prompt=${ROLLOUT_N:-2}
max_prompt_length=${MAX_PROMPT_LENGTH:-1024}
max_response_length=${MAX_RESPONSE_LENGTH:-2048}
rollout_tp=${ROLLOUT_TP:-1}
grm_tp=${GRM_TP:-1}
sp_size=${SP_SIZE:-1}
# vLLM defaults max_model_len to the model's max_position_embeddings, which OOMs the
# KV cache. Cap explicitly. NOTE the two models differ:
#   Qwen2.5-Math-7B (policy) max_position_embeddings = 4096  -> cap at 4096 (native)
#   FAPO-GenRM-4B  (reward)  max_position_embeddings = 262144 -> cap small (8192)
# Policy cap must be >= max_prompt_length + max_response_length.
max_model_len=${MAX_MODEL_LEN:-4096}
grm_max_model_len=${GRM_MAX_MODEL_LEN:-8192}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.9}
grm_gpu_mem_util=${GRM_GPU_MEM_UTIL:-0.9}
overlong_buffer_len=${OVERLONG_BUFFER_LEN:-$((max_response_length / 2))}

total_training_steps=${TOTAL_TRAINING_STEPS:-2}
# test_freq=0 disables ALL validation (incl. the forced last-step one). The test
# set's gpqa-diamond isn't scoreable by default_compute_score -> NotImplementedError
# at end of training. Keep 0 for smoke tests.
test_freq=${TEST_FREQ:-0}
save_freq=${SAVE_FREQ:--1}        # do not checkpoint during smoke run

PROJECT_NAME=${PROJECT_NAME:-FAPO-NPU}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-FAPO-7B-colocate-npu-$(date +%Y%m%d_%H%M)}

########################### launch ###########################
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
    reward.reward_model.enable=True \
    reward.reward_model.enable_resource_pool=False \
    reward.reward_model.model_path=${GRM_PATH} \
    reward.reward_model.rollout.name=vllm \
    reward.reward_model.rollout.gpu_memory_utilization=${grm_gpu_mem_util} \
    reward.reward_model.rollout.tensor_model_parallel_size=${grm_tp} \
    reward.reward_model.rollout.free_cache_engine=True \
    reward.reward_model.rollout.max_model_len=${grm_max_model_len} \
    reward.reward_manager.name=dapo \
    +reward.reward_kwargs.overlong_buffer_cfg.enable=True \
    +reward.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward.reward_kwargs.overlong_buffer_cfg.log=True \
    +reward.reward_kwargs.max_resp_len=${max_response_length} \
    reward.custom_reward_function.path=${REWARD_FN} \
    reward.custom_reward_function.name=compute_score_fapo \
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
