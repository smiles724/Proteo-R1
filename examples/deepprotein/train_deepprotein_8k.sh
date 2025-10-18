set -x


export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000

# Additional environment variables for distributed training
export NCCL_TIMEOUT=7200
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Memory management settings
export RAY_DISABLE_IMPORT_WARNING=1
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
export RAY_DEDUP_LOGS=0

# Add ProteinFM parent directory to Python path for custom model registration
# RLLM_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && cd ../../.. && pwd )
# More robust RLLM_DIR detection that works with source
# RLLM_DIR=$(python3 -c "import rllm, os; p = next(iter(getattr(rllm,'__path__',[])), None); print(os.path.dirname(p) if p else os.path.dirname(os.path.dirname(os.path.abspath(rllm.__file__))))")

# Get the directory of this script
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# Assume the project root is three levels up from the script's directory
PROJECT_ROOT=$( cd -- "$SCRIPT_DIR/../../../" &> /dev/null && pwd )
cd "${PROJECT_ROOT}"

export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH}

# Find the directory where rllm package is located
# RLLM_DIR=$(python3 -c "import rllm, os; p = next(iter(getattr(rllm,'__path__',[])), None); print(os.path.dirname(p) if p else os.path.dirname(os.path.dirname(os.path.abspath(rllm.__file__))))")

# For training with prefix MLP, use the full PLLM wrapper
# This includes: base LLM + protein encoders (frozen) + prefix MLP (trainable)
MODEL_PATH=${PROJECT_ROOT}/rllm/model/pllm

python3 -m rllm.examples.deepprotein.train_deepprotein \
    algorithm.adv_estimator=grpo \
    +data.dataset_name=thermostability \
    data.train_batch_size=128 \
    data.val_batch_size=30 \
    data.max_prompt_length=2048 \
    data.max_response_length=8192 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    +actor_rollout_ref.model.external_lib=rllm.model.register_pllm \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=30000 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    +actor_rollout_ref.actor.grad_norm_threshold=10 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode="async" \
    +actor_rollout_ref.rollout.chat_scheduler=verl.schedulers.completions_scheduler.CompletionsScheduler \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    +algorithm.mask_truncated_samples=False \
    +algorithm.clip_advantages=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='rllm-agent' \
    trainer.experiment_name='deepprotein-1.5b-8k' \
    +trainer.log_freq=5 \
    +trainer.nccl_timeout=7200 \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.default_hdfs_dir=null \
    +agent.max_steps=1 \
    +agent.use_stepwise_advantage=False \
    trainer.total_epochs=100
