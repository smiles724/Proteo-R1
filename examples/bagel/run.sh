#!/bin/bash

################################################################################
# BAGEL Model Training with FSDP2
################################################################################
#
# DESCRIPTION:
#   Train BAGEL multimodal model with unified vision understanding and
#   generation capabilities using FSDP2 distributed training.
#
# KEY FEATURES:
#   - Vision understanding and generation
#   - Qwen2-based LLM with MoT (Mixture of Tokens)
#   - SigLIP vision encoder
#   - VAE for image generation
#   - Sequence packing support
#   - FSDP2 distributed training
#   - Optional Native Sparse Attention (NSA)
#
# REQUIREMENTS:
#   - 8x GPUs (A100/H100 recommended, 80GB VRAM)
#   - flash-attn: pip install flash-attn --no-build-isolation
#   - Optional NSA: pip install git+https://github.com/XunhaoLai/native-sparse-attention-triton.git
#
# DATASET:
#   Prepare your dataset in BAGEL format (Parquet/Arrow/JSON):
#   Example dataset: https://huggingface.co/datasets/kcz358/bagel-example
#
#   Example dataset entry:
#   ```json
#   {
#     "messages": [
#       {
#         "role": "user",
#         "content": [
#           {"type": "image_url", "image_url": {"url": "path/to/image.jpg"}},
#           {"type": "text", "text": "Describe this image"}
#         ]
#       },
#       {
#         "role": "assistant",
#         "content": [{"type": "text", "text": "This image shows..."}]
#       }
#     ]
#   }
#   ```
#
# MODEL CHECKPOINT:
#   You can use either:
#   1. Original BAGEL weights (requires config.json conversion)
#      See: https://huggingface.co/kcz358/bagel_hf/blob/main/config.json
#   2. Converted HF weights: lmms-lab/BAGEL-7B-MoT-ver.LE (recommended)
#
# CONFIGURATION:
#   Edit example_config.yaml to customize:
#   - Model checkpoint: load_from_pretrained_path
#   - Dataset path: datasets[0].path
#   - Batch size: per_device_train_batch_size
#   - Packing: packing (true/false)
#   - Visual understanding: extra_kwargs.visual_und
#
# PERFORMANCE TIPS:
#   - Enable packing for better GPU utilization (packing: true)
#   - Use NSA for long sequences (enable monkey_patch_kwargs)
#   - Adjust packing_length based on GPU memory (default: 4096)
#   - Monitor memory with: watch -n 1 nvidia-smi
#
# ADVANCED FEATURES:
#   - Native Sparse Attention: Uncomment monkey_patch_kwargs in config
#   - Mixed Understanding/Generation: Set visual_und: true/false
#
################################################################################

# Number of GPUs
NGPUS=8

# Training command
torchrun --nproc_per_node=${NGPUS} \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port=12357 \
  -m lmms_engine.launch.cli \
  config_yaml=examples/bagel/example_config.yaml

################################################################################
# MULTI-NODE TRAINING:
#
# On rank 0 node:
# torchrun --nproc_per_node=8 \
#   --nnodes=2 \
#   --node_rank=0 \
#   --master_addr=<RANK_0_IP> \
#   --master_port=12357 \
#   -m lmms_engine.launch.cli \
#   config_yaml=examples/bagel/example_config.yaml
#
# On rank 1 node:
# torchrun --nproc_per_node=8 \
#   --nnodes=2 \
#   --node_rank=1 \
#   --master_addr=<RANK_0_IP> \
#   --master_port=12357 \
#   -m lmms_engine.launch.cli \
#   config_yaml=examples/bagel/example_config.yaml
#
################################################################################
#
# TROUBLESHOOTING:
#   - If config.json is not HF compatible, use the converted weights
#   - For memory issues, reduce batch size or packing_length
#   - Check docs/models/bagel.md for detailed configuration options
#
################################################################################
