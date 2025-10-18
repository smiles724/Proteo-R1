# DeepProtein Examples

This directory contains examples for training and running protein analysis agents using the RLLM framework. The protein agent uses ProteinFM (PLLM) to analyze protein sequences and predict properties like thermostability.

Our examples use the following:
* ProteinFM (PLLM) as the base model
* Protein datasets from HuggingFace (thermostability, etc.)
* Single-turn environment for protein property prediction

## Model Hosting

### Using Full PLLM Server (Recommended)

Start the full PLLM server with protein encoders:

```bash
cd /mnt/efs/erran/rllm_v02/ProteinFM/model/vllm_infer

python serve_pllm_full.py \
    --model-path ../pllm \
    --host 0.0.0.0 \
    --port 30000 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9
```

This will:
- Load the complete PLLM model (4.0 GB)
- Initialize protein encoder (ESM-2)
- Initialize structure encoder (Foldseek)
- Start vLLM for efficient text generation
- Expose OpenAI-compatible API on port 30000

The server should be accessible at `http://localhost:30000/v1`

See [ProteinFM/model/vllm_infer/README.md](../../../ProteinFM/model/vllm_infer/README.md) for more details.

## Dataset Preparation

Prepare the required protein datasets:

```bash
cd examples/deepprotein
python prepare_protein_data.py
```

This will:
- Download protein datasets from HuggingFace (thermostability by default)
- Preprocess and split into train/test sets
- Register both datasets with the RLLM DatasetRegistry

Available datasets:
- `thermostability` - Protein thermostability prediction
- `gb1` - GB1 protein fitness
- `aav` - AAV protein fitness
- And more from [fangwu97/ProteinData](https://huggingface.co/datasets/fangwu97/ProteinData)

## Running Inference

Once your model server is running and datasets are prepared, you can run inference:

```bash
cd examples/deepprotein
python run_deepprotein.py
```

### Configuration Options

You can modify the inference script parameters:

- `n_parallel_agents`: Number of parallel agents (default: 64)
- `model_name`: Model identifier for vLLM server (default: "pllm")
- `base_url`: API server URL (default: "http://localhost:30000/v1")
- `max_response_length`: Maximum response length (default: 2048)
- `max_prompt_length`: Maximum prompt length (default: 1024)
- `temperature`: Sampling temperature (default: 0.6)
- `top_p`: Top-p sampling (default: 0.95)

You can also set the dataset via environment variable:

```bash
PROTEIN_DATASET_NAME=gb1 python run_deepprotein.py
```

The script will:
1. Load the protein test dataset
2. Create protein analysis tasks
3. Execute tasks using the AgentExecutionEngine
4. Compute pass@k metrics

## Training

To train a protein agent with PPO:

```bash
cd examples/deepprotein

# Prepare data first
python prepare_protein_data.py

# Train with default settings
bash train_deepprotein_8k.sh
```

### Training Configuration

The training script uses Hydra for configuration. Key parameters:

```bash
python -m examples.deepprotein.train_deepprotein \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=128 \
    data.val_batch_size=30 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=/path/to/pllm \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.6 \
    +data.dataset_name=thermostability
```

See `train_deepprotein_8k.sh` for a complete example.

## Project Structure

```
deepprotein/
├── README.md                      # This file
├── run_deepprotein.py            # Inference script
├── train_deepprotein.py          # Training script
├── prepare_protein_data.py       # Dataset preparation
├── train_deepprotein_8k.sh       # Training script (8k context)
└── data/
    ├── dataset.md                # Dataset documentation
    └── protein.py                # Dataset classes
```

## Protein Agent

The `ProteinAgent` class handles:
- Protein sequence input formatting
- Structure sequence (3Di tokens) processing
- Interaction with ProteinFM model
- Property prediction and analysis

Example task format:

```python
{
    "prompt": "Predict the thermostability of this protein:",
    "aa_seq": "MKTFFVAIATGAFSATA...",
    "stru_seq": "ACDEFGHIKLMNPQRSTVWY...",  # Optional
    "ground_truth": 41.95,  # Target value
}
```

## Reward Function

The `protein_reward_fn` evaluates protein predictions:
- Compares predicted values to ground truth
- Calculates error metrics (MSE, MAE, etc.)
- Supports various protein property types

## Performance Tips

### For Inference

1. **Use vLLM server** for best performance (100+ req/s)
2. **Adjust n_parallel_agents** based on your hardware
3. **Tune temperature** for exploration vs exploitation
4. **Use appropriate max_response_length** for your task

### For Training

1. **Start with smaller context** (8k) before scaling to 24k
2. **Use FSDP** for large models
3. **Enable gradient checkpointing** to save memory
4. **Tune learning rate** (typically 1e-6 for protein tasks)

## Troubleshooting

### vLLM Server Not Running

```
Error: Connection refused to http://localhost:30000/v1
```

**Solution:** Start the full PLLM server first:
```bash
cd /mnt/efs/erran/rllm_v02/ProteinFM/model/vllm_infer
python serve_pllm_full.py --model-path ../pllm --port 30000
```

### Dataset Not Found

```
Dataset not found, preparing dataset...
```

**Solution:** Run dataset preparation:
```bash
python prepare_protein_data.py
```

### CUDA Out of Memory

**Solutions:**
1. Reduce `n_parallel_agents`
2. Reduce `max_response_length`
3. Use smaller batch size for training
4. Enable gradient checkpointing

### Import Errors

Make sure you're running from the correct directory:
```bash
cd /mnt/efs/erran/rllm_v02/rllm
python -m examples.deepprotein.run_deepprotein
```

## Example Output

```
Loading protein test dataset: thermostability
Dataset loaded: 1000 samples
Starting inference with 64 parallel agents...

Progress: 100%|████████████| 16000/16000 [10:30<00:00, 25.4it/s]

Results:
  Total tasks: 16000
  Successful: 15800 (98.75%)
  Pass@1: 0.75
  Pass@5: 0.89
  Pass@10: 0.94
  Pass@16: 0.97
```

## Resources

- [ProteinFM Model](../../../ProteinFM/model/README.md)
- [vLLM Serving Guide](../../../ProteinFM/model/vllm_infer/README.md)
- [RLLM Framework](../../README.md)
- [Protein Datasets](https://huggingface.co/datasets/fangwu97/ProteinData)

## Citation

If you use this code, please cite:

```bibtex
@software{deepprotein2025,
  title={DeepProtein: Protein Analysis with Reinforcement Learning},
  author={Your Name},
  year={2025}
}
```

---

**Status:** Production Ready ✅  
**Last Updated:** October 13, 2025

