# protein_llm

## Environment Setup
```
conda create -n protein_llm python=3.10 -y && conda activate protein_llm
(protein_llm): pip install vllm==0.10.1
# install torch later than vllm to prevent from wrong torch installation activated by vllm
# cu128 is for the latest version, downgrade it if conflicting with your environment
(protein_llm): pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
(protein_llm): pip install llamafactory==0.9.3
# Finally, install other dependencies to make sure the correct version
(protein_llm): pip install -e .

# (optional) if you meet issues of MPI when running deepspeed
(protein_llm): conda install -c conda-forge openmpi mpi4py
(protein_llm): conda install cuda-cudart cuda-version=12
```

## Data
```
data/
|___pdb_sft_850k_1021.jsonl
```

## Pretrained Models
Before training, please prepare the following pretrained weights:
```
pretrained/
|___ProTrek_650M
    |___esm2_t33_650M_UR50D
    |___foldseek_t30_150M
    |___ProTrek_650M.pt
```
Simply download everything from https://huggingface.co/westlake-repl/ProTrek_650M to `pretrained/ProTrek_650M` by:
```
mkdir -p pretrained/ProTrek_650M
huggingface download westlake-repl/ProTrek_650M --local-dir pretrained/ProTrek_650M
```


## Train
Please refer to the `src/protein_llm/train.py` for training details.
```
# train PLLM with Qwen2.5-3B-Instuct as the base llm by a H100 GPU, reduce trainer.per_device_train_batch_size if meeting OOM error
(protein_llm): deepspeed --num_gpus=1 src/protein_llm/train.py model=qwen25_3b trainer.per_device_train_batch_size=4 trainer.num_train_epochs=10

# train PLLM with Qwen2.5-0.5B-Instuct as the base llm by a 4090 GPU
(protein_llm): deepspeed --num_gpus=1 src/protein_llm/train.py model=qwen25_05b trainer.per_device_train_batch_size=2 trainer.num_train_epochs=10
```

After training, logging information and checkpoints will be saved to `./results`. For example, the two commands above will give:
```
results/
|___qwen25_05b_model-qwen25_05b_trainer.per_device_train_batch_size-2_trainer.num_train_epochs-10
|   |___checkpoint-*
|   |   |___*           # model weights and configuration files
|   |___config.yaml     # experimental configuration
|___qwen25_3b_model-qwen25_3b_trainer.per_device_train_batch_size-4_trainer.num_train_epochs-10``
```

### Train by your own data
Please form your data as a `.jsonl` file with exact the same structure and field names with `train_subset_1k.jsonl` and place your `.jsonl` file at `./data`.
Then, run the command by:
```
# Train a PLLM model with Qwen2.5-3B-Instruct as the base llm with your new data
(protein_llm): deepspeed --num_gpus=1 src/protein_llm/train.py model=qwen25_3b trainer.per_device_train_batch_size=4 trainer.num_train_epochs=10 dataset="{your-jsonl-name}.jsonl"
```


### Resume training
Simply run the same command again to resume the training from the latest checkpoint.


### Distributed training
DDP training is under development, so the performance may not be optimal. Please refer to the following commands:
```
# single-node DDP training with 4 H100 GPUs
(protein_llm): deepspeed --num_gpus=4 src/protein_llm/train.py model=qwen25_3b trainer.per_device_train_batch_size=4 trainer.num_train_epochs=10


# multi-node DDP training (master node)
(protein_llm): torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
    --master_addr=<master-ip-address> --master_port=29500 \
    src/protein_llm/train.py model=qwen25_3b trainer.per_device_train_batch_size=4 trainer.num_train_epochs=10

# multi-node DDP training (other nodes)
(protein_llm): torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
    --master_addr=<master-ip-address> --master_port=29500 \
    src/protein_llm/train.py model=qwen25_3b trainer.per_device_train_batch_size=4 trainer.num_train_epochs=10
```

## Inference
We support transformers-style inference by `generate()` APIs. 
Please refer to `src/protein_llm/models/modeling_pllm.py` for usage:

## TODOs
- [ ] lower lr for protein & structure encoder during SFT
- [ ] batch-level generate by left-padding tokenization 
