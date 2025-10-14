# protein_llm

## Environment Setup
```
conda create -n protein_llm python=3.10 -y && conda activate protein_llm
# cu128 is for the latest version, downgrade it if conflicting with your environment
(protein_llm): pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
(protein_llm): pip install -e .
```

## Data
```
data/
|___train_subset_1k.jsonl # an example dataset with 1000 samples
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
(protein_llm): python -m protein_llm.train model=qwen25_3b trainer.per_device_train_batch_size=4 trainer.num_train_epochs=10

# train PLLM with Qwen2.5-0.5B-Instuct as the base llm by a 4090 GPU
(protein_llm): python -m protein_llm.train model=qwen25_05b trainer.per_device_train_batch_size=2 trainer.num_train_epochs=10
```

After training, logging information and checkpoints will be saved to `./results`. For example, the two commands above will give:
```
results/
|___qwen25_05b_model-qwen25_05b_trainer.per_device_train_batch_size-2_trainer.num_train_epochs-10
|   |___checkpoint-*
|   |   |___*           # model weights and configuration files
|   |___config.yaml     # experimental configuration
|___qwen25_3b_model-qwen25_3b_trainer.per_device_train_batch_size-4_trainer.num_train_epochs-10
```

### Train by your own data
Please form your data as a `.jsonl` file with exact the same structure and field names with `train_subset_1k.jsonl` and place your `.jsonl` file at `./data`.
Then, run the command by:
```
# Train a PLLM model with Qwen2.5-3B-Instruct as the base llm with your new data
(protein_llm): python -m protein_llm.train model=qwen25_3b trainer.per_device_train_batch_size=4 trainer.num_train_epochs=10 dataset="{your-jsonl-name}.jsonl"
```


### Resume training
Simply run the same command again to resume the training from the latest checkpoint.


### Distributed training
DDP training is under development, so the performance may not be optimal. Please refer to the following commands:
```
# single-node DDP training with 4 H100 GPUs
(protein_llm): torchrun --nproc_per_node=4 src/protein_llm/train.py model=qwen25_3b trainer.per_device_train_batch_size=4 trainer.num_train_epochs=10


# multi-node DDP training (master node)
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
    --master_addr=<master-ip-address> --master_port=29500 \
    src/protein_llm/train.py model=qwen25_3b trainer.per_device_train_batch_size=4 trainer.num_train_epochs=10

# multi-node DDP training (other nodes)
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
    --master_addr=<master-ip-address> --master_port=29500 \
    src/protein_llm/train.py model=qwen25_3b trainer.per_device_train_batch_size=4 trainer.num_train_epochs=10
```

## Inference
We support transformers-style inference by `generate()` APIs. Please refer to the following example codes for usage:
```python
from protein_llm.models.modeling_pllm import PLLM
from transformers import AutoTokenizer

model_path = "{path-to-your-trained-checkpoint}"  # e.g., ./results/qwen25_05b_model-qwen25_05b_trainer.per_device_train_batch_size-2_trainer.num_train_epochs-10/checkpoint-500
pllm = PLLM.from_pretrained(
    model_path, load_pretrained=False # necessary for calling from_pretrained()
)
pllm = pllm.to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

# the first sample in train_subset_1k.jsonl here for illustration
user_content = (
    "You are a professional protein biologist. "
    "Based only on the provided inputs, produce a natural, concise, and biologically accurate description of the protein. "
    "First reason step by step inside a <thinking> block using sequence-derived evidence and structural cues, "
    "then provide the final 2–4 sentence description inside an <answer> block.\n\n"
    "Inputs:\n"
    "Protein: <protein>\n"
    "Structure: <structure>"
)

# Ground-Truth of this sample, used to compare with the generated response below
assistant_content = (
    "<thinking>\n"
    "1. **Signal Peptide and Localization**: The sequence starts with methionine (M), but there is no clear signal "
    "peptide sequence that would suggest secretion or targeting to specific organelles. This suggests a cytoplasmic "
    "or nuclear localization.\n\n"
    "2. **Transmembrane Helices**: The sequence does not show characteristics of "
    "transmembrane helices, such as stretches of hydrophobic residues typically found in membrane-spanning regions. "
    "This suggests the protein is not membrane-bound.\n\n"
    "3. **Repeats and Low-Complexity Segments**: The sequence "
    "does not contain obvious repetitive motifs or low-complexity regions that are often associated with structural "
    "or functional repeats.\n\n"
    "4. **Catalytic Motifs/Domains**: The sequence contains cysteine residues (C) and "
    "histidine (H) that could potentially form a zinc finger or other metal-binding motif, but there is no clear "
    "pattern indicating a known catalytic domain.\n\n"
    "5. **Family and Function**: The sequence contains a segment "
    "\"PCPCG\" which is a characteristic motif found in some proteins of the UPF0225 family. This family is known "
    "for proteins with unknown functions, often involved in stress responses or regulatory roles.\n\n"
    "6. **Overall "
    "Function**: Given the lack of clear catalytic motifs or transmembrane regions, and the presence of a UPF0225 "
    "family motif, the protein is likely involved in a regulatory or structural role within the cell, potentially "
    "related to stress response or protein-protein interactions.\n"
    "</thinking>\n\n"
    "<answer>\n"
    "Belongs to the UPF0225 family.\n"
    "</answer>"
)

aa_seq = "MSKGTPSRGKRQTQTHLTCRRCGRMSYHKRHKICSSCGFGRSTRMRSYGWITKRPKVATH"
structure = "<|chain:A|> <|chain_sep|> #ddddvvvvpppddqfdqdppprdraqgpvqragpqqggpndpggdddpvvddddpdddd"

test_prompt = tokenizer.apply_chat_template(
    [dict(role="user", content=user_content)], add_generation_prompt=True, tokenize=False
)
test_inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")
generated_ids = pllm.generate(
    **test_inputs,
    aa_seq=[aa_seq],
    stru_str=[structure],
    max_new_tokens=1024
)

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(test_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
```
