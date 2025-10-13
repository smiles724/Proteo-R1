

In the data preprocessing folder, there are **four Colab notebooks** that form an end-to-end pipeline from dataset checks to CoT generation, structure tokenization (Foldseek 3Di), and a forward/loss smoke test for understanding model.

**Run them in this order:**
1. **`ProteinDT_molreasoner_pro2dec_demo_colab.ipynb`** — Demo / check all APIs and dataset
2. **`ProteinDT_molreasoner_pro2dec_cot_colab.ipynb`** — Generate Chain-of-Thought (CoT)
3. **`ProteinDT_molreasoner_pro2dec_PDB3Di_colab.ipynb`** — AlphaFold PDB → Foldseek 3Di
4. **`BigProteinQwen_Colab_Debugging.ipynb`** — Test each model component; forward & loss

## Quick Start

1. Open each notebook in **Google Colab**.
2. Set the paths at the top (e.g., `BASE_DIR`) and required creds (e.g., `OPENAI_API_KEY`).
3. Run the notebooks **in order** (1→4). Notebooks are resume-friendly and avoid redoing work where possible.

---

## Notebooks

### 1) `ProteinDT_molreasoner_pro2dec_demo_colab.ipynb` — Demo / data & API checks
**Purpose**
- Sanity-check the **ProteinDT** dataset (e.g., SwissProtCLAP `protein_sequence.txt` ↔ `text_sequence.txt` alignment, counts).
- Verify external structure APIs:
  - **AlphaFold** (view/download a sample structure)
  - **RCSB/PDBe** (map UniProt → experimental PDB IDs)
- Confirm Google Drive I/O and basic utilities.

**Inputs**
- Local copy of ProteinDT under `BASE_DIR`.

**Outputs**
- Printed summaries, small CSVs, and optional sample downloads under:
  - `BASE_DIR/`
  - `BASE_DIR/downloads/<UniProt>/`

---

### 2) `ProteinDT_molreasoner_pro2dec_cot_colab.ipynb` — Generate CoT
**Purpose**
- Build prompts from ProteinDT (sequence → description).
- Submit **OpenAI Batch** jobs (chunked, resume-safe) and **fetch** results.
- Produce two SFT files:
  - **Parsed CoT**: `<thinking>…</thinking>\n\n<answer>…</answer>`
  - **Raw**: unmodified model output for audit.

**Key Features**
- Loose parsing (tolerates missing closing tags or answer-only).
- Optional **fixed-answer** mode: provide the ground-truth `<answer>`, model fills only `<thinking>`.

**Outputs** (under something like)
- `BASE_DIR/gpt_batch_protein2desc*/`
  - `*_input.jsonl`, `*_meta.json`, `*_batch_info.json`
  - `protein2desc_cot_sft.json` (parsed CoT)
  - `protein2desc_cot_raw.json` (raw text)

---

### 3) `ProteinDT_molreasoner_pro2dec_PDB3Di_colab.ipynb` — AlphaFold PDB → Foldseek 3Di
**Purpose**
- For your UniProt set (e.g., from the CoT batches), ensure **AlphaFold PDB** exists locally (v4→v3→v2; **PDB only**).
- Convert structures to **Foldseek 3Di tokens** for **all chains** found (A–Z, 0–9).
- Concatenate chains with special tokens:
  - chain separator: `<|chain_sep|>`
  - optional per-chain tag: `<|chain:{ID}|>`

**Outputs** (under)
- `BASE_DIR/sft_build/foldseek_3di_pdb/`
  - `<UID>.3di.txt` — concatenated 3Di
  - `<UID>.aa.txt` — concatenated AA (aligned with 3Di)
  - `<UID>.chains.json` — chain metadata (IDs, lengths, file used)
  - `3di_manifest_pdb.csv` — manifest
  - `3di_failed_pdb.txt` — failures (if any)

---

### 4) `BigProteinQwen_Colab_Debugging.ipynb` — Model components & forward/loss
**Purpose**
- Unit-style checks for model components (tokenizers/encoders).
- Data collation from your SFT (CoT + 3Di).
- **Forward pass** and **loss** computation on a mini-batch (smoke test).

**Outputs**
- Printed shapes, losses, and sanity diagnostics. Optional logs under your `BASE_DIR`.

---

## Requirements

- **Colab** (recommended) or Python 3.10+.
- Common packages: `pandas`, `tqdm`, `requests`, `torch`, `openai`.
- **Foldseek** binary for 3Di:
  - Place at `bin/foldseek` (or update the path in notebook)
  - Make it executable:
    ```bash
    chmod +x bin/foldseek
    ```
- **OpenAI API key** (for CoT):
  ```python
  import os
  os.environ["OPENAI_API_KEY"] = "sk-..."

- **Training Env**
- All required package/version and python torch version are all in requirement.txt

- **Encoder Weight**
- To load encoders weight, First download weight from Protrek HF repo.
- https://huggingface.co/westlake-repl/ProTrek_35M
- https://huggingface.co/westlake-repl/ProTrek_650M
- And set:
- 35M Encoder
- --protein_config "/protrek/weights/ProTrek_35M/esm2_t12_35M_UR50D" \
  --structure_config "/protrek/weights/ProTrek_35M/foldseek_t12_35M" \
  --protrek_ckpt "/protrek/weights/ProTrek_35M/ProTrek_35M.pt" \
  
- 650M Encoder
  
  --protein_config "/protrek/weights/ProTrek_35M/esm2_t33_650M_UR50D" \
  --structure_config "/protrek/weights/ProTrek_35M/foldseek_t30_150M" \
  --protrek_ckpt "/protrek/weights/ProTrek_35M/ProTrek_650M.pt" \

```python
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel setuptools

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
```


This repo provides three entrypoints:

- **`train_prefix_qwen.py`** — single‑GPU trainer (gradient accumulation; optional encoder finetune).  
- **`train_prefix_qwen_fsdp_vanilla_ckf.py`** — multi‑GPU trainer using only FSDP  
- **`train_prefix_qwen_fsdp_offload1.py`** — FSDP full‑shard with **CPU offload** (plus optional 8‑bit/Adafactor optimizers). (Haven't test yet)

---

## How it works (high‑level)
- Each example contains a **prompt**, **response**, and **protein inputs**:
  - `aa_seq` — amino‑acid sequence (string) → **1024‑D** embedding
  - `stru_str` — Foldseek 3Di structural string (string) → **1024‑D** embedding
- The two 1024‑D embeddings are **concatenated** to **2048‑D**; if one modality is missing, it is **zero‑padded**.
- A small **projector MLP** maps 2048‑D → `prefix_len × hidden_size` and yields a **soft prefix** (one or more learned tokens) directly in the LLM’s hidden space.
- At training time, the prefix is **prepended** to the token embeddings of the textual prompt.
- **Loss is masked** over the prefix and prompt (and its EOS) so cross‑entropy supervises **only the response tokens**.

---

## File layout
```
protein_encoder.py                 # sequence encoder wrapper
structure_encoder.py               # 3Di structure encoder wrapper
train_prefix_qwen.py               # single‑GPU trainer
train_prefix_qwen_fsdp_vanilla_ckf.py  # multi‑GPU trainer (FSDP)
train_prefix_qwen_fsdp_offload1.py # multi‑GPU trainer (FSDP + CPU offload)
```

Both multi‑GPU scripts load encoder weights from a **ProTrek** `.pt` checkpoint using **slot IDs** (e.g., `prot_slot=1`, `stru_slot=3`).

---

## Data format (JSONL)
Each line is a JSON object:
```json
{
  "prompt": "Describe the likely function of this protein.",
  "response": "This appears to be an enzyme with possible hydrolase activity.",
  "aa_seq": "MGDVEK...",         // protein sequence
  "stru_str": "ACDEFGH..."       // Foldseek 3Di string
}
```
The collator builds `[prompt + EOS] + [response + EOS]` and masks labels to supervise **only** the response (EOS included).

---

## Encoder initialization (ProTrek `.pt` + slots)
- **Architectures** come from `--protein-config` and `--structure-config` (ESM‑style configs or local dirs).
- **Weights** come from a ProTrek **`.pt`** checkpoint passed via `--protrek-ckpt`. The state dict is split by numeric **slot IDs**; `--prot-slot` and `--stru-slot` choose which sub‑model to load for **protein** and **structure** encoders respectively.
- If a slot is missing, the script prints a warning and keeps random init for that encoder.

## Rank‑sharded streaming
### Arguments & What They Mean

#### Data & I/O
- `--train-file <path>` — Path to training JSONL.
- `--save-dir <dir>` — Where checkpoints/logs are written (created if missing).
- `--log-every <int>` — Print/record metrics every _N_ steps.

#### Base LLM
- `--model-name <hf_id_or_path>` — Hugging Face model (e.g., `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen2.5-14B-Instruct`).
- `--dtype {fp32,fp16,bf16}` — Compute/load dtype; **bf16** is recommended on Ampere/Hopper; **fp32** is heaviest.

#### Protein & Structure Encoders
- `--protein-config <hf_id_or_path>` — Protein encoder config/weights. _Server default: `esm2_t33_650M_UR50D`._
- `--structure-config <hf_id_or_path>` — Structure encoder config/weights. _Server default: `foldseek_t30_150M`._
- `--train-encoders` — If set, encoders are trainable (more VRAM/optimizer state). Omit to freeze.
- `--protrek-ckpt <path>` — path to a ProTrek checkpoint (`ProTrek_650M.pt`) to initialize adapters/heads.
- `--prot-slot <int>` / `--stru-slot <int>` — Which prefix “slot” each encoder feeds Default set: --prot-slot 1 / --stru-slot 3.

#### Prefix Conditioning (projection head)
- `--prefix-len <int>` — How many virtual tokens to prepend as the prefix.
- `--proj-hid <int>` — Hidden size of the projection MLP from encoders → prefix space.
- `--dropout <float>` — Dropout rate inside the projection/prefix modules.
- `--single-token-prefix` — Use a 1-token prefix (if enabled); otherwise `--prefix-len` controls length.

#### Optimization & Schedule
- `--batch-size <int>` — **Global** batch size (across all ranks). If OOM, reduce or increase `--accum-steps`.
- `--accum-steps <int>` — Gradient accumulation steps (effective batch = `batch_size × accum_steps`).
- `--max-len <int>` — Tokenized sequence length (larger → more activations/VRAM).
- `--epochs <int>` — Number of full passes over the dataset (or use `--steps-per-epoch` to cap).
- `--lr <float>` — Base learning rate.
- `--warmup-ratio <float>` — Fraction of total steps used for LR warmup.
- `--weight-decay <float>` — AdamW weight decay.

#### Weights & Biases (optional)
- `--wandb` — Enable W&B logging (**requires** `WANDB_API_KEY` in env).
- `--wandb-project <str>` / `--wandb-run-name <str>` / `--wandb-entity <str>` — W&B project/run metadata.
- `--wandb-tags tag1,tag2,...` — Comma-separated tags.  
  _Environment tips:_ `WANDB_MODE=online`, set `WANDB_DIR` to a fast disk, and `export WANDB_API_KEY=<your_key>` before launch.

#### Runtime Notes
- If use wandb, need to login before running the script
- FSDP sharding is configured inside the script/policy; `torchrun --nproc_per_node <N> ...` controls the number of GPUs used on the node.
- For large models (e.g., 14B in `fp32`), start with smaller `--batch-size` and `--max-len`, and prefer `bf16` on supported GPUs.
- Activation checkpointing (if enabled in the policy) further reduces VRAM.



---

## Quickstart

### A) Single‑GPU
```bash
python train_prefix_qwen.py \
  --train-file /data/train.jsonl \
  --val-file   /data/val.jsonl \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --protein-config   esm2_t33_650M_UR50D \
  --structure-config foldseek_t30_150M \
  --protrek-ckpt     /weights/ProTrek_35M.pt \
  --prot-slot 1 --stru-slot 3 \
  --dtype fp32 \
  --prefix-len 4 \
  --batch-size 4 --accum-steps 4 --max-len 1024 \
  --epochs 1 --lr 1e-5 \
  --save-dir ./runs --save-every 1000
```

### B) Multi‑GPU (FSDP) (Main training file)
```bash
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" train_prefix_qwen_fsdp_vanilla_ckf.py \
  --train-file level2_10k_train.jsonl \
  --model-name Qwen/Qwen2.5-14B-Instruct \
  --protein-config   esm2_t33_650M_UR50D \
  --structure-config foldseek_t30_150M \
  --train-encoders \
  --dtype fp32 \
  --prefix-len 8 \
  --proj-hid 2048 \
  --dropout 0.10 \
  --batch-size 16 --accum-steps 1 --max-len 2048 \
  --epochs 1 --lr 1e-5 --warmup-ratio 0.03 --weight-decay 0.05 \
  --save-dir qwen2p5_14b_vanilla_fsdp_fp32 \
  --log-every 20 \
  --wandb --wandb-project MyProject --wandb-run-name qwen2p5_13b_sft_fp32 \
  --wandb-entity team_name --wandb-tags qwen2p5,fsdp,fp32,train-encoders \
  --protrek-ckpt ProTrek_650M.pt \
  --prot-slot 1 --stru-slot 3
```

**`accelerate_fsdp_bf16.yaml` (example):**
```yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
num_processes: 4             # number of GPUs
mixed_precision: bf16
gradient_accumulation_steps: 8
downcast_bf16: 'no'
fsdp_config:
  sharding_strategy: FULL_SHARD
  auto_wrap_policy: TRANSFORMER_BASED_WRAP
  transformer_layer_cls_to_wrap: Qwen2DecoderLayer,LlamaDecoderLayer,GPTNeoXLayer
  backward_prefetch: BACKWARD_PRE
  state_dict_type: FULL_STATE_DICT
  sync_module_states: true
  cpu_offload: false
  use_orig_params: true
  limit_all_gathers: true
```

### C) FSDP + **CPU offload**
```bash
accelerate launch --config_file accelerate_cpu_offload_bf16.yaml train_prefix_qwen_fsdp_offload1.py \
  --train-file /data/train.jsonl \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --protein-config   esm2_t33_650M_UR50D \
  --structure-config foldseek_t30_150M \
  --protrek-ckpt     /weights/ProTrek_35M.pt \
  --prot-slot 1 --stru-slot 3 \
  --prefix-len 4 --batch-size 1 --accum-steps 8 \
  --max-len 1024 --epochs 1 --lr 1e-4 \
  --optimizer adam8bit --train-encoders \
  --save-dir ./runs_offload
```
*This variant performs selected computations/parameters on CPU to fit larger models; expect slower throughput.*

---

## Prompts used for CoT generation

### System prompt (with inlined example)
```python
You are a professional protein biologist. Your task is to generate a natural, concise, and biologically accurate description of a protein based **only** on its amino-acid sequence.
```

### Example:
```python
Protein UniProt: {ex_uid}
Protein sequence: {trunc(ex_seq, 1200)}
Answer: <answer> {ex_gt} </answer>
```
### Now try this:
Given ONLY the sequence, first think step by step about plausible features (signal peptides, transmembrane helices, repeats, low-complexity segments, catalytic motifs/domains, localization signals), then produce a polished 2–4 sentence description.

Your final answer **must** be returned in the format:
```python
<thinking>
[steps: Your reasoning steps using sequence-derived evidence]
</thinking>

<answer>
[2–4 sentences; A natural, concise scientific description of the protein]
</answer>
```

### User message template (fixed-answer CoT run)
```python
Protein UniProt: {uniprot_id}
Protein sequence (truncated to 2000 aa if long):
{trunc(seq, 2000)}

<thinking>
[Your reasoning steps using sequence-derived evidence]
</thinking>

<answer>
{ground_truth}
</answer>
```

### Result CoT example
```jsonl
{
  "response": "<thinking>\n- Sequence length ~246 aa (exact shown ~246), typical size of 14-3-3 proteins (~246–255 aa).\n- N-terminal region begins MDKSELVQKAKLSEQAERY... matches highly conserved 14-3-3 consensus N-terminus (acidic/charged helix).\n- Overall composition: acidic and helical-favoring residues, no signal peptide or transmembrane segments → cytosolic.\n- Contains conserved motifs and residues characteristic of 14-3-3: charged ridge and amphipathic grooves for phospho-peptide binding.\n- Sequence lacks signal peptides, targeting peptides, or enzymatic catalytic motifs, consistent with a scaffold/adaptor role.\n</thinking>\n\n<answer>\nProbable 14-3-3 family adaptor protein (~246 aa) that resides in the cytosol and binds phosphoserine/phosphothreonine motifs in diverse client proteins. Functions as a dimeric scaffold to modulate target localization, stability and activity in multiple signaling pathways. Belongs to the 14-3-3 protein family.\n</answer>"
}
```
Thinking:

Sequence length ~246 aa (exact shown ~246), typical size of 14-3-3 proteins (~246–255 aa).
N-terminal region begins MDKSELVQKAKLSEQAERY... matches highly conserved 14-3-3 consensus N-terminus (acidic/charged helix).
Overall composition: acidic and helical-favoring residues, no signal peptide or transmembrane segments → cytosolic.
Contains conserved motifs and residues characteristic of 14-3-3: charged ridge and amphipathic grooves for phospho-peptide binding.\n- Sequence lacks signal peptides, targeting peptides, or enzymatic catalytic motifs, consistent with a scaffold/adaptor role.

Answer:

Probable 14-3-3 family adaptor protein (~246 aa) that resides in the cytosol and binds phosphoserine/phosphothreonine motifs in diverse client proteins. Functions as a dimeric scaffold to modulate target localization, stability and activity in multiple signaling pathways. Belongs to the 14-3-3 protein family.

### Prompt used for training

**Training JSONL schema (example)**

```jsonl
{"prompt": "You are a professional protein biologist. Based only on the provided inputs, produce a natural, concise, and biologically accurate description of the protein. First reason step by step inside a <thinking> block using sequence-derived evidence and structural cues, then provide the final 2–4 sentence description inside an <answer> block.", "response": "<thinking>…</thinking>\n\n<answer>…</answer>", "aa_seq": "…", "stru_str": "…", "uniprot_id": "…", "caption": "…", "_af2_pdb_path": "…", "_3di_chain_meta": [...]}
```

