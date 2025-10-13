# PLLM Inference Guide

This guide explains how to load a saved PLLM model and run inference.

## Prerequisites

Make sure you have trained and saved a model using `example.py`:

```bash
python example.py
```

This will save the model to `./pllm/`

## Inference Scripts

### 1. Simple Inference (`simple_inference.py`)

Basic inference script that demonstrates:
- Loading a saved model
- Running forward passes on single examples
- Batch inference
- Protein encoding

**Usage:**

```bash
cd /mnt/efs/erran/rllm_v02/ProteinFM/model
python simple_inference.py
```

**What it does:**
- Loads the model from `./pllm/`
- Runs inference on 2 example proteins
- Shows output shapes and predictions
- Displays protein encoding information

### 2. Advanced Inference (`inference_example.py`)

Full-featured inference script with text generation:
- Autoregressive text generation
- Temperature and top-p sampling
- Customizable generation parameters

**Usage:**

```bash
# Basic usage
python inference_example.py

# Custom parameters
python inference_example.py \
    --model-path ./pllm \
    --device cuda \
    --max-new-tokens 150 \
    --temperature 0.8 \
    --top-p 0.95
```

**Arguments:**
- `--model-path`: Path to saved model (default: `./pllm`)
- `--device`: Device to use (`cuda` or `cpu`, default: `cuda`)
- `--max-new-tokens`: Maximum tokens to generate (default: 100)
- `--temperature`: Sampling temperature (default: 0.7)
- `--top-p`: Nucleus sampling parameter (default: 0.9)

## Model Structure

The saved model in `./pllm/` contains:

```
pllm/
├── config.json              # Model configuration
├── model.safetensors        # Model weights
├── llm/                     # LLM tokenizer and config
│   ├── config.json
│   ├── tokenizer.json
│   └── ...
└── README.md
```

## Example Usage in Your Code

```python
import torch
from transformers import AutoTokenizer
from proteinLLM_pllm import PLLM

# Load model
model = PLLM.from_pretrained("./pllm").to("cuda")
model.eval()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("./pllm/llm")

# Prepare inputs
aa_seq = "MKTFFVAIATGAFSATA"
stru_seq = "ACDEFGHIKLMNPQRSTVWY"
prompt = "Explain the function of this protein."

inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"].to("cuda")
attention_mask = inputs["attention_mask"].to("cuda")

# Run inference
with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        aa_seq=[aa_seq],
        stru_str=[stru_seq],
    )

# Get logits
logits = outputs.logits  # Shape: (batch_size, seq_len, vocab_size)
```

## Input Format

### Amino Acid Sequences (`aa_seq`)
- Standard single-letter amino acid codes
- Example: `"MKTFFVAIATGAFSATA"`

### Structure Sequences (`stru_str`)
- 3Di structure tokens from Foldseek
- Can use AA alphabet as placeholder for testing
- Example: `"ACDEFGHIKLMNPQRSTVWY"`

### Prompts
- Natural language questions about the protein
- Examples:
  - `"Explain the likely function of this protein."`
  - `"Which domain could this protein contain?"`
  - `"What is the biological role of this protein?"`

## Batch Processing

Both scripts support batch inference:

```python
aa_sequences = ["MKTFFVAIATGAFSATA", "MGDVEKGKKIFIMKCSQCHTVEK"]
stru_sequences = ["ACDEFGHIKLMNPQRSTVWY", "ACDEFGHIKLMNP"]
prompts = ["Prompt 1", "Prompt 2"]

batch_inputs = tokenizer(prompts, return_tensors="pt", padding=True)

with torch.no_grad():
    outputs = model(
        input_ids=batch_inputs["input_ids"].to("cuda"),
        attention_mask=batch_inputs["attention_mask"].to("cuda"),
        aa_seq=aa_sequences,
        stru_str=stru_sequences,
    )
```

## Troubleshooting

### Model Not Found
```
FileNotFoundError: ./pllm not found
```
**Solution:** Run `python example.py` first to train and save the model.

### Tokenizer Not Found
```
Warning: Tokenizer not found in model directory
```
**Solution:** The script will automatically fall back to the default tokenizer. This is normal.

### CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution:** 
- Use `--device cpu` for CPU inference
- Reduce batch size
- Use a smaller model

### Import Error
```
ModuleNotFoundError: No module named 'proteinLLM_pllm'
```
**Solution:** Make sure you're running from the correct directory:
```bash
cd /mnt/efs/erran/rllm_v02/ProteinFM/model
python simple_inference.py
```

## Performance Tips

1. **Use GPU**: CUDA inference is much faster than CPU
2. **Batch Processing**: Process multiple proteins at once
3. **Half Precision**: Use `torch.float16` or `torch.bfloat16` for faster inference
4. **Frozen Encoders**: Keep encoders frozen (default) for faster inference

## Next Steps

- Integrate into your application
- Fine-tune on your specific protein dataset
- Export to ONNX for production deployment
- Add custom generation strategies

