# HuggingFace Inference for PLLM

Direct inference using the PLLM model with HuggingFace Transformers.

## Quick Start

```bash
cd /mnt/efs/erran/rllm_v02/ProteinFM/model/hf_infer

# Simple inference
python simple_inference.py

# Advanced generation
python inference_example.py --max-new-tokens 100
```

## Scripts

### `simple_inference.py`

Basic inference examples:
- Single protein analysis
- Batch processing
- Protein encoding

**Usage:**
```bash
python simple_inference.py
```

**Output:**
```
✅ Forward pass successful!
   Output shape: torch.Size([1, 31, 151936])
   Next token prediction: ' The'
```

### `inference_example.py`

Advanced generation with sampling:
- Autoregressive generation
- Temperature and top-p sampling
- Custom generation parameters

**Usage:**
```bash
python inference_example.py \
    --max-new-tokens 150 \
    --temperature 0.8 \
    --top-p 0.95
```

## Code Examples

### Basic Inference

```python
from proteinLLM_pllm import PLLM
import torch

# Load model
model = PLLM.from_pretrained('../pllm').to('cuda')
model.eval()

# Prepare inputs
aa_seq = "MKTFFVAIATGAFSATA"
stru_seq = "ACDEFGHIKLMNPQRSTVWY"
prompt = "What is the function of this protein?"

inputs = model.tokenizer(prompt, return_tensors="pt")

# Run inference
with torch.no_grad():
    outputs = model(
        input_ids=inputs["input_ids"].to('cuda'),
        attention_mask=inputs["attention_mask"].to('cuda'),
        aa_seq=[aa_seq],
        stru_str=[stru_seq],
        labels=None,
    )

# Get logits
logits = outputs.logits  # Shape: (batch, seq_len, vocab_size)
```

### Text Generation

```python
# Generate with sampling
generated_text = generate_text(
    model=model,
    tokenizer=model.tokenizer,
    prompt="Analyze this protein",
    aa_seq="MKTFFVAIATGAFSATA",
    stru_seq="ACDEFGHIKLMNPQRSTVWY",
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.9,
)
```

### Batch Processing

```python
# Process multiple proteins
proteins = [
    ("MKTFFVAIATGAFSATA", "ACDEFGHIKLMNPQRSTVWY"),
    ("MGDVEKGKKIFIMKCSQCHTVEK", "ACDEFGHIKLMNP"),
]

for aa_seq, stru_seq in proteins:
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        aa_seq=[aa_seq],
        stru_str=[stru_seq],
        labels=None,
    )
    # Process outputs...
```

## Input Format

### Amino Acid Sequences (`aa_seq`)
- Standard single-letter codes
- Example: `"MKTFFVAIATGAFSATA"`

### Structure Sequences (`stru_seq`)
- 3Di tokens from Foldseek
- Can use AA alphabet as placeholder
- Example: `"ACDEFGHIKLMNPQRSTVWY"`

### Text Prompts
- Natural language questions
- Examples:
  - "What is the function of this protein?"
  - "Analyze the domain structure"
  - "Predict subcellular localization"

## Generation Parameters

```python
generate_text(
    model=model,
    tokenizer=tokenizer,
    prompt="Your prompt",
    aa_seq="...",
    stru_seq="...",
    max_new_tokens=100,     # Maximum tokens to generate
    temperature=0.7,        # Sampling temperature (0.0 = greedy)
    top_p=0.9,             # Nucleus sampling
    top_k=50,              # Top-k sampling
    repetition_penalty=1.0, # Penalty for repetition
)
```

## Performance

| Metric | Value |
|--------|-------|
| **Load Time** | ~20 seconds |
| **Inference** | ~500ms per request |
| **Memory** | ~8GB GPU (bfloat16) |
| **Throughput** | 1-10 requests/sec |

## When to Use

✅ **Good for:**
- Development and testing
- Single requests
- Simple setup
- Full control over generation

❌ **Not ideal for:**
- Production deployment
- High throughput (>10 req/s)
- Multiple concurrent users
- Low latency requirements (<100ms)

→ For production, use [vLLM serving](../vllm_infer/README.md)

## Troubleshooting

### CUDA Out of Memory

**Solution:** Use CPU or reduce batch size
```python
model = PLLM.from_pretrained('../pllm').to('cpu')
```

### Slow Inference

**Solutions:**
1. Use GPU instead of CPU
2. Use bfloat16 dtype
3. Reduce max_new_tokens
4. Consider vLLM for production

### Import Errors

**Solution:** Make sure you're in the correct directory
```bash
cd /mnt/efs/erran/rllm_v02/ProteinFM/model/hf_infer
python simple_inference.py
```

## Model Architecture

The PLLM model consists of:
1. **Protein Encoder** (ESM2-650M) - Encodes amino acid sequences
2. **Structure Encoder** (Foldseek-150M) - Encodes 3Di structure tokens
3. **Prefix MLP** - Projects encodings to LLM hidden size
4. **Base LLM** (Qwen2.5-0.5B) - Generates text

Total: ~1.3B parameters, 4GB on disk

## Next Steps

1. ✅ Run `simple_inference.py` to test
2. ✅ Try `inference_example.py` for generation
3. 📊 Benchmark on your use case
4. 🚀 Move to vLLM for production

---

**For production deployment, see:** [vLLM Serving Guide](../vllm_infer/README.md)

