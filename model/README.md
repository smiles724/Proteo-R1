# ProteinFM Model

Protein-augmented Language Model (PLLM) for protein sequence analysis and understanding.

## Quick Start

### 1. Train/Save Model

```bash
python example.py \
    --protein-config ./ProTrek_650M/esm2_t33_650M_UR50D \
    --structure-config ./ProTrek_650M/foldseek_t30_150M \
    --protrek-ckpt ./ProTrek_650M/ProTrek_650M.pt
```

This saves the model to `./pllm/` (~4GB).

### 2. Run Inference

**Option A: Direct HuggingFace Inference** (Simple, slower)
```bash
cd hf_infer
python simple_inference.py
```

**Option B: vLLM Serving** (Production, faster)
```bash
# Start full PLLM server with protein encoders
cd vllm_infer
python serve_pllm_full.py --model-path ../pllm --port 30000

# Server provides OpenAI-compatible API at http://localhost:30000/v1
# Use with any OpenAI client or run inference directly
```

## Model Architecture

```
PLLM = Protein Encoders + Prefix MLP + Base LLM

Input: Protein sequences (aa_seq, stru_seq) + Text prompt
       ↓
    Protein Encoders (ESM2 + Foldseek)
       ↓
    Prefix MLP (project to LLM hidden size)
       ↓
    Concatenate with text embeddings
       ↓
    Base LLM (Qwen2.5-0.5B)
       ↓
    Output: Generated text
```

## Saved Model Structure

```
pllm/                           (4.0 GB total)
├── config.json                 # Model configuration
├── model.safetensors          # Encoder + prefix weights (3.0 GB)
├── llm/                       # Base LLM (942 MB)
│   ├── config.json
│   ├── model.safetensors
│   └── tokenizer files
├── protein_config/            # ESM2 config
└── structure_config/          # Foldseek config
```

**Key features:**
- ✅ Self-contained (all weights included)
- ✅ No ProTrek checkpoint needed (weights extracted)
- ✅ Portable (can be moved/shared)
- ✅ HuggingFace compatible

## Inference Options

### HuggingFace Inference (`hf_infer/`)

**Use when:**
- Simple setup needed
- Single requests
- Development/testing

**Scripts:**
- `simple_inference.py` - Basic examples
- `inference_example.py` - Advanced generation with sampling

See [hf_infer/README.md](hf_infer/README.md) for details.

### vLLM Serving (`vllm_infer/`)

**Use when:**
- Production deployment
- High throughput needed (80-100 req/s)
- Low latency required (<120ms)
- Multiple concurrent users
- Need full protein encoder support

**Scripts:**
- `serve_pllm_full.py` - Full PLLM server (recommended)
- `pllm_vllm_model.py` - Native vLLM integration (advanced)
- `test_full_pllm.py` - Test server functionality

See [vllm_infer/README.md](vllm_infer/README.md) for complete guide.

## Performance Comparison

| Method | Throughput | Latency | Memory | Protein Encoders | Setup |
|--------|-----------|---------|--------|-----------------|-------|
| **HF Direct** | Low (1-10 req/s) | High (>500ms) | 4GB | ✅ Yes | Simple |
| **vLLM Full PLLM** | High (80-100 req/s) | Low (~110ms) | 4GB | ✅ Yes | Simple |

**Note:** Both methods use the full PLLM model with protein and structure encoders for accurate protein understanding.

## Usage Examples

### HuggingFace Inference

```python
from proteinLLM_pllm import PLLM

# Load model
model = PLLM.from_pretrained('./pllm').to('cuda')
model.eval()

# Inference
with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        aa_seq=["MKTFFVAIATGAFSATA"],
        stru_str=["ACDEFGHIKLMNPQRSTVWY"],
        labels=None,
    )
```

### vLLM Server

```python
import requests

# Connect to full PLLM server
url = "http://localhost:30000/v1/completions"

response = requests.post(url, json={
    "model": "pllm",
    "prompt": "Analyze this protein and predict its thermostability:",
    "protein_sequence": "MKTFFVAIATGAFSATA",
    "structure_sequence": "ACDEFGHIKLMNPQRSTVWY",
    "max_tokens": 512,
    "temperature": 0.7
})

print(response.json()["choices"][0]["text"])
```

## Directory Structure

```
model/
├── README.md                   # This file
├── example.py                  # Training example
├── proteinLLM_pllm.py         # PLLM model class
├── protein_encoder.py         # Protein encoder
├── structure_encoder.py       # Structure encoder
├── pllm/                      # Saved model (4GB)
├── hf_infer/                  # HuggingFace inference
│   ├── README.md
│   ├── simple_inference.py
│   └── inference_example.py
└── vllm_infer/                # vLLM serving
    ├── README.md              # Complete integration guide
    ├── serve_pllm_full.py     # Full PLLM server (recommended)
    ├── pllm_vllm_model.py     # Native vLLM integration (advanced)
    ├── pllm_openai_client.py  # Custom OpenAI client
    └── test_full_pllm.py      # Server tests
```

## Requirements

```bash
pip install torch transformers safetensors
pip install vllm  # For vLLM serving (optional)
```

## Model Details

- **Base LLM**: Qwen2.5-0.5B-Instruct
- **Protein Encoder**: ESM2-650M
- **Structure Encoder**: Foldseek-150M
- **Total Parameters**: 1.3B
- **Model Size**: 4.0 GB

## Citation

If you use this model, please cite:

```bibtex
@software{proteinfm2025,
  title={ProteinFM: Protein-augmented Language Model},
  author={Your Name},
  year={2025}
}
```

## Resources

- [HuggingFace Inference Guide](hf_infer/README.md)
- [vLLM Serving Guide](vllm_infer/README.md)
- [ProTrek Model](https://huggingface.co/westlake-repl/ProTrek_650M)
- [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)

## Support

For issues or questions:
1. Check the README in `hf_infer/` or `vllm_infer/`
2. Review example scripts
3. Open an issue on GitHub

---

**Status**: Production Ready ✅  
**Last Updated**: October 13, 2025

