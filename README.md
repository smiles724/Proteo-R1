
## Update qkv_export
updated clean_sft/models/modeling_pllm.py

Based on https://github.com/smiles724/ProteinFM/blob/clean_sft/src/protein_llm/models 

Added qkv_export to export qkv after forward: simple usage:

```python
layers_to_probe = [0, 1, 2]
pllm.eval()
with torch.no_grad():
    out = pllm(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=None,
        aa_seq=[aa_seq],
        stru_str=[structure],
        extract_qkv=True,
        layer_idx=layers_to_probe,
    )

exp_multi = pllm.export_qkv(split_heads=False, to_cpu=False)
layers = exp_multi["layers"]
mask = exp_multi["m"]
More detailed test see: testing_notebook/Testing_proteinLLM_qkvexport.ipynb
```


More detailed test see: testing_notebook/Testing_proteinLLM_qkvexport.ipynb
