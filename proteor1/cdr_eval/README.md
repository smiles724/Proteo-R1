# ProteoR1 CDR Evaluation Utilities

This package contains the CDR evaluation helper modules shipped with ProteoR1:

| Stage | Module | Purpose |
| --- | --- | --- |
| 1 | `data_preparation.py` | Parse split JSON files and validate CIF availability. |
| 2 | `chain_extractor.py` | Extract antibody and antigen chains from CIF files. |
| 3 | `cdr_masking.py` | Identify CDR regions and create masked Protenix input JSON. |
| 4 | `batch_inference.py` | Run Protenix prediction over prepared inputs. |
| 5 | `remap_cif_chains.py` | Restore original chain IDs in Protenix output CIF files. |

## Imports

```python
from proteor1.cdr_eval import (
    EntryInfo,
    extract_chains_from_cif,
    extract_and_mask_cdr,
    load_entries_for_split,
    remap_cif_chain_ids,
    run_batch_inference_cli,
)
```

The package-level namespace also re-exports the five shipped modules:

```python
from proteor1.cdr_eval import (
    batch_inference,
    cdr_masking,
    chain_extractor,
    data_preparation,
    remap_cif_chains,
)
```

## Command Line Usage

Prepare split entries and inspect basic statistics:

```bash
python -m proteor1.cdr_eval.data_preparation path/to/split.json path/to/cif_dir
```

Extract chains for one structure:

```bash
python -m proteor1.cdr_eval.chain_extractor single \
  path/to/input.cif \
  path/to/extracted.cif \
  --heavy H \
  --light L \
  --antigen A
```

Extract chains for a split:

```bash
python -m proteor1.cdr_eval.chain_extractor batch \
  path/to/split.json \
  path/to/cif_dir \
  path/to/extracted_dir
```

Create CDR-masked Protenix inputs:

```bash
python -m proteor1.cdr_eval.cdr_masking \
  --split_json path/to/split.json \
  --cif_dir path/to/extracted_cif_dir \
  --output_dir path/to/cdr_masking
```

Run Protenix over prepared JSON files:

```python
from proteor1.cdr_eval.batch_inference import run_batch_inference_cli

run_batch_inference_cli(
    input_json_dir="path/to/json_inputs",
    output_dir="path/to/predictions",
    model_name="path/to/checkpoint",
)
```

Restore chain IDs in a predicted CIF:

```bash
python -m proteor1.cdr_eval.remap_cif_chains \
  --pred_cif path/to/prediction.cif \
  --cdr_info path/to/cdr_info.json \
  --output path/to/remapped.cif
```

## Notes

CDR region detection uses `abnumber` when CDR masking functions are executed.
Importing `proteor1.cdr_eval` does not construct abnumber `Chain` objects or
run HMMER subprocesses.
