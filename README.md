# ProteinFM RLLM Training Guide

This guide provides step-by-step instructions to set up the environment, prepare the data, and run the training for the ProteinFM model.

## 1. Prepare a Test SFT Model

First, we need to prepare the Supervised Fine-Tuning (SFT) model. This involves downloading a pre-trained protein model and then using it to generate the final model configuration for our training.

1.  **Navigate to the model directory.**
    ```bash
    cd rllm/model
    ```

2.  **Download the `ProTrek_650M` Model.**
    This model contains the protein sequence and structure encoders. You will need `git` and `git-lfs`.
    ```bash
    git lfs install
    git clone https://huggingface.co/westlake-repl/ProTrek_650M
    ```

3.  **Generate the `pllm` Model Directory.**
    Copy the example script from the `ProteinFM` directory and run it. This script wraps the downloaded encoders with a base language model and saves the final configuration required for training into a new `pllm` directory.
    ```bash
    cp ../../ProteinFM/model/example.py .
    python3 example.py
    ```

4.  **Return to the project root.**
    ```bash
    cd ../..
    ```
    You should now have a `rllm/model/pllm` directory, which will be used as the model path for training.

## 2. Patch vLLM for Custom Model Support

To ensure the custom ProteinLLM model is correctly registered within the vLLM engine, you need to apply a patch to the vLLM source code.

Modify the file `/usr/local/lib/python3.10/dist-packages/vllm/v1/engine/core.py`.

Find the `run_engine_core` function and add the following code block at the beginning of the function, right after `maybe_register_config_serialize_by_value()`:

```python
    # ... inside run_engine_core function ...
    
    # Ensure we can serialize transformer config after spawning
    maybe_register_config_serialize_by_value()
    
    # MODIFIED: Import custom model libraries based on config.
    # This ensures that custom models, like PLLM, are registered in the
    # EngineCore process before the model loader tries to find them.
    try:
        vllm_config = kwargs.get("vllm_config")
        # The user's framework sets external_lib for HF models. We need to
        # import the corresponding vLLM-specific module.
        if vllm_config and "ProteinFM" in vllm_config.model_config.model:
             logger.info("Importing ProteinFM.model.vllm_pllm for custom model registration.")
             import ProteinFM.model.vllm_pllm
             logger.info("Successfully imported custom model library.")
    except Exception as e:
        logger.warning(f"Failed to import custom model library: {e}")

    # ... rest of the function continues here ...
```
*Note: The code snippet above uses a `logger` object. Ensure it is available in the scope or import it if necessary.*


## 3. Prepare Protein Data

Next, prepare the dataset for training.

1.  Navigate to the data preparation directory.
    ```bash
    cd rllm/examples/deepprotein/
    ```

2.  Run the data preparation script.
    ```bash
    python3 prepare_protein_data.py
    ```

3.  Return to the project root.
    ```bash
    cd ../../..
    ```

## 4. Run Training

Once the model and data are prepared, you can start the training process. The script is designed to be run from different locations.

**From the project root:**
```bash
bash rllm/examples/deepprotein/train_deepprotein_8k.sh
```

**From a subdirectory (e.g., `rllm/`):**
```bash
cd rllm
bash examples/deepprotein/train_deepprotein_8k.sh
```
