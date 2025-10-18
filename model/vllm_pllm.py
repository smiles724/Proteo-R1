"""
vLLM integration for PLLM (Protein Language Learning Model).

This module implements vLLM's multimodal interface to support PLLM models
with protein and structure encoders in vLLM's inference engine.
"""

from collections.abc import Iterable, Mapping, Sequence
from typing import Optional

import torch
import torch.nn as nn

try:
    import copy
    import os

    from safetensors import safe_open
    from transformers import AutoConfig
    from vllm.config import VllmConfig
    from vllm.model_executor.models.interfaces import SupportsMultiModal

    # We will instantiate the language model directly to avoid naming conflicts.
    from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
    from vllm.model_executor.models.utils import merge_multimodal_embeddings
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargs
    from vllm.multimodal.parse import MultiModalDataItems
    from vllm.multimodal.processing import (
        BaseMultiModalProcessor,
        BaseProcessingInfo,
        PromptReplacement,
        PromptUpdate,
    )
    from vllm.multimodal.profiling import BaseDummyInputsBuilder
    from vllm.sequence import IntermediateTensors

    VLLM_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  vLLM not available due to import error: {e}")
    VLLM_AVAILABLE = False
    SupportsMultiModal = object


if VLLM_AVAILABLE:

    class PLLMProcessingInfo(BaseProcessingInfo):
        """Processing information for PLLM multimodal inputs."""

        def get_supported_mm_limits(self) -> Mapping[str, int | None]:
            return {"protein_sequence": 1, "structure_data": 1}

        def get_mm_max_tokens_per_item(self, seq_len: int) -> Mapping[str, int]:
            hf_cfg = self.get_hf_config()
            prefix_len = getattr(hf_cfg, "prefix_len", 4)
            return {"protein_sequence": prefix_len, "structure_data": prefix_len}

    class PLLMDummyInputsBuilder(BaseDummyInputsBuilder[PLLMProcessingInfo]):
        """Builder for dummy inputs for PLLM memory profiling."""

        def get_dummy_processor_inputs(self, seq_len: int, mm_counts: Mapping[str, int]) -> object:
            from vllm.multimodal.profiling import ProcessorInputs

            # Return only a small text prompt; skip multimodal inputs so vLLM's generic parser does not complain.
            return ProcessorInputs(prompt_text="Hello", mm_data={})

    class PLLMMultiModalProcessor(BaseMultiModalProcessor[PLLMProcessingInfo]):
        """Processor for PLLM multimodal inputs."""

        def _get_mm_fields_config(self, hf_inputs: dict, hf_processor_mm_kwargs: Mapping[str, object]) -> Mapping[str, MultiModalFieldConfig]:
            return {
                "protein_sequence": MultiModalFieldConfig.batched("protein_sequence"),
                "structure_data": MultiModalFieldConfig.batched("structure_data"),
            }

        def _get_prompt_updates(self, mm_items: MultiModalDataItems, hf_processor_mm_kwargs: Mapping[str, object], out_mm_kwargs: MultiModalKwargs) -> Sequence[PromptUpdate]:
            hf_cfg = self.info.get_hf_config()
            tokenizer = self.info.get_tokenizer()
            pad_id: int = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            prefix_len = int(getattr(hf_cfg, "prefix_len", 4))

            def get_replacement(_item_idx: int):
                return [pad_id] * prefix_len

            updates = []
            if mm_items.get_items("protein_sequence", list) is not None:
                updates.append(PromptReplacement(modality="protein_sequence", target=[pad_id], replacement=get_replacement))
            if mm_items.get_items("structure_data", list) is not None:
                updates.append(PromptReplacement(modality="structure_data", target=[pad_id], replacement=get_replacement))
            return updates

        # Bypass HF Processor: directly tokenize prompt and return minimal MultiModalInputs
        def apply(self, prompt, mm_data, hf_processor_mm_kwargs, return_mm_hashes: bool = False):
            from vllm.multimodal.processing import MultiModalInputs

            tokenizer = self.info.get_tokenizer()
            if isinstance(prompt, str):
                prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
            elif isinstance(prompt, list):
                prompt_ids = prompt
                # Best-effort human-readable prompt for logs
                try:
                    prompt = tokenizer.decode(prompt_ids, skip_special_tokens=False)
                except Exception:
                    prompt = ""
            else:
                prompt = str(prompt)
                prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)

            return MultiModalInputs(
                type="multimodal",
                prompt=prompt,
                prompt_token_ids=prompt_ids,
                mm_kwargs=mm_data,
                mm_hashes=None,
                mm_placeholders={},
            )

    @MULTIMODAL_REGISTRY.register_processor(PLLMMultiModalProcessor, info=PLLMProcessingInfo, dummy_inputs=PLLMDummyInputsBuilder)
    class VLLMPLLMForCausalLM(nn.Module, SupportsMultiModal):
        """vLLM multimodal model that composes Qwen2 LLM with PLLM encoders."""

        @staticmethod
        def is_backend_compatible() -> bool:
            # Indicate to vLLM that this model is compatible with the current backend
            return True

        def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
            super().__init__()
            self.vllm_config = vllm_config
            self.hf_config = vllm_config.model_config.hf_config

            # Create a dedicated vLLM config for the inner language model.
            base_model_path = self.hf_config.base_model_name_or_path
            llm_vllm_config = copy.deepcopy(vllm_config)
            llm_vllm_config.model_config.hf_config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=vllm_config.model_config.trust_remote_code)

            # CRITICAL FIX: Instantiate the language model directly, bypassing the problematic helper.
            # We also add it to a module dict to ensure PyTorch handles it correctly.
            self.sub_modules = nn.ModuleDict({"language_model": Qwen2ForCausalLM(vllm_config=llm_vllm_config)})
            self.language_model = self.sub_modules["language_model"]

            self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors

            from ProteinFM.model.protein_encoder import ProteinEncoder
            from ProteinFM.model.proteinLLM_pllm import PrefixProjector
            from ProteinFM.model.structure_encoder import StructureEncoder

            proj_hid = int(getattr(self.hf_config, "proj_hid", 1024))
            hidden_size = int(getattr(self.hf_config, "hidden_size", 896))

            base_model_dir = vllm_config.model_config.model
            protein_cfg = getattr(self.hf_config, "protein_config", None)
            structure_cfg = getattr(self.hf_config, "structure_config", None)

            if protein_cfg:
                protein_cfg = os.path.join(base_model_dir, protein_cfg)
            if structure_cfg:
                structure_cfg = os.path.join(base_model_dir, structure_cfg)

            self.protein_encoder = ProteinEncoder(protein_cfg, out_dim=1024, load_pretrained=False) if protein_cfg is not None else None
            self.structure_encoder = StructureEncoder(structure_cfg, out_dim=1024, load_pretrained=False) if structure_cfg is not None else None

            self.prefix_mlp = PrefixProjector(
                in_dim=1024,
                mid_dim=proj_hid,
                out_hidden=hidden_size,
                dropout=0.1,
            )

            try:
                tok = vllm_config.model_config.get_tokenizer()
                self._placeholder_token_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
            except Exception:
                self._placeholder_token_id = int(getattr(self.hf_config, "pad_token_id", 0))

        def get_language_model(self) -> nn.Module:
            return self.language_model

        @torch.no_grad()
        def _encode_seqlevel(self, reps: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            valid = mask.float()
            denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
            pooled = (reps * valid.unsqueeze(-1)).sum(dim=1) / denom
            return pooled

        @torch.no_grad()
        def get_multimodal_embeddings(self, **kwargs: object) -> tuple[torch.Tensor, ...] | None:
            prefix_len = int(getattr(self.hf_config, "prefix_len", 4))
            device = next(self.language_model.parameters()).device

            if "protein_sequence" in kwargs and self.protein_encoder is not None:
                seqs: list[str] = kwargs["protein_sequence"]
                reps, mask, _ = self.protein_encoder(seqs, device=device)
                pooled = self._encode_seqlevel(reps, mask)
                expanded = pooled.unsqueeze(1).expand(-1, prefix_len, -1)
                embeds = self.prefix_mlp(expanded)
                return tuple(embeds.split(1, dim=0)[i].squeeze(0) for i in range(embeds.shape[0]))

            if "structure_data" in kwargs and self.structure_encoder is not None:
                seqs: list[str] = kwargs["structure_data"]
                reps, mask, _ = self.structure_encoder(seqs, device=device)
                pooled = self._encode_seqlevel(reps, mask)
                expanded = pooled.unsqueeze(1).expand(-1, prefix_len, -1)
                embeds = self.prefix_mlp(expanded)
                return tuple(embeds.split(1, dim=0)[i].squeeze(0) for i in range(embeds.shape[0]))
            return None

        def get_input_embeddings(self, input_ids: torch.Tensor, multimodal_embeddings: tuple[torch.Tensor, ...] | None = None, attn_metadata: Optional["IntermediateTensors"] = None) -> torch.Tensor:
            text_embeds = self.language_model.get_input_embeddings(input_ids)
            if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
                return text_embeds
            return merge_multimodal_embeddings(input_ids, text_embeds, multimodal_embeddings, placeholder_token_id=self._placeholder_token_id)

        def forward(self, input_ids: torch.Tensor | None, positions: torch.Tensor, intermediate_tensors: IntermediateTensors | None = None, inputs_embeds: torch.Tensor | None = None) -> torch.Tensor | IntermediateTensors:
            return self.language_model(input_ids, positions, intermediate_tensors, inputs_embeds)

        def compute_logits(self, hidden_states: torch.Tensor, sampling_metadata) -> torch.Tensor | None:
            return self.language_model.compute_logits(hidden_states, sampling_metadata)

        def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
            loaded_params = set()
            custom_weights_from_ckpt = dict(weights)

            for module_name, module in [
                ("protein_encoder", self.protein_encoder),
                ("structure_encoder", self.structure_encoder),
                ("prefix_mlp", self.prefix_mlp),
            ]:
                if module is None:
                    continue
                state_dict = {}
                for name_in_ckpt in list(custom_weights_from_ckpt.keys()):
                    if name_in_ckpt.startswith(f"{module_name}."):
                        sub_name = name_in_ckpt.removeprefix(f"{module_name}.")
                        state_dict[sub_name] = custom_weights_from_ckpt.pop(name_in_ckpt)
                        loaded_params.add(name_in_ckpt)
                if state_dict:
                    module.load_state_dict(state_dict)

            base_model_dir = self.vllm_config.model_config.model
            lm_dir = os.path.join(base_model_dir, "llm")

            if os.path.isdir(lm_dir):
                lm_safetensors_files = sorted([f for f in os.listdir(lm_dir) if f.endswith(".safetensors")])

                def lm_weight_generator():
                    for filename in lm_safetensors_files:
                        with safe_open(os.path.join(lm_dir, filename), framework="pt", device="cpu") as f:
                            for key in f.keys():
                                yield key, f.get_tensor(key)

                loaded_lm_sub_params = self.language_model.load_weights(lm_weight_generator())

                for sub_param in loaded_lm_sub_params:
                    # The name in the state_dict of a submodule doesn't have the prefix, so we add it back.
                    # PyTorch adds the `sub_modules` prefix automatically.
                    loaded_params.add(f"sub_modules.language_model.{sub_param}")

            return loaded_params


def register_vllm_pllm():
    """Register PLLM architecture with vLLM's model registry."""
    if not VLLM_AVAILABLE:
        print("⚠️  vLLM not available; skipping PLLMForCausalLM registry")
        return
    try:
        from vllm.model_executor.models.registry import ModelRegistry

        ModelRegistry.register_model("PLLMForCausalLM", VLLMPLLMForCausalLM)
        print("✅ Registered VLLMPLLMForCausalLM with vLLM ModelRegistry for key 'PLLMForCausalLM'")

        print("✅ PLLM multimodal processor registered via decorator")
    except Exception as e:
        import traceback

        print(f"⚠️  Failed to register PLLMForCausalLM with vLLM: {e}")
        traceback.print_exc()


# Auto-register when module is imported
register_vllm_pllm()
