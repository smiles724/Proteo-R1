import argparse
import os

from os.path import dirname, relpath
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, GenerationMixin, AutoConfig, \
    EsmTokenizer, Cache

import lmms_engine
from lmms_engine.models.pllm.configuration_pllm import PLLMConfig
import lmms_engine.models.pllm.protein_encoder as protein_encoder_mod
import lmms_engine.models.pllm.structure_encoder as structure_encoder_mod
from lmms_engine.models.pllm.processing_pllm import PLLMProcessor

try:
    import flash_attn
except (ImportError, ModuleNotFoundError):
    flash_attn = None


from transformers.utils import logging
logger = logging.get_logger(__name__)


class PrefixProjector(nn.Module):
    """
    Simple per-token MLP: (B, L, D) -> (B, L, H)
    """

    def __init__(self, in_dim: int, mid_dim: int, out_hidden: int, dropout: float = 0.1, dtype=None, device=None):
        super().__init__()
        self.out_hidden = out_hidden
        self.net = nn.Sequential(nn.Linear(in_dim, mid_dim, bias=True), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(mid_dim, out_hidden, bias=True), )
        if dtype is not None or device is not None:
            self.to(device=device, dtype=dtype)

    def forward(self, protein_vec: torch.Tensor) -> torch.Tensor:
        # protein_vec: [B, L, D]
        return self.net(protein_vec)


class PLLM(PreTrainedModel, GenerationMixin):
    config_class = PLLMConfig
    base_model_prefix = "pllm"
    supports_gradient_checkpointing = True
    _no_split_modules = []
    _supports_flash_attn_2 = True

    def __init__(self, config: PLLMConfig):
        super().__init__(config)

        attn_implementation = "flash_attention_2" if flash_attn is not None else "sdpa"
        if attn_implementation == "sdpa":
            logger.warning(
                f"flash_attention_2 is not activated for {self.__class__.__name__} since flash_attn is not supported!"
            )

        if config.load_pretrained:
            self.llm = AutoModelForCausalLM.from_pretrained(
                config.base_model_name_or_path,
                torch_dtype=torch.bfloat16 if attn_implementation == "flash_attention_2" else None,
                low_cpu_mem_usage=True,
                attn_implementation=attn_implementation,
            )
        else:
            llm_config = AutoConfig.from_pretrained(config.base_model_name_or_path)
            # If vocab_size is specified in PLLM config, override the LLM config
            # This ensures correct vocab_size when loading from PT checkpoint
            if config.vocab_size is not None:
                llm_config.vocab_size = config.vocab_size
                logger.info(
                    f"Using vocab_size from PLLM config ({config.vocab_size}) to initialized base llm "
                    f"{config.base_model_name_or_path}"
                )
            self.llm = AutoModelForCausalLM.from_config(
                llm_config,
                torch_dtype=torch.bfloat16 if attn_implementation == "flash_attention_2" else None,
                attn_implementation=attn_implementation
            )

        # Sync vocab_size from LLM config to PLLM config (if not already set)
        if self.config.vocab_size is None and hasattr(self.llm.config, 'vocab_size'):
            self.config.vocab_size = self.llm.config.vocab_size

        if hasattr(self.llm.config, "use_cache"):
            self.llm.config.use_cache = False

        self.hidden_size = self.llm.config.hidden_size
        self.prefix_len = 1 if config.single_token_prefix else config.prefix_len # currently unused, for future case

        # Encoders (arch from configs; weights from ProTrek slots below)
        # Note: gradient_checkpointing=False here, will be enabled by Trainer if needed
        self.protein_encoder = protein_encoder_mod.ProteinEncoder(
            config.protein_config,
            out_dim=1024,
            load_pretrained=False,
            gradient_checkpointing=False,
            attn_implementation=attn_implementation,
        )
        self.structure_encoder = structure_encoder_mod.StructureEncoder(
            config.structure_config,
            out_dim=1024,
            load_pretrained=False,
            gradient_checkpointing=False,
            attn_implementation=attn_implementation,
        )

        if config.load_pretrained:
            self.load_protrek_weights()

        # Per-token projector: 1024 -> hidden size (dtype aligned to LLM)
        model_dtype = next(self.llm.parameters()).dtype

        # 🔥 Convert encoders to match LLM dtype for FSDP2 compatibility
        self.protein_encoder = self.protein_encoder.to(dtype=model_dtype)
        self.structure_encoder = self.structure_encoder.to(dtype=model_dtype)

        if self.config.joint_projector:
            self.prefix_mlp = PrefixProjector(
                in_dim=1024, mid_dim=config.proj_hid, out_hidden=self.hidden_size,
                dropout=config.dropout, dtype=model_dtype,
            )
        else:
            self.prefix_mlp = torch.nn.ModuleDict(
                dict(
                    seq=PrefixProjector(
                        in_dim=1024, mid_dim=config.proj_hid, out_hidden=self.hidden_size,
                        dropout=config.dropout, dtype=model_dtype,
                    ),
                    struct=PrefixProjector(
                        in_dim=1024, mid_dim=config.proj_hid, out_hidden=self.hidden_size,
                        dropout=config.dropout, dtype=model_dtype,
                    ),
                )
            )

        # Freeze specific part if requested
        self.freeze_params()

        # Configuration correctness insurance
        self.config.load_pretrained = False  # set to False for calling from_pretrained() by a pretrained checkpoint
        self.config.freeze_choice = "none"  # reset freeze_choice to avoid mistakenly freeze params in the next loading
        self.config.hidden_size = self.llm.config.hidden_size  # for deepspeed
        if os.path.isabs(self.config.protein_config):
            self.config.protein_config = relpath(os.path.curdir, self.config.protein_config)
        if os.path.isabs(self.config.structure_config):
            self.config.structure_config = relpath(os.path.curdir, self.config.structure_config)

        # Sync generation config from base LLM
        self._sync_generation_config()

        # Dynamically merge _no_split_modules from all sub-models
        all_no_split_modules = []

        # Merge from LLM
        if hasattr(self.llm, '_no_split_modules') and self.llm._no_split_modules:
            all_no_split_modules.extend(self.llm._no_split_modules)

        # Merge from protein encoder
        if hasattr(self.protein_encoder, 'model') and hasattr(self.protein_encoder.model, '_no_split_modules'):
            if self.protein_encoder.model._no_split_modules:
                all_no_split_modules.extend(self.protein_encoder.model._no_split_modules)

        # Merge from structure encoder
        if hasattr(self.structure_encoder, 'model') and hasattr(self.structure_encoder.model, '_no_split_modules'):
            if self.structure_encoder.model._no_split_modules:
                all_no_split_modules.extend(self.structure_encoder.model._no_split_modules)

        # Remove duplicates and update
        self._no_split_modules = list(set(all_no_split_modules))
        logger.info(f"Merged _no_split_modules: {self._no_split_modules}")

    def _tie_weights(self):
        """Tie weights for encoder modules that have tied weights internally.
        ESM models have tied weights between embeddings and lm_head.decoder.
        """
        # Tie weights for protein encoder (ESM model)
        if hasattr(self.protein_encoder, 'model') and hasattr(self.protein_encoder.model, 'lm_head'):
            self._tie_or_clone_weights(
                self.protein_encoder.model.lm_head.decoder,
                self.protein_encoder.model.esm.embeddings.word_embeddings
            )

        # Tie weights for structure encoder (ESM model)
        if hasattr(self.structure_encoder, 'model') and hasattr(self.structure_encoder.model, 'lm_head'):
            self._tie_or_clone_weights(
                self.structure_encoder.model.lm_head.decoder,
                self.structure_encoder.model.esm.embeddings.word_embeddings
            )

    def _sync_generation_config(self):
        """Sync generation config from base LLM to PLLM.
        This ensures generate() uses the correct eos_token_id and other generation parameters.
        Should be called after __init__ and after from_pretrained loads the model.
        """
        # 首先尝试加载保存的 generation_config（如果存在）
        if hasattr(self.config, '_name_or_path') and self.config._name_or_path:
            try:
                from transformers import GenerationConfig
                saved_gen_config = GenerationConfig.from_pretrained(self.config._name_or_path)
                self.generation_config = saved_gen_config
                print(f"[PLLM] Loaded generation_config from checkpoint: {self.config._name_or_path}")
                return  # 使用保存的 config，不再同步
            except Exception:
                pass  # 如果没有保存的 config，继续同步

        # If base_model_name_or_path is set, load fresh generation config from the base model
        if hasattr(self.config, 'base_model_name_or_path') and self.config.base_model_name_or_path:
            try:
                from transformers import GenerationConfig
                base_gen_config = GenerationConfig.from_pretrained(self.config.base_model_name_or_path)
                self.generation_config = base_gen_config
                print(f"[PLLM] Loaded generation_config from base model: {self.config.base_model_name_or_path}")
            except Exception as e:
                print(f"[PLLM] Warning: Could not load generation_config from base model: {e}")
                # Fall back to LLM's generation_config
                if hasattr(self.llm, 'generation_config') and self.llm.generation_config is not None:
                    self.generation_config = self.llm.generation_config
        else:
            # Fall back to LLM's generation_config
            if hasattr(self.llm, 'generation_config') and self.llm.generation_config is not None:
                self.generation_config = self.llm.generation_config

        # Also sync important config attributes
        if hasattr(self.llm.config, 'eos_token_id'):
            self.config.eos_token_id = self.llm.config.eos_token_id
        if getattr(self.config, 'eos_token_id', None) is None and hasattr(self.generation_config, 'eos_token_id'):
            self.config.eos_token_id = self.generation_config.eos_token_id

        if hasattr(self.llm.config, 'pad_token_id'):
            self.config.pad_token_id = self.llm.config.pad_token_id
        if getattr(self.config, 'pad_token_id', None) is None and hasattr(self.generation_config, 'pad_token_id'):
            self.config.pad_token_id = self.generation_config.pad_token_id

        if hasattr(self.llm.config, 'bos_token_id'):
            self.config.bos_token_id = self.llm.config.bos_token_id
        if getattr(self.config, 'bos_token_id', None) is None and hasattr(self.generation_config, 'bos_token_id'):
            self.config.bos_token_id = self.generation_config.bos_token_id

    def _post_init(self):
        """Called after from_pretrained loads the model."""
        super()._post_init()
        # Re-sync generation config after loading from checkpoint
        self._sync_generation_config()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing for all components."""
        # Enable for LLM
        if hasattr(self.llm, "gradient_checkpointing_enable"):
            self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

        # Only enable gradient checkpointing for encoders if they have trainable parameters
        # Enable for protein encoder
        if hasattr(self.protein_encoder.model, "gradient_checkpointing_enable"):
            self.protein_encoder.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

        # Enable for structure encoder
        if hasattr(self.structure_encoder.model, "gradient_checkpointing_enable"):
            self.structure_encoder.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for all components."""
        # Disable for LLM
        if hasattr(self.llm, "gradient_checkpointing_disable"):
            self.llm.gradient_checkpointing_disable()

        # Disable for protein encoder
        if hasattr(self.protein_encoder.model, "gradient_checkpointing_disable"):
            self.protein_encoder.model.gradient_checkpointing_disable()

        # Disable for structure encoder
        if hasattr(self.structure_encoder.model, "gradient_checkpointing_disable"):
            self.structure_encoder.model.gradient_checkpointing_disable()

    def load_protrek_weights(self):
        if self.config.protrek_ckpt and os.path.exists(self.config.protrek_ckpt):
            sd_raw = torch.load(self.config.protrek_ckpt, map_location="cpu")
            sd = sd_raw.get("model", sd_raw.get("state_dict", sd_raw))
            slots = {}
            for k, v in sd.items():
                head = k.split(".", 1)[0]
                if head.isdigit():
                    slots.setdefault(int(head), {})[k[len(head) + 1:]] = v

            def drop_extras(sub):
                return {k: v for k, v in sub.items() if "embeddings.position_ids" not in k}

            if self.config.prot_slot in slots:
                mp, up = self.protein_encoder.load_state_dict(drop_extras(slots[self.config.prot_slot]), strict=False)
                logger.info(f"[ProteinEncoder] loaded from ProTrek slot {self.config.prot_slot} | missing={mp} unexpected={up}")
            else:
                logger.info(f"[ProteinEncoder] WARNING: ProTrek slot {self.config.prot_slot} not found; skipping ckpt load.")

            if self.config.stru_slot in slots:
                ms, us = self.structure_encoder.load_state_dict(drop_extras(slots[self.config.stru_slot]), strict=False)
                logger.info(f"[StructureEncoder] loaded from ProTrek slot {self.config.stru_slot} | missing={ms} unexpected={us}")
            else:
                logger.info(f"[StructureEncoder] WARNING: ProTrek slot {self.config.stru_slot} not found; skipping ckpt load.")
        else:
            raise ValueError("No ProTrek checkpoint provided or path not found; encoders stay random-init.")

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.llm.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.llm.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.llm.set_output_embeddings(new_embeddings)

    def resize_token_embeddings(
            self,
            new_num_tokens: Optional[int] = None,
            pad_to_multiple_of: Optional[int] = None,
            mean_resizing: bool = True,
    ) -> nn.Embedding:
        """
        Resize input and output token embeddings matrix of the LLM.
        Automatically syncs vocab_size to PLLM config.

        Args:
            new_num_tokens: New number of tokens in the embedding matrix.
            pad_to_multiple_of: If set, will pad to a multiple of this value.

        Returns:
            torch.nn.Embedding: Pointer to the input embeddings Module of the model.
        """
        model_embeds = self.llm.resize_token_embeddings(new_num_tokens, pad_to_multiple_of, mean_resizing)

        # Sync the vocab_size from LLM config to PLLM config
        if hasattr(self.llm.config, 'vocab_size'):
            self.config.vocab_size = self.llm.config.vocab_size
            logger.info(f"Synced vocab_size to PLLM config: {self.config.vocab_size}")

        return model_embeds

    def get_decoder(self):
        return self.llm.model

    def set_decoder(self, decoder):
        self.llm.model = decoder

    @property
    def device(self):
        return next(self.llm.parameters()).device

    def train(self, mode: bool = True):
        """Override train() to manage use_cache."""
        super().train(mode)
        if hasattr(self.llm.config, "use_cache"):
            self.llm.config.use_cache = not mode  # 训练时 False，推理时 True
        return self

    def eval(self):
        """Override eval() to manage use_cache."""
        super().eval()
        if hasattr(self.llm.config, "use_cache"):
            self.llm.config.use_cache = True
        return self

    def freeze_params(self, freeze_choice: str = None):
        if freeze_choice is None:
            freeze_choice = self.config.freeze_choice

        if freeze_choice == "none":
            for name, param in self.named_parameters():
                param.requires_grad = True

        elif freeze_choice == "encoder":
            for name, param in self.named_parameters():
                module_name = name.split('.')[0]
                param.requires_grad = module_name not in ["protein_encoder", "structure_encoder"]

        elif freeze_choice == "non_projector":
            for name, param in self.named_parameters():
                module_name = name.split('.')[0]
                param.requires_grad = module_name in ["prefix_mlp"]

        else:
            raise ValueError(f"Unknown freeze_choice: {freeze_choice}")

    def build_prefix(self, protein_tokens: torch.Tensor, branch: str = None) -> torch.Tensor:
        if not self.config.joint_projector:
            if branch is None:
                raise RuntimeError("")
            if branch not in self.prefix_mlp:
                raise ValueError("")
            return self.prefix_mlp[branch](protein_tokens)

        else:
            # Per-token MLP to LLM hidden size
            return self.prefix_mlp(protein_tokens)  # (B, L', H)

    def forward(
            self,
            input_ids: torch.Tensor = None,
            attention_mask: torch.Tensor = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Tuple] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: torch.Tensor = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            protein_input_ids: torch.Tensor = None,
            protein_attention_mask: torch.Tensor = None,
            structure_input_ids: torch.Tensor = None,
            structure_attention_mask: torch.Tensor = None,
            **kwargs,
    ) -> torch.Tensor:
        if input_ids is None and inputs_embeds is None:
            raise RuntimeError("input_ids and inputs_embeds must not be None at the same time!")

        text_embeds = None
        if input_ids is not None:
            text_embeds = self.get_input_embeddings()(input_ids)

        is_first_forward = (
            past_key_values is None
            or (hasattr(past_key_values, 'get_seq_length') and past_key_values.get_seq_length() == 0)
        )

        if inputs_embeds is None and is_first_forward:
            seq_tok, seq_mask, _ = self.protein_encoder(
                input_ids=protein_input_ids,
                attention_mask=protein_attention_mask,
                get_mask_logits=False
            )  # (N, Ls, 1024)

            struct_tok, struct_mask, _ = self.structure_encoder.get_repr(
                input_ids=structure_input_ids,
                attention_mask=structure_attention_mask,
            )  # (N, Lt, 1024)

            # Project seq_tok and struct_tok through prefix_mlp
            seq_prefix = self.build_prefix(seq_tok, branch="seq")  # [N, Ls, H]
            struct_prefix = self.build_prefix(struct_tok, branch="struct")  # [N, Lt, H]

            valid_seq_prefix = seq_prefix[seq_mask]
            valid_struct_prefix = struct_prefix[struct_mask]

            valid_seq_token_count = valid_seq_prefix.size(0)
            valid_struct_token_count = valid_struct_prefix.size(0)

            # Get position masks for protein and structure tokens
            seq_token_mask = (input_ids == self.config.seq_token_id)  # [B, T]
            structure_token_mask = (input_ids == self.config.struct_token_id)  # [B, T]

            # Check if token counts match
            expected_seq_token_count = seq_token_mask.sum()
            expected_structure_token_count = structure_token_mask.sum()

            if expected_seq_token_count != valid_seq_token_count:
                raise ValueError(
                    f"seq_token_id count mismatch. Expected {expected_seq_token_count}, got {valid_seq_token_count}"
                )
            if expected_structure_token_count != valid_struct_token_count:
                raise ValueError(
                    f"struct_token_id count mismatch. Expected {expected_structure_token_count}, got {valid_struct_token_count}"
                )

            text_embeds[seq_token_mask] = valid_seq_prefix
            text_embeds[structure_token_mask] = valid_struct_prefix

        if inputs_embeds is None and text_embeds is not None:
            inputs_embeds = text_embeds

        # Qwen adds position ids internally; no manual position_ids needed
        out = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs
        )
        return out

    def prepare_inputs_for_generation(
            self,
            input_ids: torch.LongTensor,
            past_key_values: Optional[Cache] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            cache_position: Optional[torch.LongTensor] = None,
            protein_input_ids: torch.Tensor = None,
            protein_attention_mask: torch.Tensor = None,
            structure_input_ids: torch.Tensor = None,
            structure_attention_mask: torch.Tensor = None,
            **kwargs
    ):
        model_inputs = self.llm.prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs
        )

        model_inputs.update(
            protein_input_ids=protein_input_ids,
            protein_attention_mask=protein_attention_mask,
            structure_input_ids=structure_input_ids,
            structure_attention_mask=structure_attention_mask,
        )

        return model_inputs
