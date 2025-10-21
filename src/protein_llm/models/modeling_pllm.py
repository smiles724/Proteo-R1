import argparse
import os

from os.path import dirname
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, GenerationMixin, AutoConfig

import protein_llm
from protein_llm.data.data_collator import ProteinLLMChainDataCollator
from protein_llm.data.dataset import ProteinLLMChainDataset
from protein_llm.models.configuration_pllm import PLLMConfig

import protein_llm.models.protein_encoder as protein_encoder_mod
import protein_llm.models.structure_encoder as structure_encoder_mod

try:
    import flash_attn
except (ImportError, ModuleNotFoundError):
    flash_attn = None
    print("[WARN] flash_attn is not installed, flash_attn will not work")


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
    _no_split_modules = ["EsmForMaskedLM"]
    _supports_flash_attn_2 = True

    def __init__(self, config: PLLMConfig):
        super().__init__(config)

        attn_impl = "flash_attention_2" if flash_attn is not None else "sdpa"
        if attn_impl == "sdpa":
            print("[WARN] flash_attention_2 is not activated since flash_attn is not supported!")

        if config.load_pretrained:
            self.llm = AutoModelForCausalLM.from_pretrained(
                config.base_model_name_or_path,
                torch_dtype="bfloat16" if attn_impl == "flash_attention_2" else "auto",
                low_cpu_mem_usage=True,
                attn_implementation=attn_impl,
            )
        else:
            llm_config = AutoConfig.from_pretrained(
                config.base_model_name_or_path,
                torch_dtype="bfloat16" if attn_impl == "flash_attention_2" else None,
                attn_implementation=attn_impl
            )
            self.llm = AutoModelForCausalLM.from_config(llm_config)

        # set to False for from_pretrained() after training
        self.config.load_pretrained = False
        self.config.hidden_size = self.llm.config.hidden_size # for deepspeed

        if hasattr(self.llm.config, "use_cache"):
            self.llm.config.use_cache = False
        self.llm.gradient_checkpointing_enable()

        self.hidden_size = self.llm.config.hidden_size
        self.prefix_len = 1 if config.single_token_prefix else config.prefix_len

        # Encoders (arch from configs; weights from ProTrek slots below)
        self.protein_encoder = protein_encoder_mod.ProteinEncoder(
            config.protein_config, out_dim=1024, load_pretrained=False, gradient_checkpointing=True
        )
        self.structure_encoder = structure_encoder_mod.StructureEncoder(
            config.structure_config, out_dim=1024, load_pretrained=False, gradient_checkpointing=True
        )

        # Per-token projector: 1024 -> hidden size (dtype aligned to LLM)
        model_dtype = next(self.llm.parameters()).dtype
        self.prefix_mlp = PrefixProjector(
            in_dim=1024, mid_dim=config.proj_hid, out_hidden=self.hidden_size,
            dropout=config.dropout, dtype=model_dtype,
        )

        # Freeze encoders if requested
        if not config.train_encoders:
            for p in self.protein_encoder.parameters():
                p.requires_grad = False
            for p in self.structure_encoder.parameters():
                p.requires_grad = False

        # Sync generation config from base LLM
        self._sync_generation_config()

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
                print(f"[ProteinEncoder] loaded from slot {self.config.prot_slot} | missing={mp} unexpected={up}")
            else:
                print(f"[ProteinEncoder] WARNING: slot {self.config.prot_slot} not found; skipping ckpt load.")

            if self.config.stru_slot in slots:
                ms, us = self.structure_encoder.load_state_dict(drop_extras(slots[self.config.stru_slot]), strict=False)
                print(f"[StructureEncoder] loaded from slot {self.config.stru_slot} | missing={ms} unexpected={us}")
            else:
                print(f"[StructureEncoder] WARNING: slot {self.config.stru_slot} not found; skipping ckpt load.")
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

    def get_decoder(self):
        return self.llm.model

    def set_decoder(self, decoder):
        self.llm.model = decoder

    @property
    def device(self):
        return next(self.llm.parameters()).device

    def freeze_params(self, freeze_choice: str):
        raise NotImplementedError("")

    def encode_protein_batch(self, aa_list, stru_list):
        """
        Return token-level embeddings and mask:
          prot_tok:  (B, L_total, 1024)
          prot_mask: (B, L_total) boolean
        """
        B = len(aa_list)
        device = next(self.llm.parameters()).device
        dtype = next(self.llm.parameters()).dtype

        idx_seq = [i for i, a in enumerate(aa_list) if a is not None and len(a) > 0]
        idx_stru = [i for i, s in enumerate(stru_list) if s is not None and len(s) > 0]

        # sequences
        if len(idx_seq) > 0:
            seqs = [aa_list[i] for i in idx_seq]
            emb_s, m_s, _ = self.protein_encoder(seqs, get_mask_logits=False, device=device)  # (N, Ls, 1024)
            Ls = emb_s.size(1)
        else:
            emb_s = m_s = None
            Ls = 0
        seq_tok = torch.zeros(B, Ls, 1024, device=device, dtype=dtype)
        seq_mask = torch.zeros(B, Ls, device=device, dtype=torch.bool)
        if len(idx_seq) > 0:
            seq_tok[idx_seq, :Ls, :] = emb_s.to(device=device, dtype=dtype)
            seq_mask[idx_seq, :Ls] = m_s.to(device=device)

        # structures
        if len(idx_stru) > 0:
            strs = [stru_list[i] for i in idx_stru]
            emb_t, m_t, _ = self.structure_encoder.get_repr(strs, batch_size=max(1, len(strs)),
                                                            verbose=False)  # (N, Lt, 1024)
            Lt = emb_t.size(1)
        else:
            emb_t = m_t = None
            Lt = 0
        struct_tok = torch.zeros(B, Lt, 1024, device=device, dtype=dtype)
        struct_mask = torch.zeros(B, Lt, device=device, dtype=torch.bool)
        if len(idx_stru) > 0:
            struct_tok[idx_stru, :Lt, :] = emb_t.to(device=device, dtype=dtype)
            struct_mask[idx_stru, :Lt] = m_t.to(device=device)

        # concatenate along sequence axis
        # prot_tok = torch.cat([seq_tok, struct_tok], dim=1)  # (B, L', 1024)
        # prot_mask = torch.cat([seq_mask, struct_mask], dim=1)  # (B, L')
        # return prot_tok, prot_mask

        return seq_tok, seq_mask, struct_tok, struct_mask

    def build_prefix(self, protein_tokens: torch.Tensor) -> torch.Tensor:
        # Per-token MLP to LLM hidden size
        return self.prefix_mlp(protein_tokens)  # (B, L', H)

    def forward(
            self,
            input_ids: torch.Tensor = None,
            attention_mask: torch.Tensor = None,
            labels: torch.Tensor = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            past_key_values: Optional[Tuple] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            aa_seq: List[Optional[str]] = None,
            stru_str: List[Optional[str]] = None,
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
            if aa_seq is not None and stru_str is not None:
                seq_tok, seq_mask, struct_tok, struct_mask = self.encode_protein_batch(aa_seq, stru_str)
                # N is the summation of all chains in B, i.e., N >= B
                # seq_tok [N, L, C]
                # seq_mask [N, L]
                # struct_tok [N, L', C]
                # struct_mask [N, L']

                B = input_ids.size(0)

                # Get position masks for protein and structure tokens
                seq_token_mask = (input_ids == self.config.seq_token_id)  # [B, T]
                structure_token_mask = (input_ids == self.config.struct_token_id)  # [B, T]

                # Check if token counts match
                seq_token_count = seq_token_mask.sum()
                structure_token_count = structure_token_mask.sum()

                if seq_token_count != seq_tok.size(0):
                    raise ValueError(
                        f"seq_token_id count mismatch. Expected {seq_tok.size(0)}, got {seq_token_count}"
                    )
                if structure_token_count != struct_tok.size(0):
                    raise ValueError(
                        f"struct_token_id count mismatch. Expected {struct_tok.size(0)}, got {structure_token_count}"
                    )

                # Project seq_tok and struct_tok through prefix_mlp
                seq_prefix = self.build_prefix(seq_tok)  # [N, L, H]
                struct_prefix = self.build_prefix(struct_tok)  # [N, L', H]

                seq_counter = 0
                struct_counter = 0
                batch_to_chains = []  # [(seq_indices, struct_indices), ...]

                # Step 1: Build batch to chain mapping
                for b_idx in range(B):
                    seq_count = seq_token_mask[b_idx].sum().item()
                    struct_count = structure_token_mask[b_idx].sum().item()

                    seq_indices = list(range(seq_counter, seq_counter + seq_count))
                    struct_indices = list(range(struct_counter, struct_counter + struct_count))
                    batch_to_chains.append((seq_indices, struct_indices))

                    seq_counter += seq_count
                    struct_counter += struct_count

                # Step 2: Calculate total valid length to insert for each sample
                total_insert_lens = []
                for b_idx in range(B):
                    seq_indices, struct_indices = batch_to_chains[b_idx]

                    # Calculate sum of valid lengths for all seqs
                    seq_valid_len = seq_mask[seq_indices].sum().item() if seq_indices else 0
                    # Calculate sum of valid lengths for all structs
                    struct_valid_len = struct_mask[struct_indices].sum().item() if struct_indices else 0
                    total_insert_lens.append(seq_valid_len + struct_valid_len)

                # Step 3: Calculate text valid lengths and simulate max length after insertion
                text_valid_lens = attention_mask.sum(dim=1).tolist() if attention_mask is not None \
                                  else [input_ids.size(1)] * B
                new_lens = []

                for b_idx in range(B):
                    # new_length = original_text_valid_length - num_special_tokens_replaced + inserted_valid_length
                    num_special_tokens = seq_token_mask[b_idx].sum().item() + structure_token_mask[b_idx].sum().item()
                    new_len = text_valid_lens[b_idx] - num_special_tokens + total_insert_lens[b_idx]
                    new_lens.append(new_len)

                max_len = max(new_lens)

                # Step 4: Create canvas
                pad_token_id = self.config.pad_token_id
                if pad_token_id is None:
                    pad_token_id = input_ids[~attention_mask.bool()][0]
                dummy_pad_ids = torch.full((1, 1), fill_value=pad_token_id, dtype=input_ids.dtype, device=input_ids.device)
                inputs_embeds = self.get_input_embeddings()(dummy_pad_ids).expand(B, max_len, -1).clone()
                new_attention_mask = torch.zeros(B, max_len, dtype=torch.long, device=text_embeds.device)
                if labels is not None:
                    new_labels = torch.full((B, max_len), -100, dtype=labels.dtype, device=labels.device)

                # Step 5: Build global bool masks and source index mapping
                text_fill_mask = torch.zeros(B, max_len, dtype=torch.bool, device=text_embeds.device)
                seq_fill_mask = torch.zeros(B, max_len, dtype=torch.bool, device=text_embeds.device)
                struct_fill_mask = torch.zeros(B, max_len, dtype=torch.bool, device=text_embeds.device)

                for b_idx in range(B):
                    seq_indices, struct_indices = batch_to_chains[b_idx]
                    seq_positions = torch.where(seq_token_mask[b_idx])[0].tolist()
                    struct_positions = torch.where(structure_token_mask[b_idx])[0].tolist()

                    # Build insertion point info
                    insertions = []
                    for pos, chain_idx in zip(seq_positions, seq_indices):
                        valid_len = seq_mask[chain_idx].sum().item()
                        if valid_len > 0:
                            insertions.append((pos, 'seq', chain_idx, valid_len))

                    for pos, chain_idx in zip(struct_positions, struct_indices):
                        valid_len = struct_mask[chain_idx].sum().item()
                        if valid_len > 0:
                            insertions.append((pos, 'struct', chain_idx, valid_len))

                    insertions.sort(key=lambda x: x[0])

                    # Set masks and source indices
                    new_pos = 0
                    src_pos = 0

                    for orig_pos, insert_type, chain_idx, valid_len in insertions:
                        # Set text part
                        text_fill_len = orig_pos - src_pos
                        text_fill_mask[b_idx, new_pos: new_pos + text_fill_len] = True
                        new_pos += text_fill_len

                        if insert_type == 'seq':
                            seq_fill_mask[b_idx, new_pos: new_pos + valid_len] = True
                        else:  # 'struct'
                            struct_fill_mask[b_idx, new_pos: new_pos + valid_len] = True
                        new_pos += valid_len

                        src_pos = orig_pos + 1

                    # Remaining text part
                    text_fill_len = attention_mask[b_idx].sum().item() - src_pos
                    text_fill_mask[b_idx, new_pos: new_pos + text_fill_len] = True
                    new_pos += text_fill_len

                    assert new_pos == new_lens[b_idx], \
                        f"Position mismatch at batch {b_idx}: new_pos={new_pos}, expected={new_lens[b_idx]}"

                # Step 6: Batch fill using bool masks
                non_protein_token_mask = torch.logical_and(~seq_token_mask, ~structure_token_mask)  # [B, T]
                valid_non_protein_token_mask = torch.logical_and(non_protein_token_mask, attention_mask.bool())  # [B, T]

                # # Debug check: verify mask size consistency
                # assert text_fill_mask.sum() == valid_non_protein_token_mask.sum(), \
                #     f"Text mask size mismatch: {text_fill_mask.sum()} vs {valid_non_protein_token_mask.sum()}"
                # assert seq_fill_mask.sum() == seq_mask.sum(), \
                #     f"Seq mask size mismatch: {seq_fill_mask.sum()} vs {seq_mask.sum()}"
                # assert struct_fill_mask.sum() == struct_mask.sum(), \
                #     f"Struct mask size mismatch: {struct_fill_mask.sum()} vs {struct_mask.sum()}"
                #
                # # Check total fill mask sum for each batch
                # for b_idx in range(B):
                #     total_fill = (text_fill_mask[b_idx].sum() + seq_fill_mask[b_idx].sum() + struct_fill_mask[b_idx].sum()).item()
                #     assert total_fill == new_lens[b_idx], \
                #         f"Batch {b_idx} fill mask sum mismatch: {total_fill} vs {new_lens[b_idx]}"

                # Fill text
                if text_fill_mask.any():
                    inputs_embeds[text_fill_mask] = text_embeds[valid_non_protein_token_mask]
                    if attention_mask is not None:
                        new_attention_mask[text_fill_mask] = attention_mask[valid_non_protein_token_mask]
                    else:
                        new_attention_mask[text_fill_mask] = 1
                    if labels is not None:
                        new_labels[text_fill_mask] = labels[valid_non_protein_token_mask]

                # Fill seq
                if seq_fill_mask.any():
                    inputs_embeds[seq_fill_mask] = seq_prefix[seq_mask]
                    new_attention_mask[seq_fill_mask] = 1

                # Fill struct
                if struct_fill_mask.any():
                    inputs_embeds[struct_fill_mask] = struct_prefix[struct_mask]
                    new_attention_mask[struct_fill_mask] = 1

                # # Debug checking: verify masks are mutually exclusive and contiguous
                # assert torch.logical_and(text_fill_mask, seq_fill_mask).sum() == 0
                # assert torch.logical_and(text_fill_mask, struct_fill_mask).sum() == 0
                # assert torch.logical_and(seq_fill_mask, struct_fill_mask).sum() == 0
                #
                # for b_idx in range(B):
                #     assert new_attention_mask[b_idx].sum() == new_lens[b_idx]
                #
                #     # Check attention_mask contiguity: 0s and 1s should each be contiguous
                #     attn_mask_seq = new_attention_mask[b_idx]
                #     if len(attn_mask_seq) > 1:
                #         # Find all transition points
                #         transitions = (attn_mask_seq[1:] - attn_mask_seq[:-1]).nonzero(as_tuple=True)[0]
                #         # Contiguous 0s and 1s means at most one transition (0->1 or 1->0)
                #         assert len(transitions) <= 1, \
                #             f"attention_mask is not contiguous at batch {b_idx}, found {len(transitions)} transitions"
                #
                #     # Check labels contiguity: -100 and non-(-100) should each be contiguous
                #     if labels is not None:
                #         labels_seq = new_labels[b_idx]
                #         if len(labels_seq) > 1:
                #             # Check if it's -100
                #             is_ignore = (labels_seq == -100)
                #             # Find all transition points
                #             transitions = (is_ignore[1:].long() - is_ignore[:-1].long()).nonzero(as_tuple=True)[0]
                #             assert len(transitions) <= 2, \
                #                 f"labels is not contiguous at batch {b_idx}, found {len(transitions)} transitions"
                #
                #         valid_labels_seq = labels_seq[attn_mask_seq.bool()]
                #         if len(valid_labels_seq) > 1:
                #             # Check if it's -100
                #             is_ignore = (valid_labels_seq == -100)
                #             # Find all transition points
                #             transitions = (is_ignore[1:].long() - is_ignore[:-1].long()).nonzero(as_tuple=True)[0]
                #             # Contiguous -100 and non-(-100) means at most one transition (ignore->valid or valid->ignore)
                #             assert len(transitions) <= 1, \
                #                 f"labels is not contiguous at batch {b_idx}, found {len(transitions)} transitions"

                # Update
                attention_mask = new_attention_mask
                if labels is not None:
                    labels = new_labels

            elif aa_seq is not None or stru_str is not None:
                raise RuntimeError("aa_seq and stru_str must be given at the same time!")

        if inputs_embeds is None and text_embeds is not None:
            inputs_embeds = text_embeds

        # Qwen adds position ids internally; no manual position_ids needed
        out = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return out

    def prepare_inputs_for_generation(
            self,
            input_ids: torch.LongTensor,
            past_key_values: Optional[Tuple] = None,
            attention_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            **kwargs
    ):
        if input_ids.size(0) > 1:
            raise NotImplementedError("Batch-level generate with left-padding has not been implemented yet.")

        aa_seq = kwargs.pop("aa_seq", None)
        stru_str = kwargs.pop("stru_str", None)

        model_inputs = super().prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

        model_inputs.update(
            aa_seq=aa_seq,
            stru_str=stru_str,
        )

        return model_inputs


def test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    args = parser.parse_args()

    print(args.llm_name)

    pretrained_root = f"{dirname(protein_llm.__file__)}/../../pretrained"
    data_root = f"{dirname(protein_llm.__file__)}/../../data"

    tokenizer = AutoTokenizer.from_pretrained(args.llm_name, use_fast=True)

    dataset = ProteinLLMChainDataset(
        data_path=f"{data_root}/pdb_sft_debug100_1021.jsonl",
        tokenizer=tokenizer,
    )
    data_collator = ProteinLLMChainDataCollator(tokenizer=tokenizer)

    seq_token_id = tokenizer("<aa_seq>", add_special_tokens=False).input_ids
    assert len(seq_token_id) == 1
    seq_token_id = seq_token_id[-1]

    struct_token_id = tokenizer("<3d_struct>", add_special_tokens=False).input_ids
    assert len(struct_token_id) == 1
    struct_token_id = struct_token_id[-1]

    if os.path.isdir(args.llm_name):
        pllm = PLLM.from_pretrained(args.llm_name, load_pretrained=False)
    else:
        pllm_config = PLLMConfig(
            base_model_name_or_path=args.llm_name,
            protein_config=f"{pretrained_root}/ProTrek_650M/esm2_t33_650M_UR50D",
            structure_config=f"{pretrained_root}/ProTrek_650M/foldseek_t30_150M",
            protrek_ckpt=f"{pretrained_root}/ProTrek_650M/ProTrek_650M.pt",
            prot_slot=1,
            stru_slot=3,
            train_encoders=False,
            proj_hid=1024,
            dropout=0.10,
            seq_token_id=seq_token_id,
            struct_token_id=struct_token_id,
        )
        pllm = PLLM(pllm_config)
        pllm.load_protrek_weights()
        pllm.llm.resize_token_embeddings(len(tokenizer))  # IMPORTANT after adding tokens

    pllm = pllm.to(device="cuda", dtype=torch.bfloat16)

    # temp_root = f"{dirname(protein_llm.__file__)}/../../temp"
    # pllm.config.load_pretrained = False
    # pllm.save_pretrained(f"{temp_root}/pllm")
    # pllm = PLLM.from_pretrained(f"{temp_root}/pllm")

    batch = data_collator([dataset[i] for i in range(3)])
    batch = {k: v.to(device=pllm.device, dtype=pllm.dtype if "float" in str(v.dtype) else v.dtype) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    train_outputs = pllm(**batch)
    print(train_outputs.loss)

    dataset.inference_mode = True
    batch = data_collator([dataset[i] for i in range(1)])
    batch = {k: v.to(device=pllm.device, dtype=pllm.dtype if "float" in str(v.dtype) else v.dtype) if isinstance(v, torch.Tensor) else v
             for k, v in batch.items()}
    generated_ids = pllm.generate(**batch, max_new_tokens=1024)

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(batch["input_ids"], generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    print(response)


if __name__ == '__main__':
    test()
