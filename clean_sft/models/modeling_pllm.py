import argparse
import os

from os.path import dirname
from typing import List, Optional, Tuple, Sequence, Union

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, GenerationMixin, AutoConfig

import protein_llm
from protein_llm.models.configuration_pllm import PLLMConfig

import protein_llm.models.protein_encoder as protein_encoder_mod
import protein_llm.models.structure_encoder as structure_encoder_mod
from protein_llm.utils.sft import create_sft_training_data_simple

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
        if config.load_pretrained:
            self.llm = AutoModelForCausalLM.from_pretrained(
                config.base_model_name_or_path,
                torch_dtype="auto",
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
        if hasattr(self.llm.config, 'pad_token_id'):
            self.config.pad_token_id = self.llm.config.pad_token_id
        if hasattr(self.llm.config, 'bos_token_id'):
            self.config.bos_token_id = self.llm.config.bos_token_id

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
            extract_qkv: bool = False,
            layer_idx: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        if input_ids is None and inputs_embeds is None:
            raise RuntimeError("input_ids and inputs_embeds must not be None at the same time!")

        text_embeds = None
        if input_ids is not None:
            text_embeds = self.get_input_embeddings()(input_ids)

        if inputs_embeds is None and past_key_values is None:
            if aa_seq is not None and stru_str is not None:
                # prot_tok, prot_mask = self.encode_protein_batch(aa_seq, stru_str)
                # pref = self.build_prefix(prot_tok)
                # inputs_embeds = torch.cat([pref, text_embeds], dim=1)
                # attention_mask = torch.cat([prot_mask.to(attention_mask.dtype), attention_mask], dim=1)
                #
                # if labels is not None:
                #     pad = torch.full((labels.size(0), pref.size(1)), -100, dtype=labels.dtype, device=labels.device)
                #     labels = torch.cat([pad, labels], dim=1)

                seq_tok, seq_mask, struct_tok, struct_mask = self.encode_protein_batch(aa_seq, stru_str)
                # seq_tok [B, L, C]
                # seq_mask [B, L]
                # struct_tok [B, L', C]
                # struct_mask [B, L']

                B = input_ids.size(0)

                # Get position masks for protein and structure tokens
                protein_token_mask = (input_ids == self.config.protein_token_id)  # [B, T]
                structure_token_mask = (input_ids == self.config.structure_token_id)  # [B, T]

                # Check if token counts match
                protein_token_count = protein_token_mask.sum(dim=1)  # [B]
                structure_token_count = structure_token_mask.sum(dim=1)  # [B]
                for b_idx in range(B):
                    # Check token count for each sample
                    expected_protein = 1 if aa_seq[b_idx] is not None and len(aa_seq[b_idx]) > 0 else 0
                    expected_structure = 1 if stru_str[b_idx] is not None and len(stru_str[b_idx]) > 0 else 0

                    if protein_token_count[b_idx] != expected_protein:
                        raise ValueError(
                            f"Sample {b_idx}: protein_token_id count mismatch. "
                            f"Expected {expected_protein}, got {protein_token_count[b_idx]}"
                        )
                    if structure_token_count[b_idx] != expected_structure:
                        raise ValueError(
                            f"Sample {b_idx}: structure_token_id count mismatch. "
                            f"Expected {expected_structure}, got {structure_token_count[b_idx]}"
                        )

                # Project seq_tok and struct_tok through prefix_mlp
                seq_prefix = self.build_prefix(seq_tok)  # [B, L, H]
                struct_prefix = self.build_prefix(struct_tok)  # [B, L', H]

                # Build new sequence for each sample
                new_embeds_list = []
                new_labels_list = [] if labels is not None else None
                new_attn_mask_list = []

                for b_idx in range(B):
                    curr_embeds = text_embeds[b_idx]  # [T, H]
                    curr_attn = attention_mask[b_idx] if attention_mask is not None else None  # [T]
                    curr_labels = labels[b_idx] if labels is not None else None  # [T]

                    # Collect positions and content to insert
                    insertions = []  # (position, embed_tensor, label_tensor)

                    # protein token
                    if protein_token_count[b_idx] > 0:
                        protein_pos = torch.where(protein_token_mask[b_idx])[0][0].item()
                        seq_valid_len = seq_mask[b_idx].sum().item()
                        if seq_valid_len > 0:
                            insertions.append((
                                protein_pos,
                                seq_prefix[b_idx, :seq_valid_len],  # [L, H]
                                torch.full((seq_valid_len,), -100, dtype=labels.dtype, device=labels.device) if labels is not None else None
                            ))

                    # structure token
                    if structure_token_count[b_idx] > 0:
                        structure_pos = torch.where(structure_token_mask[b_idx])[0][0].item()
                        struct_valid_len = struct_mask[b_idx].sum().item()
                        if struct_valid_len > 0:
                            insertions.append((
                                structure_pos,
                                struct_prefix[b_idx, :struct_valid_len],  # [L', H]
                                torch.full((struct_valid_len,), -100, dtype=labels.dtype, device=labels.device) if labels is not None else None
                            ))

                    # Sort by position
                    insertions.sort(key=lambda x: x[0])

                    # Perform replacement: process from back to front to avoid position shift
                    embed_parts = []
                    label_parts = [] if labels is not None else None
                    attn_parts = []

                    last_pos = 0
                    for pos, embed_insert, label_insert in insertions:
                        # Add previous part
                        if pos > last_pos:
                            embed_parts.append(curr_embeds[last_pos:pos])
                            attn_parts.append(curr_attn[last_pos:pos] if curr_attn is not None else torch.ones(pos - last_pos, device=curr_embeds.device))
                            if labels is not None:
                                label_parts.append(curr_labels[last_pos:pos])

                        # Add inserted part (replace single token)
                        embed_parts.append(embed_insert)
                        attn_parts.append(torch.ones(embed_insert.size(0), dtype=curr_attn.dtype if curr_attn is not None else torch.long, device=curr_embeds.device))
                        if labels is not None:
                            label_parts.append(label_insert)

                        last_pos = pos + 1  # Skip replaced token

                    # Add remaining part
                    if last_pos < curr_embeds.size(0):
                        embed_parts.append(curr_embeds[last_pos:])
                        attn_parts.append(curr_attn[last_pos:] if curr_attn is not None else torch.ones(curr_embeds.size(0) - last_pos, device=curr_embeds.device))
                        if labels is not None:
                            label_parts.append(curr_labels[last_pos:])

                    # Concatenate
                    new_embeds_list.append(torch.cat(embed_parts, dim=0))
                    new_attn_mask_list.append(torch.cat(attn_parts, dim=0))
                    if labels is not None:
                        new_labels_list.append(torch.cat(label_parts, dim=0))

                # Pad all samples to same length
                max_len = max(e.size(0) for e in new_embeds_list)
                inputs_embeds = torch.zeros(B, max_len, self.hidden_size, dtype=text_embeds.dtype, device=text_embeds.device)
                attention_mask = torch.zeros(B, max_len, dtype=attention_mask.dtype if attention_mask is not None else torch.long, device=text_embeds.device)
                if labels is not None:
                    new_labels = torch.full((B, max_len), -100, dtype=labels.dtype, device=labels.device)

                for b_idx in range(B):
                    curr_len = new_embeds_list[b_idx].size(0)
                    inputs_embeds[b_idx, :curr_len] = new_embeds_list[b_idx]
                    attention_mask[b_idx, :curr_len] = new_attn_mask_list[b_idx]
                    if labels is not None:
                        new_labels[b_idx, :curr_len] = new_labels_list[b_idx]

                if labels is not None:
                    labels = new_labels

            elif aa_seq is not None or stru_str is not None:
                raise RuntimeError("aa_seq and stru_str must be given at the same time!")

        if inputs_embeds is None and text_embeds is not None:
            inputs_embeds = text_embeds


        # ====== QKV capture block ======
        handles = None
        if extract_qkv:
            if layer_idx is None or len(layer_idx) == 0:
                raise ValueError("When extract_qkv=True, please provide a non-empty list for layer_idx.")

            # Prepare cache containers and validate indices
            if not (hasattr(self.llm, "model") and hasattr(self.llm.model, "layers")):
                raise RuntimeError("Unsupported LLM structure: expected self.llm.model.layers to exist.")
            layers = self.llm.model.layers
            for li in layer_idx:
                if not (0 <= li < len(layers)):
                    raise IndexError(f"layer_idx {li} out of range [0, {len(layers)-1}].")
                if not hasattr(layers[li], "self_attn"):
                    raise RuntimeError(f"Layer {li} has no 'self_attn' submodule.")

            self._last_qkv = {li: {"q": None, "k": None, "v": None} for li in layer_idx}
            self._last_attention_mask = attention_mask.detach() if attention_mask is not None else None
            self._last_qkv_layers = list(layer_idx)
            # Build kwargs-aware pre-hooks to collect hidden_states -> q_proj/k_proj/v_proj
            def make_pre_hook(li: int):
                def _pre_hook(module, args, kwargs):
                    # Qwen-style: kwargs-only call; fall back to args[0] if ever present
                    hs = kwargs.get("hidden_states", None)
                    if hs is None and isinstance(args, (tuple, list)) and len(args) > 0:
                        hs = args[0]
                    if hs is None:
                        return
                    # Compute the tensors attention will use; detach to avoid graph retention
                    q = module.q_proj(hs).detach()
                    k = module.k_proj(hs).detach()
                    v = module.v_proj(hs).detach()
                    self._last_qkv[li]["q"] = q
                    self._last_qkv[li]["k"] = k
                    self._last_qkv[li]["v"] = v
                return _pre_hook
            # Qwen2.5 attn:
            # self.self_attn(
            #     hidden_states=hidden_states,          
            #     attention_mask=causal_mask,           
            #     position_ids=...,
            #     ...
            #     )

            handles = []
            for li in layer_idx:
                h = layers[li].self_attn.register_forward_pre_hook(make_pre_hook(li), with_kwargs=True)
                handles.append(h)
        # ====== end QKV capture block end ======


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


    def export_qkv(
        self,
        *,
        split_heads: bool = False,
        detach: bool = True,
        to_cpu: bool = False,
    ) -> dict:
        """
        Export the Q/K/V captured on-the-fly during the most recent
        forward(..., extract_qkv=True, layer_idx=[...]) call.

        Returns (always the same shape, even for a single layer):
        {
            "layers": {
            <layer_idx>: {
                "q": Tensor,      # [B,S,Hq] or [B,S,num_heads,head_dim_q] if split_heads
                "k": Tensor,      # [B,S,Hk] or [B,S,num_kv_heads,head_dim_kv]
                "v": Tensor,      # [B,S,Hv] or [B,S,num_kv_heads,head_dim_kv]
                "meta": {
                "layer_idx": int,
                "split_heads": bool,
                "num_heads": int | None,
                "num_key_value_heads": int | None,
                "head_dim_q": int | None,
                "head_dim_kv": int | None,
                "seq_len": int,
                }
            },
            ...
            },
            "m": Tensor | None   # attention mask used in that forward (B,S)
        }
        """
        if not hasattr(self, "_last_qkv") or not self._last_qkv:
            raise RuntimeError("No captured Q/K/V found. Run forward(..., extract_qkv=True, layer_idx=[...]) first.")
        if not hasattr(self, "_last_qkv_layers") or not self._last_qkv_layers:
            # Fallback: infer from keys (older cache); keep deterministic order
            layer_list = sorted(self._last_qkv.keys())
        else:
            layer_list = list(self._last_qkv_layers)

        # Config for (G)QA shapes
        n_q  = getattr(self.llm.config, "num_attention_heads", None)
        n_kv = getattr(self.llm.config, "num_key_value_heads", n_q)

        def process(li: int):
            entry = self._last_qkv.get(li, None)
            if entry is None or entry["q"] is None or entry["k"] is None or entry["v"] is None:
                raise RuntimeError(
                    f"Q/K/V for layer {li} not captured. Ensure this layer was included in the last forward(..., layer_idx=[...])."
                )
            q, k, v = entry["q"], entry["k"], entry["v"]

            # Optional head-splitting with GQA awareness
            head_dim_q = head_dim_kv = None
            if split_heads:
                if n_q is None:
                    raise RuntimeError("Model config lacks 'num_attention_heads'; cannot split heads.")
                Hq = q.size(-1)
                if Hq % n_q != 0:
                    raise RuntimeError(f"Q hidden_size ({Hq}) not divisible by num_attention_heads ({n_q}).")
                head_dim_q = Hq // n_q

                nkv = n_kv if n_kv is not None else n_q
                Hk, Hv = k.size(-1), v.size(-1)
                if Hk % nkv != 0 or Hv % nkv != 0:
                    raise RuntimeError(
                        f"K/V hidden sizes ({Hk}/{Hv}) not divisible by num_key_value_heads ({nkv})."
                    )
                head_dim_kv = Hk // nkv

                def _split_q(x):  return x.view(x.size(0), x.size(1), n_q,  head_dim_q)
                def _split_kv(x): return x.view(x.size(0), x.size(1), nkv, head_dim_kv)
                q, k, v = _split_q(q), _split_kv(k), _split_kv(v)

            # Optional detach / device move
            if detach:
                q, k, v = q.detach(), k.detach(), v.detach()
            if to_cpu:
                q, k, v = q.cpu(), k.cpu(), v.cpu()

            meta = {
                "layer_idx": li,
                "split_heads": split_heads,
                "num_heads": n_q,
                "num_key_value_heads": n_kv,
                "head_dim_q": head_dim_q,
                "head_dim_kv": head_dim_kv,
                "seq_len": q.size(1),
            }
            return {"q": q, "k": k, "v": v, "meta": meta}

        # Build layers output in the same order requested in forward
        layers_out = {}
        for li in layer_list:
            layers_out[li] = process(li)

        m = getattr(self, "_last_attention_mask", None)
        if m is not None:
            m = m.detach()
            if to_cpu:
                m = m.cpu()

        return {"layers": layers_out, "m": m}




def test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    args = parser.parse_args()

    print(args.llm_name)

    pretrained_root = f"{dirname(protein_llm.__file__)}/../../pretrained"

    user_prompt = (
        "You are a professional protein biologist. "
        "Based only on the provided inputs, produce a natural, concise, and biologically accurate description of the protein. "
        "First reason step by step inside a <thinking> block using sequence-derived evidence and structural cues, "
        "then provide the final 2–4 sentence description inside an <answer> block.\n\n"
        "Inputs:\n"
        "Protein: <protein>\n"
        "Structure: <structure>"
    )

    aa_seq = "MSKGTPSRGKRQTQTHLTCRRCGRMSYHKRHKICSSCGFGRSTRMRSYGWITKRPKVATH"
    structure = "<|chain:A|> <|chain_sep|> #ddddvvvvpppddqfdqdppprdraqgpvqragpqqggpndpggdddpvvddddpdddd"

    tokenizer = AutoTokenizer.from_pretrained(args.llm_name, use_fast=True)
    tokenizer.add_tokens("<protein>")
    tokenizer.add_tokens("<structure>")
    protein_token_id = tokenizer("<protein>", add_special_tokens=False).input_ids[-1]
    structure_token_id = tokenizer("<structure>", add_special_tokens=False).input_ids[-1]

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
            protein_token_id=protein_token_id,
            structure_token_id=structure_token_id,
        )
        pllm = PLLM(pllm_config)
        pllm.load_protrek_weights()
    pllm = pllm.to("cuda")

    # temp_root = f"{dirname(protein_llm.__file__)}/../../temp"
    # pllm.config.load_pretrained = False
    # pllm.save_pretrained(f"{temp_root}/pllm")
    # pllm = PLLM.from_pretrained(f"{temp_root}/pllm")

    gt_ans = (
        "<thinking>\n"
        "1. **Signal Peptide and Localization**: The sequence starts with methionine (M), but there is no clear signal "
        "peptide sequence that would suggest secretion or targeting to specific organelles. This suggests a cytoplasmic "
        "or nuclear localization.\n\n"
        "2. **Transmembrane Helices**: The sequence does not show characteristics of "
        "transmembrane helices, such as stretches of hydrophobic residues typically found in membrane-spanning regions. "
        "This suggests the protein is not membrane-bound.\n\n"
        "3. **Repeats and Low-Complexity Segments**: The sequence "
        "does not contain obvious repetitive motifs or low-complexity regions that are often associated with structural "
        "or functional repeats.\n\n"
        "4. **Catalytic Motifs/Domains**: The sequence contains cysteine residues (C) and "
        "histidine (H) that could potentially form a zinc finger or other metal-binding motif, but there is no clear "
        "pattern indicating a known catalytic domain.\n\n"
        "5. **Family and Function**: The sequence contains a segment "
        "\"PCPCG\" which is a characteristic motif found in some proteins of the UPF0225 family. This family is known "
        "for proteins with unknown functions, often involved in stress responses or regulatory roles.\n\n"
        "6. **Overall "
        "Function**: Given the lack of clear catalytic motifs or transmembrane regions, and the presence of a UPF0225 "
        "family motif, the protein is likely involved in a regulatory or structural role within the cell, potentially "
        "related to stress response or protein-protein interactions.\n"
        "</thinking>\n\n"
        "<answer>\n"
        "Belongs to the UPF0225 family.\n"
        "</answer>"
    )

    input_ids, labels = create_sft_training_data_simple(
        tokenizer=tokenizer,
        messages=[dict(role="user", content=user_prompt), dict(role="assistant", content=gt_ans)],
    )
    input_ids, labels = torch.tensor([input_ids], dtype=torch.long, device="cuda"), torch.tensor([labels], dtype=torch.long, device="cuda")
    attention_mask = torch.ones_like(input_ids)

    train_outputs = pllm(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        aa_seq=[aa_seq],
        stru_str=[structure]
    )
    print(train_outputs.loss)

    test_prompt = tokenizer.apply_chat_template(
        [dict(role="user", content=user_prompt)], add_generation_prompt=True, tokenize=False
    )
    test_inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")
    generated_ids = pllm.generate(
        # inputs=test_inputs.input_ids,
        # attention_mask=test_inputs.attention_mask,
        **test_inputs,
        aa_seq=[aa_seq],
        stru_str=[structure],
        max_new_tokens=1024
    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(test_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)
    print(generated_ids[0].tolist())


if __name__ == '__main__':
    test()
