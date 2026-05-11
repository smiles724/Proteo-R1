import gc
import os
import random
from typing import Any, Dict, Optional

import torch
import torch._dynamo

from torch import Tensor, nn
from torchmetrics import MeanMetric

import proteor1.generate.layers.initialize as init
from proteor1.generate.configuration import ATOM_ENCODER_DIM
from proteor1.generate.data_load import const
from proteor1.generate.utils import save_hyperparameters, AttributeDict
from proteor1.generate.data_load.feature.symmetry import (
    minimum_lddt_symmetry_coords,
    minimum_symmetry_coords,
)

from proteor1.generate.loss.validation import (
    compute_pae_mae,
    compute_pde_mae,
    compute_plddt_mae,
    factored_lddt_loss,
    factored_token_lddt_dist_loss,
    weighted_minimum_rmsd,
)
from proteor1.generate.modules.confidence import ConfidenceModule
from proteor1.generate.modules.diffusion import AtomDiffusion
from proteor1.generate.modules.encoders import RelativePositionEncoder
from proteor1.generate.modules.trunk import (
    DistogramModule,
    InputEmbedder,
    MSAModule,
    PairformerModule,
)
from proteor1.generate.modules.utils import ExponentialMovingAverage

from transformers.utils import logging
logger = logging.get_logger(__name__)


class Boltz1(torch.nn.Module):
    def __init__(  # noqa: PLR0915, C901, PLR0912
        self,
        atom_s: int,
        atom_z: int,
        token_s: int,
        token_z: int,
        num_bins: int,
        training_args: dict[str, Any],
        validation_args: dict[str, Any],
        embedder_args: dict[str, Any],
        msa_args: dict[str, Any],
        pairformer_args: dict[str, Any],
        score_model_args: dict[str, Any],
        diffusion_process_args: dict[str, Any],
        diffusion_loss_args: dict[str, Any],
        confidence_model_args: dict[str, Any],
        atom_feature_dim: int = 128,
        confidence_prediction: bool = False,
        sequence_prediction_training: bool = False,
        confidence_imitate_trunk: bool = False,
        alpha_pae: float = 0.0,
        structure_inpainting: bool = False,
        structure_prediction_training: bool = True,
        atoms_per_window_queries: int = 32,
        atoms_per_window_keys: int = 128,
        compile_pairformer: bool = False,
        compile_structure: bool = False,
        compile_confidence: bool = False,
        nucleotide_rmsd_weight: float = 5.0,
        ligand_rmsd_weight: float = 10.0,
        no_msa: bool = False,
        no_atom_encoder: bool = False,
        ema: bool = False,
        ema_decay: float = 0.999,
        min_dist: float = 2.0,
        max_dist: float = 22.0,
        predict_args: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        if diffusion_process_args.get("noise_type") == 'discrete':
            logger.warning("`discrete` is not a valid noise_type! Casted it to `discrete_absorb`")
            diffusion_process_args["noise_type"] = "discrete_absorb"

        # Store all hyperparameters (mimics PyTorch Lightning's save_hyperparameters)
        # Automatically captures all __init__ arguments using inspect
        save_hyperparameters(self)

        self.lddt = nn.ModuleDict()
        self.disto_lddt = nn.ModuleDict()
        self.complex_lddt = nn.ModuleDict()
        if confidence_prediction:
            self.top1_lddt = nn.ModuleDict()
            self.iplddt_top1_lddt = nn.ModuleDict()
            self.ipde_top1_lddt = nn.ModuleDict()
            self.pde_top1_lddt = nn.ModuleDict()
            self.ptm_top1_lddt = nn.ModuleDict()
            self.iptm_top1_lddt = nn.ModuleDict()
            self.ligand_iptm_top1_lddt = nn.ModuleDict()
            self.protein_iptm_top1_lddt = nn.ModuleDict()
            self.avg_lddt = nn.ModuleDict()
            self.plddt_mae = nn.ModuleDict()
            self.pde_mae = nn.ModuleDict()
            self.pae_mae = nn.ModuleDict()
        for m in const.out_types + ["pocket_ligand_protein"]:
            self.lddt[m] = MeanMetric()
            self.disto_lddt[m] = MeanMetric()
            self.complex_lddt[m] = MeanMetric()
            if confidence_prediction:
                self.top1_lddt[m] = MeanMetric()
                self.iplddt_top1_lddt[m] = MeanMetric()
                self.ipde_top1_lddt[m] = MeanMetric()
                self.pde_top1_lddt[m] = MeanMetric()
                self.ptm_top1_lddt[m] = MeanMetric()
                self.iptm_top1_lddt[m] = MeanMetric()
                self.ligand_iptm_top1_lddt[m] = MeanMetric()
                self.protein_iptm_top1_lddt[m] = MeanMetric()
                self.avg_lddt[m] = MeanMetric()
                self.pde_mae[m] = MeanMetric()
                self.pae_mae[m] = MeanMetric()
        for m in const.out_single_types:
            if confidence_prediction:
                self.plddt_mae[m] = MeanMetric()
        self.rmsd = MeanMetric()
        self.best_rmsd = MeanMetric()

        self.train_confidence_loss_logger = MeanMetric()
        self.train_confidence_loss_dict_logger = nn.ModuleDict()
        for m in [
            "plddt_loss",
            "resolved_loss",
            "pde_loss",
            "pae_loss",
            "rel_plddt_loss",
            "rel_pde_loss",
            "rel_pae_loss",
        ]:
            self.train_confidence_loss_dict_logger[m] = MeanMetric()

        self.ema = None
        self.use_ema = ema
        self.ema_decay = ema_decay

        self.training_args = training_args
        self.validation_args = validation_args
        self.diffusion_loss_args = diffusion_loss_args
        self.predict_args = predict_args

        self.nucleotide_rmsd_weight = nucleotide_rmsd_weight
        self.ligand_rmsd_weight = ligand_rmsd_weight

        self.num_bins = num_bins
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.is_pairformer_compiled = False

        # Input projections
        s_input_dim = (
            token_s + 2 * const.num_tokens + 1 + len(const.pocket_contact_info)
        )
        self.s_init = nn.Linear(s_input_dim, token_s, bias=False)
        self.z_init_1 = nn.Linear(s_input_dim, token_z, bias=False)
        self.z_init_2 = nn.Linear(s_input_dim, token_z, bias=False)

        # Input embeddings
        full_embedder_args = {
            "atom_s": atom_s,
            "atom_z": atom_z,
            "token_s": token_s,
            "token_z": token_z,
            "atoms_per_window_queries": atoms_per_window_queries,
            "atoms_per_window_keys": atoms_per_window_keys,
            "atom_feature_dim": atom_feature_dim,
            "no_atom_encoder": no_atom_encoder,
            **embedder_args,
        }
        self.input_embedder = InputEmbedder(**full_embedder_args)
        self.rel_pos = RelativePositionEncoder(token_z)
        self.token_bonds = nn.Linear(1, token_z, bias=False)

        # Normalization layers
        self.s_norm = nn.LayerNorm(token_s)
        self.z_norm = nn.LayerNorm(token_z)

        # Recycling projections
        self.s_recycle = nn.Linear(token_s, token_s, bias=False)
        self.z_recycle = nn.Linear(token_z, token_z, bias=False)
        init.gating_init_(self.s_recycle.weight)
        init.gating_init_(self.z_recycle.weight)

        # Pairwise stack
        self.no_msa = no_msa
        if not no_msa:
            self.msa_module = MSAModule(
                token_z=token_z,
                s_input_dim=s_input_dim,
                **msa_args,
            )
        self.pairformer_module = PairformerModule(token_s, token_z, **pairformer_args)
        if compile_pairformer:
            # Big models hit the default cache limit (8)
            self.is_pairformer_compiled = True
            torch._dynamo.config.cache_size_limit = 512
            torch._dynamo.config.accumulated_cache_size_limit = 512
            self.pairformer_module = torch.compile(
                self.pairformer_module,
                dynamic=False,
                fullgraph=False,
            )

        if confidence_prediction:
            confidence_model_args["use_s_diffusion"] = True
            if "use_gaussian" in confidence_model_args:
                confidence_model_args.pop("use_gaussian")
            if "relative_confidence" in confidence_model_args["confidence_args"]:
                confidence_model_args["confidence_args"].pop("relative_confidence")

        # Output modules
        self.structure_module = AtomDiffusion(
            score_model_args={
                "token_z": token_z,
                "token_s": token_s,
                "atom_z": atom_z,
                "atom_s": atom_s,
                "atoms_per_window_queries": atoms_per_window_queries,
                "atoms_per_window_keys": atoms_per_window_keys,
                "atom_feature_dim": atom_feature_dim,
                "sequence_train": sequence_prediction_training,
                "structure_train": structure_prediction_training,
                **score_model_args,
            },
            compile_score=compile_structure,
            accumulate_token_repr="use_s_diffusion" in confidence_model_args
            and confidence_model_args["use_s_diffusion"],
            **diffusion_process_args,
        )
        self.distogram_module = DistogramModule(token_z, num_bins)
        self.confidence_prediction = confidence_prediction
        self.sequence_prediction_training = sequence_prediction_training
        self.alpha_pae = alpha_pae

        self.structure_inpainting = structure_inpainting
        self.structure_prediction_training = structure_prediction_training
        self.confidence_imitate_trunk = confidence_imitate_trunk
        if self.confidence_prediction:
            if self.confidence_imitate_trunk:
                self.confidence_module = ConfidenceModule(
                    token_s,
                    token_z,
                    compute_pae=alpha_pae > 0,
                    imitate_trunk=True,
                    pairformer_args=pairformer_args,
                    full_embedder_args=full_embedder_args,
                    msa_args=msa_args,
                    **confidence_model_args,
                )
            else:
                self.confidence_module = ConfidenceModule(
                    token_s,
                    token_z,
                    compute_pae=alpha_pae > 0,
                    **confidence_model_args,
                )
            if compile_confidence:
                self.confidence_module = torch.compile(
                    self.confidence_module, dynamic=False, fullgraph=False
                )

        for name, param in self.named_parameters():
           param.requires_grad = False

        # Remove grad from weights they are not trained for ddp
        if confidence_prediction:
            for name, param in self.named_parameters():
                if name.split(".")[0] == "confidence_module":
                    param.requires_grad = True
        if sequence_prediction_training:
            self.training_args.diffusion_multiplicity = 1 # When predicting antibodies, we don't need mini-rollout
            self.training_args.diffusion_samples = 1
            self.validation_args.diffusion_samples = 1
            for name, param in self.named_parameters():
                if "sequence_model" in name:
                    param.requires_grad = True
        if structure_prediction_training:
            for name, param in self.named_parameters():
                if name.split(".")[0] != "confidence_module" and "sequence_model" not in name:
                    param.requires_grad = True

    @classmethod
    def from_pretrained(cls, ckpt_path, load_ema=False, **override_kwargs):
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        state_dict = checkpoint["state_dict"]
        hyper_parameters = checkpoint["hyper_parameters"]
        hyper_parameters.update(**override_kwargs)

        model = cls(**hyper_parameters)
        model.on_load_checkpoint()
        missing_keys, unexpected_keys = model.load_state_dict(state_dict=state_dict, strict=False)

        if missing_keys:
            print(f"[Boltz1.from_pretrained] Missing keys ({len(missing_keys)}): {missing_keys[:10]}{'...' if len(missing_keys) > 10 else ''}")

        if unexpected_keys:
            print(f"[Boltz1.from_pretrained] Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:10]}{'...' if len(unexpected_keys) > 10 else ''}")

        # Safe EMA loading: only load if checkpoint has EMA and model uses EMA
        if load_ema:
            if not model.use_ema:
                print(
                    "[Boltz1.from_pretrained] Warning: Model uses EMA but checkpoint does not contain EMA state. Initializing EMA from scratch.")

            if "ema" in checkpoint:
                assert model.ema is not None
                model.ema.load_state_dict(checkpoint["ema"])
            else:
                print("[Boltz1.from_pretrained] Warning: Model uses EMA but checkpoint does not contain EMA state. Initializing EMA from scratch.")

        return model

    @property
    def hparams(self) -> AttributeDict:
        """The collection of hyperparameters saved with save_hyperparameters.

        Returns:
            Mutable hyperparameters dictionary with dot-notation access.
        """
        if not hasattr(self, "_hparams"):
            self._hparams = AttributeDict()
        return self._hparams

    def get_hyperparameters(self) -> dict:
        """
        Return the model's hyperparameters.

        This method mimics PyTorch Lightning's save_hyperparameters behavior,
        returning a dictionary of all __init__ parameters.

        Returns
        -------
        dict
            Dictionary containing all hyperparameters.
        """
        return dict(self._hparams)

    def forward(
        self,
        feats: dict[str, Tensor],
        recycling_steps: int = 0,
        num_sampling_steps: Optional[int] = None,
        multiplicity_diffusion_train: int = 1,
        diffusion_samples: int = 1,
        run_confidence_sequentially: bool = False,
        text_conditioning: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        """
        Forward pass of Boltz1.

        Parameters
        ----------
        feats : dict[str, Tensor]
            Feature dictionary containing input features.
        recycling_steps : int
            Number of recycling iterations (default 0).
        num_sampling_steps : Optional[int]
            Number of diffusion sampling steps.
        multiplicity_diffusion_train : int
            Multiplicity for diffusion training (default 1).
        diffusion_samples : int
            Number of diffusion samples (default 1).
        run_confidence_sequentially : bool
            Whether to run confidence module sequentially (default False).
        text_conditioning : Optional[Tensor]
            If provided, add this tensor to the first 384 dimensions of s_inputs.
            This injects text conditioning from the Understanding Model into
            the AtomAttentionEncoder's continuous output, while preserving
            the discrete features in s_inputs[:, :, 384:].
            Shape: [B, N_token, 384].
            When None, original Boltz behavior is preserved (no conditioning).

        Returns
        -------
        dict[str, Tensor]
            Output dictionary containing predictions.
        """
        dict_out = {}

        # Compute input embeddings
        with torch.set_grad_enabled(
            self.training and self.structure_prediction_training
        ):
            # Compute s_inputs from input_embedder
            s_inputs = self.input_embedder(feats)

            # Inject text_conditioning into first ATOM_ENCODER_DIM dimensions if provided
            # s_inputs[:, :, :ATOM_ENCODER_DIM] = AtomAttentionEncoder continuous output
            # s_inputs[:, :, ATOM_ENCODER_DIM:] = discrete features (preserved unchanged)
            if text_conditioning is not None:
                # Validate shape
                B, N_token, c_s_inputs = s_inputs.shape
                assert text_conditioning.shape == (B, N_token, ATOM_ENCODER_DIM), (
                    f"text_conditioning shape mismatch: expected ({B}, {N_token}, {ATOM_ENCODER_DIM}), "
                    f"got {text_conditioning.shape}"
                )
                # Ensure dtype compatibility with s_inputs (text_conditioning may be bf16 from Understanding Model)
                if text_conditioning.dtype != s_inputs.dtype:
                    text_conditioning = text_conditioning.to(s_inputs.dtype)
                # Add text_conditioning to first ATOM_ENCODER_DIM dims (AtomAttentionEncoder output)
                s_inputs = s_inputs.clone()  # Avoid in-place modification
                s_inputs[:, :, :ATOM_ENCODER_DIM] = s_inputs[:, :, :ATOM_ENCODER_DIM] + text_conditioning

            # Initialize the sequence and pairwise embeddings
            s_init = self.s_init(s_inputs)  # self.s_init is nn.Linear
            z_init = (
                self.z_init_1(s_inputs)[:, :, None]   # self.z_init is nn.Linear
                + self.z_init_2(s_inputs)[:, None, :] # z_init pairwise
            )
            relative_position_encoding = self.rel_pos(feats)
            z_init = z_init + relative_position_encoding   # Introduce the position info.
            z_init = z_init + self.token_bonds(feats["token_bonds"].float())   # self.token_bonds is nn.Linear

            # Perform rounds of the pairwise stack
            s = torch.zeros_like(s_init)
            z = torch.zeros_like(z_init)

            # Compute pairwise mask
            mask = feats["token_pad_mask"].float()
            pair_mask = mask[:, :, None] * mask[:, None, :]

            for i in range(recycling_steps + 1):
                with torch.set_grad_enabled(self.training and (i == recycling_steps)):
                    # Fixes an issue with unused parameters in autocast
                    if (
                        self.training
                        and (i == recycling_steps)
                        and torch.is_autocast_enabled()
                    ):
                        torch.clear_autocast_cache()  # ?

                    # Apply recycling
                    s = s_init + self.s_recycle(self.s_norm(s))
                    z = z_init + self.z_recycle(self.z_norm(z))

                    # Compute pairwise stack
                    if not self.no_msa:
                        z = z + self.msa_module(z, s_inputs, feats)

                    # Revert to uncompiled version for validation
                    if self.is_pairformer_compiled and not self.training:
                        pairformer_module = self.pairformer_module._orig_mod  # noqa: SLF001
                    else:
                        pairformer_module = self.pairformer_module

                    if int(os.getenv("SKIP_PAIRFORMER", "0")) == 0:
                        s, z = pairformer_module(s, z, mask=mask, pair_mask=pair_mask)

            pdistogram = self.distogram_module(z)
            dict_out = {"pdistogram": pdistogram}

        # Compute structure module
        if self.training and (self.structure_prediction_training or self.sequence_prediction_training):
            dict_out.update(
                self.structure_module(
                    s_trunk=s,  # singe seq
                    z_trunk=z,  # pairwise embedding
                    s_inputs=s_inputs,
                    feats=feats,
                    relative_position_encoding=relative_position_encoding,
                    num_sampling_steps=num_sampling_steps,
                    multiplicity=multiplicity_diffusion_train,
                )
            )

        if (not self.training) or self.confidence_prediction:
            dict_out.update(
                self.structure_module.sample(
                    s_trunk=s,
                    z_trunk=z,
                    s_inputs=s_inputs,
                    feats=feats,
                    relative_position_encoding=relative_position_encoding,
                    num_sampling_steps=num_sampling_steps,
                    atom_mask=feats["atom_pad_mask"],
                    multiplicity=diffusion_samples,
                    train_accumulate_token_repr=self.training,
                    inpaint=self.structure_inpainting,
                )
            )

        if self.confidence_prediction:
            dict_out.update(
                self.confidence_module(
                    s_inputs=s_inputs.detach(),
                    s=s.detach(),
                    z=z.detach(),
                    s_diffusion=(
                        dict_out["diff_token_repr"]
                        if self.confidence_module.use_s_diffusion
                        else None
                    ),
                    x_pred=dict_out["sample_atom_coords"].detach(),
                    feats=feats,
                    pred_distogram_logits=dict_out["pdistogram"].detach(),
                    multiplicity=diffusion_samples,
                    run_sequentially=run_confidence_sequentially,
                )
            )
        # if self.confidence_prediction and self.confidence_module.use_s_diffusion:
        #     dict_out.pop("diff_token_repr", None)
        return dict_out

    def get_true_coordinates(
        self,
        batch,
        out,
        diffusion_samples,
        symmetry_correction,
        lddt_minimization=True,
    ):
        if symmetry_correction:
            min_coords_routine = (
                minimum_lddt_symmetry_coords
                if lddt_minimization
                else minimum_symmetry_coords
            )
            true_coords = []
            true_coords_resolved_mask = []
            rmsds, best_rmsds = [], []
            for idx in range(batch["token_index"].shape[0]):
                best_rmsd = float("inf")
                for rep in range(diffusion_samples):
                    i = idx * diffusion_samples + rep
                    best_true_coords, rmsd, best_true_coords_resolved_mask = (
                        min_coords_routine(
                            coords=out["sample_atom_coords"][i : i + 1],
                            feats=batch,
                            index_batch=idx,
                            nucleotide_weight=self.nucleotide_rmsd_weight,
                            ligand_weight=self.ligand_rmsd_weight,
                        )
                    )
                    rmsds.append(rmsd)
                    true_coords.append(best_true_coords)
                    true_coords_resolved_mask.append(best_true_coords_resolved_mask)
                    if rmsd < best_rmsd:
                        best_rmsd = rmsd
                best_rmsds.append(best_rmsd)
            true_coords = torch.cat(true_coords, dim=0)
            true_coords_resolved_mask = torch.cat(true_coords_resolved_mask, dim=0)
        else:
            true_coords = (
                batch["coords"].squeeze(1).repeat_interleave(diffusion_samples, 0)
            )

            true_coords_resolved_mask = batch["atom_resolved_mask"].repeat_interleave(
                diffusion_samples, 0
            )
            rmsds, best_rmsds = weighted_minimum_rmsd(
                out["sample_atom_coords"],
                batch,
                multiplicity=diffusion_samples,
                nucleotide_weight=self.nucleotide_rmsd_weight,
                ligand_weight=self.ligand_rmsd_weight,
            )

        return true_coords, rmsds, best_rmsds, true_coords_resolved_mask

    def gradient_norm(self, module) -> float:
        # Only compute over parameters that are being trained
        parameters = filter(lambda p: p.requires_grad, module.parameters())
        parameters = filter(lambda p: p.grad is not None, parameters)
        norm = torch.tensor([p.grad.norm(p=2) ** 2 for p in parameters]).sum().sqrt()
        return norm

    def parameter_norm(self, module) -> float:
        # Only compute over parameters that are being trained
        parameters = filter(lambda p: p.requires_grad, module.parameters())
        norm = torch.tensor([p.norm(p=2) ** 2 for p in parameters]).sum().sqrt()
        return norm

    def predict(
        self,
        batch: Any,
        text_conditioning: Optional[Tensor] = None,
        recycling_steps: Optional[int] = None,
        sampling_steps: Optional[int] = None,
        diffusion_samples: Optional[int] = None,
    ) -> Any:
        """
        Run prediction on input batch with optional text conditioning.

        Parameters
        ----------
        batch : Any
            Feature dictionary from BoltzFeaturizer
        text_conditioning : Optional[Tensor]
            If provided, add this tensor to the first 384 dimensions of s_inputs.
            Shape: [B, N_token, 384]. When None, original Boltz behavior is preserved.
        recycling_steps : Optional[int]
            Number of recycling iterations. If None, uses predict_args default.
        sampling_steps : Optional[int]
            Number of diffusion sampling steps. If None, uses predict_args default.
        diffusion_samples : Optional[int]
            Number of diffusion samples. If None, uses predict_args default.

        Returns
        -------
        dict
            Prediction dictionary containing coordinates and confidence scores
        """
        # Use provided values or fall back to predict_args defaults
        if recycling_steps is None:
            recycling_steps = self.predict_args["recycling_steps"]
        if sampling_steps is None:
            sampling_steps = self.predict_args["sampling_steps"]
        if diffusion_samples is None:
            diffusion_samples = self.predict_args["diffusion_samples"]

        try:
            out = self(
                batch,
                recycling_steps=recycling_steps,
                num_sampling_steps=sampling_steps,
                diffusion_samples=diffusion_samples,
                run_confidence_sequentially=True,
                text_conditioning=text_conditioning,
            )
            pred_dict = {"exception": False}
            pred_dict["masks"] = batch["atom_pad_mask"]
            pred_dict["coords"] = out["sample_atom_coords"]
            pred_dict["seqs"] = out["sample_seqs"]
            if self.predict_args.get("write_confidence_summary", True):
                pred_dict["confidence_score"] = (
                    4 * out["complex_plddt"] +
                    (out["iptm"] if not torch.allclose(out["iptm"], torch.zeros_like(out["iptm"])) else out["ptm"])
                ) / 5
                for key in [
                    "ptm",
                    "iptm",
                    "ligand_iptm",
                    "protein_iptm",
                    "pair_chains_iptm",
                    "complex_plddt",
                    "complex_iplddt",
                    "complex_pde",
                    "complex_ipde",
                    "plddt",
                ]:
                    pred_dict[key] = out[key]
            if self.predict_args.get("write_full_pae", True):
                pred_dict["pae"] = out["pae"]
            if self.predict_args.get("write_full_pde", False):
                pred_dict["pde"] = out["pde"]

            return pred_dict

        except RuntimeError as e:  # catch out of memory exceptions
            if "out of memory" in str(e):
                print("| WARNING: ran out of memory, skipping batch")
                torch.cuda.empty_cache()
                gc.collect()
                return {"exception": True}
            else:
                raise {"exception": True}

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if self.use_ema:
            checkpoint["ema"] = self.ema.state_dict()

    def on_load_checkpoint(self) -> None:
        if self.use_ema and self.ema is None:
            self.ema = ExponentialMovingAverage(
                parameters=self.parameters(), decay=self.ema_decay
            )

    def on_train_start(self):
        if self.use_ema and self.ema is None:
            self.ema = ExponentialMovingAverage(
                parameters=self.parameters(), decay=self.ema_decay
            )
        elif self.use_ema:
            # Move EMA to the same device as model parameters
            device = next(self.parameters()).device
            self.ema.to(device)

    def on_train_epoch_start(self) -> None:
        if self.use_ema:
            self.ema.restore(self.parameters())

    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int) -> None:
        # Updates EMA parameters after optimizer.step()
        if self.use_ema:
            self.ema.update(self.parameters())

    def prepare_eval(self) -> None:
        if self.use_ema and self.ema is None:
            self.ema = ExponentialMovingAverage(
                parameters=self.parameters(), decay=self.ema_decay
            )

        if self.use_ema:
            self.ema.store(self.parameters())
            self.ema.copy_to(self.parameters())

    def on_validation_start(self):
        self.prepare_eval()

    def on_predict_start(self) -> None:
        self.prepare_eval()

    def on_test_start(self) -> None:
        self.prepare_eval()

