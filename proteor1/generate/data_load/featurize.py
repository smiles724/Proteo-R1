"""
BoltzFeaturizer - Generates model input features from Tokenized data.

This module implements the featurization logic for converting tokenized
protein structure data into tensor features suitable for the Boltz model.

Core components:
1. process_token_features() - Token-level features
2. process_atom_features() - Atom-level features
3. process_msa_features() - MSA features
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.nn.functional import one_hot, pad

from ..constants import (
    chain_type_ids,
    num_elements,
    num_tokens,
    pocket_contact_info,
    ref_atoms,
    token_ids,
    tokens,
)
from ..utils import center_random_augmentation
from .types import (
    MSA,
    MSADeletion,
    MSAResidue,
    MSASequence,
    Tokenized,
)


####################################################################################################
# HELPER FUNCTIONS
####################################################################################################


def pad_dim(data: Tensor, dim: int, pad_len: int, value: float = 0) -> Tensor:
    """Pad a tensor along a given dimension.

    Parameters
    ----------
    data : Tensor
        The input tensor.
    dim : int
        The dimension to pad.
    pad_len : int
        The padding length.
    value : float, optional
        The value to pad with. Default is 0.

    Returns
    -------
    Tensor
        The padded tensor.
    """
    if pad_len <= 0:
        return data

    total_dims = len(data.shape)
    padding = [0] * (2 * (total_dims - dim))
    padding[2 * (total_dims - 1 - dim) + 1] = pad_len
    return pad(data, tuple(padding), value=value)


def pad_to_max(data: List[Tensor], value: float = 0) -> Tuple[Tensor, Tensor]:
    """Pad the data in all dimensions to the maximum found.

    Parameters
    ----------
    data : List[Tensor]
        List of tensors to pad.
    value : float
        The value to use for padding.

    Returns
    -------
    Tuple[Tensor, Tensor]
        - The padded tensor (stacked).
        - The padding mask.
    """
    if isinstance(data[0], str):
        return data, 0

    # Check if all have the same shape
    if all(d.shape == data[0].shape for d in data):
        return torch.stack(data, dim=0), 0

    # Get the maximum in each dimension
    num_dims = len(data[0].shape)
    max_dims = [max(d.shape[i] for d in data) for i in range(num_dims)]

    # Get the padding lengths
    pad_lengths = []
    for d in data:
        dims = []
        for i in range(num_dims):
            dims.append(0)
            dims.append(max_dims[num_dims - i - 1] - d.shape[num_dims - i - 1])
        pad_lengths.append(dims)

    # Pad the data
    padding = [
        pad(torch.ones_like(d), pad_len, value=0)
        for d, pad_len in zip(data, pad_lengths)
    ]
    data = [pad(d, pad_len, value=value) for d, pad_len in zip(data, pad_lengths)]

    # Stack the data
    padding = torch.stack(padding, dim=0)
    data = torch.stack(data, dim=0)

    return data, padding


def compute_collinear_mask(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Check if vectors are collinear (for frame validity).

    Parameters
    ----------
    v1, v2 : np.ndarray
        Vectors to check for collinearity.

    Returns
    -------
    np.ndarray
        Boolean mask where True means vectors are NOT collinear.
    """
    norm1 = np.linalg.norm(v1, axis=1, keepdims=True)
    norm2 = np.linalg.norm(v2, axis=1, keepdims=True)
    v1 = v1 / (norm1 + 1e-6)
    v2 = v2 / (norm2 + 1e-6)
    mask_angle = np.abs(np.sum(v1 * v2, axis=1)) < 0.9063
    mask_overlap1 = norm1.reshape(-1) > 1e-2
    mask_overlap2 = norm2.reshape(-1) > 1e-2
    return mask_angle & mask_overlap1 & mask_overlap2


def compute_frames_nonpolymer(
    data: Tokenized,
    coords: np.ndarray,
    resolved_mask: np.ndarray,
    atom_to_token: np.ndarray,
    frame_data: List,
    resolved_frame_data: List,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute frames for non-polymer tokens.

    For non-polymer molecules (ligands), frames are computed based on
    the nearest atoms rather than predefined backbone atoms.

    Parameters
    ----------
    data : Tokenized
        The tokenized data.
    coords : np.ndarray
        Atom coordinates.
    resolved_mask : np.ndarray
        Mask indicating resolved atoms.
    atom_to_token : np.ndarray
        Mapping from atom to token indices.
    frame_data : List
        Current frame data.
    resolved_frame_data : List
        Current frame resolved mask.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Updated frame data and resolved frame mask.
    """
    frame_data = np.array(frame_data)
    resolved_frame_data = np.array(resolved_frame_data)
    asym_id_token = data.tokens["asym_id"]
    asym_id_atom = data.tokens["asym_id"][atom_to_token]
    token_idx = 0
    atom_idx = 0

    for chain_id in np.unique(data.tokens["asym_id"]):
        mask_chain_token = asym_id_token == chain_id
        mask_chain_atom = asym_id_atom == chain_id
        num_tokens_chain = mask_chain_token.sum()
        num_atoms_chain = mask_chain_atom.sum()

        # Skip non-NONPOLYMER chains or chains with < 3 atoms
        if (
            data.tokens[token_idx]["mol_type"] != chain_type_ids["NONPOLYMER"]
            or num_atoms_chain < 3
        ):
            token_idx += num_tokens_chain
            atom_idx += num_atoms_chain
            continue

        # Compute distance matrix between atoms in chain
        chain_coords = coords.reshape(-1, 3)[mask_chain_atom]
        dist_mat = (
            (chain_coords[:, None, :] - chain_coords[None, :, :]) ** 2
        ).sum(-1) ** 0.5

        # Mask out unresolved pairs
        chain_resolved = resolved_mask[mask_chain_atom]
        resolved_pair = 1 - (
            chain_resolved[None, :] * chain_resolved[:, None]
        ).astype(np.float32)
        resolved_pair[resolved_pair == 1] = math.inf

        # Find nearest atoms for frame construction
        indices = np.argsort(dist_mat + resolved_pair, axis=1)
        frames = (
            np.concatenate(
                [
                    indices[:, 1:2],
                    indices[:, 0:1],
                    indices[:, 2:3],
                ],
                axis=1,
            )
            + atom_idx
        )

        frame_data[token_idx : token_idx + num_atoms_chain, :] = frames
        resolved_frame_data[token_idx : token_idx + num_atoms_chain] = resolved_mask[
            frames
        ].all(axis=1)

        token_idx += num_tokens_chain
        atom_idx += num_atoms_chain

    # Check for collinearity
    frames_expanded = coords.reshape(-1, 3)[frame_data]
    mask_collinear = compute_collinear_mask(
        frames_expanded[:, 1] - frames_expanded[:, 0],
        frames_expanded[:, 1] - frames_expanded[:, 2],
    )

    return frame_data, resolved_frame_data & mask_collinear


def dummy_msa(residues: np.ndarray) -> MSA:
    """Create a dummy MSA for a chain without MSA data.

    Parameters
    ----------
    residues : np.ndarray
        The residues for the chain.

    Returns
    -------
    MSA
        A dummy MSA with just the sequence.
    """
    res_types = [res["res_type"] for res in residues]
    deletions = []
    sequences = [(0, -1, 0, len(res_types), 0, 0)]

    return MSA(
        residues=np.array(res_types, dtype=MSAResidue),
        deletions=np.array(deletions, dtype=MSADeletion),
        sequences=np.array(sequences, dtype=MSASequence),
    )


def construct_paired_msa(
    data: Tokenized,
    max_seqs: int,
    max_pairs: int = 8192,
    max_total: int = 16384,
    random_subset: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Construct paired MSA from tokenized data.

    This function pairs MSA sequences from different chains based on
    their taxonomy IDs.

    Parameters
    ----------
    data : Tokenized
        The tokenized data.
    max_seqs : int
        Maximum number of sequences to include.
    max_pairs : int, optional
        Maximum number of paired sequences. Default is 8192.
    max_total : int, optional
        Maximum total sequences. Default is 16384.
    random_subset : bool, optional
        Whether to randomly sample sequences. Default is False.

    Returns
    -------
    Tuple[Tensor, Tensor, Tensor]
        - msa_data: [N_token, N_msa] residue type indices
        - del_data: [N_token, N_msa] deletion counts
        - paired_data: [N_token, N_msa] paired mask
    """
    # Get unique chains
    chain_ids = np.unique(data.tokens["asym_id"])

    # Get relevant MSA, create dummy for chains without
    msa = {k: data.msa[k] for k in chain_ids if k in data.msa}
    for chain_id in chain_ids:
        if chain_id not in msa:
            chain = data.structure.chains[chain_id]
            res_start = chain["res_idx"]
            res_end = res_start + chain["res_num"]
            residues = data.structure.residues[res_start:res_end]
            msa[chain_id] = dummy_msa(residues)

    # Map taxonomies to (chain_id, seq_idx)
    taxonomy_map: Dict[int, List] = {}
    for chain_id, chain_msa in msa.items():
        sequences = chain_msa.sequences
        sequences = sequences[sequences["taxonomy"] != -1]
        for sequence in sequences:
            seq_idx = sequence["seq_idx"]
            taxon = sequence["taxonomy"]
            taxonomy_map.setdefault(taxon, []).append((chain_id, seq_idx))

    # Remove taxonomies with only one sequence
    taxonomy_map = {k: v for k, v in taxonomy_map.items() if len(v) > 1}
    taxonomy_map = sorted(
        taxonomy_map.items(),
        key=lambda x: len({c for c, _ in x[1]}),
        reverse=True,
    )

    # Track available sequences per chain
    visited = {(c, s) for c, items in taxonomy_map for s in items}
    available = {}
    for c in chain_ids:
        available[c] = [
            i for i in range(1, len(msa[c].sequences)) if (c, i) not in visited
        ]

    # Create sequence pairs
    is_paired = []
    pairing = []

    # Start with the first sequence for each chain
    is_paired.append({c: 1 for c in chain_ids})
    pairing.append({c: 0 for c in chain_ids})

    # Add paired rows based on taxonomy
    for _, pairs in taxonomy_map:
        chain_occurences = {}
        for chain_id, seq_idx in pairs:
            chain_occurences.setdefault(chain_id, []).append(seq_idx)

        max_occurences = max(len(v) for v in chain_occurences.values())
        for i in range(max_occurences):
            row_pairing = {}
            row_is_paired = {}

            for chain_id, seq_idxs in chain_occurences.items():
                idx = i % len(seq_idxs)
                seq_idx = seq_idxs[idx]
                row_pairing[chain_id] = seq_idx
                row_is_paired[chain_id] = 1

            for chain_id in chain_ids:
                if chain_id not in row_pairing:
                    row_is_paired[chain_id] = 0
                    if available[chain_id]:
                        seq_idx = available[chain_id].pop(0)
                        row_pairing[chain_id] = seq_idx
                    else:
                        row_pairing[chain_id] = -1

            pairing.append(row_pairing)
            is_paired.append(row_is_paired)

            if len(pairing) >= max_pairs:
                break

        if len(pairing) >= max_pairs:
            break

    # Add unpaired rows
    max_left = max(len(v) for v in available.values()) if available else 0
    for _ in range(min(max_total - len(pairing), max_left)):
        row_pairing = {}
        row_is_paired = {}
        for chain_id in chain_ids:
            row_is_paired[chain_id] = 0
            if available[chain_id]:
                seq_idx = available[chain_id].pop(0)
                row_pairing[chain_id] = seq_idx
            else:
                row_pairing[chain_id] = -1

        pairing.append(row_pairing)
        is_paired.append(row_is_paired)

        if len(pairing) >= max_total:
            break

    # Deterministic downsample to max_seqs
    if not random_subset:
        pairing = pairing[:max_seqs]
        is_paired = is_paired[:max_seqs]
    else:
        num_seqs = len(pairing)
        if num_seqs > max_seqs:
            indices = np.random.choice(list(range(1, num_seqs)), size=max_seqs - 1, replace=False)
            pairing = [pairing[0]] + [pairing[i] for i in indices]
            is_paired = [is_paired[0]] + [is_paired[i] for i in indices]

    # Map (chain_id, seq_idx, res_idx) to deletion
    deletions = {}
    for chain_id, chain_msa in msa.items():
        for sequence in chain_msa.sequences:
            del_start = sequence["del_start"]
            del_end = sequence["del_end"]
            chain_deletions = chain_msa.deletions[del_start:del_end]
            for deletion_data in chain_deletions:
                seq_idx = sequence["seq_idx"]
                res_idx = deletion_data["res_idx"]
                deletion = deletion_data["deletion"]
                deletions[(chain_id, seq_idx, res_idx)] = deletion

    # Build MSA data per token
    gap_token_id = token_ids["-"]
    msa_data = []
    del_data = []
    paired_data = []

    for token in data.tokens:
        token_res_types = []
        token_deletions = []
        token_is_paired = []

        for row_pairing, row_is_paired in zip(pairing, is_paired):
            res_idx = int(token["res_idx"])
            chain_id = int(token["asym_id"])
            seq_idx = row_pairing[chain_id]
            token_is_paired.append(row_is_paired[chain_id])

            if seq_idx == -1:
                token_res_types.append(gap_token_id)
                token_deletions.append(0)
            else:
                sequence = msa[chain_id].sequences[seq_idx]
                res_start = sequence["res_start"]
                res_type = msa[chain_id].residues[res_start + res_idx][0]
                deletion = deletions.get((chain_id, seq_idx, res_idx), 0)
                token_res_types.append(res_type)
                token_deletions.append(deletion)

        msa_data.append(token_res_types)
        del_data.append(token_deletions)
        paired_data.append(token_is_paired)

    msa_data = torch.tensor(msa_data, dtype=torch.long)
    del_data = torch.tensor(del_data, dtype=torch.float)
    paired_data = torch.tensor(paired_data, dtype=torch.float)

    return msa_data, del_data, paired_data


####################################################################################################
# FEATURE PROCESSING FUNCTIONS
####################################################################################################


def process_token_features(
    data: Tokenized,
    max_tokens: Optional[int] = None,
    inference_binder: Optional[List[int]] = None,
    inference_pocket: Optional[List[Tuple[int, int]]] = None,
) -> Dict[str, Tensor]:
    """Process token-level features.

    Parameters
    ----------
    data : Tokenized
        The tokenized data.
    max_tokens : int, optional
        Maximum number of tokens for padding.
    inference_binder : List[int], optional
        List of binder chain IDs for inference.
    inference_pocket : List[Tuple[int, int]], optional
        List of (chain_id, res_idx) for pocket residues.

    Returns
    -------
    Dict[str, Tensor]
        Dictionary of token features.
    """
    token_data = data.tokens
    token_bonds = data.bonds

    # Core token features
    token_index = torch.arange(len(token_data), dtype=torch.long)
    residue_index = torch.from_numpy(token_data["res_idx"].astype(np.int64))
    asym_id = torch.from_numpy(token_data["asym_id"].astype(np.int64))
    entity_id = torch.from_numpy(token_data["entity_id"].astype(np.int64))
    sym_id = torch.from_numpy(token_data["sym_id"].astype(np.int64))
    mol_type = torch.from_numpy(token_data["mol_type"].astype(np.int64))
    res_type = torch.from_numpy(token_data["res_type"].astype(np.int64))
    res_type_onehot = one_hot(res_type, num_classes=num_tokens).float()
    disto_center = torch.from_numpy(token_data["disto_coords"].astype(np.float32))

    # Mask features
    pad_mask = torch.ones(len(token_data), dtype=torch.float)
    resolved_mask = torch.from_numpy(token_data["resolved_mask"].astype(np.float32))
    disto_mask = torch.from_numpy(token_data["disto_mask"].astype(np.float32))

    # Token bond features
    if max_tokens is not None:
        pad_len = max_tokens - len(token_data)
        num_tokens_actual = max_tokens if pad_len > 0 else len(token_data)
    else:
        num_tokens_actual = len(token_data)

    tok_to_idx = {tok["token_idx"]: idx for idx, tok in enumerate(token_data)}
    bonds = torch.zeros(num_tokens_actual, num_tokens_actual, dtype=torch.float)
    for token_bond in token_bonds:
        token_1 = tok_to_idx.get(token_bond["token_1"])
        token_2 = tok_to_idx.get(token_bond["token_2"])
        if token_1 is not None and token_2 is not None:
            bonds[token_1, token_2] = 1
            bonds[token_2, token_1] = 1
    bonds = bonds.unsqueeze(-1)

    # Pocket conditioned feature
    pocket_feature = torch.zeros(len(token_data), len(pocket_contact_info))

    if inference_binder is not None and inference_pocket is not None:
        pocket_residues = set(inference_pocket)
        for idx, token in enumerate(token_data):
            if token["asym_id"] in inference_binder:
                pocket_feature[idx, pocket_contact_info["BINDER"]] = 1.0
            elif (token["asym_id"], token["res_idx"]) in pocket_residues:
                pocket_feature[idx, pocket_contact_info["POCKET"]] = 1.0
            else:
                pocket_feature[idx, pocket_contact_info["UNSELECTED"]] = 1.0
    else:
        pocket_feature[:, pocket_contact_info["UNSPECIFIED"]] = 1.0

    # Apply padding if needed
    if max_tokens is not None:
        pad_len = max_tokens - len(token_data)
        if pad_len > 0:
            token_index = pad_dim(token_index, 0, pad_len)
            residue_index = pad_dim(residue_index, 0, pad_len)
            asym_id = pad_dim(asym_id, 0, pad_len)
            entity_id = pad_dim(entity_id, 0, pad_len)
            sym_id = pad_dim(sym_id, 0, pad_len)
            mol_type = pad_dim(mol_type, 0, pad_len)
            res_type_onehot = pad_dim(res_type_onehot, 0, pad_len)
            disto_center = pad_dim(disto_center, 0, pad_len)
            pad_mask = pad_dim(pad_mask, 0, pad_len)
            resolved_mask = pad_dim(resolved_mask, 0, pad_len)
            disto_mask = pad_dim(disto_mask, 0, pad_len)
            pocket_feature = pad_dim(pocket_feature, 0, pad_len)

    return {
        "token_index": token_index,
        "residue_index": residue_index,
        "asym_id": asym_id,
        "entity_id": entity_id,
        "sym_id": sym_id,
        "mol_type": mol_type,
        "res_type": res_type_onehot,
        "disto_center": disto_center,
        "token_bonds": bonds,
        "token_pad_mask": pad_mask,
        "token_resolved_mask": resolved_mask,
        "token_disto_mask": disto_mask,
        "pocket_feature": pocket_feature,
    }


def process_atom_features(
    data: Tokenized,
    atoms_per_window_queries: int = 32,
    min_dist: float = 2.0,
    max_dist: float = 22.0,
    num_bins: int = 64,
    max_atoms: Optional[int] = None,
    max_tokens: Optional[int] = None,
    gt_oracle_cdr_mask: Optional[np.ndarray] = None,
) -> Dict[str, Tensor]:
    """Process atom-level features.

    Parameters
    ----------
    data : Tokenized
        The tokenized data.
    atoms_per_window_queries : int
        Atoms per window for attention. Default is 32.
    min_dist : float
        Minimum distance for distogram. Default is 2.0.
    max_dist : float
        Maximum distance for distogram. Default is 22.0.
    num_bins : int
        Number of distogram bins. Default is 64.
    max_atoms : int, optional
        Maximum number of atoms for padding.
    max_tokens : int, optional
        Maximum number of tokens for padding.
    gt_oracle_cdr_mask : np.ndarray, optional
        Token-level CDR mask whose atoms should use GT coordinates as ref_pos.

    Returns
    -------
    Dict[str, Tensor]
        Dictionary of atom features.
    """
    # Collect atom data
    atom_data = []
    ref_space_uid = []
    coord_data = []
    frame_data = []
    resolved_frame_data = []
    atom_to_token = []
    token_to_rep_atom = []
    disto_coords = []
    atom_idx = 0

    chain_res_ids = {}
    for token_id, token in enumerate(data.tokens):
        chain_idx, res_id = token["asym_id"], token["res_idx"]
        chain = data.structure.chains[chain_idx]

        if (chain_idx, res_id) not in chain_res_ids:
            new_idx = len(chain_res_ids)
            chain_res_ids[(chain_idx, res_id)] = new_idx
        else:
            new_idx = chain_res_ids[(chain_idx, res_id)]

        # Map atoms to ref space and tokens
        ref_space_uid.extend([new_idx] * token["atom_num"])
        atom_to_token.extend([token_id] * token["atom_num"])

        # Get atom data for this token
        start = token["atom_idx"]
        end = token["atom_idx"] + token["atom_num"]
        token_atoms = data.structure.atoms[start:end]

        # Map token to representative atom
        token_to_rep_atom.append(atom_idx + token["disto_idx"] - start)

        # Get token coordinates
        token_coords = np.array([token_atoms["coords"]])
        coord_data.append(token_coords)

        # Compute frame data
        res_type = tokens[token["res_type"]]

        if token["atom_num"] < 3 or res_type in ["<pad>", "UNK", "-"]:
            idx_frame_a, idx_frame_b, idx_frame_c = 0, 0, 0
            mask_frame = False
        elif token["mol_type"] == chain_type_ids["PROTEIN"] and res_type in ref_atoms:
            idx_frame_a = ref_atoms[res_type].index("N")
            idx_frame_b = ref_atoms[res_type].index("CA")
            idx_frame_c = ref_atoms[res_type].index("C")
            mask_frame = (
                token_atoms["is_present"][idx_frame_a]
                and token_atoms["is_present"][idx_frame_b]
                and token_atoms["is_present"][idx_frame_c]
            )
        elif (
            token["mol_type"] == chain_type_ids["DNA"]
            or token["mol_type"] == chain_type_ids["RNA"]
        ) and res_type in ref_atoms:
            idx_frame_a = ref_atoms[res_type].index("C1'")
            idx_frame_b = ref_atoms[res_type].index("C3'")
            idx_frame_c = ref_atoms[res_type].index("C4'")
            mask_frame = (
                token_atoms["is_present"][idx_frame_a]
                and token_atoms["is_present"][idx_frame_b]
                and token_atoms["is_present"][idx_frame_c]
            )
        else:
            idx_frame_a, idx_frame_b, idx_frame_c = 0, 0, 0
            mask_frame = False

        frame_data.append([idx_frame_a + atom_idx, idx_frame_b + atom_idx, idx_frame_c + atom_idx])
        resolved_frame_data.append(mask_frame)

        # Get distogram coordinates
        disto_coords_tok = data.structure.atoms[token["disto_idx"]]["coords"]
        disto_coords.append(disto_coords_tok)

        # Update atom data
        token_atoms = token_atoms.copy()
        token_atoms["coords"] = token_coords[0]
        atom_data.append(token_atoms)
        atom_idx += len(token_atoms)

    disto_coords = np.array(disto_coords)

    # Compute distogram
    t_center = torch.tensor(disto_coords, dtype=torch.float32)
    t_dists = torch.cdist(t_center, t_center)
    boundaries = torch.linspace(min_dist, max_dist, num_bins - 1)
    distogram = (t_dists.unsqueeze(-1) > boundaries).sum(dim=-1).long()
    disto_target = one_hot(distogram, num_classes=num_bins).float()

    atom_data = np.concatenate(atom_data)
    coord_data = np.concatenate(coord_data, axis=1)
    ref_space_uid = np.array(ref_space_uid)

    # Convert to tensors
    ref_atom_name_chars = torch.from_numpy(atom_data["name"].astype(np.int64))
    ref_element = torch.from_numpy(atom_data["element"].astype(np.int64))
    ref_charge = torch.from_numpy(atom_data["charge"].astype(np.float32))
    ref_pos = torch.from_numpy(atom_data["conformer"].copy().astype(np.float32))
    ref_space_uid_tensor = torch.from_numpy(ref_space_uid.astype(np.int64))
    coords = torch.from_numpy(coord_data.copy().astype(np.float32))
    resolved_mask = torch.from_numpy(atom_data["is_present"].astype(np.float32))
    pad_mask = torch.ones(len(atom_data), dtype=torch.float)
    atom_to_token_tensor = torch.tensor(atom_to_token, dtype=torch.long)
    token_to_rep_atom_tensor = torch.tensor(token_to_rep_atom, dtype=torch.long)
    cdr_atom_mask = None
    if gt_oracle_cdr_mask is not None:
        cdr_token_mask = torch.as_tensor(gt_oracle_cdr_mask, dtype=torch.bool)
        assert cdr_token_mask.numel() == len(data.tokens)
        cdr_atom_mask = cdr_token_mask[atom_to_token_tensor]

    # Compute frames for non-polymer tokens
    frame_data, resolved_frame_data = compute_frames_nonpolymer(
        data,
        coord_data,
        atom_data["is_present"],
        atom_to_token_tensor.numpy(),
        frame_data,
        resolved_frame_data,
    )
    frames = torch.from_numpy(frame_data.copy().astype(np.int64))
    frame_resolved_mask = torch.from_numpy(resolved_frame_data.copy().astype(np.float32))

    # One-hot encoding
    ref_atom_name_chars = one_hot(ref_atom_name_chars % num_bins, num_classes=num_bins).float()
    ref_element = one_hot(ref_element.clamp(0, num_elements - 1), num_classes=num_elements).float()
    num_tokens_actual = len(data.tokens)
    atom_to_token_onehot = one_hot(atom_to_token_tensor, num_classes=num_tokens_actual).float()
    token_to_rep_atom_onehot = one_hot(token_to_rep_atom_tensor, num_classes=len(atom_data)).float()

    # Center the ground truth coordinates
    center = (coords * resolved_mask[None, :, None]).sum(dim=1)
    center = center / resolved_mask.sum().clamp(min=1)
    coords = coords - center[:, None]
    if cdr_atom_mask is not None:
        ref_pos[cdr_atom_mask] = coords[0, cdr_atom_mask]

    # Apply random roto-translation to the input atoms
    ref_pos = center_random_augmentation(
        ref_pos[None], resolved_mask[None], centering=False
    )[0]
    if cdr_atom_mask is not None:
        ref_pos[cdr_atom_mask] = coords[0, cdr_atom_mask]

    # Compute padding for atoms
    if max_atoms is not None:
        pad_len = max_atoms - len(atom_data)
    else:
        pad_len = (
            (len(atom_data) - 1) // atoms_per_window_queries + 1
        ) * atoms_per_window_queries - len(atom_data)

    if pad_len > 0:
        pad_mask = pad_dim(pad_mask, 0, pad_len)
        ref_pos = pad_dim(ref_pos, 0, pad_len)
        resolved_mask = pad_dim(resolved_mask, 0, pad_len)
        ref_element = pad_dim(ref_element, 0, pad_len)
        ref_charge = pad_dim(ref_charge, 0, pad_len)
        ref_atom_name_chars = pad_dim(ref_atom_name_chars, 0, pad_len)
        ref_space_uid_tensor = pad_dim(ref_space_uid_tensor, 0, pad_len)
        coords = pad_dim(coords, 1, pad_len)
        atom_to_token_onehot = pad_dim(atom_to_token_onehot, 0, pad_len)
        token_to_rep_atom_onehot = pad_dim(token_to_rep_atom_onehot, 1, pad_len)

    if max_tokens is not None:
        token_pad_len = max_tokens - token_to_rep_atom_onehot.shape[0]
        if token_pad_len > 0:
            atom_to_token_onehot = pad_dim(atom_to_token_onehot, 1, token_pad_len)
            token_to_rep_atom_onehot = pad_dim(token_to_rep_atom_onehot, 0, token_pad_len)
            disto_target = pad_dim(pad_dim(disto_target, 0, token_pad_len), 1, token_pad_len)
            frames = pad_dim(frames, 0, token_pad_len)
            frame_resolved_mask = pad_dim(frame_resolved_mask, 0, token_pad_len)

    return {
        "ref_pos": ref_pos,
        "atom_resolved_mask": resolved_mask,
        "ref_element": ref_element,
        "ref_charge": ref_charge,
        "ref_atom_name_chars": ref_atom_name_chars,
        "ref_space_uid": ref_space_uid_tensor,
        "coords": coords,
        "atom_pad_mask": pad_mask,
        "atom_to_token": atom_to_token_onehot,
        "token_to_rep_atom": token_to_rep_atom_onehot,
        "disto_target": disto_target,
        "frames_idx": frames,
        "frame_resolved_mask": frame_resolved_mask,
    }


def process_msa_features(
    data: Tokenized,
    max_seqs_batch: int,
    max_seqs: int,
    max_tokens: Optional[int] = None,
    pad_to_max_seqs: bool = False,
) -> Dict[str, Tensor]:
    """Process MSA features.

    Parameters
    ----------
    data : Tokenized
        The tokenized data.
    max_seqs_batch : int
        Number of sequences for this batch.
    max_seqs : int
        Maximum number of sequences.
    max_tokens : int, optional
        Maximum number of tokens for padding.
    pad_to_max_seqs : bool
        Whether to pad to max_seqs. Default is False.

    Returns
    -------
    Dict[str, Tensor]
        Dictionary of MSA features.
    """
    # Construct paired MSA
    msa, deletion, paired = construct_paired_msa(data, max_seqs_batch)
    msa = msa.transpose(1, 0)  # [N_MSA, N_token]
    deletion = deletion.transpose(1, 0)
    paired = paired.transpose(1, 0)

    # Prepare features
    msa_onehot = one_hot(msa, num_classes=num_tokens).float()
    msa_mask = torch.ones_like(paired)
    profile = msa_onehot.mean(dim=0)
    has_deletion = (deletion > 0).float()
    deletion = math.pi / 2 * torch.arctan(deletion / 3)
    deletion_mean = deletion.mean(dim=0)

    # Pad in MSA dimension
    if pad_to_max_seqs:
        pad_len = max_seqs - msa_onehot.shape[0]
        if pad_len > 0:
            msa_onehot = pad_dim(msa_onehot, 0, pad_len, token_ids["-"])
            paired = pad_dim(paired, 0, pad_len)
            msa_mask = pad_dim(msa_mask, 0, pad_len)
            has_deletion = pad_dim(has_deletion, 0, pad_len)
            deletion = pad_dim(deletion, 0, pad_len)

    # Pad in token dimension
    if max_tokens is not None:
        token_pad_len = max_tokens - msa_onehot.shape[1]
        if token_pad_len > 0:
            msa_onehot = pad_dim(msa_onehot, 1, token_pad_len, token_ids["-"])
            paired = pad_dim(paired, 1, token_pad_len)
            msa_mask = pad_dim(msa_mask, 1, token_pad_len)
            has_deletion = pad_dim(has_deletion, 1, token_pad_len)
            deletion = pad_dim(deletion, 1, token_pad_len)
            profile = pad_dim(profile, 0, token_pad_len)
            deletion_mean = pad_dim(deletion_mean, 0, token_pad_len)

    return {
        "msa": msa_onehot,
        "msa_paired": paired,
        "deletion_value": deletion,
        "has_deletion": has_deletion,
        "deletion_mean": deletion_mean,
        "profile": profile,
        "msa_mask": msa_mask,
    }


####################################################################################################
# MAIN FEATURIZER CLASS
####################################################################################################


class BoltzFeaturizer:
    """Boltz featurizer - generates model features from tokenized data.

    This class combines token, atom, and MSA feature processing into
    a single interface for feature generation.
    """

    def process(
        self,
        data: Tokenized,
        training: bool = False,
        max_seqs: int = 4096,
        atoms_per_window_queries: int = 32,
        min_dist: float = 2.0,
        max_dist: float = 22.0,
        num_bins: int = 64,
        max_tokens: Optional[int] = None,
        max_atoms: Optional[int] = None,
        pad_to_max_seqs: bool = False,
        inference_binder: Optional[List[int]] = None,
        inference_pocket: Optional[List[Tuple[int, int]]] = None,
        # upstream Boltz1 compatibility parameters (unused but kept for API compatibility)
        symmetries: Optional[Dict] = None,
        compute_symmetries: bool = False,
        binder_pocket_conditioned_prop: Optional[float] = 0.0,
        binder_pocket_cutoff: Optional[float] = 6.0,
        binder_pocket_sampling_geometric_p: Optional[float] = 0.0,
        only_ligand_binder_pocket: Optional[bool] = False,
        gt_oracle_cdr_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, Tensor]:
        """Compute model features from tokenized data.

        Parameters
        ----------
        data : Tokenized
            The tokenized data.
        training : bool
            Whether in training mode. Default is False.
        max_seqs : int
            Maximum MSA sequences. Default is 4096.
        atoms_per_window_queries : int
            Atoms per attention window. Default is 32.
        min_dist : float
            Minimum distance for distogram. Default is 2.0.
        max_dist : float
            Maximum distance for distogram. Default is 22.0.
        num_bins : int
            Number of distogram bins. Default is 64.
        max_tokens : int, optional
            Maximum tokens for padding.
        max_atoms : int, optional
            Maximum atoms for padding.
        pad_to_max_seqs : bool
            Whether to pad to max_seqs. Default is False.
        inference_binder : List[int], optional
            Binder chain IDs for inference.
        inference_pocket : List[Tuple[int, int]], optional
            Pocket residue positions for inference.
        symmetries : Dict, optional
            Symmetry information (upstream Boltz1 compatibility, unused).
        compute_symmetries : bool
            Whether to compute symmetries (upstream Boltz1 compatibility, unused).
        binder_pocket_conditioned_prop : float, optional
            Binder pocket conditioning probability (upstream Boltz1 compatibility, unused).
        binder_pocket_cutoff : float, optional
            Binder pocket cutoff distance (upstream Boltz1 compatibility, unused).
        binder_pocket_sampling_geometric_p : float, optional
            Geometric sampling probability (upstream Boltz1 compatibility, unused).
        only_ligand_binder_pocket : bool, optional
            Only use ligand binder pocket (upstream Boltz1 compatibility, unused).
        gt_oracle_cdr_mask : np.ndarray, optional
            Token-level CDR mask for GT oracle ref_pos overlay.

        Returns
        -------
        Dict[str, Tensor]
            Dictionary of model features.
        """
        # Compute random number of sequences for training
        if training and max_seqs is not None:
            max_seqs_batch = np.random.randint(1, max_seqs + 1)
        else:
            max_seqs_batch = max_seqs

        # Compute token features
        token_features = process_token_features(
            data,
            max_tokens,
            inference_binder=inference_binder,
            inference_pocket=inference_pocket,
        )

        # Compute atom features
        atom_features = process_atom_features(
            data,
            atoms_per_window_queries,
            min_dist,
            max_dist,
            num_bins,
            max_atoms,
            max_tokens,
            gt_oracle_cdr_mask,
        )

        # Compute MSA features
        msa_features = process_msa_features(
            data,
            max_seqs_batch,
            max_seqs,
            max_tokens,
            pad_to_max_seqs,
        )

        # Add masked_seq for sequence prediction (copy of res_type indices)
        # This is used by the diffusion module during inference
        masked_seq = torch.from_numpy(data.tokens["res_type"].copy()).long()
        if max_tokens is not None:
            pad_len = max_tokens - len(masked_seq)
            if pad_len > 0:
                masked_seq = pad_dim(masked_seq, 0, pad_len)

        # Add attention mask (all True for valid tokens)
        attn_mask = torch.ones_like(masked_seq).bool()

        return {
            **token_features,
            **atom_features,
            **msa_features,
            "masked_seq": masked_seq,
            "attn_mask": attn_mask,
        }
