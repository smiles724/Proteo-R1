"""
Utility functions for upstream Boltz1 data loading.

This module provides helper functions for data loading and preprocessing
that are compatible with upstream Boltz1's PredictionDataset.
"""

from pathlib import Path
from typing import Optional

import numpy as np
from numpy import ndarray

from .types import (
    MSA,
    Input,
    Record,
    Structure,
)


def load_input(
    record: Record,
    target_dir: Path,
    msa_dir: Path,
    no_msa: bool = False,
) -> Input:
    """Load the input data for a given record.

    This function is compatible with upstream Boltz1's load_input() from
    boltz/data/module/training.py and boltz/data/module/inference.py.

    Parameters
    ----------
    record : Record
        The record to load.
    target_dir : Path
        The path to the target directory containing structure NPZ files.
        For upstream Boltz1 inference, this should be the directory containing
        the NPZ files directly (not structures/ subdirectory).
    msa_dir : Path
        The path to the MSA directory.
    no_msa : bool
        If True, skip loading MSA data.

    Returns
    -------
    Input
        The loaded input data.
    """
    # Load the structure
    # Note: upstream Boltz1 inference uses target_dir/{record.id}.npz directly
    # while upstream Boltz1 training uses target_dir/structures/{record.id}.npz
    structure_path = target_dir / f"{record.id}.npz"
    if not structure_path.exists():
        # Try the training path format
        structure_path = target_dir / "structures" / f"{record.id}.npz"

    structure_data = np.load(structure_path)
    structure = Structure(
        atoms=structure_data["atoms"],
        bonds=structure_data["bonds"],
        residues=structure_data["residues"],
        chains=structure_data["chains"],
        connections=structure_data["connections"],
        interfaces=structure_data["interfaces"],
        mask=structure_data["mask"],
    )

    msas = {}
    if no_msa:
        return Input(structure, msas)

    for chain in record.chains:
        msa_id = chain.msa_id
        # Load the MSA for this chain, if any
        if msa_id != -1:
            msa_path = msa_dir / f"{msa_id}.npz"
            if msa_path.exists():
                msa_data = np.load(msa_path)
                msas[chain.chain_id] = MSA(**msa_data)

    return Input(structure, msas)


def ab_region_type(
    token: ndarray,
    spec_mask: ndarray,
    chain_id: int,
) -> ndarray:
    """Get antibody region labels for a chain.

    This function assigns region labels to each token based on the spec_mask.
    The regions are labeled as follows:
        FR1:  1
        CDR1: 2
        FR2:  3
        CDR2: 4
        FR3:  5
        CDR3: 6
        FR4:  7

    Parameters
    ----------
    token : ndarray
        The tokenized data array with 'asym_id' field.
    spec_mask : ndarray
        The spec mask indicating CDR regions (1) vs framework regions (0).
    chain_id : int
        The chain ID to process.

    Returns
    -------
    ndarray
        Array of region labels (same length as spec_mask).
    """
    indices = [i for i, x in enumerate(token) if x["asym_id"] == chain_id]
    masks = spec_mask[indices]

    # Compute segment IDs based on transitions in the mask
    diff = np.diff(masks.astype(int))
    diff = np.concatenate(([1], diff))
    segment_ids = np.cumsum(diff != 0)

    label = np.zeros_like(spec_mask, dtype=int)
    label[indices] = segment_ids

    return label


def ag_region_type(
    token: ndarray,
    spec_mask: ndarray,
    ab_chain_ids: list,
    add_epitope: bool = True,
) -> ndarray:
    """Get antigen region labels.

    This function assigns region labels to antigen residues:
        non-epitope: 8
        epitope: 9

    Parameters
    ----------
    token : ndarray
        The tokenized data array with 'asym_id' field.
    spec_mask : ndarray
        The spec mask indicating epitope regions (1) vs non-epitope (0).
    ab_chain_ids : list
        List of antibody chain IDs to exclude.
    add_epitope : bool
        If True, distinguish epitope (9) from non-epitope (8).
        If False, all antigen residues get label 8.

    Returns
    -------
    ndarray
        Array of region labels (same length as spec_mask).
    """
    indices = [i for i, x in enumerate(token) if x["asym_id"] not in ab_chain_ids]
    masks = spec_mask[indices]

    label = np.zeros_like(spec_mask, dtype=int)
    if add_epitope:
        label[indices] = masks + 8
    else:
        label[indices] = 8

    return label
