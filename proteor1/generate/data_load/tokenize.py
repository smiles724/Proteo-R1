"""
BoltzTokenizer - Converts Structure to Token sequences.

This module implements the tokenization logic for converting raw protein
structure data into a sequence of tokens suitable for the Boltz model.

Core logic:
1. Iterate over valid chains
2. For each residue in the chain:
   - Standard residues -> 1 token
   - Non-standard residues -> 1 token per atom (UNK type)
3. Create token bonds from atom bonds and connections
"""
from abc import ABC, abstractmethod
from dataclasses import astuple, dataclass
from typing import Tuple

import numpy as np

from ..constants import token_ids, unk_token
from .types import Input, Token, TokenBond, Tokenized


@dataclass
class TokenData:
    """Intermediate token data representation.

    This dataclass holds token data before conversion to structured array.
    """
    token_idx: int
    atom_idx: int
    atom_num: int
    res_idx: int
    res_type: int
    sym_id: int
    asym_id: int
    entity_id: int
    mol_type: int
    center_idx: int
    disto_idx: int
    center_coords: np.ndarray
    disto_coords: np.ndarray
    resolved_mask: bool
    disto_mask: bool


class Tokenizer(ABC):
    """Tokenize an input structure for training."""

    @abstractmethod
    def tokenize(self, data: Input) -> Tokenized:
        """Tokenize the input data.

        Parameters
        ----------
        data : Input
            The input data.

        Returns
        -------
        Tokenized
            The tokenized data.

        """
        raise NotImplementedError


class BoltzTokenizer(Tokenizer):
    """Tokenize an input structure for training/inference.

    The tokenizer converts a Structure into a sequence of tokens where:
    - Standard residues become single tokens
    - Non-standard residues are tokenized per-atom with UNK type
    - Token bonds are derived from atom bonds and inter-chain connections
    """

    def tokenize(self, data: Input) -> Tuple[Tokenized, np.ndarray]:
        """Tokenize the input data.

        Parameters
        ----------
        data : Input
            The input data containing structure and MSA.

        Returns
        -------
        Tuple[Tokenized, np.ndarray]
            A tuple of:
            - tokenized: The tokenized data
            - token_mask: Boolean array indicating CDR tokens (True for CDR residues)
        """
        struct = data.structure

        # Create token data storage
        token_data = []
        token_mask = []
        token_idx = 0
        atom_to_token = {}

        # Filter to valid chains only
        chains = struct.chains[struct.mask]

        for chain in chains:
            # Get residue indices for this chain
            res_start = chain["res_idx"]
            res_end = chain["res_idx"] + chain["res_num"]

            for res in struct.residues[res_start:res_end]:
                # Get atom indices for this residue
                atom_start = res["atom_idx"]
                atom_end = res["atom_idx"] + res["atom_num"]

                # Standard residues are tokenized as single tokens
                if res["is_standard"]:
                    # Get center and disto atoms
                    center = struct.atoms[res["atom_center"]]
                    disto = struct.atoms[res["atom_disto"]]

                    # Token is present if residue and center atom are present
                    is_present = res["is_present"] & center["is_present"]
                    is_disto_present = res["is_present"] & disto["is_present"]

                    # Get coordinates
                    c_coords = center["coords"]
                    d_coords = disto["coords"]

                    # Create token
                    token = TokenData(
                        token_idx=token_idx,
                        atom_idx=res["atom_idx"],
                        atom_num=res["atom_num"],
                        res_idx=res["res_idx"],
                        res_type=res["res_type"],
                        sym_id=chain["sym_id"],
                        asym_id=chain["asym_id"],
                        entity_id=chain["entity_id"],
                        mol_type=chain["mol_type"],
                        center_idx=res["atom_center"],
                        disto_idx=res["atom_disto"],
                        center_coords=c_coords,
                        disto_coords=d_coords,
                        resolved_mask=is_present,
                        disto_mask=is_disto_present,
                    )
                    token_data.append(astuple(token))

                    # Track CDR mask
                    token_mask.append(res["is_cdr_residue"])

                    # Map all atoms in this residue to this token
                    for atom_idx in range(atom_start, atom_end):
                        atom_to_token[atom_idx] = token_idx

                    token_idx += 1

                # Non-standard residues are tokenized per atom
                else:
                    # Use UNK protein token as res_type
                    unk_token_name = unk_token["PROTEIN"]
                    unk_id = token_ids[unk_token_name]

                    # Get atom data
                    atom_data = struct.atoms[atom_start:atom_end]
                    atom_coords = atom_data["coords"]

                    # Tokenize each atom
                    for i, atom in enumerate(atom_data):
                        # Token is present if residue and atom are present
                        is_present = res["is_present"] & atom["is_present"]
                        index = atom_start + i

                        # Create token
                        token = TokenData(
                            token_idx=token_idx,
                            atom_idx=index,
                            atom_num=1,
                            res_idx=res["res_idx"],
                            res_type=unk_id,
                            sym_id=chain["sym_id"],
                            asym_id=chain["asym_id"],
                            entity_id=chain["entity_id"],
                            mol_type=chain["mol_type"],
                            center_idx=index,
                            disto_idx=index,
                            center_coords=atom_coords[i],
                            disto_coords=atom_coords[i],
                            resolved_mask=is_present,
                            disto_mask=is_present,
                        )
                        token_data.append(astuple(token))

                        # Track CDR mask from atom
                        token_mask.append(atom["is_cdr_atom"])

                        # Map this atom to this token
                        atom_to_token[index] = token_idx
                        token_idx += 1

        # Create token bonds
        token_bonds = []

        # Add bonds from atom-atom bonds (ligands, etc.)
        for bond in struct.bonds:
            atom1 = bond["atom_1"]
            atom2 = bond["atom_2"]
            if atom1 in atom_to_token and atom2 in atom_to_token:
                token_bond = (
                    atom_to_token[atom1],
                    atom_to_token[atom2],
                )
                token_bonds.append(token_bond)

        # Add bonds from inter-chain connections (covalent)
        for conn in struct.connections:
            atom1 = conn["atom_1"]
            atom2 = conn["atom_2"]
            if atom1 in atom_to_token and atom2 in atom_to_token:
                token_bond = (
                    atom_to_token[atom1],
                    atom_to_token[atom2],
                )
                token_bonds.append(token_bond)

        # Convert to numpy structured arrays
        token_mask = np.array(token_mask, dtype=bool)
        token_data = np.array(token_data, dtype=Token)
        token_bonds = np.array(token_bonds, dtype=TokenBond) if token_bonds else np.array([], dtype=TokenBond)

        tokenized = Tokenized(
            tokens=token_data,
            bonds=token_bonds,
            structure=data.structure,
            msa=data.msa,
        )

        return tokenized, token_mask
