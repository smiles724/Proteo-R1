"""
Constants for Boltz model.

Defines token vocabularies, chain types, and other constants used
throughout the model for feature encoding and decoding.

This module is designed to be compatible with upstream's boltz.data.const.
"""

from typing import Dict, List

####################################################################################################
# TOKEN DEFINITIONS (reference-impl compatible)
####################################################################################################

# Token vocabulary (matches upstream)
tokens = [
    "<pad>",
    "-",
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "UNK",  # unknown protein token (idx 22)
    "A",
    "G",
    "C",
    "U",
    "N",  # unknown rna token
    "DA",
    "DG",
    "DC",
    "DT",
    "DN",  # unknown dna token
]

token_ids = {token: i for i, token in enumerate(tokens)}
num_tokens = len(tokens)

# Unknown token mappings
unk_token = {"PROTEIN": "UNK", "DNA": "DN", "RNA": "N"}
unk_token_ids = {m: token_ids[t] for m, t in unk_token.items()}

# Special token indices
PAD_TOKEN = 0
GAP_TOKEN = 1
UNK_TOKEN = 22

# Standard amino acid vocabulary (1-letter codes)
# NOTE: This follows alphabetical order for AA_VOCAB, but the actual token indices
# in the tokens list follow a different order (ALA=2, ARG=3, ASN=4, ...).
# For correct encoding/decoding, use token_ids and prot_token_to_letter instead.
AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"

# Correct token index to amino acid mapping based on tokens list order:
# tokens = ["<pad>", "-", "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
#           "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
#           "TYR", "VAL", "UNK", ...]
IDX_TO_AA = {
    0: "",    # <pad>
    1: "-",   # gap
    2: "A",   # ALA
    3: "R",   # ARG
    4: "N",   # ASN
    5: "D",   # ASP
    6: "C",   # CYS
    7: "Q",   # GLN
    8: "E",   # GLU
    9: "G",   # GLY
    10: "H",  # HIS
    11: "I",  # ILE
    12: "L",  # LEU
    13: "K",  # LYS
    14: "M",  # MET
    15: "F",  # PHE
    16: "P",  # PRO
    17: "S",  # SER
    18: "T",  # THR
    19: "W",  # TRP
    20: "Y",  # TYR
    21: "V",  # VAL
    22: "X",  # UNK
}
AA_TO_IDX = {v: k for k, v in IDX_TO_AA.items() if v and v not in ("", "-", "X")}

# Protein letter to token mapping
prot_letter_to_token = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "E": "GLU",
    "Q": "GLN",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
    "X": "UNK",
    "J": "UNK",
    "B": "UNK",
    "Z": "UNK",
    "O": "UNK",
    "U": "UNK",
    "-": "-",
}

# Reverse mapping: token to letter (for sequence decoding)
prot_token_to_letter = {v: k for k, v in prot_letter_to_token.items()}
prot_token_to_letter["UNK"] = "X"
prot_token_to_letter["<pad>"] = ""

####################################################################################################
# CHAIN TYPES
####################################################################################################

chain_types = [
    "PROTEIN",
    "DNA",
    "RNA",
    "NONPOLYMER",
]
chain_type_ids: Dict[str, int] = {chain: i for i, chain in enumerate(chain_types)}

# Pocket contact information for antibody design
pocket_contact_info: Dict[str, int] = {
    "UNSPECIFIED": 0,
    "UNSELECTED": 1,
    "POCKET": 2,
    "BINDER": 3,
}

# Output types for metrics
out_types: List[str] = [
    "protein",
    "dna",
    "rna",
    "ligand",
    "complex",
]

####################################################################################################
# MODEL DIMENSIONS
####################################################################################################

DEFAULT_ATOM_S = 64
DEFAULT_ATOM_Z = 64
DEFAULT_TOKEN_S = 384
DEFAULT_TOKEN_Z = 128
DEFAULT_NUM_BINS = 64

# Default feature dimensions
ATOM_FEATURE_DIM = 128 + 3 + 1 + 1 + 128 + 64 * 4  # ref_pos + charge + mask + element + atom_name_chars

# Reference atom per residue type (for computing center of mass)
# Maps residue type index to list of reference atom indices
REFERENCE_ATOMS = {
    # Amino acids typically use CA (alpha carbon) as reference
    # This is simplified - actual implementation may need more detail
}

####################################################################################################
# DIFFUSION CONSTANTS (EDM schedule)
####################################################################################################

SIGMA_DATA = 16.0
SIGMA_MIN = 0.0004
SIGMA_MAX = 160.0
P_MEAN = -1.2
P_STD = 1.5  # reference value from upstream
RHO = 7.0

####################################################################################################
# ATOMS (from upstream)
####################################################################################################

num_elements = 128

# Reference atoms for each residue type (for frame computation)
# fmt: off
ref_atoms = {
    "PAD": [],
    "UNK": ["N", "CA", "C", "O", "CB"],
    "-": [],
    "ALA": ["N", "CA", "C", "O", "CB"],
    "ARG": ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
    "ASN": ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"],
    "ASP": ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"],
    "CYS": ["N", "CA", "C", "O", "CB", "SG"],
    "GLN": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"],
    "GLU": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"],
    "GLY": ["N", "CA", "C", "O"],
    "HIS": ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"],
    "ILE": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"],
    "LEU": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"],
    "LYS": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"],
    "MET": ["N", "CA", "C", "O", "CB", "CG", "SD", "CE"],
    "PHE": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD"],
    "SER": ["N", "CA", "C", "O", "CB", "OG"],
    "THR": ["N", "CA", "C", "O", "CB", "OG1", "CG2"],
    "TRP": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
    "TYR": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
    "VAL": ["N", "CA", "C", "O", "CB", "CG1", "CG2"],
    "A": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"],
    "G": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"],
    "C": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"],
    "U": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6"],
    "N": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'"],
    "DA": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"],
    "DG": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"],
    "DC": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"],
    "DT": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "N1", "C2", "O2", "N3", "C4", "O4", "C5", "C7", "C6"],
    "DN": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'"]
}
# fmt: on

# Maps residue to center atom name
res_to_center_atom = {
    "UNK": "CA",
    "ALA": "CA",
    "ARG": "CA",
    "ASN": "CA",
    "ASP": "CA",
    "CYS": "CA",
    "GLN": "CA",
    "GLU": "CA",
    "GLY": "CA",
    "HIS": "CA",
    "ILE": "CA",
    "LEU": "CA",
    "LYS": "CA",
    "MET": "CA",
    "PHE": "CA",
    "PRO": "CA",
    "SER": "CA",
    "THR": "CA",
    "TRP": "CA",
    "TYR": "CA",
    "VAL": "CA",
    "A": "C1'",
    "G": "C1'",
    "C": "C1'",
    "U": "C1'",
    "N": "C1'",
    "DA": "C1'",
    "DG": "C1'",
    "DC": "C1'",
    "DT": "C1'",
    "DN": "C1'"
}

# Maps residue to disto atom name
res_to_disto_atom = {
    "UNK": "CB",
    "ALA": "CB",
    "ARG": "CB",
    "ASN": "CB",
    "ASP": "CB",
    "CYS": "CB",
    "GLN": "CB",
    "GLU": "CB",
    "GLY": "CA",
    "HIS": "CB",
    "ILE": "CB",
    "LEU": "CB",
    "LYS": "CB",
    "MET": "CB",
    "PHE": "CB",
    "PRO": "CB",
    "SER": "CB",
    "THR": "CB",
    "TRP": "CB",
    "TYR": "CB",
    "VAL": "CB",
    "A": "C4",
    "G": "C4",
    "C": "C2",
    "U": "C2",
    "N": "C1'",
    "DA": "C4",
    "DG": "C4",
    "DC": "C2",
    "DT": "C2",
    "DN": "C1'"
}

####################################################################################################
# COMPUTED ATOM INDICES
####################################################################################################

# Maps residue name to center atom INDEX in ref_atoms list
res_to_center_atom_id = {
    res: ref_atoms[res].index(atom)
    for res, atom in res_to_center_atom.items()
    if res in ref_atoms and atom in ref_atoms[res]
}

# Maps residue name to disto atom INDEX in ref_atoms list
res_to_disto_atom_id = {
    res: ref_atoms[res].index(atom)
    for res, atom in res_to_disto_atom.items()
    if res in ref_atoms and atom in ref_atoms[res]
}

####################################################################################################
# CHIRALITY
####################################################################################################

chirality_types = [
    "CHI_UNSPECIFIED",
    "CHI_TETRAHEDRAL_CW",
    "CHI_TETRAHEDRAL_CCW",
    "CHI_OTHER",
]
chirality_type_ids = {chirality: i for i, chirality in enumerate(chirality_types)}
unk_chirality_type = "CHI_UNSPECIFIED"

####################################################################################################
# BONDS
####################################################################################################

bond_types = [
    "OTHER",
    "SINGLE",
    "DOUBLE",
    "TRIPLE",
    "AROMATIC",
]
bond_type_ids = {bond: i for i, bond in enumerate(bond_types)}
unk_bond_type = "OTHER"

####################################################################################################
# MSA CONSTANTS
####################################################################################################

max_msa_seqs = 16384
max_paired_seqs = 8192

####################################################################################################
# TRAINING DEFAULTS
####################################################################################################

DEFAULT_RECYCLING_STEPS = 3
DEFAULT_SAMPLING_STEPS = 200
DEFAULT_DIFFUSION_SAMPLES = 5
