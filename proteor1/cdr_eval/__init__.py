"""CDR evaluation helpers for antibody structure prediction workflows.

This package ships the data preparation, chain extraction, CDR masking,
batch inference, and CIF chain remapping utilities used by the ProteoR1
CDR evaluation flow.
"""

from .data_preparation import (
    DuplicateAnalysisResult,
    DuplicatePdbInfo,
    EntryInfo,
    ParseResult,
    ValidationResult,
    analyze_duplicate_pdb_ids,
    get_entry_stats,
    load_entries_for_split,
    parse_entry_string,
    parse_split_json,
    print_duplicate_analysis,
    validate_cif_existence,
)
from .chain_extractor import (
    BatchResult,
    ExtractionResult,
    apply_standard_filters,
    extract_chains_by_auth_asym_id,
    extract_chains_from_cif,
    get_chains_to_extract,
    process_entries_batch,
)
from .cdr_masking import (
    CDRMaskingResult,
    CDRRegionInfo,
    ChainCDRInfo,
    cif_to_protenix_json,
    extract_and_mask_cdr,
    filter_json_by_chains,
    find_entity_by_chain_id,
    find_matching_entity_by_sequence,
    get_cdr_indices_from_sequence,
    get_cdr_info_for_lddt,
    get_cdr_summary,
    get_chain_mapping_from_json,
    get_entry_chain_ids,
    mask_sequence,
    process_entry_cdr_masking,
    save_cdr_info_and_chain_mapping,
)
from .batch_inference import (
    BatchVerificationResult,
    InferenceResult,
    PredictionVerification,
    get_best_sample_info,
    get_best_sample_path,
    load_confidence_scores,
    run_batch_inference_cli,
    verify_prediction_outputs_batch,
    verify_single_entry_prediction,
)
from .remap_cif_chains import (
    RemapResult,
    load_chain_mapping,
    remap_batch,
    remap_cif_chain_ids,
    remap_single_entry,
)
from . import (
    batch_inference,
    cdr_masking,
    chain_extractor,
    data_preparation,
    remap_cif_chains,
)

__all__ = [
    # modules
    "batch_inference",
    "cdr_masking",
    "chain_extractor",
    "data_preparation",
    "remap_cif_chains",
    # data_preparation
    "DuplicateAnalysisResult",
    "DuplicatePdbInfo",
    "EntryInfo",
    "ParseResult",
    "ValidationResult",
    "analyze_duplicate_pdb_ids",
    "get_entry_stats",
    "load_entries_for_split",
    "parse_entry_string",
    "parse_split_json",
    "print_duplicate_analysis",
    "validate_cif_existence",
    # chain_extractor
    "BatchResult",
    "ExtractionResult",
    "apply_standard_filters",
    "extract_chains_by_auth_asym_id",
    "extract_chains_from_cif",
    "get_chains_to_extract",
    "process_entries_batch",
    # cdr_masking
    "CDRMaskingResult",
    "CDRRegionInfo",
    "ChainCDRInfo",
    "cif_to_protenix_json",
    "extract_and_mask_cdr",
    "filter_json_by_chains",
    "find_entity_by_chain_id",
    "find_matching_entity_by_sequence",
    "get_cdr_indices_from_sequence",
    "get_cdr_info_for_lddt",
    "get_cdr_summary",
    "get_chain_mapping_from_json",
    "get_entry_chain_ids",
    "mask_sequence",
    "process_entry_cdr_masking",
    "save_cdr_info_and_chain_mapping",
    # batch_inference
    "BatchVerificationResult",
    "InferenceResult",
    "PredictionVerification",
    "get_best_sample_info",
    "get_best_sample_path",
    "load_confidence_scores",
    "run_batch_inference_cli",
    "verify_prediction_outputs_batch",
    "verify_single_entry_prediction",
    # remap_cif_chains
    "RemapResult",
    "load_chain_mapping",
    "remap_batch",
    "remap_cif_chain_ids",
    "remap_single_entry",
]
