"""
ProteoR1Understand Configuration

Based on the PLLMConfig design but simplified to a single Protenix encoder architecture.

Key changes:
1. Removed protein_config, structure_config, protrek_ckpt, prot_slot, stru_slot
2. Added protenix_encoder_path (pretrained directory of the Protenix encoder)
3. Simplified to a single protein_token_id (replacing seq_token_id + struct_token_id)
4. Updated projector input dim (384 for Protenix s, optional 2560 for ESM)
5. Built on Qwen3 (requires transformers>=4.51.0)
6. Added protein_start_token_id, protein_end_token_id for boundary markers
7. Added Position Embedding config (use_protenix_pos_embed, max_residues, max_chains)
"""
from typing import Optional

from transformers import PretrainedConfig


class ProteoR1UnderstandConfig(PretrainedConfig):
    """
    Configuration for the ProteoR1Understand model.

    Args:
        base_model_name_or_path: HuggingFace name or local path of the backbone LLM.
            Defaults to Qwen3-4B-Instruct-2507.
        protenix_encoder_path: pretrained directory of the Protenix encoder.
            Expected layout:
                protenix_encoder_path/
                ├── config.json
                ├── protenix_mini_ism_v0.5.0.pt
                └── esm2_t36_3B_UR50D_ism.pt
        load_esm: whether to load the ESM encoder; default True.
        protenix_s_dim: Protenix s embedding dimensionality; default 384.
        esm_embedding_dim: ESM embedding dimensionality; default 2560.
        use_esm_embedding: whether to use the ESM embedding (in addition to Protenix s); default False.
            Backwards-compatible bool: True == "esm", False == "s".
        embedding_mode: embedding selection mode; default None (decided by use_esm_embedding).
            - "s": Protenix s embedding only (384)
            - "a": Protenix a_token embedding only (768)
            - "esm": ESM embedding only (2560)
            - "esm+s": ESM + s concatenated (2944)
            - "esm+a": ESM + a_token concatenated (3328)
        proj_hid: projector hidden dimensionality; default None (use the LLM hidden_size).
        dropout: projector dropout probability.
        protein_token_id: id of the protein placeholder token (set at runtime by the processor).
        protein_start_token_id: id of the protein-region start marker token (set at runtime by the processor).
        protein_end_token_id: id of the protein-region end marker token (set at runtime by the processor).
        load_pretrained: whether to load LLM weights from the pretrained checkpoint.
        freeze_choice: freezing strategy.
            - "none": freeze nothing
            - "encoder": freeze the Protenix encoder
            - "non_projector": train the projector only
        vocab_size: vocabulary size (synced at runtime from the LLM config, or used when restoring from a checkpoint).
        use_protenix_pos_embed: whether to enable the Protenix position embedding (hierarchical absolute positions).
        max_residues: maximum residues per chain (used by the position embedding).
        max_chains: maximum number of chains (used by the position embedding).
        use_cdr_mask_embedding: whether to enable the CDR mask embedding (used to mask CDR regions).
            When True, the model registers a frozen mask_embedding parameter and replaces the embedding
            at CDR positions with mask_embedding during forward.

    Note:
        - protein_token_id, protein_start_token_id, protein_end_token_id and vocab_size are None at __init__.
        - The token IDs are set externally after the processor adds special tokens.
        - vocab_size is synced from the LLM config at model init time, ensuring correct serialization after
          resize_token_embeddings.
    """

    model_type = "proteor1_understand"

    def __init__(
        self,
        # LLM configuration
        base_model_name_or_path: str = "Qwen/Qwen3-4B-Instruct-2507",
        load_pretrained: bool = True,
        vocab_size: int = None,  # synced from the LLM at runtime
        # Protenix encoder configuration
        protenix_encoder_path: str = None,
        load_esm: bool = True,
        protenix_s_dim: int = 384,
        esm_embedding_dim: int = 2560,
        protenix_a_dim: int = 768,
        use_esm_embedding: bool = False,
        embedding_mode: str = None,  # "s", "esm", or "concat"
        triangle_by_torch: Optional[bool] = None,
        # Projector configuration
        proj_hid: int = None,  # None = use the LLM hidden_size
        dropout: float = 0.0,
        # Token configuration (set at runtime)
        protein_token_id: int = None,  # set at runtime by the processor
        protein_start_token_id: int = None,  # set at runtime by the processor
        protein_end_token_id: int = None,  # set at runtime by the processor
        protein_token: str = "<protein>",
        protein_start_token: str = "<protein_start>",
        protein_end_token: str = "<protein_end>",
        # Position Embedding configuration
        use_protenix_pos_embed: bool = False,  # whether to enable the Protenix position embedding
        max_residues: int = 4096,  # maximum residues per chain
        max_chains: int = 64,  # maximum number of chains
        # CDR Mask Embedding configuration
        use_cdr_mask_embedding: bool = False,  # whether to enable the CDR mask embedding
        # Training configuration
        freeze_choice: str = "none",
        **kwargs
    ):
        super().__init__(**kwargs)

        # LLM configuration
        self.base_model_name_or_path = base_model_name_or_path
        self.load_pretrained = load_pretrained
        self.vocab_size = vocab_size

        # Protenix encoder configuration
        self.protenix_encoder_path = protenix_encoder_path
        self.load_esm = load_esm
        self.protenix_s_dim = protenix_s_dim
        self.esm_embedding_dim = esm_embedding_dim
        self.protenix_a_dim = protenix_a_dim
        self.use_esm_embedding = use_esm_embedding
        self.embedding_mode = embedding_mode
        self.triangle_by_torch = triangle_by_torch

        # Projector configuration
        self.proj_hid = proj_hid
        self.dropout = dropout

        # Token configuration
        self.protein_token_id = protein_token_id
        self.protein_start_token_id = protein_start_token_id
        self.protein_end_token_id = protein_end_token_id
        self.protein_token = protein_token
        self.protein_start_token = protein_start_token
        self.protein_end_token = protein_end_token

        # Position Embedding configuration
        self.use_protenix_pos_embed = use_protenix_pos_embed
        self.max_residues = max_residues
        self.max_chains = max_chains

        # CDR Mask Embedding configuration
        self.use_cdr_mask_embedding = use_cdr_mask_embedding

        # Training configuration
        self.freeze_choice = freeze_choice

    def get_embedding_mode(self) -> str:
        """
        Return the embedding mode. Prefer embedding_mode; otherwise derive from use_esm_embedding for compatibility.

        Returns:
            "s": Protenix s embedding only (384)
            "a": Protenix a_token embedding only (768)
            "esm": ESM embedding only (2560)
            "esm+s": ESM + s concatenated (2944)
            "esm+a": ESM + a_token concatenated (3328)
        """
        if self.embedding_mode is not None:
            return self.embedding_mode
        # Backwards compatibility: use_esm_embedding bool fallback.
        return "esm" if self.use_esm_embedding else "s"

    @property
    def projector_input_dim(self) -> int:
        """
        Compute the projector's input dimensionality.

        - "s": protenix_s_dim = 384
        - "a": protenix_a_dim = 768
        - "esm": esm_embedding_dim = 2560
        - "esm+s": esm_embedding_dim + protenix_s_dim = 2944
        - "esm+a": esm_embedding_dim + protenix_a_dim = 3328
        """
        mode = self.get_embedding_mode()
        if mode == "esm":
            return self.esm_embedding_dim
        elif mode == "s":
            return self.protenix_s_dim
        elif mode == "a":
            return self.protenix_a_dim
        elif mode == "esm+s":
            return self.esm_embedding_dim + self.protenix_s_dim
        elif mode == "esm+a":
            return self.esm_embedding_dim + self.protenix_a_dim
        else:
            raise ValueError(mode)


if __name__ == '__main__':
    import shutil

    # Smoke-test config creation and save.
    # Note: protein_token_id is set by the processor in real usage; here it is only for testing.
    config = ProteoR1UnderstandConfig(
        base_model_name_or_path="Qwen/Qwen3-4B-Instruct-2507",
        protenix_encoder_path="pretrained/protenix_mini_ism_v0.5.0",
        proj_hid=1024,
        dropout=0.1,
        # Leave protein_token_id and vocab_size as None to mirror the real flow.
    )

    test_dir = "temp/test_protenix_qwen_config"
    shutil.rmtree(test_dir, ignore_errors=True)
    config.save_pretrained(test_dir)
    print(f"Saved config to {test_dir}")

    # Test loading.
    loaded = ProteoR1UnderstandConfig.from_pretrained(test_dir)
    print(f"Loaded config:")
    print(f"  base_model_name_or_path: {loaded.base_model_name_or_path}")
    print(f"  protenix_encoder_path: {loaded.protenix_encoder_path}")
    print(f"  projector_input_dim: {loaded.projector_input_dim}")
    print(f"  protein_token_id: {loaded.protein_token_id} (None = to be set by processor)")
    print(f"  vocab_size: {loaded.vocab_size} (None = to be synced from LLM)")

    # Simulate runtime setup.
    loaded.protein_token_id = 151936  # pretend the processor assigned this ID after adding the token
    loaded.vocab_size = 151937  # pretend vocab_size after resize
    print(f"\nAfter runtime setup:")
    print(f"  protein_token_id: {loaded.protein_token_id}")
    print(f"  vocab_size: {loaded.vocab_size}")

    # Cleanup.
    shutil.rmtree(test_dir, ignore_errors=True)
    print("\nTest passed!")
