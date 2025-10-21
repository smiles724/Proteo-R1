from transformers import PretrainedConfig

class PLLMConfig(PretrainedConfig):
    model_type = "pllm"

    def __init__(
        self,
        base_model_name_or_path: str = None,
        protein_config: str = None,
        structure_config: str = None,
        protrek_ckpt: str = None,
        prot_slot: int = 1,
        stru_slot: int = 3,
        single_token_prefix: bool = False,
        prefix_len: int = 4,
        proj_hid: int = 1024,
        dropout: float = 0.1,
        train_encoders: bool = True,
        load_pretrained: bool = True,
        protein_token_id: int = None,
        structure_token_id: int = None,
        # NEW: chain-aware special tokens (ALL are LLM-tokenizer IDs)
        chain_bos_id: int = None,
        chain_eos_id: int = None,
        seq_bos_id: int = None,
        seq_eos_id: int = None,
        struct_bos_id: int = None,
        struct_eos_id: int = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.base_model_name_or_path = base_model_name_or_path
        self.protein_config = protein_config
        self.structure_config = structure_config
        self.protrek_ckpt = protrek_ckpt
        self.prot_slot = prot_slot
        self.stru_slot = stru_slot
        self.single_token_prefix = single_token_prefix
        self.prefix_len = prefix_len
        self.proj_hid = proj_hid
        self.dropout = dropout
        self.train_encoders = train_encoders
        self.load_pretrained = load_pretrained
        self.protein_token_id = protein_token_id
        self.structure_token_id = structure_token_id
        # NEW: chain/seq/struct LLM token ids
        self.chain_bos_id = chain_bos_id
        self.chain_eos_id = chain_eos_id
        self.seq_bos_id = seq_bos_id
        self.seq_eos_id = seq_eos_id
        self.struct_bos_id = struct_bos_id
        self.struct_eos_id = struct_eos_id


if __name__ == '__main__':
    config = PLLMConfig(base_model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct")
    config.save_pretrained("./test_config")
    loaded = PLLMConfig.from_pretrained("./test_config")
