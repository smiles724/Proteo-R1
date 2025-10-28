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
        load_pretrained: bool = True,
        seq_token_id: int = None,
        struct_token_id: int = None,
        joint_projector: bool = False,
        freeze_choice: str = "none",
        vocab_size: int = None,
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
        self.load_pretrained = load_pretrained
        self.seq_token_id = seq_token_id
        self.struct_token_id = struct_token_id
        self.joint_projector = joint_projector
        self.freeze_choice = freeze_choice
        self.vocab_size = vocab_size


if __name__ == '__main__':
    config = PLLMConfig(base_model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct")
    config.save_pretrained("./test_config")
    loaded = PLLMConfig.from_pretrained("./test_config")
