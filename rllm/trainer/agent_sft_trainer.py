import torch
from torch.distributed.device_mesh import init_device_mesh

from rllm.agents.agent import Trajectory
from rllm.parser.chat_template.parser import ChatTemplateParser
from verl.trainer.fsdp_sft_trainer import FSDPSFTTrainer
from verl.utils import hf_tokenizer
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.device import get_device_name
from verl.utils.distributed import initialize_global_process_group
from verl.utils.fs import copy_to_local


class RLLMMultiTurnSFTDataset(MultiTurnSFTDataset):
    """
    Dataset for multi-turn conversations using rllm chat template parser
    """

    def __init__(self, parquet_files, tokenizer, config=None):
        # Initialize the chat template parser
        self.chat_parser = ChatTemplateParser.get_parser(tokenizer, disable_thinking=False)
        print(f"Using chat parser: {type(self.chat_parser).__name__}")

        # Initialize verl's MultiTurnSFTDataset
        super().__init__(parquet_files, tokenizer, config)

    def __getitem__(self, item):
        tokenizer = self.tokenizer
        messages = self.messages[item]

        # Get the full conversation tokens using chat template parser
        full_text = self.chat_parser.parse(messages, add_generation_prompt=False, is_first_msg=True)
        tokens = tokenizer.encode(full_text, add_special_tokens=False)
        input_ids = torch.tensor(tokens, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        # Create loss mask by identifying assistant responses
        loss_mask = torch.zeros_like(input_ids, dtype=torch.long)

        # Process each message to find assistant responses
        for i, msg in enumerate(messages):
            # Get tokens for messages up to this point
            prefix_messages = messages[: i + 1]
            prefix_text = self.chat_parser.parse(prefix_messages, add_generation_prompt=False, is_first_msg=True)
            prefix_tokens = tokenizer.encode(prefix_text, add_special_tokens=False)

            # Get tokens for messages up to previous point
            if i > 0:
                prev_messages = messages[:i]
                prev_text = self.chat_parser.parse(prev_messages, add_generation_prompt=False, is_first_msg=True)
                prev_tokens = tokenizer.encode(prev_text, add_special_tokens=False)
                start_pos = len(prev_tokens)
            else:
                start_pos = 0

            end_pos = len(prefix_tokens)

            # If this is an assistant message, set loss mask
            if msg["role"] == "assistant":
                loss_mask[start_pos:end_pos] = 1

        # Handle sequence length
        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            # Pad sequences
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            padded_input_ids = torch.ones(size=(self.max_length - sequence_length,), dtype=input_ids.dtype) * pad_token_id
            padded_attention_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=attention_mask.dtype)
            padded_loss_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=loss_mask.dtype)

            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
            loss_mask = torch.cat((loss_mask, padded_loss_mask))
        elif sequence_length > self.max_length:
            if self.truncation == "left":
                input_ids = input_ids[-self.max_length :]
                attention_mask = attention_mask[-self.max_length :]
                loss_mask = loss_mask[-self.max_length :]
            elif self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
                loss_mask = loss_mask[: self.max_length]
            elif self.truncation == "error":
                raise ValueError(f"{sequence_length=} is larger than {self.max_length=}")
            else:
                raise ValueError(f"Unknown truncation method {self.truncation}")

        # Create position IDs
        position_ids = torch.arange(len(input_ids), dtype=torch.long)
        # Zero out position IDs for padding
        position_ids = position_ids * attention_mask

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }


class AgentSFTTrainer(FSDPSFTTrainer):
    def __init__(self, config):
        # Initialize distributed training
        device_name = get_device_name()
        local_rank, rank, world_size = initialize_global_process_group()
        device_mesh = init_device_mesh(device_type=device_name, mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
        dp_size = world_size // config.ulysses_sequence_parallel_size
        ulysses_device_mesh = init_device_mesh(device_type=device_name, mesh_shape=(dp_size, config.ulysses_sequence_parallel_size), mesh_dim_names=("dp", "sp"))

        # Build tokenizer and datasets
        local_model_path = copy_to_local(src=config.model.partial_pretrain, verbose=True)
        tokenizer = hf_tokenizer(local_model_path, trust_remote_code=config.model.trust_remote_code)

        # Use RLLM chat template parser
        train_dataset = RLLMMultiTurnSFTDataset(parquet_files=config.data.train_files, tokenizer=tokenizer, config=config.data)
        val_dataset = RLLMMultiTurnSFTDataset(parquet_files=config.data.val_files, tokenizer=tokenizer, config=config.data)

        # Initialize parent class
        super().__init__(config=config, device_mesh=device_mesh, ulysses_device_mesh=ulysses_device_mesh, tokenizer=tokenizer, train_dataset=train_dataset, val_dataset=val_dataset)

    @staticmethod
    def process_trajectories(trajectories: list[Trajectory], reward_threshold: float, filter_tool_calls: bool = False):
        """Process trajectories into SFT format."""
        sft_data = []

        for traj in trajectories:
            # Skip empty trajectories
            if not traj or not traj.steps:
                continue

            # Filter by reward threshold
            if traj.reward < reward_threshold:
                continue

            # Get messages from the last step
            last_step = traj.steps[-1]
            if not hasattr(last_step, "chat_completions") or not last_step.chat_completions:
                continue

            messages = last_step.chat_completions

            # Filter by tool calls
            if filter_tool_calls:
                has_tool_calls = any(msg.get("role") == "tool" for msg in messages)
                if not has_tool_calls:
                    continue

            # Clean and format messages
            clean_messages = []
            for msg in messages:
                if isinstance(msg, dict) and msg.get("role") and str(msg.get("content", "")).strip():
                    clean_messages.append({"role": msg["role"], "content": str(msg["content"]).strip()})

            # Need at least one user and assistant messages
            if len(clean_messages) >= 2:
                sft_data.append({"messages": clean_messages})

        print(f"Processed {len(trajectories)} trajectories -> {len(sft_data)} valid examples")
        return sft_data

    def train(self):
        """Start training."""
        self.fit()
