import torch


def create_pretraining_data_with_masking(
        tokenizer,
        user_text: str,
        assistant_text: str,
        add_eos: bool = True,
        ignore_index: int = -100,
):
    """
    Create pretraining-style data (no chat template) but mask user part.

    Args:
        tokenizer: Tokenizer instance
        user_text: User/prompt text (will be masked in labels)
        assistant_text: Assistant/response text (will compute loss)
        add_eos: Whether to add EOS token at the end
        ignore_index: Mask value for user part

    Returns:
        (input_ids, labels): user part is masked, assistant part computes loss
    """
    # Tokenize user and assistant separately
    user_ids = tokenizer(user_text, add_special_tokens=False)["input_ids"]
    assistant_ids = tokenizer(assistant_text, add_special_tokens=False)["input_ids"]

    # Add BOS if needed
    if getattr(tokenizer, "add_bos_token", False):
        user_ids = [tokenizer.bos_token_id] + user_ids

    # Add EOS if needed
    if add_eos:
        assistant_ids = assistant_ids + [tokenizer.eos_token_id]

    # Concatenate
    input_ids = user_ids + assistant_ids

    # Create labels: mask user part, keep assistant part
    labels = [ignore_index] * len(user_ids) + assistant_ids

    return torch.LongTensor(input_ids), torch.LongTensor(labels)
