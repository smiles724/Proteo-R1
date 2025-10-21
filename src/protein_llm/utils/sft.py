from typing import Optional, Callable

from llamafactory.data.template import get_template_and_fix_tokenizer
from llamafactory.hparams import DataArguments




def create_sft_labels_with_masking(
    encoded_pairs: list[tuple[list[int], list[int]]],
    tokenizer_eos_token_id: int,
    ignore_index: int = -100,
    train_on_prompt: bool = False,
    mask_history: bool = False,
    efficient_eos: bool = False,
) -> tuple[list[int], list[int]]:
    """
    Create SFT training input_ids and labels from encoded message pairs, masking non-assistant parts.

    This function implements LLaMA-Factory's core masking logic.
    Source code reference: src/llamafactory/data/processor/supervised.py:33-86

    Note: train_on_prompt and mask_history cannot both be True (consistent with LLaMA-Factory)

    Args:
        encoded_pairs: List of encoded conversation pairs, each element is (source_ids, target_ids)
                      source_ids is the prompt part (user message), target_ids is the response part (assistant message)
                      Note: Each turn's source_ids should only contain the current turn's user message, not history
        tokenizer_eos_token_id: Tokenizer's eos_token_id
        ignore_index: Index value for masking, default -100 (PyTorch CrossEntropyLoss default)
        train_on_prompt: Whether to train on prompts, default False (i.e., mask out prompts)
        mask_history: Whether to only train on the last turn, default False
        efficient_eos: Whether to use efficient eos handling, default False

    Returns:
        (input_ids, labels): Training input_ids and corresponding labels

    Example:
        >>> # Assume two conversation turns
        >>> encoded_pairs = [
        ...     ([1, 2, 3], [4, 5, 6]),  # Turn 0: user1 -> assistant1
        ...     ([7, 8], [9, 10]),        # Turn 1: user2 -> assistant2
        ... ]
        >>> input_ids, labels = create_sft_labels_with_masking(
        ...     encoded_pairs, tokenizer_eos_token_id=2, ignore_index=-100
        ... )
        >>> # input_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> # labels    = [-100, -100, -100, 4, 5, 6, -100, -100, 9, 10]
    """
    # Parameter validation (consistent with LLaMA-Factory)
    if mask_history and train_on_prompt:
        raise ValueError("`mask_history` is incompatible with `train_on_prompt`.")

    input_ids = []
    labels = []

    if mask_history:
        encoded_pairs = encoded_pairs[::-1]  # Reverse order, prioritize last turn

    for turn_idx, (source_ids, target_ids) in enumerate(encoded_pairs):
        source_len = len(source_ids)
        target_len = len(target_ids)

        # Process source (prompt/user) labels
        if train_on_prompt:
            # Also compute loss on prompts
            source_label = source_ids
        elif efficient_eos and turn_idx != 0:
            # Keep first token as eos (for efficient_eos mode)
            source_label = [tokenizer_eos_token_id] + [ignore_index] * (source_len - 1)
        else:
            # Default: mask out all source tokens
            source_label = [ignore_index] * source_len

        # Process target (response/assistant) labels
        if mask_history and turn_idx != 0:
            # Only train on last turn (turn_idx == 0, since already reversed)
            target_label = [ignore_index] * target_len
        else:
            # Keep target labels, compute loss
            target_label = target_ids

        # Concatenate to input_ids and labels
        if mask_history:  # Reverse mode (concatenate from back to front)
            input_ids = source_ids + target_ids + input_ids
            labels = source_label + target_label + labels
        else:  # Normal order
            input_ids = input_ids + source_ids + target_ids
            labels = labels + source_label + target_label

    # Add ending eos token
    if efficient_eos:
        input_ids += [tokenizer_eos_token_id]
        labels += [tokenizer_eos_token_id]

    return input_ids, labels
