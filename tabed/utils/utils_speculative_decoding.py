"""Utility functions for speculative decoding operations."""

from typing import Any, Dict, List, Optional, Tuple

import torch


def init_outputs_dict(prompt_length: int, **kwargs) -> Dict[str, Any]:
    """Initialize the outputs dictionary for speculative decoding.

    Args:
        prompt_length: Length of the input prompt.
        **kwargs: Additional keyword arguments (unused).

    Returns:
        Dictionary with empty lists for metric accumulation.
    """
    return {
        'num_accepted_tokens': [],
        'num_generated_tokens': [],
        'num_prefill_tokens': None,
        'ids_accepted_tokens': [],
        'ids_first_rejected_tokens': [],
        'first_rejected_tokens': [],
        'prompt_length': prompt_length,
        'time_drf_generate': [],
        'time_tgt_forward': [],
        'time_spec_decode': None,
    }


def update_outputs_dict(
    outputs_dict: Dict[str, Any],
    n_matches: torch.Tensor,
    first_rejected_token: Optional[torch.Tensor],
    outputs_drf: Any,
    outputs_tgt: Any,
    batch: Dict[str, torch.Tensor],
) -> None:
    """Update outputs dictionary with results from one decoding step.

    Args:
        outputs_dict: Dictionary to update with new metrics.
        n_matches: Number of accepted tokens in this step.
        first_rejected_token: The first token that was rejected (if any).
        outputs_drf: Outputs from the draft model.
        outputs_tgt: Outputs from the target model.
        batch: Input batch containing input_ids.
    """
    outputs_dict['num_accepted_tokens'].append(n_matches.item())
    outputs_dict['num_generated_tokens'].append(len(outputs_drf.logits))

    if outputs_dict['num_prefill_tokens'] is None:
        kv_length = outputs_drf.past_key_values[0][0].shape[2]
        outputs_dict['num_prefill_tokens'] = kv_length - len(outputs_drf.logits) + 1

    input_length = batch['input_ids'].shape[1]
    outputs_dict['ids_accepted_tokens'].extend([
        i + input_length for i in range(n_matches)
    ])

    if first_rejected_token is not None:
        outputs_dict['ids_first_rejected_tokens'].append(
            input_length + n_matches.item()
        )
        outputs_dict['first_rejected_tokens'].append(first_rejected_token.item())

    outputs_dict['time_drf_generate'].append(outputs_drf['time_drf_generate'])
    outputs_dict['time_tgt_forward'].append(outputs_tgt['time_tgt_forward'])


def finalize_sd_outputs(
    outputs_dict: Dict[str, Any],
    batch: Any,
    tokenizer: Any,
    do_print: bool = False,
) -> None:
    """Finalize speculative decoding outputs.

    Args:
        outputs_dict: Dictionary containing accumulated metrics.
        batch: Batch object with input_ids.
        tokenizer: Tokenizer for decoding sequences.
        do_print: Whether to print output details.
    """
    outputs_dict['sequences'] = batch.input_ids.tolist()
    outputs_dict['num_target_tokens'] = (
        batch.input_ids.shape[1] - outputs_dict['prompt_length']
    )

    if do_print:
        print_sd_outputs(tokenizer, outputs_dict)


def get_model_kwargs() -> Dict[str, Any]:
    """Get default model kwargs for speculative decoding.

    Returns:
        Dictionary with default model keyword arguments.
    """
    return {
        'num_accepted_tokens': None,
        'past_key_values': None,
    }


def crop_past_key_values(
    model: Any,
    past_key_values: Tuple,
    maximum_length: int,
) -> Tuple:
    """Crop past key values to a maximum sequence length.

    Args:
        model: The model (used to check if encoder-decoder).
        past_key_values: Tuple of past key-value tensors.
        maximum_length: Maximum sequence length to keep.

    Returns:
        Cropped past key values tuple.
    """
    new_past = []

    if model.config.is_encoder_decoder:
        for idx in range(len(past_key_values)):
            new_past.append((
                past_key_values[idx][0][:, :, :maximum_length, :],
                past_key_values[idx][1][:, :, :maximum_length, :],
                past_key_values[idx][2],
                past_key_values[idx][3],
            ))
    else:
        for idx in range(len(past_key_values)):
            new_past.append((
                past_key_values[idx][0][:, :, :maximum_length, :],
                past_key_values[idx][1][:, :, :maximum_length, :],
            ))

    return tuple(new_past)


def print_sd_outputs(tokenizer: Any, outputs_dict: Dict[str, Any]) -> None:
    """Print speculative decoding outputs for debugging.

    Args:
        tokenizer: Tokenizer for decoding sequences.
        outputs_dict: Dictionary containing accumulated metrics.
    """
    sequences = outputs_dict['sequences']
    num_accepted_tokens = outputs_dict.get('num_accepted_tokens')
    num_generated_tokens = outputs_dict.get('num_generated_tokens')
    ids_accepted_tokens = outputs_dict.get('ids_accepted_tokens')

    print("#" * 50)
    decoded = tokenizer.batch_decode(sequences, skip_special_tokens=False)[0]
    print(f"Decoded sequence: {decoded}")
    print(f"Number of accepted tokens: {num_accepted_tokens}")
    print(f"Number of generated tokens: {num_generated_tokens}")
    print(f"Sum of total tokens: {len(sequences[0])}")
    print(f"Sum of accepted tokens: {sum(num_accepted_tokens)}")
    print(f"Sum of generated tokens: {sum(num_generated_tokens)}")
    print(f"Accepted token ids: {ids_accepted_tokens}")
    print("#" * 50)
