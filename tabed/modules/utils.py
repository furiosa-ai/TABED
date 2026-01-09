"""Utility classes and dataclasses for generation outputs."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from transformers.utils import ModelOutput


@dataclass
class GenerateDecoderOnlyOutput(ModelOutput):
    """Outputs of decoder-only generation models using non-beam methods.

    Args:
        sequences: Generated token sequences of shape (batch_size, sequence_length).
            The second dimension is either equal to max_length or shorter if all
            batches finished early due to eos_token_id.
        scores: Processed prediction scores of the language modeling head
            (scores for each vocabulary token before SoftMax) at each generation
            step. Tuple of torch.FloatTensor with up to max_new_tokens elements,
            each of shape (batch_size, config.vocab_size).
        logits: Unprocessed prediction scores of the language modeling head
            at each generation step. Tuple of torch.FloatTensor with up to
            max_new_tokens elements, each of shape (batch_size, config.vocab_size).
        attentions: Tuple (one element for each generated token) of tuples
            (one element for each layer) of torch.FloatTensor of shape
            (batch_size, num_heads, generated_length, sequence_length).
        hidden_states: Tuple (one element for each generated token) of tuples
            (one element for each layer) of torch.FloatTensor of shape
            (batch_size, generated_length, hidden_size).
        past_key_values: Tuple (one element for each layer) of tuples
            (two elements, key and value tensors). Each tensor has shape
            (batch_size, num_heads, sequence_length, embed_size_per_head).
        time_prefill: Time taken for prefill phase in seconds.
        num_prefill_tokens: Number of tokens processed during prefill.
        metrics: Dictionary of additional metrics.
        per_token_decoding_latency: Average latency per token in seconds.
    """

    sequences: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[Tuple[Tuple[Tuple[torch.FloatTensor]]]] = None
    time_prefill: Optional[float] = None
    num_prefill_tokens: Optional[int] = None
    metrics: Optional[Dict[str, Any]] = None
    per_token_decoding_latency: Optional[float] = None
