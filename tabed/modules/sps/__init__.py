"""Speculative decoding implementation for vision-language models."""

from .candidate_generator_vlmsd import (
    AssistedCandidateGeneratorVLM,
    AssistedCandidateGeneratorTabedVLM,
)
from .decoding import _assisted_decoding
from .utils import _get_candidate_generator_vlm, _validate_assistant_vlm
from .modeling_llava_vlmsd import (
    forward,
    prepare_inputs_for_generation,
    _get_initial_cache_position,
    _update_model_kwargs_for_generation,
)

__all__ = [
    # Candidate generators
    "AssistedCandidateGeneratorVLM",
    "AssistedCandidateGeneratorTabedVLM",
    # Decoding
    "_assisted_decoding",
    # Utilities
    "_get_candidate_generator_vlm",
    "_validate_assistant_vlm",
    # Model modifications
    "forward",
    "prepare_inputs_for_generation",
    "_get_initial_cache_position",
    "_update_model_kwargs_for_generation",
]
