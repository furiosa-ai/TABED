"""Utility functions for speculative decoding with VLM models."""

from typing import Dict

import torch
from transformers.generation.candidate_generator import CandidateGenerator
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList
from transformers.modeling_utils import PreTrainedModel

from .candidate_generator_vlmsd import (
    AssistedCandidateGeneratorTabedVLM,
    AssistedCandidateGeneratorVLM,
)


def _get_candidate_generator_vlm(
    self,
    generation_config: GenerationConfig,
    input_ids: torch.LongTensor,
    inputs_tensor: torch.Tensor,
    assistant_model: PreTrainedModel,
    logits_processor: LogitsProcessorList,
    model_kwargs: Dict,
) -> CandidateGenerator:
    """Get the candidate generator for assisted generation.

    Creates either a single-mode or TABED candidate generator based on
    the drafting configuration.

    Args:
        generation_config: Generation configuration parameters.
        input_ids: Input token IDs.
        inputs_tensor: Input tensor for the model.
        assistant_model: The draft/assistant model.
        logits_processor: List of logits processors.
        model_kwargs: Additional model keyword arguments.

    Returns:
        CandidateGenerator instance appropriate for the drafting mode.
    """
    try:
        config = getattr(self, '_config', None)
        if config is None:
            from ...config import capture_config
            config = capture_config()
    except Exception:
        config = getattr(assistant_model, '_config', {})

    drafting = assistant_model._config['drafting']

    if isinstance(drafting, list):
        candidate_generator = AssistedCandidateGeneratorTabedVLM(
            input_ids=input_ids,
            assistant_model=assistant_model,
            generation_config=generation_config,
            model_kwargs=model_kwargs,
            inputs_tensor=inputs_tensor,
            logits_processor=logits_processor,
            prompt_setter=assistant_model.prompt_setter,
            config=config,
        )
    else:
        candidate_generator = AssistedCandidateGeneratorVLM(
            input_ids=input_ids,
            assistant_model=assistant_model,
            generation_config=generation_config,
            model_kwargs=model_kwargs,
            inputs_tensor=inputs_tensor,
            logits_processor=logits_processor,
            prompt_setter=assistant_model.prompt_setter,
            config=config,
        )

    return candidate_generator


def _validate_assistant_vlm(self, assistant_model: PreTrainedModel) -> None:
    """Validate that the assistant model is compatible with the main model.

    Checks encoder compatibility for encoder-decoder models and validates
    that both models use the same tokenizer vocabulary.

    Args:
        assistant_model: The draft/assistant model to validate.

    Raises:
        ValueError: If models are incompatible.
    """
    if assistant_model is None:
        return

    if self.config.is_encoder_decoder and not assistant_model.config.is_encoder_decoder:
        attributes_to_check = [
            "encoder_attention_heads",
            "encoder_ffn_dim",
            "encoder_layers",
        ]
        attributes_to_check = [
            attr for attr in dir(assistant_model.config)
            if attr in attributes_to_check
        ]
        are_equal = all(
            getattr(self.config, attr) == getattr(assistant_model.config, attr)
            for attr in attributes_to_check
        )
        if not are_equal:
            raise ValueError(
                "The main model and the assistant don't have compatible "
                "encoder-dependent input shapes. Ensure you load the assistant "
                "with the correct encoder-decoder class, e.g. "
                "`AutoModelForSpeechSeq2Seq` for Whisper."
            )

    if hasattr(assistant_model.config, "vocab_size"):
        # Text-only drafting
        if self.config.text_config.vocab_size != assistant_model.config.vocab_size:
            raise ValueError(
                "Make sure the main and assistant model use the same tokenizer"
            )
    else:
        # Multimodal drafting
        if (
            self.config.text_config.vocab_size
            != assistant_model.config.text_config.vocab_size
        ):
            raise ValueError(
                "Make sure the main and assistant model use the same tokenizer"
            )
