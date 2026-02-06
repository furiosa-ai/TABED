"""Evaluator utility functions for building models and components."""

from typing import Any, Dict, Tuple, Type

import torch
from absl import logging
from transformers import PreTrainedModel

from ..criterions import BaseCriterion
from ..modules import (
    AutoregressiveDecoding,
    SpeculativeDecoding,
    load_image_processor,
    load_pretrained_model,
    load_tokenizer,
)


def build_models(_config: Dict[str, Any]) -> Dict[str, PreTrainedModel]:
    """Build and initialize draft and target models.

    Args:
        _config: Configuration dictionary with model specifications.

    Returns:
        Dictionary mapping role names to loaded models.
    """
    logging.info("[Evaluator] Build models")
    models = {}

    for role in ['drf', 'tgt']:
        if _config[role] is None:
            continue
        model = load_pretrained_model(_config, role=role).cuda().eval()
        models[role] = model

    return models


def build_tokenizers(
    _config: Dict[str, Any],
) -> Tuple[Dict[str, Any], int, int]:
    """Build tokenizers for draft and target models.

    Args:
        _config: Configuration dictionary with tokenizer specifications.

    Returns:
        Tuple of (tokenizers dict, eos_token_id, pad_token_id).
    """
    logging.info("[Evaluator] Build tokenizers")
    tokenizers = {}
    eos_token_id = None
    pad_token_id = None

    for role in ['drf', 'tgt']:
        if _config[role] is None:
            continue
        tokenizers[role], eos_token_id, pad_token_id = load_tokenizer(
            _config,
            max_target_length=_config['max_target_length'],
            role=role,
        )

    return tokenizers, eos_token_id, pad_token_id


def build_image_processors(_config: Dict[str, Any]) -> Dict[str, Any]:
    """Build image processors for draft and target models.

    Args:
        _config: Configuration dictionary with processor specifications.

    Returns:
        Dictionary mapping role names to image processors.
    """
    logging.info("[Evaluator] Build image processors")
    image_processors = {}

    for role in ['drf', 'tgt']:
        if _config[role] is None:
            continue
        image_processors[role] = load_image_processor(_config, role=role)

    return image_processors


def get_decoding_class(_config: Dict[str, Any]) -> Type:
    """Get the appropriate decoding class based on configuration.

    Args:
        _config: Configuration dictionary with decoding type.

    Returns:
        Decoding class (AutoregressiveDecoding or SpeculativeDecoding).

    Raises:
        ValueError: If decoding type is invalid.
    """
    logging.info("[Evaluator] Build decoding")

    if _config['decoding'] == 'ard':
        return AutoregressiveDecoding
    elif _config['decoding'] == 'sd':
        return SpeculativeDecoding
    else:
        raise ValueError(f"Invalid decoding type: {_config['decoding']}")


def get_criterion(_config: Dict[str, Any]) -> BaseCriterion:
    """Create a criterion instance for metric tracking.

    Args:
        _config: Configuration dictionary.

    Returns:
        BaseCriterion instance.
    """
    return BaseCriterion(_config)


def warmup_generation(
    model: PreTrainedModel,
    tokenizer: Any,
    warmup_steps: int = 10,
) -> None:
    """Warmup model with dummy generation steps.

    Performs warmup generation to initialize CUDA kernels and optimize
    subsequent inference performance.

    Args:
        model: The model to warmup.
        tokenizer: Tokenizer for creating dummy inputs.
        warmup_steps: Number of warmup iterations.
    """
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt").to(model.device)
    logging.info(f"[Evaluator] Warmup steps: {warmup_steps}")

    for _ in range(warmup_steps):
        _ = model.generate(**inputs)

    torch.cuda.empty_cache()
