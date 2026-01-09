"""Utilities for loading pretrained models, tokenizers, and image processors."""

import logging
import os
from typing import Any, Optional, Tuple, Type

import torch
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoProcessor,
    AutoTokenizer,
    Blip2ForConditionalGeneration,
    Blip2Processor,
    BlipForConditionalGeneration,
    BlipProcessor,
    LlavaForConditionalGeneration,
)


def load_pretrained_model(config: dict, role: str):
    """Load a pretrained model based on configuration.

    Args:
        config: Configuration dictionary containing model paths and settings.
        role: Model role ('drf', 'tgt', or 'captioning_model').

    Returns:
        Loaded model, potentially the language model component for text-only modes.
    """
    model_name = config[role]
    dtype_str = config[role + '_dtype']
    dtype = _get_dtype(dtype_str)

    ModelClass = _get_pretrained_classes(model_name, 'model')
    local_path = _get_checkpoint_path(config, model_name, ModelClass)
    model = _load_model(ModelClass, local_path, model_name, config, dtype)

    if role == 'captioning_model':
        pass
    elif role == 'drf' and config['is_drf_text_only'] and config['is_drf_from_mllm']:
        return model.language_model
    elif role == 'tgt' and config['is_tgt_text_only']:
        return model.language_model

    return model


def load_tokenizer(
    _config: dict,
    max_target_length: Optional[int],
    role: str = 'drf',
) -> Tuple[Any, Optional[int], Optional[int]]:
    """Load tokenizer and get special token IDs.

    Args:
        _config: Configuration dictionary.
        max_target_length: Maximum target length (unused, kept for compatibility).
        role: Model role ('drf', 'tgt', or 'captioning_model').

    Returns:
        Tuple of (tokenizer, eos_token_id, pad_token_id).
    """
    TokenizerClass = _get_pretrained_classes(_config[role], 'tokenizer')

    # Load tokenizer with appropriate settings
    if role == 'captioning_model' and 'lorence-2' in _config[role]:
        drf_tokenizer = TokenizerClass.from_pretrained(
            _config[role], trust_remote_code=True
        )
    else:
        drf_tokenizer = _load_tokenizer_for_role(_config, role, TokenizerClass)

    # Return early for non-MLLM draft models
    if (not _config['is_drf_from_mllm']) and role == 'drf':
        return drf_tokenizer, None, None

    eos_token_id, pad_token_id = _get_special_token_ids(drf_tokenizer, TokenizerClass)

    # Validate vocab size for speculative decoding
    if _config['decoding'] == 'sd' and role != 'captioning_model':
        tgt_tokenizer = _load_tokenizer_for_role(_config, 'tgt', TokenizerClass)
        _validate_vocab_size(drf_tokenizer, tgt_tokenizer, TokenizerClass)

    return drf_tokenizer, eos_token_id, pad_token_id


def load_image_processor(_config: dict, role: str):
    """Load image processor for a model.

    Args:
        _config: Configuration dictionary.
        role: Model role ('drf' or 'tgt').

    Returns:
        Image processor instance.
    """
    model_name = _config[role]
    ProcessorClass = _get_pretrained_classes(model_name, 'image_processor')

    if _config[role] == "llava-hf/llava-1.5-7b-hf":
        return ProcessorClass.from_pretrained(
            model_name, revision='a272c74'
        ).image_processor
    elif _config['tgt'] == "llava-hf/llava-1.5-13b-hf":
        return ProcessorClass.from_pretrained(
            _config['drf']
        ).image_processor
    else:
        return ProcessorClass.from_pretrained(model_name).image_processor


def _get_dtype(dtype_str: str) -> torch.dtype:
    """Convert dtype string to torch dtype.

    Args:
        dtype_str: String representation ('fp32', 'fp16', or 'bf16').

    Returns:
        Corresponding torch.dtype.
    """
    dtype_map = {
        'fp32': torch.float32,
        'fp16': torch.float16,
        'bf16': torch.bfloat16,
    }
    return dtype_map.get(dtype_str, torch.float32)


def _load_tokenizer_for_role(_config: dict, role: str, TokenizerClass: Type):
    """Load tokenizer with role-specific settings.

    Args:
        _config: Configuration dictionary.
        role: Model role.
        TokenizerClass: Tokenizer class to use.

    Returns:
        Loaded tokenizer instance.
    """
    model_name = _config[role]

    if model_name == "llava-hf/llava-1.5-7b-hf":
        return TokenizerClass.from_pretrained(model_name, revision='a272c74')
    elif _config['tgt'] == "llava-hf/llava-1.5-13b-hf":
        return TokenizerClass.from_pretrained(_config['drf'])
    else:
        return TokenizerClass.from_pretrained(model_name)


def _get_pretrained_classes(model_name: str, class_type: str) -> Type:
    """Get the appropriate class for a model and class type.

    Args:
        model_name: Name/path of the model.
        class_type: Type of class ('model', 'tokenizer', or 'image_processor').

    Returns:
        The appropriate class for the model.

    Raises:
        ValueError: If class_type is not supported for the model.
    """
    class_mapping = {
        'llava-hf': {
            'model': LlavaForConditionalGeneration,
            'tokenizer': AutoProcessor,
            'image_processor': AutoProcessor,
        },
        'mjbooo': {
            'model': LlavaForConditionalGeneration,
            'tokenizer': AutoProcessor,
            'image_processor': AutoProcessor,
        },
        'InternVL2': {
            'model': AutoModel,
            'tokenizer': AutoTokenizer,
            'image_processor': AutoProcessor,
        },
        'google/t5': {
            'model': AutoModelForSeq2SeqLM,
            'tokenizer': AutoTokenizer,
            'image_processor': None,
        },
        'JackFram/llama': {
            'model': AutoModelForCausalLM,
            'tokenizer': AutoProcessor,
            'image_processor': AutoProcessor,
        },
        'double7/vicuna': {
            'model': AutoModelForCausalLM,
            'tokenizer': AutoProcessor,
            'image_processor': AutoProcessor,
        },
        'Salesforce/blip2-opt': {
            'model': Blip2ForConditionalGeneration,
            'tokenizer': Blip2Processor,
            'image_processor': Blip2Processor,
        },
        'Salesforce/blip-': {
            'model': BlipForConditionalGeneration,
            'tokenizer': BlipProcessor,
            'image_processor': BlipProcessor,
        },
        'microsoft/Florence': {
            'model': AutoModelForCausalLM,
            'tokenizer': AutoProcessor,
            'image_processor': AutoProcessor,
        },
        'ljnlonoljpiljm/florence': {
            'model': AutoModelForCausalLM,
            'tokenizer': AutoProcessor,
            'image_processor': AutoProcessor,
        },
    }

    default_mapping = {
        'model': AutoModelForCausalLM,
        'tokenizer': AutoTokenizer,
        'image_processor': None,
    }

    # Find matching model mapping
    model_mapping = default_mapping
    for key in class_mapping:
        if key in model_name:
            model_mapping = class_mapping[key]
            break

    requested_class = model_mapping.get(class_type)

    if requested_class is None:
        raise ValueError(
            f"Class type '{class_type}' not supported for model '{model_name}'."
        )

    return requested_class


def _get_checkpoint_path(config: dict, model_name: str, ModelClass: Type) -> str:
    """Get the checkpoint path for a model.

    Args:
        config: Configuration dictionary.
        model_name: Name of the model.
        ModelClass: Model class for downloading if needed.

    Returns:
        Path to the model checkpoint.
    """
    if config['ckpt_dir'] is not None:
        logging.info(f"Loading selected model checkpoint from {config['ckpt_dir']}...")
        return f"{config['ckpt_dir']}/{model_name}"
    else:
        local_path = f"{config['root']}/data/MSD/checkpoint/{model_name}"
        if not os.path.exists(local_path) and "lorence-2" not in model_name:
            _download_and_save_model(ModelClass, model_name, local_path)
        return local_path


def _download_and_save_model(ModelClass: Type, model_name: str, local_path: str):
    """Download and save a model to local path.

    Args:
        ModelClass: Model class to use for loading.
        model_name: Name of the model to download.
        local_path: Path to save the model.
    """
    logging.info(f"No model found in local path! Downloading & Saving {model_name}...")
    model = ModelClass.from_pretrained(model_name)
    model.save_pretrained(local_path)


def _load_model(
    ModelClass: Type,
    local_path: str,
    model_name: str,
    _config: dict,
    _dtype: torch.dtype,
):
    """Load a model with appropriate settings.

    Args:
        ModelClass: Model class to use.
        local_path: Path to load from.
        model_name: Name of the model.
        _config: Configuration dictionary.
        _dtype: Data type for the model.

    Returns:
        Loaded model instance.

    Raises:
        ValueError: If model name is not supported.
    """
    if 'llava-hf' in model_name or 'mjbooo' in model_name:
        return ModelClass.from_pretrained(
            local_path,
            torch_dtype=_dtype,
            low_cpu_mem_usage=True,
        )
    elif 'Salesforce/blip2' in model_name:
        return ModelClass.from_pretrained(
            local_path,
            torch_dtype=torch.float16,
        )
    elif 'Salesforce/blip-' in model_name:
        return ModelClass.from_pretrained(
            local_path,
            torch_dtype=torch.float16,
        )
    elif 'lorence-2' in model_name:
        return ModelClass.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
    else:
        raise ValueError(f"Invalid model name: {model_name}")


def _get_special_token_ids(
    tokenizer,
    tokenizer_class: Type,
) -> Tuple[Optional[int], Optional[int]]:
    """Get EOS and PAD token IDs from tokenizer.

    Args:
        tokenizer: Tokenizer instance.
        tokenizer_class: Class of the tokenizer.

    Returns:
        Tuple of (eos_token_id, pad_token_id).
    """
    if tokenizer_class in [AutoProcessor, Blip2Processor, BlipProcessor]:
        eos_token_id = tokenizer.tokenizer.eos_token_id
        pad_token_id = tokenizer.tokenizer.pad_token_id
    else:
        eos_token_id = tokenizer.eos_token_id
        pad_token_id = tokenizer.pad_token_id

    return eos_token_id, pad_token_id


def _validate_vocab_size(drf_tokenizer, tgt_tokenizer, tokenizer_class: Type):
    """Validate that draft and target tokenizers have matching vocab sizes.

    Args:
        drf_tokenizer: Draft model tokenizer.
        tgt_tokenizer: Target model tokenizer.
        tokenizer_class: Class of the tokenizers.

    Raises:
        AssertionError: If vocab sizes don't match.
    """
    message = "Vocab mismatch between drf and tgt models"

    if tokenizer_class == AutoTokenizer:
        assert drf_tokenizer.vocab_size == tgt_tokenizer.vocab_size, message
    elif tokenizer_class == AutoProcessor:
        assert (
            drf_tokenizer.tokenizer.vocab_size == tgt_tokenizer.tokenizer.vocab_size
        ), message


