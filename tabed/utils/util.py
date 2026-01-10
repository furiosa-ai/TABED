"""Utility functions and mappings for TABED experiments."""

import os
import random
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.multiprocessing as mp
import yaml


# =============================================================================
# Model and Dataset Name Mappings
# =============================================================================

MODEL_NAME_MAP = {
    # Target models (LLaVA)
    "llava-hf/llava-1.5-7b-hf": "llava-llama-7b",
    "llava-hf/llava-1.5-13b-hf": "llava-llama-13b",
    "llava-hf/llava-v1.6-vicuna-7b-hf": "llava-vicuna-7b",
    "llava-hf/llava-v1.6-vicuna-13b-hf": "llava-vicuna-13b",

    # Draft models
    "mjbooo/lvlm68m": "llava-68m",
    "mjbooo/lvlm160m-bf16": "llava-160m",
    "mjbooo/lvlm290m": "llava-290m",
    "mjbooo/lvlm68m-ov": "llava-68m-ov",

    # Captioning models
    "microsoft/Florence-2-large-ft": "florence2-0.77b",
    "microsoft/Florence-2-large": "florence2-0.77b-noft",

    # Caption types
    "<CAPTION>": "C",
    "<DETAILED_CAPTION>": "DC",
    "<MORE_DETAILED_CAPTION>": "MDC",
    "<OCR>": "OCR",
}


# Combined map for backward compatibility
map_name_task = {**MODEL_NAME_MAP}


# =============================================================================
# Name Resolution Functions
# =============================================================================

def get_short_name(name_obj: str) -> str:
    """Get the short name for a model or dataset.

    Args:
        name_obj: Full name of the model or dataset.

    Returns:
        Short name if mapping exists, otherwise the original name.
    """
    if name_obj not in map_name_task:
        return name_obj

    mapped = map_name_task[name_obj]
    if isinstance(mapped, tuple):
        return mapped[0]
    return mapped


def get_tabed_name(_config: Dict[str, Any]) -> str:
    """Generate a descriptive name for TABED drafting configuration.

    Args:
        _config: Configuration dictionary.

    Returns:
        Hyphen-separated string describing the TABED configuration.
    """
    sorted_list = sorted(_config['drafting'])

    # Handle caption drafting
    if 'caption' in _config['drafting']:
        idx = sorted_list.index('caption')
        caption_name = f"caption-{get_short_name(_config['captioning_model'])}"
        if 'lorence-2' in _config['captioning_model']:
            caption_name += f"-{get_short_name(_config['caption_type'])}"
        sorted_list[idx] = caption_name

    # Handle image-pool drafting
    if 'image-pool' in _config['drafting']:
        idx = sorted_list.index('image-pool')
        pool_name = (
            f"image-pool-{_config['target_dim_image_pooling']}d-"
            f"{_config['image_pool_type']}"
        )
        sorted_list[idx] = pool_name

    # Build TABED options
    if _config['tabed_rule'] == 'mm-weight':
        tabed_options = [_config['tabed_rule'], str(_config['mm_weight_policy'])]
    elif _config['tabed_rule'] in ['conf-weight', 'confidence']:
        tabed_options = [_config['tabed_rule'], str(_config['confidence_type'])]
    else:
        tabed_options = [_config['tabed_rule']]

    # Add history-dependent options
    if _config['history_dependent']:
        tabed_options += [
            'history',
            _config['history_item'],
            _config['history_unit'],
            str(_config['history_window']),
        ]

        if 'adaboost' in _config['history_item']:
            tabed_options.append(
                f"C{str(_config['history_adaboost_constant_weight'])}"
            )
        elif 'w-grid' in _config['history_item']:
            tabed_options.append(
                f"grid-{str(_config['history_w_num_grid'])}-"
                f"{str(_config['history_w_grid_measure'])}"
            )

    if _config['mm_weight_k'] is not None:
        tabed_options.append(f"{_config['mm_weight_k']}x")

    tabed_options.append(f"tdw{_config['temperature_drafting_weight']}")
    tabed_options.append('tabed')

    return '-'.join(sorted_list + tabed_options)


def get_ckpt_name(
    _config: Dict[str, Any],
    is_phase_2: bool = True,
    dataset: Optional[str] = None,
) -> str:
    """Generate a checkpoint name based on configuration.

    Args:
        _config: Configuration dictionary.
        is_phase_2: Whether this is phase 2 training (unused).
        dataset: Optional dataset name override.

    Returns:
        Underscore-separated checkpoint name string.
    """
    ckpt_dataset = dataset or _config['dataset']

    # Build drafting name
    if isinstance(_config['drafting'], list):
        name_drafting = get_tabed_name(_config)
    elif _config['drafting'] == 'caption':
        name_drafting = f"{_config['drafting']}-{get_short_name(_config['captioning_model'])}"
        if 'lorence-2' in _config['captioning_model']:
            name_drafting += f"-{get_short_name(_config['caption_type'])}"
    elif _config['drafting'] == 'image-pool':
        name_drafting = (
            f"{_config['drafting']}-{_config['target_dim_image_pooling']}d-"
            f"{_config['image_pool_type']}"
        )
    else:
        name_drafting = _config['drafting']

    if _config['image_top_k_attention']:
        name_drafting += f"-top-{_config['image_top_k_attention']}"

    name_drafting += '-drafting'

    # Build dataset name with multi-turn suffix
    dataset_name = get_short_name(ckpt_dataset)
    if _config['multi_turn_task']:
        dataset_name += f"-{_config['multi_turn_task']}"

    # Build dtype string
    dtype_prefix = 'fp' if 'fp' in _config['drf_dtype'] else 'bf'
    dtype_str = f"{dtype_prefix}{_config['drf_dtype'][-2:]}-{_config['tgt_dtype'][-2:]}"

    factors = [
        _config['decoding'],
        get_short_name(_config['drf']),
        get_short_name(_config['tgt']),
        name_drafting,
        dataset_name,
        f"mtl-{_config['max_target_length']}",
        f"gamma-{_config['max_chunk_length']}",
        f"t{int(_config['temperature'])}",
        dtype_str,
        _config['seed'],
    ]

    if _config['is_time_factorized']:
        factors.insert(-1, 'time-factorized')
    if _config['is_tgt_text_only']:
        factors.insert(-1, 'text-verify')
    if _config['tiny_data']:
        factors.append('tiny_data')

    return "_".join(map(str, factors))


# =============================================================================
# Token ID Functions
# =============================================================================

def get_image_escape_token_num(model_name: str) -> Tuple[int, int]:
    """Get the number of tokens for image and escape sequences.

    Args:
        model_name: Name of the model.

    Returns:
        Tuple of (image_token_count, escape_token_count).

    Raises:
        NotImplementedError: If model is not supported.
    """
    if any(x in model_name for x in ["mjbooo/lvlm", "llava-hf/llava"]):
        return 2, 3
    else:
        raise NotImplementedError(
            f"get_image_escape_token_num not implemented for {model_name}"
        )


def get_caption_prefix_ids(model_name: str) -> torch.LongTensor:
    """Get caption prefix token IDs for a model.

    Args:
        model_name: Name of the model.

    Returns:
        LongTensor of caption prefix token IDs.

    Raises:
        NotImplementedError: If model is not supported.
    """
    if any(x in model_name for x in ["mjbooo/lvlm", "llava-hf/llava"]):
        return torch.LongTensor([1967, 29901, 29871])
    else:
        raise NotImplementedError(
            f"get_caption_prefix_ids not implemented for {model_name}"
        )


def get_pseudo_image_text_token_ids(model_name: str) -> torch.LongTensor:
    """Get pseudo image text token IDs for a model.

    Args:
        model_name: Name of the model.

    Returns:
        LongTensor of pseudo image text token IDs.

    Raises:
        NotImplementedError: If model is not supported.
    """
    if any(x in model_name for x in ["mjbooo/lvlm", "llava-hf/llava"]):
        return torch.LongTensor([529, 3027, 29958])
    else:
        raise NotImplementedError(
            f"get_pseudo_image_text_token_ids not implemented for {model_name}"
        )


# =============================================================================
# Utility Functions
# =============================================================================

def avg(values: List[float]) -> float:
    """Calculate average of a list of values.

    Args:
        values: List of numeric values.

    Returns:
        Average value.
    """
    return sum(values) / len(values)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    mp.set_sharing_strategy('file_system')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _save(
    model: Any,
    optimizer: Any,
    lr_scheduler: Any,
    metric: Any,
    save_dir: str,
    config: Dict[str, Any],
) -> None:
    """Save model, optimizer, scheduler, and config to disk.

    Args:
        model: Model to save.
        optimizer: Optimizer to save.
        lr_scheduler: Learning rate scheduler to save.
        metric: Metric tracker to save.
        save_dir: Directory to save to.
        config: Experiment configuration.
    """
    os.makedirs(save_dir, exist_ok=True)
    state_dict = model.state_dict()

    # Save model
    model.save_pretrained(save_dir, state_dict=state_dict, safe_serialization=True)

    # Save optimizer and scheduler
    torch.save({
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': lr_scheduler.state_dict(),
    }, os.path.join(save_dir, "optimizers.pt"))

    # Save metric
    torch.save(metric.state_dict(), os.path.join(save_dir, "metric.pt"))

    # Save config
    with open(os.path.join(save_dir, 'config_sacred.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


# =============================================================================
# Context Managers
# =============================================================================

@contextmanager
def noop_context():
    """A no-operation context manager."""
    yield


@contextmanager
def patch_function(
    target_object: Any,
    function_name: str,
    custom_function: Any,
    *args,
    **kwargs,
):
    """Context manager to temporarily patch a function.

    Args:
        target_object: Object or module containing the function.
        function_name: Name of the function to patch.
        custom_function: Replacement function.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments passed to custom function.

    Yields:
        None
    """
    original_function = getattr(target_object, function_name)
    custom_function.kwargs = kwargs
    setattr(target_object, function_name, custom_function)

    try:
        yield
    finally:
        setattr(target_object, function_name, original_function)
