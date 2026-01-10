"""Utility modules for TABED experiments."""

from .util import (
    # Name mappings
    map_name_task,
    MODEL_NAME_MAP,
    # Name functions
    get_short_name,
    get_tabed_name,
    get_ckpt_name,
    # Token functions
    get_image_escape_token_num,
    get_caption_prefix_ids,
    get_pseudo_image_text_token_ids,
    # Utilities
    avg,
    set_seed,
    _save,
    noop_context,
    patch_function,
)

from .saver import Saver

from .utils_evaluator import (
    build_models,
    build_tokenizers,
    build_image_processors,
    get_decoding_class,
    get_criterion,
    warmup_generation,
)

from .utils_speculative_decoding import (
    init_outputs_dict,
    update_outputs_dict,
    finalize_sd_outputs,
    get_model_kwargs,
    crop_past_key_values,
    print_sd_outputs,
)

__all__ = [
    # Name mappings
    "map_name_task",
    "MODEL_NAME_MAP",
    # Name functions
    "get_short_name",
    "get_tabed_name",
    "get_ckpt_name",
    # Token functions
    "get_image_escape_token_num",
    "get_caption_prefix_ids",
    "get_pseudo_image_text_token_ids",
    # General utilities
    "avg",
    "set_seed",
    "_save",
    "noop_context",
    "patch_function",
    # Saver
    "Saver",
    # Evaluator utilities
    "build_models",
    "build_tokenizers",
    "build_image_processors",
    "get_decoding_class",
    "get_criterion",
    "warmup_generation",
    # Speculative decoding utilities
    "init_outputs_dict",
    "update_outputs_dict",
    "finalize_sd_outputs",
    "get_model_kwargs",
    "crop_past_key_values",
    "print_sd_outputs",
]
