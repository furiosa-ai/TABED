"""TABED: Token-level Adaptive Block Efficiency Drafting for Speculative Decoding.

This package implements speculative decoding strategies for vision-language
models with various drafting modes including multimodal, text-only, caption,
and image-pool approaches.
"""

__version__ = "1.0.0"


def get_experiment():
    """Lazy load the Sacred experiment to avoid circular imports.

    Returns:
        Tuple of (ex, capture_config) from config module.
    """
    from .config import ex, capture_config
    return ex, capture_config


__all__ = ["get_experiment", "__version__"]
