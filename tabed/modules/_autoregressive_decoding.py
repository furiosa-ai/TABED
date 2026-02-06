"""Autoregressive decoding implementation."""

from typing import Any, Dict

from .decoding import Decoding
from .mllm import MLLM


class AutoregressiveDecoding(Decoding):
    """Autoregressive decoding module for standard token-by-token generation.

    Args:
        _config: Configuration dictionary.
        models: Dictionary containing 'drf' (draft) and 'tgt' (target) models.
        tokenizers: Dictionary of tokenizers for each model.
        image_processors: Dictionary of image processors.
        **kwargs: Additional arguments passed to parent class.
    """

    def __init__(
        self,
        _config: Dict,
        models: Dict[str, MLLM],
        tokenizers: Dict[str, Any],
        image_processors: Dict[str, Any],
        **kwargs,
    ):
        """Initialize autoregressive decoding module."""
        super().__init__(
            _config=_config,
            models=models,
            tokenizers=tokenizers,
            image_processors=image_processors,
            **kwargs,
        )

    def decode(self, batch: Dict, **kwargs) -> Dict:
        """Perform autoregressive decoding on the input batch.

        Args:
            batch: Input batch containing tokens and optionally pixel values.
            **kwargs: Additional decoding arguments.

        Returns:
            Generation outputs including sequences and timing metrics.
        """
        if self._config['is_drf_text_only']:
            batch.pop('pixel_values')

        outputs_generate = self.models['drf'].generate(
            **batch,
            **self.generate_config,
        )

        outputs_generate.metrics = {}
        outputs_generate.metrics['time_prefill_drf'] = outputs_generate.time_prefill
        outputs_generate.metrics['num_prefill_tokens_drf'] = (
            outputs_generate.num_prefill_tokens
        )

        return outputs_generate
