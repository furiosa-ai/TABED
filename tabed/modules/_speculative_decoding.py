"""Speculative decoding implementation for accelerated inference."""

from typing import Any, Dict

from transformers import GenerationMixin
from transformers.models.llava.modeling_llava import LlavaForConditionalGeneration

from .decoding import Decoding
from .eval_specbench import measure_time
from .mllm import MLLM
from .sps.decoding import _assisted_decoding
from .sps.modeling_llava_vlmsd import (
    _get_initial_cache_position,
    _update_model_kwargs_for_generation,
    forward,
    prepare_inputs_for_generation,
)
from .sps.utils import _get_candidate_generator_vlm, _validate_assistant_vlm


class SpeculativeDecoding(Decoding):
    """Speculative decoding module for accelerated inference.

    Uses a smaller draft model to generate candidate tokens that are then
    verified by the larger target model.

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
        """Initialize speculative decoding module."""
        super().__init__(
            _config=_config,
            models=models,
            tokenizers=tokenizers,
            image_processors=image_processors,
            **kwargs,
        )

        self.generate_config['num_assistant_tokens'] = _config['max_chunk_length']

        # Patch model methods for speculative decoding
        LlavaForConditionalGeneration.forward = forward
        LlavaForConditionalGeneration.prepare_inputs_for_generation = (
            prepare_inputs_for_generation
        )
        LlavaForConditionalGeneration._get_initial_cache_position = (
            _get_initial_cache_position
        )
        LlavaForConditionalGeneration._update_model_kwargs_for_generation = (
            _update_model_kwargs_for_generation
        )
        GenerationMixin._assisted_decoding = _assisted_decoding
        GenerationMixin._get_candidate_generator = _get_candidate_generator_vlm
        GenerationMixin._validate_assistant = _validate_assistant_vlm

    def decode(self, batch: Dict, **kwargs) -> Dict:
        """Perform speculative decoding on the input batch.

        Args:
            batch: Input batch containing tokens and pixel values.
            **kwargs: Additional decoding arguments.

        Returns:
            Generation outputs including sequences, timing metrics,
            and acceptance statistics.
        """
        _, time_prompt_process = measure_time(self.prompt_setter.set_batch, batch)

        outputs_generate = self.models['tgt'].generate(
            **batch,
            **self.generate_config,
            assistant_model=self.models['drf'],
        )

        if outputs_generate.metrics['time_prompt_process'] is None:
            assert self.prompt_setter.drafting not in ['multimodal', 'image-pool']
            outputs_generate.metrics['time_prompt_process'] = time_prompt_process

        return outputs_generate
