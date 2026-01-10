"""Base decoding module for language model generation."""

from typing import Any, Dict

import torch
from absl import logging
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.llava.modeling_llava import LlavaForConditionalGeneration

from ..utils.util import (
    get_caption_prefix_ids,
    get_image_escape_token_num,
    get_pseudo_image_text_token_ids,
)
from .mllm import MLLM
from .promptsetter import PromptSetter
from .sample import _sample


class Decoding:
    """Base class for decoding strategies.

    Provides common functionality for autoregressive and speculative decoding,
    including model setup, tokenizer configuration, and batch processing.

    Args:
        _config: Configuration dictionary.
        models: Dictionary containing 'drf' (draft) and 'tgt' (target) models.
        tokenizers: Dictionary of tokenizers for each model.
        image_processors: Dictionary of image processors.
        **kwargs: Additional arguments including eos_token_id.
    """

    def __init__(
        self,
        _config: Dict,
        models: Dict[str, MLLM],
        tokenizers: Dict[str, Any],
        image_processors: Dict[str, Any],
        **kwargs,
    ):
        """Initialize the decoding module."""
        self._config = _config
        self.models = models
        self.tokenizers = tokenizers
        self.image_processors = image_processors

        device = self.models['drf'].device

        # Validate tokenization
        image_tokenized_ids = self.tokenizers['drf']("<image>").input_ids
        escape_tokenized_ids = self.tokenizers['drf']("\n").input_ids

        num_image_str_tokenized, num_escape_tokenized = get_image_escape_token_num(
            _config['drf']
        )

        assert image_tokenized_ids.size(1) == num_image_str_tokenized, (
            f"Tokenizing '<image>' should result in {num_image_str_tokenized} tokens"
        )
        assert escape_tokenized_ids.size(1) == num_escape_tokenized, (
            f"Tokenizing '\\n' should result in {num_escape_tokenized} tokens"
        )

        caption_prefix_ids = get_caption_prefix_ids(_config['drf']).to(device)

        self.image_token_id = image_tokenized_ids[0, -1].item()
        self.escape_token_id = escape_tokenized_ids[0, -1].item()
        self.pseudo_image_text_token_ids = get_pseudo_image_text_token_ids(
            _config['drf']
        ).to(device)

        prompt_kwargs = {
            'image_token_id': self.image_token_id,
            'escape_token_id': self.escape_token_id,
            'pseudo_image_text_token_ids': self.pseudo_image_text_token_ids,
            'caption_prefix_ids': caption_prefix_ids,
            'device': device,
        }

        self.prompt_setter = PromptSetter(
            _config,
            tokenizer=self.tokenizers['drf'],
            **prompt_kwargs,
        )
        self.models['drf'].prompt_setter = self.prompt_setter
        self.models['drf']._config = _config
        self.models['tgt']._config = _config

        self.generate_config = {
            'do_sample': (_config['temperature'] == 1),
            'use_cache': True,
            'max_new_tokens': _config['max_target_length'],
            'return_dict_in_generate': True,
            'pad_token_id': kwargs['eos_token_id'],
        }

        # Patch sampling methods
        LlamaForCausalLM._sample = _sample
        LlavaForConditionalGeneration._sample = _sample

        logging.info(f"[Decoding] image_token_id: {self.image_token_id}")
        logging.info(f"[Decoding] escape_token_id: {self.escape_token_id}")

    def load_batch_to_device(self, batch: Dict) -> Dict:
        """Load batch tensors to the appropriate device.

        Args:
            batch: Dictionary of batch data.

        Returns:
            Batch with tensors moved to device.
        """
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.models['drf'].device)

            if k == 'pixel_values':
                if (
                    'lvlm160m-bf16' in self._config['drf']
                    and self._config['drf_dtype'] == 'bf16'
                ):
                    batch[k] = batch[k].bfloat16()

        return batch

    def decode(self, **kwargs) -> Dict:
        """Decode input to generate output.

        Must be implemented by subclasses.

        Args:
            **kwargs: Decoding arguments.

        Returns:
            Dictionary containing generation outputs.

        Raises:
            NotImplementedError: This method must be overridden.
        """
        raise NotImplementedError
