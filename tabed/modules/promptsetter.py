"""Prompt manipulation for different drafting strategies in speculative decoding."""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from absl import logging
from einops import rearrange

from .load_pretrained import (
    load_image_processor,
    load_pretrained_model,
    load_tokenizer,
)


class PromptSetter:
    """Handles prompt manipulation for various drafting strategies.

    Supports multimodal, text-only, caption-based, and image-pooling
    strategies for speculative decoding.

    Args:
        _config: Configuration dictionary containing drafting settings.
        tokenizer: Tokenizer for text processing.
        **kwargs: Additional arguments including image_token_id, escape_token_id,
            pseudo_image_text_token_ids, caption_prefix_ids, and device.
    """

    def __init__(self, _config: Dict, tokenizer, **kwargs):
        """Initialize the PromptSetter with configuration."""
        self._config = _config
        self.drafting = _config.get('drafting')

        # Validate drafting mode configuration
        if self.drafting in ['text-only', 'tokenized-image', 'special-token', 'caption']:
            assert _config['is_drf_text_only'], (
                "tokenized-image mode requires DRF to be text-only."
            )
        else:
            assert not _config['is_drf_text_only'], (
                "Non-text-only mode requires DRF to be multimodal."
            )

        self.tokenizer = tokenizer
        self.device = kwargs.get('device')
        self.image_token_id = kwargs.get('image_token_id')
        self.escape_token_id = kwargs.get('escape_token_id')
        self.pseudo_image_text_token_ids = kwargs.get('pseudo_image_text_token_ids')
        self.caption_prefix_ids = kwargs.get('caption_prefix_ids')

        if 'caption' in _config['drafting']:
            self.captioning_model, self.captioning_processor = (
                self.load_captioning_model()
            )

        self.target_dim = _config.get('target_dim_image_pooling')

        # Initialize batch-specific properties
        self.input_ids = None
        self.input_ids_length_initial = None
        self.attention_mask_initial = None
        self.image_ids = None
        self.manipulated_input_ids = None
        self.manipulated_input_ids_length_initial = None

    def load_captioning_model(self) -> Tuple[Any, Any]:
        """Load the captioning model and processor.

        Returns:
            Tuple of (captioning_model, captioning_processor).
        """
        logging.info("[PromptSetter] Build captioning model and processor ...")
        model = load_pretrained_model(self._config, 'captioning_model').to(self.device)
        processor, _, _ = load_tokenizer(self._config, None, 'captioning_model')
        return model, processor

    def set_batch(self, batch: Dict):
        """Set the input data from the provided batch.

        Processes the batch to extract input_ids, attention_mask, and image
        positions, then applies the configured drafting strategy.

        Args:
            batch: Dictionary containing 'input_ids', 'attention_mask', and
                optional image-related data.
        """
        self.input_ids = batch['input_ids']
        self.input_ids_length_initial = self.input_ids.size(1)

        # Reset manipulated_input_ids
        self.manipulated_input_ids = None
        self.manipulated_input_ids_length_initial = None

        # Create image_mask based on image_token_id
        self.image_ids = (batch['input_ids'][0] == self.image_token_id).nonzero()

        # Save pixel values for image tokens
        self.pixel_values_caption = batch.pop('pixel_values_caption', None)
        self.input_ids_caption = batch.pop('input_ids_caption', None)

        (
            self.manipulated_input_ids,
            self.manipulated_input_ids_length_initial,
            self.attention_mask_initial,
        ) = self._process_prompt(self.drafting)

    def get_resulting_input(self, new_input_ids: torch.Tensor) -> torch.Tensor:
        """Get the resulting input by combining manipulated prompt with new tokens.

        Args:
            new_input_ids: New input IDs including generated tokens.

        Returns:
            Concatenated tensor of manipulated prompt and new remainder.
        """
        new_input_ids_remainder = new_input_ids[:, self.input_ids_length_initial:]
        return torch.cat((self.manipulated_input_ids, new_input_ids_remainder), dim=1)

    def replace_image_tokens(
        self,
        drafting: str,
        value: Optional[Union[int, List, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Replace image tokens in input_ids based on drafting strategy.

        Args:
            drafting: The drafting mode ('text-only', 'tokenized-image', 'caption').
            value: Replacement value(s) for image tokens.

        Returns:
            Modified input_ids tensor with image tokens replaced.

        Raises:
            ValueError: If drafting mode is unsupported.
        """
        assert self.manipulated_input_ids is None, (
            "Manipulated input IDs must be None for replacement."
        )
        assert value is not None, "Value must be provided for replacement."

        input_ids = self.input_ids.clone()

        if drafting == 'text-only':
            input_ids[:, self.image_ids] = value

        elif drafting == 'tokenized-image':
            input_ids = self._replace_with_tokenized_image(input_ids, value)

        elif drafting == 'caption':
            input_ids = self._replace_with_caption(input_ids, value)

        else:
            raise ValueError(f"Unsupported mode: {drafting}")

        return input_ids

    def _replace_with_tokenized_image(
        self,
        input_ids: torch.Tensor,
        value: List[int],
    ) -> torch.Tensor:
        """Replace image tokens with tokenized image representation.

        Args:
            input_ids: Original input IDs.
            value: List of token IDs to replace each image token.

        Returns:
            Modified input_ids with expanded replacements.
        """
        device = input_ids.device
        input_ids_flat = input_ids.flatten()
        indices_to_replace = self.image_ids.flatten().to(device)
        replacement_list = torch.tensor(value, device=device)

        replacement_length = len(value)
        original_length = input_ids_flat.size(0)
        num_replacements = len(indices_to_replace)
        new_length = original_length + num_replacements * (replacement_length - 1)

        new_tensor = torch.empty(new_length, dtype=input_ids.dtype, device=device)
        mask = torch.ones(new_length, dtype=torch.bool, device=device)

        insertion_offsets = torch.arange(
            len(indices_to_replace), device=device
        ) * (replacement_length - 1)
        insert_indices = indices_to_replace + insertion_offsets

        for i in range(replacement_length):
            mask[insert_indices + i] = 0

        remaining_indices = mask.nonzero(as_tuple=True)[0]
        new_tensor[remaining_indices] = input_ids_flat[
            input_ids_flat != self.image_token_id
        ]

        for i in range(replacement_length):
            new_tensor[insert_indices + i] = replacement_list[i]

        return new_tensor.unsqueeze(0)

    def _replace_with_caption(
        self,
        input_ids: torch.Tensor,
        value: List[List[int]],
    ) -> torch.Tensor:
        """Replace image tokens with caption token sequences.

        Args:
            input_ids: Original input IDs.
            value: List of token ID sequences, one per image token.

        Returns:
            Modified input_ids with caption replacements.
        """
        device = input_ids.device
        input_ids_flat = input_ids.flatten()
        indices_to_replace = self.image_ids.flatten().to(device)

        replacement_list_tensors = [
            torch.tensor(sublist, device=device) for sublist in value
        ]

        assert len(replacement_list_tensors) == len(indices_to_replace), (
            "Each replacement must correspond to an index to replace."
        )

        replacement_lengths = [len(sublist) for sublist in value]
        original_length = input_ids_flat.size(0)
        num_replacements = len(indices_to_replace)
        new_length = original_length + sum(replacement_lengths) - num_replacements

        new_tensor = torch.empty(new_length, dtype=input_ids.dtype, device=device)
        mask = torch.ones(new_length, dtype=torch.bool, device=device)

        offset = 0
        for idx, replace_index in enumerate(indices_to_replace):
            current_replacement = replacement_list_tensors[idx]
            current_replacement_length = replacement_lengths[idx]
            insertion_index = replace_index + offset

            mask[insertion_index:insertion_index + current_replacement_length] = 0
            new_tensor[
                insertion_index:insertion_index + current_replacement_length
            ] = current_replacement

            offset += current_replacement_length - 1

        remaining_indices = mask.nonzero(as_tuple=True)[0]
        new_tensor[remaining_indices] = input_ids_flat[
            input_ids_flat != self.image_token_id
        ]

        return new_tensor.unsqueeze(0)

    def generate_caption(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> List[str]:
        """Generate captions for images using the captioning model.

        Args:
            pixel_values: Image pixel values tensor.
            input_ids: Input IDs for captioning (used by some models).

        Returns:
            List of generated caption strings.

        Raises:
            ValueError: If captioning model is not set.
        """
        if self.captioning_model is None:
            raise ValueError(
                "Captioning model is not set. "
                "Please provide a captioning model for 'caption' mode."
            )

        inputs_caption = {
            "pixel_values": pixel_values.to(dtype=self.captioning_model.dtype),
        }

        if 'lorence-2' in self._config['captioning_model']:
            inputs_caption['input_ids'] = input_ids.expand(
                (pixel_values.size(0), input_ids.size(1))
            )
            inputs_caption['do_sample'] = False
            inputs_caption['max_new_tokens'] = 1024
            inputs_caption['num_beams'] = 3

        generated_ids = self.captioning_model.generate(**inputs_caption)
        generated_text = self.captioning_processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        return generated_text

    def rollback_to_original_prompt(
        self,
        candidate_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Rollback to original prompt while appending new sequences.

        Args:
            candidate_ids: Generated sequences after the manipulated prompt.

        Returns:
            Concatenated input with original prompt and new sequence remainder.
        """
        new_sequences_remainder = candidate_ids[
            :, self.manipulated_input_ids_length_initial:
        ]
        return torch.cat((self.input_ids, new_sequences_remainder), dim=1)

    def pool_image_embedding(self, image_embedding: torch.Tensor) -> torch.Tensor:
        """Apply pooling on image embedding to reduce sequence length.

        Args:
            image_embedding: Image embedding tensor of shape (1, S, E).

        Returns:
            Pooled embedding of shape (1, target_dim, E).

        Raises:
            ValueError: If target_dim > S or unsupported pooling type.
        """
        S, E = image_embedding.size(1), image_embedding.size(2)

        if self.target_dim > S:
            raise ValueError(
                "Target dimension must be less than or equal to sequence length."
            )

        if self._config['image_pool_type'] == 'avg1d':
            pooled_embedding = F.adaptive_avg_pool1d(
                image_embedding.permute(0, 2, 1),
                output_size=self.target_dim,
            )
            pooled_embedding = pooled_embedding.permute(0, 2, 1)

        elif self._config['image_pool_type'] == 'avg2d':
            s = image_embedding.size(1) ** 0.5
            assert s == int(s), "Image embedding must be square for 2D pooling."

            image_embedding = rearrange(
                image_embedding, 'b (h w) e -> b e h w', h=int(s)
            )

            l = self.target_dim ** 0.5
            assert l == int(l), "Target dimension must be square for 2D pooling."

            pooled_embedding = F.adaptive_avg_pool2d(
                image_embedding, output_size=(int(l), int(l))
            )
            pooled_embedding = rearrange(pooled_embedding, 'b e h w -> b (h w) e')

        else:
            raise ValueError(
                f"Unsupported image pooling type: {self._config['image_pool_type']}"
            )

        return pooled_embedding

    def _process_prompt(
        self,
        drafting: Union[str, List[str]],
    ) -> Tuple[
        Union[torch.Tensor, Dict],
        Union[int, Dict],
        Optional[Union[torch.Tensor, Dict]],
    ]:
        """Process the prompt based on the given drafting mode.

        Args:
            drafting: Drafting mode string or list of modes for TABED.

        Returns:
            Tuple of (manipulated_input_ids, length_initial, attention_mask_initial).

        Raises:
            ValueError: If drafting mode is unsupported.
        """
        mode = drafting
        attention_mask_initial = None

        if isinstance(mode, str):
            return self._process_single_mode(mode)

        elif isinstance(mode, list):
            return self._process_tabed_mode(mode)

        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def _process_single_mode(
        self,
        mode: str,
    ) -> Tuple[torch.Tensor, int, Optional[torch.Tensor]]:
        """Process a single drafting mode.

        Args:
            mode: Drafting mode string.

        Returns:
            Tuple of (manipulated_input_ids, length_initial, attention_mask_initial).
        """
        attention_mask_initial = None

        if mode in [
            'multimodal', 'multimodal-debug', 'multimodal-debug2',
            'special-token', 'image-pool'
        ]:
            manipulated_input_ids = self.input_ids

        elif mode == 'text-only':
            manipulated_input_ids = self.replace_image_tokens(
                mode, self.escape_token_id
            )

        elif mode == 'tokenized-image':
            manipulated_input_ids = self.replace_image_tokens(
                mode, self.pseudo_image_text_token_ids
            )
            attention_mask_initial = torch.ones_like(
                manipulated_input_ids,
                dtype=torch.long,
                device=manipulated_input_ids.device,
            )

        elif mode == 'caption':
            caption = self.generate_caption(
                self.pixel_values_caption,
                self.input_ids_caption,
            )
            caption_tokenized = self.tokenizer(caption, return_tensors='np').input_ids
            caption_tokenized = [
                torch.cat((
                    self.caption_prefix_ids,
                    torch.tensor(l[1:], device=self.caption_prefix_ids.device),
                ))
                for l in caption_tokenized
            ]

            manipulated_input_ids = self.replace_image_tokens(mode, caption_tokenized)
            attention_mask_initial = torch.ones_like(
                manipulated_input_ids,
                dtype=torch.long,
                device=manipulated_input_ids.device,
            )

        else:
            raise ValueError(f"Unsupported mode: {mode}")

        manipulated_input_ids_length_initial = manipulated_input_ids.size(1)

        return manipulated_input_ids, manipulated_input_ids_length_initial, attention_mask_initial

    def _process_tabed_mode(
        self,
        modes: List[str],
    ) -> Tuple[Dict, Dict, Dict]:
        """Process TABED drafting with multiple modes.

        Args:
            modes: List of drafting mode strings.

        Returns:
            Tuple of dictionaries mapping modes to their processed values.
        """
        input_tabed = defaultdict(dict)

        for submode in modes:
            (
                manipulated_input_ids_submode,
                manipulated_input_ids_length_initial_submode,
                attention_mask_initial_submode,
            ) = self._process_prompt(submode)

            input_tabed['manipulated_input_ids'][submode] = (
                manipulated_input_ids_submode
            )
            input_tabed['manipulated_input_ids_length_initial'][submode] = (
                manipulated_input_ids_length_initial_submode
            )
            input_tabed['attention_mask_initial'][submode] = (
                attention_mask_initial_submode
            )

        return (
            input_tabed['manipulated_input_ids'],
            input_tabed['manipulated_input_ids_length_initial'],
            input_tabed['attention_mask_initial'],
        )
