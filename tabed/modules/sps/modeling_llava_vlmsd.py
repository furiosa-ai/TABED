from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Dict, Any
import time

import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F

from transformers.modeling_utils import PreTrainedModel
from transformers.models.llava.modeling_llava import LlavaCausalLMOutputWithPast, LLAVA_INPUTS_DOCSTRING
from transformers.utils import (
    ModelOutput,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
    is_torchdynamo_compiling,
)
from transformers.cache_utils import Cache

from ...utils.util import noop_context, patch_function

_CONFIG_FOR_DOC = "LlavaConfig"

"""
_get_initial_cache_position
prepare_inputs_for_generation
_update_model_kwargs_for_generation
forward

"""

def _merge_input_ids_with_image_features_image_top_k(self, image_features, inputs_embeds, input_ids, attention_mask, labels):
    num_images, num_image_patches, embed_dim = image_features.shape
    batch_size, sequence_length = input_ids.shape
    left_padding = not torch.sum(input_ids[:, -1] == torch.tensor(self.pad_token_id))
    # 1. Create a mask to know where special image tokens are
    special_image_token_mask = input_ids == self.config.image_token_index
    num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
    # Compute the maximum embed dimension
    max_embed_dim = (num_special_image_tokens.max() * (num_image_patches - 1)) + sequence_length
    batch_indices, non_image_indices = torch.where(input_ids != self.config.image_token_index)

    # 2. Compute the positions where text should be written
    # Calculate new positions for text tokens in merged image-text sequence.
    # `special_image_token_mask` identifies image tokens. Each image token will be replaced by `nb_text_tokens_per_images - 1` text tokens.
    # `torch.cumsum` computes how each image token shifts subsequent text token positions.
    # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
    new_token_positions = torch.cumsum((special_image_token_mask * (num_image_patches - 1) + 1), -1) - 1
    nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
    if left_padding:
        new_token_positions += nb_image_pad[:, None]  # offset for left padding
    text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

    # 3. Create the full embedding, already padded to the maximum position
    final_embedding = torch.zeros(
        batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
    )
    final_attention_mask = torch.zeros(
        batch_size, max_embed_dim, dtype=attention_mask.dtype, device=inputs_embeds.device
    )
    if labels is not None:
        final_labels = torch.full(
            (batch_size, max_embed_dim), self.config.ignore_index, dtype=input_ids.dtype, device=input_ids.device
        )
    # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
    # set the corresponding tensors into their correct target device.
    target_device = inputs_embeds.device
    batch_indices, non_image_indices, text_to_overwrite = (
        batch_indices.to(target_device),
        non_image_indices.to(target_device),
        text_to_overwrite.to(target_device),
    )
    attention_mask = attention_mask.to(target_device)

    # 4. Fill the embeddings based on the mask. If we have ["hey" "<image>", "how", "are"]
    # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the image features
    final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
    final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]
    if labels is not None:
        final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_image_indices]

    # 5. Fill the embeddings corresponding to the images. Anything that is not `text_positions` needs filling (#29835)
    image_to_overwrite = torch.full(
        (batch_size, max_embed_dim), True, dtype=torch.bool, device=inputs_embeds.device
    )
    image_to_overwrite[batch_indices, text_to_overwrite] = False
    image_to_overwrite &= image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[:, None].to(target_device)
    if image_to_overwrite.sum() != image_features.shape[:-1].numel():
        raise ValueError(
            f"The input provided to the model are wrong. The number of image tokens is {torch.sum(special_image_token_mask)} while"
            f" the number of image given to the model is {num_images}. This prevents correct indexing and breaks batch generation."
        )

    final_embedding[image_to_overwrite] = image_features.contiguous().reshape(-1, embed_dim).to(target_device)
    final_attention_mask |= image_to_overwrite
    position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

    # 6. Mask out the embedding at padding positions, as we later use the past_key_value value to determine the non-attended tokens.
    batch_indices, pad_indices = torch.where(input_ids == self.pad_token_id)
    indices_to_mask = new_token_positions[batch_indices, pad_indices]

    final_embedding[batch_indices, indices_to_mask] = 0

    if labels is None:
        final_labels = None

    kwargs_image_token_mask = {
        "image_token_mask": image_to_overwrite, 
        "num_images": num_images,
    }

    return final_embedding, final_attention_mask, final_labels, position_ids, kwargs_image_token_mask

def _get_initial_cache_position(self, input_ids, model_kwargs):
    """Calculates `cache_position` for the pre-fill stage based on `input_ids` and optionally past length"""
    # `torch.compile`-friendly `torch.arange` from a shape -- the lines below are equivalent to `torch.arange`
    if "inputs_embeds" in model_kwargs:
        cache_position = torch.ones_like(model_kwargs["inputs_embeds"][0, :, 0], dtype=torch.int64).cumsum(0) - 1
    else:
        cache_length = model_kwargs['past_key_values'].get_seq_length()
        if input_ids.shape[1] > cache_length:
            # prefill
            cache_position = torch.ones_like(input_ids[0, :], dtype=torch.int64).cumsum(0) - 1
        else:
            # decode subsequent chunks for VLMSD only for draft model (not target model)
            num_additional_cache = 2 
            cache_position = torch.arange(cache_length + num_additional_cache, device=input_ids.device, dtype=torch.int64)

    past_length = 0
    if model_kwargs.get("past_key_values") is not None:
        cache = model_kwargs["past_key_values"]
        past_length = 0
        if not isinstance(cache, Cache):
            past_length = cache[0][0].shape[2]
        elif hasattr(cache, "get_seq_length") and cache.get_seq_length() is not None:
            past_length = cache.get_seq_length()

        # TODO(joao): this is not torch.compile-friendly, find a work-around. If the cache is not empty,
        # end-to-end compilation will yield bad results because `cache_position` will be incorrect.
        if not is_torchdynamo_compiling():
            cache_position = cache_position[past_length:]

    model_kwargs["cache_position"] = cache_position
    return model_kwargs

def prepare_inputs_for_generation(
    self, input_ids, past_key_values=None, inputs_embeds=None, pixel_values=None, attention_mask=None, **kwargs
):
    if past_key_values is not None:
        if isinstance(past_key_values, Cache):
            cache_length = past_key_values.get_seq_length()
            past_length = past_key_values.seen_tokens
        else:
            cache_length = past_length = past_key_values[0][0].shape[2]

        # Keep only the unprocessed tokens:
        # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
        # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
        # input)
        if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
            input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
        # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
        # input_ids based on the past_length.
        elif past_length < input_ids.shape[1]: 
            # prefill (no kv cache)
            input_ids = input_ids[:, past_length:]
        # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.
        elif self.config.image_token_index in input_ids:
            # decode (with kv cache)
            cache_position_length = len(kwargs['cache_position'])
            input_ids = input_ids[:, input_ids.shape[1] - cache_position_length :]
        # If the cache has seen more tokens than it can hold, then the cache has a size limit. Let's discard the
        # older attention values, as their corresponding values are not part of the input.
        if cache_length < past_length and attention_mask is not None:
            attention_mask = attention_mask[:, -(cache_length + input_ids.shape[1]) :]

    position_ids = kwargs.get("position_ids", None)
    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1] :]

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}

    model_inputs.update(
        {
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }
    )
    return model_inputs

@add_start_docstrings_to_model_forward(LLAVA_INPUTS_DOCSTRING)
@replace_return_docstrings(output_type=LlavaCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
def forward(
    self,
    input_ids: torch.LongTensor = None,
    pixel_values: torch.FloatTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    vision_feature_layer: Optional[int] = None,
    vision_feature_select_strategy: Optional[str] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, LlavaCausalLMOutputWithPast]:
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    Returns:

    Example:

    ```python
    >>> from PIL import Image
    >>> import requests
    >>> from transformers import AutoProcessor, LlavaForConditionalGeneration

    >>> model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
    >>> processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

    >>> prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"
    >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)

    >>> inputs = processor(text=prompt, images=image, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(**inputs, max_new_tokens=15)
    >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "USER:  \nWhat's the content of the image? ASSISTANT: The image features a busy city street with a stop sign prominently displayed"
    ```"""

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    vision_feature_layer = (
        vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
    )
    vision_feature_select_strategy = (
        vision_feature_select_strategy
        if vision_feature_select_strategy is not None
        else self.config.vision_feature_select_strategy
    )

    if inputs_embeds is None:
        # 1. Extra the input embeddings
        inputs_embeds = self.get_input_embeddings()(input_ids)
        has_past_key_values = past_key_values.get_seq_length() != 0

        # 2. Merge text and images
        if not has_past_key_values and pixel_values is not None and input_ids.shape[1] != 1:
            if hasattr(self, 'prompt_setter'):
                torch.cuda.synchronize()
                start_time_time_prompt_process = time.time()
            
            image_outputs = self.vision_tower(pixel_values, output_hidden_states=True) # Zv = Encoder(Xv)
            # this is not memory efficient at all (output_hidden_states=True) will save all the hidden stated.
            selected_image_feature = image_outputs.hidden_states[vision_feature_layer]

            if vision_feature_select_strategy == "default":
                selected_image_feature = selected_image_feature[:, 1:]
            elif vision_feature_select_strategy == "full":
                selected_image_feature = selected_image_feature
            else:
                raise ValueError(
                    f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}"
                )

            if (hasattr(self, 'prompt_setter') and self.prompt_setter.drafting == "image-pool") \
                or (hasattr(self, 'tabed_current_drafting') and self.tabed_current_drafting == "image-pool"):
                selected_image_feature = self.prompt_setter.pool_image_embedding(selected_image_feature)
            
            image_features = self.multi_modal_projector(selected_image_feature) # Hv = W * Zv

            if hasattr(self, 'prompt_setter'):
                torch.cuda.synchronize()
                self.time_prompt_process = time.time() - start_time_time_prompt_process
                
            inputs_embeds = inputs_embeds.to(image_features.dtype)

            # top-k attention if enabled
            if hasattr(self, 'prompt_setter') and (self._config['output_image_attentions'] or self._config['image_top_k_attention']):
                inputs_embeds, attention_mask, labels, position_ids, kwargs_image_token_mask = self._merge_input_ids_with_image_features(
                    image_features, inputs_embeds, input_ids, attention_mask, labels
                )

                image_regions = get_image_region_ids(**kwargs_image_token_mask)
                self.attention_kwargs = {
                    'image_top_k_attention': self._config['image_top_k_attention'],
                    'image_regions': image_regions,
                    'num_images': kwargs_image_token_mask['num_images'],
                } 
                        
            else:
                inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_image_features(
                    image_features, inputs_embeds, input_ids, attention_mask, labels
                )

        # In case input_ids.shape[1] == 1 & pixel_values==None & past_key_values != None, we are in the case of
        # generation with cache
        elif past_key_values is not None and pixel_values is not None:
            # Retrieve the first layer to inspect the logits and mask out the hidden states
            # that are set to 0
            first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

            # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
            batch_index, non_attended_tokens = torch.where(first_layer_past_key_value.float().sum(-2) == 0)

            # Get the target length
            target_length = input_ids.shape[1]
            past_length = first_layer_past_key_value.shape[-1]

            extended_attention_mask = torch.ones(
                (attention_mask.shape[0], past_length),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )

            # Filter out only the tokens that can be un-attended, this can happen
            # if one uses Llava + Fused modules where the cache on the
            # first iteration is already big enough, or if one passes custom cache
            valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
            new_batch_index = batch_index[valid_indices]
            new_non_attended_tokens = non_attended_tokens[valid_indices]

            # Zero-out the places where we don't need to attend
            if hasattr(self, 'prompt_setter') and self._config['solution_type'] == "sol2":
                extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

            attention_mask = torch.cat((extended_attention_mask, attention_mask[:, -target_length:]), dim=1)
            if target_length == 1:
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            else:
                position_ids = attention_mask.cumsum(-1) - 1
                position_ids = position_ids[:, -target_length:]
            """
            if len(new_batch_index) > 0:
                position_ids += len(new_batch_index)   
            """

    outputs = self.language_model(
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    logits = outputs[0]

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        if attention_mask is not None:
            shift_attention_mask = attention_mask[..., 1:]
            shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
            shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
        else:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
        )

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return LlavaCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

def _update_model_kwargs_for_generation(
    self,
    outputs: ModelOutput,
    model_kwargs: Dict[str, Any],
    is_encoder_decoder: bool = False,
    num_new_tokens: int = 1,
) -> Dict[str, Any]:
    # update past_key_values keeping its naming used in model code
    cache_name, cache = self._extract_past_from_model_output(outputs)
    model_kwargs[cache_name] = cache
    if getattr(outputs, "state", None) is not None:
        model_kwargs["state"] = outputs.state

    # update token_type_ids with last value
    if "token_type_ids" in model_kwargs:
        token_type_ids = model_kwargs["token_type_ids"]
        model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

    if not is_encoder_decoder:
        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )
    else:
        # update decoder attention mask
        if "decoder_attention_mask" in model_kwargs:
            decoder_attention_mask = model_kwargs["decoder_attention_mask"]
            model_kwargs["decoder_attention_mask"] = torch.cat(
                [decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1))],
                dim=-1,
            )

    if model_kwargs.get("use_cache", True):
        model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + num_new_tokens
    else:
        past_positions = model_kwargs.pop("cache_position")
        new_positions = torch.arange(
            past_positions[-1] + 1, past_positions[-1] + num_new_tokens + 1, dtype=past_positions.dtype
        ).to(past_positions.device)
        model_kwargs["cache_position"] = torch.cat((past_positions, new_positions))
    return model_kwargs

def get_image_region_ids(image_token_mask, num_images):
    """
    Identifies contiguous regions of True values in the image_token_mask and computes the region length.
    
    Args:
        image_token_mask: Boolean tensor of shape (1, S) indicating image token positions.
    
    Returns:
        region_indices: Tensor of shape (num_regions, region_length) containing indices for each contiguous region.
    """

    region_indices = image_token_mask.nonzero(as_tuple=True)[1].unsqueeze(0)
    
    return region_indices

@add_start_docstrings_to_model_forward(LLAVA_INPUTS_DOCSTRING)
@replace_return_docstrings(output_type=LlavaCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
def forward_pooling(
    self,
    input_ids: torch.LongTensor = None,
    pixel_values: torch.FloatTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    vision_feature_layer: Optional[int] = None,
    vision_feature_select_strategy: Optional[str] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, LlavaCausalLMOutputWithPast]:
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    Returns:

    Example:

    ```python
    >>> from PIL import Image
    >>> import requests
    >>> from transformers import AutoProcessor, LlavaForConditionalGeneration

    >>> model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
    >>> processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

    >>> prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"
    >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)

    >>> inputs = processor(text=prompt, images=image, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(**inputs, max_new_tokens=15)
    >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "USER:  \nWhat's the content of the image? ASSISTANT: The image features a busy city street with a stop sign prominently displayed"
    ```"""

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    vision_feature_layer = (
        vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
    )
    vision_feature_select_strategy = (
        vision_feature_select_strategy
        if vision_feature_select_strategy is not None
        else self.config.vision_feature_select_strategy
    )

    if inputs_embeds is None:
        # 1. Extra the input embeddings
        inputs_embeds = self.get_input_embeddings()(input_ids)

        # 2. Merge text and images
        if pixel_values is not None and input_ids.shape[1] != 1:
            image_outputs = self.vision_tower(pixel_values, output_hidden_states=True)
            # this is not memory efficient at all (output_hidden_states=True) will save all the hidden stated.
            selected_image_feature = image_outputs.hidden_states[vision_feature_layer]

            if vision_feature_select_strategy == "default":
                selected_image_feature = selected_image_feature[:, 1:]
            elif vision_feature_select_strategy == "full":
                selected_image_feature = selected_image_feature
            else:
                raise ValueError(
                    f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}"
                )

            if (hasattr(self, 'prompt_setter') and self.prompt_setter.drafting == "image-pool") \
                or (hasattr(self, 'tabed_current_drafting') and self.tabed_current_drafting == "image-pool"):
                selected_image_feature = self.prompt_setter.pool_image_embedding(selected_image_feature)

            image_features = self.multi_modal_projector(selected_image_feature)
            inputs_embeds = inputs_embeds.to(image_features.dtype)
            inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_image_features(
                image_features, inputs_embeds, input_ids, attention_mask, labels
            )

        # In case input_ids.shape[1] == 1 & pixel_values==None & past_key_values != None, we are in the case of
        # generation with cache
        elif past_key_values is not None and pixel_values is not None and input_ids.shape[1] == 1:
            # Retrieve the first layer to inspect the logits and mask out the hidden states
            # that are set to 0
            first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

            # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
            batch_index, non_attended_tokens = torch.where(first_layer_past_key_value.float().sum(-2) == 0)

            # Get the target length
            target_length = input_ids.shape[1]
            past_length = first_layer_past_key_value.shape[-1]

            extended_attention_mask = torch.ones(
                (attention_mask.shape[0], past_length),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )

            # Filter out only the tokens that can be un-attended, this can happen
            # if one uses Llava + Fused modules where the cache on the
            # first iteration is already big enough, or if one passes custom cache
            valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
            new_batch_index = batch_index[valid_indices]
            new_non_attended_tokens = non_attended_tokens[valid_indices]

            # Zero-out the places where we don't need to attend
            extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

            attention_mask = torch.cat((extended_attention_mask, attention_mask[:, -target_length:]), dim=1)
            position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1

    outputs = self.language_model(
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    logits = outputs[0]

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        if attention_mask is not None:
            shift_attention_mask = attention_mask[..., 1:]
            shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
            shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
        else:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
        )

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return LlavaCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )