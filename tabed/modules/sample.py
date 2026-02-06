"""Custom sampling implementation for generation with timing metrics."""

from typing import Optional, Union

import torch
from torch import nn
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.utils import (
    GenerateEncoderDecoderOutput,
    GenerateNonBeamOutput,
    LogitsProcessorList,
    StoppingCriteriaList,
)

from .eval_specbench import measure_time
from .utils import GenerateDecoderOnlyOutput


def _sample(
    self,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfig,
    synced_gpus: bool,
    streamer: Optional["BaseStreamer"],
    logits_warper: Optional[LogitsProcessorList],
    **model_kwargs,
) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
    """Generate sequences using multinomial sampling.

    Generates sequences of token ids for models with a language modeling head
    using multinomial sampling. Can be used for text-decoder, text-to-text,
    speech-to-text, and vision-to-text models.

    Args:
        input_ids: The sequence used as a prompt for generation.
            Shape: (batch_size, sequence_length).
        logits_processor: List of LogitsProcessor instances used to modify
            prediction scores at each generation step.
        stopping_criteria: List of StoppingCriteria instances used to
            determine when to stop generation.
        generation_config: The generation configuration parameters.
        synced_gpus: Whether to continue the while loop until max_length
            (needed for ZeRO stage 3).
        streamer: Streamer object for streaming generated sequences.
        logits_warper: List of LogitsWarper instances used to warp the
            prediction score distribution before multinomial sampling.
        **model_kwargs: Additional model-specific kwargs forwarded to
            the forward function.

    Returns:
        GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput, or
        torch.LongTensor containing the generated tokens.
    """
    # Initialize values from generation config
    pad_token_id = generation_config._pad_token_tensor
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    max_length = generation_config.max_length
    has_eos_stopping_criteria = any(
        hasattr(criteria, "eos_token_id") for criteria in stopping_criteria
    )
    do_sample = generation_config.do_sample

    if do_sample is True and not isinstance(logits_warper, LogitsProcessorList):
        raise ValueError(
            "`do_sample` is set to `True`, `logits_warper` must be a "
            f"`LogitsProcessorList` instance (it is {logits_warper})."
        )

    # Initialize output tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = (
        () if (return_dict_in_generate and output_hidden_states) else None
    )

    # Handle encoder-decoder models
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = (
            model_kwargs["encoder_outputs"].get("attentions")
            if output_attentions else None
        )
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states")
            if output_hidden_states else None
        )

    # Track finished sequences
    batch_size, cur_len = input_ids.shape
    this_peer_finished = False
    unfinished_sequences = torch.ones(
        batch_size, dtype=torch.long, device=input_ids.device
    )
    model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)
    time_prefill = None
    num_prefill_tokens = None

    while self._has_unfinished_sequences(
        this_peer_finished,
        synced_gpus,
        device=input_ids.device,
        cur_len=cur_len,
        max_length=max_length,
    ):
        # Prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # Prepare variable output controls
        if output_attentions:
            model_inputs["output_attentions"] = output_attentions
        if output_hidden_states:
            model_inputs["output_hidden_states"] = output_hidden_states

        # Forward pass to get next token
        has_past_key_values = (
            model_kwargs.get("past_key_values").get_seq_length() > 0
        )
        if has_past_key_values:
            outputs = self(**model_inputs, return_dict=True)
        else:
            outputs, time_prefill = measure_time(
                self, **model_inputs, return_dict=True
            )
            num_prefill_tokens = outputs.get("past_key_values").get_seq_length()

        if synced_gpus and this_peer_finished:
            continue

        # Clone to avoid keeping large logits reference
        next_token_logits = outputs.logits[:, -1, :].clone()

        # Pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)
        if do_sample:
            next_token_scores = logits_warper(input_ids, next_token_scores)

        # Store outputs when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_logits:
                raw_logits += (next_token_logits,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,)
                    if self.config.is_encoder_decoder
                    else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)
            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # Token selection
        if do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)

        # Handle finished sentences
        if has_eos_stopping_criteria:
            next_tokens = (
                next_tokens * unfinished_sequences
                + pad_token_id * (1 - unfinished_sequences)
            )

        # Update for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())

        model_kwargs = self._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=self.config.is_encoder_decoder,
        )

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(
            input_ids, scores
        )
        this_peer_finished = unfinished_sequences.max() == 0
        cur_len += 1

        # Delete outputs to free memory
        del outputs

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GenerateEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
                time_prefill=time_prefill,
                num_prefill_tokens=num_prefill_tokens,
            )
    else:
        return input_ids
