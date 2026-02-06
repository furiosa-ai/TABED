import torch
import torch.nn.functional as F
import copy
import warnings

from dataclasses import dataclass

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList, ModelOutput, GenerateEncoderDecoderOutput, GenerateNonBeamOutput
from transformers.generation.configuration_utils import GenerationConfig
from transformers.cache_utils import (
    DynamicCache,
    EncoderDecoderCache,
)
from transformers.generation.candidate_generator import (
    _crop_past_key_values,
    _prepare_attention_mask,
    _prepare_token_type_ids,
)
import torch.distributed as dist
import os, time
FUNC_MAP = {}
CONFIG_MAP = {}
COLOR_PRINT = int(os.environ.get("COLOR_PRINT", 0))

from .candidate_generator_vlmsd import CandidateGenerator, AssistedCandidateGeneratorVLM, AssistedCandidateGeneratorTabedVLM

from ..eval_specbench import measure_time
from ..utils import GenerateDecoderOnlyOutput

def _assisted_decoding(
    self,
    input_ids: torch.LongTensor,
    candidate_generator: CandidateGenerator,
    logits_processor: LogitsProcessorList,
    logits_warper: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfig,
    synced_gpus: bool,
    streamer: Optional["BaseStreamer"],
    **model_kwargs,
) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
    r"""
    Generates sequences of token ids for models with a language modeling head using **greedy decoding** or
    **sample** (depending on `do_sample`), assisted by candidate sequences. Assisted generation is an example of a
    candidate decoding strategy. Can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text
    models.

    Parameters:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The sequence used as a prompt for the generation.
        candidate_generator (`CandidateGenerator`):
            A derived instance of [`CandidateGenerator`] that defines how candidate sequences are generated. For
            more information, the documentation of [`CandidateGenerator`] should be read.
        logits_processor (`LogitsProcessorList`):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        logits_warper (`LogitsProcessorList`):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
            to warp the prediction score distribution of the language modeling head applied before multinomial
            sampling at each generation step. Only used if sampling is active.
        stopping_criteria (`StoppingCriteriaList`):
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
            used to tell if the generation loop should stop.
        generation_config ([`~generation.GenerationConfig`]):
            The generation configuration to be used as parametrization of the decoding method.
        synced_gpus (`bool`):
            Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
        streamer (`BaseStreamer`, *optional*):
            Streamer object that will be used to stream the generated sequences. Generated tokens are passed
            through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
        model_kwargs:
            Additional model specific keyword arguments will be forwarded to the `forward` function of the model.
            If model is an encoder-decoder model the kwargs should include `encoder_outputs`.

    Return:
        [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or
        `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
        [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
        `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
        `model.config.is_encoder_decoder=True`.
    """
    # init values
    do_sample = logits_warper is not None
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    batch_size = input_ids.shape[0]
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
    model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

    # This is needed if return_dict_in_generate is True
    start_from_empty_dynamic_cache = False
    past_key_values = model_kwargs.get("past_key_values", None)
    if isinstance(past_key_values, DynamicCache) or (
        isinstance(past_key_values, EncoderDecoderCache)
        and isinstance(past_key_values.self_attention_cache, DynamicCache)
    ):
        if len(past_key_values) == 0:
            start_from_empty_dynamic_cache = True

    this_peer_finished = False 
    metrics = {
        'num_accepted_tokens': [],
        'num_prefill_tokens_drf': None, 
        'num_prefill_tokens_tgt': None, 
        'ids_accepted_tokens': [], 
        'ids_first_rejected': [], 
        'tokens_first_rejected': [], 
        'time_total': None, # time_decode = time_total - time_prefill_drf - time_prefill_tgt
        "time_prefill_drf": None, 
        "time_prefill_tgt": None, 
        "time_prompt_process": None,
        "value_image_attention_drf_accepted": [],
        "value_image_attention_drf_first_rejected": [],
        "ids_image_attention_drf_accepted": [],

        # top-k logging
        'tokens_accepted_tokens_topk': {} if isinstance(candidate_generator, AssistedCandidateGeneratorTabedVLM) else [],
        'value_probability_accepted_topk': {} if isinstance(candidate_generator, AssistedCandidateGeneratorTabedVLM) else [],
        'tokens_rejected_tokens_topk': {} if isinstance(candidate_generator, AssistedCandidateGeneratorTabedVLM) else [],
        'value_probability_rejected_topk': {} if isinstance(candidate_generator, AssistedCandidateGeneratorTabedVLM) else [],
        
        "value_probability_ratio_accepted": [],
        "value_probability_ratio_first_rejected": [],
        "value_probability_accepted_drf": [],
        "value_probability_accepted_tgt": [],
        "value_probability_first_rejected_drf": [],
        "value_probability_first_rejected_tgt": [],

        "history_dependent_weights": {},
    }
    is_time_factorized = candidate_generator._config['is_time_factorized']
    do_print = candidate_generator._config['do_print']
    
    
    if is_time_factorized:
        metrics['time_generate_drf'] = [] # time_decode_drf = time_generate_drf - time_prefill_drf
        metrics['time_verify_tgt'] = [] # time_decode_tgt = time_verify_tgt - time_prefill_tgt

    while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        cur_len = input_ids.shape[-1]

        #  1. Fetch candidate sequences from a `CandidateGenerator`
        last_target_logits = new_logits if self._config['history_dependent'] and 'new_logits' in locals() else None
        if not is_time_factorized:
            candidate_outputs = candidate_generator.get_candidates(input_ids, last_target_logits)
        else:
            candidate_outputs, time_generate_drf = measure_time(candidate_generator.get_candidates, input_ids, last_target_logits)
            metrics['time_generate_drf'].append(time_generate_drf)
            
        candidate_input_ids = candidate_outputs["candidate_ids"]
        candidate_logits = candidate_outputs["candidate_logits"]
        new_attentions = candidate_outputs.get('draft_image_attentions')

        if candidate_outputs["time_prefill"] is not None:
            assert metrics["time_prefill_drf"] is None
            metrics['num_prefill_tokens_drf'] = candidate_outputs["num_prefill_tokens"]
            metrics["time_prefill_drf"] = candidate_outputs["time_prefill"]
            metrics["time_prompt_process"] = candidate_outputs["time_prompt_process"]
        
        if candidate_logits is not None:
            candidate_logits = candidate_logits.to(self.device)

        candidate_length = candidate_input_ids.shape[1] - input_ids.shape[1]
        assert candidate_length <= self._config['max_chunk_length'], f"candidate_length: {candidate_length} > max_chunk_length: {self._config['max_chunk_length']}"
        
        is_done_candidate = stopping_criteria(candidate_input_ids, None)

        # 2. Use the original model to obtain the next token logits given the candidate sequence. We obtain
        # `candidate_length + 1` relevant logits from this process: in the event that all candidates are correct,
        # we use this forward pass to also pick the subsequent logits in the original model.

        # 2.1. Prepare the model inputs
        candidate_kwargs = copy.copy(model_kwargs)
        candidate_kwargs = _prepare_attention_mask(
            candidate_kwargs, candidate_input_ids.shape[1], self.config.is_encoder_decoder
        )
        candidate_kwargs = _prepare_token_type_ids(candidate_kwargs, candidate_input_ids.shape[1])
        if "cache_position" in candidate_kwargs:
            candidate_kwargs["cache_position"] = torch.cat(
                (
                    candidate_kwargs["cache_position"],
                    torch.arange(cur_len, cur_len + candidate_length, device=input_ids.device, dtype=torch.long),
                ),
                dim=0,
            )

        model_inputs = self.prepare_inputs_for_generation(candidate_input_ids, **candidate_kwargs)
        if "num_logits_to_keep" in model_inputs:
            model_inputs["num_logits_to_keep"] = candidate_length + 1

        # 2.2. Run a forward pass on the candidate sequence
        # prepare variable output controls (note: some models won't accept all output controls)
        model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
        model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

        if model_inputs['past_key_values'].get_seq_length() > 0:
            if is_time_factorized:
                outputs, time_verify_tgt = measure_time(self, **model_inputs)
                metrics["time_verify_tgt"].append(time_verify_tgt)
            else:
                outputs = self(**model_inputs)
        else:
            outputs, time_prefill = measure_time(self, **model_inputs)
            assert metrics["time_prefill_tgt"] is None
            metrics["time_prefill_tgt"] = time_prefill
            metrics['num_prefill_tokens_tgt'] = outputs["past_key_values"].get_seq_length()
            if is_time_factorized:
                metrics["time_verify_tgt"].append(time_prefill)

        # 2.3. Process the new logits
        new_logits = outputs.logits[:, -candidate_length - 1 :]  # excludes the input prompt if present
        next_token_logits = new_logits.clone()
        if len(logits_processor) > 0:
            for i in range(candidate_length + 1):
                new_logits[:, i, :] = logits_processor(candidate_input_ids[:, : cur_len + i], new_logits[:, i, :])
        if do_sample and len(logits_warper) > 0:
            for i in range(candidate_length + 1):
                new_logits[:, i, :] = logits_warper(candidate_input_ids[:, : cur_len + i], new_logits[:, i, :])

        # 3. Select the accepted tokens. There are two possible cases:
        # Case 1: `do_sample=True` and we have logits for the candidates (originally from speculative decoding)
        # 👉 Apply algorithm 1 from the speculative decoding paper (https://arxiv.org/pdf/2211.17192.pdf).
        if do_sample and candidate_logits is not None:
            valid_tokens, n_matches = _speculative_sampling(
                candidate_input_ids,
                candidate_logits,
                candidate_length,
                new_logits,
                is_done_candidate,
            )

        # Case 2: all other cases (originally from assisted generation) 👉 Compare the tokens selected from the
        # original model logits with the candidate tokens. We can keep the candidate tokens until the first
        # mismatch, or until the max length is reached.
        else:
            if do_sample:
                probs = new_logits.softmax(dim=-1)
                selected_tokens = torch.multinomial(probs[0, :, :], num_samples=1).squeeze(1)[None, :]
            else:
                selected_tokens = new_logits.argmax(dim=-1)

            candidate_new_tokens = candidate_input_ids[:, cur_len:]
            n_matches = ((~(candidate_new_tokens == selected_tokens[:, :-1])).cumsum(dim=-1) < 1).sum()
            # for debugging
            if do_print and candidate_logits is not None:
                if isinstance(candidate_generator, AssistedCandidateGeneratorTabedVLM) and candidate_generator.tabed_rule in ['mm-weight', 'dist-sum']:
                    mm_index = list(candidate_generator.candidate_generator_tabed.keys()).index('multimodal')
                    mm_weight = self._config['mm_weight_policy']
                    _, _, probability_ratio = get_probability_along_tokens(candidate_logits, new_logits, candidate_new_tokens, candidate_length, output_ratio=True, merge_prob=True, mm_index=mm_index, mm_weight=mm_weight)
                else:
                    _, _, probability_ratio = get_probability_along_tokens(candidate_logits, new_logits, candidate_new_tokens, candidate_length, output_ratio=True)
                print([f"{ratio:.4f}" for ratio in probability_ratio])
            # Ensure we don't generate beyond max_len or an EOS token
            if is_done_candidate and n_matches == candidate_length:
                n_matches -= 1
            valid_tokens = selected_tokens[:, : n_matches + 1]

        # 4. Update variables according to the number of matching assistant tokens. Remember: the token generated
        # by the model after the last candidate match is also valid, as it is generated from a correct sequence.
        # Because of this last token, assisted generation search reduces to a normal greedy search/sample if there
        # is no match.

        # 4.1. Get the valid continuation, after the matching tokens
        n_matches_int = n_matches.item()
        
        metrics['ids_accepted_tokens'].extend(list(range(input_ids.shape[1], input_ids.shape[1] + n_matches_int)))
        metrics['num_accepted_tokens'].append(n_matches_int)
        if candidate_generator._config['history_dependent']:
            metrics['history_dependent_weights'].update(candidate_generator.cum_history_dependent_weight)

        # Assuming we want to log topk indices and probabilities for accepted tokens only
        # Slicing out the top-k for only accepted tokens from candidate_topk_probs and candidate_topk_indices
        if self._config['logging_top_k'] is not None and candidate_logits is not None:
            if 'candidate_logits_total' not in candidate_outputs:
                # single drafting
                candidate_probs = F.softmax(candidate_logits, dim=-1)
                candidate_topk_probs, candidate_topk_indices = torch.topk(candidate_probs, k=self._config['logging_top_k'] , dim=-1)

                accepted_candidate_topk_indices = candidate_topk_indices[:, :n_matches_int, :]
                accepted_candidate_topk_probs = candidate_topk_probs[:, :n_matches_int, :]

                # Log the top-k probabilities and indices for the accepted tokens
                metrics['tokens_accepted_tokens_topk'].extend(accepted_candidate_topk_indices[0].tolist())  # pick batch index 0
                metrics['value_probability_accepted_topk'].extend(accepted_candidate_topk_probs[0].tolist())  # pick batch index 0

            else:
                # TABED
                logits_total = candidate_outputs['candidate_logits_total']

                for _drafting, candidate_logits_drafting in logits_total.items():
                    candidate_probs_drafting = F.softmax(candidate_logits_drafting, dim=-1)
                    candidate_topk_probs_drafting, candidate_topk_indices_drafting = torch.topk(candidate_probs_drafting, k=self._config['logging_top_k'] , dim=-1)

                    accepted_candidate_topk_indices_drafting = candidate_topk_indices_drafting[:, :n_matches_int, :]
                    accepted_candidate_topk_probs_drafting = candidate_topk_probs_drafting[:, :n_matches_int, :]

                    # Log the top-k probabilities and indices for the accepted tokens
                    if _drafting not in metrics['tokens_accepted_tokens_topk']:
                        metrics['tokens_accepted_tokens_topk'][_drafting] = accepted_candidate_topk_indices_drafting[0].tolist()
                        metrics['value_probability_accepted_topk'][_drafting] = accepted_candidate_topk_probs_drafting[0].tolist()
                    else:
                        metrics['tokens_accepted_tokens_topk'][_drafting].extend(accepted_candidate_topk_indices_drafting[0].tolist())  # pick batch index 0
                        metrics['value_probability_accepted_topk'][_drafting].extend(accepted_candidate_topk_probs_drafting[0].tolist())  # pick batch index 0

        # if rejection occurred
        if (n_matches_int != candidate_length) and not (is_done_candidate and n_matches + 1 == candidate_length):
            idx_first_rejected = input_ids.shape[1] + n_matches_int
            metrics['ids_first_rejected'].append(idx_first_rejected)
            metrics['tokens_first_rejected'].append(candidate_new_tokens[:, n_matches_int].item())
            if self._config['logging_top_k'] is not None:
                if 'candidate_logits_total' not in candidate_outputs:
                    # Log the top-k probabilities and indices for the rejected tokens
                    rejected_candidate_topk_indices = candidate_topk_indices[:, n_matches_int, :]
                    rejected_candidate_topk_probs = candidate_topk_probs[:, n_matches_int, :]
                    metrics['tokens_rejected_tokens_topk'].extend(rejected_candidate_topk_indices.tolist())
                    metrics['value_probability_rejected_topk'].extend(rejected_candidate_topk_probs.tolist())

                else:
                    logits_total = candidate_outputs['candidate_logits_total']
                    for _drafting, candidate_logits_drafting in logits_total.items():
                        candidate_probs_drafting = F.softmax(candidate_logits_drafting, dim=-1)
                        candidate_topk_probs_drafting, candidate_topk_indices_drafting = torch.topk(candidate_probs_drafting, k=self._config['logging_top_k'] , dim=-1)

                        rejected_candidate_topk_indices_drafting = candidate_topk_indices_drafting[:, n_matches_int, :]
                        rejected_candidate_topk_probs_drafting = candidate_topk_probs_drafting[:, n_matches_int, :]

                        # Log the top-k probabilities and indices for the rejected tokens
                        if _drafting not in metrics['tokens_rejected_tokens_topk']:
                            metrics['tokens_rejected_tokens_topk'][_drafting] = rejected_candidate_topk_indices_drafting.tolist()
                            metrics['value_probability_rejected_topk'][_drafting] = rejected_candidate_topk_probs_drafting.tolist()
                        else:
                            metrics['tokens_rejected_tokens_topk'][_drafting].extend(rejected_candidate_topk_indices_drafting.tolist())  # pick batch index 0
                            metrics['value_probability_rejected_topk'][_drafting].extend(rejected_candidate_topk_probs_drafting.tolist())  # pick batch index 0

            if new_attentions is not None:
                metrics['value_image_attention_drf_first_rejected'].append(new_attentions[n_matches_int])
                q_i, p_i, probability_ratio = get_probability_along_tokens(candidate_logits, new_logits, candidate_new_tokens, candidate_length, output_ratio=True)
                metrics['value_probability_ratio_first_rejected'].append(probability_ratio[n_matches_int])
                metrics['value_probability_first_rejected_drf'].append(q_i[n_matches_int])
                metrics['value_probability_first_rejected_tgt'].append(p_i[n_matches_int])


        input_ids = torch.cat((input_ids, valid_tokens), dim=-1)
        if streamer is not None:
            streamer.put(valid_tokens.cpu())
        new_cur_len = input_ids.shape[-1]

        # 4.2. Discard past key values relative to unused assistant tokens
        candidate_generator._set_num_rejected_vlm(candidate_length - (valid_tokens.shape[1] - 1))
        new_cache_size = outputs.past_key_values.get_seq_length() - candidate_generator._get_num_rejected_vlm()
        outputs.past_key_values = _crop_past_key_values(self, outputs.past_key_values, new_cache_size)

        # 5. Update the candidate generation strategy if needed
        # candidate_generator.update_candidate_strategy(input_ids, new_logits, n_matches)

        if synced_gpus and this_peer_finished:
            continue  # don't waste resources running the code we don't need

        # Store scores, attentions and hidden_states when required
        # Assistant: modified to append one tuple element per token, as in the other generation methods.

        if new_attentions is not None and candidate_generator._config['output_image_attentions'] and n_matches > 0:
            metrics["value_image_attention_drf_accepted"].append(new_attentions[:n_matches])
            metrics["ids_image_attention_drf_accepted"].append(list(range(cur_len, cur_len + n_matches))) 
            q_i, p_i, probability_ratio = get_probability_along_tokens(candidate_logits, new_logits, candidate_new_tokens, candidate_length, output_ratio=True)
            metrics['value_probability_ratio_accepted'].append(probability_ratio[:n_matches])
            metrics['value_probability_accepted_drf'].append(q_i[:n_matches])
            metrics['value_probability_accepted_tgt'].append(p_i[:n_matches])
            
        if return_dict_in_generate:
            if output_scores:
                scores += tuple(new_logits[:, i, :] for i in range(n_matches + 1))
            if output_logits:
                raw_logits += (next_token_logits,)

            if "past_key_values" not in model_kwargs or start_from_empty_dynamic_cache:
                added_len = new_cur_len
                # set it to false for other iterations
                start_from_empty_dynamic_cache = False
            else:
                added_len = n_matches + 1

            if output_attentions:
                if self.config.is_encoder_decoder:
                    cross_attentions = _split_model_outputs(
                        cross_attentions, outputs.cross_attentions, cur_len, added_len
                    )
                    decoder_attentions = _split_model_outputs(
                        decoder_attentions,
                        outputs.decoder_attentions,
                        cur_len,
                        added_len,
                        is_decoder_attention=True,
                    )
                else:
                    decoder_attentions = _split_model_outputs(
                        decoder_attentions,
                        outputs.attentions,
                        cur_len,
                        added_len,
                        is_decoder_attention=True,
                    )
            if output_hidden_states:
                if self.config.is_encoder_decoder:
                    decoder_hidden_states = _split_model_outputs(
                        decoder_hidden_states, outputs.decoder_hidden_states, cur_len, added_len
                    )
                else:
                    decoder_hidden_states = _split_model_outputs(
                        decoder_hidden_states, outputs.hidden_states, cur_len, added_len
                    )

        model_kwargs = self._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=self.config.is_encoder_decoder,
            num_new_tokens=n_matches + 1,
        )

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
        this_peer_finished = unfinished_sequences.max() == 0

    if streamer is not None:
        streamer.end()

    if (
        hasattr(candidate_generator, "assistant_model")
        and candidate_generator.assistant_model.generation_config.num_assistant_tokens_schedule == "heuristic"
    ):
        candidate_generator.assistant_model.generation_config.num_assistant_tokens = (
            candidate_generator.num_assistant_tokens
        )
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
                metrics=metrics,
            )
    else:
        return input_ids

def get_probability_along_tokens(candidate_logits, new_logits, candidate_new_tokens, candidate_length, output_ratio=False, merge_prob=False, mm_index=None, mm_weight=None):
    
    # Compute softmax to get probabilities from logits
    q = candidate_logits.softmax(dim=-1)
    
    if merge_prob:
        # Sum the probability across axis 0 and normalize the sum to 1
        # candidate_logits.shape: torch.Size([2, 5, 32064])

        if isinstance(mm_weight, str):
            print(f"[TBD] mm_weight is {mm_weight}")
            return None, None, [-1, -1, -1, -1, -1]
        
        q[mm_index] *= mm_weight

        # Sum along axis 0 (merge the probabilities)
        q_sum = torch.sum(q, dim=0)

        # Normalize to sum to 1 along the last dimension
        q = q_sum / q_sum.sum(dim=-1, keepdim=True)
        q = q.unsqueeze(0)

    # Get the probabilities along the token indices for candidate_new_tokens
    q_i = q[:, torch.arange(candidate_length), candidate_new_tokens].squeeze(0, 1)

    # Compute softmax for new logits
    p = new_logits.softmax(dim=-1)[:, :-1]
    p_i = p[:, torch.arange(candidate_length), candidate_new_tokens].squeeze(0, 1)
    
    if output_ratio:
        # Return q_i, p_i, and the ratio between p_i and q_i
        return q_i.tolist(), p_i.tolist(), (p_i / q_i).tolist()
    
    return q_i.tolist(), p_i.tolist()

def _speculative_sampling(
    candidate_input_ids,
    candidate_logits,
    candidate_length,
    new_logits,
    is_done_candidate,
):
    """
    Applies sampling as in the speculative decoding paper (https://arxiv.org/pdf/2211.17192.pdf, algorithm 1). Returns
    the selected tokens, as well as the number of candidate matches.

    NOTE: Unless otherwise stated, the variable names match those in the paper.
    """
    new_candidate_input_ids = candidate_input_ids[:, -candidate_length:]
    # Gets the probabilities from the logits. q_i and p_i denote the assistant and model probabilities of the tokens
    # selected by the assistant, respectively.
    q = candidate_logits.softmax(dim=-1)
    q_i = q[:, torch.arange(candidate_length), new_candidate_input_ids].squeeze(0, 1)
    p = new_logits.softmax(dim=-1)
    p_i = p[:, torch.arange(candidate_length), new_candidate_input_ids].squeeze(0, 1)
    probability_ratio = p_i / q_i
    print(probability_ratio)

    # When probability_ratio > 1 (i.e. q_i(x) < p_i(x), or "assistant probability of the candidate token is smaller
    # than the model probability for the same token"), keep the token. Otherwise reject with p = 1 - probability_ratio
    # (= keep with p = probability_ratio). Keep all the tokens until the first rejection
    r_i = torch.rand_like(probability_ratio)
    is_accepted = r_i <= probability_ratio
    n_matches = ((~is_accepted).cumsum(dim=-1) < 1).sum()  # this is `n` in algorithm 1

    # Ensure we don't generate beyond max_len or an EOS token (not in algorithm 1, but needed for correct behavior)
    if is_done_candidate and n_matches == candidate_length:
        # Output length is assumed to be `n_matches + 1`. Since we won't generate another token with the target model
        # due to acceptance on EOS we fix `n_matches`
        n_matches -= 1
        valid_tokens = new_candidate_input_ids[:, : n_matches + 1]
    else:
        # Next token selection: if there is a rejection, adjust the distribution from the main model before sampling.
        gamma = candidate_logits.shape[1]
        p_n_plus_1 = p[:, n_matches, :]
        if n_matches < gamma:
            q_n_plus_1 = q[:, n_matches, :]
            p_prime = torch.clamp((p_n_plus_1 - q_n_plus_1), min=0)
            p_prime.div_(p_prime.sum())
        else:
            p_prime = p_n_plus_1
        t = torch.multinomial(p_prime, num_samples=1).squeeze(1)[None, :]

        # The selected tokens include the matches (if any) plus the next sampled tokens
        if n_matches > 0:
            valid_tokens = torch.cat((new_candidate_input_ids[:, :n_matches], t), dim=-1)
        else:
            valid_tokens = t

    return valid_tokens, n_matches


def _split_model_outputs(outputs, new_outputs, cur_len, added_len, is_decoder_attention=False):
    """
    Given the (decoder/cross attentions)/(decoder hidden states) for multiple generated tokens, splits it into a tuple
    where each member corresponds to a single generated token.
    """
    # Retrocompatibility: in our generation functions, the first iteration includes the attention/hidden states for the
    # prompt.
    if len(outputs) == 0:
        new_tuple = ()
        for layer in new_outputs:
            last_dim_size = cur_len if is_decoder_attention else layer.shape[-1]
            new_tuple += (layer[..., :cur_len, :last_dim_size],)
        outputs += (new_tuple,)
        # The first iteration contains the prompt + 1 generated token, let's update the length variables accordingly
        cur_len += 1
        added_len -= cur_len

    for i in range(added_len):
        new_tuple = ()
        for layer in new_outputs:
            last_dim_size = cur_len + i if is_decoder_attention else layer.shape[-1]
            new_tuple += (layer[..., i : i + 1, :last_dim_size],)
        outputs += (new_tuple,)
    return outputs