import copy
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, List
from types import MethodType
import torch
import torch.nn.functional as F
from einops import rearrange
import numpy as np
import math
import random
from itertools import product

from transformers.modeling_utils import PreTrainedModel
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList, MinLengthLogitsProcessor
from transformers.generation.candidate_generator import CandidateGenerator, _prepare_attention_mask, _prepare_token_type_ids, _crop_past_key_values
from transformers.utils import is_torchdynamo_compiling

from ..promptsetter import PromptSetter
from .modeling_llava_vlmsd import _merge_input_ids_with_image_features_image_top_k
from ...utils.util import patch_function, noop_context
from transformers.models.llava.modeling_llava import LlavaForConditionalGeneration
from transformers.cache_utils import Cache
from ..w_grid import generate_w_grid, get_best_from_w_grid


class CandidateGenerator:
    """Abstract base class for all candidate generators that can be applied during assisted generation."""

    def get_candidates(self, input_ids: torch.LongTensor) -> Tuple[torch.LongTensor, Optional[torch.FloatTensor]]:
        """
        Fetches the candidates to be tried for the current input.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)

        Return:
            `torch.LongTensor` of shape `(batch_size, candidate_length)` containing the candidate sequences to be
            assessed by the model and, optionally, a `torch.FloatTensor` of shape `(batch_size, candidate_length,
            vocabulary_size)` containing the logits associated to each candidate.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can call `get_candidates`."
        )

    def update_candidate_strategy(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, num_matches: int):
        """
        Updates the candidate generation strategy based on the outcomes.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
            scores (`torch.FloatTensor` of shape `(batch_size, candidate_length, config.vocab_size)`):
                Prediction scores of a language modeling head. These can be logits for each vocabulary when not using
                beam search or log softmax for each vocabulary token when using beam search
            num_matches (`int`):
                The number of matches between the candidate sequences and the model predictions.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can call "
            "`update_candidate_strategy`."
        )
    
    def _set_num_rejected_vlm(self, num_rejected_token_vlm):
        self.assistant_model.num_rejected_token_vlm = num_rejected_token_vlm

    def _get_num_rejected_vlm(self):
        return self.assistant_model.num_rejected_token_vlm
    

class AssistedCandidateGeneratorVLM(CandidateGenerator):
    """
    `CandidateGenerator` class to be used for assisted generation and speculative decoding. This class generates
    candidates through the use of a smaller model. Read the following blog post for more information:
    https://huggingface.co/blog/assisted-generation

    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
        assistant_model (`PreTrainedModel`):
            The model to be used for generating candidates. This model should be smaller than the main model.
        generation_config (`~generation.GenerationConfig`, *optional*):
            The generation configuration to be used as base parametrization for the generation call.
        logits_processor (`LogitsProcessorList`):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        model_kwargs (`Dict`):
            The keyword arguments that will be passed to the main model, and are used as base inputs for the assistant
            model as well.
        inputs_tensor (`torch.Tensor`, *optional*):
            The model input tensor. In encoder-decoder models, this is the encoder input.
    """

    def __init__(
        self,
        input_ids: torch.LongTensor,
        assistant_model: "PreTrainedModel",
        generation_config: "GenerationConfig",
        model_kwargs: Dict,
        inputs_tensor: Optional[torch.Tensor] = None,
        logits_processor: "LogitsProcessorList" = None,
        prompt_setter: "PromptSetter" = None,
        config: Dict = None,
    ):
        # VLMSD
        if config is not None:
            self._config = config
        else:
            from ...config import capture_config
            self._config = capture_config()

        # Make sure all data at the same device as assistant model
        device = assistant_model.device
        input_ids = input_ids.to(device)
        if inputs_tensor is not None:
            inputs_tensor = inputs_tensor.to(device)

        # Prepare the assistant and the starting number of candidate tokens
        self.assistant_model = assistant_model
        self.num_assistant_tokens = assistant_model.generation_config.num_assistant_tokens

        # VLMSD - drafting
        self.prompt_setter = prompt_setter
        input_ids = prompt_setter.manipulated_input_ids
        self.assistant_model._config = self._config

        if (self._config['image_top_k_attention'] > 0) or self._config['output_image_attentions']:
            assert self._config['drafting'] in ['multimodal', 'image-pool']
            self.assistant_model._merge_input_ids_with_image_features = MethodType(
                _merge_input_ids_with_image_features_image_top_k, self.assistant_model
            )

        # Set eos in assistant same as in target model
        self.assistant_model.generation_config.eos_token_id = generation_config.eos_token_id

        # Prepare the kwargs for the assistant model
        assistant_kwargs = {}
        architecture_draft_model = assistant_model.config.architectures[0]
        if not (architecture_draft_model=="LlavaForConditionalGeneration" or architecture_draft_model.endswith("ForCausalLM")):
            raise ValueError(
                "The assistant model should be a decoder-only model, like `LlavaForConditionalGeneration` or `XXForCausalLM`."
            )
        for key, value in model_kwargs.items():  # deepcopy crashes if we attempt to copy encoder outputs with grads
            if key not in ("encoder_outputs", "assistant_encoder_outputs", "past_key_values"):
                if key=='pixel_values' and prompt_setter.drafting in ['text-only', 'tokenized-image', 'special-token', 'caption']:
                    # VLMSD: Text-only drafting: no need to pass pixel_values to the assistant model
                    continue
                if key=='attention_mask' and prompt_setter.attention_mask_initial is not None and value.size(1) != prompt_setter.attention_mask_initial.size(1):
                    value = prompt_setter.attention_mask_initial
                assistant_kwargs[key] = (
                    value.detach().to(device) if isinstance(value, torch.Tensor) else copy.deepcopy(value)
                )
        if "assistant_encoder_outputs" in model_kwargs:
            assistant_kwargs["encoder_outputs"] = model_kwargs["assistant_encoder_outputs"]
        elif assistant_model.config.is_encoder_decoder:
            inputs_tensor, model_input_name, assistant_kwargs = assistant_model._prepare_model_inputs(
                inputs_tensor, assistant_model.generation_config.bos_token_id, assistant_kwargs
            )
            assistant_kwargs = assistant_model._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, assistant_kwargs, model_input_name, assistant_model.generation_config
            )
        elif "encoder_outputs" in model_kwargs:
            assistant_kwargs["encoder_outputs"] = model_kwargs["encoder_outputs"]
        self.assistant_kwargs = assistant_kwargs

        # Prepare assistant model's keys of inputs
        if assistant_model.config.is_encoder_decoder:
            # both are encoder-decoder
            self.input_ids_key = "decoder_input_ids"
        elif "encoder_outputs" in assistant_kwargs:
            # special case for encoder-decoder with decoder-only assistant (like DistilWhisper)
            self.input_ids_key = "input_ids"
            self.assistant_kwargs["attention_mask"] = self.assistant_kwargs.get(
                "decoder_attention_mask",
                torch.ones((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.long),
            )
        else:
            # both are decoder-only
            self.input_ids_key = "input_ids"

        # Prepare generation-related options.
        self.logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        self.generation_config = copy.deepcopy(generation_config)
        self.generation_config.return_dict_in_generate = True
        self.generation_config.output_scores = True

        # Disable sampling -- this implementation of assisted generation/speculative decoding uses the assistant
        # greedily to maximize matches. Disables sampling-related flags to prevent warnings
        self.generation_config.do_sample = False
        for attr in ("temperature", "top_p", "min_p", "typical_p", "top_k", "epsilon_cutoff", "eta_cutoff"):
            setattr(self.generation_config, attr, None)

        # avoid unnecessary warnings that min_length is larger than max_new_tokens
        # remove the `MinLengthLogitsProcessor` if exists (NOTE: no need to check for `MinNewTokensLogitsProcessor`)
        self.main_model_min_length = self.generation_config.min_length
        self.generation_config.min_length = 0
        self.generation_config.min_new_tokens = None
        for processor in self.logits_processor:
            if isinstance(processor, MinLengthLogitsProcessor):
                raise ValueError(
                    "Passing `MinLengthLogitsProcessor` when using `assisted_generation is disabled. "
                    "Please pass in `min_length` into `.generate()` instead"
                )

        # We need to roll back the cache in assisted generation, only DynamicCache is supported
        self.generation_config.cache_implementation = None

    def get_candidates(self, input_ids: torch.LongTensor, output_hidden_states: bool = False, last_target_logits=None) -> Tuple[torch.LongTensor, Optional[torch.FloatTensor]]:
        """
        Fetches the candidates to be tried for the current input.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)

        Return:
            `torch.LongTensor` of shape `(batch_size, candidate_length)` containing the candidate sequences to be
            assessed by the model and a `torch.FloatTensor` of shape `(batch_size, candidate_length,
            vocabulary_size)` containing the logits associated to each candidate.
        """
        input_ids = input_ids.to(self.assistant_model.device)
        # Don't generate more than `max_length - 1` candidates since the target model generates one extra token.
        max_new_tokens = min(int(self.num_assistant_tokens), self.generation_config.max_length - input_ids.size(1) - 1)
        min_new_tokens = max(min(max_new_tokens, self.main_model_min_length - input_ids.size(1)), 0)

        # VLMSD - various drafting
        input_ids = self.prompt_setter.get_resulting_input(input_ids)
        new_cur_len = input_ids.shape[-1]
        if max_new_tokens == 0:
            return dict(
                candidate_ids=self.prompt_setter.rollback_to_original_prompt(input_ids),
                candidate_logits=None,
                time_prefill=None,
                num_prefill_tokens=None,
                time_prompt_process=None,
                # Todo: output hidden states
            )

        # 1. If it is not the first round of candidate generation, prepare the inputs based on the input_ids length
        # (which implicitly contains the number of accepted candidates from the previous round)
        has_past_key_values = self.assistant_kwargs.get("past_key_values", None) is not None
        if has_past_key_values:
            new_cache_size = self.assistant_kwargs.get("past_key_values")[0][0].shape[2] - self._get_num_rejected_vlm()
            self.assistant_kwargs["past_key_values"] = _crop_past_key_values(
                self.assistant_model, self.assistant_kwargs["past_key_values"], new_cache_size
            )

            self.assistant_kwargs = _prepare_attention_mask(
                self.assistant_kwargs, new_cur_len, self.assistant_model.config.is_encoder_decoder
            )
            self.assistant_kwargs = _prepare_token_type_ids(self.assistant_kwargs, new_cur_len)

        # 2. Forecast next N tokens using the assistant model.
        assistant_generation_kwargs = {
            self.input_ids_key: input_ids,
            "min_new_tokens": min_new_tokens,
            "max_new_tokens": max_new_tokens,
            "generation_config": self.generation_config,
            "logits_processor": self.logits_processor,
            "output_attentions": self._config['output_image_attentions'],
            "output_hidden_states": output_hidden_states,
        }

        assistant_output = self.assistant_model.generate(**assistant_generation_kwargs, **self.assistant_kwargs)

        # 3. Update variables for the next round of candidaɢte generation
        self.assistant_kwargs["past_key_values"] = assistant_output.past_key_values

        # 4. Prepare variables for output
        candidate_logits = torch.stack(assistant_output.scores, dim=1)
        candidate_ids = assistant_output.sequences
        
        # VLMSD - drafting: rollback to original prompt
        candidate_ids_rollback = self.prompt_setter.rollback_to_original_prompt(candidate_ids)

        if assistant_output.time_prefill is not None and hasattr(self.assistant_model, "time_prompt_process"):
            time_prompt_process = self.assistant_model.time_prompt_process 
        else:
            time_prompt_process = None

        candidate_outputs = dict(
            candidate_ids=candidate_ids_rollback,
            candidate_logits=candidate_logits,
            time_prefill=assistant_output.time_prefill,
            num_prefill_tokens=assistant_output.num_prefill_tokens,
            time_prompt_process=time_prompt_process,
        )

        # VLMSD - drafting: store attentions for image tokens if needed
        if self._config['output_image_attentions']:
            candidate_outputs['draft_image_attentions'] = self.get_image_attentions(assistant_output.attentions, self.assistant_model.attention_kwargs)
        if output_hidden_states:
            candidate_outputs['draft_hidden_states'] = assistant_output.hidden_states
        
        return candidate_outputs

    def get_image_attentions(self, attentions, attention_kwargs):
        image_regions = attention_kwargs['image_regions'][0]  # Indices of image tokens
        
        # Initialize a list to store image attention values
        image_attentions = []

        # Iterate over generated tokens
        for token_attention in attentions:
            # Initialize a list for the current token's image attention values across all layers
            token_image_attentions = []
            
            # Iterate over layers
            for layer_attention in token_attention:
                # `layer_attention` has shape (batch_size, num_heads, generated_length, sequence_length)
                
                # Select the attention values corresponding to `image_regions` indices
                # We index the last dimension (sequence_length) with `image_regions`
                # The resulting shape will be (batch_size, num_heads, generated_length, len(image_regions))
                # Only choose the last token (for prefill)
                layer_image_attention = layer_attention[..., -1:, image_regions]
                
                # Append the result for this layer
                token_image_attentions.append(layer_image_attention.tolist())
            
            # Append the result for this token
            image_attentions.append(tuple(token_image_attentions))
        
        # Return as a tuple of tuples to match the input structure
        return tuple(image_attentions)
    
    def update_candidate_strategy(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, num_matches: int):
        """
        Updates the candidate generation strategy based on the outcomes.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
            scores (`torch.FloatTensor` of shape `(batch_size, candidate_length, config.vocab_size)`):
                Prediction scores of a language modeling head. These can be logits for each vocabulary when not using
                beam search or log softmax for each vocabulary token when using beam search
            num_matches (`int`):
                The number of matches between the candidate sequences and the model predictions.
        """
        # Adjust the max number of assistant tokens to use in the next iteration. This is a simple heuristic,
        # probably can be improved -- we want to balance the benefits of getting assistant tokens correct with the
        # cost of forecasting incorrect assistant tokens.
        if self.assistant_model.generation_config.num_assistant_tokens_schedule in {
            "heuristic",
            "heuristic_transient",
        }:
            if num_matches == int(self.num_assistant_tokens):
                self.num_assistant_tokens += 2.0
            else:
                self.num_assistant_tokens = max(1.0, self.num_assistant_tokens - 1.0)

class AssistedCandidateGeneratorTabedVLM(CandidateGenerator):
    def __init__(
        self,
        input_ids: torch.LongTensor,
        assistant_model: "PreTrainedModel",
        generation_config: "GenerationConfig",
        model_kwargs: Dict,
        inputs_tensor: Optional[torch.Tensor] = None,
        logits_processor: "LogitsProcessorList" = None,
        prompt_setter: "PromptSetter" = None,
        config: Dict = None,
    ):
        # VLMSD
        if config is not None:
            self._config = config
        else:
            from ...config import capture_config
            self._config = capture_config()
        self.prompt_setter = prompt_setter

        self.generation_config = copy.copy(generation_config)
        self.num_assistant_tokens = generation_config.num_assistant_tokens
        assistant_model.generation_config.num_assistant_tokens = 1 

        self.prompt_setter = prompt_setter
        self.num_drafting = len(prompt_setter.drafting)

        self.limit = 1e5

        # TABED functionality
        self.tabed_rule = self._config['tabed_rule']
        self.mm_weight_policy = self._config['mm_weight_policy']

        if self._config['history_dependent']:
            self.num_block = 0
            self.cum_accepted_len = 0

        self.candidate_generator_tabed = {}
        for drafting in self._config['drafting']:
            self.candidate_generator_tabed[drafting] = AssistedCandidateGeneratorVLM(
                input_ids=input_ids,
                assistant_model=copy.deepcopy(assistant_model),
                generation_config=generation_config,
                model_kwargs=model_kwargs,
                inputs_tensor=inputs_tensor,
                logits_processor=logits_processor,
                prompt_setter=self.set_prompt_setter(prompt_setter, drafting),
                config=self._config,
            )

        if self._config['history_dependent']:
            self.cum_history_dependent_weight = {drafting: [] for drafting in self.prompt_setter.drafting}

    def set_prompt_setter(self, prompt_setter_tabed, drafting):
        prompt_setter = copy.copy(prompt_setter_tabed)
        prompt_setter.drafting = drafting
        prompt_setter.manipulated_input_ids = prompt_setter_tabed.manipulated_input_ids.get(drafting)
        prompt_setter.manipulated_input_ids_length_initial = prompt_setter_tabed.manipulated_input_ids_length_initial.get(drafting)
        prompt_setter.attention_mask_initial = prompt_setter_tabed.attention_mask_initial.get(drafting)
        return prompt_setter

    def get_candidates(self, input_ids: torch.LongTensor, last_target_logits=None) -> Dict[str, Any]:
        inputs = {
            "input_ids": input_ids,
            "logits": None,  # Initialize logits to None
            "logits_total": {},
        }

        if not self._config['history_dependent']:
            self.history_dependent_weight = {key: 0 for key in self.prompt_setter.drafting}
        else:
            # get gt for each drafting and calculate the weight
            history_window = self._config['history_window']

            if self.num_block == 0:
                if self._config['history_item'] == 'w-grid':
                    self.history = {key: torch.tensor([]) for key in self.prompt_setter.drafting + ['target']}
                    interval_grid = 1 / self._config['history_w_num_grid']
                    self.w_grid_tensor = generate_w_grid(self.num_drafting, interval_grid)
                    self.w_grid_tensor = torch.from_numpy(self.w_grid_tensor).float().to(input_ids.device)
                else:
                    self.history = {key: [] for key in self.prompt_setter.drafting}
                self.history_dependent_weight = {key: 0 for key in self.prompt_setter.drafting}
                
            else:
                num_new_tokens = input_ids.size(1) - self.last_input_ids_len
                self.cum_accepted_len += min(num_new_tokens, self.num_assistant_tokens)
                p_prob_except_bonus_token = F.softmax(last_target_logits[:, :min(num_new_tokens, self.num_assistant_tokens)], dim=-1)
                num_vocab = p_prob_except_bonus_token.size(-1)
                
                if self._config['history_item'] == 'random':
                    # Generate random weights that sum to 1
                    keys = list(self.prompt_setter.drafting)
                    random_values = [random.random() for _ in keys]
                    total = sum(random_values)
                    self.history_dependent_weight = {key: val / total for key, val in zip(keys, random_values)}
                
                elif self._config['history_item'] in ['block-efficiency', 'kld', 'tvd', 'adaboost']:
                    if self._config['history_item'] in ['block-efficiency', 'adaboost']:
                        accepted_ids = input_ids[:, self.last_input_ids_len: self.last_input_ids_len+5]
                        current_step = {k: (v[:accepted_ids.size(1)] == accepted_ids).sum(dim=-1).tolist() for k, v in self.last_drafting_ids.items()}
                        """
                        αₘ = ln((1 − εₘ) / εₘ) + ln(K − 1)
                        - εₘ: Error rate of each model for history window => accepted_ids의 length cum 필요?
                        - K: vocab size
                        """
                        if self._config['history_item'] == 'block-efficiency': 
                            agg_ftn = sum
                        
                        elif self._config['history_item'] == 'adaboost':
                            agg_ftn = lambda x: (
                                self.limit if (x_sum := sum(x)) == self.cum_accepted_len else
                                - self.limit if x_sum == 0 else
                                math.log(x_sum / (self.cum_accepted_len - x_sum)) + self._config['history_adaboost_constant_weight'] * math.log(num_vocab - 1)
                            )
                        else:
                            raise NotImplementedError(f"History item {self._config['history_item']} is not implemented.")

                    elif self._config['history_item'] in ['kld', 'tvd']:
                        # get gt history
                        loss_ftn = torch.nn.KLDivLoss if self._config['history_item'] == 'kld' else torch.nn.L1Loss
                        left_term_ftn = F.log_softmax if self._config['history_item'] == 'kld' else F.softmax

                        if self._config['history_unit']=="block":
                            reduction = 'sum'
                            reduction_ftn = lambda x: [x.item()]
                        else:
                            reduction = 'none'
                            reduction_ftn = lambda x: x.sum(dim=-1).squeeze(0).tolist()

                        criterion = loss_ftn(reduction=reduction)
                        current_step = {
                            k: reduction_ftn(
                                criterion(
                                    left_term_ftn(v[:, :num_new_tokens], dim=-1),
                                    p_prob_except_bonus_token,
                                )
                            ) for k, v in self.last_drafting_logits.items()
                        }

                        # sum and inverse
                        agg_ftn = lambda x: 1/sum(x) if sum(x) != 0 else 1e11
                
                    self.history = {k: self.history[k] + current_step[k] for k in self.history}
                    self.history_dependent_weight = {k: agg_ftn(v[-self._config['history_window']:]) for k, v in self.history.items()}

                elif self._config['history_item'] == 'w-grid':
                    # get all probabilities
                    current_step = {k: F.softmax(v, dim=-1)[:, :num_new_tokens].cpu() for k, v in self.last_drafting_logits.items()}
                    current_step['target'] = p_prob_except_bonus_token.cpu()

                    # con6 self.history[k] and current_step[k]
                    self.history = {k: torch.cat([self.history[k], current_step[k]], dim=1) for k in self.history}

                    # modify the history (window, top1-matching, reshape - optional)
                    _history_current, is_history_available = get_current_history(self.history, self._config)

                    if not is_history_available:
                        self.history_dependent_weight = {key: 1 for key in self.prompt_setter.drafting}

                    else:
                        _history_current_drafting_concat = np.concatenate([rearrange(_history_current[k], "b s v -> s b v") for k in self.history.keys() if k != 'target'], axis=1) # (history len, num_drafting, vocab_size)=
                        best_w_from_grid = get_best_from_w_grid(
                            self.w_grid_tensor,
                            _history_current_drafting_concat,
                            _history_current['target'],
                            history_w_grid_measure=self._config['history_w_grid_measure'],
                            history_w_grid_num_accepted_order=self._config['history_w_grid_num_accepted_order'],
                            history_w_grid_num_accepted_lenience=self._config['history_w_grid_num_accepted_lenience'],
                        )
                        self.history_dependent_weight = {
                            k: best_w_from_grid[i].item() for i, k in enumerate(self.history.keys()) if k != 'target'
                        }

            
            if set(self.history_dependent_weight.values()) in [{0}, {-self.limit}]:
                # Your code here when the condition is satisfied
                self.history_dependent_weight = {key: 1 for key in self.prompt_setter.drafting}

            self.num_block += 1
            self.last_input_ids_len = input_ids.size(1)
            self.last_drafting_ids = {key: None for key in self.prompt_setter.drafting}
            self.last_drafting_logits = {key: None for key in self.prompt_setter.drafting}
            for drafting, weight in self.history_dependent_weight.items():
                self.cum_history_dependent_weight[drafting].append(weight) 

        eos_token_id = self.generation_config.eos_token_id  # Get the EOS token ID from the generation config

        for i in range(self.num_assistant_tokens):
            if i == 0:
                context = noop_context()
            else:
                context = patch_function(
                    LlavaForConditionalGeneration, 
                    '_get_initial_cache_position', 
                    _get_initial_cache_position_tabed,
                )
            
            with context:
                next_tokens_outputs = self.get_next_tokens(inputs['input_ids'])

                next_token, best_index = self.pick_next_token(next_tokens_outputs)

                if next_token is None:
                    break                    
                
                if self._config['history_dependent']:
                    # Update the drafting history with next_token_id for this iteration
                    for key, outputs in next_tokens_outputs.items():
                        next_token_id = outputs['next_token_id'].squeeze()  # Remove unnecessary dimensions

                        # Initialize history for the key if it's the first iteration
                        if self.last_drafting_ids[key] is None:
                            self.last_drafting_ids[key] = next_token_id.unsqueeze(0)  # Add a batch dimension (for stacking later)
                        else:
                            self.last_drafting_ids[key] = torch.cat((self.last_drafting_ids[key], next_token_id.unsqueeze(0)), dim=0)

                        if self.last_drafting_logits[key] is None:
                            self.last_drafting_logits[key] = outputs['next_token_logit'].unsqueeze(0)
                        else:
                            self.last_drafting_logits[key] = torch.cat((self.last_drafting_logits[key], outputs['next_token_logit'].unsqueeze(0)), dim=1)                        

                # Update num_rejected_vlm for the candidate generator
                self._set_num_rejected_vlm(best_index=best_index)

                # Update inputs, including candidate_logits
                inputs = self.update_inputs(inputs, next_token, next_tokens_outputs, best_index)

                # Break if next_token == eos_token
                if (next_token == eos_token_id).all():
                    break

        candidate_outputs = dict(
            candidate_ids=inputs['input_ids'],
            candidate_logits=inputs['logits'],  # Updated to include combined logits of picked tokens
            candidate_logits_total=inputs['logits_total'],  # Include all logits for debugging
            time_prefill=None,
            num_prefill_tokens=None,
        )

        return candidate_outputs

    def get_next_tokens(self, input_ids: torch.LongTensor):
        per_drafting_outputs = {}

        for drafting, candidate_generator in self.candidate_generator_tabed.items():
            candidate_generator.assistant_model.tabed_current_drafting = drafting
            candidate_outputs = candidate_generator.get_candidates(input_ids)
            del candidate_generator.assistant_model.tabed_current_drafting
            candidate_ids = candidate_outputs['candidate_ids']
            candidate_logits = candidate_outputs['candidate_logits']

            if candidate_logits is None:
                per_drafting_outputs[drafting] = {
                    'next_token_id': None,
                    'next_token_log_prob_value': None,
                    'next_token_logit': None
                }
                continue

            # Get the next token (the last token in candidate_ids)
            next_token_id = candidate_ids[:, -1].unsqueeze(-1)  # shape (batch_size, 1)
            next_token_logit = candidate_logits[:, -1, :]  # shape (batch_size, vocab_size)

            # Convert logits to probabilities
            next_token_log_probs = F.log_softmax(next_token_logit, dim=-1)
            # Get the log probability assigned to next_token_id
            next_token_log_prob_value = next_token_log_probs.gather(1, next_token_id)  # shape (batch_size, 1)

            per_drafting_outputs[drafting] = {
                'next_token_id': next_token_id,
                'next_token_log_prob_value': next_token_log_prob_value,
                'next_token_logit': next_token_logit  # Include logits for updating inputs
            }

        return per_drafting_outputs

    def pick_next_token(self, next_tokens_outputs):
        per_drafting_outputs = next_tokens_outputs

        # Collect tokens and their log probabilities, ignoring None values
        tokens = []
        logits = []
        log_probs = []
        valid_draftings = []  # Keep track of valid draftings (i.e., where logits are not None)

        for drafting in per_drafting_outputs:
            if per_drafting_outputs[drafting]['next_token_id'] is not None:
                tokens.append(per_drafting_outputs[drafting]['next_token_id'])  # shape (batch_size, 1)
                logits.append(per_drafting_outputs[drafting]['next_token_logit'])  # shape (batch_size, vocab_size)
                log_probs.append(per_drafting_outputs[drafting]['next_token_log_prob_value'])  # shape (batch_size, 1)
                valid_draftings.append(drafting)  # Add only valid draftings
        # Stack tokens and log probabilities along a new dimension if there are valid draftings
        if tokens:
            tokens = torch.cat(tokens, dim=1)  # shape (batch_size, num_draftings)
            log_probs = torch.cat(log_probs, dim=1)  # shape (batch_size, num_draftings)
            logits = torch.stack(logits, dim=1)  # shape (batch_size, num_draftings, vocab_size)

            # Convert history_dependent_weight values to a tensor and normalize the weight to sum to 1 vector
            history_dependent_weight_tensor = torch.tensor(
                [self.history_dependent_weight[key] for key in self.history_dependent_weight.keys()],
                device=tokens.device 
            ).view(1, -1, 1) if self._config['history_dependent'] else torch.ones((1, self.num_drafting, 1), device=tokens.device)
            history_dependent_weight_tensor = history_dependent_weight_tensor/history_dependent_weight_tensor.sum()

            if self.tabed_rule == 'confidence':
                probs = F.softmax(logits, dim=-1)  # shape (batch_size, num_draftings, vocab_size)
                confidence = get_confidence(probs, self._config['confidence_type'])
                history_dependent_weight_tensor_temperatured = get_temperatured_weight(
                    history_dependent_weight_tensor.squeeze(-1), 
                    self._config['temperature_drafting_weight']
                )
                history_weighted_probs = confidence * history_dependent_weight_tensor_temperatured  # shape (batch_size, num_draftings)
                
                best_indices = torch.argmax(history_weighted_probs, dim=1)  # shape (batch_size,)
                best_next_tokens = tokens[torch.arange(tokens.size(0)), best_indices].unsqueeze(-1)  # shape (batch_size, 1)

                # Convert best_indices to correspond to valid_draftings
                best_indices = [valid_draftings[idx] for idx in best_indices.tolist()]  # List of drafting names
            
            elif self.tabed_rule in ['mm-weight', 'conf-weight']:
                if self.tabed_rule == 'mm-weight':
                    # Find the index of the multimodal drafting in the valid_draftings list
                    if 'multimodal' in valid_draftings:
                        dim_1_index_multimodel_drafting = valid_draftings.index('multimodal')
                    else:
                        dim_1_index_multimodel_drafting = valid_draftings.index('caption')

                    if isinstance(self.mm_weight_policy, int) or isinstance(self.mm_weight_policy, float):
                        mm_weight = self.mm_weight_policy
                    
                    elif self.mm_weight_policy == 'img-nec':
                        # Apply softmax to logits to get probabilities
                        probs = F.softmax(logits, dim=-1)  # shape (batch_size, num_draftings, vocab_size)

                        # Get the index for the counterpart drafting (the one that is not dim_1_index_multimodel_drafting)
                        ids_prob = {i for i in range(probs.size(1))}  # Create set of drafting indices
                        ids_prob.remove(dim_1_index_multimodel_drafting)  # Remove multimodel drafting index

                        # Ensure there is exactly one counterpart drafting index
                        assert len(ids_prob) == 1, f"Invalid number of draftings: {len(ids_prob)}"
                        dim_1_index_counterpart_drafting = ids_prob.pop()  # Get the counterpart drafting index

                        # Find the best token from counterpart drafting
                        best_indices_img_nec = torch.argmax(probs, dim=-1)  # shape (batch_size, num_draftings)
                        best_token_from_counterpart = best_indices_img_nec[0, dim_1_index_counterpart_drafting]

                        # Get the probability of the best token from multimodel drafting
                        prob_best_token_mm = probs[0, dim_1_index_multimodel_drafting].max().item()
                        prob_indexed_by_best_token_counterpart = probs[0, dim_1_index_multimodel_drafting, best_token_from_counterpart].item()

                        # Calculate the mm_weight
                        if self._config['mm_weight_k'] is not None:
                            mm_weight = 1 + (prob_best_token_mm - prob_indexed_by_best_token_counterpart) * self._config['mm_weight_k']
                        else:
                            mm_weight = 1 + (prob_best_token_mm - prob_indexed_by_best_token_counterpart)

                    else:
                        raise ValueError(f"Invalid multimodal weight policy: {self.mm_weight_policy}")
                
                if self._config['tabed_input'] == 'probability':
                    # Convert logits to probabilities using softmax
                    probs = F.softmax(logits, dim=-1)  # shape (batch_size, num_draftings, vocab_size)
                    total_weight_tensor = history_dependent_weight_tensor

                    # Weighting
                    if self.tabed_rule == 'mm-weight':
                        total_weight_tensor[:, dim_1_index_multimodel_drafting] *= mm_weight
                    elif self.tabed_rule == 'conf-weight':
                        confidence = get_confidence(probs, self._config['confidence_type'])
                        total_weight_tensor *= confidence.unsqueeze(-1)

                    if 'adaboost' not in self._config['history_item']:  # negative w occurs
                        total_weight_tensor = get_temperatured_weight(total_weight_tensor.squeeze(-1), self._config['temperature_drafting_weight']).unsqueeze(-1)
                    total_weighted_probs = probs * total_weight_tensor  # shape (batch_size, num_draftings)
                    
                    # Sum probabilities across the drafting dimension
                    summed_probs = total_weighted_probs.sum(dim=1)  # shape (batch_size, vocab_size)
                
                elif self._config['tabed_input'] == 'logit':
                    logits[:, dim_1_index_multimodel_drafting] *= mm_weight
                    
                    summed_logits = logits.sum(dim=1)
                    
                    summed_probs = F.softmax(summed_logits, dim=-1)

                else:
                    raise ValueError(f"Invalid TABED input: {self._config['tabed_input']}")

                # Pick the token with the highest summed probability
                best_indices = valid_draftings # List of all drafting names
                best_next_tokens = torch.argmax(summed_probs, dim=-1).unsqueeze(-1)  # shape (batch_size, 1)
                
            else:
                raise ValueError(f"Invalid TABED rule: {self.tabed_rule}")

        else:
            # Handle the case where there are no valid tokens (all None)
            best_next_tokens = None
            best_indices = []

        return best_next_tokens, best_indices

    def update_inputs(self, inputs, next_token, next_tokens_outputs, best_index):
        # Update input_ids by concatenating next_token
        input_ids = inputs['input_ids']
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        inputs['input_ids'] = input_ids

        # Update candidate_logits by concatenating logits of picked tokens (only for best_index)
        logits = inputs['logits']
        logits_total = inputs.get('logits_total', {})  # Initialize logits_total if not already present
        selected_logits = []

        # (1) Gather the logits corresponding to the best_index (2) Store all logits
        for drafting, next_token_output in next_tokens_outputs.items():
            next_token_logits = next_token_output['next_token_logit']  # shape (batch_size, vocab_size)
            # Stack logits_total here for each drafting
            if drafting not in logits_total:
                # If this is the first time encountering the drafting, initialize it
                logits_total[drafting] = next_token_logits.unsqueeze(1)  # shape (batch_size, 1, vocab_size)
            else:
                # Otherwise, concatenate the new logits to the previous ones
                logits_total[drafting] = torch.cat([logits_total[drafting], next_token_logits.unsqueeze(1)], dim=1)

            # If the current drafting is in the best_index, save its logits for selection
            if drafting in best_index:
                selected_logits.append(next_token_logits)

        # Stack selected logits from the drafts that are part of best_index
        selected_logits = torch.stack(selected_logits, dim=0)  # shape (1, vocab_size)

        if logits is None:
            logits = selected_logits  # shape (batch_size, 1, vocab_size)
        else:
            logits = torch.cat([logits, selected_logits], dim=1)  # shape (batch_size, sequence, vocab_size)

        # Update the logits and logits_total in the inputs
        inputs['logits'] = logits
        inputs['logits_total'] = logits_total

        return inputs


    def _set_num_rejected_vlm(self, num_rejected_token_vlm=None, best_index=None):
        """
        Sets `num_rejected_token_vlm` for the current instance or for all instances in the TABED generator.

        Args:
            num_rejected_token_vlm (int): The value to set for `num_rejected_token_vlm`.
            best_index (Optional[List[str]]): If provided, sets `num_rejected_vlm` for each
                                            `AssistantCandidateGeneratorVLM` in the TABED generator.
        """
        if best_index is None:
            # Set num_rejected_vlm for the current instance and all in the TABED generator
            for candidate_generator in self.candidate_generator_tabed.values():
                candidate_generator._set_num_rejected_vlm(num_rejected_token_vlm)
        else:
            # TABED functionality: Set num_rejected_vlm to 1 for all initially
            for candidate_generator in self.candidate_generator_tabed.values():
                candidate_generator._set_num_rejected_vlm(1)
            
            # Set num_rejected_vlm to 0 for candidate generators whose token was selected
            for drafting in set(best_index):
                candidate_generator = self.candidate_generator_tabed[drafting]
                candidate_generator._set_num_rejected_vlm(0)
    
    def _get_num_rejected_vlm(self):
        for candidate_generator in self.candidate_generator_tabed.values():
            return candidate_generator._get_num_rejected_vlm()


def _get_initial_cache_position_tabed(self, input_ids, model_kwargs):
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
            # decode subsequent chunks for VLMSD
            is_fully_accepted = self.num_rejected_token_vlm == 0
            num_additional_cache = 1 if is_fully_accepted else 2
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

def get_confidence(probs, confidence_type='top1-prob'):
    """
    probs: torch.FloatTensor of shape (batch_size, num_draftings, vocab_size)
    """
    if confidence_type == 'top1-prob':
        confidence = probs.max(-1)[0]
    
    elif confidence_type == 'entropy': # 1/Entropy
        epsilon = 1e-12
        probs = probs + epsilon
        # Calculate the entropy along the last dimension (vocab_size)
        confidence = -1/torch.sum(probs * torch.log2(probs), dim=-1) # heuristics: 1/entropy
    
    elif confidence_type == 'top1-prob-cross':
        n = probs.shape[1]

        # Initialize a tensor to store confidence values
        confidence = torch.zeros(1, n, device=probs.device)

        # Calculate confidence for each drafting
        # probs[0, :, probs.argmax(-1)[0]].sum(0)
        for i in range(n):
            current_probs = probs[0, i, :]
            top_index = current_probs.argmax()
            confidence[0, i] = probs[0, :, top_index].sum()

    return confidence

def get_temperatured_weight(weights: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    if temperature == 0:
        # Create a "hard" one-hot encoded tensor for the largest weight in each batch
        max_indices = torch.argmax(weights, dim=1, keepdim=True)
        one_hot = torch.zeros_like(weights).scatter_(-1, max_indices, 1.0)
        return one_hot
    
    logits = torch.log(weights + 1e-12)  # Add a small constant to prevent log(0)
    scaled_logits = logits / temperature
    confidence_temperatured = F.softmax(scaled_logits, dim=1)

    return confidence_temperatured


def get_current_history(history, _config) -> Dict[str, np.ndarray]:
    history_window = _config['history_window']
    history_filter_top1_match = _config['history_filter_top1_match']

    # windowing
    _history_current = {k: v[:, -history_window:].numpy() for k, v in history.items()}

    # indexing by top1 match
    if history_filter_top1_match:
        is_top1_match = check_common_element(
            _history_current['target'].argmax(-1),
            *[_history_current[key].argmax(-1) for key in _history_current.keys() if key != 'target']
        )

        _history_current = {k: v[:, is_top1_match] for k, v in _history_current.items()}

        if is_top1_match.sum() == 0:
            return _history_current, False # _history_current, is_history_available

    return _history_current, True


def check_common_element(target_array:np.ndarray, *arrays):
    results = []
    for i in range(target_array.shape[1]):
        # Create a set for target elements at step i
        target_set = set(target_array[:, i])
        
        # Check for intersection with other arrays at step i
        is_common = any(set(arr[:, i]).intersection(target_set) for arr in arrays)
        
        results.append(is_common)

    return np.array(results)