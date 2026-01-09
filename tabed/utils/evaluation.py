import os
import sys
import re
from typing import List, Dict, Any
from itertools import product
import copy

import numpy as np
import pandas as pd
import logging
from datetime import datetime
import pytz
from transformers import AutoProcessor
from huggingface_hub import login

import matplotlib.pyplot as plt
import io
from PIL import Image

# add root
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from config import ex
from util import avg, get_ckpt_name, get_short_name, get_tabed_name
from util_evaluation import load_metrics, get_size, save_images, _get_markdown, set_config_itr, _merge_list, save_pil_image
from criterions.base_criterion import BaseCriterion


def get_block_efficiency_info_per_sample(metrics):
    """Calculate block efficiency."""
    block_efficiency_info_per_sample = []
    for i, num_accepted_token_list in enumerate(metrics["num_accepted_tokens"]):
        block_efficiency_info_per_sample.append(
            {
                "num_chunks": len(num_accepted_token_list),
                "block_efficiency": 1 + avg(num_accepted_token_list),
            }
        )
    
    return block_efficiency_info_per_sample

def _factorize_time(times: list, prefix: str):
    # Todo: noqa - split the prefill when logging
    time_decode_first_chunk = times[0]
    time_decode_subs_chunks = times[1:] if len(times) > 1 else None # coule be empty

    time_decode_subs_chunks_avg = avg(time_decode_subs_chunks) if time_decode_subs_chunks is not None else None
    time_prefill_approx = (time_decode_first_chunk - time_decode_subs_chunks_avg) if time_decode_subs_chunks_avg is not None else None

    # Todo: decode_subs_chunks_avg
    return {    
        # f"{prefix}_total": times,
        # f"{prefix}_num_chunks": len(times),
        # f"{prefix}_first_chunk": time_decode_first_chunk,
        f"time_decode_chunks_{prefix}": time_decode_subs_chunks,
        # f"{prefix}_prefill_approx": time_prefill_approx,
        # f"{prefix}_decode_subs_chunks_avg": time_decode_subs_chunks_avg,
    }

def get_time_info_per_sample(metrics, _config):
    time_info = []
    
    for i, time in enumerate(metrics['time_total']):
        # basic time info
        time_total = metrics['time_total'][i]
        # time_prefill = metrics['time_prefill_drf'][i] if _config['decoding'] == 'ard' else metrics['time_prefill_drf'][i] + metrics['time_prefill_tgt'][i]
        time_prefill = -1
        time_prompt_process = metrics['time_prompt_process'][i] if 'time_prompt_process' in metrics else 0
        time_decode = time_total - time_prefill - time_prompt_process
        time_info_sample = {
            "time_total": time_total,
            "time_prefill": time_prefill,
            "time_decode": time_decode,
            "time_prompt_process": time_prompt_process,
        }

        if _config['decoding'] == 'sd':
            time_info_sample.update(
                {
                    'time_prefill_drf': -1,
                    # 'time_prefill_drf': metrics['time_prefill_drf'][i],
                    'time_prefill_tgt': metrics['time_prefill_tgt'][i],
                }
            )

        # optional: factorize
        if _config['is_time_factorized']:
            # Todo: debug w/ factorize data 
            time_info_sample.update(_factorize_time(metrics['time_generate_drf'][i], prefix="drf"))
            time_info_sample.update(_factorize_time(metrics['time_verify_tgt'][i], prefix="tgt"))

        time_info.append(time_info_sample)
            
    return time_info


def get_history_info_per_sample(metrics, _config):
    history_info = []
    
    for i, history_info_sample in enumerate(metrics['history_dependent_weights']):
        history_info.append({f"{drafting}_history_weight": weights for drafting, weights in history_info_sample.items()})

    return history_info

def get_sequence_info_per_sample(metrics, _config):
    #(1) whole generated sequence (2) accepted tokens from draft model (3) the first rejected tokens
    tokenizer = AutoProcessor.from_pretrained(_config['drf'])
    sequence_info = []

    for i, sequence in enumerate(metrics['sequences']):
        sequence_info_sample = {
            'sequences_decoded': tokenizer.batch_decode(sequence),
        }

        if _config['decoding'] == 'sd':
            # factorize the sequence for sd
            sequences_decoded_factorized = _factorize_squence_for_sd(
                sequence=sequence,
                ids_accepted_tokens=metrics['ids_accepted_tokens'][i],
                ids_first_rejected_tokens=metrics["ids_first_rejected"][i],
                first_rejected_tokens=metrics["tokens_first_rejected"][i],
                tokenizer=tokenizer,
            )
            # visually separate the factorized elements via markdown
            sequences_decoded_markdown = _get_markdown(sequences_decoded_factorized)
            
            sequence_info_sample.update(
                {
                    'sequences_decoded_factorized': sequences_decoded_factorized,
                    'sequences_decoded_markdown': sequences_decoded_markdown,
                }
            )

            if 'tokens_accepted_tokens_topk' in metrics:
                sequences_decoded_factorized_topk = _factorize_squence_for_sd_topk(
                    sequence=sequence, 
                    ids_accepted_tokens=metrics['ids_accepted_tokens'][i],
                    ids_first_rejected_tokens=metrics["ids_first_rejected"][i],
                    first_rejected_tokens=metrics["tokens_first_rejected"][i],
                    tokens_accepted_tokens_topk=metrics['tokens_accepted_tokens_topk'][i],
                    value_probability_accepted_topk=metrics['value_probability_accepted_topk'][i],
                    tokens_rejected_tokens_topk=metrics['tokens_rejected_tokens_topk'][i],
                    value_probability_rejected_topk=metrics['value_probability_rejected_topk'][i],
                    tokenizer=tokenizer,
                )

                sequence_info_sample.update(
                    {
                        'sequences_decoded_factorized_topk': sequences_decoded_factorized_topk,
                    }
                )

        """if _config['output_image_attentions']:
            # sequence_info_sample['sequences_decoded']: List[str]
            # metrics['value_image_attention_drf_accepted']: List[List[Tuple[Tuple[List[List[List[float]]]]]]] # num_samples x num chunks x num accepted X num layers x num heads x query len (=1) x key len (num image tokens)
            # metrics['ids_image_attention_drf_accepted']: List[List[List[int]]] 

            val_image_attention_drf = aggregate_attention_values(
                metrics['value_image_attention_drf_accepted'][i],
                metrics['ids_image_attention_drf_accepted'][i],
                sequence_info_sample['sequences_decoded'],
            )

            img_image_attention_drf = visualize_attention_values_per_layer(
                val_image_attention_drf,
                metrics['ids_image_attention_drf_accepted'][i],
                sequence_info_sample['sequences_decoded'],
            )
            
            sequence_info_sample.update(
                {
                    "val_image_attention_drf": val_image_attention_drf,
                    "img_image_attention_drf": img_image_attention_drf,
                }
            )"""

        sequence_info.append(sequence_info_sample)
    
    
    return sequence_info

def visualize_attention_values_per_layer(image_attention_drf, ids_image_attention_drf_accepted, sequences_decoded):
    """
    Visualize the attention values for each decoded sequence that attended to image tokens for each separate layer
    and return the visualizations as PIL Image objects.

    Args:
    - image_attention_drf: List of tuples where each tuple contains a string (token) and a list of attention arrays for each layer.
    - ids_image_attention_drf_accepted: List of lists of indices that map each value in image_attention_drf to its corresponding decoded token.
    - sequences_decoded: List of decoded token sequences.
    
    Returns:
    - List of PIL.Image objects for each layer.
    """

    # Merge ids_image_attention_drf_accepted to a single list
    merged_indices = [index for sublist in ids_image_attention_drf_accepted for index in sublist]

    # Extract the number of layers from the first attention value entry
    num_layers = len(image_attention_drf[0][1])

    # Determine the global min and max values for consistent color scaling
    global_min = float('inf')
    global_max = float('-inf')
    for _, attentions in image_attention_drf:
        for layer_attentions in attentions:
            min_val = layer_attentions.min()
            max_val = layer_attentions.max()
            if min_val < global_min:
                global_min = min_val
            if max_val > global_max:
                global_max = max_val

    # List to store PIL Image objects
    pil_images = []

    # Create subplots for all layers with a wider figure size to avoid crowded X-axis labels
    fig, axs = plt.subplots(num_layers, 1, figsize=(int(len(sequences_decoded) * 0.25), 15))  # Widen the figure size
    fig.suptitle("Attention to Image Tokens by Decoded Tokens for Each Layer", fontsize=16)

    # Iterate over each layer
    for layer_idx in range(num_layers):
        # Prepare a blank matrix for attention values with dimensions (num_decoded_tokens, max_num_image_tokens)
        num_decoded_tokens = len(sequences_decoded)
        max_num_image_tokens = max(len(att[layer_idx]) for _, att in image_attention_drf)  # Determine max number of image tokens across all attentions

        # Initialize matrix with a special color value (e.g., -1) for indices not in ids_image_attention_drf_accepted
        attention_matrix = np.full((num_decoded_tokens, max_num_image_tokens), -1.0)  # Use -1.0 to indicate missing indices

        # Fill the attention matrix using merged_indices as the index mapper
        for (token, attentions), index in zip(image_attention_drf, merged_indices):
            if index < num_decoded_tokens:  # Ensure index is within range
                attention_matrix[index, :len(attentions[layer_idx])] = attentions[layer_idx].flatten()  # Populate the relevant rows

        # Convert the matrix to 2D numpy array for easier plotting
        attention_scores = np.ma.masked_less(attention_matrix, 0)  # Mask values less than 0 for special color

        # Create the heatmap for attention scores for the current layer
        ax = axs[layer_idx] if num_layers > 1 else axs  # Handle the case when num_layers is 1
        cmap = plt.cm.viridis  # Use the viridis color map
        cmap.set_bad(color='gray')  # Set the color for masked (invalid) values
        cax = ax.imshow(attention_scores.T, aspect='auto', cmap=cmap, vmin=global_min, vmax=global_max)  # Set global vmin, vmax for consistent color range
        fig.colorbar(cax, ax=ax, orientation='vertical', label='Attention Score')

        # Set the tick labels for decoded tokens, tilting them to avoid crowding
        ax.set_xticks(np.arange(len(sequences_decoded)))
        ax.set_xticklabels(sequences_decoded, rotation=45, ha='right')  # Tilt labels by 45 degrees

        # Set the tick labels for image tokens (indices)
        ax.set_yticks(np.arange(attention_scores.shape[1]))
        ax.set_yticklabels([f"Image Token {i}" for i in range(attention_scores.shape[1])])

        # Set labels for each subplot
        ax.set_title(f"Layer {layer_idx}")
        ax.set_xlabel("Decoded Token")
        ax.set_ylabel("Attention to Image Tokens")

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the figure to a bytes buffer and convert it to a PIL Image object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    pil_images.append(image)

    # Note: Do not close `buf` or `plt` to keep the objects accessible
    # Do not call plt.clf() or plt.close() to avoid losing the figure

    return pil_images

def aggregate_attention_values(sample_value_image_attention, sample_ids_image_attention, sequences_decoded):
        
    # Remove the split by chunks
    attention_output_for_sample = []
    sample_attention_merged = sum([list(_tuple) for _tuple in sample_value_image_attention], []) # containing all tokens
    sample_indices_merged = sum(sample_ids_image_attention, []) # containing all tokens

    for accepted_idx, layer_attentions in enumerate(sample_attention_merged):
        # Get the corresponding index of the decoded sequence that attended to image tokens
        decoded_idx = sample_indices_merged[accepted_idx]
        decoded_token = sequences_decoded[decoded_idx]

        # Initialize a list to store max attention across layers
        max_attention_across_layers = []

        # Iterate over layers
        for layer_attention in layer_attentions:
            # Convert the list to a numpy array for easier manipulation
            layer_attention_np = np.array(layer_attention)

            # Find max attention values along the heads (axis=1)
            max_attention_heads = layer_attention_np.max(axis=1)  # Shape: (query_len (=1), num_image_tokens)
            # max_attention_heads = layer_attention_np.mean(axis=1)  # Shape: (query_len (=1), num_image_tokens)

            # Squeeze the query length dimension since it is 1
            max_attention_heads = max_attention_heads.squeeze(axis=0)

            # Append the max attention values for this layer
            max_attention_across_layers.append(max_attention_heads)

        # Aggregate across layers by averaging
        # max_attention_across_layers = np.mean(max_attention_across_layers, axis=0)

        # Store the aggregated max attention for this decoded token
        attention_output_for_sample.append((decoded_token, max_attention_across_layers))

    return attention_output_for_sample

def _factorize_squence_for_sd(sequence, ids_accepted_tokens, ids_first_rejected_tokens, first_rejected_tokens, tokenizer):
    """Return tuples of (final_decoded_token, is_accepted, first_rejected_tokens if exists) for each token in a single sequence."""
    
    # ids_first_rejected_tokens = metrics.get("ids_first_rejected_tokens")[i]
    # first_rejected_tokens = metrics.get("first_rejected_tokens")[i]
    return [
        (
            tokenizer.decode([token_id], skip_special_tokens=False), 
            idx in ids_accepted_tokens,
            tokenizer.decode(first_rejected_tokens[ids_first_rejected_tokens.index(idx)], skip_special_tokens=False) if idx in ids_first_rejected_tokens else None,
        )
        for idx, token_id in enumerate(sequence)
    ]

def _factorize_squence_for_sd_topk(sequence, **kwargs):
    tokenizer = kwargs['tokenizer']
    ids_accepted_tokens = kwargs['ids_accepted_tokens']
    ids_first_rejected_tokens = kwargs['ids_first_rejected_tokens']
    
    resulting_list = []

    for idx, token_id in enumerate(sequence):
        # 1. Decode the token
        resulting_token_decoded = tokenizer.decode([token_id], skip_special_tokens=False)

        # 2. Check if the token is accepted
        is_accepted = idx in ids_accepted_tokens

        # 3. Get top-k info using the refactored get_top_k_info function
        if is_accepted or idx in ids_first_rejected_tokens:
            top_k_info = get_top_k_info(idx=idx, is_accepted=is_accepted, **kwargs)
        else:
            top_k_info = None
        
        resulting_list.append((resulting_token_decoded, is_accepted, top_k_info))

    return resulting_list

def get_top_k_info(**kwargs):
    idx = kwargs['idx']
    is_accepted = kwargs['is_accepted']
    tokenizer = kwargs['tokenizer']
    
    if is_accepted:
        # Accepted tokens
        ids_accepted_tokens = kwargs['ids_accepted_tokens']
        tokens_accepted_tokens_topk = kwargs['tokens_accepted_tokens_topk']
        value_probability_accepted_topk = kwargs['value_probability_accepted_topk']

        if isinstance(tokens_accepted_tokens_topk, list):
            # Handle case where top-k tokens are in a list
            _top_k_tokens = tokenizer.batch_decode(tokens_accepted_tokens_topk[ids_accepted_tokens.index(idx)], skip_special_tokens=False)
            _top_k_prob = value_probability_accepted_topk[ids_accepted_tokens.index(idx)]
            top_k_info = [(t, p) for t, p in zip(_top_k_tokens, _top_k_prob)]

        elif isinstance(tokens_accepted_tokens_topk, dict):
            # Handle case where top-k tokens are in a dict of draftings
            top_k_info = []
            for drafting, tokens in tokens_accepted_tokens_topk.items():
                _top_k_tokens = tokenizer.batch_decode(tokens[ids_accepted_tokens.index(idx)], skip_special_tokens=False)
                _top_k_prob = value_probability_accepted_topk[drafting][ids_accepted_tokens.index(idx)]
                top_k_info.append((drafting, [(t, p) for t, p in zip(_top_k_tokens, _top_k_prob)]))
    
    else:
        # Rejected tokens
        ids_first_rejected_tokens = kwargs['ids_first_rejected_tokens']
        tokens_rejected_tokens_topk = kwargs['tokens_rejected_tokens_topk']
        value_probability_rejected_topk = kwargs['value_probability_rejected_topk']

        if isinstance(tokens_rejected_tokens_topk, list):
            # Handle case where top-k tokens are in a list
            _top_k_tokens = tokenizer.batch_decode(tokens_rejected_tokens_topk[ids_first_rejected_tokens.index(idx)], skip_special_tokens=False)
            _top_k_prob = value_probability_rejected_topk[ids_first_rejected_tokens.index(idx)]
            top_k_info = [(t, p) for t, p in zip(_top_k_tokens, _top_k_prob)]

        elif isinstance(tokens_rejected_tokens_topk, dict):
            # Handle case where top-k tokens are in a dict of draftings
            top_k_info = []
            for drafting, tokens in tokens_rejected_tokens_topk.items():
                _top_k_tokens = tokenizer.batch_decode(tokens[ids_first_rejected_tokens.index(idx)], skip_special_tokens=False)
                _top_k_prob = value_probability_rejected_topk[drafting][ids_first_rejected_tokens.index(idx)]
                top_k_info.append((drafting, [(t, p) for t, p in zip(_top_k_tokens, _top_k_prob)]))

    return top_k_info

def _gather_samples(_key, info: List[Dict[str, Any]]):
    value_class = type(info[0][_key])
    if value_class in [int, float]:  
        list_filtered = [_d[_key] for _d in info if _d[_key] is not None]
    elif value_class in [list, np.ndarray]:
        list_filtered = [sum(_d[_key]) for _d in info if _d[_key] is not None]
    else:
        raise ValueError(f"Unexpected type: {value_class}")
    return list_filtered


def get_aggregated_info(
        metrics,
        time_info_per_sample: List[Dict[str, Any]],
    ):
    is_sd = 'time_prefill_tgt' in time_info_per_sample[0]
    is_time_factorized = 'time_decode_chunks_drf' in time_info_per_sample[0]
    
    num_target_tokens_decode = [
        len(sequence) - num_prompt_token - 1
        for sequence, num_prompt_token
        in zip(metrics['sequences'], metrics['num_prompt_tokens'])
    ]
    times_total = metrics['time_total']
    times_prefill = [time_info['time_prefill'] for time_info in time_info_per_sample]
    times_decode = [time_info['time_decode'] for time_info in time_info_per_sample]
    time_prompt_process = [time_info['time_prompt_process'] for time_info in time_info_per_sample]
    
    if is_sd:
        times_prefill_drf = [time_info['time_prefill_drf'] for time_info in time_info_per_sample]
        times_prefill_tgt = [time_info['time_prefill_tgt'] for time_info in time_info_per_sample]
        nums_chunk = [len(chunks) for chunks in metrics['num_accepted_tokens']]
    
    if is_sd and is_time_factorized:
        times_decode_chunks_drf = [time_info['time_decode_chunks_drf'] for time_info in time_info_per_sample]
        times_decode_chunks_tgt = [time_info['time_decode_chunks_tgt'] for time_info in time_info_per_sample]
    
    aggregated_info = {
        "token_rate_prefill_agg": len(times_prefill)/sum(times_prefill),
        "time_total_agg": avg(times_total),
        "time_decode_agg": avg(times_decode),
        "time_prefill_agg": avg(times_prefill),
        "time_prompt_process": avg(time_prompt_process),
        "block_efficiency_agg": 1 + avg(_merge_list(metrics['num_accepted_tokens'])) if is_sd else ['-'],
        "token_rate_prefill_drf_agg": len(times_prefill_drf)/sum(times_prefill_drf) if is_sd else ['-'],
        "token_rate_prefill_tgt_agg": len(times_prefill_tgt)/sum(times_prefill_tgt) if is_sd else ['-'],
        "token_rate_decode_agg": sum(num_target_tokens_decode)/sum(times_decode),
        "time_prefill_drf_agg": avg(times_prefill_drf) if is_sd else ['-'],
        "time_prefill_tgt_agg": avg(times_prefill_tgt) if is_sd else ['-'],
        "time_decode_chunks_drf_agg": avg(_merge_list(times_decode_chunks_drf)) if is_sd and is_time_factorized else ['-'],
        "time_decode_chunks_tgt_agg": avg(_merge_list(times_decode_chunks_tgt)) if is_sd and is_time_factorized else ['-'],
        "num_chunks_agg": avg(nums_chunk) if is_sd else ['-'],
        "num_target_tokens_decode_agg": avg(num_target_tokens_decode),
    }
    return aggregated_info

def get_attention_info(metrics, _config):
    """
    1. binary vs. att values
        aggregation type for att values : all, mean, max
    2. prob ratio vs. att values
        metrics["value_probability_ratio_accepted"]
        metrics["value_probability_ratio_first_rejected"]
    """

    all_value_image_attention_drf_accepted = gather_attention_values(metrics["value_image_attention_drf_accepted"])
    all_value_image_attention_drf_first_rejected = gather_attention_values(metrics["value_image_attention_drf_first_rejected"], input_chunk_separated=False)
    
    
    all_value_image_attention_drf_accepted_layerwise = gather_attention_values(metrics["value_image_attention_drf_accepted"], separate_layer=True)
    all_value_image_attention_drf_first_rejected_layerwise = gather_attention_values(metrics["value_image_attention_drf_first_rejected"], input_chunk_separated=False, separate_layer=True)
    all_value_probability_ratio_accepted = sum([sum(sublist, [])  for sublist in metrics["value_probability_ratio_accepted"]], [])
    all_value_probability_ratio_first_rejected = sum(metrics["value_probability_ratio_first_rejected"], [])
    all_value_probability_accepted_drf = sum([sum(sublist, [])  for sublist in metrics["value_probability_accepted_drf"]], [])
    all_value_probability_first_rejected_drf = sum(metrics["value_probability_first_rejected_drf"], [])

    assert len(all_value_image_attention_drf_accepted_layerwise) == len(all_value_probability_ratio_accepted), "Length mismatch"
    assert len(all_value_image_attention_drf_first_rejected_layerwise) == len(all_value_probability_ratio_first_rejected), "Length mismatch"

    histogram = visualize_attention_hist(all_value_image_attention_drf_accepted, all_value_image_attention_drf_first_rejected)
    scatter_plot = visualize_attention_scatter(
        all_value_image_attention_drf_accepted_layerwise, 
        # all_value_probability_ratio_accepted,
        all_value_probability_accepted_drf,

        all_value_image_attention_drf_first_rejected_layerwise,
        # all_value_probability_ratio_first_rejected,
        all_value_probability_first_rejected_drf,
    )

    return {
        "histogram": histogram,
        "scatter_plot": scatter_plot,
    }


def visualize_attention_scatter(attention_accepted, probability_ratio_accepted, attention_rejected, probability_ratio_rejected):
    """
    Visualizes attention values with scatter plots for each layer and an averaged plot. 
    The output image contains (num_layer + 1) subplots.

    Args:
    - attention_accepted (List[List[int]]): Attention values for accepted samples, where each sublist corresponds to a specific layer.
    - probability_ratio_accepted (List[int]): Probability ratios corresponding to accepted samples.
    - attention_rejected (List[List[int]]): Attention values for rejected samples, where each sublist corresponds to a specific layer.
    - probability_ratio_rejected (List[int]): Probability ratios corresponding to rejected samples.

    Returns:
    - image: A PIL.Image object containing the scatter plot visualization.
    """
    num_layers = len(attention_accepted[0])  # Number of layers
    if num_layers != 2:
        print()

    # Create subplots with (num_layers + 1) subplots
    fig, axs = plt.subplots(1, num_layers + 1, figsize=(5 * (num_layers + 1), 5), sharey=True)
    fig.suptitle('Attention vs Probability Ratio Scatter Plots', fontsize=16)

    # Plot for each layer
    for layer_idx in range(num_layers):
        # Accepted scatter plot
        axs[layer_idx].scatter(
            [attention[layer_idx] for attention in attention_accepted],
            probability_ratio_accepted,
            color='blue',
            alpha=0.7,
            label='Accepted'
        )
        
        # Rejected scatter plot
        axs[layer_idx].scatter(
            [attention[layer_idx] for attention in attention_rejected],
            probability_ratio_rejected,
            color='red',
            alpha=0.7,
            label='Rejected'
        )
        
        axs[layer_idx].set_title(f'Layer {layer_idx + 1}')
        axs[layer_idx].set_xlabel('Attention Value')
        axs[layer_idx].set_ylabel('Probability Ratio')
        axs[layer_idx].legend()

    # Plot for averaged values over all layers
    avg_attention_accepted = [np.mean(attention) for attention in attention_accepted]
    avg_attention_rejected = [np.mean(attention) for attention in attention_rejected]

    # Averaged accepted scatter plot
    axs[num_layers].scatter(
        avg_attention_accepted,
        probability_ratio_accepted,
        color='blue',
        alpha=0.7,
        label='Accepted'
    )

    # Averaged rejected scatter plot
    axs[num_layers].scatter(
        avg_attention_rejected,
        probability_ratio_rejected,
        color='red',
        alpha=0.7,
        label='Rejected'
    )

    axs[num_layers].set_title('Average Across All Layers')
    axs[num_layers].set_xlabel('Average Attention Value')
    axs[num_layers].set_ylabel('Probability Ratio')
    axs[num_layers].legend()

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the plot to a BytesIO buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Convert buffer to a PIL Image object
    image = Image.open(buf)

    return image


# (num_samples, num_chunks, num_accepted, num_layers, num_heads, query_len (=1), key_len (num_image_tokens))
def gather_attention_values(attention_data, aggregation_type='max', input_chunk_separated=True, separate_layer=False):
    """
    Gathers all attention values from the attention data and aggregates them by mean or max across the head axis.

    Args:
    - attention_data: List of lists containing attention values with shape 
                      (num_samples, num_chunks, num_accepted, num_layers, num_heads, query_len (=1), key_len (num image tokens)).
    - aggregation_type: str, type of aggregation ('mean' or 'max'). Default is 'max'.

    Returns:
    - all_attention_values: Flattened list of aggregated attention values across all samples.
    """
    all_attention_values = []

    # Iterate over each sample
    for sample in attention_data:  # num_samples
        # Iterate over each chunk in the sample
        if input_chunk_separated: 
            merged_chunks = [chunk for chunks in sample for chunk in chunks]
        else:
            merged_chunks = sample
        

        # Iterate over each accepted entry in the merged chunk
        for accepted_entry in merged_chunks:  # num_accepted
            layer_attention_values = []
            # Iterate over each layer's attention
            for layer_attention in accepted_entry:  # num_layers
                assert np.array(layer_attention).ndim == 4, f"Invalid shape: layer_attention should have 4 dimensions, but got {np.array(layer_attention).ndim}."
                # Aggregate across heads based on the specified type
                if aggregation_type == 'mean':
                    aggregated_attention = np.mean(layer_attention, axis=1)  # Aggregate by mean along the head axis (axis=4)
                elif aggregation_type == 'max':
                    aggregated_attention = np.max(layer_attention, axis=1)  # Aggregate by max along the head axis (axis=4)
                elif aggregation_type == 'all':
                    aggregated_attention = np.array(layer_attention)  # Aggregate by max along the head axis (axis=4)
                else:
                    raise ValueError("Invalid aggregation type. Use 'mean' or 'max'.")

                # Flatten the aggregated attention values and gather them
                if separate_layer == False:
                    all_attention_values.extend(aggregated_attention.flatten())
                else:
                    layer_attention_values.extend(aggregated_attention.flatten())
            
            if separate_layer:
                all_attention_values.append(layer_attention_values)

    return all_attention_values

def visualize_attention_hist(accepted_values, rejected_values):
    """
    Draws histograms containing two graphs (one for accepted, another for first_rejected) and returns them as a PIL Image.
    The histograms show density (probability) rather than absolute frequency, and the x-axis is set from 0 to 1.

    Args:
    - accepted_values: List of attention values for accepted samples.
    - rejected_values: List of attention values for first rejected samples.

    Returns:
    - image: A PIL.Image object containing the histograms.
    """

    assert all(x >=0 or x < 1 for x in accepted_values), "Not all elements are < 0 or > 1"
    assert all(x >=0 or x < 1 for x in rejected_values), "Not all elements are < 0 or > 1"

    # Create two subplots side-by-side
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    # Plot histogram for accepted values with density
    axs[0].hist(accepted_values, bins=30, color='blue', alpha=0.7, density=True)
    axs[0].set_title('Accepted Attention Values')
    axs[0].set_xlabel('Attention Score')
    axs[0].set_ylabel('Density')
    axs[0].set_xlim(0, 1)  # Set x-axis range from 0 to 1

    # Plot histogram for first rejected values with density
    axs[1].hist(rejected_values, bins=30, color='red', alpha=0.7, density=True)
    axs[1].set_title('First Rejected Attention Values')
    axs[1].set_xlabel('Attention Score')
    axs[1].set_xlim(0, 1)  # Set x-axis range from 0 to 1

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the plot to a BytesIO buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Convert buffer to a PIL Image object
    image = Image.open(buf)

    # Note: Do not close `buf` or `plt` to keep the objects accessible
    # plt.close(fig)  # Close the figure to free up memory

    return image

# Example usage:
# accepted_values = np.random.rand(100).tolist()  # Example random data
# rejected_values = np.random.rand(100).tolist()  # Example random data
# image = visualize_attention_hist(accepted_values, rejected_values)
# image.show()  # Display the image
# image.save('attention_histogram.png')  # Save the image to a file



def save_eval_datasets_info(_config, info_list: list, is_phase_2):
    # filename = model name + YYMMDDHH
    root = _config['root']
    formatted_date_kst = datetime.now(pytz.timezone('Asia/Seoul')).strftime("%y%m%d%H")
    csv_dir = f"{root}/data/MSD/csv/{_config['exp_title']}"
    csv_per_sample_dir= os.path.join(csv_dir, f'per_sample_{formatted_date_kst}')
    filename = f"aggregated_{formatted_date_kst}.csv"
    csv_path = os.path.join(csv_dir, filename)
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(csv_per_sample_dir, exist_ok=True)

    col_img_image_attention_drf = 'img_image_attention_drf'

    config_columns = [
        'dataset', # config
        'decoding',
        'drafting', 
        'drf_model',
        'drf_model_size',
        'tgt_model',
        'tgt_model_size',
        'gamma', 
    ]
    repr_metrics_columns = [
        'block_efficiency_agg', # representative metrics
        'token_rate_decode_agg', 
        "token_rate_prefill_drf_agg",
        "token_rate_prefill_tgt_agg",
        'time_total_agg', 
        'time_prefill_agg', 
        'time_decode_agg',
        'token_rate_prefill_agg', # token rate
        "token_rate_prefill_drf_agg",
        "token_rate_prefill_tgt_agg",
        'token_rate_decode_agg', 
        'time_total_agg', # time
        'time_prefill_agg', 
        'time_decode_agg',
        "time_prefill_drf_agg",
        "time_prefill_tgt_agg",
        'time_prompt_process',
        "num_target_tokens_decode_agg",
    ]
    factorized_time_columns = [
        "time_decode_chunks_drf_agg",
        "time_decode_chunks_tgt_agg",
    ]
    etc_columns = [
        'num_chunks_agg', # etc
        'exp_title',
        'ckpt_name', 
        'sequences_decoded_markdown', # qualitative analysis
        # 'val_image_attention_drf', 
        # 'img_image_attention_drf',
    ]
    
    if _config['decoding'] == 'sd' and _config['is_time_factorized']:
        evaluation_column_order = config_columns + repr_metrics_columns + etc_columns + factorized_time_columns
    else:
        evaluation_column_order = config_columns + repr_metrics_columns + etc_columns

    df_cum = pd.DataFrame()
    list_hist_image_attention = {}
    list_scatter_image_attention = {}
    dict_image_attention = {}

    for info_dict in info_list:
        _config_itr = info_dict['_config']

        # 1. Aggregae info
        df = pd.DataFrame.from_dict(info_dict['aggregated_info'])
        df['dataset'] = _config_itr['dataset']
        df['decoding'] = _config_itr['decoding']
        df['drafting'] = _config_itr['drafting'] if isinstance(_config_itr['drafting'], str) else get_tabed_name(_config_itr)
        # df['drafting'] = 'multimodal' if not _config_itr['is_drf_text_only'] else 'text-only'
        df['drf_model'] = get_short_name(_config_itr['drf'])
        df['drf_model_size'] = get_size(_config_itr['drf'])
        df['tgt_model'] = get_short_name(_config_itr['tgt']) if ('sd' in _config_itr['decoding']) else ''
        df['tgt_model_size'] = get_size(_config_itr['tgt']) if len(df['tgt_model']) > 0 else ''
        df['gamma'] = _config_itr['max_chunk_length']
        df['ckpt_name'] = get_ckpt_name(_config_itr, is_phase_2)
        df['exp_title'] = _config_itr['exp_title']
        df_cum = pd.concat([df_cum, df], axis=0)

        
        # 2. Per-sample info
        filename_per_sample = f"{get_ckpt_name(_config_itr, is_phase_2)}"
        df_per_sample = pd.DataFrame()
        df_per_sample = pd.concat([df_per_sample, pd.DataFrame.from_dict(info_dict['time_per_sample'])], axis=1)
        df_per_sample = pd.concat([df_per_sample, pd.DataFrame.from_dict(info_dict['block_efficiency_per_sample'])], axis=1)
        df_per_sample = pd.concat([df_per_sample, pd.DataFrame.from_dict(info_dict['sequence_per_sample'])], axis=1)
        df_per_sample = pd.concat([df_per_sample, pd.DataFrame.from_dict(info_dict['history_per_sample'])], axis=1)

        csv_per_sample_path= os.path.join(csv_per_sample_dir, f"{filename_per_sample}.csv")
        
        try:
            df_per_sample.to_csv(csv_per_sample_path)
                
        except OSError as e:
            if "File name too long" in str(e):
                print(f"Error: {e}. Splitting the save operation into smaller parts...")
                csv_per_sample_path = csv_per_sample_path.replace('oNone', '')
                csv_per_sample_path = csv_per_sample_path.replace('l0', '')
                csv_per_sample_path = csv_per_sample_path.replace('tm-False', '')
                csv_per_sample_path = csv_per_sample_path.replace('gamma-5', '')
                
                df_per_sample.to_csv(csv_per_sample_path)

            else:
                # Re-raise other OSError exceptions
                raise

        # Save attention images if enabled
        # list_img_image_attention_drf = [_sample.get(col_img_image_attention_drf) for _sample in info_dict['sequence_per_sample']] if col_img_image_attention_drf in info_dict['sequence_per_sample'][0] else None
        # list_hist_image_attention[filename_per_sample] = info_dict['attention_info']['histogram'] if _config['output_image_attentions'] else None
        # list_scatter_image_attention[filename_per_sample] = info_dict['attention_info']['scatter_plot'] if _config['output_image_attentions'] else None

        for plot in ['histogram', 'scatter_plot']:
            if filename_per_sample not in dict_image_attention:
                dict_image_attention[filename_per_sample] = {}
            dict_image_attention[filename_per_sample][plot] = info_dict['attention_info'][plot] if _config['output_image_attentions'] else None
        
        # 3. Save attention images if enabled
        # if list_img_image_attention_drf is not None:
        #     save_images(list_img_image_attention_drf, os.path.join(csv_per_sample_dir, f"{filename_per_sample}_attentions"))
    
    if _config['output_image_attentions']:
        for filename_per_sample, plots_dict in dict_image_attention.items():
            for plot, plot_image in plots_dict.items():
                csv_plot_dir = os.path.join(csv_dir, f'{plot}_{formatted_date_kst}')
                logging.info(f"[Evaluation] Saving attention {plot} to: {csv_plot_dir}")
                os.makedirs(csv_plot_dir, exist_ok=True)
                save_pil_image(plot_image, os.path.join(csv_plot_dir, f"{filename_per_sample}_{plot}.png"))

    # csv_hist_dir= os.path.join(csv_dir, f'hist_{formatted_date_kst}')
    # logging.info(f"[Evaluation] Saving attention histograms to: {csv_hist_dir}")
    # os.makedirs(csv_hist_dir, exist_ok=True)
    # for _key, hist_image_attention in list_hist_image_attention.items():
    #     save_pil_image(hist_image_attention, os.path.join(csv_hist_dir, f"{_key}_histogram.png"))
    
    df_cum_reordered = df_cum[evaluation_column_order]

    os.makedirs(csv_dir, exist_ok=True)
    logging.info(f"[Evaluation] Saving dataset: {_config['eval_datasets']}")
    df_cum_reordered = df_cum_reordered.sort_values(by=['dataset', 'decoding', 'drf_model', 'tgt_model', 'drf_model_size', 'tgt_model_size', 'drafting', 'gamma']).reset_index(drop=True)
    logging.info(f"[Evaluation] Saving aggregated info to: {csv_path}")
    logging.info(f"[Evaluation] Saving per-sample info to: {csv_per_sample_dir}")
    df_cum_reordered.to_csv(csv_path, index=False)

def get_info_dataset(metrics: Dict[str, np.array], _config):

    info_dict = {}
    
    # 1. time analysis (per sample)
    time_info_per_sample = get_time_info_per_sample(metrics, _config)
    
    # aggreagated analysis (per dataset)
    aggregated_info = get_aggregated_info(
        metrics,
        time_info_per_sample,
    )

    if _config['decoding'] == 'sd':
        # 2. block efficiency (per sample)'
        block_efficiency_info_per_sample = get_block_efficiency_info_per_sample(metrics)

        # 3. qualitative analysis (per sample)
        random_ids = [0]
        sequence_info_per_sample = get_sequence_info_per_sample(metrics, _config)
        sequences_decoded_markdown_single_sample = [sequence_info_per_sample[random_idx]['sequences_decoded_markdown'] for random_idx in random_ids]

        # Todo: If history_dependent, add history_info
        if _config['history_dependent']:
            history_info_per_sample = get_history_info_per_sample(metrics, _config)

    
    aggregated_info['sequences_decoded_markdown'] = sequences_decoded_markdown_single_sample if _config['decoding'] == 'sd' else '-'
    
    info_dict = {
        "_config": _config,
        "time_per_sample": time_info_per_sample,
        "block_efficiency_per_sample": block_efficiency_info_per_sample if _config['decoding'] == 'sd' else None,
        "sequence_per_sample": sequence_info_per_sample if _config['decoding'] == 'sd' else None,
        "history_per_sample": history_info_per_sample if (_config['decoding'] == 'sd') and _config['history_dependent'] else None,
        "aggregated_info": aggregated_info,
    }

    # 4. attention analysis (per sample)
    if _config['output_image_attentions']:
        info_dict['attention_info'] = get_attention_info(metrics, _config)

    return info_dict


def get_number_before_ft(s):
    # Search for a number followed by "-ft" and return the number
    match = re.search(r'(\d+)-ft', s)
    if match:
        return match.group(1)
    return None

def parse_eval_list(policy_str):
    # Split the input by commas
    if isinstance(policy_str, str):
        elements = policy_str.split(',')
        
        # Try to convert each element to an integer, if it fails, treat it as a string
        parsed_policy = []
        for element in elements:
            element = element.strip()  # Remove any leading/trailing whitespace
            try:
                # Try to convert to int or float
                if '.' in element:  # For floats
                    parsed_policy.append(float(element))
                else:  # For integers
                    parsed_policy.append(int(element))
            except ValueError:
                # If it fails, treat it as a string and remove surrounding quotes
                parsed_policy.append(element.strip("'").strip('"'))
    elif isinstance(policy_str, int):
        parsed_policy = [policy_str]
    else:
        parsed_policy = policy_str
    
    return parsed_policy


@ex.automain
def main(_config):
    login(os.environ.get("HF_TOKEN"))

    # Root directory
    root_dir = f"{_config['root']}/data/MSD/npy"

    is_phase_2 = len(_config['eval_drafting']) > 0

    var_names = [
        'models',
        'datasets',
        'drafting',
        'max_chunk_length',
        'target_dim_image_pooling',
        'captioning_model',
        'caption_type',
        'image_top_k_attention',
        'tabed_rule',
        'mm_weight_policy',
        'mm_weight_k',
    ]

    metric_names = BaseCriterion(_config)._init_metrics().keys()
    base_combinations = list(product(
            _config['eval_models'],
            parse_eval_list(_config['eval_datasets']),
            parse_eval_list(_config['eval_drafting']),
            parse_eval_list(_config['eval_max_chunk_length']),
            parse_eval_list(_config['eval_target_dim_image_pooling']),
            parse_eval_list(_config['eval_captioning_model']),
            parse_eval_list(_config['eval_caption_type']),
            parse_eval_list(_config['eval_image_top_k_attention']),
            parse_eval_list(_config['eval_tabed_rule']),
            parse_eval_list(_config['eval_mm_weight_policy']),
            parse_eval_list(_config['eval_mm_weight_k']),
            parse_eval_list(_config['eval_history_dependent']),
            parse_eval_list(_config['eval_history_window']),
            parse_eval_list(_config['eval_history_item']),
            parse_eval_list(_config['eval_temperature_drafting_weight']),
            parse_eval_list(_config['eval_history_filter_top1_match']),
            parse_eval_list(_config['eval_history_adaboost_constant_weight']),
            parse_eval_list(_config['eval_history_w_grid_measure']),
            parse_eval_list(_config['eval_history_w_grid_num_accepted_lenience']),
            parse_eval_list(_config['eval_history_w_grid_num_accepted_order']),
            parse_eval_list(_config['eval_multi_turn_task']),
        )
    )

    info_list = []
    ckpt_name_list = []
    for _comb in base_combinations:
        # Setting
        _config_itr = set_config_itr(_comb, _config)
        
        ckpt_name = get_ckpt_name(_config_itr, is_phase_2)
        ft_num = get_number_before_ft(_config_itr['drf']) 
        if ckpt_name in ckpt_name_list:
            # logging.info(f"[Evaluation] ckpt is already processed: {ckpt_name}")
            continue
        elif _config_itr['drafting']=='image-pool':
            if ft_num and str(ft_num) != str(_config_itr['target_dim_image_pooling']):
                continue
    
        ckpt_name_list.append(ckpt_name)
        npy_dir = f"{root_dir}/{_config_itr['exp_title']}/{ckpt_name}"
        logging.info(f"[Evaluation] Config: {ckpt_name}")
        
        # Load metrics from .npy files
        metrics_loaded = load_metrics(npy_dir, metric_names, tiny_data=_config['tiny_data'])

        # Get information for each dataset
        info_dict = get_info_dataset(metrics_loaded, _config_itr)
        info_list.append(
            info_dict
        )
    
    save_eval_datasets_info(_config, info_list, is_phase_2)
