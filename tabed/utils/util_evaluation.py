import os
from PIL import Image
import logging
import numpy as np
import copy
from typing import List, Any

def load_metrics(npy_dir, metric_names, tiny_data):
    """Load the metrics from the specified directory."""
    if any(x in npy_dir for x in ["multimodal", "text-only"]):
        metrics_exclude = ['time_prompt_process']
    else:
        metrics_exclude = []

    metrics = {}
    for metric_name in metric_names:
        if metric_name in metrics_exclude:
            continue
        npy_path = f"{npy_dir}/{metric_name}.npy"

        try:
            npy_metric = np.load(npy_path, allow_pickle=True)
        except:
            if any(x in npy_dir for x in ["multimodal", "text-only"]):
                # logging.info(f"File {npy_dir}/{metric_name}.npy not found. Changed 'text-img' to 'multimodal'.")
                if "multimodal" in npy_path:
                    npy_path_corrected = npy_path.replace("multimodal-drafting", "text-img")
                elif "text-only" in npy_path:
                    npy_path_corrected = npy_path.replace("text-only-drafting", "text-only")
                npy_metric = np.load(npy_path_corrected, allow_pickle=True)
        
        if tiny_data:
            npy_metric = npy_metric[:5]
            
        metrics[metric_name] = npy_metric

    # {metric_name: np.load(f"{npy_dir}/{metric_name}.npy", allow_pickle=True) for metric_name in metric_names if metric_name not in metrics_exclude}
    return metrics

def set_config_itr(comb_itr, _config):
    _config_itr = copy.copy(_config)

    var_names = [
        'model_pairs',
        'dataset',
        'drafting',
        'max_chunk_length',
        'target_dim_image_pooling',
        'captioning_model',
        'caption_type',
        'image_top_k_attention',
        'tabed_rule',
        'mm_weight_policy',
        'mm_weight_k',
        'history_dependent',
        'history_window',
        'history_item',
        'temperature_drafting_weight',
        'history_filter_top1_match',
        'history_adaboost_constant_weight',
        'history_w_grid_measure',
        'history_w_grid_num_accepted_lenience',
        'history_w_grid_num_accepted_order',
        'multi_turn_task',
    ]
    # match comb_itr with var_names
    _dict_itr_except_model_pairs  = {var_name: comb_itr[i] for i, var_name in enumerate(var_names) if i>0}
    
    _dict_itr = {
        "drf": comb_itr[0][0],
        "tgt": comb_itr[0][1],
    }
    _dict_itr.update(_dict_itr_except_model_pairs)
    
    _config_itr.update(_dict_itr)
    return _config_itr


def save_pil_image(image, file_path):
    """
    Save a single PIL Image to a specified file path.

    Args:
    - image (PIL.Image): The PIL Image object to save.
    - file_path (str): The path where the image should be saved, including the file name and extension (e.g., 'output_image.png').

    Returns:
    - None
    """
    try:
        # Save the image to the specified file path
        image.save(file_path)
        # print(f"Image successfully saved to {file_path}.")
    except Exception as e:
        print(f"Error saving image: {e}")

        

def save_images(list_img_image_attention_drf, output_dir):
    """
    Save a list of PIL Image objects to a specified directory.

    Args:
    - list_img_image_attention_drf: List of lists, where each sublist contains one PIL Image object to be saved.
    - output_dir: Directory where the images will be saved.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over each sublist of images
    logging.info(f"[Evaluation] Saving figures wrt attentions to: {output_dir}")
    for sample_idx, img_list in enumerate(list_img_image_attention_drf):
        # Ensure the current item is a list and contains exactly one PIL.Image object
        if not isinstance(img_list, list) or len(img_list) != 1 or not isinstance(img_list[0], Image.Image):
            raise ValueError(f"Each item in list_img_image_attention_drf must be a list containing one PIL Image object. Error at index {sample_idx}.")

        # Get the image from the list
        image = img_list[0]

        # Define a unique filename for each image
        output_file_path = os.path.join(output_dir, f'sample_{sample_idx}.png')

        # Save the image to the output directory
        image.save(output_file_path)

def _get_markdown(sequences_decoded_factorized):
    list_connected = _connect_consecutive_ids(sequences_decoded_factorized)
    md = ' '.join(list_connected)

    return md

def _connect_consecutive_ids(sequences_decoded_factorized):
    # Initialize an empty list to store the final output
    output = []

    # Initialize variables to keep track of the current label and concatenated text
    current_label = sequences_decoded_factorized[0][1]
    current_text = sequences_decoded_factorized[0][0]

    # Iterate through the sequence starting from the second element
    for item, label, _ in sequences_decoded_factorized[1:]:
        if label == current_label:
            # If the label is the same as the current label, concatenate the text
            current_text += " " + item
        else:
            # If the label changes, append the current text and label to the output
            if current_label:
                # Bold if it is an accepted token generated from draft model
                output.append("**"+current_text.strip()+"**")
            else:
                output.append(current_text.strip())
            # Update the current label and text
            current_label = label
            current_text = item

    # Append the last group to the output
    if current_label:
        # Bold if it is an accepted token generated from draft model
        output.append("**"+current_text.strip()+"**")
    else:
        output.append(current_text.strip())

    return output


def _merge_list(ll: List[List[Any]]):
    return [item for sublist in ll for item in sublist]

def get_size(model_name):
    if "68m" in model_name:
        return 0.068
    elif "160m" in model_name:
        return 0.16
    elif "290m" in model_name:
        return 0.29
    elif "7b" in model_name:
        return 7
    elif "13b" in model_name:
        return 13
    