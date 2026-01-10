"""Sacred experiment configuration for TABED.

This module defines the experiment configuration using the Sacred framework,
including model settings, data configurations, and evaluation parameters.
"""

import os
import sys

import torch
from sacred import Experiment

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

# Direct import from util module to avoid circular imports through utils/__init__.py
from tabed.utils.util import get_short_name, map_name_task


# =============================================================================
# Path Configuration (auto-detected from main.py location)
# =============================================================================
TABED_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABED_DATA_DIR = os.path.join(TABED_ROOT, 'datasets')
TABED_RESULTS_DIR = os.path.join(TABED_ROOT, 'results')
TABED_CHECKPOINT_DIR = os.path.join(TABED_ROOT, 'checkpoint')
TABED_NPY_DIR = os.path.join(TABED_ROOT, 'results_npy')


ex = Experiment("METER", save_git_info=False)


@ex.config
def config():
    """Base configuration for TABED experiments."""
    # Model config
    drf = "llava-hf/llava-1.5-7b-hf"
    tgt = "llava-hf/llava-1.5-7b-hf"
    captioning_model = "microsoft/Florence-2-large-ft"
    caption_type = "<CAPTION>"  # "<DETAILED_CAPTION>", ""

    drf_dtype = "fp16"
    tgt_dtype = "fp16"
    captioning_model_dtype = "fp16"
    caption_prefix = "image: "
    assistant_dtype = "fp32"

    # Drafting options: 'multimodal', 'text-only', 'caption',
    drafting = 'multimodal'
    image_top_k_attention = 0  # llama: 576
    is_drf_text_only = drafting in [
        'text-only', 'special-token', 'caption', 'tokenized-image'
    ]
    is_tgt_text_only = False
    is_drf_from_mllm = True
    drf_aux_tokenizer = None  # for base lm

    # Drafting settings
    target_dim_image_pooling = 144
    image_pool_type = 'avg2d'
    output_image_attentions = False
    logging_top_k = 5

    # Draft generation config
    max_prompt_length = 2048
    max_target_length = 128
    min_target_length = None
    assistant = None
    temperature = 0.0

    # TABED config
    tabed_rule = 'mm-weight'  # 'confidence', 'mm-weight', 'conf-weight'
    temperature_drafting_weight = 1
    mm_weight_policy = None
    mm_weight_k = None
    tabed_input = 'probability'  # 'logit', 'probability'
    confidence_type = 'top1-prob'  # 'top1-prob', 'entropy', 'top1-prob-cross'

    # History-dependent settings
    history_dependent = False
    history_window = 0  # 0 for all, else number of blocks
    history_item = 'block-efficiency'  # 'block-efficiency', 'kld', 'tvd',
    #   'adaboost', 'w-grid'
    history_unit = "token"  # 'block', 'token'
    history_filter_top1_match = False
    history_adaboost_constant_weight = 1
    history_w_num_grid = 10
    history_w_grid_measure = 'num-accepted'  # 'num_accepted', 'kld', 'tvd',
    #   'num-accepted-kld'
    history_w_grid_num_accepted_order = None  # 'first', 'last', 'middle'
    history_w_grid_num_accepted_lenience = 0

    # Decoding config
    decoding = 'sd'  # 'sd', 'ard'
    max_chunk_length = 5
    solution_type = 'sol1'

    # Experiment config
    batch_size = 1
    seed = 2024
    debug = True
    test_only = True
    do_print = False
    save_steps = 2500
    logging_steps = 10
    is_time_factorized = False

    metric = [
        'sequences',
        'num_prompt_tokens',
        'num_accepted_tokens',
        'num_prefill_tokens_drf',
        'num_prefill_tokens_tgt',
        'ids_accepted_tokens',
        'ids_first_rejected',
        'tokens_first_rejected',
        'time_total',
        'time_prefill_drf',
        'time_prefill_tgt',
        'time_generate_drf',
        'time_verify_tgt',
        'time_prompt_process',
        'tokens_accepted_tokens_topk',
        'value_probability_accepted_topk',
        'tokens_rejected_tokens_topk',
        'value_probability_rejected_topk',
        "value_image_attention_drf_accepted",
        "value_image_attention_drf_first_rejected",
        "ids_image_attention_drf_accepted",
        "value_probability_ratio_accepted",
        "value_probability_ratio_first_rejected",
        'value_probability_accepted_drf',
        'value_probability_accepted_tgt',
        'value_probability_first_rejected_drf',
        'value_probability_first_rejected_tgt',
        'history_dependent_weights',
    ]

    # Dataset config
    dataset = "LLaVA-Instruct-150K"
    ensemble_train_split = False
    multi_turn_task = None
    multi_turn_prev_dir = None
    multi_turn_data_root = os.path.join(TABED_RESULTS_DIR, "multiturn_data")

    # Dataset options:
    # Single-image: "LLaVA-Instruct-150K", "COCO2014", "ScienceQA"
    # Evaluation: "VibeEval", "DC100_EN", "LLaVA-Bench-Wilder"
    # Multi-image: 'Spot-the-Diff', 'Birds-to-Words', 'CLEVR-Change',
    #   'HQ-Edit', 'MagicBrush', 'IEdit', 'AESOP', 'FlintstonesSV',
    #   'PororoSV', 'VIST', 'WebQA'

    # Evaluation config
    eval_models = [None]
    eval_datasets = [None]
    eval_is_drf_text_only = [None]
    eval_drafting = [None]
    eval_max_chunk_length = [None]
    eval_target_dim_image_pooling = [None]
    eval_captioning_model = [None]
    eval_caption_type = [None]
    eval_image_top_k_attention = [None]
    eval_mm_weight_policy = [None]
    eval_tabed_rule = [None]
    eval_mm_weight_k = [None]
    eval_history_dependent = [False]
    eval_history_window = [None]
    eval_history_item = [None]
    eval_temperature_drafting_weight = ["1"]
    eval_history_filter_top1_match = [False]
    eval_history_adaboost_constant_weight = [None]
    eval_history_w_grid_measure = [None]
    eval_history_w_grid_num_accepted_lenience = ['0']
    eval_history_w_grid_num_accepted_order = [None]
    eval_multi_turn_task = [None]
    exp_title = ''
    empty_cache = None

    tiny_data = False
    reduce_data = None  # llava test set: 15772

    # Logging config
    wandb_project_name = "TABED"

    # Path config (configurable via environment variables)
    root = TABED_ROOT
    input_datasets_dir = os.path.join(TABED_DATA_DIR, dataset)
    ensemble_train_split_dir = os.path.join(TABED_DATA_DIR, "ensemble_train_split")
    ckpt_dir = None
    npy_save_dir = TABED_NPY_DIR


@ex.capture
def capture_config(_config):
    """Capture and return the current configuration."""
    return _config


# =============================================================================
# Model Named Configs
# =============================================================================

@ex.named_config
def Llava68m():
    """68M parameter LLaVA draft model."""
    drf = "mjbooo/lvlm68m"


@ex.named_config
def Llava68mOv():
    """68M parameter LLaVA OneVision draft model."""
    drf = "mjbooo/lvlm68m-ov"


@ex.named_config
def Llava160mBf():
    """160M parameter LLaVA draft model with bfloat16."""
    drf = "mjbooo/lvlm160m-bf16"
    drf_dtype = "bf16"
    tgt_dtype = "bf16"


@ex.named_config
def Llava290m():
    """290M parameter LLaVA draft model."""
    drf = "mjbooo/lvlm290m"


# =============================================================================
# Decoding Named Configs
# =============================================================================

@ex.named_config
def ARDecoding():
    """Autoregressive decoding mode."""
    decoding = 'ard'


@ex.named_config
def SpecDecoding():
    """Speculative decoding mode."""
    decoding = 'sd'


# =============================================================================
# TABED Named Configs
# =============================================================================

@ex.named_config
def TabedMT():
    """TABED: multimodal -> text-only."""
    drafting = ['multimodal', 'text-only']


@ex.named_config
def TabedMC():
    """TABED: multimodal -> caption."""
    drafting = ['multimodal', 'caption']


@ex.named_config
def TabedTC():
    """TABED: text-only -> caption."""
    drafting = ['text-only', 'caption']


@ex.named_config
def TabedCP():
    """TABED: caption -> image-pool."""
    drafting = ['caption', 'image-pool']


@ex.named_config
def TabedMTP():
    """TABED: multimodal -> text-only -> image-pool."""
    drafting = ['multimodal', 'text-only', 'image-pool']


@ex.named_config
def TabedMCP():
    """TABED: multimodal -> caption -> image-pool."""
    drafting = ['multimodal', 'caption', 'image-pool']


@ex.named_config
def TabedMTC():
    """TABED: multimodal -> text-only -> caption."""
    drafting = ['multimodal', 'text-only', 'caption']


@ex.named_config
def TabedTCP():
    """TABED: text-only -> caption -> image-pool."""
    drafting = ['text-only', 'caption', 'image-pool']


@ex.named_config
def TabedMTCP():
    """TABED: multimodal -> text-only -> caption -> image-pool."""
    drafting = ['multimodal', 'text-only', 'caption', 'image-pool']


# =============================================================================
# Evaluation Model Named Configs
# =============================================================================

@ex.named_config
def EvalLvlm():
    """Evaluation with 68M LVLM to 7B LLaVA."""
    eval_models = [
        ("mjbooo/lvlm68m", "llava-hf/llava-1.5-7b-hf"),
    ]


@ex.named_config
def EvalLvlmOv():
    """Evaluation with OneVision LVLM."""
    eval_models = [
        ("mjbooo/lvlm68m-ov", "llava-hf/llava-1.5-7b-hf"),
    ]


@ex.named_config
def EvalLvlmOvNextVicuna7b():
    """Evaluation with OneVision LVLM to Next Vicuna 7B."""
    eval_models = [
        ("mjbooo/lvlm68m-ov", "llava-hf/llava-v1.6-vicuna-7b-hf"),
    ]


@ex.named_config
def EvalLvlm290m():
    """Evaluation with 290M LVLM."""
    eval_models = [
        ("mjbooo/lvlm290m", "llava-hf/llava-1.5-7b-hf"),
    ]


@ex.named_config
def EvalLvlm160mBf():
    """Evaluation with 160M bfloat16 LVLM."""
    eval_models = [
        ("mjbooo/lvlm160m-bf16", "llava-hf/llava-1.5-7b-hf"),
    ]
    drf_dtype = "bf16"
    tgt_dtype = "bf16"


@ex.named_config
def EvalLvlm160mBfNextVicuna7b():
    """Evaluation with 160M bfloat16 LVLM to Next Vicuna 7B."""
    eval_models = [
        ("mjbooo/lvlm160m-bf16", "llava-hf/llava-v1.6-vicuna-7b-hf"),
    ]
    drf_dtype = "bf16"
    tgt_dtype = "bf16"


@ex.named_config
def EvalLvlm13b():
    """Evaluation with 68M LVLM to 13B LLaVA."""
    eval_models = [
        ("mjbooo/lvlm68m", "llava-hf/llava-1.5-13b-hf"),
    ]


@ex.named_config
def EvalLvlmOv13b():
    """Evaluation with OneVision LVLM to 13B LLaVA."""
    eval_models = [
        ("mjbooo/lvlm68m-ov", "llava-hf/llava-1.5-13b-hf"),
    ]


@ex.named_config
def EvalLvlm290m13b():
    """Evaluation with 290M LVLM to 13B LLaVA."""
    eval_models = [
        ("mjbooo/lvlm290m", "llava-hf/llava-1.5-13b-hf"),
    ]


@ex.named_config
def EvalLvlm160mBf13b():
    """Evaluation with 160M bfloat16 LVLM to 13B LLaVA."""
    eval_models = [
        ("mjbooo/lvlm160m-bf16", "llava-hf/llava-1.5-13b-hf"),
    ]
    drf_dtype = "bf16"
    tgt_dtype = "bf16"


# =============================================================================
# Evaluation TABED Named Configs
# =============================================================================

@ex.named_config
def EvalTabedMT():
    """Evaluation TABED: multimodal -> text-only."""
    eval_drafting = [['multimodal', 'text-only']]


@ex.named_config
def EvalTabedMC():
    """Evaluation TABED: multimodal -> caption."""
    eval_drafting = [['multimodal', 'caption']]


@ex.named_config
def EvalTabedTC():
    """Evaluation TABED: text-only -> caption."""
    eval_drafting = [['text-only', 'caption']]


@ex.named_config
def EvalTabedCP():
    """Evaluation TABED: caption -> image-pool."""
    eval_drafting = [['caption', 'image-pool']]


@ex.named_config
def EvalTabedMTP():
    """Evaluation TABED: multimodal -> text-only -> image-pool."""
    eval_drafting = [['multimodal', 'text-only', 'image-pool']]


@ex.named_config
def EvalTabedMTC():
    """Evaluation TABED: multimodal -> text-only -> caption."""
    eval_drafting = [['multimodal', 'text-only', 'caption']]


@ex.named_config
def EvalTabedMCP():
    """Evaluation TABED: multimodal -> caption -> image-pool."""
    eval_drafting = [['multimodal', 'caption', 'image-pool']]


@ex.named_config
def EvalTabedTCP():
    """Evaluation TABED: text-only -> caption -> image-pool."""
    eval_drafting = [['text-only', 'caption', 'image-pool']]


@ex.named_config
def EvalTabedMTCP():
    """Evaluation TABED: multimodal -> text-only -> caption -> image-pool."""
    eval_drafting = [['multimodal', 'text-only', 'caption', 'image-pool']]


# =============================================================================
# Drafting Named Configs
# =============================================================================

@ex.named_config
def MultimodalDraft():
    """Multimodal drafting mode."""
    drafting = 'multimodal'


@ex.named_config
def TextOnlyDraft():
    """Text-only drafting mode."""
    drafting = 'text-only'


@ex.named_config
def CaptionDraft():
    """Caption-based drafting mode."""
    drafting = 'caption'


@ex.named_config
def InferencePoolDraft():
    """Image pooling drafting mode."""
    drafting = 'image-pool'


@ex.named_config
def TextOnlyVerify():
    """Text-only verification mode."""
    is_tgt_text_only = True


@ex.named_config
def HalfPrecision():
    """Half precision (FP16) mode."""
    drf_dtype = "fp16"
    tgt_dtype = "fp16"


# =============================================================================
# Dataset Named Configs
# =============================================================================

@ex.named_config
def LlavaBenchInTheWildData():
    """LLaVA-Bench-in-the-Wild dataset."""
    dataset = 'llava-bench-in-the-wild'


@ex.named_config
def DocVQAData():
    """DocVQA validation dataset."""
    dataset = 'docvqa_val'


@ex.named_config
def PopeData():
    """POPE dataset."""
    dataset = "POPE"


@ex.named_config
def MMVetData():
    """MM-Vet dataset."""
    dataset = "MMVet"


@ex.named_config
def IEditData():
    """IEdit dataset."""
    dataset = "IEdit"


@ex.named_config
def MagicBrushData():
    """MagicBrush dataset."""
    dataset = "MagicBrush"


@ex.named_config
def SpotTheDiffData():
    """Spot-the-Diff dataset."""
    dataset = "Spot-the-Diff"


@ex.named_config
def PororoSVData():
    """PororoSV dataset."""
    dataset = "PororoSV"


@ex.named_config
def VISTData():
    """VIST dataset."""
    dataset = "VIST"


# =============================================================================
# Evaluation Named Configs
# =============================================================================

@ex.named_config
def EvaluationSD():
    """Speculative decoding evaluation.

    Run: python3 tabed/utils/evaluation.py with EvaluationSD
    """
    drf = None
    tgt = None
    decoding = 'sd'
    eval_models = [
        ("mjbooo/lvlm68m-ov", "llava-hf/llava-1.5-7b-hf"),
    ]
    output_image_attentions = False
    eval_is_drf_text_only = [False]
    eval_max_chunk_length = [5]
    is_time_factorized = False
    exp_title = "fp16-mm-weight-tabed"
    eval_datasets = [
        'llava-bench-in-the-wild', 'docvqa_val', 'POPE', 'MMVet',
        'IEdit', 'MagicBrush', 'Spot-the-Diff', 'PororoSV', 'VIST'
    ]
    npy_save_dir = TABED_NPY_DIR
    do_print = False


@ex.named_config
def Debug():
    """Debug mode with tiny data."""
    debug = True
    tiny_data = True
