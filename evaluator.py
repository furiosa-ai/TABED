import os
import warnings
import math
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
from collections import defaultdict

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import time

from datasets.formatting import get_formatter, query_table, format_table
from accelerate.utils import tqdm
from absl import logging
import wandb

from tabed.modules.mllm import MLLM
from tabed.modules._speculative_decoding import SpeculativeDecoding
from tabed.modules.eval_specbench import measure_time
from tabed.modules.load_pretrained import load_pretrained_model, load_tokenizer, load_image_processor
from tabed.datamodules.dataset import create_data_loaders
from tabed.criterions.base_criterion import BaseCriterion
from tabed.utils.util import _save, get_ckpt_name
from tabed.utils.saver import Saver
from tabed.utils.utils_evaluator import build_models, build_tokenizers, build_image_processors, get_decoding_class, get_criterion, warmup_generation


class Evaluator(object):
    def __init__(self, _config):
        self._config = _config
        ckpt_save = get_ckpt_name(_config)

        logging.info(f"[Evaluator] Config: {ckpt_save}")
        logging.info(f"[Evaluator] Exp_title: {_config['exp_title']}")

        # Models
        self.models = build_models(_config)
        
        # Tokenizers
        self.tokenizers, eos_token_id, pad_token_id = build_tokenizers(_config)

        # Image Processors
        self.image_processors = build_image_processors(_config)

        # Decoding
        self.decoding = get_decoding_class(_config)(
            _config=_config,
            models=self.models,
            tokenizers=self.tokenizers,
            image_processors=self.image_processors,
            eos_token_id=eos_token_id,
        )
        
        # DataLoaders
        self.data_loaders = create_data_loaders(_config, self.tokenizers['drf'], self.image_processors['drf'])

        # metrics
        self.criterion = get_criterion(_config)

        # Saver
        self.saver = Saver(_config, ckpt_save)

        # wandb
        if not _config['debug']:
            wandb.init(
                entity="furiosaai",
                project=str(_config['wandb_project_name']),
                config=_config,
                reinit=True
            )
            wandb.run.name = ckpt_save

    def train(self):
        logging.info(f"[Evaluator] Training starts ...")
        pass
    
    def inference(self, split: Literal["valid", "test"]):
        logging.info(f"[Evaluator] Inference on ({split}) starts ...")
        # warmup for measuring time
        if not self._config['debug']:
            warmup_generation(self.models['drf'], self.tokenizers['drf'])

        for i, batch in enumerate(tqdm(iterable=self.data_loaders[split])):
            batch = self.decoding.load_batch_to_device(batch)
            results, time_total = measure_time(self.decoding.decode, batch)
            
            results.metrics['num_prompt_tokens'] = batch['input_ids'].shape[1]
            results.metrics['time_total'] = time_total
            results.metrics['sequences'] = results['sequences'].tolist()[0]
            self.criterion(results.metrics)  # Pass the accumulated results to BaseCriterion
            # self.tokenizers['drf'].batch_decode(self.models['drf'].generate(**batch, **self.decoding.generate_config)['sequences'])[0]
            # self.tokenizers['drf'].batch_decode(self.models['tgt'].generate(**batch, **self.decoding.generate_config)['sequences'])[0]

            if i > 0 and i % self._config['save_steps'] == 0:
                logging.info(f"[Evaluator] iteration {i:05d} saving metrics ...")
                self._save_metrics(i, reset=False)

        logging.info(f"[Evaluator] last iteration {i:05d} saving metrics ...")
        self._save_metrics(reset=True)  # Final save with reset

    def _save_metrics(self, step: int=None, reset: bool=False):
        """
        Helper method to save metrics and optionally reset the internal state.
        """
        self.saver.save_metrics(self.criterion, step)  # Save detailed metrics as .npy
        # epoch_dict = self.criterion.get_epoch_dict(reset=reset)

    def validate(self):
        self.inference("valid")

    def test(self):
        self.inference("test")

