import os
import glob
import sys
import random

import torch
import torch.multiprocessing as mp
import numpy as np
from absl import logging
import wandb

from tabed.config import ex
from tabed.utils.util import set_seed
from evaluator import Evaluator

def setup_hf_token():
    """Setup HuggingFace token from environment variable without interactive login."""
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        os.environ["HF_TOKEN"] = hf_token


def get_cuda_device_count():
    """Get the number of available CUDA devices"""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0


def worker_process(rank, world_size, _config):
    """Worker process for multiprocessing"""
    # Set CUDA device for this process
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)

    # Set seed with rank offset for reproducibility
    set_seed(_config['seed'] + rank)
    setup_hf_token()

    # Create modified config for this worker
    worker_config = _config.copy()
    worker_config['rank'] = rank
    worker_config['world_size'] = world_size

    # Initialize evaluator with worker-specific config
    evaluator = Evaluator(worker_config)

    if not worker_config['test_only']:
        evaluator.train()
    else:
        evaluator.test()


@ex.automain
def main(_config):
	# Get number of available CUDA devices
	num_devices = get_cuda_device_count()

	if num_devices <= 1:
		# Single device or no CUDA - keep original behavior
		set_seed(_config['seed'])
		setup_hf_token()

		evaluator = Evaluator(_config)

		if not _config['test_only']:
			evaluator.train() # train, save, test
		else:
			evaluator.test() # test

		if not _config['debug']:
			wandb.finish(0)
	else:
		# Multiple devices - use multiprocessing
		logging.info(f"[Main] Found {num_devices} CUDA devices, starting multiprocessing")
		mp.set_start_method('spawn', force=True)

		processes = []
		for rank in range(num_devices):
			p = mp.Process(target=worker_process, args=(rank, num_devices, _config))
			p.start()
			processes.append(p)

		# Wait for all processes to complete
		for p in processes:
			p.join()

		if not _config['debug']:
			wandb.finish(0)