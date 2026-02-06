"""Metric saving utilities for experiment results."""

import logging
import os
from typing import Any, Dict, Optional

import numpy as np


class Saver:
    """Saves experiment metrics to numpy files.

    Args:
        _config: Configuration dictionary containing save paths.
        ckpt_save: Checkpoint save identifier for directory naming.
    """

    def __init__(self, _config: Dict[str, Any], ckpt_save: str):
        """Initialize the saver with output directory."""
        self.npy_save_dir = (
            f"{_config['root']}/data/MSD/npy/{_config['exp_title']}/{ckpt_save}"
        )
        os.makedirs(self.npy_save_dir, exist_ok=True)

    def save_metrics(self, criterion, step: Optional[int]) -> None:
        """Save all metrics from a criterion to numpy files.

        Args:
            criterion: Criterion object containing metrics and config.
            step: Current step number (None for final save).
        """
        # Save configuration
        config_path = os.path.join(self.npy_save_dir, "config.npy")
        self._save_npy(config_path, 'config', criterion._config)

        # Save each metric
        for key, value in criterion.metrics.items():
            if step is not None:
                filename = f"{key}_{step:05d}.npy"
            else:
                filename = f"{key}.npy"
            npy_path = os.path.join(self.npy_save_dir, filename)
            self._save_npy(npy_path, key, value)

    def _save_npy(self, npy_path: str, key: str, value: Any) -> None:
        """Save a value to a numpy file.

        Args:
            npy_path: Path to save the numpy file.
            key: Name of the metric being saved.
            value: Value to save.
        """
        np.save(npy_path, np.array(value, dtype=object))
        logging.info(f"[Saver] Save {key} to {npy_path} ...")
