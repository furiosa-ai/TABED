"""Timing utilities for model evaluation and benchmarking."""

import time

import torch


def measure_time(func, *args, **kwargs):
    """Measure the time taken for a function call using CUDA synchronization.

    Ensures accurate timing by synchronizing CUDA operations before and after
    the function execution.

    Args:
        func: A callable to measure the time for.
        *args: Positional arguments to pass to the callable.
        **kwargs: Keyword arguments to pass to the callable.

    Returns:
        Tuple of (function output, elapsed time in seconds).
    """
    torch.cuda.synchronize()
    start_time = time.time()

    outputs = func(*args, **kwargs)

    torch.cuda.synchronize()
    end_time = time.time()

    elapsed_time = end_time - start_time
    return outputs, elapsed_time
