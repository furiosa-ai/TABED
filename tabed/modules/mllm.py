"""MLLM wrapper module for unified model interface."""

import torch.nn as nn
from transformers import PreTrainedModel


class MLLM(nn.Module):
    """Wrapper for MLLM/LLM models to support various architectures.

    Provides a unified interface following the same API as
    ConditionalGeneration classes in transformers.

    Args:
        model: A pretrained model from transformers library.
    """

    def __init__(self, model: PreTrainedModel):
        """Initialize the MLLM wrapper."""
        super(MLLM, self).__init__()
        self.mllm = model.cuda().eval()

    def generate(self, *args, **kwargs):
        """Generate text using the wrapped model.

        Args:
            *args: Positional arguments passed to model.generate().
            **kwargs: Keyword arguments passed to model.generate().

        Returns:
            Generated output from the model.
        """
        return self.mllm.generate(*args, **kwargs)
