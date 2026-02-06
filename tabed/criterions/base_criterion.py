"""Base criterion class for metric computation and logging."""

from typing import Any, Dict, List


class BaseCriterion:
    """Base class for computing and tracking experiment metrics.

    Handles metric initialization based on decoding configuration,
    metric accumulation during training/evaluation, and reset functionality.

    Args:
        _config: Configuration dictionary containing experiment settings.
    """

    def __init__(self, _config: Dict[str, Any]):
        """Initialize the criterion with configuration."""
        self._config = _config
        self.metrics = self._init_metrics()

    def _init_metrics(self) -> Dict[str, List]:
        """Initialize metrics dictionary based on configuration.

        Returns:
            Dictionary mapping metric names to empty lists for accumulation.
        """
        metrics_exclude = []

        if self._config['decoding'] == 'ard':
            metrics_exclude = [
                'num_accepted_tokens',
                'num_prefill_tokens_tgt',
                'ids_accepted_tokens',
                'ids_first_rejected',
                'tokens_first_rejected',
                'time_prefill_tgt',
                'time_generate_drf',
                'time_verify_tgt',
            ]
        elif (self._config['decoding'] == 'sd'
              and not self._config['is_time_factorized']):
            metrics_exclude = ['time_generate_drf', 'time_verify_tgt']

        if not self._config['output_image_attentions']:
            metrics_exclude += [
                "value_image_attention_drf_accepted",
                "value_image_attention_drf_first_rejected",
                "ids_image_attention_drf_accepted",
                "value_probability_ratio_accepted",
                "value_probability_ratio_first_rejected",
                'value_probability_accepted_drf',
                'value_probability_accepted_tgt',
                'value_probability_first_rejected_drf',
                'value_probability_first_rejected_tgt',
            ]

        return {
            m: [] for m in self._config['metric']
            if m not in metrics_exclude
        }

    def __call__(self, results: Dict[str, List]) -> None:
        """Accumulate metrics from a batch of results.

        Args:
            results: Dictionary mapping metric names to their values.
        """
        for m in self.metrics:
            if m in results:
                self.metrics[m].append(results[m])

    def compute_metrics(self) -> Dict[str, float]:
        """Compute aggregated metrics from accumulated values.

        Returns:
            Dictionary containing computed metrics including total tokens
            and block efficiency.
        """
        num_accepted_tokens = self.metrics['num_accepted_tokens']
        num_generated_tokens = self.metrics['num_generated_tokens']

        total_accepted_tokens = sum(sum(tokens) for tokens in num_accepted_tokens)
        total_generated_tokens = sum(sum(tokens) for tokens in num_generated_tokens)

        # Block efficiency calculation: 1 + average(num_accepted_tokens)
        if len(num_accepted_tokens) > 0:
            block_efficiency = 1 + (total_accepted_tokens / len(num_accepted_tokens))
        else:
            block_efficiency = 1.0

        return {
            'total_accepted_tokens': total_accepted_tokens,
            'total_generated_tokens': total_generated_tokens,
            'block_efficiency': block_efficiency,
        }

    def get_epoch_dict(self, reset: bool = False) -> Dict[str, float]:
        """Get metrics dictionary for the epoch.

        Args:
            reset: If True, reset internal state after returning metrics.

        Returns:
            Dictionary of computed metrics.
        """
        metrics = self.compute_metrics()
        if reset:
            self.reset()
        return metrics

    def reset(self) -> None:
        """Reset all metrics to initial empty state."""
        self.metrics = self._init_metrics()
