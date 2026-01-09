"""Grid-based weight search for ensemble drafting strategies."""

from itertools import product
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange


def generate_w_grid(num_drafting: int, interval_grid: float) -> np.ndarray:
    """Generate a grid of weight combinations for ensemble drafting.

    Creates all possible weight combinations for the specified number of
    drafting strategies where weights sum to 1.

    Args:
        num_drafting: Number of drafting strategies to combine.
        interval_grid: Step size for weight values (e.g., 0.1 for 10 steps).

    Returns:
        NumPy array of shape (n_combinations, num_drafting) containing
        all valid weight combinations.
    """
    candidates = np.arange(0, 1 + interval_grid, interval_grid)
    initial_combinations = product(candidates, repeat=num_drafting)
    w_grid = [comb for comb in initial_combinations if np.isclose(sum(comb), 1)]

    return np.array(w_grid)


def get_best_from_w_grid(
    w_grid_tensor: torch.Tensor,
    _history_current_drafting_concat: np.ndarray,
    _history_current_target: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """Find the best weight combination from the grid based on history.

    Evaluates all weight combinations in the grid against historical
    predictions and targets to find the optimal ensemble weights.

    Args:
        w_grid_tensor: Tensor of weight combinations, shape (n_combinations, n_drafts).
        _history_current_drafting_concat: Historical drafting predictions.
        _history_current_target: Historical target distributions.
        **kwargs: Additional arguments:
            - history_w_grid_measure: Measure type ('num-accepted', 'kld', 'tvd').
            - history_w_grid_num_accepted_order: Order for tie-breaking.
            - history_w_grid_num_accepted_lenience: Lenience threshold.

    Returns:
        Best weight vector as a NumPy array.
    """
    device = w_grid_tensor.device

    _history_current_drafting_concat = (
        torch.from_numpy(_history_current_drafting_concat).float().to(device)
    )
    _history_current_target = (
        torch.from_numpy(_history_current_target).float().to(device)
    )

    history_w_grid_measure = kwargs.get('history_w_grid_measure')
    history_w_grid_num_accepted_order = kwargs.get(
        'history_w_grid_num_accepted_order', 'first'
    )
    history_w_grid_num_accepted_lenience = kwargs.get(
        'history_w_grid_num_accepted_lenience', 0.0
    )

    # Rearrange target from (B, history_window, vocab_size) to
    # (history_window, B, vocab_size)
    _history_current_target = rearrange(_history_current_target, "b h v -> h b v")

    # Compute ensemble predictions for all weight combinations
    ensembled_probs_candidate_w_grid = w_grid_tensor @ _history_current_drafting_concat

    if 'num-accepted' in history_w_grid_measure:
        w_best_index = _get_best_by_num_accepted(
            w_grid_tensor,
            ensembled_probs_candidate_w_grid,
            _history_current_target,
            history_w_grid_measure,
            history_w_grid_num_accepted_order,
            history_w_grid_num_accepted_lenience,
        )
    elif history_w_grid_measure in ['kld', 'tvd']:
        w_best_index = _get_best_by_divergence(
            ensembled_probs_candidate_w_grid,
            _history_current_target,
            history_w_grid_measure,
        )
    else:
        raise ValueError(f"Unsupported measure: {history_w_grid_measure}")

    return w_grid_tensor[w_best_index].cpu().numpy()


def _get_best_by_num_accepted(
    w_grid_tensor: torch.Tensor,
    ensembled_probs: torch.Tensor,
    target: torch.Tensor,
    measure: str,
    order: Optional[str],
    lenience: float,
) -> int:
    """Find best weights by number of accepted tokens.

    Args:
        w_grid_tensor: Weight combinations tensor.
        ensembled_probs: Ensemble predictions.
        target: Target distributions.
        measure: Measure type (may include 'kld' or 'tvd').
        order: Tie-breaking order.
        lenience: Lenience threshold for score comparison.

    Returns:
        Index of the best weight combination.
    """
    pred_w_grid = ensembled_probs.argmax(dim=-1)
    hard_label = target.argmax(dim=-1)

    scores = (pred_w_grid == hard_label).sum(dim=0)
    score_threshold = scores.max() * (1.0 - lenience)
    max_ids_torch = torch.where(scores >= score_threshold)[0]
    max_ids = max_ids_torch.cpu().numpy()

    if 'kld' in measure:
        return _get_best_with_kld(
            ensembled_probs, target, max_ids_torch, max_ids, order
        )
    elif 'tvd' in measure:
        return _get_best_with_tvd(
            ensembled_probs, target, max_ids_torch, max_ids, order
        )
    elif order == 'none':
        return _get_best_with_tracking(
            w_grid_tensor, max_ids_torch, max_ids, scores
        )
    elif order == 'first':
        return max_ids[0]
    elif order == 'last':
        return max_ids[-1]
    elif order == 'middle':
        return max_ids[len(max_ids) // 2]
    elif order == 'random':
        return np.random.choice(max_ids)
    else:
        return max_ids[0]


def _get_best_with_kld(
    ensembled_probs: torch.Tensor,
    target: torch.Tensor,
    max_ids_torch: torch.Tensor,
    max_ids: np.ndarray,
    order: Optional[str],
) -> int:
    """Get best weight index using KL divergence for tie-breaking.

    Args:
        ensembled_probs: Ensemble predictions.
        target: Target distributions.
        max_ids_torch: Indices of best candidates (torch).
        max_ids: Indices of best candidates (numpy).
        order: Tie-breaking order.

    Returns:
        Index of the best weight combination.
    """
    pred_subset = ensembled_probs[:, max_ids_torch, :]
    target_subset = target.expand_as(pred_subset)

    kl_divergence = F.kl_div(pred_subset.log(), target_subset, reduction='none')
    scores_kld = kl_divergence.mean(dim=-1)
    summed_scores_kld = scores_kld.sum(dim=0)

    if order is not None and 'local' in str(order):
        order_value = float(order.split('local')[-1])
        w_best_local = get_percentile_indices(scores_kld[-1, :], order_value)
    elif order is not None:
        w_best_local = get_percentile_indices(summed_scores_kld, order)
    else:
        w_best_local = scores_kld.sum(dim=0).argmin().item()

    if isinstance(w_best_local, torch.Tensor):
        w_best_local = w_best_local.item()

    return max_ids[w_best_local]


def _get_best_with_tvd(
    ensembled_probs: torch.Tensor,
    target: torch.Tensor,
    max_ids_torch: torch.Tensor,
    max_ids: np.ndarray,
    order: Optional[str],
) -> int:
    """Get best weight index using total variation distance for tie-breaking.

    Args:
        ensembled_probs: Ensemble predictions.
        target: Target distributions.
        max_ids_torch: Indices of best candidates (torch).
        max_ids: Indices of best candidates (numpy).
        order: Tie-breaking order.

    Returns:
        Index of the best weight combination.
    """
    pred_subset = ensembled_probs[:, max_ids_torch, :]
    target_subset = target.expand_as(pred_subset)

    absolute_differences = torch.abs(pred_subset - target_subset)
    scores_tvd = 0.5 * absolute_differences.sum(dim=-1)
    summed_scores_tvd = scores_tvd.sum(dim=0)

    if order is not None and 'local' in str(order):
        order_value = float(order.split('local')[-1])
        w_best_local = get_percentile_indices(scores_tvd[-1, :], order_value)
    elif order is not None:
        w_best_local = get_percentile_indices(summed_scores_tvd, order)
    else:
        w_best_local = scores_tvd.sum(dim=0).argmin().item()

    if isinstance(w_best_local, torch.Tensor):
        w_best_local = w_best_local.item()

    return max_ids[w_best_local]


def _get_best_with_tracking(
    w_grid_tensor: torch.Tensor,
    max_ids_torch: torch.Tensor,
    max_ids: np.ndarray,
    scores: torch.Tensor,
) -> int:
    """Get best weight index while tracking all candidates for oracle testing.

    Args:
        w_grid_tensor: Weight combinations tensor.
        max_ids_torch: Indices of best candidates (torch).
        max_ids: Indices of best candidates (numpy).
        scores: Match scores for each candidate.

    Returns:
        Index of the first best weight combination.
    """
    if not hasattr(get_best_from_w_grid, '_all_candidates'):
        get_best_from_w_grid._all_candidates = []

    best_candidates = w_grid_tensor[max_ids_torch]

    if scores.max() != 0:
        get_best_from_w_grid._all_candidates.append(best_candidates.cpu().numpy())

    return max_ids[0]


def _get_best_by_divergence(
    ensembled_probs: torch.Tensor,
    target: torch.Tensor,
    measure: str,
) -> int:
    """Get best weight index by minimizing divergence measure.

    Args:
        ensembled_probs: Ensemble predictions.
        target: Target distributions.
        measure: Divergence measure ('kld' or 'tvd').

    Returns:
        Index of the best weight combination.
    """
    pred_w_grid = ensembled_probs
    soft_label = target
    soft_label_expanded = soft_label.expand_as(pred_w_grid)

    if measure == 'kld':
        kl_divergence = F.kl_div(
            pred_w_grid.log(), soft_label_expanded, reduction='none'
        )
        scores_reversed = kl_divergence.mean(dim=-1)
    else:  # 'tvd'
        absolute_differences = torch.abs(pred_w_grid - soft_label_expanded)
        scores_reversed = 0.5 * absolute_differences.sum(dim=-1)

    w_best_index_torch = scores_reversed.sum(dim=0).argmin()
    return w_best_index_torch.item()


def get_percentile_indices(scores: torch.Tensor, order: Union[str, float]) -> int:
    """Get the index at a specified percentile of scores.

    Args:
        scores: Tensor of scores to sort.
        order: Percentile value (0.0 to 1.0) as string or float.

    Returns:
        Index corresponding to the specified percentile.
    """
    _idx = round(float(order) * scores.size(0))
    sorted_indices = np.argsort(scores.cpu().numpy())
    return sorted_indices[_idx].item()
