"""
SDIQA Evaluation Metrics
========================

Standard quality-assessment correlation metrics:

* PLCC - Pearson Linear Correlation Coefficient
* SRCC - Spearman Rank-order Correlation Coefficient

For a stable training signal, both metrics are reported as their absolute
value: a near-zero variance in either input array yields ``0.0``.
"""

from typing import List, Tuple

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr


def compute_plcc(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Pearson linear correlation between two 1-D arrays."""
    if np.std(predictions) < 1e-8 or np.std(targets) < 1e-8:
        return 0.0
    plcc, _ = pearsonr(predictions, targets)
    return abs(plcc) if not np.isnan(plcc) else 0.0


def compute_srcc(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Spearman rank correlation between two 1-D arrays."""
    if np.std(predictions) < 1e-8 or np.std(targets) < 1e-8:
        return 0.0
    srcc, _ = spearmanr(predictions, targets)
    return abs(srcc) if not np.isnan(srcc) else 0.0


def compute_plcc_per_dim(
    predictions: torch.Tensor, targets: torch.Tensor
) -> Tuple[List[float], float]:
    """Per-dimension PLCC and the average over all dimensions."""
    preds_np = predictions.cpu().numpy()
    targets_np = targets.cpu().numpy()
    num_dims = preds_np.shape[1]

    plcc_scores = [
        compute_plcc(preds_np[:, i], targets_np[:, i]) for i in range(num_dims)
    ]
    return plcc_scores, float(np.mean(plcc_scores))


def compute_srcc_per_dim(
    predictions: torch.Tensor, targets: torch.Tensor
) -> Tuple[List[float], float]:
    """Per-dimension SRCC and the average over all dimensions."""
    preds_np = predictions.cpu().numpy()
    targets_np = targets.cpu().numpy()
    num_dims = preds_np.shape[1]

    srcc_scores = [
        compute_srcc(preds_np[:, i], targets_np[:, i]) for i in range(num_dims)
    ]
    return srcc_scores, float(np.mean(srcc_scores))
