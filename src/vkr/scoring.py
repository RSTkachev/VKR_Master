"""Retrieval scoring utilities: rank computation and standard recall metrics."""

from typing import Dict, Optional

import numpy as np


def retrieval_metrics_from_ranks(
    ranks: np.ndarray, levels: Optional[list] = None
) -> Dict[str, float]:
    """Compute recall-at-K, MRR and rank summary statistics.

    Args:
        ranks: 1-based ranks of the ground-truth item per query.
        levels: extra K values for ``recall@K`` on top of the defaults 1, 5, 10.

    Return: dictionary with ``r_at_K`` entries plus ``mrr``, ``median_rank``, ``mean_rank``.
    """
    recall_levels = [1, 5, 10]
    if levels is not None:
        recall_levels.extend(levels)
    recall_levels = sorted(set(recall_levels))

    results = {
        f"r_at_{level}": float(np.mean(ranks <= level)) for level in recall_levels
    }
    results["mrr"] = float(np.mean(1.0 / ranks))
    results["median_rank"] = float(np.median(ranks))
    results["mean_rank"] = float(np.mean(ranks))
    return results


def compute_ranks(similarity: np.ndarray) -> np.ndarray:
    """Compute the 1-based rank of the diagonal entry in each row of a similarity matrix.

    Args:
        similarity: square similarity matrix; row ``i`` and column ``i`` are the ground-truth pair.

    Return: int32 array of 1-based ranks per row.
    """
    sorted_indices = np.argsort(-similarity, axis=1)
    gt = np.arange(similarity.shape[0])[:, None]
    ranks = (sorted_indices == gt).argmax(axis=1) + 1
    return ranks.astype(np.int32)
