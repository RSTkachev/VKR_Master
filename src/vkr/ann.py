"""Approximate nearest-neighbour retrieval over normalised embeddings using FAISS HNSW."""

import faiss
import numpy as np

from vkr.utils import _to_numpy_f32


def build_hnsw_index(embeds: np.ndarray, M: int = 32):
    """Build a FAISS HNSW index with inner-product metric.

    Args:
        embeds: float32 matrix of shape ``(N, D)`` with pre-normalised vectors.
        M: number of neighbours per HNSW node.

    Return: populated ``faiss.IndexHNSWFlat``.
    """
    d = embeds.shape[1]
    index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 200
    index.add(embeds)
    return index


def ann_ranks_for_aligned_pairs(
    query_embeds,
    target_embeds,
    top_k: int = 100,
):
    """Compute ranks of aligned (query, target) pairs via HNSW search.

    Args:
        query_embeds: query embeddings of shape ``(N, D)``.
        target_embeds: target embeddings; row ``i`` is the ground truth for query ``i``.
        top_k: number of neighbours retrieved per query; missing positives receive rank ``top_k + 1``.

    Return: tuple ``(ranks, retrieved)`` where ``ranks`` has shape ``(N,)`` int32
        and ``retrieved`` has shape ``(N, top_k)`` with target indices per query.
    """
    q = _to_numpy_f32(query_embeds)
    t = _to_numpy_f32(target_embeds)

    index = build_hnsw_index(t)

    index.hnsw.efSearch = max(64, top_k * 2)

    _, retrieved = index.search(q, top_k)
    gt = np.arange(q.shape[0], dtype=np.int64)[:, None]

    matches = retrieved == gt
    found = matches.any(axis=1)
    pos = matches.argmax(axis=1)

    ranks = np.where(found, pos + 1, top_k + 1).astype(np.int32)
    return ranks, retrieved
