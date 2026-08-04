from __future__ import annotations

from functools import lru_cache
from typing import Callable, Dict, List

import numpy as np

from .alignment import align_titles
from .tree import extract_papers, iter_leaf_paths


def _fallback_similarity(a: str, b: str) -> float:
    from .alignment import title_similarity
    return title_similarity(a, b)


def _path_cost(short: List[str], long: List[str], similarity: Callable[[str, str], float], unmatched_penalty: float) -> float:
    n, m = len(short), len(long)
    if n == 0:
        return m * unmatched_penalty
    if m == 0:
        return n * unmatched_penalty
    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, :] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if j < i:
                continue
            match = dp[i - 1, j - 1] + (1.0 - max(0.0, similarity(short[i - 1], long[j - 1])))
            skip = dp[i, j - 1]
            dp[i, j] = min(match, skip)
    return float(dp[n, m] + unmatched_penalty * (m - n))


def sem_path_score(reference_tree: dict, prediction_tree: dict, threshold: float = 0.92, unmatched_penalty: float = 1.0, similarity: Callable[[str, str], float] | None = None) -> Dict[str, float]:
    """Compute a lightweight Sem-Path score for aligned paper paths.

    For exact paper-level reproduction, use the same embedding model and title
    alignment settings described in the paper. This public implementation uses a
    deterministic string-similarity fallback unless a custom similarity function
    is supplied.
    """
    similarity = similarity or _fallback_similarity
    ref_paths = iter_leaf_paths(reference_tree)
    pred_paths = iter_leaf_paths(prediction_tree)
    alignments = align_titles(list(ref_paths), list(pred_paths), threshold=threshold)
    sims: List[float] = []
    for row in alignments:
        pred = row["prediction_title"]
        if pred is None:
            continue
        best = 0.0
        for rp in ref_paths[str(row["reference_title"] )]:
            for pp in pred_paths[str(pred)]:
                r_anc = rp[:-1]
                p_anc = pp[:-1]
                short, long = (r_anc, p_anc) if len(r_anc) <= len(p_anc) else (p_anc, r_anc)
                cost = _path_cost(short, long, similarity, unmatched_penalty)
                best = max(best, 1.0 / (1.0 + cost))
        sims.append(best)
    return {"sem_path": float(np.mean(sims)) if sims else 0.0, "n_aligned": float(len(sims)), "n_reference": float(len(ref_paths))}
