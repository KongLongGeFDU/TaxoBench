from __future__ import annotations

from typing import Dict, Sequence

from .alignment import align_titles


def retrieval_metrics(reference_titles: Sequence[str], retrieved_titles: Sequence[str], threshold: float = 0.92) -> Dict[str, float]:
    rows = align_titles(reference_titles, retrieved_titles, threshold=threshold)
    hits = sum(1 for row in rows if row["prediction_title"] is not None)
    recall = hits / len(reference_titles) if reference_titles else 0.0
    precision = hits / len(retrieved_titles) if retrieved_titles else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"recall": recall, "precision": precision, "f1": f1, "n_hits": float(hits), "n_reference": float(len(reference_titles)), "n_retrieved": float(len(retrieved_titles))}
