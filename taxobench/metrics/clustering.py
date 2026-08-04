from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from sklearn.metrics import adjusted_rand_score, completeness_score, homogeneity_score, v_measure_score

from .alignment import align_titles
from .tree import extract_clusters, extract_papers


def _cluster_map(clusters: Sequence[Sequence[str]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for cluster_id, cluster in enumerate(clusters):
        for title in cluster:
            out[str(title)] = cluster_id
    return out


def clustering_metrics(reference_tree: dict, prediction_tree: dict, threshold: float = 0.92, include_unretrieved: bool = True) -> Dict[str, float]:
    ref_clusters = extract_clusters(reference_tree)
    pred_clusters = extract_clusters(prediction_tree)
    ref_map = _cluster_map(ref_clusters)
    pred_map = _cluster_map(pred_clusters)
    pred_titles = list(pred_map.keys())
    alignments = align_titles(list(ref_map.keys()), pred_titles, threshold=threshold)
    y_true: List[int] = []
    y_pred: List[int] = []
    missing_cluster = len(pred_clusters)
    for row in alignments:
        ref = row["reference_title"]
        pred = row["prediction_title"]
        if pred is None and not include_unretrieved:
            continue
        y_true.append(ref_map[str(ref)])
        y_pred.append(pred_map[str(pred)] if pred is not None else missing_cluster)
    if len(y_true) < 2:
        return {"ari": 0.0, "v_measure": 0.0, "homogeneity": 0.0, "completeness": 0.0, "n_aligned": float(len(y_true))}
    return {
        "ari": float(adjusted_rand_score(y_true, y_pred)),
        "v_measure": float(v_measure_score(y_true, y_pred)),
        "homogeneity": float(homogeneity_score(y_true, y_pred)),
        "completeness": float(completeness_score(y_true, y_pred)),
        "n_aligned": float(sum(1 for row in alignments if row["prediction_title"] is not None)),
        "n_reference": float(len(ref_map)),
        "n_prediction": float(len(pred_titles)),
    }
