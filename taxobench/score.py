from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

from .metrics.clustering import clustering_metrics
from .metrics.retrieval import retrieval_metrics
from .metrics.sem_path import sem_path_score
from .metrics.tree import extract_papers


def load_jsonl(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Score TaxoBench prediction JSONL files.")
    parser.add_argument("--data", type=Path, required=True, help="TaxoBench dataset JSONL.")
    parser.add_argument("--predictions", type=Path, required=True, help="Prediction JSONL with id and hierarchy_tree fields.")
    parser.add_argument("--threshold", type=float, default=0.92, help="Title alignment threshold.")
    parser.add_argument("--output", type=Path, default=None, help="Optional per-survey JSONL output.")
    args = parser.parse_args()

    refs = {item["id"]: item for item in load_jsonl(args.data)}
    preds = {item["id"]: item for item in load_jsonl(args.predictions)}
    rows: List[dict] = []
    for survey_id, ref in refs.items():
        pred = preds.get(survey_id)
        if pred is None:
            continue
        ref_tree = ref["gt"]
        pred_tree = pred.get("hierarchy_tree") or pred.get("tree")
        if not isinstance(pred_tree, dict):
            continue
        row = {"id": survey_id, "survey_topic": ref.get("survey_topic")}
        row.update({f"leaf_{k}": v for k, v in clustering_metrics(ref_tree, pred_tree, threshold=args.threshold).items()})
        row.update({f"sem_path_{k}": v for k, v in sem_path_score(ref_tree, pred_tree, threshold=args.threshold).items()})
        retrieved = pred.get("retrieved_papers") or extract_papers(pred_tree)
        row.update({f"retrieval_{k}": v for k, v in retrieval_metrics(extract_papers(ref_tree), retrieved, threshold=args.threshold).items()})
        rows.append(row)

    def mean(key: str) -> float:
        vals = [float(r[key]) for r in rows if key in r]
        return sum(vals) / len(vals) if vals else 0.0

    summary = {
        "n_scored": len(rows),
        "leaf_ari": mean("leaf_ari"),
        "leaf_v_measure": mean("leaf_v_measure"),
        "leaf_homogeneity": mean("leaf_homogeneity"),
        "leaf_completeness": mean("leaf_completeness"),
        "sem_path": mean("sem_path_sem_path"),
        "retrieval_recall": mean("retrieval_recall"),
        "retrieval_precision": mean("retrieval_precision"),
        "retrieval_f1": mean("retrieval_f1"),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.output:
        with args.output.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
