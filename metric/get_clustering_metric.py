#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import sys
import warnings
from typing import List, Tuple, Dict, Any

try:
    from sklearn import metrics
    # Suppress UserWarning in sklearn clustering metrics
    warnings.filterwarnings('ignore', message='.*The number of unique classes is greater than 50% of the number of samples.*', category=UserWarning)
except Exception as import_error:
    sys.stderr.write(
        "Failed to import scikit-learn. Please install it first, e.g.:\n"
        "  pip install scikit-learn\n"
        f"Details: {import_error}\n"
    )
    sys.exit(1)


def compute_clustering_metrics(alignments: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Extract labels from alignments and compute clustering metrics
    
    Args:
        alignments: List of alignment data containing gt_cluster_id and model_cluster_id
    
    Returns:
        Dictionary containing ARI and V-measure
    """
    labels_true: List[int] = []
    labels_pred: List[int] = []
    
    for item in alignments:
        try:
            gt_cluster_id = int(item["gt_cluster_id"])
            model_cluster_id = int(item["model_cluster_id"])
            labels_true.append(gt_cluster_id)
            labels_pred.append(model_cluster_id)
        except (KeyError, ValueError) as e:
            continue
    
    if len(labels_true) == 0:
        return {
            "ari": 0.0,
            "v_measure": 0.0,
            "homogeneity": 0.0,
            "completeness": 0.0,
            "num_samples": 0
        }
    
    # Compute metrics
    ari = metrics.adjusted_rand_score(labels_true, labels_pred)
    v_measure = metrics.v_measure_score(labels_true, labels_pred)
    homogeneity = metrics.homogeneity_score(labels_true, labels_pred)
    completeness = metrics.completeness_score(labels_true, labels_pred)
    
    return {
        "ari": float(ari),
        "v_measure": float(v_measure),
        "homogeneity": float(homogeneity),
        "completeness": float(completeness),
        "num_samples": len(labels_true)
    }


def process_paper_alignment_file(input_file: str, output_file: str) -> List[Dict[str, Any]]:
    """
    Process alignment_fixed.jsonl file and compute clustering metrics for each survey
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSON file
    
    Returns:
        List of processing results
    """
    # Read input file (JSONL format, one JSON object per line)
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                survey = json.loads(line)
                data.append(survey)
            except json.JSONDecodeError as e:
                continue
    
    if not data:
        return []
    
    results = []
    
    # Process each survey
    for survey in data:
        try:
            survey_id = survey.get("survey_id")
            survey_topic = survey.get("survey_topic", "")
            gt_cluster_count = survey.get("gt_cluster_count", 0)
            model_cluster_count = survey.get("model_cluster_count", 0)
            alignments = survey.get("alignments", [])
            
            if not alignments:
                continue
            
            # Compute clustering metrics
            metrics_result = compute_clustering_metrics(alignments)
            
            # Build result
            result = {
                "survey_id": survey_id,
                "survey_topic": survey_topic,
                "gt_cluster_count": gt_cluster_count,
                "model_cluster_count": model_cluster_count,
                "metrics": metrics_result
            }
            
            results.append(result)
            
        except Exception as e:
            continue
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Compute average metrics
    if results:
        avg_ari = sum(r["metrics"]["ari"] for r in results) / len(results)
        avg_v_measure = sum(r["metrics"]["v_measure"] for r in results) / len(results)
        avg_homogeneity = sum(r["metrics"]["homogeneity"] for r in results) / len(results)
        avg_completeness = sum(r["metrics"]["completeness"] for r in results) / len(results)
        
        print("\nAverage metrics:")
        print(f"  ARI (Adjusted Rand Index): {avg_ari:.6f}")
        print(f"  V-measure: {avg_v_measure:.6f}")
        print(f"  Homogeneity: {avg_homogeneity:.6f}")
        print(f"  Completeness: {avg_completeness:.6f}")
    
    return results


def process_single_model(model_name: str, model_dir: str):
    """
    Process data for a single model
    
    Args:
        model_name: Model name
        model_dir: Model folder path
    """
    input_file = os.path.join(model_dir, "paper_alignment_all.json")
    output_file = os.path.join(model_dir, "clustering_metrics.json")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        return
    
    try:
        process_paper_alignment_file(input_file, output_file)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise


def main():
    """
    Batch process data for all models
    """
    # Base path and model list
    base_dir = "model/output"
    models = ["deepseek", "deepseek-thinking", "kimi", "kimi-thinking", "qwen", "qwen-thinking"]
    
    # Iterate through all models
    for model_name in models:
        model_dir = os.path.join(base_dir, model_name)
        
        # Check if model directory exists
        if not os.path.exists(model_dir):
            continue
        
        try:
            process_single_model(model_name, model_dir)
        except Exception as e:
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()

