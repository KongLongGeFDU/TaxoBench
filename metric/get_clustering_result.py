import json
import re
import os
from pathlib import Path
from difflib import SequenceMatcher
from typing import List, Dict, Tuple

def extract_clusters_from_tree(tree: Dict) -> List[List[str]]:
    """
    Recursively extract papers from all leaf nodes in the tree structure, each leaf node represents a cluster
    Returns: List[List[str]] - Each sublist is a cluster containing all papers in that cluster
    """
    clusters = []
    
    def traverse(node: Dict):
        if "papers" in node:
            # Leaf node, contains papers field
            clusters.append(node["papers"])
        elif "subtopics" in node:
            # Non-leaf node, continue traversing child nodes
            for subtopic in node["subtopics"]:
                traverse(subtopic)
    
    traverse(tree)
    return clusters

def normalize_paper_name(name: str) -> str:
    """
    Normalize paper name for matching
    Remove punctuation, convert to lowercase, remove extra spaces
    """
    # Convert to lowercase
    name = name.lower()
    # Remove punctuation (keep letters, numbers and spaces)
    name = re.sub(r'[^\w\s]', '', name)
    # Remove extra spaces
    name = re.sub(r'\s+', ' ', name).strip()
    return name

def find_best_match(gt_paper: str, model_papers: List[str], used_indices: set) -> Tuple[str, int, float]:
    """
    Find the best matching paper in the model paper list for the gt paper
    Returns: (matched paper name, index, similarity score)
    """
    gt_normalized = normalize_paper_name(gt_paper)
    best_match = None
    best_score = 0.0
    best_idx = -1
    
    for idx, model_paper in enumerate(model_papers):
        if idx in used_indices:
            continue
        
        model_normalized = normalize_paper_name(model_paper)
        # Use SequenceMatcher to compute similarity
        score = SequenceMatcher(None, gt_normalized, model_normalized).ratio()
        
        if score > best_score:
            best_score = score
            best_match = model_paper
            best_idx = idx
    
    return best_match, best_idx, best_score

def check_contains_relationship(gt_paper: str, model_paper: str) -> bool:
    """
    Check if two paper names have a containment relationship
    Returns: True if model_paper contains gt_paper or gt_paper contains model_paper (entire string)
    """
    gt_normalized = normalize_paper_name(gt_paper)
    model_normalized = normalize_paper_name(model_paper)
    
    # Check if model_paper contains gt_paper (entire string)
    if gt_normalized in model_normalized:
        return True
    
    # Check if gt_paper contains model_paper (entire string)
    if model_normalized in gt_normalized:
        return True
    
    return False

def align_papers(gt_clusters: List[List[str]], model_clusters: List[List[str]], 
                 threshold: float = 1.0) -> List[Dict]:
    """
    Align papers between gt and model
    Returns: List[Dict] - Each element contains gt paper name, model paper name, gt cluster id, model cluster id
    """
    # Create mapping from paper to cluster id
    gt_paper_to_cluster = {}
    for cluster_id, cluster in enumerate(gt_clusters):
        for paper in cluster:
            gt_paper_to_cluster[paper] = cluster_id
    
    model_paper_to_cluster = {}
    for cluster_id, cluster in enumerate(model_clusters):
        for paper in cluster:
            model_paper_to_cluster[paper] = cluster_id
    
    # Align papers
    alignments = []
    used_model_papers = set()
    
    # Iterate through all gt papers to find corresponding model papers
    for gt_paper in gt_paper_to_cluster.keys():
        gt_cluster_id = gt_paper_to_cluster[gt_paper]
        
        # Find best match among all model papers
        all_model_papers = []
        for cluster in model_clusters:
            all_model_papers.extend(cluster)
        
        best_match, best_idx, best_score = find_best_match(
            gt_paper, all_model_papers, used_model_papers
        )
        
        # Determine if matched
        is_matched = False
        
        if best_match:
            # Case 1: Exact match (similarity >= 1.0)
            if best_score >= threshold:
                is_matched = True
            # Case 2: Similarity between 0.6 and 1.0, check containment relationship
            elif 0.6 <= best_score < threshold:
                if check_contains_relationship(gt_paper, best_match):
                    is_matched = True
        
        if is_matched:
            model_cluster_id = model_paper_to_cluster[best_match]
            alignments.append({
                "gt_paper": gt_paper,
                "model_paper": best_match,
                "gt_cluster_id": gt_cluster_id,
                "model_cluster_id": model_cluster_id,
                "similarity": best_score
            })
            used_model_papers.add(best_idx)
        else:
            # No matching paper found
            alignments.append({
                "gt_paper": gt_paper,
                "model_paper": None,
                "gt_cluster_id": gt_cluster_id,
                "model_cluster_id": None,
                "similarity": best_score if best_match else 0.0
            })
    
    return alignments

def process_single_model(model_name: str, model_dir: str):
    """
    Process data for a single model
    
    Args:
        model_name: Model name
        model_dir: Model folder path
    """
    input_file = os.path.join(model_dir, "merged.jsonl")
    output_file = os.path.join(model_dir, "paper_alignment_all.json")
    txt_output_file = os.path.join(model_dir, "paper_alignment_all.txt")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        return
    
    all_results = []
    
    # Read all data
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                survey_id = data.get("idx", 0)
                
                # Extract gt and hierarchy_tree
                gt_tree = data["gt"]
                model_tree = data["hierarchy_tree"]
                
                # Extract clusters
                gt_clusters = extract_clusters_from_tree(gt_tree)
                model_clusters = extract_clusters_from_tree(model_tree)
                
                # Align papers
                alignments = align_papers(gt_clusters, model_clusters)
                
                # Build output data structure
                output_data = {
                    "survey_id": survey_id,
                    "survey_topic": data.get("survey_topic", ""),
                    "gt_cluster_count": len(gt_clusters),
                    "model_cluster_count": len(model_clusters),
                    "alignments": alignments
                }
                
                all_results.append(output_data)
                
            except json.JSONDecodeError as e:
                continue
            except Exception as e:
                import traceback
                traceback.print_exc()
                continue
    
    # Save all results to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    # Save all results to text file: gt_paper_name-model_paper_name-gt_cluster_id-model_cluster_id
    with open(txt_output_file, 'w', encoding='utf-8') as f:
        for result in all_results:
            survey_id = result['survey_id']
            # Add separator marker for each survey
            f.write(f"# Survey ID: {survey_id}\n")
            for alignment in result['alignments']:
                gt_paper = alignment['gt_paper']
                model_paper = alignment['model_paper'] if alignment['model_paper'] else 'N/A'
                gt_cluster_id = alignment['gt_cluster_id']
                model_cluster_id = alignment['model_cluster_id'] if alignment['model_cluster_id'] is not None else 'N/A'
                f.write(f"{gt_paper}-{model_paper}-{gt_cluster_id}-{model_cluster_id}\n")
            f.write("\n")  # Empty line between surveys
    
    if all_results:
        total_matched = sum(sum(1 for a in r['alignments'] if a['model_paper'] is not None) for r in all_results)
        total_unmatched = sum(sum(1 for a in r['alignments'] if a['model_paper'] is None) for r in all_results)
        total_papers = total_matched + total_unmatched
        
        print(f"\nModel {model_name} overall statistics:")
        print(f"  Total papers: {total_papers}")
        print(f"  Successfully aligned: {total_matched} ({total_matched/total_papers*100:.2f}%)")
        print(f"  Unaligned: {total_unmatched} ({total_unmatched/total_papers*100:.2f}%)")


def process_all_data():
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
    process_all_data()

