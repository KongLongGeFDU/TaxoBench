# This version does not calculate the paper node for the two sequences; it still only counts recall.
import os
import torch 
import numpy as np
import argparse
import json
from collections import defaultdict
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None  # type: ignore


# ----------------------------
# Basic Components: Tree and Vector Model
# ----------------------------
class TreeNode:
    def __init__(self, name: str, children: Optional[List["TreeNode"]] = None):
        self.name = name
        self.children: List["TreeNode"] = children or []


class Embedder:
    """
    Uses only SentenceTransformer.
    """

    def __init__(self, model_name: str = "/root/new_metric/code/models--sentence-transformers--all-MiniLM-L6-v2"):
        if SentenceTransformer is None:
            raise ImportError(
                "Please install sentence-transformers library first, e.g.:\n"
                "  pip install -U sentence-transformers\n"
                "or in conda environment:\n"
                "  conda install -c conda-forge sentence-transformers"
            )
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load SentenceTransformer: {e}") from e

    @lru_cache(maxsize=4096)
    def encode(self, text: str):
        return self.model.encode(text)

    def cosine(self, a: str, b: str) -> float:
        va = self.encode(a)
        vb = self.encode(b)
        va_arr = np.array(va)
        vb_arr = np.array(vb)
        denom = (np.linalg.norm(va_arr) * np.linalg.norm(vb_arr))
        if denom == 0:
            return 0.0
        return float(np.dot(va_arr, vb_arr) / denom)


# ----------------------------
# Tree Building and Path Collection
# ----------------------------
def build_tree(node_dict: Dict | List) -> TreeNode:
    """
    Build TreeNode from JSON tree node.
    Includes full tree structure: name, subtopics, and papers (as leaf nodes).
    Supports dictionary and list formats.
    """
    if isinstance(node_dict, list):
        if not node_dict:
            return TreeNode(name="", children=[])
        if len(node_dict) == 1:
            return build_tree(node_dict[0])
        children = [build_tree(child) for child in node_dict]
        return TreeNode(name="", children=children)

    if not isinstance(node_dict, dict):
        return TreeNode(name="", children=[])

    name = node_dict.get("name", "")
    children: List[TreeNode] = []

    subtopics_data = node_dict.get("subtopics", [])
    if subtopics_data:
        children.extend([build_tree(child) for child in subtopics_data])

    papers_data = node_dict.get("papers", [])
    if papers_data:
        for paper_title in papers_data:
            if paper_title:
                children.append(TreeNode(name=str(paper_title), children=[]))

    return TreeNode(name=name, children=children)


def _collect_paths(node: TreeNode, current: List[str], result: Dict[str, List[List[str]]]):
    """
    DFS collection of paths from root to paper leaves.
    result maps paper_title -> list of multiple paths.
    """
    new_path = current + ([node.name] if node.name else [])
    if not node.children:
        # Treat leaves as papers
        paper_name = node.name.strip()
        if paper_name:
            result[paper_name].append(new_path)
        return

    for child in node.children:
        _collect_paths(child, new_path, result)


def extract_paper_paths(root: TreeNode) -> Dict[str, List[List[str]]]:
    paths: Dict[str, List[List[str]]] = defaultdict(list)
    _collect_paths(root, [], paths)
    return paths


# ----------------------------
# Paper Alignment
# ----------------------------
def _normalize(text: str) -> str:
    return text.lower().strip()


def is_same_paper(p1: str, p2: str, embedder: Embedder) -> bool:
    """
    Determine if two papers can be considered the same citation:
    1) Text exact match or inclusion
    2) Embedding cosine similarity reaches 1 (strict requirement)
    """
    t1 = _normalize(p1)
    t2 = _normalize(p2)
    if not t1 or not t2:
        return False
    if t1 == t2 or t1 in t2 or t2 in t1:
        return True
    cos = embedder.cosine(t1, t2)
    return cos >= 0.999999


def find_common_papers(
    gt_map: Dict[str, List[List[str]]],
    model_map: Dict[str, List[List[str]]],
    embedder: Embedder,
) -> List[Tuple[str, str]]:
    """
    Returns list of aligned paper pairs (gt_title, model_title).
    """
    matched_pairs: List[Tuple[str, str]] = []
    model_titles = list(model_map.keys())
    for gt_title in gt_map.keys():
        for model_title in model_titles:
            if is_same_paper(gt_title, model_title, embedder):
                matched_pairs.append((gt_title, model_title))
                break  # Match the same expert paper to the first model paper found
    return matched_pairs


# ----------------------------
# Path Similarity (DTW + Cosine Distance)
# ----------------------------
def embed_path(path: List[str], embedder: Embedder) -> List[np.ndarray]:
    return [np.array(embedder.encode(p)) for p in path]

def cosine_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    denom = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if denom == 0:
        return 1.0
    cos_sim = float(np.dot(vec1, vec2) / denom)
    cos_sim = max(cos_sim, 0.0)  # clamp negative to 0
    return 1.0 - cos_sim


def calculate_min_alignment_cost(gt_vecs: list, model_vecs: list) -> float:
    """
    Calculate minimal alignment cost between two Embedding sequences.
    Maintains strict partial order relationship.
    
    Penalty for extra nodes set to 1.0 (represents irrelevant/orthogonal).
    """
    len_gt = len(gt_vecs)
    len_model = len(model_vecs)

    # If both are empty, cost is 0
    if len_gt == 0 and len_model == 0:
        return 0.0, 0
    # If one is empty, all nodes are treated as unmatched (Penalty = 1.0)
    if len_gt == 0:
        return float(len_model * 1.0), len_model
    if len_model == 0:
        return float(len_gt * 1.0), len_gt
    
    # --- Case 1: Equal length ---
    if len_gt == len_model:
        cost = 0.0
        for v1, v2 in zip(gt_vecs, model_vecs):
            cost += cosine_distance(v1, v2)
        return cost, len_gt

    # --- Case 2: Unequal length ---
    # Determine short sequence and long sequence
    if len_gt < len_model:
        short_vecs, long_vecs = gt_vecs, model_vecs
    else:
        short_vecs, long_vecs = model_vecs, gt_vecs
        
    n = len(short_vecs)
    m = len(long_vecs)
    
    # [Modification] Fixed penalty for unmatched nodes
    # Since unmatched is treated as "irrelevant", the cost for each extra node is 1.0
    unmatched_penalty = (m - n) * 1.0
    
    # dp[i][j] represents: minimal [matching cost] for the subsequence of the first i items
    # of the short sequence matching the first j items of the long sequence
    # Note: Only calculates sum of distances for "successfully paired" parts; penalty for unmatched parts is added at the end
    dp = np.full((n + 1, m + 1), np.inf)
    
    # Initialization
    for j in range(m + 1):
        dp[0][j] = 0.0
        
    # Dynamic programming table filling
    for i in range(1, n + 1):      # short sequence
        for j in range(1, m + 1):  # long sequence
            
            if j < i: # Pruning: Remaining length of long sequence is insufficient to match remaining part of short sequence
                continue
            
            # Option 1: Match
            # Pair short[i] with long[j]
            match_cost = dp[i-1][j-1] + cosine_distance(short_vecs[i-1], long_vecs[j-1])
            
            # Option 2: Skip
            # long[j] does not participate in pairing (treated as extra node, will be counted in unmatched_penalty at the end)
            # Try to make up the first i items of short using the first j-1 items of long
            skip_cost = dp[i][j-1]
            
            dp[i][j] = min(match_cost, skip_cost)
            
    # Final result = sum of best pairing distances + (number of unmatched nodes * 1.0)
    # Because no matter how paired, as long as N are selected, the remaining M-N must be unmatched, so this penalty is constant
    min_match_cost = dp[n][m]
    
    return min_match_cost + unmatched_penalty, m  # Total steps is the length of the long sequence

def path_similarity(gt_path: List[str], model_path: List[str], embedder: Embedder) -> float:
    # Remove the last paper node in the path
    gt_path_no_paper = gt_path[:-1] if len(gt_path) > 1 else gt_path
    model_path_no_paper = model_path[:-1] if len(model_path) > 1 else model_path
    
    # If path is empty after removing paper, return 0
    if not gt_path_no_paper or not model_path_no_paper:
        return 0.0
    
    gt_vecs = embed_path(gt_path_no_paper, embedder)
    model_vecs = embed_path(model_path_no_paper, embedder)
    cum_dist, step_cnt = calculate_min_alignment_cost(gt_vecs, model_vecs)
    if step_cnt == 0 or not np.isfinite(cum_dist):
        return 0.0
    return 1.0 / (1.0 + cum_dist)


# ----------------------------
# Tree Path Structure Similarity for Single Tree
# ----------------------------
def tree_path_similarity_from_alignment(
    gt_root: TreeNode,
    model_root: TreeNode,
    alignments: List[Dict],
    embedder: Embedder,
) -> Tuple[float, int, int]:
    map_gt = extract_paper_paths(gt_root)
    map_mod = extract_paper_paths(model_root)

    # Expert paper count changed to statistics based on gt_paper in alignment file (count after deduplication)
    gt_papers_from_alignment = {ali.get("gt_paper") for ali in alignments if ali.get("gt_paper")}
    gt_total_papers = len(gt_papers_from_alignment)
    total_sim = 0.0
    align_count = 0

    printed_first_pair = False
    for ali in alignments:
        gt_title = ali.get("gt_paper")
        model_title = ali.get("model_paper")
        if not gt_title or not model_title:
            continue
        gt_paths = map_gt.get(gt_title, [])
        model_paths = map_mod.get(model_title, [])
        if not gt_paths or not model_paths:
            continue
        best_sim = 0.0
        best_gp: Optional[List[str]] = None
        best_mp: Optional[List[str]] = None
        for gp in gt_paths:
            for mp in model_paths:
                sim = path_similarity(gp, mp, embedder)
                if sim > best_sim:
                    best_sim = sim
                    best_gp = gp
                    best_mp = mp
        total_sim += best_sim
        align_count += 1

        if not printed_first_pair and best_gp is not None and best_mp is not None:
            printed_first_pair = True

    # Changed to divide by "alignment count" instead of expert paper count; return 0 if alignment is 0
    # NOTE: Changed calculation of similarity for a single survey
    denom = align_count if align_count > 0 else 1
    alignment_sim = total_sim / denom
    expert_sim = total_sim / gt_total_papers
    # NOTE: Check return values
    return alignment_sim, expert_sim, align_count, gt_total_papers


# ----------------------------
# I/O and Batch Processing
# ----------------------------
def load_trees_from_jsonl(path: str) -> List[Tuple[TreeNode, TreeNode, int, str]]:
    items: List[Tuple[TreeNode, TreeNode, int, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            gt_tree_dict = data.get("gt")
            model_tree_dict = data.get("hierarchy_tree")
            if not gt_tree_dict or not model_tree_dict:
                continue
            survey_topic = data.get("survey_topic") or data.get("survey_topic_path") or ""
            gt_tree = build_tree(gt_tree_dict)
            model_tree = build_tree(model_tree_dict)
            survey_id = data.get("id") or data.get("idx") or line_num
            items.append((gt_tree, model_tree, survey_id, survey_topic))
    return items


def process_single_model(model_name: str, model_dir: str, embedder: Embedder) -> Dict[str, float]:
    """
    Process single model: Read paper_alignment_all.json to get alignment pairs,
    then use trees from merged.jsonl to build paths and calculate path similarity.
    """
    merged_file = os.path.join(model_dir, "merged.jsonl")
    align_file = os.path.join(model_dir, "paper_alignment_all.json")

    if not os.path.exists(merged_file):
        print(f"Warning: merged.jsonl file for {model_name} does not exist: {merged_file}")
        return {"avg_path_sim": 0.0, "count": 0, "total_align": 0, "total_gt_papers": 0}
    if not os.path.exists(align_file):
        print(f"Warning: paper_alignment_all.json file for {model_name} does not exist: {align_file}")
        return {"avg_path_sim": 0.0, "count": 0, "total_align": 0, "total_gt_papers": 0}

    items = load_trees_from_jsonl(merged_file)
    if not items:
        print(f"Warning: No valid merged data read for {model_name}.")
        return {"avg_path_sim": 0.0, "count": 0, "total_align": 0, "total_gt_papers": 0}

    # survey_topic -> (gt_tree, model_tree, survey_id)
    topic_map: Dict[str, Tuple[TreeNode, TreeNode, int]] = {}
    for gt_tree, model_tree, survey_id, survey_topic in items:
        topic_map[survey_topic] = (gt_tree, model_tree, survey_id)

    with open(align_file, "r", encoding="utf-8") as f:
        try:
            align_data = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to read {align_file}: {e}")
            return {"avg_path_sim": 0.0, "count": 0, "total_align": 0, "total_gt_papers": 0}

    alignment_sim_list = []
    expert_sim_list = []
    total_align = 0
    total_gt_papers = 0
    survey_count = 0

    for entry in align_data:
        survey_topic = entry.get("survey_topic", "")
        alignments = entry.get("alignments", [])
        if survey_topic not in topic_map:
            print(f"  Skipped: survey_topic in alignment file not found in merged: {survey_topic}")
            continue
        gt_tree, model_tree, survey_id = topic_map[survey_topic]

        alignment_sim, expert_sim, align_cnt, gt_cnt = tree_path_similarity_from_alignment(
            gt_tree, model_tree, alignments, embedder
        )

        alignment_sim_list.append(alignment_sim)
        expert_sim_list.append(expert_sim)
        total_align += align_cnt
        total_gt_papers += gt_cnt
        survey_count += 1
        #print(f"  Survey {survey_id} | topic={survey_topic} | path similarity = {sim:.4f} | align_count={align_cnt} | gt_count={gt_cnt}")

    avg_alignment_sim = (sum(alignment_sim_list) / len(alignment_sim_list)) if alignment_sim_list else 0.0
    avg_expert_sim = (sum(expert_sim_list) / len(expert_sim_list)) if expert_sim_list else 0.0
    return {
        "avg_alignment_sim": avg_alignment_sim,
        "avg_expert_sim": avg_expert_sim,
        "count": survey_count,
        "total_align": total_align,
        "total_gt_papers": total_gt_papers,
    }


def main():
    parser = argparse.ArgumentParser(description="Tree structure similarity based on path semantics")
    parser.add_argument(
        "--model_dir",
        default="/root/new_metric/model/deep-research",
        help="Root directory path containing model folders",
    )
    parser.add_argument(
        "--model",
        default="/root/new_metric/code/models--sentence-transformers--all-MiniLM-L6-v2",
        help="SentenceTransformer model name (used for similarity calculation)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        raise FileNotFoundError(f"Model directory does not exist: {args.model_dir}")
    
    model_names =["test"]
    embedder = Embedder(model_name=args.model)

    results: Dict[str, Dict[str, float]] = {}

    print("=" * 80)
    print(f"Start batch processing {len(model_names)} models")
    print("=" * 80)

    for model_name in model_names:
        model_path = os.path.join(args.model_dir, model_name)
        print(f"\nProcessing model: {model_name}")
        print("-" * 80)

        result = process_single_model(model_name, model_path, embedder)
        results[model_name] = result
        print("Result:", results[model_name])
        # Recalled papers / Number of recalled papers
        # Recalled papers / Total number of gt papers
        print(
            f"Model {model_name}: Total {result['count']} items, Average Recall Similarity: {result['avg_alignment_sim']:.4f} ",
            f"Model {model_name}: Total {result['count']} items, Average Expert Similarity: {result['avg_expert_sim']:.4f}"
        )

    print("\n" + "=" * 80)
    print("Summary Results")
    print("=" * 80)
    for model_name in model_names:
        if model_name in results:
            result = results[model_name]
            print(
                f"{model_name:20s}: Avg Recall Sim: {result['avg_alignment_sim']:.4f} "
                f"{model_name:20s}: Avg Expert Sim: {result['avg_expert_sim']:.4f} "
                f"(Total {result['count']} items)"
            )
    print("=" * 80)


if __name__ == "__main__":
    main()