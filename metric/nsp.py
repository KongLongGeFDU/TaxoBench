import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

# Accelerate mirror and cache (keep environment settings consistent with ted.py)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))


# ----------------------------
# Tree and Node Collection
# ----------------------------
class TreeNode:
    def __init__(self, name: str, children: List["TreeNode"] | None = None):
        self.name = name
        self.children = children or []


def build_tree(node_dict: Dict) -> TreeNode:
    """Only build topic level (ignore papers list)."""
    name = node_dict.get("name", "")
    children_data = node_dict.get("subtopics", [])
    children = [build_tree(child) for child in children_data]
    return TreeNode(name=name, children=children)


def collect_node_names(root: TreeNode) -> List[str]:
    """Pre-order collect all node names (including root), excluding papers."""
    names = [root.name]
    for child in root.children:
        names.extend(collect_node_names(child))
    return names


# ----------------------------
# Soft Cardinality and NSP
# ----------------------------
def soft_cardinality(names: List[str], model: SentenceTransformer) -> float:
    """
    Compute soft cardinality of list using SentenceTransformer (no longer normalized).
    c = sum_i 1 / sum_j sim(i, j); if row sum is 0, set row weight to 0.
    """
    if not names:
        return 0.0
    embeddings = model.encode(names, normalize_embeddings=True)
    emb = np.array(embeddings)
    sim = emb @ emb.T  # Cosine similarity matrix
    row_sums = sim.sum(axis=1)
    weights = np.zeros_like(row_sums)
    nonzero = row_sums != 0
    weights[nonzero] = 1.0 / row_sums[nonzero]
    c=float(weights.sum())
    #c = float(weights.sum() / len(names))
    return c


def NSP(gt_names: List[str], mt_names: List[str], model: SentenceTransformer) -> float:
    c_gt = soft_cardinality(gt_names, model)
    c_mt = soft_cardinality(mt_names, model)
    c_union = soft_cardinality(gt_names + mt_names, model)
    # Traditional NSP: denominator uses c_gt
    if c_gt == 0:
        return 0.0
    return (c_gt + c_mt - c_union) / c_mt


# ----------------------------
# Data Reading
# ----------------------------
def load_pairs(path: str) -> List[Tuple[List[str], List[str], int]]:
    pairs: List[Tuple[List[str], List[str], int]] = []
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
            gt_tree = build_tree(gt_tree_dict)
            mt_tree = build_tree(model_tree_dict)
            gt_names = collect_node_names(gt_tree)
            mt_names = collect_node_names(mt_tree)
            survey_id = data.get("id", line_num)
            pairs.append((gt_names, mt_names, survey_id))
    return pairs


# ----------------------------
# Main Process
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Compute Node Soft Recall (NSP)")
    parser.add_argument(
        "--input",
        default="model/output/model_name/merged.jsonl",
        help="Path to jsonl file containing gt and hierarchy_tree",
    )
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file does not exist: {args.input}")

    model = SentenceTransformer(args.model)
    pairs = load_pairs(args.input)
    if not pairs:
        return

    NSP_list: List[float] = []
    for gt_names, mt_names, survey_id in pairs:
        val = NSP(gt_names, mt_names, model)
        NSP_list.append(val)

    avg = sum(NSP_list) / len(NSP_list)
    print(f"Total {len(NSP_list)} records, average NSP: {avg:.4f}")


if __name__ == "__main__":
    main()

