import argparse
import json
import os
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None  # type: ignore

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# Set HuggingFace cache directory so SentenceTransformer can recognize downloaded models
os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")
# ----------------------------
# Data Structures
# ----------------------------
class TreeNode:
    def __init__(self, name: str, children: Optional[List["TreeNode"]] = None):
        self.name = name
        self.children: List["TreeNode"] = children or []


class Embedder:
    """
    Only use SentenceTransformer.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        if SentenceTransformer is None:
            raise ImportError("Please install sentence-transformers library first")
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
        import numpy as np

        va_arr = np.array(va)
        vb_arr = np.array(vb)
        denom = (np.linalg.norm(va_arr) * np.linalg.norm(vb_arr))
        if denom == 0:
            return 0.0
        return float(np.dot(va_arr, vb_arr) / denom)


# ----------------------------
# Tree Construction
# ----------------------------
def build_tree(node_dict: Dict) -> TreeNode:
    """
    Build TreeNode from JSON tree node.
    Only keep structure: name and subtopics. papers is only used to determine if it's a leaf, not to generate child nodes.
    """
    name = node_dict.get("name", "")
    children_data = node_dict.get("subtopics", [])
    children = [build_tree(child) for child in children_data]
    return TreeNode(name=name, children=children)


# ----------------------------
# Zhang-Shasha Ordered Tree Edit Distance
# ----------------------------
def _post_order(node: TreeNode, nodes: List[TreeNode], l: List[int]) -> int:
    """
    Post-order traversal, returns the position of the leftmost leaf node of current subtree (index in nodes list).
    """
    if not node.children:
        idx = len(nodes)
        nodes.append(node)
        l.append(idx)
        return idx

    first_child_left = None
    for child in node.children:
        child_left = _post_order(child, nodes, l)
        if first_child_left is None:
            first_child_left = child_left
    idx = len(nodes)
    nodes.append(node)
    l.append(first_child_left if first_child_left is not None else idx)
    return first_child_left if first_child_left is not None else idx


def _compute_keyroots(l: List[int]) -> List[int]:
    """
    Returns the "last occurrence" position of each leftmost leaf index, ensuring root node is included.
    Zhang-Shasha requires keyroots to select the rightmost occurrence node, otherwise treedist[m-1][n-1]
    may remain inf (root not processed).
    """
    keyroots: List[int] = []
    seen = set()
    # Reverse order first occurrence = forward order last occurrence
    for i in range(len(l) - 1, -1, -1):
        left = l[i]
        if left not in seen:
            keyroots.append(i)
            seen.add(left)
    keyroots.reverse()
    return keyroots


def tree_edit_distance(
    t1: TreeNode,
    t2: TreeNode,
    embedder: Embedder,
    insert_cost: float = 1.0,
    delete_cost: float = 1.0,
) -> float:
    """
    Zhang-Shasha ordered tree edit distance, rename cost = 1 - cos_sim.
    """
    nodes1: List[TreeNode] = []
    l1: List[int] = []
    _post_order(t1, nodes1, l1)

    nodes2: List[TreeNode] = []
    l2: List[int] = []
    _post_order(t2, nodes2, l2)

    keyroots1 = _compute_keyroots(l1)
    keyroots2 = _compute_keyroots(l2)

    # Cache for treedist
    m = len(nodes1)
    n = len(nodes2)
    # Initialize to infinity to avoid unassigned 0 being treated as valid optimal distance
    treedist = [[float("inf") for _ in range(n)] for _ in range(m)]

    def rename_cost(i: int, j: int) -> float:
        cos = embedder.cosine(nodes1[i].name, nodes2[j].name)
        return 1.0 - cos

    for i_key in keyroots1:
        for j_key in keyroots2:
            _forest_dist(i_key, j_key, nodes1, nodes2, l1, l2, treedist, insert_cost, delete_cost, rename_cost)

    return treedist[m - 1][n - 1]


def _forest_dist(
    i_key: int,
    j_key: int,
    nodes1: List[TreeNode],
    nodes2: List[TreeNode],
    l1: List[int],
    l2: List[int],
    treedist: List[List[float]],
    insert_cost: float,
    delete_cost: float,
    rename_cost_fn,
):
    """
    Compute subtree distance (Zhang-Shasha).
    """
    i_start = l1[i_key]
    j_start = l2[j_key]

    fd = [[0.0 for _ in range(j_key - j_start + 2)] for _ in range(i_key - i_start + 2)]

    fd[0][0] = 0
    for i in range(1, i_key - i_start + 2):
        fd[i][0] = fd[i - 1][0] + delete_cost
    for j in range(1, j_key - j_start + 2):
        fd[0][j] = fd[0][j - 1] + insert_cost

    for i in range(1, i_key - i_start + 2):
        for j in range(1, j_key - j_start + 2):
            i_idx = i_start + i - 1
            j_idx = j_start + j - 1

            if l1[i_idx] == i_start and l2[j_idx] == j_start:
                cost_ren = rename_cost_fn(i_idx, j_idx)
                fd[i][j] = min(
                    fd[i - 1][j] + delete_cost,
                    fd[i][j - 1] + insert_cost,
                    fd[i - 1][j - 1] + cost_ren,
                )
                treedist[i_idx][j_idx] = fd[i][j]
            else:
                # If treedist has not been updated, fallback to current rename cost
                subtree_cost = (
                    treedist[i_idx][j_idx]
                    if treedist[i_idx][j_idx] != float("inf")
                    else rename_cost_fn(i_idx, j_idx)
                )
                fd[i][j] = min(
                    fd[i - 1][j] + delete_cost,
                    fd[i][j - 1] + insert_cost,
                    fd[i - 1][j - 1] + subtree_cost,
                )


# ----------------------------
# Normalized TED
# ----------------------------
def normalized_ted(t1: TreeNode, t2: TreeNode, embedder: Embedder) -> float:
    ted = tree_edit_distance(t1, t2, embedder)
    size = max(count_nodes(t1) + count_nodes(t2), 1)
    return ted / size


def count_nodes(root: TreeNode) -> int:
    return 1 + sum(count_nodes(c) for c in root.children)


# ----------------------------
# Tree Printing (for debugging)
# ----------------------------
def _format_tree_lines(node: TreeNode, depth: int = 0) -> List[str]:
    prefix = "  " * depth + "- "
    lines = [f"{prefix}{node.name}"]
    for child in node.children:
        lines.extend(_format_tree_lines(child, depth + 1))
    return lines


def print_tree(node: TreeNode, title: str) -> None:
    print(title)
    for line in _format_tree_lines(node):
        print(line)
    print("-" * 40)


# ----------------------------
# I/O and Main Process
# ----------------------------
def load_trees_from_jsonl(path: str) -> List[Tuple[TreeNode, TreeNode, int]]:
    items: List[Tuple[TreeNode, TreeNode, int]] = []
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
            model_tree = build_tree(model_tree_dict)
            survey_id = data.get("idx", line_num)
            items.append((gt_tree, model_tree, survey_id))
    return items


def main():
    parser = argparse.ArgumentParser(description="Compute improved tree edit distance (with semantic rename cost)")
    parser.add_argument(
        "--input",
        default="model/output/model_name/merged.jsonl",
        help="Path to jsonl file containing gt and hierarchy_tree",
    )
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name; if unavailable will automatically use character 3-gram fallback",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file does not exist: {args.input}")

    embedder = Embedder(model_name=args.model)
    items = load_trees_from_jsonl(args.input)
    if not items:
        return

    ted_list = []
    for gt_tree, model_tree, survey_id in items:
        n_ted = normalized_ted(gt_tree, model_tree, embedder)
        ted_list.append(n_ted)

    avg_ted = sum(ted_list) / len(ted_list)
    print(f"Total {len(ted_list)} records, average normalized TED: {avg_ted:.4f}")


if __name__ == "__main__":
    main()

