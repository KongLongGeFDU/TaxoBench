from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple

Tree = Dict[str, Any]


def extract_papers(tree: Tree) -> List[str]:
    papers: List[str] = []

    def visit(node: Tree) -> None:
        if not isinstance(node, dict):
            return
        if "papers" in node:
            papers.extend(str(p) for p in node.get("papers", []) if p)
        for child in node.get("subtopics", []) or []:
            visit(child)

    visit(tree)
    return papers


def extract_clusters(tree: Tree) -> List[List[str]]:
    clusters: List[List[str]] = []

    def visit(node: Tree) -> None:
        if not isinstance(node, dict):
            return
        if "papers" in node:
            cluster = [str(p) for p in node.get("papers", []) if p]
            if cluster:
                clusters.append(cluster)
        for child in node.get("subtopics", []) or []:
            visit(child)

    visit(tree)
    return clusters


def iter_leaf_paths(tree: Tree) -> Dict[str, List[List[str]]]:
    """Map each paper title to one or more root-to-paper paths."""
    paths: Dict[str, List[List[str]]] = defaultdict(list)

    def visit(node: Tree, prefix: List[str]) -> None:
        if not isinstance(node, dict):
            return
        name = str(node.get("name", "")).strip()
        current = prefix + ([name] if name else [])
        for paper in node.get("papers", []) or []:
            paper_title = str(paper).strip()
            if paper_title:
                paths[paper_title].append(current + [paper_title])
        for child in node.get("subtopics", []) or []:
            visit(child, current)

    visit(tree, [])
    return dict(paths)
