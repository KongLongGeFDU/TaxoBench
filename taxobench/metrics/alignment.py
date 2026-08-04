from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def normalize_title(title: str) -> str:
    title = title.lower()
    title = re.sub(r"[^\w\s]", " ", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def title_similarity(a: str, b: str) -> float:
    na, nb = normalize_title(a), normalize_title(b)
    if not na or not nb:
        return 0.0
    if na == nb or na in nb or nb in na:
        return 1.0
    return SequenceMatcher(None, na, nb).ratio()


def align_titles(reference_titles: Sequence[str], prediction_titles: Sequence[str], threshold: float = 0.92) -> List[Dict[str, object]]:
    """Greedy one-to-one title alignment used by the public scorer.

    The paper reports a stricter internal alignment audit; this public utility is
    intended for reproducible scoring of user predictions without releasing raw
    model logs.
    """
    used: set[int] = set()
    rows: List[Dict[str, object]] = []
    for ref_idx, ref in enumerate(reference_titles):
        best_idx: Optional[int] = None
        best_score = 0.0
        for pred_idx, pred in enumerate(prediction_titles):
            if pred_idx in used:
                continue
            score = title_similarity(ref, pred)
            if score > best_score:
                best_score = score
                best_idx = pred_idx
        if best_idx is not None and best_score >= threshold:
            used.add(best_idx)
            rows.append({"reference_title": ref, "prediction_title": prediction_titles[best_idx], "reference_index": ref_idx, "prediction_index": best_idx, "score": best_score})
        else:
            rows.append({"reference_title": ref, "prediction_title": None, "reference_index": ref_idx, "prediction_index": None, "score": best_score})
    return rows
