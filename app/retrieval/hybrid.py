"""Reciprocal Rank Fusion: score(d) = sum_i  w_i / (k + rank_i(d))."""
from __future__ import annotations


def rrf_fuse(ranked_lists: dict[str, list[dict]], weights: dict[str, float], k: int = 60,
             limit: int | None = None) -> list[dict]:
    """Merge ranked result lists by rank (not raw score: BM25 and cosine scales are incomparable).

    Each input doc must have an "_id". Returns copies with "rrf" and "ranks" ({list name: rank}).
    Ties keep first-seen order (stable sort), so results are deterministic.
    """
    fused: dict = {}
    for name, results in ranked_lists.items():
        w = weights.get(name, 1.0)
        for rank, doc in enumerate(results, start=1):
            entry = fused.get(doc["_id"])
            if entry is None:
                entry = fused[doc["_id"]] = {**doc, "rrf": 0.0, "ranks": {}, "scores": {}}
            entry["rrf"] += w / (k + rank)
            entry["ranks"][name] = rank
            if "score" in doc:
                entry["scores"][name] = doc["score"]
    ranked = sorted(fused.values(), key=lambda e: e["rrf"], reverse=True)
    for e in ranked:
        e.pop("score", None)
        e["score"] = e["rrf"]
    return ranked[:limit] if limit else ranked
