"""Retrieval metrics with page-level gold labels (valid across every chunking configuration)."""
from __future__ import annotations


def is_hit(result: dict, gold: list[dict]) -> bool:
    """Hit = chunk from a gold file whose page_start..page_end overlaps any gold page."""
    for g in gold:
        if result["filename"] == g["filename"] and any(
                result["page_start"] <= p <= result["page_end"] for p in g["pages"]):
            return True
    return False


def first_hit_rank(results: list[dict], gold: list[dict]) -> int | None:
    for rank, r in enumerate(results, 1):
        if is_hit(r, gold):
            return rank
    return None


def hit_at_k(ranks: list[int | None], k: int = 3) -> float:
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks) if ranks else 0.0


def mrr_at_k(ranks: list[int | None], k: int = 10) -> float:
    return sum(1 / r for r in ranks if r is not None and r <= k) / len(ranks) if ranks else 0.0


def best_threshold(answerable: list[float], unanswerable: list[float]) -> tuple[float, float]:
    """Threshold on top-1 vector score that best separates the two groups. Returns (threshold, accuracy)."""
    scores = sorted(set(answerable + unanswerable))
    if not scores:
        return 0.0, 0.0
    candidates = [scores[0] - 1e-6] + [(a + b) / 2 for a, b in zip(scores, scores[1:])] + [scores[-1] + 1e-6]
    total = len(answerable) + len(unanswerable)

    def acc(t: float) -> float:
        return (sum(s >= t for s in answerable) + sum(s < t for s in unanswerable)) / total

    best = max(candidates, key=acc)
    return round(best, 4), acc(best)
