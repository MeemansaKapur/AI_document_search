"""Aggregation pipelines for $vectorSearch (Atlas Vector Search) and $search (Atlas Search)."""
from __future__ import annotations

# Inclusion projection: never ship the 1,536-float embedding back to the app.
RESULT_FIELDS = ["doc_id", "filename", "title", "course", "doc_type", "page_start", "page_end",
                 "section", "chunk_index", "text"]


def _project(meta: str) -> dict:
    return {"$project": {**{f: 1 for f in RESULT_FIELDS}, "score": {"$meta": meta}}}


def vector_pipeline(qvec: list[float], k: int, *, index: str = "vector_index",
                    num_candidates_multiplier: int = 15, course: str | None = None) -> list[dict]:
    stage = {"index": index, "path": "embedding", "queryVector": qvec,
             "numCandidates": max(k, k * num_candidates_multiplier), "limit": k}
    if course:
        stage["filter"] = {"course": course}  # pre-filter inside the HNSW search
    return [{"$vectorSearch": stage}, _project("vectorSearchScore")]


def text_pipeline(q: str, k: int, *, index: str = "text_index", course: str | None = None) -> list[dict]:
    text = {"text": {"query": q, "path": ["text", "title", "section"]}}
    if course:
        search = {"index": index, "compound": {
            "must": [text], "filter": [{"equals": {"path": "course", "value": course}}]}}
    else:
        search = {"index": index, **text}
    return [{"$search": search}, {"$limit": k}, _project("searchScore")]
