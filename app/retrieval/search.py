"""Keyword, vector and hybrid retrieval over the chunks collection, with per-stage timings."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum

from app.config import Settings
from app.retrieval.cache import QueryEmbedder
from app.retrieval.hybrid import rrf_fuse
from app.retrieval.pipelines import text_pipeline, vector_pipeline


class Mode(str, Enum):
    keyword = "keyword"
    vector = "vector"
    hybrid = "hybrid"


@dataclass
class SearchOutcome:
    results: list[dict]
    timings_ms: dict[str, float] = field(default_factory=dict)
    top_vector_score: float | None = None  # used by the /ask confidence gate
    cache_hit: bool | None = None


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)


class Retriever:
    def __init__(self, chunks, embedder: QueryEmbedder, settings: Settings):
        self.chunks = chunks
        self.embedder = embedder
        self.s = settings

    async def _aggregate(self, pipeline: list[dict]) -> list[dict]:
        cursor = await self.chunks.aggregate(pipeline)
        return await cursor.to_list(None)

    async def _vector(self, q: str, k: int, course: str | None, ncm: int, timings: dict):
        t0 = time.perf_counter()
        qvec, hit = await self.embedder.embed(q)
        timings["embed"] = _ms(t0)
        t1 = time.perf_counter()
        res = await self._aggregate(vector_pipeline(
            qvec, k, index=self.s.vector_index, num_candidates_multiplier=ncm, course=course))
        timings["vector"] = _ms(t1)
        return res, hit

    async def _text(self, q: str, k: int, course: str | None, timings: dict):
        t0 = time.perf_counter()
        res = await self._aggregate(text_pipeline(q, k, index=self.s.text_index, course=course))
        timings["text"] = _ms(t0)
        return res

    async def search(self, q: str, mode: Mode | str = Mode.hybrid, k: int = 5, course: str | None = None,
                     *, num_candidates_multiplier: int | None = None, w_vector: float | None = None,
                     w_text: float | None = None, rrf_k: int | None = None,
                     concurrent: bool = True) -> SearchOutcome:
        mode = Mode(mode)
        ncm = num_candidates_multiplier or self.s.num_candidates_multiplier
        timings: dict[str, float] = {}
        t0 = time.perf_counter()

        if mode is Mode.keyword:
            results = self._rank(await self._text(q, k, course, timings), "keyword")
            out = SearchOutcome(results, timings)
        elif mode is Mode.vector:
            res, hit = await self._vector(q, k, course, ncm, timings)
            out = SearchOutcome(self._rank(res, "vector"), timings,
                                res[0]["score"] if res else None, hit)
        else:
            fetch = k * 2  # a doc ranked 12th by one method may belong in the fused top k
            if concurrent:  # text search runs while the query is being embedded + vector-searched
                (vec, hit), txt = await asyncio.gather(
                    self._vector(q, fetch, course, ncm, timings), self._text(q, fetch, course, timings))
            else:
                vec, hit = await self._vector(q, fetch, course, ncm, timings)
                txt = await self._text(q, fetch, course, timings)
            t1 = time.perf_counter()
            results = rrf_fuse({"vector": vec, "keyword": txt},
                               {"vector": w_vector if w_vector is not None else self.s.w_vector,
                                "keyword": w_text if w_text is not None else self.s.w_text},
                               k=rrf_k or self.s.rrf_k, limit=k)
            timings["fuse"] = _ms(t1)
            out = SearchOutcome(results, timings, vec[0]["score"] if vec else None, hit)

        timings["total"] = _ms(t0)
        return out

    @staticmethod
    def _rank(results: list[dict], name: str) -> list[dict]:
        for i, r in enumerate(results, 1):
            r["ranks"] = {name: i}
        return results
