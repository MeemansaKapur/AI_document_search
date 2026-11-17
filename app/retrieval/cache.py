"""Query embedding with an in-memory LRU cache (the embedding API call is often the slowest step)."""
from __future__ import annotations

from collections import OrderedDict

from openai import AsyncOpenAI


def normalize(q: str) -> str:
    return " ".join(q.lower().split())


class QueryEmbedder:
    def __init__(self, client: AsyncOpenAI, model: str, maxsize: int = 1024):
        self.client = client
        self.model = model
        self.maxsize = maxsize
        self._cache: OrderedDict[str, list[float]] = OrderedDict()

    async def embed(self, q: str) -> tuple[list[float], bool]:
        """Returns (vector, cache_hit)."""
        key = normalize(q)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key], True
        resp = await self.client.embeddings.create(model=self.model, input=[key])
        vec = resp.data[0].embedding
        self._cache[key] = vec
        if len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)
        return vec, False
