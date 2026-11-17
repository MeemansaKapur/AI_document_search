"""Batched OpenAI embeddings with exponential backoff + jitter."""
from __future__ import annotations

import logging
import random
import time

import openai
from openai import OpenAI

log = logging.getLogger(__name__)

RETRYABLE = (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError,
             openai.InternalServerError)


class Embedder:
    def __init__(self, client: OpenAI, model: str, batch_size: int = 100, max_retries: int = 6):
        self.client = client
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.tokens_used = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        if any(not t.strip() for t in texts):
            raise ValueError("cannot embed empty text")
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            out.extend(self._embed_batch(texts[i:i + self.batch_size]))
        return out

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.client.embeddings.create(model=self.model, input=batch)
                self.tokens_used += resp.usage.total_tokens
                return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]
            except RETRYABLE as e:
                if attempt == self.max_retries:
                    raise
                delay = 2 ** attempt + random.random()  # jitter avoids synchronized retries
                log.warning("embedding call failed (%s); retrying in %.1fs", type(e).__name__, delay)
                time.sleep(delay)
        raise AssertionError("unreachable")
