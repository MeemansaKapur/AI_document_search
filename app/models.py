"""Request / response schemas."""
from __future__ import annotations

from pydantic import BaseModel, Field

from app.retrieval.search import Mode


class SearchRequest(BaseModel):
    q: str = Field(min_length=3, max_length=500)
    mode: Mode = Mode.hybrid
    k: int = Field(5, ge=1, le=20)
    course: str | None = None


class AskRequest(BaseModel):
    q: str = Field(min_length=3, max_length=500)
    course: str | None = None


class Passage(BaseModel):
    id: str
    n: int | None = None  # citation number in /ask
    title: str
    filename: str
    course: str
    doc_type: str
    pages: str
    page_start: int
    page_end: int
    section: str | None = None
    snippet: str
    score: float
    ranks: dict[str, int] = {}


class SearchResponse(BaseModel):
    query: str
    mode: Mode
    results: list[Passage]
    timings_ms: dict[str, float]
    cache_hit: bool | None = None


class AskResponse(BaseModel):
    query: str
    answer: str
    abstained: bool
    abstain_reason: str | None = None
    citations: list[Passage]
    cited: list[int] = []
    invalid_citations: list[int] = []
    top_vector_score: float | None = None
    timings_ms: dict[str, float]
