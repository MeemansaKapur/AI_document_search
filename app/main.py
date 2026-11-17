"""FastAPI app: /health, /search, /ask, /courses, and the static UI."""
from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import openai
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI
from pymongo import AsyncMongoClient

from app.config import get_settings
from app.generate import generate_answer, pages_label
from app.models import AskRequest, AskResponse, Passage, SearchRequest, SearchResponse
from app.retrieval.cache import QueryEmbedder
from app.retrieval.search import Mode, Retriever

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
log = logging.getLogger("api")
logging.basicConfig(level=logging.INFO, format="%(message)s")


class IndexStatus:
    """Cached check that both search indexes are queryable (they build asynchronously)."""

    def __init__(self, chunks, settings, ttl_s: float = 15):
        self.chunks, self.s, self.ttl = chunks, settings, ttl_s
        self._at, self._status = 0.0, {}

    async def get(self) -> dict[str, str]:
        if time.monotonic() - self._at > self.ttl or not self.ready_from(self._status):
            cursor = await self.chunks.list_search_indexes()
            found = {ix["name"]: ix for ix in await cursor.to_list(None)}
            self._status = {
                name: "missing" if name not in found else ("ready" if found[name].get("queryable") else "building")
                for name in (self.s.vector_index, self.s.text_index)}
            self._at = time.monotonic()
        return self._status

    @staticmethod
    def ready_from(status: dict) -> bool:
        return bool(status) and all(v == "ready" for v in status.values())


class RateLimiter:
    """Per-IP token bucket (per worker process). /ask costs an LLM call; /search does not."""

    def __init__(self, per_minute: int):
        self.rate, self.capacity = per_minute / 60.0, float(per_minute)
        self.buckets: dict[str, tuple[float, float]] = {}

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        tokens, last = self.buckets.get(key, (self.capacity, now))
        tokens = min(self.capacity, tokens + (now - last) * self.rate)
        if len(self.buckets) > 10_000:
            self.buckets.clear()
        if tokens < 1:
            self.buckets[key] = (tokens, now)
            return False
        self.buckets[key] = (tokens - 1, now)
        return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    # Clients are created once: they hold connection pools (no TLS handshake per request).
    mongo = AsyncMongoClient(s.mongodb_uri, appname="atlas-doc-search-api", serverSelectionTimeoutMS=5000)
    oai = AsyncOpenAI(api_key=s.openai_api_key or None, max_retries=2, timeout=30)
    db = mongo[s.mongodb_db]
    app.state.settings, app.state.db, app.state.oai = s, db, oai
    app.state.retriever = Retriever(db.chunks, QueryEmbedder(oai, s.embedding_model, s.query_cache_size), s)
    app.state.index_status = IndexStatus(db.chunks, s)
    app.state.limiter = RateLimiter(s.ask_rate_limit_per_min)
    yield
    await mongo.close()
    await oai.close()


app = FastAPI(title="Atlas Doc Search", version="1.0", lifespan=lifespan)


@app.exception_handler(openai.OpenAIError)
async def openai_unavailable(request: Request, exc: openai.OpenAIError):
    # Query embedding failed (outage, rate limit, bad key). Keyword mode needs no OpenAI call.
    log.error(json.dumps({"rid": getattr(request.state, "rid", None), "openai_error": type(exc).__name__}))
    return JSONResponse(status_code=503, content={"detail": {
        "error": f"embedding service unavailable ({type(exc).__name__})", "hint": "mode=keyword still works"}})


@app.middleware("http")
async def access_log(request: Request, call_next):
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    request.state.rid, request.state.log = rid, None
    t0 = time.perf_counter()
    response = await call_next(request)
    entry = {"rid": rid, "method": request.method, "path": request.url.path, "status": response.status_code,
             "ms": round((time.perf_counter() - t0) * 1000, 1), **(request.state.log or {})}
    log.info(json.dumps(entry))
    response.headers["x-request-id"] = rid
    return response


def client_ip(request: Request) -> str:
    if request.app.state.settings.trust_forwarded_for:
        fwd = request.headers.get("x-forwarded-for")
        if fwd:
            return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


async def require_indexes(request: Request) -> None:
    status = await request.app.state.index_status.get()
    if not IndexStatus.ready_from(status):
        raise HTTPException(503, detail={"error": "search indexes not ready", "indexes": status})


def to_passage(r: dict, n: int | None = None) -> Passage:
    return Passage(
        id=r["_id"], n=n, title=r["title"], filename=r["filename"], course=r["course"], doc_type=r["doc_type"],
        pages=pages_label(r["page_start"], r["page_end"]), page_start=r["page_start"], page_end=r["page_end"],
        section=r.get("section"), snippet=r["text"][:500], score=round(float(r["score"]), 5),
        ranks=r.get("ranks", {}))


@app.get("/health")
async def health(request: Request):
    db_ok = "ok"
    try:
        await request.app.state.db.command("ping")
        indexes = await request.app.state.index_status.get()
    except Exception as e:  # noqa: BLE001
        db_ok, indexes = f"error: {type(e).__name__}", {}
    ready = db_ok == "ok" and IndexStatus.ready_from(indexes)
    return {"status": "ok" if ready else "degraded", "db": db_ok,
            "index": "ready" if ready else "building", "indexes": indexes}


@app.get("/courses")
async def courses(request: Request) -> list[str]:
    return sorted(c for c in await request.app.state.db.chunks.distinct("course") if c)


@app.post("/search", response_model=SearchResponse)
async def search(body: SearchRequest, request: Request):
    await require_indexes(request)
    out = await request.app.state.retriever.search(body.q, body.mode, body.k, body.course)
    request.state.log = {"mode": body.mode.value, "results": len(out.results), "timings": out.timings_ms,
                         "cache_hit": out.cache_hit, **({"q": body.q} if request.app.state.settings.log_questions else {})}
    return SearchResponse(query=body.q, mode=body.mode, results=[to_passage(r) for r in out.results],
                          timings_ms=out.timings_ms, cache_hit=out.cache_hit)


@app.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest, request: Request):
    s = request.app.state.settings
    if not request.app.state.limiter.allow(client_ip(request)):
        raise HTTPException(429, detail=f"rate limit: {s.ask_rate_limit_per_min} questions per minute")
    await require_indexes(request)

    t0 = time.perf_counter()
    out = await request.app.state.retriever.search(body.q, Mode.hybrid, s.ask_top_k, body.course)
    timings = dict(out.timings_ms)
    passages = [to_passage(r, i) for i, r in enumerate(out.results, 1)]
    log_extra = {"top_vector_score": out.top_vector_score, "results": len(passages)}

    # Confidence gate: don't let the LLM stitch weak passages into a confident wrong answer.
    if not passages or out.top_vector_score is None or out.top_vector_score < s.abstain_threshold:
        timings["total"] = round((time.perf_counter() - t0) * 1000, 1)
        request.state.log = {**log_extra, "abstained": True, "timings": timings}
        return AskResponse(query=body.q, answer="I couldn't find this in the course documents.", abstained=True,
                           abstain_reason="low_retrieval_confidence", citations=passages,
                           top_vector_score=out.top_vector_score, timings_ms=timings)

    t1 = time.perf_counter()
    try:
        gen = await generate_answer(request.app.state.oai, s.chat_model, body.q, out.results,
                                    max_tokens=s.answer_max_tokens, temperature=s.answer_temperature)
    except openai.OpenAIError as e:
        # Graceful degradation: retrieval still works when the LLM is unavailable.
        raise HTTPException(503, detail={"error": f"answer generation unavailable ({type(e).__name__})",
                                         "passages": [p.model_dump() for p in passages]}) from e
    timings["llm"] = round((time.perf_counter() - t1) * 1000, 1)
    timings["total"] = round((time.perf_counter() - t0) * 1000, 1)
    request.state.log = {**log_extra, "abstained": gen["refused"], "timings": timings,
                         "invalid_citations": gen["invalid_citations"]}
    return AskResponse(query=body.q, answer=gen["answer"], abstained=gen["refused"],
                       abstain_reason="model_refused" if gen["refused"] else None, citations=passages,
                       cited=gen["cited"], invalid_citations=gen["invalid_citations"],
                       top_vector_score=out.top_vector_score, timings_ms=timings)


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
