"""API tests with fake retriever / LLM (no network, no database)."""
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import main
from app.config import Settings
from app.retrieval.search import SearchOutcome

RESULT = {"_id": "d:00001", "title": "OS", "filename": "os.pdf", "course": "OS", "doc_type": "slides",
          "page_start": 14, "page_end": 15, "section": None, "text": "LRU evicts...", "score": 0.03,
          "ranks": {"vector": 1, "keyword": 2}}


class FakeRetriever:
    def __init__(self, top):
        self.top = top

    async def search(self, q, mode="hybrid", k=5, course=None, **kw):
        return SearchOutcome([dict(RESULT)], {"embed": 1.0, "total": 2.0}, self.top, False)


class ReadyIndexes:
    async def get(self):
        return {"vector_index": "ready", "text_index": "ready"}


@pytest.fixture
def client(monkeypatch):
    async def fake_generate(*a, **kw):
        return {"answer": "LRU [1].", "cited": [1], "invalid_citations": [], "refused": False}

    monkeypatch.setattr(main, "generate_answer", fake_generate)
    app = main.FastAPI()  # bare app reusing the routes, without the real lifespan
    app.router.routes.extend(main.app.router.routes)
    app.middleware("http")(main.access_log)
    app.add_exception_handler(main.openai.OpenAIError, main.openai_unavailable)
    s = Settings(abstain_threshold=0.7, ask_rate_limit_per_min=2)
    app.state.settings, app.state.oai = s, SimpleNamespace()
    app.state.index_status = ReadyIndexes()
    app.state.limiter = main.RateLimiter(2)
    app.state.retriever = FakeRetriever(top=0.9)
    return TestClient(app)


def test_search_validation(client):
    assert client.post("/search", json={"q": "hi"}).status_code == 422
    assert client.post("/search", json={"q": "valid q", "k": 50}).status_code == 422
    assert client.post("/search", json={"q": "valid q", "mode": "bogus"}).status_code == 422


def test_search_returns_passages(client):
    d = client.post("/search", json={"q": "page eviction", "mode": "hybrid"}).json()
    assert d["results"][0]["pages"] == "pp. 14–15"
    assert "embedding" not in d["results"][0]
    assert "total" in d["timings_ms"]


def test_ask_answers_then_rate_limits(client):
    d = client.post("/ask", json={"q": "page eviction"}).json()
    assert d["abstained"] is False and d["cited"] == [1]
    assert client.post("/ask", json={"q": "page eviction"}).status_code == 200
    assert client.post("/ask", json={"q": "page eviction"}).status_code == 429


def test_openai_failure_returns_503(client):
    import httpx
    import openai

    class Down(FakeRetriever):
        async def search(self, *a, **kw):
            raise openai.APIConnectionError(request=httpx.Request("POST", "https://api.openai.com"))

    client.app.state.retriever = Down(0.9)
    r = client.post("/search", json={"q": "page eviction", "mode": "vector"})
    assert r.status_code == 503 and "keyword" in r.json()["detail"]["hint"]


def test_ask_abstains_on_low_confidence(client):
    client.app.state.retriever = FakeRetriever(top=0.5)
    d = client.post("/ask", json={"q": "robotics elective"}).json()
    assert d["abstained"] is True and d["abstain_reason"] == "low_retrieval_confidence"
    assert "llm" not in d["timings_ms"]
