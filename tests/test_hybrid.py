import pytest

from app.retrieval.hybrid import rrf_fuse
from app.retrieval.pipelines import text_pipeline, vector_pipeline


def docs(*ids):
    return [{"_id": i, "score": 1.0} for i in ids]


def test_doc_first_in_both_beats_doc_first_in_one():
    out = rrf_fuse({"vector": docs("a", "b"), "keyword": docs("a", "c")}, {})
    assert out[0]["_id"] == "a"
    assert out[0]["ranks"] == {"vector": 1, "keyword": 1}
    assert out[0]["score"] == pytest.approx(2 / 61)


def test_single_list_doc_still_gets_credit():
    out = rrf_fuse({"vector": docs("a"), "keyword": docs("b")}, {})
    assert {d["_id"] for d in out} == {"a", "b"}


def test_weights_respected():
    lists = {"vector": docs("v"), "keyword": docs("k")}
    assert rrf_fuse(lists, {"vector": 2.0, "keyword": 1.0})[0]["_id"] == "v"
    assert rrf_fuse(lists, {"vector": 1.0, "keyword": 2.0})[0]["_id"] == "k"


def test_ties_are_stable_and_limit_applies():
    out = rrf_fuse({"vector": docs("a", "b"), "keyword": docs("b", "a")}, {}, limit=1)
    assert len(out) == 1 and out[0]["_id"] == "a"


def test_pipelines_exclude_embedding_and_prefilter():
    vp = vector_pipeline([0.1] * 4, 10, num_candidates_multiplier=15, course="CS F372")
    assert vp[0]["$vectorSearch"]["numCandidates"] == 150
    assert vp[0]["$vectorSearch"]["filter"] == {"course": "CS F372"}
    assert "embedding" not in vp[1]["$project"]
    tp = text_pipeline("lru", 10, course="CS F372")
    assert tp[0]["$search"]["compound"]["filter"][0]["equals"]["value"] == "CS F372"
    assert "embedding" not in tp[2]["$project"]
