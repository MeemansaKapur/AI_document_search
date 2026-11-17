import pytest

from app.evaluate.metrics import best_threshold, first_hit_rank, hit_at_k, is_hit, mrr_at_k
from app.generate import REFUSAL, build_messages, is_refusal, pages_label, parse_citations

GOLD = [{"filename": "os.pdf", "pages": [14, 15]}]


def r(fn, a, b):
    return {"filename": fn, "page_start": a, "page_end": b}


def test_is_hit_page_overlap():
    assert is_hit(r("os.pdf", 13, 14), GOLD)
    assert not is_hit(r("os.pdf", 16, 17), GOLD)
    assert not is_hit(r("db.pdf", 14, 14), GOLD)


def test_first_hit_rank():
    assert first_hit_rank([r("db.pdf", 1, 1), r("os.pdf", 15, 15)], GOLD) == 2
    assert first_hit_rank([r("db.pdf", 1, 1)], GOLD) is None


def test_hit_and_mrr_hand_computed():
    ranks = [1, 3, None, 5]
    assert hit_at_k(ranks, 3) == 0.5
    assert mrr_at_k(ranks, 10) == pytest.approx((1 + 1 / 3 + 0 + 1 / 5) / 4)
    assert mrr_at_k([11], 10) == 0.0
    assert hit_at_k([], 3) == 0.0


def test_best_threshold_separates():
    t, acc = best_threshold([0.8, 0.85, 0.9], [0.6, 0.65])
    assert 0.65 < t < 0.8 and acc == 1.0


def test_citations_and_refusal():
    assert parse_citations("A [1] and B [2][3], also [7].", 5) == ([1, 2, 3], [7])
    assert is_refusal(REFUSAL) and is_refusal("not found in the provided documents")
    assert not is_refusal("LRU evicts the least recently used page [1].")
    assert pages_label(3, 3) == "p. 3" and pages_label(3, 4) == "pp. 3–4"


def test_prompt_numbers_and_labels_passages():
    msgs = build_messages("q?", [{"title": "OS", "page_start": 1, "page_end": 2, "text": "body"}])
    assert msgs[0]["role"] == "system" and "[1] (OS, pp. 1–2)" in msgs[1]["content"]
