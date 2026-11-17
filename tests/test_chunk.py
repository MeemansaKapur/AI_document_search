from app.ingest.chunk import chunk_pages, contextual_text, count_tokens
from app.ingest.clean import CleanPage


def para(word: str, n: int) -> str:
    return " ".join(f"{word}{i}." for i in range(n))


def test_chunks_respect_size_and_track_pages():
    pages = [CleanPage(1, [para("alpha", 80), para("beta", 80)], "Intro"),
             CleanPage(2, [para("gamma", 80)], None),
             CleanPage(3, [para("delta", 80), para("eps", 80)], "Next")]
    chunks = chunk_pages(pages, size=200, overlap=30)
    assert len(chunks) > 1
    assert all(c.token_count <= 200 * 1.05 for c in chunks)
    assert chunks[0].page_start == 1 and chunks[-1].page_end == 3
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert all(c.page_start <= c.page_end for c in chunks)
    assert chunks[0].section == "Intro"


def test_overlap_present_between_consecutive_chunks():
    pages = [CleanPage(1, [para(f"p{j}w", 10) for j in range(20)], None)]
    chunks = chunk_pages(pages, size=150, overlap=60)
    for a, b in zip(chunks, chunks[1:]):
        last_para = a.text.split("\n\n")[-1]
        assert b.text.startswith(last_para)


def test_huge_paragraph_is_split():
    pages = [CleanPage(1, [" ".join(f"Sentence number {i} is here." for i in range(400))], None)]
    chunks = chunk_pages(pages, size=120, overlap=20)
    assert len(chunks) >= 10
    assert all(c.token_count <= 120 * 1.05 for c in chunks)


def test_empty_pages_handled():
    assert chunk_pages([CleanPage(1, [], None), CleanPage(2, [], None)]) == []
    chunks = chunk_pages([CleanPage(1, [], None), CleanPage(2, ["Only text here."], None)])
    assert len(chunks) == 1 and chunks[0].page_start == 2


def test_contextual_header():
    assert contextual_text("OS Unit 3", "Page Replacement", "body") == "OS Unit 3 > Page Replacement\n\nbody"
    assert contextual_text("OS Unit 3", None, "body") == "OS Unit 3\n\nbody"
    assert count_tokens("hello world") == 2
