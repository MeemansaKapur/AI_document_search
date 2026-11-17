from app.ingest.clean import clean_pages, join_lines
from app.ingest.parse import Page, detect_heading


def make_page(n: int, body: list[str], heading=None) -> Page:
    return Page(n, [["BITS Pilani Hyderabad Campus"], body, [f"CS F372 Operating Systems  Page {n} of 4"]], heading)


def test_repeated_header_and_footer_removed_unique_line_kept():
    topics = ["paging", "segmentation", "TLB lookups", "thrashing"]
    pages = [make_page(i, [f"Notes on {topics[i - 1]}"]) for i in range(1, 5)]
    pages[1].blocks.insert(1, ["A line appearing only once"])
    cleaned = clean_pages(pages)
    text = [" ".join(p.paragraphs) for p in cleaned]
    assert all("BITS Pilani" not in t and "Page" not in t for t in text)
    assert "Notes on TLB lookups" in text[2]
    assert "A line appearing only once" in text[1]


def test_short_documents_keep_all_lines():
    cleaned = clean_pages([make_page(1, ["x"]), make_page(2, ["y"])])
    assert "BITS Pilani Hyderabad Campus" in cleaned[0].paragraphs


def test_hyphenation_joined_and_bullets_kept():
    assert join_lines(["page replace-", "ment policy"]) == "page replacement policy"
    assert join_lines(["Algorithms:", "• LRU", "• FIFO"]) == "Algorithms:\n• LRU\n• FIFO"
    assert join_lines(["wrapped   line", "continues"]) == "wrapped line continues"
    assert join_lines(["Self-", "Study"]) == "Self- Study"  # capitalized: not a hyphenation break


def test_repeated_heading_dropped():
    pages = [make_page(i, ["body"], heading="Course Title") for i in range(1, 5)]
    for p in pages:
        p.blocks.insert(0, ["Course Title"])
    assert all(p.heading is None for p in clean_pages(pages, edge=3))


def test_detect_heading():
    lines = [("Page Replacement", 24.0)] + [("body text " * 5, 11.0)] * 10
    assert detect_heading(lines) == "Page Replacement"
    assert detect_heading([("all same", 11.0), ("size here", 11.0)]) is None
