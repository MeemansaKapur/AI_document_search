"""Remove repeated headers/footers, join hyphenated words, normalize whitespace."""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from app.ingest.parse import Page

_NUM = re.compile(r"\d+")
_BULLET = re.compile(r"^\s*([•▪◦●○■□➢►\-–*·]|\d+[.)]|[a-zA-Z][.)])\s+")
_SPACES = re.compile(r"[ \t ]+")


@dataclass
class CleanPage:
    number: int
    paragraphs: list[str]
    heading: str | None


def _norm(line: str) -> str:
    """Normalize for header/footer matching: 'CS F372 Page 3 of 40' and '... Page 4 of 40' compare
    equal. Only lines in the first/last `edge` positions of a page are ever candidates."""
    return _NUM.sub("#", " ".join(line.lower().split()))


def _edge_positions(page: Page, edge: int) -> list[tuple[int, int]]:
    flat = [(bi, li) for bi, block in enumerate(page.blocks) for li in range(len(block))]
    if len(flat) <= 2 * edge:
        return flat
    return flat[:edge] + flat[-edge:]


def find_repeated_edge_lines(pages: list[Page], edge: int = 2, min_ratio: float = 0.5,
                             min_pages: int = 3) -> set[str]:
    """Lines that appear in the first/last `edge` lines of more than `min_ratio` of pages."""
    if len(pages) < min_pages:
        return set()
    counts: Counter[str] = Counter()
    for page in pages:
        counts.update({_norm(page.blocks[bi][li]) for bi, li in _edge_positions(page, edge)})
    return {k for k, c in counts.items() if k and c / len(pages) > min_ratio}


def join_lines(lines: list[str]) -> str:
    """Join a block's lines: fix 'replace-\\nment', keep bullets on their own lines."""
    out = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if not out:
            out = line
        elif out.endswith("-") and len(out) > 1 and out[-2].isalpha() and line[0].islower():
            out = out[:-1] + line
        elif _BULLET.match(line):
            out += "\n" + line
        else:
            out += " " + line
    return _SPACES.sub(" ", out).strip()


def clean_pages(pages: list[Page], edge: int = 2) -> list[CleanPage]:
    repeated = find_repeated_edge_lines(pages, edge=edge)
    cleaned = []
    for page in pages:
        drop = {pos for pos in _edge_positions(page, edge)
                if _norm(page.blocks[pos[0]][pos[1]]) in repeated}
        paragraphs = []
        for bi, block in enumerate(page.blocks):
            kept = [line for li, line in enumerate(block) if (bi, li) not in drop]
            text = join_lines(kept)
            if text:
                paragraphs.append(text)
        heading = page.heading
        if heading and _norm(heading) in repeated:
            heading = None  # e.g. a deck title repeated on every slide
        cleaned.append(CleanPage(number=page.number, paragraphs=paragraphs, heading=heading))
    return cleaned
