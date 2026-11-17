"""PDF text extraction with PyMuPDF, page by page, plus scanned-PDF detection."""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path

import pymupdf


@dataclass
class Page:
    number: int                 # 1-based
    blocks: list[list[str]]     # text blocks, each a list of lines
    heading: str | None = None  # largest-font line, if clearly larger than body text


@dataclass
class ParsedPDF:
    pages: list[Page]
    metadata_title: str | None

    @property
    def char_count(self) -> int:
        return sum(len(line) for p in self.pages for b in p.blocks for line in b)

    def looks_scanned(self, min_chars_per_page: int = 50) -> bool:
        """Images of pages yield almost no text; OCR is out of scope."""
        return not self.pages or self.char_count / len(self.pages) < min_chars_per_page


def parse_pdf(path: str | Path) -> ParsedPDF:
    with pymupdf.open(path) as doc:
        if doc.needs_pass:
            raise ValueError("encrypted PDF")
        title = ((doc.metadata or {}).get("title") or "").strip() or None
        pages = [_parse_page(i, page) for i, page in enumerate(doc, start=1)]
    return ParsedPDF(pages=pages, metadata_title=title)


def _parse_page(number: int, page) -> Page:
    d = page.get_text("dict", sort=True)
    blocks: list[list[str]] = []
    sized_lines: list[tuple[str, float]] = []
    for b in d.get("blocks", []):
        if b.get("type") != 0:  # 0 = text, 1 = image
            continue
        lines = []
        for line in b.get("lines", []):
            spans = line.get("spans", [])
            text = "".join(s.get("text", "") for s in spans).strip()
            if not text:
                continue
            lines.append(text)
            sized_lines.append((text, max(s.get("size", 0.0) for s in spans)))
        if lines:
            blocks.append(lines)
    return Page(number=number, blocks=blocks, heading=detect_heading(sized_lines))


def detect_heading(sized_lines: list[tuple[str, float]], ratio: float = 1.2) -> str | None:
    """Return the first line whose font is >= ratio x the (character-weighted) median size."""
    if len(sized_lines) < 2:
        return None
    sizes: list[float] = []
    for text, size in sized_lines:
        sizes.extend([round(size, 1)] * min(len(text), 200))
    body = statistics.median(sizes)
    for text, size in sized_lines:
        if size >= body * ratio and 3 <= len(text) <= 120 and any(c.isalpha() for c in text):
            return text
    return None
