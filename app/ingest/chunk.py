"""Paragraph-aware chunking with token overlap and page-range tracking."""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import tiktoken

from app.ingest.clean import CleanPage

SEP = "\n\n"
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'\[])")


@lru_cache
def _enc():
    # text-embedding-3-* models use cl100k_base
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_enc().encode(text, disallowed_special=()))


@dataclass
class Unit:
    text: str
    page: int
    section: str | None
    ntok: int


@dataclass
class Chunk:
    index: int
    text: str
    page_start: int
    page_end: int
    section: str | None
    token_count: int


def _split_long(text: str, max_tokens: int) -> list[str]:
    """Split an oversized paragraph by sentences (hard token split as a last resort)."""
    pieces: list[str] = []
    for sent in _SENTENCE.split(text):
        if count_tokens(sent) <= max_tokens:
            pieces.append(sent)
        else:
            toks = _enc().encode(sent, disallowed_special=())
            pieces.extend(_enc().decode(toks[i:i + max_tokens]) for i in range(0, len(toks), max_tokens))
    merged: list[str] = []
    cur = ""
    for p in pieces:
        cand = f"{cur} {p}" if cur else p
        if cur and count_tokens(cand) > max_tokens:
            merged.append(cur)
            cur = p
        else:
            cur = cand
    if cur:
        merged.append(cur)
    return merged


def _units(pages: list[CleanPage], max_tokens: int) -> list[Unit]:
    units: list[Unit] = []
    section = None
    for page in pages:
        if page.heading:
            section = page.heading
        for para in page.paragraphs:
            n = count_tokens(para)
            if n <= max_tokens:
                units.append(Unit(para, page.number, section, n))
            else:
                units.extend(Unit(t, page.number, section, count_tokens(t))
                             for t in _split_long(para, max_tokens))
    return units


def _size(units: list[Unit]) -> int:
    return sum(u.ntok for u in units) + max(0, len(units) - 1)  # SEP is one token


def _tail(units: list[Unit], overlap: int) -> list[Unit]:
    """Last ~`overlap` tokens of a chunk, kept on whole paragraphs where possible."""
    if overlap <= 0 or not units:
        return []
    tail: list[Unit] = []
    for u in reversed(units):
        if _size([u, *tail]) > overlap:
            break
        tail.insert(0, u)
    if tail:
        return tail
    last = units[-1]
    toks = _enc().encode(last.text, disallowed_special=())
    text = _enc().decode(toks[-overlap:])
    if " " in text:  # don't start mid-word
        text = text.split(" ", 1)[1]
    text = text.strip()
    return [Unit(text, last.page, last.section, count_tokens(text))] if text else []


def chunk_pages(pages: list[CleanPage], size: int = 600, overlap: int = 90) -> list[Chunk]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    chunks: list[Chunk] = []

    def emit(units: list[Unit]) -> None:
        text = SEP.join(u.text for u in units)
        chunks.append(Chunk(
            index=len(chunks), text=text,
            page_start=min(u.page for u in units), page_end=max(u.page for u in units),
            section=units[0].section, token_count=count_tokens(text)))

    cur: list[Unit] = []
    fresh = 0  # units in `cur` that are not overlap carried from the previous chunk
    for u in _units(pages, size):
        if fresh and _size(cur) + 1 + u.ntok > size:
            emit(cur)
            cur, fresh = _tail(cur, overlap), 0
        while cur and _size(cur) + 1 + u.ntok > size:
            cur.pop(0)  # overlap too large to fit next to this paragraph
        cur.append(u)
        fresh += 1
    if fresh:
        emit(cur)
    return chunks


def contextual_text(title: str, section: str | None, text: str) -> str:
    """'Contextual chunk header': what gets embedded is title > section + text."""
    header = f"{title} > {section}" if section else title
    return f"{header}\n\n{text}"
