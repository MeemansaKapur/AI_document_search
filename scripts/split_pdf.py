"""Split a large textbook PDF into one PDF per chapter using its table of contents.

  python scripts/split_pdf.py book.pdf --out data/pdfs/networks --prefix openstax_cs --level 1

Useful for openly licensed textbooks (e.g. OpenStax, CC BY) so each chapter is a manageable document.
"""
import argparse
import re
from pathlib import Path

import pymupdf


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:60]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("pdf", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--prefix", default="")
    p.add_argument("--level", type=int, default=1, help="TOC level that marks a chapter")
    p.add_argument("--min-pages", type=int, default=3)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    with pymupdf.open(args.pdf) as doc:
        toc = [(title, page) for lvl, title, page in doc.get_toc() if lvl == args.level and page > 0]
        if not toc:
            raise SystemExit("no table of contents at that level; split manually")
        bounds = [(t, start, (toc[i + 1][1] - 1) if i + 1 < len(toc) else doc.page_count)
                  for i, (t, start) in enumerate(toc)]
        n = 0
        for i, (title, start, end) in enumerate(bounds, 1):
            if end - start + 1 < args.min_pages:
                continue
            part = pymupdf.open()
            part.insert_pdf(doc, from_page=start - 1, to_page=end - 1)
            part.set_metadata({"title": title})
            name = f"{args.prefix + '_' if args.prefix else ''}{i:02d}_{slug(title)}.pdf"
            part.save(args.out / name)
            part.close()
            n += 1
            print(f"{name}: pages {start}-{end}")
    print(f"wrote {n} chapter PDFs to {args.out}")


if __name__ == "__main__":
    main()
