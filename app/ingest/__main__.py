"""Ingest a folder of PDFs into MongoDB Atlas: parse -> clean -> chunk -> embed -> store.

Usage:
  python -m app.ingest --dir data/pdfs/os --course "CS F372" --doc-type slides
  python -m app.ingest --dir data/pdfs --recursive          # course = sub-folder name

Idempotent: a file already ingested with the same chunk config is skipped; chunk _ids are
deterministic (<file sha1>:<chunk index>) so re-runs overwrite instead of duplicating.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI
from pymongo import MongoClient, ReplaceOne

from app.config import get_settings
from app.ingest.chunk import chunk_pages, contextual_text
from app.ingest.clean import clean_pages
from app.ingest.embed import Embedder
from app.ingest.parse import parse_pdf

CHUNK_CONFIG_VERSION = 1
log = logging.getLogger("ingest")


def sha1(data: bytes | str) -> str:
    return hashlib.sha1(data.encode() if isinstance(data, str) else data).hexdigest()


def pretty_title(path: Path, metadata_title: str | None) -> str:
    bad = ("untitled", "microsoft", "powerpoint", "slide 1", ".doc", ".ppt")
    if metadata_title and len(metadata_title) > 3 and not any(b in metadata_title.lower() for b in bad):
        return metadata_title
    return path.stem.replace("_", " ").replace("-", " ").strip().title()


def ingest_file(path: Path, *, db, embedder: Embedder, course: str, doc_type: str, source: str,
                source_url: str | None, cfg: dict, model: str, force: bool) -> str:
    data = path.read_bytes()
    doc_id = sha1(data)
    documents, chunks = db.documents, db.chunks

    existing = documents.find_one({"_id": doc_id})
    if existing and existing.get("status") == "ok" and existing.get("chunk_config") == cfg and not force:
        return "skipped_unchanged"

    # An edited file has new bytes -> new doc_id; drop the old version's chunks.
    for old in documents.find({"filename": path.name, "course": course, "_id": {"$ne": doc_id}}, {"_id": 1}):
        chunks.delete_many({"doc_id": old["_id"]})
        documents.delete_one({"_id": old["_id"]})

    base = {"_id": doc_id, "filename": path.name, "course": course, "doc_type": doc_type,
            "source": source, "source_url": source_url, "chunk_config": cfg,
            "ingested_at": datetime.now(timezone.utc)}
    try:
        parsed = parse_pdf(path)
        title = pretty_title(path, parsed.metadata_title)
        base.update(title=title, pages=len(parsed.pages))
        if parsed.looks_scanned():
            documents.replace_one({"_id": doc_id}, {**base, "status": "skipped_scanned"}, upsert=True)
            return "skipped_scanned"

        pieces = chunk_pages(clean_pages(parsed.pages), size=cfg["size"], overlap=cfg["overlap"])
        if not pieces:
            documents.replace_one({"_id": doc_id}, {**base, "status": "skipped_empty"}, upsert=True)
            return "skipped_empty"

        records = []
        for c in pieces:
            embed_text = contextual_text(title, c.section, c.text) if cfg["contextual_headers"] else c.text
            records.append({
                "_id": f"{doc_id}:{c.index:05d}", "doc_id": doc_id, "filename": path.name,
                "title": title, "course": course, "doc_type": doc_type,
                "page_start": c.page_start, "page_end": c.page_end, "section": c.section,
                "chunk_index": c.index, "text": c.text, "token_count": c.token_count,
                "content_hash": sha1(embed_text), "embedding_model": model,
                "_embed_text": embed_text,
            })

        # Reuse vectors for chunks whose embedded text is unchanged (saves money on re-runs).
        prior = {d["_id"]: d for d in chunks.find(
            {"doc_id": doc_id, "embedding_model": model}, {"content_hash": 1, "embedding": 1})}
        todo = []
        for r in records:
            p = prior.get(r["_id"])
            if p and p.get("content_hash") == r["content_hash"] and p.get("embedding"):
                r["embedding"] = p["embedding"]
            else:
                todo.append(r)
        if todo:
            for r, vec in zip(todo, embedder.embed([r["_embed_text"] for r in todo]), strict=True):
                r["embedding"] = vec

        for r in records:
            del r["_embed_text"]
        for i in range(0, len(records), 200):
            chunks.bulk_write([ReplaceOne({"_id": r["_id"]}, r, upsert=True) for r in records[i:i + 200]],
                              ordered=False)
        chunks.delete_many({"doc_id": doc_id, "chunk_index": {"$gte": len(records)}})

        documents.replace_one({"_id": doc_id}, {
            **base, "status": "ok", "chunks": len(records),
            "tokens": sum(r["token_count"] for r in records), "embedded_new": len(todo)}, upsert=True)
        return "ok"
    except Exception as e:  # corrupt / encrypted PDFs etc.: record and continue
        log.exception("failed: %s", path)
        documents.replace_one({"_id": doc_id}, {**base, "status": "failed", "error": str(e)[:500]},
                              upsert=True)
        return "failed"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", required=True, type=Path)
    p.add_argument("--recursive", action="store_true",
                   help="walk sub-folders; course defaults to each file's parent folder name")
    p.add_argument("--course", help="course label (default: parent folder name)")
    p.add_argument("--doc-type", default="notes", choices=["slides", "notes", "textbook", "paper"])
    p.add_argument("--source", default="own-notes", help="e.g. own-notes, openstax")
    p.add_argument("--source-url")
    p.add_argument("--chunk-size", type=int, default=600)
    p.add_argument("--overlap", type=int, help="tokens of overlap (default 15%% of chunk size)")
    p.add_argument("--no-headers", action="store_true", help="disable contextual chunk headers")
    p.add_argument("--db", help="database name (overrides MONGODB_DB); use one per experiment")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    settings = get_settings()
    if args.db:
        settings = settings.model_copy(update={"mongodb_db": args.db})
    if not settings.openai_api_key:
        sys.exit("OPENAI_API_KEY is not set")

    cfg = {"size": args.chunk_size,
           "overlap": args.overlap if args.overlap is not None else round(args.chunk_size * 0.15),
           "contextual_headers": not args.no_headers, "version": CHUNK_CONFIG_VERSION}
    db = MongoClient(settings.mongodb_uri, appname="atlas-doc-search-ingest")[settings.mongodb_db]
    embedder = Embedder(OpenAI(api_key=settings.openai_api_key, max_retries=0), settings.embedding_model)

    pattern = "**/*.pdf" if args.recursive else "*.pdf"
    files = sorted(f for f in args.dir.glob(pattern) if f.is_file())
    if not files:
        sys.exit(f"no PDFs found in {args.dir}")

    counts: dict[str, int] = {}
    for i, f in enumerate(files, 1):
        status = ingest_file(f, db=db, embedder=embedder, course=args.course or f.parent.name,
                             doc_type=args.doc_type, source=args.source, source_url=args.source_url,
                             cfg=cfg, model=settings.embedding_model, force=args.force)
        counts[status] = counts.get(status, 0) + 1
        log.info("[%d/%d] %s: %s", i, len(files), f.name, status)

    log.info("done %s | embedding tokens used: %d | db=%s documents ok=%d chunks=%d",
             counts, embedder.tokens_used, settings.mongodb_db,
             db.documents.count_documents({"status": "ok"}), db.chunks.count_documents({}))


if __name__ == "__main__":
    main()
