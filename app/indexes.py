"""Create and check the Atlas Vector Search and Atlas Search indexes.

Usage: python -m app.indexes [--db NAME] [--wait]
"""
from __future__ import annotations

import argparse
import time

from pymongo import ASCENDING, MongoClient
from pymongo.collection import Collection
from pymongo.errors import CollectionInvalid
from pymongo.operations import SearchIndexModel

from app.config import Settings, get_settings


def vector_index_definition(dims: int) -> dict:
    return {
        "fields": [
            {"type": "vector", "path": "embedding", "numDimensions": dims, "similarity": "cosine"},
            {"type": "filter", "path": "course"},
            {"type": "filter", "path": "doc_type"},
        ]
    }


TEXT_INDEX_DEFINITION = {
    "mappings": {
        "dynamic": False,  # never index the embedding array
        "fields": {
            "text": {"type": "string", "analyzer": "lucene.english"},
            "title": {"type": "string", "analyzer": "lucene.english"},
            "section": {"type": "string", "analyzer": "lucene.english"},
            "course": {"type": "token"},
        },
    }
}


def ensure_indexes(db, settings: Settings) -> None:
    for name in ("documents", "chunks"):
        try:
            db.create_collection(name)
        except CollectionInvalid:
            pass

    db.chunks.create_index([("doc_id", ASCENDING), ("chunk_index", ASCENDING)])
    db.documents.create_index([("filename", ASCENDING)])

    chunks: Collection = db.chunks
    existing = {ix["name"] for ix in chunks.list_search_indexes()}
    if settings.vector_index not in existing:
        chunks.create_search_index(SearchIndexModel(
            name=settings.vector_index, type="vectorSearch",
            definition=vector_index_definition(settings.embedding_dims)))
        print(f"created {settings.vector_index}")
    if settings.text_index not in existing:
        chunks.create_search_index(SearchIndexModel(
            name=settings.text_index, type="search", definition=TEXT_INDEX_DEFINITION))
        print(f"created {settings.text_index}")


def index_status(chunks: Collection, settings: Settings) -> dict[str, str]:
    found = {ix["name"]: ix for ix in chunks.list_search_indexes()}
    out = {}
    for name in (settings.vector_index, settings.text_index):
        ix = found.get(name)
        out[name] = "missing" if ix is None else ("ready" if ix.get("queryable") else "building")
    return out


def wait_until_ready(chunks: Collection, settings: Settings, timeout_s: int = 600) -> dict[str, str]:
    """Search indexes build asynchronously; poll until both are queryable."""
    deadline = time.monotonic() + timeout_s
    while True:
        status = index_status(chunks, settings)
        if all(v == "ready" for v in status.values()):
            return status
        if time.monotonic() > deadline:
            raise TimeoutError(f"indexes not ready: {status}")
        time.sleep(3)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", help="database name (overrides MONGODB_DB)")
    p.add_argument("--wait", action="store_true", help="block until indexes are queryable")
    args = p.parse_args()

    settings = get_settings()
    if args.db:
        settings = settings.model_copy(update={"mongodb_db": args.db})
    client = MongoClient(settings.mongodb_uri)
    db = client[settings.mongodb_db]
    ensure_indexes(db, settings)
    status = wait_until_ready(db.chunks, settings) if args.wait else index_status(db.chunks, settings)
    print(status)


if __name__ == "__main__":
    main()
