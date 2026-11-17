"""Print the corpus numbers the resume cites (documents ingested, chunks), straight from the DB.

  python scripts/stats.py [--db NAME]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pymongo import MongoClient  # noqa: E402

from app.config import get_settings  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db")
    args = p.parse_args()
    s = get_settings()
    db = MongoClient(s.mongodb_uri)[args.db or s.mongodb_db]

    status = {d["_id"]: d["n"] for d in db.documents.aggregate([{"$group": {"_id": "$status", "n": {"$sum": 1}}}])}
    by_course = list(db.documents.aggregate([
        {"$match": {"status": "ok"}},
        {"$group": {"_id": "$course", "docs": {"$sum": 1}, "chunks": {"$sum": "$chunks"}, "pages": {"$sum": "$pages"}}},
        {"$sort": {"_id": 1}}]))
    tok = next(db.chunks.aggregate([{"$group": {"_id": None, "avg": {"$avg": "$token_count"},
                                                "sum": {"$sum": "$token_count"}}}]), {"avg": 0, "sum": 0})
    models = db.chunks.distinct("embedding_model")

    print(f"database: {db.name}")
    print(f"documents by status: {status}")
    print(f"PDFs ingested (status ok): {status.get('ok', 0)}")
    print(f"chunks: {db.chunks.count_documents({})}  (avg {tok['avg']:.0f} tokens, total {tok['sum']:,} tokens)")
    print(f"embedding models in index: {models}" + ("  <-- WARNING: mixed models" if len(models) > 1 else ""))
    for c in by_course:
        print(f"  {c['_id']}: {c['docs']} docs, {c['pages']} pages, {c['chunks']} chunks")


if __name__ == "__main__":
    main()
