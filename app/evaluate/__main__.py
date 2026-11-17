"""Evaluate keyword / vector / hybrid retrieval on hand-labelled questions.

  python -m app.evaluate --split test                       # main mode comparison
  python -m app.evaluate --split dev --calibrate            # abstain threshold
  python -m app.evaluate --modes vector --ncm 5,15,50       # numCandidates: recall vs latency
  python -m app.evaluate --modes hybrid --weights 1:1,1:2,2:1
  python -m app.evaluate --db docsearch_c300 --label c300   # chunk-size experiment (separate DB)

Writes results/<label>_<timestamp>.md and .json.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

from openai import AsyncOpenAI
from pymongo import AsyncMongoClient

from app.config import get_settings
from app.evaluate.metrics import best_threshold, first_hit_rank, hit_at_k, mrr_at_k
from app.retrieval.cache import QueryEmbedder
from app.retrieval.search import Retriever

DEFAULT_QUESTIONS = Path(__file__).with_name("questions.jsonl")
TYPE_ORDER = ["exact", "paraphrase", "multi"]


def load_questions(path: Path, split: str) -> list[dict]:
    qs = [json.loads(line) for line in path.read_text().splitlines() if line.strip() and not line.startswith("//")]
    ids = [q["id"] for q in qs]
    if len(ids) != len(set(ids)):
        sys.exit("duplicate question ids")
    return [q for q in qs if split == "all" or q.get("split", "test") == split]


def pct(x: float) -> str:
    return f"{x * 100:.0f}%"


async def run(args) -> None:
    s = get_settings()
    if args.db:
        s = s.model_copy(update={"mongodb_db": args.db})
    questions = load_questions(args.questions, args.split)
    answerable = [q for q in questions if q["gold"]]
    unanswerable = [q for q in questions if not q["gold"]]
    if len(answerable) < 30:
        print(f"note: {len(answerable)} answerable questions in split '{args.split}' (the resume claims 30)")

    mongo = AsyncMongoClient(s.mongodb_uri)
    oai = AsyncOpenAI(api_key=s.openai_api_key or None)
    retriever = Retriever(mongo[s.mongodb_db].chunks, QueryEmbedder(oai, s.embedding_model, 4096), s)

    configs = []
    for mode in args.modes.split(","):
        for ncm in ([int(x) for x in args.ncm.split(",")] if args.ncm else [s.num_candidates_multiplier]):
            weight_list = args.weights.split(",") if (mode == "hybrid" and args.weights) else [f"{s.w_text:g}:{s.w_vector:g}"]
            for w in weight_list:
                wt, wv = (float(x) for x in w.split(":"))
                name = mode + (f" ncm={ncm}" if args.ncm else "") + (f" text:vec={w}" if args.weights and mode == "hybrid" else "")
                configs.append({"name": name, "mode": mode, "ncm": ncm, "w_text": wt, "w_vector": wv})

    # Warm the query-embedding cache once so per-config latency reflects the database, not the API.
    for q in questions:
        await retriever.embedder.embed(q["question"])

    rows, details = [], {}
    for cfg in configs:
        per_q = []
        for q in answerable:
            out = await retriever.search(
                q["question"], cfg["mode"], k=10, course=q.get("course") if args.course_filter else None,
                num_candidates_multiplier=cfg["ncm"], w_vector=cfg["w_vector"], w_text=cfg["w_text"],
                rrf_k=args.rrf_k)
            rank = first_hit_rank(out.results, q["gold"])
            db_ms = max(out.timings_ms.get("vector", 0), out.timings_ms.get("text", 0))
            per_q.append({"id": q["id"], "type": q.get("type"), "rank": rank, "db_ms": db_ms,
                          "top3": [f"{r['filename']} p{r['page_start']}-{r['page_end']}" for r in out.results[:3]]})
        ranks = [p["rank"] for p in per_q]
        by_type = {t: hit_at_k([p["rank"] for p in per_q if p["type"] == t], 3)
                   for t in TYPE_ORDER if any(p["type"] == t for p in per_q)}
        rows.append({"config": cfg["name"], "n": len(ranks), "hit@3": hit_at_k(ranks, 3),
                     "mrr@10": mrr_at_k(ranks, 10), "by_type": by_type,
                     "db_ms_p50": statistics.median(p["db_ms"] for p in per_q) if per_q else 0})
        details[cfg["name"]] = per_q

    types = [t for t in TYPE_ORDER if any(t in r["by_type"] for r in rows)]
    lines = [f"Split: `{args.split}`, {len(answerable)} answerable questions, db `{s.mongodb_db}`\n",
             "| Config | Hit@3 | MRR@10 | " + " | ".join(t.capitalize() for t in types) + " | DB stage p50 (ms) |",
             "|---|---|---|" + "---|" * len(types) + "---|"]
    for r in rows:
        lines.append(f"| {r['config']} | {pct(r['hit@3'])} | {r['mrr@10']:.2f} | "
                     + " | ".join(pct(r["by_type"][t]) if t in r["by_type"] else "-" for t in types)
                     + f" | {r['db_ms_p50']:.0f} |")
    type_counts = {t: sum(1 for q in answerable if q.get("type") == t) for t in types}
    lines.append(f"\nQuestions per type: {type_counts}")

    calibration = None
    if args.calibrate:
        top = {}
        for q in questions:
            out = await retriever.search(q["question"], "vector", k=1)
            top[q["id"]] = out.top_vector_score or 0.0
        a = [top[q["id"]] for q in answerable]
        u = [top[q["id"]] for q in unanswerable]
        t, acc = best_threshold(a, u)
        calibration = {"answerable": sorted(a), "unanswerable": sorted(u), "threshold": t, "accuracy": acc}
        lines += ["\n### Abstain threshold calibration (top-1 vector score)",
                  f"- answerable: min {min(a, default=0):.3f}, median {statistics.median(a) if a else 0:.3f}",
                  f"- unanswerable: max {max(u, default=0):.3f}, median {statistics.median(u) if u else 0:.3f}",
                  f"- best threshold: **{t}** (accuracy {pct(acc)}); set ABSTAIN_THRESHOLD={t}"]

    report = "\n".join(lines)
    print(report)
    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.out / f"{args.label}_{args.split}_{datetime.now():%Y%m%d_%H%M%S}"
    stem.with_suffix(".md").write_text(report + "\n")
    stem.with_suffix(".json").write_text(json.dumps(
        {"rows": rows, "details": details, "calibration": calibration, "db": s.mongodb_db}, indent=2))
    print(f"\nwrote {stem}.md / .json")
    await mongo.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    p.add_argument("--split", default="test", choices=["dev", "test", "all"])
    p.add_argument("--modes", default="keyword,vector,hybrid")
    p.add_argument("--ncm", help="comma list of numCandidates multipliers, e.g. 5,15,50")
    p.add_argument("--weights", help="hybrid text:vector weights, e.g. 1:1,1:2,2:1")
    p.add_argument("--rrf-k", type=int, default=None)
    p.add_argument("--course-filter", action="store_true", help="restrict each question to its 'course'")
    p.add_argument("--calibrate", action="store_true")
    p.add_argument("--db")
    p.add_argument("--label", default="eval")
    p.add_argument("--out", type=Path, default=Path("results"))
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
