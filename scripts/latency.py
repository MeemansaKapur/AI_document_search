"""Client-side latency benchmark against a running API (run it against the DEPLOYED URL).

  python scripts/latency.py --url https://your-host --n 200 --concurrency 1,5
  python scripts/latency.py --url https://your-host --endpoint ask --n 40

Each request's server-reported cache_hit splits results into cold vs warm (the query-embedding
cache is per worker process, so this is more honest than guessing from send order).
Writes results/latency_<endpoint>_<timestamp>.csv and prints p50/p95.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import time
from datetime import datetime
from pathlib import Path

import httpx

VARIANTS = ["{q}", "{q} explain", "In the course notes, {q}", "{q} (briefly)", "Question: {q}",
            "{q} with an example", "Can you tell me: {q}"]


def pctl(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    return s[max(0, math.ceil(p / 100 * len(s)) - 1)]  # nearest-rank


def build_queries(path: Path, n: int) -> list[str]:
    qs = [json.loads(line)["question"] for line in path.read_text().splitlines()
          if line.strip() and not line.startswith("//")]
    out = []
    for v in VARIANTS:
        out += [v.format(q=q) for q in qs]
    return (out * math.ceil(n / len(out)))[:n]


async def bench(url: str, endpoint: str, queries: list[str], concurrency: int, mode: str) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)
    rows: list[dict] = []
    async with httpx.AsyncClient(base_url=url, timeout=60) as client:
        async def one(q: str):
            async with sem:
                body = {"q": q, "mode": mode, "k": 5} if endpoint == "search" else {"q": q}
                t0 = time.perf_counter()
                r = await client.post(f"/{endpoint}", json=body)
                ms = (time.perf_counter() - t0) * 1000
                d = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
                rows.append({"concurrency": concurrency, "status": r.status_code, "client_ms": round(ms, 1),
                             "server_total_ms": d.get("timings_ms", {}).get("total"),
                             "cache_hit": d.get("cache_hit"), "q": q})
        await asyncio.gather(*(one(q) for q in queries))
    return rows


def summarize(rows: list[dict], label: str) -> None:
    ok = [r for r in rows if r["status"] == 200]
    groups = {"all": ok, "cold": [r for r in ok if r["cache_hit"] is False],
              "warm": [r for r in ok if r["cache_hit"] is True]}
    print(f"\n{label}: {len(ok)}/{len(rows)} OK")
    print("| subset | n | p50 ms | p95 ms |\n|---|---|---|---|")
    for name, g in groups.items():
        if g:
            lat = [r["client_ms"] for r in g]
            print(f"| {name} | {len(g)} | {pctl(lat, 50):.0f} | {pctl(lat, 95):.0f} |")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", required=True)
    p.add_argument("--endpoint", default="search", choices=["search", "ask"])
    p.add_argument("--mode", default="hybrid")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--concurrency", default="1,5")
    p.add_argument("--questions", type=Path, default=Path("app/evaluate/questions.jsonl"))
    p.add_argument("--warm-pass", action="store_true", help="re-send the same queries to measure warm cache")
    args = p.parse_args()

    queries = build_queries(args.questions, args.n)
    all_rows = []
    for c in (int(x) for x in args.concurrency.split(",")):
        # unique prefix per run so each concurrency level starts cold
        qs = [f"{q} #{c}" if c > 1 else q for q in queries]
        rows = asyncio.run(bench(args.url, args.endpoint, qs, c, args.mode))
        summarize(rows, f"/{args.endpoint} concurrency={c}")
        all_rows += rows
        if args.warm_pass:
            rows = asyncio.run(bench(args.url, args.endpoint, qs, c, args.mode))
            summarize(rows, f"/{args.endpoint} concurrency={c} (repeat pass)")
            all_rows += rows

    out = Path("results") / f"latency_{args.endpoint}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    out.parent.mkdir(exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nwrote {out}  (URL: {args.url})")


if __name__ == "__main__":
    main()
