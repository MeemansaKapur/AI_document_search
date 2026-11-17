# Course Search: hybrid RAG retrieval over lecture PDFs with MongoDB Atlas Vector Search

Ask a question about course material and get the exact passages (with page numbers) plus a short answer that cites them.
Keyword, vector and hybrid retrieval, measured on a hand-labelled question set.

**Stack:** Python · FastAPI · MongoDB Atlas Vector Search + Atlas Search · OpenAI API · Docker · AWS EC2

**Live demo:** `<url>`  **Corpus:** `<N>` PDFs, `<N>` chunks (`python scripts/stats.py`)

## Results
<!-- paste from results/eval_test_*.md: 30 answerable test questions, never used for tuning -->
| Mode | Hit@3 | MRR@10 | Exact-term | Paraphrase | Multi-concept |
|---|---|---|---|---|---|
| keyword | | | | | |
| vector | | | | | |
| hybrid | | | | | |

Why hybrid wins: *(2–3 sentences with a real example question from the per-type breakdown)*

## Latency
<!-- paste from scripts/latency.py run against the deployed URL; state regions + instance type -->
| Endpoint | Cache | p50 | p95 |
|---|---|---|---|
| `/search` (hybrid) | cold | | |
| `/search` (hybrid) | warm | | |
| `/ask` | cold | | |

Measured from a client against EC2 `<instance, region>`; Atlas cluster in `<region>`.

## Architecture
```
INGESTION (CLI, run once)                              QUERY (FastAPI)
PDFs → parse (PyMuPDF) → clean → chunk → embed ─┐       question
       page by page      headers   600 tok  batch │          │ embed (LRU cache)
                         footers   +15%     100   ▼          ▼
                                   overlap   MongoDB Atlas: chunks ◀─ $vectorSearch ─┐ concurrently
                                             (text, pages, embedding) ◀─ $search ────┤
                                                                         RRF fusion (Python)
                                                                                     │
                                                         confidence gate (abstain) ──┤
                                                                LLM + top-5 passages → answer [1][2]
```

## Design decisions
- **Vectors live in MongoDB**, next to the text and metadata: one system, one query can pre-filter by course and return passage text; nothing to keep in sync with a separate vector DB.
- **Chunking:** ~600 tokens, 15% overlap, paragraph-aware, with *contextual headers* (`title > section`) prepended before embedding. Page ranges tracked per chunk for citations.
- **Idempotent ingestion:** document `_id` = SHA-1 of file bytes; chunk `_id` = `<doc>:<index>`; unchanged chunks reuse their stored embedding (`content_hash`).
- **Hybrid = Reciprocal Rank Fusion in my own code** (`app/retrieval/hybrid.py`): BM25 and cosine scores aren't comparable, ranks are. Each method fetches 2k candidates; the two searches run concurrently (`asyncio.gather`), with keyword search overlapping the query-embedding call.
- **Guardrails:** `/ask` abstains without calling the LLM when the top vector score is below a threshold calibrated on the dev split; the prompt requires citations and an exact refusal string; out-of-range citations are flagged; passages are delimited as untrusted data.
- **No LangChain/LlamaIndex**: the pipeline is small enough to own and explain end to end.

## Evaluation method
- Hand-written questions (`app/evaluate/questions.jsonl`): exact-term, paraphrase, multi-concept, and unanswerable.
- **Gold labels are pages, not chunk IDs**, so one question set is valid across every chunking experiment. A result is a hit if it's from the gold file and its page range overlaps the gold pages.
- **Dev/test split:** weights, `numCandidates` and the abstain threshold are tuned on `dev`; the table above reports `test` only.
- Metrics: Hit@3, MRR@10, per-type Hit@3; abstain accuracy.

```bash
python -m app.evaluate --split test                         # mode comparison
python -m app.evaluate --split dev --calibrate              # abstain threshold
python -m app.evaluate --split dev --modes vector --ncm 5,15,50
python -m app.evaluate --split dev --modes hybrid --weights 1:1,1:2,2:1
```

## Run locally
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env                      # add OPENAI_API_KEY
docker compose up -d atlas                # Atlas Search + Vector Search on your laptop
python -m app.indexes --wait
python -m app.ingest --dir data/pdfs --recursive --doc-type slides   # course = sub-folder name
python scripts/stats.py
uvicorn app.main:app --reload             # http://localhost:8000  (API docs at /docs)
pytest -q
```
Deploying: see [deploy/EC2.md](deploy/EC2.md).

## API
| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | | db + search-index status |
| POST | `/search` | `{q, mode: keyword\|vector\|hybrid, k, course?}` | ranked passages, per-method ranks, `timings_ms` |
| POST | `/ask` | `{q, course?}` | answer, citations, `abstained`, `timings_ms` |
| GET | `/courses` | | distinct courses |

## Limitations
Scanned PDFs are skipped (no OCR); equations and code extract poorly; `lucene.english` keyword search is English-only; the query cache and rate limiter are per process.
