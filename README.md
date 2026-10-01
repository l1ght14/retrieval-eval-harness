# Groundtruth — a retrieval evaluation harness that tells you which number moved

A RAG system's quality is two numbers, not one: **did retrieval find the right passage**, and
**did the model use it correctly**. This project measures the first rigorously on a real
corpus with human relevance labels, keeps the second in separate columns, and ships a CI gate
that fails when retrieval silently degrades.

It also produced a result I did not expect and initially got wrong. See [Findings](#findings).

---

## The result

Golden set: **200 real user questions** from FiQA (BEIR), mean 2.17 human-annotated relevant
documents per query. Corpus: **57,638 documents**, full. Every interval is a 2000-sample
paired bootstrap.

| config | mode | recall@10 | hit@10 | MRR | nDCG@10 | p50 latency |
|---|---|---|---|---|---|---|
| `bm25_256` | BM25 | 0.2949 | 0.4450 | 0.3078 | 0.2499 | **1.2 ms** |
| `dense_256` | dense only | 0.4716 | 0.6500 | 0.4549 | 0.3949 | **22 ms** |
| `dense_256_rerank` | + cross-encoder | **0.4863** | 0.6650 | 0.4501 | **0.3966** | 8810 ms |
| `hybrid_256` | BM25 + dense | 0.4647 | 0.6400 | 0.4351 | 0.3787 | 25 ms |
| `hybrid_256_rerank` | + cross-encoder | 0.4547 | 0.6100 | 0.4194 | 0.3730 | 6543 ms |

Dense beats BM25 by **+0.1767 recall@10, 95% CI [+0.1182, +0.2344]** — the one comparison here
that is not in doubt.

**Nothing added on top of dense produced a change this sample can confirm.** Against
`dense_256` as baseline:

| config | Δ recall@10 | 95% CI | verdict | MRR better/worse |
|---|---|---|---|---|
| `dense_256_rerank` | +0.0148 | [−0.0013, +0.0350] | not significant | 36 / 29 |
| `hybrid_256_rerank` | −0.0169 | [−0.0532, +0.0208] | not significant | 29 / 49 |
| `hybrid_256` | −0.0069 | [−0.0350, +0.0258] | not significant | 24 / 51 |
| `hybrid_256_w03` (BM25 weight 0.3) | +0.0000 | [+0.0000, +0.0000] | identical | 15 / 22 |

Reranking is directionally positive and its win/loss split is 36/29, but at **400× the
latency** (22 ms → 8,810 ms) for a gain the lower bound of the interval cannot separate from
zero. The shipped recommendation is dense-only, and the argument for it is cost, not a claim
that the alternatives were worse.

---

## Findings

**1. Chunk size pointed the opposite way from the usual advice.** 128-word chunks scored 0.2704,
256-word 0.2949, 512-word 0.3085 — and 128-vs-512 is significant. *Smaller chunks were worse.*
FiQA's median document is 90 words, so splitting at 128 words severs 34% of documents from
their own context, and BM25 needs the surrounding words to score a term well. At 512 words only
2% of documents are touched, which makes the best setting on this corpus effectively *no
chunking*. A harness that had only tried chunk sizes familiar from long documents would have
tuned this axis backwards.

**2. Hybrid fusion bought nothing at any weighting.** At equal RRF weights it was slightly
worse than dense alone; at BM25 weight 0.3 it returned a delta of exactly `+0.0000` with a
zero-width interval — the lexical arm changed the top-10 membership on *no* query. It was not
being outvoted, it was contributing nothing.

**3. A bug in my own reranker path was suppressing the effect I was reporting on.** The first
run put `dense_256_rerank` at 0.4470 — *below* dense retrieval — and I wrote that reranking
"made retrieval worse". The cause: the cross-encoder was scoring the retrieved **chunk** while
`context_for` handed the model the whole **document**, so the reranker was judging text the
reader never sees. Fixing it moved the score from 0.4470 to **0.4863** and reversed the sign of
the effect.

The paired significance test did not catch this — the broken version was confidently
non-significant in the wrong direction. The lesson is not "always align your stages"; it is
that a negative result got reported before anyone checked whether the thing under test was
wired up correctly.

**4. I got the headline conclusion wrong first, twice.** The original findings claimed fusion
and reranking were *harmful* on point estimates of −0.007 and −0.025. The paired tests showed
every interval containing zero. Rewritten to "no measurable benefit" — then corrected again
once finding 3 surfaced the bug. `eval/report.py` now generates the expected claim text from
the stored results and fails if the prose does not contain it: the error class this project
exists to prevent, caught by the project itself.

### Wrong along the way

- The reranker text mismatch above — reported a negative result on a mis-wired component.
- An O(n × chunks) scan in the chunker's stats function — would have taken hours on 57k chunks.
- BM25 scored each query by masking a CSR column per term: O(terms × total_postings). Rewritten
  to per-term postings, 0.4 ms/query.
- `top_k_from_scores` broke ties by index on one branch and by id on the other, so a ranking
  depended on corpus read order. A test caught it.
- The reranked and non-reranked `search()` paths returned different chunk ids for the same
  document (`setdefault` vs dict comprehension). Found in review.
- `sum` chunk aggregation claimed to sum *reciprocal* ranks but summed raw ranks — it would
  have favoured the opposite of its documented intent.
- `np.save` appends `.npy` to any path not already ending in it, so an atomic-cache temp file
  named `*.npy.tmp` wrote `*.npy.tmp.npy` and the rename failed. Cost one hour of embedding.
- `LLMClient("mock")` constructed cleanly, reported `available=True`, then raised
  `OpenAIError` on first use.
- Installing `optimum[onnxruntime]` to chase a speedup silently downgraded `huggingface-hub`
  and broke sentence-transformers. It bought 1.3× and cost more than it gave.
- Throughput measured 10 chunks/s in one run and 20 chunks/s in another on the same machine.
  Every throughput claim here carries the measurement that produced it.

---

## Why these choices

**FiQA, not a synthetic corpus.** The queries are real user questions ("What is considered a
business expense on a business trip?") with human relevance labels, so provenance is a property
of the dataset rather than a claim in a README. CC BY-NC 3.0, recorded in the golden set's
provenance block.

**Metrics implemented, not imported.** recall@k, MRR, nDCG, precision@k, BM25 and reciprocal
rank fusion are written out and pinned against hand-computed fixtures. A metric suite that
asserts whatever the implementation produces would pass just as happily against an off-by-one
in the log base.

**Document-level qrels reconciled to chunk-level retrieval.** Scoring chunks against
document-level labels compares `"566392#2"` against `"566392"`, registers a miss, and yields a
plausible wrong number rather than a crash. Every chunk score is folded to its parent document
first; `max` and `sum` folds are both implemented and both are reported.

**Provenance baked into the data.** `golden_v1.meta.json` records the SHA-256 of the raw corpus
and every source file. The CI gate fails if it moves.

**Free-tier LLM, honestly.** Keys read from the environment at call time, never written to
disk. Groq's 30 RPM is respected by pacing, 429s are retried with exponential backoff and
`retry-after`, and every response is cached. With no key at all the harness runs fully offline
on a deterministic mock — retrieval metrics are unaffected, and generation metrics that could
not run report `null`, never a passing default.

---

## Running it

```bash
pip install -r requirements.txt

python scripts/build_golden.py                  # fetch FiQA, mint the versioned golden set
python scripts/embed_corpus.py --status         # check the embedding cache
python scripts/embed_corpus.py --shard 0        # ~10 min per shard, resumable
python scripts/embed_corpus.py --all            # ~1 hour total on 8 CPU threads

python eval/run_eval.py --headline              # the four mode-axis configs
python eval/run_eval.py --chunk-axis            # BM25 chunk study, seconds
python eval/compare.py --baseline bm25_256
python eval/compare.py --baseline dense_256 --out reports/comparisons_vs_dense.json
python eval/report.py                           # RESULTS.md + figures
python eval/gate.py                             # exits 1 on regression

pytest                                          # 174 tests, 90% library coverage
```

Optional: set `GROQ_API_KEY` or `GEMINI_API_KEY` to enable the generation layer's LLM judge.
Without a key the offline mock runs and the judge columns stay `null`.

---

## Structure

```
src/
  corpus.py      BEIR loading, golden-set minting, provenance
  chunking.py    word-window chunking + the measurement that FiQA is already chunk-sized
  lexical.py     BM25 (per-term postings) and reciprocal rank fusion
  dense.py       sentence-transformers index, sharded disk cache, cross-encoder
  retriever.py   BM25 / dense / hybrid, chunk→document reconciliation
  metrics.py     recall@k, MRR, nDCG, bootstrap and paired bootstrap
  generate.py    generation + the retrieval/generation split
  llm.py         cached, retrying, key-safe client for Groq and Gemini
eval/
  configs.py     the configuration space
  run_eval.py    run one or more configs, write per-query rows
  compare.py     paired comparisons from stored rows
  gate.py        CI regression gate
  thresholds.json  floors, derived from the measured baseline
  report.py      RESULTS.md, figures, and the consistency check
scripts/         build_golden.py, embed_corpus.py
data/golden/     golden_v1.jsonl + .meta.json (committed)
```

## Limitations

- **The dense arms are single-model.** `all-MiniLM-L6-v2` was chosen so the whole thing runs on
  a laptop. A stronger embedder would raise these numbers; the table would need re-deriving.
- **Faithfulness is lexical when offline.** Content-word overlap is a proxy. It cannot
  distinguish a correct paraphrase from a wrong one sharing vocabulary. The LLM judge is
  separate and reported separately, never silently substituted.
- **Two-stage retrieval was ruled out but not measured end-to-end.** BM25's recall ceiling here
  is 0.58 at 200, so a BM25→dense pipeline would cap every system below what dense alone
  reaches. That is a strong argument against it on this corpus, not a measurement of one.
- **The consistency check is a presence check**, so a number repeated in the prose and edited
  in one place can slip through. Documented rather than papered over with a markdown parser.
- **200 queries is a modest sample.** Intervals are wide enough that the significant differences
  are the only ones claimed.

## Licence and attribution

Code: MIT. FiQA is CC BY-NC 3.0 (Macedo, Costa & Appert, ESOC 2019), distributed via BEIR; the
corpus is not vendored here — `scripts/build_golden.py` fetches it and records its SHA-256.
