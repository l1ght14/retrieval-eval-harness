"""Run one retrieval configuration over the golden set and write its result JSON.

Each config writes `reports/eval_<config>.json` containing the aggregate summary, the
per-query rows, and the paired comparison against the first config. Keeping every config's
raw rows on disk is what lets the report be regenerated later - including the error analysis
and the significance tests - without re-running the retrieval, which is the expensive part.

Run directly:
    python eval/run_eval.py --config hybrid_256_rerank
    python eval/run_eval.py --headline
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.chunking import chunk_stats, split_document  # noqa: E402
from src.corpus import DocumentStore, GoldenSet  # noqa: E402
from src.dense import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_RERANKER,
    CrossEncoderReranker,
    DenseIndex,
    EmbeddingCache,
    build_dense_index,
    embed_queries,
    load_embedded_corpus,
)
from src.metrics import aggregate, bootstrap_ci, paired_bootstrap_delta  # noqa: E402
from src.retriever import HybridRetriever  # noqa: E402
from eval.configs import ABLATIONS, CHUNK_AXIS, HEADLINE, get  # noqa: E402

RAW = ROOT / "data" / "raw"
GOLDEN = ROOT / "data" / "golden" / "golden_v1.jsonl"
REPORTS = ROOT / "reports"
CACHE = ROOT / ".cache" / "embeddings"


def build_retriever(cfg, store: DocumentStore, cache: EmbeddingCache, show: bool = True):
    """Chunk the corpus and construct the retriever for one config.

    Dense vectors come from the sharded cache in `.cache/embeddings` when it is complete, and
    fall back to a fresh encode only when it is not. Every dense config at 256 words reads
    the *same* shards, so the expensive pass is paid once across the whole mode axis rather
    than once per config.
    """
    chunks = []
    for doc_id in store.ids():
        chunks.extend(split_document(doc_id, store.get(doc_id), cfg.chunk_size, cfg.chunk_overlap))

    dense_index = None
    if cfg.mode in ("dense", "hybrid"):
        model_name = cfg.embedding_model or DEFAULT_MODEL
        ids = [c.chunk_id for c in chunks]
        texts = [c.text for c in chunks]
        matrix = load_embedded_corpus(ids, texts, CACHE / "manifest.json", CACHE, model_name)
        if matrix is not None:
            dense_index = DenseIndex(ids, matrix, model_name)
            if show:
                print(f"    dense vectors from sharded cache ({len(ids)} x {matrix.shape[1]})")
        else:
            if show:
                print("    sharded cache incomplete - encoding now (this takes about an hour)")
            dense_index = build_dense_index(
                ids, texts, model_name=model_name, cache=cache, show_progress=show
            )

    reranker = CrossEncoderReranker(DEFAULT_RERANKER) if cfg.rerank else None
    return HybridRetriever(
        chunks, cfg, dense_index=dense_index, reranker=reranker,
        doc_texts={d: store.get(d) for d in store.ids()},
    )


def evaluate(cfg, golden: GoldenSet, store: DocumentStore, k: int = 10, show: bool = True) -> dict:
    cache = EmbeddingCache(CACHE)
    t0 = time.time()
    retriever = build_retriever(cfg, store, cache, show=show)
    build_s = time.time() - t0

    needs_vec = cfg.mode in ("dense", "hybrid")
    if needs_vec:
        qvecs = embed_queries([q.text for q in golden.queries], cfg.embedding_model or DEFAULT_MODEL)
    else:
        qvecs = None

    runs, latencies = [], []
    for i, q in enumerate(golden.queries):
        started = time.time()
        ranked = retriever.retrieve_ids(
            q.text, k=k, query_vec=qvecs[i] if qvecs is not None else None
        )
        latencies.append(time.time() - started)
        runs.append(
            {
                "query_id": q.query_id,
                "ranked_ids": ranked,
                "relevant": q.relevant,
                "category": q.category,
            }
        )
        if show and (i + 1) % 50 == 0:
            print(f"    {i + 1}/{len(golden.queries)} queries", flush=True)

    result = aggregate(runs)
    lat = np.asarray(latencies)
    payload = {
        "config": cfg.to_dict(),
        "k": k,
        "chunk_stats": chunk_stats(retriever.chunks),
        "index_build_seconds": round(build_s, 2),
        "query_latency_ms": {
            "mean": round(float(lat.mean()) * 1000, 1),
            "p50": round(float(np.percentile(lat, 50)) * 1000, 1),
            "p95": round(float(np.percentile(lat, 95)) * 1000, 1),
        },
        "summary": result["summary"],
        "per_query": result["per_query"],
        "retriever": retriever.describe(),
    }

    for metric in ("recall@10", "mrr", "ndcg@10"):
        lo, hi = bootstrap_ci([r[metric] for r in result["per_query"]])
        payload["summary"][f"{metric}_ci"] = [round(lo, 4), round(hi, 4)]
    return payload


def compare(payloads: dict[str, dict], baseline: str) -> dict:
    """Paired comparisons of every config against the baseline, per metric.

    Paired because both configs are scored on the identical 200 queries, so the query
    difficulty cancels and the interval reflects only the configuration change. An
    unpaired interval here would be roughly twice as wide and would hide real differences.
    """
    if baseline not in payloads:
        return {}
    base_rows = {r["query_id"]: r for r in payloads[baseline]["per_query"]}
    out = {"baseline": baseline, "comparisons": {}}
    for name, payload in payloads.items():
        if name == baseline:
            continue
        entry = {}
        for metric in ("recall@10", "mrr", "ndcg@10", "hit@10"):
            b = [base_rows[r["query_id"]][metric] for r in payload["per_query"]]
            a = [r[metric] for r in payload["per_query"]]
            entry[metric] = paired_bootstrap_delta(a, b)
        out["comparisons"][name] = entry
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="single config name")
    ap.add_argument("--headline", action="store_true",
                    help="the mode axis: bm25_256, dense_256, hybrid_256, hybrid_256_rerank")
    ap.add_argument("--chunk-axis", action="store_true",
                    help="BM25-only chunk-size study: 128 / 256 / 512 words, no embeddings")
    ap.add_argument("--ablations", action="store_true", help="the ablation set")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--limit", type=int, help="debug: use only the first N queries")
    args = ap.parse_args()

    if not GOLDEN.exists():
        print(f"missing {GOLDEN}; build it first with scripts/build_golden.py", file=sys.stderr)
        return 2

    golden = GoldenSet.load(GOLDEN)
    if args.limit:
        golden.queries = golden.queries[: args.limit]
    print(f"golden set: {len(golden)} queries, corpus from {GOLDEN.name}")

    store = DocumentStore(RAW)
    print(f"corpus: {len(store)} documents\n")

    names = (
        list(HEADLINE) if args.headline
        else list(CHUNK_AXIS) if args.chunk_axis
        else list(ABLATIONS) if args.ablations
        else [args.config or "hybrid_256"]
    )

    REPORTS.mkdir(parents=True, exist_ok=True)
    payloads = {}
    for name in names:
        cfg = get(name)
        print(f"=== {name} ({cfg.mode}, chunk={cfg.chunk_size}, rerank={cfg.rerank}) ===", flush=True)
        t0 = time.time()
        payload = evaluate(cfg, golden, store, k=args.k)
        payloads[name] = payload
        s = payload["summary"]
        print(
            f"    recall@10 {s['recall@10']:.4f}  mrr {s['mrr']:.4f}  "
            f"ndcg@10 {s['ndcg@10']:.4f}  | {time.time() - t0:.0f}s  "
            f"{payload['query_latency_ms']['p50']}ms p50"
        )
        out = REPORTS / f"eval_{name}.json"
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"    -> {out.relative_to(ROOT)}")

    if len(payloads) > 1:
        baseline = names[0]
        cmp_payload = compare(payloads, baseline)
        (REPORTS / "comparisons.json").write_text(
            json.dumps(cmp_payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\npaired comparisons vs {baseline} -> reports/comparisons.json")
        for name, entry in cmp_payload["comparisons"].items():
            d = entry["recall@10"]
            verdict = "significant" if d["significant"] else "not significant"
            print(
                f"  {name:<20} recall@10 {d['delta']:+.4f} "
                f"[{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]  {verdict}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
