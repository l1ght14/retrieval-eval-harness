"""The configuration space, arranged so that each axis is measured on the arm that can
afford it.

The first draft of this file ran `dense_512`, `hybrid_128` and `hybrid_256_rerank`. That was
wrong on two counts, both visible only after measuring the corpus:

**It was expensive for nothing.** Each distinct chunk size needs its own dense embedding of
the whole corpus. At ~18 chunks/s on 8 CPU threads that is roughly 55-85 minutes per
granularity, so the original three configs cost about 3.4 hours of embedding to vary the
chunk-size axis. But FiQA is already chunk-sized - p50 is 90 words, p90 is 270 - so that axis
barely exists here. Paying three hours to demonstrate that is not a result, it is a
transcription error.

**It split the arms.** Running the chunk axis on the dense arm and the mode axis at a single
chunk size is both cheaper and easier to read: the chunk axis is measured where it is
actually free to measure, and the mode/rerank axis is measured at one fixed granularity so
all three arms share a single cached embedding.

So the design is:

  Chunk-size axis, BM25 over the full corpus (seconds, no embeddings)
      bm25_128 / bm25_256 / bm25_512

  Retrieval-mode axis, all at 256 words sharing one embedding pass (~1 hour, once)
      bm25_256 / dense_256 / hybrid_256 / hybrid_256_rerank

The headline table is the mode axis. The chunk table is a separate finding, and its finding
is that 512 is a no-op here - which is a result worth stating, not a gap to paper over.
"""
from __future__ import annotations

from src.retriever import RetrieverConfig

# --------------------------------------------------------------- mode axis (headline)


def _cfg(**kw) -> RetrieverConfig:
    return RetrieverConfig(**kw)


HEADLINE: dict[str, RetrieverConfig] = {
    "bm25_256": _cfg(
        name="bm25_256", chunk_size=256, mode="bm25",
        notes="lexical baseline, full corpus, no embeddings",
    ),
    "dense_256": _cfg(
        name="dense_256", chunk_size=256, mode="dense",
        notes="dense only, same 256-word chunking as the other arms",
    ),
    "hybrid_256": _cfg(
        name="hybrid_256", chunk_size=256, mode="hybrid",
        notes="BM25 + dense fused by reciprocal rank; isolates fusion from either arm alone",
    ),
    "hybrid_256_rerank": _cfg(
        name="hybrid_256_rerank", chunk_size=256, mode="hybrid", rerank=True,
        rerank_candidates=30,
        notes="hybrid plus a cross-encoder over the top 30; isolates reranking from fusion",
    ),
}

# --------------------------------------------------------------- chunk axis (BM25 only)

CHUNK_AXIS: dict[str, RetrieverConfig] = {
    "bm25_128": _cfg(
        name="bm25_128", chunk_size=128, mode="bm25",
        notes="splits 34% of documents - the aggressive end of the axis",
    ),
    "bm25_256": HEADLINE["bm25_256"],
    "bm25_512": _cfg(
        name="bm25_512", chunk_size=512, mode="bm25",
        notes="splits 2% of documents, so within noise of no chunking at all",
    ),
}

# --------------------------------------------------------------- secondary ablations

# The first run of the mode axis produced a result that the four headline cells could not
# explain: fusing BM25 in *lowered* recall (0.4647 vs dense's 0.4716) and adding a
# cross-encoder lowered it again (0.4440), while costing 3.1 seconds a query.
#
# Two candidate causes, and the headline design conflated them:
#
#   1. BM25 fusion is genuinely harmful on FiQA - plausible, since BM25 is far weaker here
#      and reciprocal rank fusion weights the two lists equally.
#   2. The reranker is fine but was handed a worse candidate list, because the top-30 it saw
#      came from RRF rather than from the dense ranking alone.
#
# These two cells separate them. `dense_256_rerank` is the cross-encoder applied to the dense
# arm's own top-30 - so if it beats dense_256, the reranker was fine and fusion was the
# culprit. `hybrid_256_w03` down-weights BM25 to 0.3, which tests whether fusion stops being
# harmful once it is not given an equal vote.
#
# Together they turn a confusing negative into a 2x2 factorial (dense/hybrid x no-rerank/
# rerank) plus one weighting arm, which is the minimum needed to name the cause.

ABLATIONS: dict[str, RetrieverConfig] = {
    "dense_256_rerank": _cfg(
        name="dense_256_rerank", chunk_size=256, mode="dense", rerank=True,
        rerank_candidates=30,
        notes="cross-encoder over the dense arm's own top-30; isolates reranking from fusion",
    ),
    "hybrid_256_w03": _cfg(
        name="hybrid_256_w03", chunk_size=256, mode="hybrid",
        bm25_weight=0.3, dense_weight=1.0,
        notes="BM25 down-weighted to 0.3 in fusion; tests whether equal weighting was the problem",
    ),
    "hybrid_256_sum": _cfg(
        name="hybrid_256_sum", chunk_size=256, mode="hybrid", doc_aggregation="sum",
        notes="fold chunks to documents by sum of ranks instead of by best chunk",
    ),
    "bm25_256_rrf_k10": _cfg(
        name="bm25_256_rrf_k10", chunk_size=256, mode="hybrid", rrf_k=10,
        notes="sharp RRF, so the top rank dominates fusion harder",
    ),
}

ALL_CONFIGS = {**HEADLINE, **CHUNK_AXIS, **ABLATIONS}


def get(name: str) -> RetrieverConfig:
    if name not in ALL_CONFIGS:
        raise KeyError(f"unknown config {name!r}; have {sorted(ALL_CONFIGS)}")
    return ALL_CONFIGS[name]

