"""Hybrid retrieval over chunks, reconciled back to documents.

The subtlety this module exists to get right: the golden set's relevance judgements are
keyed by *document* id, while the retriever ranks *chunks*. Scoring chunks directly against
document-level qrels would compare `"566392#2"` against the label `"566392"`, register a
miss, and produce a recall number that is wrong rather than merely low. Every chunk score is
therefore folded to its parent document before metrics are computed.

Two folds are supported, and the choice is reported rather than buried:

- `max`  - a document scores as its best chunk. Standard for passage retrieval, and it is
  what "the user would see the right text somewhere on the page" actually means.
- `sum`  - a document scores as the total evidence across its chunks. Fairer when a query
  needs several passages of one document to answer, and it makes long documents harder,
  which interacts with the chunk-size axis. On FiQA the interaction is small because 89% of
  documents are a single chunk at 256 words.

Fusion is reciprocal rank, not a weighted sum of scores, because a BM25 score and a cosine
similarity are not commensurable. See `lexical.reciprocal_rank_fusion`.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .chunking import Chunk
from .dense import CrossEncoderReranker, DenseIndex
from .lexical import BM25Index, reciprocal_rank_fusion


@dataclass
class RetrieverConfig:
    """One point in the configuration space. Every field here is a reported axis."""

    name: str
    chunk_size: int = 256
    chunk_overlap: int = 0
    mode: str = "hybrid"  # bm25 | dense | hybrid
    fusion: str = "rrf"  # rrf
    rrf_k: int = 60
    doc_aggregation: str = "max"  # max | sum
    bm25_weight: float = 1.0
    dense_weight: float = 1.0
    rerank: bool = False
    rerank_candidates: int = 30
    embedding_model: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if self.mode not in ("bm25", "dense", "hybrid"):
            raise ValueError(f"mode must be bm25|dense|hybrid, got {self.mode!r}")
        if self.doc_aggregation not in ("max", "sum"):
            raise ValueError("doc_aggregation must be max|sum")
        if self.mode == "hybrid" and self.fusion != "rrf":
            raise ValueError("only rrf fusion is implemented; it is the only one without a "
                             "score-scale assumption to tune")

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


class HybridRetriever:
    """Builds a BM25 and/or dense index over a chunk set and answers queries against them."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        config: RetrieverConfig,
        dense_index: DenseIndex | None = None,
        reranker: CrossEncoderReranker | None = None,
        doc_texts: dict[str, str] | None = None,
    ) -> None:
        self.config = config
        self.chunks = list(chunks)
        self.chunk_ids = [c.chunk_id for c in self.chunks]
        self.text_by_chunk = {c.chunk_id: c.text for c in self.chunks}
        self.parent = {c.chunk_id: c.doc_id for c in self.chunks}
        self._chunks_of_doc: dict[str, list[str]] = {}
        for c in self.chunks:
            self._chunks_of_doc.setdefault(c.doc_id, []).append(c.chunk_id)
        # Full document text, for the generation context. Reconstructed by joining the
        # document's chunks in position order, so a retriever built from a chunk list alone
        # still hands the model whole documents rather than the first fragment.
        self.doc_texts = doc_texts or {
            d: " ".join(
                self.text_by_chunk[c] for c in sorted(cids, key=lambda x: x.rsplit("#", 1)[-1])
            )
            for d, cids in self._chunks_of_doc.items()
        }

        self.bm25: BM25Index | None = None
        if config.mode in ("bm25", "hybrid"):
            self.bm25 = BM25Index([(c.chunk_id, c.text) for c in self.chunks])
        self.dense = dense_index if config.mode in ("dense", "hybrid") else None
        if config.mode in ("dense", "hybrid") and self.dense is None:
            raise ValueError("dense or hybrid mode needs a dense index")

        self.reranker = reranker if (config.rerank and reranker) else None

    # ------------------------------------------------------------------ chunk level

    def _rank_chunks(self, query: str, query_vec: np.ndarray | None, k: int) -> list[str]:
        mode = self.config.mode
        depth = max(k, self.config.rerank_candidates if self.config.rerank else k)

        if mode == "bm25":
            return self.bm25.top_k_ids(query, depth)
        if mode == "dense":
            if query_vec is None:
                raise ValueError("dense mode needs an embedded query")
            return [cid for cid, _ in self.dense.search(query_vec, depth)]

        lexical = self.bm25.top_k_ids(query, depth)
        dense = [cid for cid, _ in self.dense.search(query_vec, depth)]
        fused = reciprocal_rank_fusion(
            [lexical, dense],
            k=self.config.rrf_k,
            weights=[self.config.bm25_weight, self.config.dense_weight],
        )
        return [cid for cid, _ in fused[:depth]]

    # ------------------------------------------------------------------ doc level

    def _aggregate(self, ranked_chunks: Sequence[str], k: int) -> list[str]:
        """Fold chunk ranks to document ranks.

        A document inherits the best rank among its chunks, and two chunks of the same
        document never occupy two slots - otherwise a single 3,000-word document would
        occupy the top three results and crowd out other documents, which is not what a user
        experiences.
        """
        if self.config.doc_aggregation == "max":
            best: dict[str, int] = {}
            for rank, cid in enumerate(ranked_chunks):
                doc = self.parent[cid]
                if doc not in best:
                    best[doc] = rank
            return sorted(best, key=lambda d: best[d])[:k]

        # `sum` of reciprocal ranks across the document's chunks, so a document with several
        # individually-weak but mutually-consistent chunks can outrank one strong chunk.
        # Summing *reciprocal* ranks (not raw ranks) is what makes that true: a chunk at
        # rank 2 contributes 0.5 and a chunk at rank 20 contributes 0.05, so a document
        # with two mid-ranked chunks can out-score a document with one lonely chunk near
        # the top. Summing raw ranks would reward exactly the opposite.
        scores: dict[str, float] = {}
        for rank, cid in enumerate(ranked_chunks, start=1):
            doc = self.parent[cid]
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (self.config.rrf_k + rank)
        ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return [d for d, _ in ordered[:k]]

    # ------------------------------------------------------------------ public

    def search(
        self, query: str, k: int = 10, query_vec: np.ndarray | None = None
    ) -> list[tuple[str, str]]:
        """Return up to k (doc_id, best_matching_chunk_id) pairs, best first."""
        chunk_ranked = self._rank_chunks(query, query_vec, k)
        if not chunk_ranked:
            return []

        # Best chunk per document, in rank order. `setdefault` keeps the FIRST chunk, which is
        # the one that earned the document its rank. A plain dict comprehension here would
        # keep the last instead, so the reranked and non-reranked paths would report
        # different chunk ids for the same document.
        best_chunk: dict[str, str] = {}
        for cid in chunk_ranked:
            best_chunk.setdefault(self.parent[cid], cid)

        if self.reranker is not None:
            # The cross-encoder scores whole documents, matching what `context_for` later
            # hands to the model. Scoring it on a 200-word fragment instead would rank the
            # document by text the reader never sees.
            text_by_doc = {d: self.doc_texts.get(d, "") for d in self._aggregate(chunk_ranked, k)}
            docs = self._aggregate(chunk_ranked, self.config.rerank_candidates)
            reranked = self.reranker.rerank(query, docs, text_by_doc, top_n=len(docs))
            doc_order = [d for d, _ in reranked][:k]
            return [(d, best_chunk.get(d, d)) for d in doc_order]

        return [(doc, best_chunk.get(doc, doc)) for doc in self._aggregate(chunk_ranked, k)]

    def retrieve_ids(self, query: str, k: int = 10, query_vec: np.ndarray | None = None) -> list[str]:
        return [d for d, _ in self.search(query, k=k, query_vec=query_vec)]

    def context_for(
        self, doc_ids: Sequence[str], max_chars: int = 4000
    ) -> str:
        """The text handed to the LLM. Assembled from full documents, not the retrieved
        chunk, because a generated answer quoted from a 200-word fragment of a 400-word
        document is a faithfulness artefact rather than a real failure."""
        parts: list[str] = []
        used = 0
        for i, doc in enumerate(doc_ids, start=1):
            text = self.doc_texts.get(doc, "")
            if not text:
                continue
            block = f"[{i}] {text}"
            if used + len(block) > max_chars:
                break
            parts.append(block)
            used += len(block)
        return "\n\n".join(parts)

    def describe(self) -> dict:
        return {
            "config": self.config.to_dict(),
            "n_chunks": len(self.chunks),
            "n_docs": len(self._chunks_of_doc),
            "bm25_vocab": self.bm25.vocabulary_size() if self.bm25 else 0,
            "dense_dim": self.dense.dim if self.dense else 0,
            "reranker": self.reranker.model_name if self.reranker else None,
        }
