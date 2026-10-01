"""BM25 and reciprocal-rank-fusion, written out so the "hybrid" arm of the comparison is
legible.

`rank_bm25` exists and is fast. It is not used here for two reasons, both of which matter
more than the convenience: this project's entire claim is that retrieval quality is
measurable, so the retrieval function is part of what is being measured and should be
inspectable; and a portfolio project that hides BM25 behind an import cannot show it knows
why the k1 and b parameters are where they are.

BM25 as implemented is Robertson & Zaragoza's formulation:

    score(D, Q) = sum over q in Q of  IDF(q) * ( f(q,D) * (k1 + 1)
                                                 / ( f(q,D) + k1 * (1 - b + b * dl/avgdl) ) )

where f(q,D) is the term frequency in document D, dl the document length, avgdl the mean
document length, and IDF is the standard BM25 IDF with the +1 that keeps it positive for
very common terms. k1 controls term-frequency saturation and b controls length
normalisation; k1=1.2 and b=0.75 are the defaults from the original experiments and are left
at those values so the numbers are comparable to published BM25 results.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")

# Deliberately simple. No stemming and no stop-word list beyond the words BM25's IDF already
# discounts. A heavier pipeline would conflate the BM25 implementation's own choices with the
# configuration being compared, and the whole point of the comparison is to vary one thing.
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have he in is it its of on that the to was were "
    "will with this that these those you your we our they their".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]


class BM25Index:
    """Sparse BM25 index over a fixed collection, stored as per-term postings.

    The layout is a dict of `term -> (doc_indices, weights)`. Scoring a query then touches
    only the postings of the terms the query actually contains, which for FiQA is a few
    thousand entries rather than the ~1M in the whole collection. The alternative - one CSR
    matrix of (doc, term) pairs - has to mask the index column per query term, which makes
    every query cost O(terms x total_postings) and turned a 200-query eval into minutes.
    """

    def __init__(self, documents: Sequence[str], k1: float = 1.2, b: float = 0.75) -> None:
        if k1 < 0:
            raise ValueError("k1 must be non-negative")
        if not 0.0 <= b <= 1.0:
            raise ValueError("b must be in [0, 1]")
        self.k1 = k1
        self.b = b
        self.vocab: dict[str, int] = {}
        self.doc_ids: list[str] = []
        lengths: list[int] = []
        postings: dict[int, list[tuple[int, float]]] = {}

        for doc_idx, (doc_id, text) in enumerate(documents):
            toks = tokenize(text)
            self.doc_ids.append(doc_id)
            lengths.append(len(toks))
            for term, freq in Counter(toks).items():
                col = self.vocab.setdefault(term, len(self.vocab))
                postings.setdefault(col, []).append((doc_idx, float(freq)))

        self.n_docs = len(self.doc_ids)
        self.doc_len = np.asarray(lengths, dtype=np.float64)
        # avgdl guards the degenerate case of an empty collection, where the mean is zero and
        # every document would divide by zero.
        self.avgdl = float(self.doc_len.mean()) if self.n_docs else 0.0
        # The length-normalisation denominator is constant per document, so it can be folded
        # into the term weights once here instead of on every query.
        if self.avgdl > 0:
            self.length_norm = 1.0 - b + b * (self.doc_len / self.avgdl)
        else:
            self.length_norm = np.ones(self.n_docs)

        self._postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.idf: dict[str, float] = {}
        for term, col in self.vocab.items():
            entries = postings[col]
            docs = np.fromiter((d for d, _ in entries), dtype=np.int32, count=len(entries))
            freqs = np.fromiter((f for _, f in entries), dtype=np.float64, count=len(entries))
            df = docs.size
            idf = math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))
            denom = freqs + self.k1 * self.length_norm[docs]
            sat = freqs * (self.k1 + 1.0) / np.maximum(denom, 1e-12)
            self._postings[term] = (docs, sat * idf)
            self.idf[term] = idf

    def score(self, query: str) -> np.ndarray:
        """BM25 score for every document. Documents with no query term score 0."""
        scores = np.zeros(self.n_docs, dtype=np.float64)
        for term in set(tokenize(query)):
            posting = self._postings.get(term)
            if posting is None:
                continue
            docs, weights = posting
            scores[docs] += weights
        return scores

    def top_k(self, query: str, k: int) -> list[tuple[str, float]]:
        """Ids paired with their scores, in descending score order."""
        if k <= 0:
            return []
        scores = self.score(query)
        k = min(k, self.n_docs)
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.lexsort((np.asarray(self.doc_ids)[idx], -scores[idx]))]
        return [(self.doc_ids[i], float(scores[i])) for i in idx]

    def top_k_ids(self, query: str, k: int) -> list[str]:
        if k <= 0:
            return []
        scores = self.score(query)
        k = min(k, self.n_docs)
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [self.doc_ids[i] for i in idx]

    def vocabulary_size(self) -> int:
        return len(self.vocab)


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[str]], k: int = 60, weights: Sequence[float] | None = None
) -> list[tuple[str, float]]:
    """Reciprocal rank fusion.

    Combines ranked lists by summing 1/(k + rank) rather than by comparing raw scores. That
    matters: a BM25 score and a cosine similarity are not on the same scale, so any
    score-weighted fusion hides a decision about relative trust inside a tuning parameter.
    RRF has no such parameter - `k` controls only how sharply the top rank dominates, and 60
    is the value from the original Cormack et al. paper.
    """
    if not ranked_lists:
        return []
    weights = list(weights) if weights else [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("one weight per ranked list")
    fused: dict[str, float] = {}
    for w, lst in zip(weights, ranked_lists):
        for rank, doc_id in enumerate(lst, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + w / (k + rank)
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))


def top_k_from_scores(doc_ids: Sequence[str], scores: np.ndarray, k: int) -> list[str]:
    """Ids of the k highest-scoring documents, ties broken by id for determinism.

    Determinism is not cosmetic here: a non-deterministic tie-break makes the CI regression
    gate flaky, and a flaky gate gets deleted.
    """
    if k <= 0:
        return []
    # Coerced because the callers are inconsistent - the retriever hands over a numpy array,
    # a caller reading results from JSON hands over a list - and unary minus on a list raises.
    scores = np.asarray(scores, dtype=np.float64)
    ids = list(doc_ids)
    id_arr = np.asarray(ids)
    k = min(k, len(ids))
    # One tie-break rule for both branches: by id, never by position. Breaking ties by index
    # makes the result depend on the order the corpus happened to be read in, which silently
    # changes a ranking the moment the same documents arrive in a different order.
    if k == len(ids):
        order = np.lexsort((id_arr, -scores))
        return [ids[i] for i in order]
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.lexsort((id_arr[idx], -scores[idx]))]
    return [ids[i] for i in idx]


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def idf_dispersion(idf: dict[str, float]) -> float:
    """Coefficient of variation of the IDF distribution, for the report.

    A collection whose terms mostly share a near-identical IDF cannot discriminate between
    queries, so a flat distribution is the earliest sign that a config change has broken the
    index rather than merely lowered its accuracy.
    """
    if not idf:
        return 0.0
    values = np.asarray(list(idf.values()), dtype=float)
    return float(values.std() / (values.mean() + 1e-12))
