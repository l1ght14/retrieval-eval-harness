"""Retrieval metrics, implemented rather than imported.

This is the heart of the project, so the definitions live where they can be read and
unit-tested. Every function is pinned against a hand-computed fixture in the test suite.

The distinction the whole project rests on: **retrieval metrics and generation metrics are
different questions.** recall@k and nDCG ask whether the right passage made it into the
result list. Faithfulness and answer relevance ask what the model did with it. A system can
score perfectly on one and fail the other, and reporting them together hides which one broke.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

# Rank positions are 1-indexed in the IR literature, which is a recurring source of
# off-by-one errors. Positions here are 1-based; the functions take 0-based lists and
# document the conversion.


def _dcg(gains: Sequence[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def recall_at_k(ranked_ids: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of the relevant documents that appear in the top k.

    The plainest retrieval metric and the one the CI gate is written against. It is
    deliberately insensitive to *where* in the top k a hit lands, which is why it is paired
    with MRR and nDCG rather than used alone: a system that puts the right passage at rank 10
    and one that puts it at rank 1 score identically here.
    """
    if not relevant:
        raise ValueError("recall is undefined for a query with no relevant documents")
    if k <= 0:
        raise ValueError("k must be positive")
    top = set(ranked_ids[:k])
    return len(top & relevant) / len(relevant)


def hit_rate_at_k(ranked_ids: Sequence[str], relevant: set[str], k: int) -> float:
    """1.0 if any relevant document appears in the top k, else 0.0.

    Reported alongside recall@k because the two answer different questions. A query with
    five relevant passages can have recall 0.2 at k=10 and still be a success for a user who
    only needed one of them.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    return 1.0 if set(ranked_ids[:k]) & relevant else 0.0


def reciprocal_rank(ranked_ids: Sequence[str], relevant: set[str]) -> float:
    """1 / rank of the first relevant document, or 0.0 if none appears.

    Rewards putting the answer near the top, which recall@k does not.
    """
    for i, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def mrr(ranked_ids: Sequence[str], relevant: set[str]) -> float:
    """Reciprocal rank for one query. The MRR over a query set is the mean; see `mean_mrr`."""
    return reciprocal_rank(ranked_ids, relevant)


def ndcg_at_k(ranked_ids: Sequence[str], gains: dict[str, float], k: int) -> float:
    """Normalised discounted cumulative gain, for graded relevance.

    `gains` maps document id to a gain, so binary relevance is gains={d: 1.0} and graded
    relevance is gains={d: 3.0}. Normalising by the ideal ordering makes scores comparable
    across queries with different numbers of relevant documents, which is what lets a mean
    over a query set mean anything.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    dcg = _dcg([gains.get(doc_id, 0.0) for doc_id in ranked_ids[:k]])
    ideal = sorted(gains.values(), reverse=True)[:k]
    idcg = _dcg(ideal)
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(ranked_ids: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of the top k that is relevant. Included because users see a ranked list, not
    a set, and low precision means they scroll past the answer."""
    if k <= 0:
        raise ValueError("k must be positive")
    top = ranked_ids[:k]
    return sum(1 for d in top if d in relevant) / len(top) if top else 0.0


def mean(values: Sequence[float]) -> float:
    if not len(values):
        return 0.0
    return float(np.mean(values))


# ------------------------------------------------------------------ aggregates


def aggregate(runs: Sequence[dict]) -> dict:
    """Mean metrics across a query set, plus the per-query rows that produced them.

    `runs` is a sequence of {"ranked_ids": [...], "relevant": [...], "category": str}.
    Returning the rows alongside the means matters: a mean recall of 0.81 is unreadable on
    its own, and the error analysis in the report is only possible because the per-query
    numbers are kept.
    """
    if not runs:
        raise ValueError("no runs to aggregate")
    rows = []
    for r in runs:
        relevant = set(r["relevant"])
        row = {
            "query_id": r["query_id"],
            "category": r.get("category", "all"),
            "recall@5": recall_at_k(r["ranked_ids"], relevant, 5),
            "recall@10": recall_at_k(r["ranked_ids"], relevant, 10),
            "hit@10": hit_rate_at_k(r["ranked_ids"], relevant, 10),
            "mrr": mrr(r["ranked_ids"], relevant),
            "ndcg@10": ndcg_at_k(r["ranked_ids"], {d: 1.0 for d in relevant}, 10),
            "precision@10": precision_at_k(r["ranked_ids"], relevant, 10),
            "n_relevant": len(relevant),
            "first_relevant_rank": next(
                (i for i, d in enumerate(r["ranked_ids"], 1) if d in relevant), None
            ),
        }
        rows.append(row)

    keys = ["recall@5", "recall@10", "hit@10", "mrr", "ndcg@10", "precision@10"]
    out = {k: mean([r[k] for r in rows]) for k in keys}
    out["n_queries"] = len(rows)
    out["per_category"] = {}
    for cat in sorted({r["category"] for r in rows}):
        subset = [r for r in rows if r["category"] == cat]
        out["per_category"][cat] = {
            k: mean([r[k] for r in subset]) for k in keys
        } | {"n_queries": len(subset)}
    return {"summary": out, "per_query": rows}


# ------------------------------------------------------------------ bootstrap


def bootstrap_ci(
    values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 42
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for a mean.

    A retrieval number reported without an interval invites a question that "0.84" cannot
    answer. With ~200 queries the sampling error on a mean recall is a couple of points, and
    a difference between two configurations smaller than that is not a difference. This makes
    the comparison in the report honest about which gaps are real.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("no values to bootstrap")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi


def paired_bootstrap_delta(
    a: Sequence[float], b: Sequence[float], n_boot: int = 2000, seed: int = 42
) -> dict:
    """Bootstrap the paired difference `mean(a) - mean(b)`.

    Paired, because both configurations are scored on the *same* queries. An unpaired test
    would count query difficulty as noise and understate the precision of the comparison.
    Returns the delta and whether the interval excludes zero, which is the only claim the
    report is entitled to make about a config change.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired comparison needs equal shapes, got {a.shape} and {b.shape}")
    delta = a - b
    lo, hi = bootstrap_ci(delta, n_boot=n_boot, seed=seed)
    return {
        "delta": float(delta.mean()),
        "ci_low": lo,
        "ci_high": hi,
        "significant": bool(lo > 0 or hi < 0),
        "n": int(a.size),
    }
