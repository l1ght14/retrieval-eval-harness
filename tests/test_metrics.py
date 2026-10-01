"""Metrics pinned against hand-computed values.

Every expected number here is worked out in the docstring rather than copied from a run of
this code. A metric suite that asserts whatever the implementation happens to produce is
worthless: it would pass just as happily against an off-by-one in the log base. These
fixtures are the definition of "correct" for the rest of the project.
"""
from __future__ import annotations

import math

import pytest

from src.metrics import (
    aggregate,
    bootstrap_ci,
    hit_rate_at_k,
    mean,
    mrr,
    ndcg_at_k,
    paired_bootstrap_delta,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


# ---------------------------------------------------------------- recall


def test_recall_all_relevant_in_top_k():
    # ranked a..e, relevant = {c, e}; both inside k=5
    assert recall_at_k(["a", "b", "c", "d", "e"], {"c", "e"}, 5) == 1.0


def test_recall_none_in_top_k():
    # only 'z' is relevant and it is not in the top 2
    assert recall_at_k(["a", "b", "c"], {"z"}, 2) == 0.0


def test_recall_partial():
    # relevant = {c, z}; only 'c' is retrieved -> half the relevant set found
    assert recall_at_k(["a", "b", "c"], {"c", "z"}, 3) == 0.5


def test_recall_is_insensitive_to_rank_within_k():
    """The reason recall@k is never reported alone.

    A hit at rank 1 and a hit at rank 5 score identically. A user does not experience them
    identically, which is why MRR and nDCG exist alongside.
    """
    assert recall_at_k(["c", "a", "b"], {"c"}, 5) == recall_at_k(["a", "b", "c"], {"c"}, 5) == 1.0


def test_recall_rejects_empty_relevant_set():
    """Undefined, not zero. A query with no relevant document cannot have a recall, and
    silently scoring it 0.0 would drag every mean down for no reason."""
    with pytest.raises(ValueError, match="no relevant documents"):
        recall_at_k(["a"], set(), 1)


def test_recall_rejects_nonpositive_k():
    with pytest.raises(ValueError, match="k must be positive"):
        recall_at_k(["a"], {"a"}, 0)


# ---------------------------------------------------------------- hit rate


def test_hit_rate_is_binary_and_differs_from_recall():
    runs = {"query_id": "q", "ranked_ids": ["a", "b", "c", "d"], "relevant": ["d", "e"]}
    summary = aggregate([runs])["summary"]
    # recall@10 = 1 of 2 found; hit@10 = 1 because the user found something
    assert summary["recall@10"] == 0.5
    assert summary["hit@10"] == 1.0


def test_hit_rate_zero_when_nothing_relevant_retrieved():
    assert hit_rate_at_k(["a", "b"], {"z"}, 2) == 0.0


# ---------------------------------------------------------------- reciprocal rank


def test_reciprocal_rank_of_first_hit():
    # 'c' sits at position 3 -> 1/3
    assert reciprocal_rank(["a", "b", "c"], {"c"}) == pytest.approx(1 / 3)


def test_reciprocal_rank_takes_the_earliest_hit_only():
    """Two hits, but the score comes from the first. Otherwise a doc appearing at rank 1 and
    rank 9 would score better than the same doc appearing only at rank 1."""
    assert reciprocal_rank(["c", "x", "y", "c"], {"c"}) == pytest.approx(1.0)


def test_reciprocal_rank_zero_when_absent():
    assert reciprocal_rank(["a", "b"], {"z"}) == 0.0
    assert mrr(["a", "b"], {"z"}) == 0.0


# ---------------------------------------------------------------- nDCG


def test_ndcg_perfect_order_scores_one():
    gains = {"d1": 3.0, "d2": 1.0}
    assert ndcg_at_k(["d1", "d2"], gains, 2) == pytest.approx(1.0)


def test_ndcg_reversed_order_matches_hand_computation():
    """gains {d1:3, d2:1}, ranked [d2, d1], k=2.

    DCG  = 1/log2(2) + 3/log2(3) = 1 + 3/1.5849625 = 2.8927893
    IDCG = 3/log2(2) + 1/log2(3) = 3 + 0.6309298   = 3.6309298
    nDCG = 2.8927893 / 3.6309298 = 0.7967076
    """
    gains = {"d1": 3.0, "d2": 1.0}
    expected = (1 / math.log2(2) + 3 / math.log2(3)) / (3 / math.log2(2) + 1 / math.log2(3))
    assert ndcg_at_k(["d2", "d1"], gains, 2) == pytest.approx(0.7967076, abs=1e-5)
    assert expected == pytest.approx(0.7967076, abs=1e-5)


def test_ndcg_ignores_order_when_gains_are_equal():
    """A real and slightly counter-intuitive property, so it gets pinned.

    With every relevant document worth the same, DCG is symmetric under reversal - the
    discounts are position-based and the gains are interchangeable. nDCG only rewards
    ordering when relevance is graded, which is why binary-goldensets still benefit from
    MRR for top-heavy questions.
    """
    gains = {"d1": 1.0, "d2": 1.0}
    assert ndcg_at_k(["d1", "d2"], gains, 2) == pytest.approx(
        ndcg_at_k(["d2", "d1"], gains, 2)
    )


def test_ndcg_zero_when_nothing_relevant_appears():
    assert ndcg_at_k(["a", "b"], {"z": 1.0}, 2) == 0.0


def test_ndcg_with_no_relevant_gains_is_zero_not_error():
    """Binary helper for a query whose relevant set is empty at this k. Unlike recall@k this
    degrades to 0.0 rather than raising, because aggregate() computes it for every query."""
    assert ndcg_at_k(["a"], {}, 1) == 0.0


# ---------------------------------------------------------------- precision


def test_precision_counts_relevant_in_the_visible_window():
    assert precision_at_k(["a", "x", "c"], {"a", "c"}, 3) == pytest.approx(2 / 3)


def test_precision_of_empty_top_k_is_zero():
    assert precision_at_k([], {"a"}, 5) == 0.0


# ---------------------------------------------------------------- aggregate


def _run(qid, ranked, relevant, category="all"):
    return {"query_id": qid, "ranked_ids": ranked, "relevant": relevant, "category": category}


def test_aggregate_reports_first_relevant_rank():
    rows = aggregate([_run("q1", ["a", "b", "c"], ["c"])])["per_query"]
    assert rows[0]["first_relevant_rank"] == 3


def test_aggregate_first_relevant_rank_is_none_when_absent():
    rows = aggregate([_run("q1", ["a", "b"], ["z"])])["per_query"]
    assert rows[0]["first_relevant_rank"] is None


def test_aggregate_breaks_down_by_category():
    result = aggregate([
        _run("q1", ["a"], ["a"], category="easy"),
        _run("q2", ["x"], ["y"], category="hard"),
    ])
    per_cat = result["summary"]["per_category"]
    assert per_cat["easy"]["hit@10"] == 1.0
    assert per_cat["hard"]["hit@10"] == 0.0
    assert per_cat["easy"]["n_queries"] == 1


def test_aggregate_rejects_empty_input():
    with pytest.raises(ValueError, match="no runs"):
        aggregate([])


def test_mean_of_empty_is_zero():
    assert mean([]) == 0.0


# ---------------------------------------------------------------- bootstrap


def test_bootstrap_ci_brackets_the_mean():
    values = [0.8, 0.9, 0.85, 0.95, 0.88, 0.92, 0.87, 0.93]
    lo, hi = bootstrap_ci(values, n_boot=500, seed=7)
    assert lo < mean(values) < hi
    assert 0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0


def test_bootstrap_is_deterministic_for_a_fixed_seed():
    values = [0.1, 0.5, 0.9, 0.3, 0.7]
    assert bootstrap_ci(values, seed=1) == bootstrap_ci(values, seed=1)


def test_bootstrap_of_constant_values_is_degenerate():
    lo, hi = bootstrap_ci([0.5] * 20, n_boot=200)
    assert lo == hi == pytest.approx(0.5)


def test_paired_delta_detects_a_real_difference():
    base = [0.50, 0.52, 0.48, 0.51, 0.49, 0.50, 0.53, 0.47]
    better = [b + 0.10 for b in base]
    out = paired_bootstrap_delta(better, base, n_boot=500)
    assert out["delta"] == pytest.approx(0.10, abs=1e-9)
    assert out["significant"] is True
    assert out["ci_low"] > 0


def test_paired_delta_finds_nothing_in_noise():
    """Two configs that differ by rounding must not be reported as a change."""
    base = [0.50, 0.52, 0.48, 0.51, 0.49, 0.50, 0.53, 0.47]
    same = [0.501, 0.519, 0.481, 0.509, 0.491, 0.502, 0.529, 0.471]
    out = paired_bootstrap_delta(same, base, n_boot=500)
    assert out["significant"] is False


def test_paired_delta_requires_equal_lengths():
    with pytest.raises(ValueError, match="equal shapes"):
        paired_bootstrap_delta([0.1, 0.2], [0.1])
