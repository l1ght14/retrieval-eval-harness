"""BM25 and fusion, pinned against small collections where the ranking is checkable by hand."""
from __future__ import annotations

import math

import pytest

from src.lexical import (
    BM25Index,
    idf_dispersion,
    jaccard,
    reciprocal_rank_fusion,
    top_k_from_scores,
    tokenize,
)

DOCS = [
    ("d1", "the quick brown fox jumps over the lazy dog"),
    ("d2", "a quick brown dog barks loudly at the fox"),
    ("d3", "quantum entanglement in particle physics"),
    ("d4", "the lazy dog sleeps all day in the sun"),
]


# ---------------------------------------------------------------- tokenize


def test_tokenize_lowercases_and_splits():
    assert tokenize("The Quick, brown FOX!") == ["quick", "brown", "fox"]


def test_tokenize_drops_stopwords():
    assert "the" not in tokenize("the fox and the dog")
    assert "and" not in tokenize("the fox and the dog")


def test_tokenize_drops_single_characters():
    assert tokenize("a b fox x9") == ["fox", "x9"]


# ---------------------------------------------------------------- BM25 ranking


def test_bm25_ranks_the_document_containing_the_query_term_first():
    index = BM25Index(DOCS)
    ranked = index.top_k_ids("entanglement particle physics", 3)
    assert ranked[0] == "d3"


def test_bm25_finds_the_rare_term_over_the_common_one():
    """'fox' is in 2 of 4 documents, 'entanglement' in 1. The rarer term must dominate."""
    index = BM25Index(DOCS)
    assert index.top_k_ids("entanglement", 1)[0] == "d3"
    scores = index.score("entanglement")
    assert scores[index.doc_ids.index("d3")] > 0


def test_bm25_scores_zero_for_unknown_terms():
    index = BM25Index(DOCS)
    assert list(index.score("helicopter submarine")) == [0.0] * 4


def test_bm25_returns_nothing_for_k_zero():
    assert BM25Index(DOCS).top_k_ids("fox", 0) == []


def test_bm25_top_k_respects_k():
    assert len(BM25Index(DOCS).top_k_ids("the fox dog quantum", 2)) == 2


def test_bm25_top_k_clamps_to_collection_size():
    assert len(BM25Index(DOCS).top_k_ids("fox", 99)) == 4


def test_bm25_top_k_pairs_ids_with_descending_scores():
    top = BM25Index(DOCS).top_k("the lazy dog", 3)
    assert [s for _, s in top] == sorted((s for _, s in top), reverse=True)


def test_bm25_idf_is_higher_for_rarer_terms():
    index = BM25Index(DOCS)
    assert index.idf["entanglement"] > index.idf["dog"]


def test_bm25_matches_a_hand_computed_score():
    """Single document, single query term, so the formula can be evaluated directly.

    n=1, df=1:  IDF = ln(1 + (1 - 1 + 0.5)/(1 + 0.5)) = ln(1 + 1/3) = ln(1.33333)
    f=1, dl=1, avgdl=1: denominator = 1 + 1.2*(1 - 0.75 + 0.75*1) = 1 + 1.2 = 2.2
    numerator = 1 * (1.2 + 1) = 2.2
    saturated tf = 2.2/2.2 = 1.0
    score = 1.0 * IDF
    """
    index = BM25Index([("only", "alpha")])
    got = index.score("alpha")[0]
    expected = math.log(1.0 + 0.5 / 1.5)
    assert got == pytest.approx(expected, rel=1e-9)


def test_bm25_length_normalisation_prefers_the_shorter_document():
    """Same terms, one document twice as long. Length normalisation must score the short one
    higher - the whole reason b is not 0."""
    docs = [("short", "alpha beta"), ("long", "alpha beta " + "filler " * 40)]
    index = BM25Index(docs)
    scores = index.score("alpha beta")
    assert scores[0] > scores[1]


def test_bm25_term_frequency_saturates():
    """A term repeated ten times is more relevant than one appearing once, but not ten times
    more. This is k1's job and it is the property that distinguishes BM25 from tf-idf."""
    docs = [("many", "alpha " * 20), ("once", "alpha")]
    scores = BM25Index(docs).score("alpha")
    assert scores[0] > scores[1]
    assert scores[0] < scores[1] * 20


def test_bm25_rejects_invalid_parameters():
    with pytest.raises(ValueError, match="k1"):
        BM25Index(DOCS, k1=-1)
    with pytest.raises(ValueError, match="b"):
        BM25Index(DOCS, b=1.5)


def test_bm25_handles_an_empty_collection():
    index = BM25Index([])
    assert index.n_docs == 0
    assert index.score("anything").tolist() == []


def test_bm25_handles_documents_with_no_tokens():
    index = BM25Index([("a", ""), ("b", "")])
    assert index.vocabulary_size() == 0
    assert list(index.score("anything")) == [0.0, 0.0]


def test_bm25_vocabulary_grows_with_distinct_terms():
    assert BM25Index(DOCS).vocabulary_size() > 0


# ---------------------------------------------------------------- fusion


def test_rrf_promotes_documents_appearing_in_both_lists():
    fused = dict(reciprocal_rank_fusion([["a", "b"], ["b", "a"]]))
    assert fused["a"] == pytest.approx(fused["b"])  # symmetric here


def test_rrf_favours_higher_ranked_documents():
    fused = dict(reciprocal_rank_fusion([["a", "b", "c"]]))
    assert fused["a"] > fused["b"] > fused["c"]


def test_rrf_rewards_agreement_across_lists():
    """'x' is 2nd in one list only; 'y' is 3rd in both. Agreement should win."""
    fused = dict(reciprocal_rank_fusion([["a", "x"], ["a", "y"], ["a", "y"]]))
    assert fused["y"] > fused["x"]


def test_rrf_weights_scale_a_list():
    single = dict(reciprocal_rank_fusion([["a", "b"]]))
    doubled = dict(reciprocal_rank_fusion([["a", "b"]], weights=[2.0]))
    assert doubled["a"] == pytest.approx(2 * single["a"])


def test_rrf_requires_one_weight_per_list():
    with pytest.raises(ValueError, match="one weight per ranked list"):
        reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0])


def test_rrf_of_no_lists_is_empty():
    assert reciprocal_rank_fusion([]) == []


def test_rrf_is_deterministic_on_ties():
    """Ties broken by id, so the CI gate cannot go flaky on a reordered dict."""
    fused = reciprocal_rank_fusion([["b", "a"]])
    assert [d for d, _ in fused] == ["b", "a"]


# ---------------------------------------------------------------- score helpers


def test_top_k_from_scores_sorts_descending():
    ids = ["a", "b", "c"]
    assert top_k_from_scores(ids, [0.1, 0.9, 0.5], 3) == ["b", "c", "a"]


def test_top_k_from_scores_breaks_ties_by_id():
    assert top_k_from_scores(["z", "a"], [1.0, 1.0], 2) == ["a", "z"]


def test_top_k_from_scores_is_deterministic():
    ids = [f"d{i}" for i in range(50)]
    scores = [1.0] * 50
    assert top_k_from_scores(ids, scores, 10) == top_k_from_scores(ids, scores, 10)


def test_top_k_from_scores_k_zero():
    assert top_k_from_scores(["a"], [1.0], 0) == []


def test_top_k_from_scores_clamps_to_length():
    assert len(top_k_from_scores(["a", "b"], [1.0, 2.0], 99)) == 2


# ---------------------------------------------------------------- misc


def test_jaccard_basic():
    assert jaccard([1, 2], [2, 3]) == pytest.approx(1 / 3)
    assert jaccard([], []) == 1.0
    assert jaccard([1], []) == 0.0


def test_idf_dispersion_is_zero_for_a_uniform_collection():
    # every term in every document -> identical df -> identical idf
    docs = [("a", "x y"), ("b", "x y")]
    assert idf_dispersion(BM25Index(docs).idf) == pytest.approx(0.0, abs=1e-9)


def test_idf_dispersion_of_empty_is_zero():
    assert idf_dispersion({}) == 0.0
