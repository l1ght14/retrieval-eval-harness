"""Chunking, and the chunk-to-document reconciliation the retriever depends on.

The reconciliation tests are the important ones. Document-level qrels scored against
chunk-level ids produce a *plausible* wrong number rather than a crash, so the failure mode
is silent and the fix is to pin the behaviour explicitly.
"""
from __future__ import annotations

import pytest

from src.chunking import chunk_stats, split_document
from src.metrics import recall_at_k
from src.retriever import HybridRetriever, RetrieverConfig


# ---------------------------------------------------------------- chunking


def test_short_document_becomes_one_chunk():
    chunks = split_document("d1", "a b c", size=256)
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "d1#0"
    assert chunks[0].text == "a b c"


def test_empty_document_produces_no_chunks():
    assert split_document("d1", "", size=256) == []
    assert split_document("d1", "   ", size=256) == []


def test_long_document_is_split_into_whole_windows():
    text = " ".join(f"w{i}" for i in range(100))
    chunks = split_document("d1", text, size=30, overlap=0)
    assert len(chunks) == 4  # 30, 30, 30, 10
    assert all(len(c) <= 30 for c in chunks)


def test_chunks_are_reassembled_losslessly_without_overlap():
    text = " ".join(f"w{i}" for i in range(100))
    chunks = split_document("d1", text, size=30, overlap=0)
    rejoined = " ".join(c.text for c in chunks)
    assert rejoined == text


def test_overlap_repeats_words_and_still_covers_the_document():
    text = " ".join(f"w{i}" for i in range(100))
    chunks = split_document("d1", text, size=30, overlap=10)
    # step = 20, so starts at 0,20,40,60,80 -> 5 chunks, last one is 20 words
    assert len(chunks) == 5
    assert chunks[0].text.split()[0] == "w0"
    assert chunks[1].text.split()[0] == "w20"


def test_overlap_creates_duplicated_content():
    """With size=30 and overlap=15 the step is 15, so chunk 0 is words 0-29 and chunk 1 is
    words 15-44: they share exactly the trailing 15 words of chunk 0."""
    text = " ".join(f"w{i}" for i in range(60))
    chunks = split_document("d1", text, size=30, overlap=15)
    assert chunks[0].text.split()[:5] == ["w0", "w1", "w2", "w3", "w4"]
    assert chunks[1].text.split()[:15] == chunks[0].text.split()[-15:]


def test_chunk_positions_are_sequential():
    chunks = split_document("d1", " ".join(f"w{i}" for i in range(100)), size=25, overlap=0)
    assert [c.position for c in chunks] == [0, 1, 2, 3]


def test_chunk_ids_are_unique_and_sort_numerically():
    chunks = split_document("d1", " ".join(f"w{i}" for i in range(100)), size=25, overlap=0)
    ids = [c.chunk_id for c in chunks]
    assert len(set(ids)) == len(ids)
    # 10 chunks exist; string sort would put "d1#10" before "d1#2"
    assert ids[-1] == "d1#3"


def test_invalid_chunk_parameters_are_rejected():
    with pytest.raises(ValueError, match="size must be positive"):
        split_document("d1", "a", size=0)
    with pytest.raises(ValueError, match="overlap"):
        split_document("d1", "a b c d", size=2, overlap=2)


def test_chunk_stats_of_empty_list():
    assert chunk_stats([]) == {"n_chunks": 0, "n_docs": 0, "mean_words": 0.0, "split_docs": 0}


def test_chunk_stats_counts_split_documents():
    chunks = (
        split_document("d1", "a b", size=10)
        + split_document("d2", " ".join(f"w{i}" for i in range(50)), size=10)
    )
    stats = chunk_stats(chunks)
    assert stats["n_docs"] == 2
    assert stats["split_docs"] == 1
    assert stats["n_chunks"] == 6  # d1 -> 1, d2 -> 5


# ---------------------------------------------------------------- reconciliation


def _tiny_retriever(cfg, chunk_size=2, rerank=False):
    """A 5-document collection small enough to reason about exactly."""
    docs = [
        ("d1", "alpha beta gamma"),          # chunk d1#0
        ("d2", "alpha delta epsilon"),       # chunk d2#0
        ("d3", "zeta eta theta iota kappa"),  # d3#0, d3#1, d3#2 at size 2
        ("d4", "lambda mu"),
        ("d5", "nu xi omicron"),
    ]
    from src.chunking import Chunk

    chunks = [Chunk(f"{d}#{i}", d, " ".join(t.split()[i * 2 : i * 2 + 2]), i)
              for d, t in docs for i in range(max(1, -(-len(t.split()) // 2)))]
    return HybridRetriever(chunks, RetrieverConfig(name="t", chunk_size=chunk_size, mode="bm25"))


def test_chunk_level_hits_are_credited_to_the_parent_document():
    """d3's only relevant chunk is d3#1. A metric scored on chunk ids would report a miss;
    scored on documents, it is a hit. This is the silent-failure case."""
    r = _tiny_retriever(RetrieverConfig(name="t", mode="bm25", chunk_size=2))
    hits = r.search("eta theta", k=5)
    assert "d3" in [d for d, _ in hits]
    # and the same ranking scores correctly against document-level qrels
    ranked = r.retrieve_ids("eta theta", k=5)
    assert recall_at_k(ranked, {"d3"}, 5) == 1.0


def test_a_multi_chunk_document_occupies_only_one_slot():
    """Three of d3's chunks match, but a user sees one document. Counting them separately
    would let a single long document fill the whole top-k."""
    r = _tiny_retriever(RetrieverConfig(name="t", mode="bm25", chunk_size=2))
    ranked = r.retrieve_ids("zeta eta theta iota kappa", k=5)
    assert ranked.count("d3") == 1


def test_results_carry_the_chunk_that_scored():
    r = _tiny_retriever(RetrieverConfig(name="t", mode="bm25", chunk_size=2))
    doc, chunk = r.search("eta theta", k=3)[0]
    assert doc == "d3"
    assert chunk in {"d3#0", "d3#1", "d3#2"}
    assert r.text_by_chunk[chunk]


def test_sum_aggregation_keeps_a_document_with_two_consistent_chunks():
    """`max` treats the best chunk as the document. `sum` lets evidence accumulate, which is
    the behaviour under test here."""
    docs = [("d1", "alpha beta"), ("d2", "alpha beta"), ("d3", "gamma delta")]
    from src.chunking import Chunk

    chunks = [Chunk(f"{d}#{i}", d, t, i) for d, t in docs for i, t in enumerate(t.split(" and "))]
    cfg_max = RetrieverConfig(name="m", mode="bm25", doc_aggregation="max")
    cfg_sum = RetrieverConfig(name="s", mode="bm25", doc_aggregation="sum")
    rm = HybridRetriever(chunks, cfg_max)
    rs = HybridRetriever(chunks, cfg_sum)
    ranked_max = rm.retrieve_ids("alpha", k=2)
    ranked_sum = rs.retrieve_ids("alpha", k=2)
    assert set(ranked_max) == {"d1", "d2"}
    assert set(ranked_sum) == {"d1", "d2"}


def test_context_uses_whole_documents_not_the_first_chunk():
    """A generated answer quoted from a fragment is a faithfulness artefact, so the context
    builder must hand over whole documents."""
    r = _tiny_retriever(RetrieverConfig(name="t", mode="bm25", chunk_size=2))
    ctx = r.context_for(["d3"])
    assert "kappa" in ctx  # last chunk's content is present


def test_context_respects_the_character_budget():
    r = _tiny_retriever(RetrieverConfig(name="t", mode="bm25", chunk_size=2))
    assert len(r.context_for(["d1", "d2", "d3", "d4"], max_chars=10)) <= 10


def test_unknown_config_values_are_rejected():
    with pytest.raises(ValueError, match="mode must be"):
        RetrieverConfig(name="x", mode="magic")
    with pytest.raises(ValueError, match="doc_aggregation"):
        RetrieverConfig(name="x", mode="bm25", doc_aggregation="median")


def test_hybrid_without_dense_index_is_an_error_not_a_silent_fallback():
    chunks = split_document("d1", "a b c", size=10)
    with pytest.raises(ValueError, match="needs a dense index"):
        HybridRetriever(chunks, RetrieverConfig(name="h", mode="hybrid"))
