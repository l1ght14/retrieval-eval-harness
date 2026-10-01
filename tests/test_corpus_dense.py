"""Corpus loading, golden-set minting, and the dense index.

Everything here runs against tiny synthetic fixtures written to `tmp_path`, so the suite
needs neither the 48 MB BEIR download nor an embedding model. That is deliberate: these are
the parts where a silent mistake would corrupt every number, and they are cheap to test.

The two regressions pinned here are ones this project actually hit - the `np.save` temp-file
suffix trap, and assembling a matrix from an incomplete set of shards.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from src.corpus import (
    DocumentStore,
    GoldenSet,
    build_golden_set,
    iter_documents,
    iter_jsonl,
    load_qrels,
    load_queries,
    sha256_of,
)
from src.dense import DenseIndex, EmbeddingCache, load_embedded_corpus


# ---------------------------------------------------------------- fixtures

DOCS = {
    "1": "first document about alpha",
    "2": "second document about beta",
    "3": "third document about gamma",
}


@pytest.fixture
def raw(tmp_path):
    """A miniature BEIR directory: the same three files, the same shapes."""
    root = tmp_path / "raw"
    (root / "qrels").mkdir(parents=True)
    with (root / "corpus.jsonl").open("w", encoding="utf-8") as fh:
        for doc_id, text in DOCS.items():
            fh.write(json.dumps({"_id": doc_id, "title": "", "text": text}) + "\n")
    with (root / "queries.jsonl").open("w", encoding="utf-8") as fh:
        for qid, text in (("10", "what is alpha?"), ("11", "what is beta?")):
            fh.write(json.dumps({"_id": qid, "text": text}) + "\n")
    with (root / "qrels" / "test.tsv").open("w", encoding="utf-8") as fh:
        fh.write("query-id\tcorpus-id\tscore\n")
        fh.write("10\t1\t1\n")
        fh.write("10\t2\t1\n")
        fh.write("11\t3\t1\n")
    return root


# ---------------------------------------------------------------- jsonl / qrels


def test_iter_jsonl_skips_blank_lines(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"a":1}\n\n{"a":2}\n', encoding="utf-8")
    assert [b["a"] for b in iter_jsonl(p)] == [1, 2]


def test_load_qrels_keeps_only_positive_scores(raw):
    qrels = load_qrels(raw / "qrels" / "test.tsv")
    assert qrels == {"10": {"1", "2"}, "11": {"3"}}


def test_load_qrels_drops_zero_and_negative_judgements(tmp_path):
    p = tmp_path / "q.tsv"
    p.write_text("query-id\tcorpus-id\tscore\nq\t1\t1\nq\t2\t0\nq\t3\t-1\n", encoding="utf-8")
    assert load_qrels(p) == {"q": {"1"}}


def test_load_queries_returns_id_to_text(raw):
    assert load_queries(raw / "queries.jsonl") == {"10": "what is alpha?", "11": "what is beta?"}


def test_iter_documents_prepends_a_non_empty_title(tmp_path):
    p = tmp_path / "corpus.jsonl"
    p.write_text(
        json.dumps({"_id": "1", "title": "A Title", "text": "body text"}) + "\n"
        + json.dumps({"_id": "2", "title": "", "text": "no title"}) + "\n",
        encoding="utf-8",
    )
    got = dict(iter_documents(tmp_path))
    assert got["1"] == "A Title. body text"
    assert got["2"] == "no title"


def test_iter_documents_respects_a_limit(raw):
    assert len(list(iter_documents(raw, limit=2))) == 2


def test_sha256_is_stable_and_content_dependent(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_text("alpha", encoding="utf-8")
    b.write_text("alpha", encoding="utf-8")
    (tmp_path / "c").write_text("beta", encoding="utf-8")
    assert sha256_of(a) == sha256_of(b)
    assert sha256_of(a) != sha256_of(tmp_path / "c")


# ---------------------------------------------------------------- golden set


def test_golden_set_is_built_from_the_split(raw):
    g = build_golden_set(raw, size=10, seed=1)
    assert {q.query_id for q in g.queries} == {"10", "11"}
    assert g.queries[0].relevant == ["1", "2"]


def test_golden_set_size_is_capped(raw):
    assert len(build_golden_set(raw, size=1, seed=1)) == 1


def test_golden_set_is_deterministic_for_a_fixed_seed(raw):
    a = build_golden_set(raw, size=1, seed=7)
    b = build_golden_set(raw, size=1, seed=7)
    assert [q.query_id for q in a.queries] == [q.query_id for q in b.queries]


def test_golden_set_excludes_queries_with_no_judgement(raw):
    with (raw / "queries.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"_id": "99", "text": "unjudged"}) + "\n")
    g = build_golden_set(raw, size=10, seed=1)
    assert "99" not in {q.query_id for q in g.queries}


def test_max_relevant_excludes_the_query_rather_than_truncating_it(raw):
    """`max_relevant` drops a query wholesale. Truncating instead would keep the query while
    silently discarding the judgements that make it hard, which would flatter every config."""
    kept = build_golden_set(raw, size=10, seed=1, max_relevant=1)
    assert "10" not in {q.query_id for q in kept.queries}   # 2 relevant -> excluded
    assert "11" in {q.query_id for q in kept.queries}       # 1 relevant -> kept

    unfiltered = build_golden_set(raw, size=10, seed=1)
    q2 = next(q for q in unfiltered.queries if q.query_id == "10")
    assert len(q2.relevant) == 2


def test_a_sampled_query_retains_every_relevant_document(raw):
    """Whatever is admitted, its full relevance list survives - recall@k is scored against
    everything a user could have been shown."""
    for doc in ("4", "5"):
        with (raw / "qrels" / "test.tsv").open("a", encoding="utf-8") as fh:
            fh.write(f"11\t{doc}\t1\n")
    q = next(q for q in build_golden_set(raw, size=10, seed=1).queries if q.query_id == "11")
    assert q.relevant == ["3", "4", "5"]
    assert q.n_relevant == 3


def test_golden_set_raises_when_the_filters_exclude_everything(raw):
    with pytest.raises(ValueError, match="no queries"):
        build_golden_set(raw, size=10, min_relevant=99, max_relevant=100)


def test_relevant_ids_sort_numerically_not_lexically(raw):
    with (raw / "qrels" / "test.tsv").open("a", encoding="utf-8") as fh:
        for d in ("2", "10", "3"):
            fh.write(f"10\t{d}\t1\n")
    g = build_golden_set(raw, size=10, seed=1)
    q = next(q for q in g.queries if q.query_id == "10")
    # string sort would give 1, 10, 2, 3; numeric gives 1, 2, 3, 10
    assert q.relevant == ["1", "2", "3", "10"]


def test_golden_set_round_trips_through_disk(raw, tmp_path):
    original = build_golden_set(raw, size=10, seed=3, name="golden_test")
    path = tmp_path / "golden_test.jsonl"
    original.save(path)
    loaded = GoldenSet.load(path)
    assert [q.query_id for q in loaded.queries] == [q.query_id for q in original.queries]
    assert loaded.provenance()["corpus_sha256"] == original.provenance()["corpus_sha256"]
    assert "license" in loaded.provenance()


def test_provenance_reports_the_mean_relevant_count(raw):
    prov = build_golden_set(raw, size=10, seed=1).provenance()
    # query 10 has 2 relevant, query 11 has 1
    assert prov["mean_relevant_per_query"] == pytest.approx(1.5)


def test_empty_golden_set_provenance_does_not_divide_by_zero():
    assert GoldenSet(name="empty", queries=[]).provenance()["mean_relevant_per_query"] == 0.0


def test_golden_set_len_and_iteration(raw):
    g = build_golden_set(raw, size=10, seed=1)
    assert len(g) == 2
    assert len(list(g)) == 2


# ---------------------------------------------------------------- document store


def test_document_store_indexes_and_returns_text(raw):
    store = DocumentStore(raw)
    assert len(store) == 3
    assert "1" in store
    assert store.text("1") == DOCS["1"]
    assert store.ids() == ["1", "2", "3"]


def test_document_store_missing_id_raises_on_text_but_not_get(raw):
    store = DocumentStore(raw)
    assert store.get("nope") == ""
    assert store.get("nope", "fallback") == "fallback"
    with pytest.raises(KeyError, match="unknown document id"):
        store.text("nope")


def test_document_store_texts_preserves_order(raw):
    assert DocumentStore(raw).texts(["3", "1"]) == [DOCS["3"], DOCS["1"]]


def test_document_store_respects_a_limit(raw):
    assert len(DocumentStore(raw, limit=2)) == 2


# ---------------------------------------------------------------- embedding cache


def test_embedding_cache_round_trips(tmp_path):
    cache = EmbeddingCache(tmp_path)
    ids, texts = ["a", "b"], ["one", "two"]
    key = cache.key("model-x", ids, texts, normalized=True)
    assert cache.get(key) is None
    cache.put(key, np.eye(2, dtype=np.float32))
    loaded = cache.get(key)
    assert loaded is not None and loaded.shape == (2, 2)


def test_embedding_cache_put_leaves_no_temp_file(tmp_path):
    """The regression: np.save appends .npy to a temp name that does not end in .npy, writing
    <name>.npy.tmp.npy, and the atomic rename then fails with a missing file."""
    cache = EmbeddingCache(tmp_path)
    cache.put("k", np.zeros((1, 2), dtype=np.float32))
    names = sorted(p.name for p in tmp_path.rglob("*") if p.is_file())
    assert names == ["k.npy"]


def test_embedding_cache_key_depends_on_every_input(tmp_path):
    cache = EmbeddingCache(tmp_path)
    base = cache.key("m", ["a"], ["t"], True)
    assert base != cache.key("other", ["a"], ["t"], True)
    assert base != cache.key("m", ["b"], ["t"], True)
    assert base != cache.key("m", ["a"], ["other"], True)
    assert base != cache.key("m", ["a"], ["t"], False)


def test_corrupt_cache_returns_none_rather_than_raising(tmp_path):
    cache = EmbeddingCache(tmp_path)
    key = cache.key("m", ["a"], ["t"], True)
    cache.path(key).parent.mkdir(parents=True, exist_ok=True)
    cache.path(key).write_text("not an npy file", encoding="utf-8")
    assert cache.get(key) is None


# ---------------------------------------------------------------- sharded loading


def _write_shards(tmp_path, n_shards, rows, cols=4):
    manifest = {"model": "m", "n_chunks": n_shards * rows, "nshards": n_shards,
                "shards": {str(i): {"file": f"k.s{i}.npy", "count": rows} for i in range(n_shards)}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for i in range(n_shards):
        np.save(tmp_path / f"k.s{i}.npy", np.full((rows, cols), float(i + 1), dtype=np.float32))


def test_load_embedded_corpus_assembles_all_shards(tmp_path):
    _write_shards(tmp_path, 3, 5)
    ids = [str(i) for i in range(15)]
    matrix = load_embedded_corpus(ids, ["t"] * 15, tmp_path / "manifest.json", tmp_path, "m")
    assert matrix is not None and matrix.shape == (15, 4)
    # shard order preserved: first 5 rows are shard 0's value
    assert float(matrix[0, 0]) == 1.0
    assert float(matrix[10, 0]) == 3.0


def test_load_embedded_corpus_refuses_an_incomplete_set(tmp_path):
    """A partial matrix would have the wrong row count for the ids and produce plausible
    nonsense, so this returns None rather than the rows it does have."""
    _write_shards(tmp_path, 3, 5)
    (tmp_path / "k.s2.npy").unlink()
    ids = [str(i) for i in range(15)]
    assert load_embedded_corpus(ids, ["t"] * 15, tmp_path / "manifest.json", tmp_path, "m") is None


def test_load_embedded_corpus_returns_none_for_a_model_mismatch(tmp_path):
    _write_shards(tmp_path, 2, 5)
    ids = [str(i) for i in range(10)]
    assert load_embedded_corpus(ids, ["t"] * 10, tmp_path / "manifest.json", tmp_path,
                                "different-model") is None


def test_load_embedded_corpus_returns_none_when_the_manifest_is_absent(tmp_path):
    assert load_embedded_corpus(["a"], ["t"], tmp_path / "missing.json", tmp_path, "m") is None


# ---------------------------------------------------------------- dense index


def test_dense_index_search_finds_the_nearest_vector():
    matrix = np.eye(3, dtype=np.float32)
    index = DenseIndex(["a", "b", "c"], matrix)
    hits = index.search(np.array([0.0, 1.0, 0.0]), k=2)
    assert hits[0][0] == "b"
    assert hits[0][1] == pytest.approx(1.0, abs=1e-5)


def test_dense_index_normalises_unnormalised_input_vectors():
    """Vectors are normalised at index time so a query is one matrix product. Passing
    unnormalised rows here must still give cosine, not dot product."""
    index = DenseIndex(["a"], np.array([[3.0, 4.0]], dtype=np.float32))
    assert index.dim == 2
    assert float(np.linalg.norm(index.matrix[0])) == pytest.approx(1.0, abs=1e-6)
    assert index.search(np.array([0.6, 0.8]), k=1)[0][1] == pytest.approx(1.0, abs=1e-5)


def test_dense_index_rejects_a_length_mismatch():
    with pytest.raises(ValueError, match="id/embedding count mismatch"):
        DenseIndex(["a", "b"], np.eye(3, dtype=np.float32))


def test_dense_index_search_with_k_zero():
    assert DenseIndex(["a"], np.eye(1, dtype=np.float32)).search(np.array([1.0]), 0) == []


def test_dense_index_handles_a_zero_vector_query():
    """Must not divide by zero."""
    hits = DenseIndex(["a", "b"], np.eye(2, dtype=np.float32)).search(np.zeros(2), k=2)
    assert len(hits) == 2
