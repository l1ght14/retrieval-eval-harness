"""Dense retrieval: sentence-transformers embeddings with an on-disk cache.

The expensive part of a dense index is embedding the corpus, not querying it. On this
hardware 57,638 passages take about 2.6 minutes to embed once, and near-zero thereafter. So
embeddings are cached to `.npy` keyed by a hash of the exact model id, the chunk texts, and
the normalisation setting - a hash, not a filename, because a cache that silently survives a
model change is worse than no cache at all: it returns vectors from the previous model and
the scores become meaningless.

The default model is `all-MiniLM-L6-v2` (384 dimensions, ~90 MB). It is small on purpose:
the project's claim is that these measurement practices work on a laptop, and a reader should
be able to reproduce the table without a GPU. If it were replaced with a 7B embedder the
numbers would improve, and the report would need re-deriving.
"""
from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BATCH_SIZE = 256


class EmbeddingCache:
    """Disk-backed store of one matrix of embeddings."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def key(self, model: str, ids: Sequence[str], texts: Sequence[str], normalized: bool) -> str:
        h = hashlib.sha256()
        h.update(model.encode())
        h.update(b"|norm=" + str(normalized).encode())
        for doc_id, text in zip(ids, texts):
            h.update(b"\x00")
            h.update(doc_id.encode("utf-8", "replace"))
            h.update(b"\x01")
            h.update(text.encode("utf-8", "replace"))
        return h.hexdigest()[:32]

    def path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.npy"

    def get(self, key: str) -> np.ndarray | None:
        p = self.path(key)
        if not p.exists():
            return None
        try:
            return np.load(p)
        except (ValueError, OSError):
            # A truncated cache from an interrupted run should degrade to a recompute, not
            # crash the eval.
            return None

    def put(self, key: str, matrix: np.ndarray) -> None:
        # The temp name must still end in .npy: np.save appends .npy to any path that does
        # not already end with it, so a ".npy.tmp" name silently writes ".npy.tmp.npy" and
        # the rename below then fails with a missing file.
        final = self.path(key)
        tmp = final.with_name(final.name + ".tmp.npy")
        np.save(tmp, matrix)
        tmp.replace(final)


class DenseIndex:
    """Brute-force cosine similarity over L2-normalised embeddings.

    No ANN index. At 57k passages an exact matrix product is a few milliseconds on CPU, so
    approximate search would add a recall-versus-latency knob to defend and a way for the
    reported numbers to drift with a different library version. Brute force is the correct
    choice at this corpus size, and saying so is part of the result: the project is about
    measurement quality, and a vector database would make the measurement less reproducible
    without making it fast enough to matter.
    """

    def __init__(
        self,
        ids: Sequence[str],
        matrix: np.ndarray,
        model: str = DEFAULT_MODEL,
    ) -> None:
        if matrix.shape[0] != len(ids):
            raise ValueError(
                f"id/embedding count mismatch: {len(ids)} ids vs {matrix.shape[0]} rows"
            )
        self.ids = list(ids)
        self.matrix = np.ascontiguousarray(matrix, dtype=np.float32)
        self.model = model
        norms = np.linalg.norm(self.matrix, axis=1, keepdims=True)
        self.matrix = self.matrix / np.maximum(norms, 1e-12)
        self._index = {doc_id: i for i, doc_id in enumerate(self.ids)}

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def dim(self) -> int:
        return int(self.matrix.shape[1])

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[str, float]]:
        if k <= 0:
            return []
        q = query_vec / max(float(np.linalg.norm(query_vec)), 1e-12)
        scores = self.matrix @ q
        kk = min(k, len(self.ids))
        idx = np.argpartition(-scores, kk - 1)[:kk]
        idx = idx[np.lexsort((np.asarray(self.ids)[idx], -scores[idx]))]
        return [(self.ids[i], float(scores[i])) for i in idx]



def build_dense_index(
    ids: Sequence[str],
    texts: Sequence[str],
    model_name: str = DEFAULT_MODEL,
    cache: EmbeddingCache | None = None,
    batch_size: int = BATCH_SIZE,
    show_progress: bool = True,
) -> DenseIndex:
    """Embed a corpus, using the disk cache when the inputs are byte-identical."""
    if len(ids) != len(texts):
        raise ValueError(f"id/text count mismatch: {len(ids)} vs {len(texts)}")
    key = cache.key(model_name, ids, texts, normalized=True) if cache else None
    if key:
        hit = cache.get(key)
        if hit is not None and hit.shape[0] == len(ids):
            return DenseIndex(ids, hit, model_name)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    vectors = model.encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )
    if key:
        cache.put(key, vectors)
    return DenseIndex(ids, vectors, model_name)


def embed_queries(texts: Sequence[str], model_name: str = DEFAULT_MODEL) -> np.ndarray:
    """Embed queries with the same model used for the corpus."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return model.encode(
        list(texts), normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
    )


def load_embedded_corpus(
    ids: Sequence[str],
    texts: Sequence[str],
    manifest_path: Path,
    cache_dir: Path,
    model_name: str = DEFAULT_MODEL,
) -> np.ndarray | None:
    """Assemble a sharded embedding cache into one matrix, or None if incomplete.

    Returns None rather than a partial matrix when shards are missing. A matrix built from
    four of six shards has the right row count for those four and the wrong length for the
    rest, and the resulting scores would be plausible enough to publish - so the check is
    strict. There is deliberately no `allow_partial` escape hatch: an earlier version had one,
    and the row-count check below rejected the partial result anyway, so the flag only ever
    misled a caller into expecting behaviour it never delivered.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        return None
    import json

    man = json.loads(manifest_path.read_text(encoding="utf-8"))
    if man.get("model") != model_name:
        return None
    if man.get("n_chunks") != len(ids):
        return None

    cache_dir = Path(cache_dir).resolve()
    parts: list[np.ndarray] = []
    for i in range(man["nshards"]):
        path = (cache_dir / man["shards"][str(i)]["file"]).resolve()
        # The manifest is generated locally, so this is defence in depth rather than a fix
        # for a live vulnerability - but a filename like "../../secrets.npy" in a
        # hand-edited or corrupted manifest should not be able to read outside the cache.
        if not path.is_relative_to(cache_dir):
            return None
        if not path.exists():
            return None
        try:
            parts.append(np.load(path))
        except (ValueError, OSError):
            return None
    if not parts:
        return None

    matrix = np.vstack(parts)
    if matrix.shape[0] != len(ids):
        return None
    return matrix


class CrossEncoderReranker:
    """Optional second-stage reranker over the fused candidate list.

    A bi-encoder embeds query and document independently, so it cannot model their
    interaction. A cross-encoder reads the pair together and is markedly better at ranking,
    but costs one forward pass per candidate, which is why it only ever sees the top-N
    candidates from the first stage.
    """

    def __init__(self, model_name: str = DEFAULT_RERANKER) -> None:
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, max_length=512)
        return self._model

    def rerank(
        self, query: str, candidates: Sequence[str], texts: dict[str, str], top_n: int = 10
    ) -> list[tuple[str, float]]:
        if not candidates:
            return []
        pairs = [(query, texts.get(c, "")) for c in candidates]
        scores = self._load().predict(pairs, batch_size=32, show_progress_bar=False)
        order = np.argsort(-np.asarray(scores))
        return [(candidates[i], float(scores[i])) for i in order[:top_n]]
