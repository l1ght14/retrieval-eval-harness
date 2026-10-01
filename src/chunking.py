"""Document chunking, and an honest note about how little it does on this corpus.

Chunking exists so a passage larger than the model's useful context is not retrieved whole.
On FiQA it is close to a no-op, and the report says so rather than implying the axis was
worth tuning. Measured over the 57,638 documents:

    words:  p50 = 90   p75 = 159   p90 = 270   p95 = 368   p99 = 641   max = 2973
    docs split at 128 words: 34.0%
    docs split at 256 words: 11.1%
    docs split at 512 words:  2.0%

A 512-word chunk therefore leaves 98% of the corpus untouched, and any gain it appears to
produce over a smaller window is measurement noise, not a finding. The comparison is kept
because "the knob does nothing here, and here is the measurement that says so" is a result
the project needs; what it must not do is dress 512 up as a tuned choice.

Splitting is on word boundaries rather than sentences. Sentence-aware splitting would read
better in isolation, but FiQA posts are frequently one long run of text, and a sentence
splitter that fragments those produces chunks ending mid-clause - which hurts both the
embedding and any human reading the retrieved context.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    position: int

    def __len__(self) -> int:
        return len(self.text.split())


def split_document(
    doc_id: str, text: str, size: int = 256, overlap: int = 0
) -> list[Chunk]:
    """Split one document into overlapping word windows.

    `overlap` is in words and defaults to 0. Overlap is the usual fix for an answer being cut
    in half, but it duplicates content, which can make the same passage look like two
    independent hits in a fused ranking. FiQA's documents are short enough that the default of
    0 is defensible, so the parameter stays available and unused rather than being tuned
    against a metric that cannot resolve the effect.
    """
    if size <= 0:
        raise ValueError("chunk size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError(f"overlap must be in [0, {size})")
    words = text.split()
    if not words:
        return []
    if len(words) <= size:
        return [Chunk(f"{doc_id}#0", doc_id, text, 0)]

    step = size - overlap
    chunks: list[Chunk] = []
    for i, start in enumerate(range(0, len(words), step)):
        window = words[start : start + size]
        if not window:
            break
        chunks.append(Chunk(f"{doc_id}#{i}", doc_id, " ".join(window), i))
        if start + size >= len(words):
            break
    return chunks



def chunk_stats(chunks: list[Chunk]) -> dict:
    """Descriptive stats, reported next to the accuracy numbers.

    A config that raises recall@10 while producing 3x the chunks has not improved retrieval;
    it has made the index bigger and the reranker's job harder, and that trade only shows up
    if the chunk counts are in the report.
    """
    if not chunks:
        return {"n_chunks": 0, "n_docs": 0, "mean_words": 0.0, "split_docs": 0}
    # A Counter rather than a nested scan: counting chunks per doc the obvious way is
    # O(docs x chunks), which is hours on a 57k-document corpus.
    per_doc = Counter(c.doc_id for c in chunks)
    return {
        "n_chunks": len(chunks),
        "n_docs": len(per_doc),
        "mean_words": round(sum(len(c) for c in chunks) / len(chunks), 1),
        "split_docs": sum(1 for n in per_doc.values() if n > 1),
        "chunks_per_doc": round(len(chunks) / len(per_doc), 2),
    }
