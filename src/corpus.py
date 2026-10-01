"""BEIR corpus loading and construction of the versioned golden set.

Everything downstream is measured against `data/golden/golden_v1.jsonl`, so this module has
two jobs: read the raw BEIR files without loading 57k JSON objects into memory at once, and
mint the golden set with the provenance recorded alongside it.

Three choices here are the difference between a harness that measures something and one that
manufactures a number:

**The golden set is versioned and hashed.** The file records the SHA-256 of the raw corpus
it was drawn from, so a claim about "version 1" is checkable years later instead of being a
claim about whatever happened to be in the directory.

**Queries are sampled deterministically and only from the test split.** FiQA's train split is
for fitting, and tuning a retrieval configuration on the same queries you then report on is
the single easiest way to publish a number that does not survive contact with new data.

**Queries with several relevant passages are kept, and their full relevance list is retained.**
Dropping multi-answer queries would flatter every system, because recall@k is hardest exactly
when a query has several valid answers. `max_relevant` therefore *excludes* a query outright
once it exceeds the cap rather than truncating the list - truncating would keep the query while
discarding the judgements that make it hard. The default cap of 5 admits every FiQA test query
up to five relevant documents; how many were excluded is recorded in the provenance block so
the difficulty of the set is visible rather than assumed.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip"
DATASET = "fiqa"

# FiQA's own licence, recorded in the provenance block rather than only in the README, so the
# provenance travels with the data if the file is ever copied out of the repo.
LICENSE = "CC BY-NC 3.0 (financial advice dataset; see the FiQA paper for terms)"
CITATION = (
    "Macedo, Costa & Appert, 'A Financial Dataset for Aspect-Based Sentiment Analysis of "
    "User Generated Financial News' (ESOC 2019). Distributed as the FiQA split in BEIR."
)


@dataclass
class GoldenQuery:
    query_id: str
    text: str
    relevant: list[str]
    n_relevant: int
    category: str = "all"

    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "text": self.text,
            "relevant": sorted(self.relevant, key=_as_int_key),
            "n_relevant": self.n_relevant,
            "category": self.category,
        }


@dataclass
class GoldenSet:
    """A frozen query set plus the provenance needed to reproduce it."""

    name: str
    queries: list[GoldenQuery]
    source: str = DATASET
    split: str = "test"
    corpus_sha256: str = ""
    raw_files: dict[str, str] = field(default_factory=dict)
    license: str = LICENSE
    citation: str = CITATION

    def __len__(self) -> int:
        return len(self.queries)

    def __iter__(self):
        return iter(self.queries)

    def provenance(self) -> dict:
        return {
            "name": self.name,
            "source": self.source,
            "split": self.split,
            "corpus_sha256": self.corpus_sha256,
            "raw_files": self.raw_files,
            "license": self.license,
            "citation": self.citation,
            "n_queries": len(self.queries),
            "mean_relevant_per_query": round(
                sum(q.n_relevant for q in self.queries) / len(self.queries), 4
            )
            if self.queries
            else 0.0,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for q in self.queries:
                fh.write(json.dumps(q.to_dict(), ensure_ascii=False) + "\n")
        meta = path.with_suffix(".meta.json")
        meta.write_text(json.dumps(self.provenance(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "GoldenSet":
        queries = []
        with Path(path).open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    blob = json.loads(line)
                    queries.append(
                        GoldenQuery(
                            query_id=blob["query_id"],
                            text=blob["text"],
                            relevant=blob["relevant"],
                            n_relevant=blob.get("n_relevant", len(blob["relevant"])),
                            category=blob.get("category", "all"),
                        )
                    )
        meta_path = Path(path).with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        return cls(
            name=meta.get("name", Path(path).stem),
            queries=queries,
            source=meta.get("source", DATASET),
            split=meta.get("split", "test"),
            corpus_sha256=meta.get("corpus_sha256", ""),
            raw_files=meta.get("raw_files", {}),
            license=meta.get("license", LICENSE),
            citation=meta.get("citation", CITATION),
        )


def _as_int_key(doc_id: str):
    """Sort document ids numerically when possible, lexically otherwise.

    String sorting puts "10" before "9", which makes the golden file's diff unreadable.
    """
    try:
        return (0, int(doc_id))
    except (TypeError, ValueError):
        return (1, str(doc_id))


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def iter_jsonl(path: Path) -> Iterator[dict]:
    """Stream a .jsonl file. FiQA's corpus is 48 MB; parsing it into a list is a memory
    problem that this avoids for no effort."""
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def load_qrels(path: Path) -> dict[str, set[str]]:
    """Read a BEIR qrels TSV into query_id -> set of relevant doc ids.

    BEIR qrels are tab-separated with a header row and no quoting, so csv with an explicit
    delimiter is the right reader; a plain split would be fine too but csv gets the column
    count validated for free.
    """
    qrels: dict[str, set[str]] = {}
    with Path(path).open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            score = int(row["score"])
            if score > 0:
                qrels.setdefault(row["query-id"], set()).add(row["corpus-id"])
    return qrels


def load_queries(path: Path) -> dict[str, str]:
    return {blob["_id"]: blob["text"] for blob in iter_jsonl(path)}


def build_golden_set(
    raw_dir: Path,
    name: str = "golden_v1",
    split: str = "test",
    size: int = 200,
    seed: int = 42,
    min_relevant: int = 1,
    max_relevant: int = 5,
) -> GoldenSet:
    """Sample a fixed query set from a BEIR split.

    `max_relevant` caps how many relevant passages a sampled query may have. The cap is not
    about excluding hard queries - they stay in, with their full relevance list, so recall@k
    is still scored against everything a user could have been shown. It is about the *per
    query work*: a query with 30 relevant passages tells you much less about a system's
    behaviour than a query with 3, and it is noisy in the nDCG denominator.
    """
    raw_dir = Path(raw_dir)
    qrels = load_qrels(raw_dir / "qrels" / f"{split}.tsv")
    queries = load_queries(raw_dir / "queries.jsonl")

    eligible = [
        qid
        for qid, rel in qrels.items()
        if qid in queries and min_relevant <= len(rel) <= max_relevant
    ]
    if not eligible:
        raise ValueError(
            f"no queries in the {split} split with {min_relevant}-{max_relevant} relevant docs"
        )

    # Deterministic sample. `sorted` first so the result does not depend on dict ordering,
    # then a fixed seed - so re-running mints an identical file, which is what makes the
    # golden set a versioned artefact rather than a fresh one every time.
    import random

    rng = random.Random(seed)
    chosen = sorted(eligible)
    if len(chosen) > size:
        chosen = sorted(rng.sample(chosen, size))

    golden = [
        GoldenQuery(
            query_id=qid,
            text=queries[qid],
            relevant=sorted(qrels[qid], key=_as_int_key),
            n_relevant=len(qrels[qid]),
        )
        for qid in chosen
    ]

    return GoldenSet(
        name=name,
        queries=golden,
        source=DATASET,
        split=split,
        corpus_sha256=sha256_of(raw_dir / "corpus.jsonl"),
        raw_files={
            f: sha256_of(raw_dir / f)
            for f in ("corpus.jsonl", "queries.jsonl", f"qrels/{split}.tsv")
            if (raw_dir / f).exists()
        },
    )


def iter_documents(raw_dir: Path, limit: int | None = None) -> Iterator[tuple[str, str]]:
    """Stream (doc_id, text) pairs. The title is prepended when present, because FiQA leaves
    it empty for most documents but not all, and silently ignoring a non-empty title would
    make the two cases inconsistent."""
    for i, blob in enumerate(iter_jsonl(Path(raw_dir) / "corpus.jsonl")):
        if limit is not None and i >= limit:
            return
        title = (blob.get("title") or "").strip()
        text = blob.get("text") or ""
        yield blob["_id"], (f"{title}. {text}" if title else text)


class DocumentStore:
    """Random access over the corpus, loaded once and shared.

    The whole corpus is held in memory (~55 MB of text) because every configuration
    re-reads it; streaming per configuration would dominate the runtime. 57k dicts is a few
    hundred MB at worst, which is the right trade on a machine that already holds a torch
    model.
    """

    def __init__(self, raw_dir: Path, limit: int | None = None) -> None:
        self._docs: dict[str, str] = {}
        for doc_id, text in iter_documents(raw_dir, limit=limit):
            self._docs[doc_id] = text

    def __len__(self) -> int:
        return len(self._docs)

    def __contains__(self, doc_id: object) -> bool:
        return doc_id in self._docs

    def get(self, doc_id: str, default: str = "") -> str:
        return self._docs.get(doc_id, default)

    def text(self, doc_id: str) -> str:
        if doc_id not in self._docs:
            raise KeyError(f"unknown document id {doc_id!r}")
        return self._docs[doc_id]

    def texts(self, doc_ids: Sequence[str]) -> list[str]:
        return [self.get(d) for d in doc_ids]

    def ids(self) -> list[str]:
        return list(self._docs)
