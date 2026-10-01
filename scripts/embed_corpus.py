"""Embed the 256-word chunk set once, sharded, with each shard cached to disk.

This is the only genuinely slow step in the project: ~65,700 chunks at roughly 18 chunks/s on
8 CPU threads is a bit over an hour on one pass. Three things follow from that, and they
shape the whole design here.

**Shard it.** The corpus is split into contiguous shards of ~11k chunks, each cached
separately under a hash of its own texts. A run that dies at shard 5 of 6 has still banked
five shards, and `--shard 4` resumes instead of starting over. The first version of this
script was one monolithic pass and lost an hour of work to a killed process.

**Key on content, not filename.** Each shard's key hashes the model id, the normalisation
setting and the exact chunk texts. A cache that survives a model change is worse than no
cache: it silently returns the previous model's vectors and every downstream score becomes
meaningless while still looking plausible.

**Record a manifest.** `manifest.json` holds the full-corpus hash, the shard count and each
shard's file, so a partial run is detectable and `load_embedded_corpus` refuses to assemble
a matrix from an incomplete set.

Usage:
    python scripts/embed_corpus.py --status
    python scripts/embed_corpus.py --shard 0 --nshards 6
    python scripts/embed_corpus.py --all
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.chunking import chunk_stats, split_document  # noqa: E402
from src.corpus import DocumentStore  # noqa: E402
from src.dense import DEFAULT_MODEL  # noqa: E402

CHUNK_SIZE = 256
RAW = ROOT / "data" / "raw"
CACHE = ROOT / ".cache" / "embeddings"
MANIFEST = CACHE / "manifest.json"
DEFAULT_SHARDS = 6


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def build_chunks(store: DocumentStore) -> list:
    chunks: list = []
    for doc_id in store.ids():
        chunks.extend(split_document(doc_id, store.get(doc_id), CHUNK_SIZE, 0))
    return chunks


def shard_bounds(n: int, nshards: int) -> list[tuple[int, int]]:
    """Contiguous, near-equal shards. Even split rather than strided so that a shard's cache
    key depends only on a contiguous slice of the corpus."""
    step = -(-n // nshards)
    return [(i, min(i + step, n)) for i in range(0, n, step)]


def corpus_key(ids: list[str], texts: list[str]) -> str:
    h = hashlib.sha256()
    h.update(DEFAULT_MODEL.encode())
    h.update(b"|norm=True")
    for i, t in zip(ids, texts):
        h.update(b"\x00" + i.encode("utf-8", "replace"))
        h.update(b"\x01" + t.encode("utf-8", "replace"))
    return h.hexdigest()[:32]


def shard_filename(full_key: str, index: int) -> str:
    return f"{full_key}.s{index}.npy"


def load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {}


def save_manifest(data: dict) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(data, indent=2), encoding="utf-8")


def status() -> int:
    man = load_manifest()
    if not man:
        log("no manifest - nothing embedded yet")
        return 1
    shards = man.get("shards", {})
    total = man.get("n_chunks", 0)
    done = [s for s, meta in shards.items() if (CACHE / meta["file"]).exists()]
    log(f"corpus {total} chunks, {man['nshards']} shards, {len(done)} complete")
    for s in sorted(shards, key=int):
        exists = (CACHE / shards[s]["file"]).exists()
        log(f"  shard {s}: {'OK' if exists else 'MISSING'}  {shards[s]['count']} chunks")
    return 0 if len(done) == man["nshards"] else 1


def ensure_manifest(store: DocumentStore) -> tuple[list, dict]:
    chunks = build_chunks(store)
    ids = [c.chunk_id for c in chunks]
    texts = [c.text for c in chunks]
    full_key = corpus_key(ids, texts)
    man = load_manifest()
    if man.get("corpus_key") != full_key:
        bounds = shard_bounds(len(chunks), DEFAULT_SHARDS)
        man = {
            "corpus_key": full_key,
            "model": DEFAULT_MODEL,
            "chunk_size": CHUNK_SIZE,
            "n_chunks": len(chunks),
            "nshards": len(bounds),
            "chunk_stats": chunk_stats(chunks),
            "bounds": {str(i): [a, b] for i, (a, b) in enumerate(bounds)},
            "shards": {
                str(i): {"file": shard_filename(full_key, i), "count": b - a, "start": a}
                for i, (a, b) in enumerate(bounds)
            },
        }
        save_manifest(man)
        log(f"new manifest for corpus {full_key[:12]} - {man['nshards']} shards")
    else:
        log(f"manifest matches corpus {full_key[:12]}")
    return chunks, man


def embed_shard(chunks: list, man: dict, index: int, batch_size: int = 64) -> int:
    meta = man["shards"][str(index)]
    out = CACHE / meta["file"]
    if out.exists():
        log(f"shard {index} already cached")
        return 0
    a, b = meta["start"], meta["start"] + meta["count"]
    slice_ids = [c.chunk_id for c in chunks[a:b]]
    slice_texts = [c.text for c in chunks[a:b]]

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(man["model"])
    # MiniLM's native limit is 256 tokens; setting it explicitly avoids padding every batch
    # to the corpus maximum and is what makes the throughput honest.
    model.max_seq_length = 256

    log(f"embedding shard {index}/{man['nshards'] - 1}: {len(slice_ids)} chunks")
    t0 = time.time()
    vecs = model.encode(
        slice_texts, batch_size=batch_size, normalize_embeddings=True,
        show_progress_bar=True, convert_to_numpy=True,
    )
    dt = time.time() - t0
    CACHE.mkdir(parents=True, exist_ok=True)
    # Must end in .npy or np.save writes "<name>.npy" and the rename finds nothing there.
    tmp = out.with_name(out.name + ".tmp.npy")
    import numpy as np

    np.save(tmp, vecs)
    tmp.replace(out)
    log(f"shard {index} done in {dt / 60:.1f} min ({len(slice_ids) / dt:.1f} chunks/s) -> {out.name}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--shard", type=int)
    ap.add_argument("--nshards", type=int, default=DEFAULT_SHARDS,
                    help="informational; the manifest fixes the shard count at first run")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    if args.status:
        return status()

    CACHE.mkdir(parents=True, exist_ok=True)
    log(f"loading corpus from {RAW}")
    store = DocumentStore(RAW)
    log(f"{len(store)} documents")
    chunks, man = ensure_manifest(store)
    log(f"{len(chunks)} chunks at {CHUNK_SIZE} words  {chunk_stats(chunks)}")

    if args.all:
        rc = 0
        for i in range(man["nshards"]):
            rc |= embed_shard(chunks, man, i, args.batch_size)
        save_manifest(man)
        return rc or status()
    if args.shard is None:
        ap.error("pass --status, --shard N, or --all")
    return embed_shard(chunks, man, args.shard, args.batch_size)


if __name__ == "__main__":
    raise SystemExit(main())
