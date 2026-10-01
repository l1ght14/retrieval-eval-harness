"""Fetch the BEIR corpus and mint the versioned golden set.

Kept as a script rather than done by hand so the golden set is reproducible from the raw
files: anyone can run this and get a byte-identical `golden_v1.jsonl`, which is the property
that makes the stored thresholds in `eval/thresholds.json` meaningful.

    python scripts/build_golden.py --size 200 --seed 42

The BEIR download is skipped when `data/raw/corpus.jsonl` is already present. FiQA is
distributed under CC BY-NC 3.0, which is recorded in the golden set's provenance block so it
travels with the data.
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.corpus import BEIR_URL, build_golden_set, sha256_of  # noqa: E402

RAW = ROOT / "data" / "raw"
GOLDEN_DIR = ROOT / "data" / "golden"


def log(msg: str) -> None:
    print(f"  {msg}", flush=True)


def download(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "corpus.jsonl").exists():
        log("corpus.jsonl already present, skipping download")
        return
    archive = dest.parent / "fiqa.zip"
    log(f"downloading {BEIR_URL}")
    with urlopen(BEIR_URL, timeout=300) as resp, archive.open("wb") as fh:  # noqa: S310
        total = 0
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
            total += len(chunk)
    log(f"downloaded {total / 1e6:.1f} MB")

    # BEIR archives nest everything under a dataset-named folder; extract past it.
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            parts = Path(member).parts
            if member.endswith((".jsonl", ".tsv")) and len(parts) > 1:
                rel = Path(*parts[1:])
                target = dest / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, target.open("wb") as out:
                    out.write(src.read())
    archive.unlink(missing_ok=True)
    log(f"extracted to {dest}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=200, help="number of golden queries")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default="test", choices=["test", "dev", "train"])
    ap.add_argument("--min-relevant", type=int, default=1)
    ap.add_argument("--max-relevant", type=int, default=5)
    ap.add_argument("--name", default="golden_v1")
    args = ap.parse_args()

    log(f"corpus source: {BEIR_URL}")
    download(RAW)

    for name in ("corpus.jsonl", "queries.jsonl"):
        if not (RAW / name).exists():
            log(f"ERROR: {name} missing after extraction")
            return 2
    log(f"corpus sha256: {sha256_of(RAW / 'corpus.jsonl')}")

    golden = build_golden_set(
        RAW, name=args.name, split=args.split, size=args.size, seed=args.seed,
        min_relevant=args.min_relevant, max_relevant=args.max_relevant,
    )
    out = GOLDEN_DIR / f"{args.name}.jsonl"
    golden.save(out)
    prov = golden.provenance()
    log(f"wrote {out.relative_to(ROOT)}")
    log(f"  queries              {prov['n_queries']}")
    log(f"  mean relevant/query  {prov['mean_relevant_per_query']}")
    log(f"  corpus sha256        {prov['corpus_sha256']}")
    log(f"  provenance           {(out.with_suffix('.meta.json')).relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
