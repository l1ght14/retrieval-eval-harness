"""Regression gate: fail the build when retrieval quality moves.

A harness nobody enforces is a notebook. This is the piece that makes the harness load-bearing:
a config change, a dependency bump, or a re-embedding can silently degrade retrieval, and the
only way to notice is to have a threshold that fails.

Design decisions worth stating, because each one is a way a gate like this normally goes wrong:

**Thresholds are on absolute metric floors, not on "did this beat last time".** Comparing to the
previous run means the floor ratchets down one regression at a time and never fires.

**The margin is a fraction of the metric, not a constant.** A fixed 0.02 is meaningless at
recall 0.30 and enormous at recall 0.95. A proportional margin scales with the quantity.

**Floors are stored in a committed file, not in the code.** Changing a floor should be a
visible diff in a data file that reads like a policy, with the justification, rather than an
edit to a comparison operator buried in a module.

**Determinism is checked, not assumed.** Tie-breaking is by document id and the cache is keyed
by content, but a gate that reports a difference caused by its own non-determinism is worse
than no gate. `--repeat` re-runs the baseline and fails if the metric moves at all.

Exit codes: 0 pass, 1 gate failed, 2 configuration or data problem.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.corpus import GoldenSet  # noqa: E402

REPORTS = ROOT / "reports"
THRESHOLDS = ROOT / "eval" / "thresholds.json"

DEFAULT_THRESHOLDS = {
    "_comment": (
        "Minimum acceptable retrieval quality. Edit deliberately: a floor only ever goes up "
        "in a pull request, and the reason belongs in the commit message."
    ),
    "configs": {
        "bm25_256": {"recall@10": 0.26, "ndcg@10": 0.22, "hit@10": 0.50},
        "dense_256": {"recall@10": 0.44, "ndcg@10": 0.36, "hit@10": 0.75},
        "hybrid_256": {"recall@10": 0.42, "ndcg@10": 0.34, "hit@10": 0.73},
        "hybrid_256_rerank": {"recall@10": 0.40, "ndcg@10": 0.33, "hit@10": 0.70},
        "dense_256_rerank": {"recall@10": 0.41, "ndcg@10": 0.33, "hit@10": 0.71},
        "hybrid_256_w03": {"recall@10": 0.44, "ndcg@10": 0.36, "hit@10": 0.74},
        "bm25_128": {"recall@10": 0.24, "ndcg@10": 0.20, "hit@10": 0.47},
        "bm25_512": {"recall@10": 0.27, "ndcg@10": 0.23, "hit@10": 0.52},
        "hybrid_256_sum": {"recall@10": 0.42, "ndcg@10": 0.34, "hit@10": 0.72},
        "bm25_256_rrf_k10": {"recall@10": 0.42, "ndcg@10": 0.34, "hit@10": 0.72},
    },
    "relative_margin": 0.03,
    "_margin_note": (
        "A metric passes when it is within this fraction of its floor. Scaled rather than "
        "absolute so the gate means the same thing at recall 0.30 and at recall 0.95."
    ),
    "golden": {
        "name": "golden_v1",
        "corpus_sha256": "ff593e4df9933955",
    },
    "_golden_note": (
        "Prefix of the corpus SHA-256 recorded when the floors were set. A mismatch means the "
        "underlying data moved and every floor is suspect until they are re-derived."
    ),
}


def load_thresholds() -> dict:
    if THRESHOLDS.exists():
        return json.loads(THRESHOLDS.read_text(encoding="utf-8"))
    return DEFAULT_THRESHOLDS


def check_payload(payload: dict, floors: dict, margin: float) -> list[dict]:
    """Compare one result payload against its floors. Returns a list of failures."""
    name = payload["config"]["name"]
    if name not in floors:
        return [{"config": name, "metric": "-", "value": None, "floor": None,
                 "status": "NO FLOOR", "detail": "no threshold recorded for this config"}]
    failures = []
    for metric, floor in floors[name].items():
        value = payload["summary"].get(metric)
        if value is None:
            failures.append({"config": name, "metric": metric, "value": None, "floor": floor,
                             "status": "MISSING", "detail": "metric absent from the payload"})
            continue
        allowed = floor * (1 + margin)
        if value < allowed:
            failures.append({
                "config": name, "metric": metric, "value": round(value, 4), "floor": floor,
                "status": "FAIL",
                "detail": f"{value:.4f} < floor {floor} + {margin:.0%} margin ({allowed:.4f})",
            })
    return failures


def verify_provenance(thresholds: dict, golden: GoldenSet) -> list[dict]:
    """Fail if the golden set or corpus has moved under the recorded thresholds."""
    problems = []
    expected = (thresholds.get("golden") or {}).get("corpus_sha256", "")
    if expected and not golden.corpus_sha256.startswith(expected):
        problems.append({
            "config": "-", "metric": "provenance", "value": golden.corpus_sha256[:16],
            "floor": expected, "status": "FAIL",
            "detail": "corpus SHA-256 no longer matches the value the floors were set against; "
                      "re-derive the floors before trusting them",
        })
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", nargs="*", help="result JSONs to check (default: all)")
    ap.add_argument("--golden", default=str(ROOT / "data" / "golden" / "golden_v1.jsonl"))
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    thresholds = load_thresholds()
    margin = thresholds.get("relative_margin", 0.03)

    if not Path(args.golden).exists():
        print(f"golden set not found: {args.golden}", file=sys.stderr)
        return 2
    golden = GoldenSet.load(Path(args.golden))

    paths = (
        [Path(p) for p in args.reports]
        if args.reports
        else sorted(REPORTS.glob("eval_*.json"))
    )
    if not paths:
        print(f"no result files in {REPORTS}; run eval/run_eval.py first", file=sys.stderr)
        return 2

    failures = list(verify_provenance(thresholds, golden))
    checked = 0
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        got = check_payload(payload, thresholds["configs"], margin)
        failures.extend(got)
        checked += 1
        status = "FAIL" if got else "pass"
        if not args.json:
            s = payload["summary"]
            print(
                f"  {payload['config']['name']:<20} recall@10 {s['recall@10']:.4f}  "
                f"ndcg@10 {s['ndcg@10']:.4f}  [{status}]"
            )

    if args.json:
        print(json.dumps({"checked": checked, "failures": failures}, indent=2))
    else:
        print(f"\nchecked {checked} config(s) against floors (+{margin:.0%} margin)")
        if failures:
            print(f"\n{len(failures)} FAILURE(S):")
            for f in failures:
                print(f"  - [{f['status']}] {f['config']}.{f['metric']}: {f['detail']}")
            print("\ngate FAILED")
        else:
            print("gate PASSED")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
