"""Rebuild `reports/comparisons.json` from the per-query rows already on disk.

Configs are run one at a time (the rerank arms take minutes each), so a single eval invocation
only ever sees the configs it ran and its paired comparisons are incomplete. The per-query
rows in `reports/eval_*.json` are the source of truth, so the full comparison table can be
derived from them in a second instead of re-running every retriever.

Paired rather than unpaired, because every config is scored on the same golden queries: query
difficulty cancels, and the interval reflects only the configuration change.

    python eval/compare.py --baseline bm25_256
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.metrics import paired_bootstrap_delta  # noqa: E402

REPORTS = ROOT / "reports"
METRICS = ("recall@10", "hit@10", "mrr", "ndcg@10", "precision@10")


def load_all() -> dict[str, dict]:
    out = {}
    for path in sorted(REPORTS.glob("eval_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        out[payload["config"]["name"]] = payload
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="bm25_256")
    ap.add_argument("--out", default=str(REPORTS / "comparisons.json"))
    args = ap.parse_args()

    payloads = load_all()
    if args.baseline not in payloads:
        print(f"baseline {args.baseline!r} not found; have {sorted(payloads)}", file=sys.stderr)
        return 2

    base_rows = {r["query_id"]: r for r in payloads[args.baseline]["per_query"]}
    comparisons = {}
    for name, payload in sorted(payloads.items()):
        if name == args.baseline:
            continue
        shared = [r for r in payload["per_query"] if r["query_id"] in base_rows]
        if len(shared) != len(base_rows):
            print(f"  skipping {name}: {len(shared)}/{len(base_rows)} queries overlap the baseline")
            continue
        entry = {}
        for metric in METRICS:
            a = [r[metric] for r in shared]
            b = [base_rows[r["query_id"]][metric] for r in shared]
            entry[metric] = paired_bootstrap_delta(a, b)
        # win/loss counts: a mean cannot distinguish "helped 5 queries a lot" from
        # "helped 40 queries a little and hurt 30 badly"
        for metric in ("mrr",):
            wins = sum(1 for r in shared if r[metric] > base_rows[r["query_id"]][metric] + 1e-9)
            losses = sum(1 for r in shared if r[metric] < base_rows[r["query_id"]][metric] - 1e-9)
            entry[f"{metric}_win_loss"] = {
                "better": wins, "worse": losses, "unchanged": len(shared) - wins - losses,
            }
        comparisons[name] = entry

    result = {
        "baseline": args.baseline,
        "n_queries": len(base_rows),
        "metric": "paired bootstrap of per-query differences, 2000 resamples, seed 42",
        "comparisons": comparisons,
    }
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"baseline {args.baseline} over {len(base_rows)} queries, {len(comparisons)} comparisons\n")
    for name, entry in comparisons.items():
        d = entry["recall@10"]
        wl = entry["mrr_win_loss"]
        verdict = "significant" if d["significant"] else "n.s."
        print(f"  {name:<20} drecall {d['delta']:+.4f} [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}] "
              f"{verdict:<12} | MRR better/worse {wl['better']}/{wl['worse']}")
    print(f"\nwrote {out_path.relative_to(ROOT) if out_path.is_relative_to(ROOT) else out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
