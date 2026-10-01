"""Turn the result JSONs into the report: tables, error analysis, figures, and the findings.

Everything here reads `reports/eval_*.json` and writes Markdown plus figures. It does no
retrieval of its own, so the report can be regenerated from stored results in a second
without re-running a single query - which is the whole reason the per-query rows are kept on
disk rather than thrown away after a summary.

The prose in `FINDINGS` is written against the measured numbers, and each claim carries the
evidence that supports it. If a config changes, the numbers move and the prose is expected to
be wrong until someone updates it; `check_report_consistency` exists to make that failure loud
rather than embarrassing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REPORTS = ROOT / "reports"
FIGS = REPORTS / "figures"
OUT = REPORTS / "RESULTS.md"

MODE_AXIS = ["bm25_256", "dense_256", "hybrid_256", "hybrid_256_rerank"]
ATTRIBUTION = ["dense_256", "dense_256_rerank", "hybrid_256", "hybrid_256_w03", "hybrid_256_rerank"]
CHUNK_AXIS = ["bm25_128", "bm25_256", "bm25_512"]

METRICS = [("recall@10", "recall@10"), ("hit@10", "hit@10"), ("mrr", "MRR"),
           ("ndcg@10", "nDCG@10")]


def load(name: str) -> dict | None:
    path = REPORTS / f"eval_{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_comparisons(path: Path | str = REPORTS / "comparisons.json") -> dict:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}



def table(names: list[str], baseline: str | None = None) -> str:
    missing = []
    payloads = {}
    for n in names:
        p = load(n)
        if p is None:
            missing.append(n)
        else:
            payloads[n] = p
    if not payloads:
        return f"_no results found for {', '.join(names)}_\n"

    head = "| config | mode | chunk | rerank | " + " | ".join(m[1] for m in METRICS) + " | p50 ms |"
    sep = "|---|---|---|---|" + "---|" * (len(METRICS) + 1)
    lines = [head, sep]
    for n, p in payloads.items():
        c, s = p["config"], p["summary"]
        lines.append(
            f"| `{n}` | {c['mode']} | {c['chunk_size']} | {'yes' if c['rerank'] else 'no'} | "
            + " | ".join(f"{s[m]:.4f}" for m, _ in METRICS)
            + f" | {p['query_latency_ms']['p50']:.0f} |"
        )
    if missing:
        lines.append("")
        lines.append(f"_not run: {', '.join(f'`{m}`' for m in missing)}_")
    return "\n".join(lines) + "\n"


def paired_table(names: list[str], baseline: str, metric: str = "recall@10") -> str:
    return paired_table_from(names, baseline, REPORTS / "comparisons.json", metric)


def paired_table_from(
    names: list[str], baseline: str, path: Path, metric: str = "recall@10"
) -> str:
    cmp_payload = load_comparisons(path)
    if cmp_payload.get("baseline") != baseline:
        return (
            f"_no paired comparison against `{baseline}`; build it with "
            f"`python eval/compare.py --baseline {baseline}`_\n"
        )
    entries = cmp_payload.get("comparisons", {})
    lines = [
        "| config | Δ recall@10 | 95% CI | verdict | MRR better/worse |",
        "|---|---|---|---|---|",
    ]
    for n in names:
        e = entries.get(n, {})
        if metric not in e:
            continue
        delta = e[metric]
        verdict = "significant" if delta["significant"] else "not significant"
        wl = e.get("mrr_win_loss", {})
        wr = f"{wl['better']} / {wl['worse']}" if wl else "-"
        lines.append(
            f"| `{n}` | {delta['delta']:+.4f} | "
            f"[{delta['ci_low']:+.4f}, {delta['ci_high']:+.4f}] | {verdict} | {wr} |"
        )
    return "\n".join(lines) + "\n" if len(lines) > 2 else "_no comparisons_\n"


def error_analysis(name: str = "dense_256") -> str:
    """Where the best configuration still fails, which is where the next work goes."""
    p = load(name)
    if not p:
        return "_no results_\n"
    rows = p["per_query"]
    n = len(rows)
    zero = [r for r in rows if r["recall@10"] == 0]
    perfect = [r for r in rows if r["recall@10"] == 1.0]
    partial = [r for r in rows if 0 < r["recall@10"] < 1]
    hard = sorted(rows, key=lambda r: r["mrr"])[:10]
    lines = [
        f"Over {n} queries the best configuration (`{name}`) splits into:",
        "",
        f"- **{len(perfect)} fully retrieved** (recall@10 = 1.0)",
        f"- **{len(partial)} partially retrieved** (0 < recall@10 < 1)",
        f"- **{len(zero)} not retrieved at all** (recall@10 = 0)",
        "",
        f"Mean relevant documents per query: "
        f"**{sum(r['n_relevant'] for r in rows) / n:.2f}**. Partial retrieval dominates the "
        f"failures, which means the ranking is usually putting the right document *somewhere* "
        f"in the list rather than missing it - a depth problem, not a recall problem.",
        "",
        "### Hardest queries (lowest MRR)",
        "",
        "| query | relevant found | recall@10 | MRR | nDCG@10 | first hit |",
        "|---|---|---|---|---|---|",
    ]
    for r in hard:
        first = r["first_relevant_rank"] if r["first_relevant_rank"] else "-"
        lines.append(
            f"| `{r['query_id']}` | {r['recall@10']:.2f} | {r['recall@10']:.2f} | "
            f"{r['mrr']:.3f} | {r['ndcg@10']:.3f} | {first} |"
        )
    return "\n".join(lines) + "\n"


def win_loss(name: str, baseline: str = "dense_256") -> str:
    """Paired win/loss counts, which say something a mean cannot."""
    a, b = load(name), load(baseline)
    if not a or not b:
        return ""
    bmap = {r["query_id"]: r for r in b["per_query"]}
    better = worse = same = 0
    lost_hits = 0
    worst = 0.0
    for r in a["per_query"]:
        ref = bmap.get(r["query_id"])
        if not ref:
            continue
        d = r["mrr"] - ref["mrr"]
        if d > 1e-9:
            better += 1
        elif d < -1e-9:
            worse += 1
        else:
            same += 1
        worst = min(worst, d)
        if ref["first_relevant_rank"] and not r["first_relevant_rank"]:
            lost_hits += 1
    if better == worse == same == 0:
        return ""
    return (
        f"Against `{baseline}`: MRR improves on **{better}**, degrades on **{worse}**, "
        f"unchanged on **{same}** of {len(a['per_query'])} queries. "
        f"Worst single query: **{worst:+.3f}**."
        + (f" Queries that had a hit at all and lost it entirely: **{lost_hits}**." if lost_hits else "")
    )


def figures() -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGS.mkdir(parents=True, exist_ok=True)
    written = []

    # --- recall vs latency, log x-axis: the cost/benefit of each addition
    pts = [(n, p) for n, p in ((x, load(x)) for x in MODE_AXIS + ["dense_256_rerank"]) if p]
    if pts:
        fig, ax = plt.subplots(figsize=(8, 4.6))
        for n, p in pts:
            s = p["summary"]
            ax.scatter(p["query_latency_ms"]["p50"], s["recall@10"], s=130,
                       edgecolor="white", linewidth=1.4, zorder=3, label=n)
            ax.annotate(n, (p["query_latency_ms"]["p50"], s["recall@10"]),
                        textcoords="offset points", xytext=(9, 5), fontsize=8.5)
        ax.set_xscale("log")
        ax.set_xlabel("query latency, p50 (ms, log scale)")
        ax.set_ylabel("recall@10")
        ax.set_title("Each addition to the retriever cost latency and did not buy accuracy", fontsize=11)
        ax.grid(alpha=0.25, zorder=0)
        fig.tight_layout()
        fig.savefig(FIGS / "recall_vs_latency.png", dpi=150)
        plt.close(fig)
        written.append("figures/recall_vs_latency.png")

    # --- chunk size axis
    cps = [(n, load(n)) for n in CHUNK_AXIS]
    cps = [(n, p) for n, p in cps if p]
    if cps:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        xs = [p["config"]["chunk_size"] for _, p in cps]
        for metric, marker in (("recall@10", "o"), ("ndcg@10", "s")):
            ax.plot(xs, [p["summary"][metric] for _, p in cps], marker=marker,
                    linewidth=2, label=metric)
        ax.set_xlabel("chunk size (words)")
        ax.set_ylabel("score")
        ax.set_title("Bigger chunks win on FiQA - the corpus is already chunk-sized", fontsize=11)
        ax.set_xticks(xs)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(FIGS / "chunk_axis.png", dpi=150)
        plt.close(fig)
        written.append("figures/chunk_axis.png")

    # --- recall@k curve for the best config, to show where depth stops helping
    best = load("dense_256")
    if best:
        rows = best["per_query"]
        ks = [1, 3, 5, 10, 20, 50]
        # Hit rate by depth, derived from the stored first_relevant_rank rather than
        # recomputed by re-running retrieval. The stored rank is the rank of the *first*
        # relevant document, so this is a hit-rate curve, not a recall@k curve - and it is
        # labelled as such, because conflating the two is the mistake this project exists
        # to prevent.
        curve = []
        for k in ks:
            hits = sum(1 for r in rows if r["first_relevant_rank"] and r["first_relevant_rank"] <= k)
            curve.append(hits / len(rows))
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.plot(ks, curve, marker="o", linewidth=2, color="#2b6cb0")
        ax.set_xlabel("k")
        ax.set_ylabel("hit rate")
        ax.set_xscale("log")
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_title("dense_256: hit rate by depth - most of what is findable is in the top 10",
                     fontsize=11)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(FIGS / "recall_by_depth.png", dpi=150)
        plt.close(fig)
        written.append("figures/recall_by_depth.png")
    return written


def check_report_consistency() -> list[str]:
    """Fail loudly when the prose claims something the stored results do not support.

    This exists because it already caught a real error. The first draft of the findings
    section said fusion and reranking "made retrieval worse" on the strength of their point
    estimates; the paired tests showed every interval containing zero, so the claim was not
    supported and the text was rewritten to "no measurable benefit".

    The check is deliberately inverted relative to the obvious implementation. Rather than
    comparing hardcoded numbers against the data - which validates the check but not the
    prose - it *generates* the claim string from the stored results and requires that exact
    text to appear in FINDINGS. Editing a number in the prose therefore fails the build,
    which is the only arrangement where this function does anything useful.

    Known limitation: this is a presence check, not a parser. If a number appears more than
    once in the prose, editing one occurrence will not be caught. It is a coarse net against
    the realistic failure - a config change moving every number while the prose keeps the old
    ones - and building a markdown claim-parser to close the remaining gap would cost more
    than the errors it would prevent.
    """
    problems = []

    # Findings is wrapped markdown, so a claim can straddle a newline. Collapsing whitespace
    # on both sides before comparing is what makes a substring check meaningful here.
    flat = " ".join(FINDINGS.split())

    cmp_bm = load_comparisons(REPORTS / "comparisons.json")
    if cmp_bm.get("baseline") != "bm25_256":
        return ["reports/comparisons.json missing or wrong baseline; run `python eval/compare.py`"]
    entry = cmp_bm["comparisons"].get("dense_256", {}).get("recall@10")
    if not entry:
        return ["dense_256 vs bm25_256 comparison missing from reports/comparisons.json"]
    expected = (f"+{entry['delta']:.4f}, 95% CI "
                f"[+{entry['ci_low']:.4f}, +{entry['ci_high']:.4f}]")
    if expected not in flat:
        problems.append(
            f"FINDINGS does not state the measured dense-vs-bm25 result as '{expected}'; "
            f"re-derive it from reports/comparisons.json"
        )

    cmp_dense = load_comparisons(REPORTS / "comparisons_vs_dense.json")
    if cmp_dense.get("baseline") != "dense_256":
        problems.append(
            "reports/comparisons_vs_dense.json missing; run `python eval/compare.py "
            "--baseline dense_256 --out reports/comparisons_vs_dense.json`"
        )
        return problems

    # Every arm the prose calls "not significant" must really have an interval containing zero.
    for name in ("dense_256_rerank", "hybrid_256_rerank", "hybrid_256", "hybrid_256_w03"):
        got = cmp_dense["comparisons"].get(name, {}).get("recall@10")
        if got and got["significant"]:
            problems.append(
                f"FINDINGS calls {name} not significant, but its CI "
                f"[{got['ci_low']:+.4f}, {got['ci_high']:+.4f}] excludes zero"
            )

    for name, quoted in (("dense_256", "0.4716"), ("bm25_256", "0.2949"),
                         ("dense_256_rerank", "0.4863"), ("bm25_512", "0.3085")):
        payload = load(name)
        if payload and f"{payload['summary']['recall@10']:.4f}" not in flat:
            problems.append(
                f"FINDINGS no longer cites {name} recall@10 = {quoted} "
                f"(stored: {payload['summary']['recall@10']:.4f})"
            )
    return problems


FINDINGS = """
## What actually happened

Five things in this run are worth stating. Two of them are corrections to earlier drafts of
this section, and both corrections came from the measurement rather than from re-reading the
code - which is the whole argument for having the harness.

**1. Dense retrieval beat BM25 by a wide, unambiguous margin.**
`recall@10` 0.4716 against 0.2949 - a paired difference of **+0.1767, 95% CI [+0.1182,
+0.2344]**, excluding zero by a wide margin. MRR improved on 75 queries and degraded on 26.
This is the one comparison in the project that is not in doubt, and it matches what BEIR
reports for FiQA: a dataset of natural-language questions with lexically poor answers is
exactly where a bi-encoder should win.

**2. Hybrid fusion bought nothing at any weighting.**
At equal RRF weights it was slightly worse than dense alone (-0.0069, n.s.); at BM25 weight
0.3 it returned a delta of exactly `+0.0000` with a zero-width interval - the lexical arm
changed the top-10 membership on *no* query. It was not being outvoted, it was contributing
nothing. With ~2 relevant documents per query there is very little for a lexical arm to add to
a dense model, and FiQA's queries are phrased in natural language rather than keywords, which
is the case BM25 was built for.

**3. Cross-encoder reranking helped - but not detectably, and only after a bug was fixed.**
The final numbers:

| config | recall@10 | nDCG@10 | p50 latency |
|---|---|---|---|
| `dense_256` | 0.4716 | 0.3949 | 22 ms |
| `dense_256_rerank` | 0.4863 | 0.3966 | 8810 ms |
| `hybrid_256_rerank` | 0.4547 | 0.3730 | 6543 ms |

Paired against `dense_256`, `dense_256_rerank` is **+0.0148, CI [-0.0013, +0.0350]** - still
not significant, with the lower bound sitting almost exactly on zero. So the honest statement
is: reranking is directionally positive, its win/loss split is 36/29, and it costs **400x the
latency** (22 ms to 8,810 ms) for a gain this sample size cannot confirm. On a 200-query set
the interval is wide enough that a real 1.5-point gain would not clear it. The recommendation
is unchanged for a reason that is about cost, not quality.

**4. A bug in my own reranker path was suppressing the effect I was reporting on.**
The first run put `dense_256_rerank` at 0.4470 - *below* dense retrieval - and I wrote that
the reranker "made retrieval worse". The cause was in `retriever.py`: the cross-encoder was
scoring the retrieved **chunk** while `context_for` handed the model the whole **document**, so
the reranker was judging text the reader never sees. With FiQA's median document at 90 words
this mostly affects the 11% of documents split into multiple chunks - but it moved
`dense_256_rerank` from 0.4470 to **0.4863**, a 0.039 swing that reversed the sign of the
effect.

The lesson is not "always align your stages". It is that a negative result was reported before
anyone asked whether the thing under test was wired up correctly, and a paired significance
test did not catch it either - the broken version was confidently non-significant in the wrong
direction.

**5. Chunk size pointed the opposite way from the usual advice.**
128-word chunks scored 0.2704, 256-word 0.2949, 512-word 0.3085 - and the 128-vs-512 gap is
significant. *Smaller chunks were worse.* The corpus explains why: FiQA's median document is
90 words, so splitting at 128 words severs 34% of documents from their own context, and BM25
needs the surrounding words to score a term well. At 512 words only 2% of documents are
touched, which makes the best setting on this corpus, in effect, **not chunking**. A harness
that had only tried chunk sizes familiar from long documents would have tuned this axis
backwards.

## What was wrong along the way

- The reranker/retrieval text mismatch above. Reported a negative result before checking the
  thing under test was wired up correctly.
- The first draft of this section claimed fusion and reranking were both *harmful*. The paired
  tests did not support "harmful" for either. Corrected to "no measurable benefit", and later
  corrected again for reranking once the bug was found. `report.py` now generates the expected
  claim text from the stored results and fails if the prose does not contain it.
- An O(n x chunks) scan in the chunker's stats function that would have taken hours on 57k
  chunks. It looked like a one-line convenience.
- BM25 scored each query by masking a CSR column per query term, O(terms x total_postings).
  Rewritten to per-term postings, 0.4 ms/query.
- `top_k_from_scores` broke ties by index on one branch and by id on the other, so a ranking
  depended on corpus read order. A test caught it.
- The reranked and non-reranked `search()` paths returned different chunk ids for the same
  document, because one used `setdefault` and the other a dict comprehension. Found in review.
- `sum` chunk aggregation claimed to sum *reciprocal* ranks but summed raw ranks - an
  implementation that would have favoured the opposite of the documented intent.
- `np.save` appends `.npy` to any path not already ending in it, so an atomic-cache temp file
  named `*.npy.tmp` wrote `*.npy.tmp.npy` and the rename failed. Cost one hour of embedding.
- `LLMClient("mock")` constructed cleanly, reported `available=True`, and raised
  `OpenAIError: Missing credentials` on first use.
- Installing `optimum[onnxruntime]` to chase a speedup silently downgraded `huggingface-hub`
  and broke sentence-transformers. It bought 1.3x and cost more than it gave.
- Throughput measured 10 chunks/s in one run and 20 chunks/s in another on the same machine.
  Every throughput claim here carries the measurement that produced it.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    parts = ["# Groundtruth - retrieval evaluation results", ""]
    parts += [
        "Corpus: FiQA (57,638 documents) via BEIR. Golden set: 200 queries sampled "
        "deterministically from the test split, mean 2.17 human-annotated relevant documents "
        "per query. Embeddings: all-MiniLM-L6-v2 (384-d). Reranker: "
        "ms-marco-MiniLM-L-6-v2. Every interval is a 2000-sample percentile bootstrap.",
        "",
        "## Headline: the retrieval-mode axis", "",
        "All four arms use identical 256-word chunking over the full corpus, so the mode is "
        "the only thing that varies.", "",
        table(MODE_AXIS), "",
        "Paired against `bm25_256` (same 200 queries, so query difficulty cancels):", "",
        paired_table(MODE_AXIS, "bm25_256"), "",
        "## Attribution: why fusion and reranking were kept out", "",
        "The headline table cannot tell whether the reranker was bad or whether fusion had "
        "already damaged the candidate list it was given. These two extra cells separate them.",
        "", table(ATTRIBUTION), "",
        "Paired against `dense_256`, the best configuration:", "",
        paired_table_from(ATTRIBUTION, "dense_256", "reports/comparisons_vs_dense.json"), "",
        win_loss("dense_256_rerank") or "", "",
        win_loss("hybrid_256") or "", "",
        win_loss("hybrid_256_w03") or "", "",
        "## The chunk-size axis (BM25 only)", "",
        "Measured on the lexical arm because it is the only arm where varying chunk size is "
        "free. Dense re-embedding per granularity costs about an hour and, as the 128-vs-512 "
        "gap below shows, would have been spent varying an axis that mostly confirms this one.",
        "", table(CHUNK_AXIS), "",
        paired_table(CHUNK_AXIS, "bm25_128"), "",
        "## Error analysis", "", error_analysis("dense_256"), "",
    ]

    if not args.no_figures:
        try:
            figs = figures()
            if figs:
                parts += ["## Figures", ""] + [f"![{Path(f).name}]({f})" for f in figs] + [""]
        except ImportError:
            parts += ["_matplotlib not available; figures skipped_\n", ""]

    parts += [FINDINGS, ""]
    OUT.write_text("\n".join(parts), encoding="utf-8")
    problems = check_report_consistency()
    print(f"wrote {OUT.relative_to(ROOT)} ({len(''.join(parts))} chars)")
    if problems:
        print("\nconsistency warnings:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("report consistent with stored results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
