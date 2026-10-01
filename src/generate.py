"""Generation over retrieved context, and the split between retrieval failure and
generation failure.

This module exists because of one specific question the project has to answer honestly: when
a RAG system gives a bad answer, is the retriever at fault or the model? Those need different
fixes and they cost completely different amounts of money, so collapsing them into one
"accuracy" number destroys the information you need.

Retrieval is measured in `metrics.py` against human relevance labels. Generation is measured
here, and it is reported in separate columns. A system with recall@10 of 0.47 and faithful
answers is a retrieval problem. A system with recall@10 of 0.90 and unfaithful answers is a
prompt or model problem. Both look identical if you only publish an end-to-end score.

Faithfulness is measured two ways, because one of them is always available:

**Lexical grounding (always available, no key, deterministic).** Every sentence of the answer
must share content words with the retrieved context. A sentence with no support in the context
is flagged. This is a proxy and a weak one - it cannot tell a correct paraphrase from a wrong
one sharing vocabulary, and it will penalise a legitimate abstractive answer. It is reported
because it is reproducible in CI on any machine with no API key, and because a metric that
only exists when you have credentials is not much of a guard.

**LLM-as-judge (only with a key).** The model is asked to mark each sentence supported or
unsupported given the context. Used when a provider is configured, and reported separately so
the two are never silently mixed.

Nothing here invents a score. When no key is present the LLM-judge columns are `null`, not a
default of 1.0, because a generation metric that silently passes offline is worse than no
metric at all.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from .lexical import tokenize
from .llm import LLMResponse

ANSWER_PROMPT = """Answer the question using only the context below.

If the context does not contain the answer, say exactly: "The context does not contain \
enough information to answer."

Context:
{context}

Question: {question}

Answer in at most three sentences. Do not use any fact that is not in the context."""


@dataclass
class GenerationResult:
    query_id: str
    answer: str
    context: str
    grounded: bool | None = None
    unsupported_sentences: list[str] = field(default_factory=list)
    coverage: float = 0.0
    judge_supported: bool | None = None
    judge_rationale: str = ""
    model: str = ""
    cached: bool = False

    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "answer": self.answer,
            "grounded": self.grounded,
            "unsupported_sentences": self.unsupported_sentences,
            "coverage": round(self.coverage, 4),
            "judge_supported": self.judge_supported,
            "judge_rationale": self.judge_rationale,
            "model": self.model,
            "cached": self.cached,
            "context_chars": len(self.context),
        }


# ---------------------------------------------------------------- lexical grounding

# Sentences that carry no factual claim. Splitting on these would produce false positives on
# any answer that opens with "The context does not contain..." or similar.
_ABSTENTION = re.compile(
    r"(does not contain|enough information|cannot|i (don'?t|do not) know|no information)",
    re.I,
)


def sentences_of(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [p.strip() for p in parts if p.strip()]


def grounding_report(answer: str, context: str, threshold: float = 0.5) -> tuple[bool, list[str], float]:
    """Fraction of answer sentences whose content words appear in the retrieved context.

    `threshold` is the share of a sentence's content words that must be present in the
    context for the sentence to count as grounded. Sentences matching an abstention pattern
    are exempt, because a correct "the context is insufficient" answer contains almost no
    context vocabulary and would otherwise be scored as unfaithful - punishing the model for
    the one behaviour that indicates good calibration.
    """
    claims = [s for s in sentences_of(answer) if not _ABSTENTION.search(s)]
    if not claims:
        return True, [], 1.0
    context_tokens = set(tokenize(context))
    unsupported: list[str] = []
    for sentence in claims:
        tokens = tokenize(sentence)
        if not tokens:
            continue
        overlap = sum(1 for t in tokens if t in context_tokens) / len(tokens)
        if overlap < threshold:
            unsupported.append(sentence)
    supported_ratio = 1.0 - (len(unsupported) / len(claims))
    return supported_ratio >= 0.999, unsupported, supported_ratio


# ---------------------------------------------------------------- llm judge

JUDGE_PROMPT = """You are checking whether an answer is supported by the supplied context.

Context:
{context}

Answer:
{answer}

Reply with exactly one word, SUPPORTED or UNSUPPORTED, then a colon and a short reason."""


def parse_judge(verdict: str) -> tuple[bool, str]:
    text = (verdict or "").strip()
    if ":" in text:
        label, _, reason = text.partition(":")
    else:
        label, reason = text, ""
    return label.strip().upper().startswith("SUPPORTED"), reason.strip()


# ---------------------------------------------------------------- driver


def generate_answer(
    client,
    question: str,
    context: str,
    query_id: str = "",
    max_context_chars: int = 3000,
    judge: bool = False,
) -> GenerationResult:
    """Answer from context, then score the answer without conflating the two stages.

    `judge=False` by default even when a real model is available, because the lexical metric
    is the reproducible one and it is the one CI enforces. Opting into the LLM judge is an
    explicit choice, and its result lands in a different field.
    """
    context = (context or "")[:max_context_chars]
    response: LLMResponse = client.complete(
        [{"role": "user", "content": ANSWER_PROMPT.format(context=context, question=question)}],
        temperature=0.0,
        max_tokens=400,
    )
    grounded, unsupported, coverage = grounding_report(response.text, context)
    result = GenerationResult(
        query_id=query_id,
        answer=response.text,
        context=context,
        grounded=grounded,
        unsupported_sentences=unsupported,
        coverage=coverage,
        model=response.model,
        cached=response.cached,
    )

    if judge and getattr(client, "available", False) and client.provider != "mock":
        verdict = client.complete(
            [{"role": "user", "content": JUDGE_PROMPT.format(context=context, answer=response.text)}],
            temperature=0.0,
            max_tokens=120,
        )
        result.judge_supported, result.judge_rationale = parse_judge(verdict.text)
    return result


def summarize_generation(results: Sequence[GenerationResult]) -> dict:
    """Aggregate generation metrics, keeping the unavailable judge column explicitly null."""
    if not results:
        raise ValueError("no generation results to summarise")
    judged = [r for r in results if r.judge_supported is not None]
    refused = sum(
        1 for r in results if _ABSTENTION.search(r.answer or "")
    )
    return {
        "n": len(results),
        "grounded_rate": round(sum(1 for r in results if r.grounded) / len(results), 4),
        "mean_sentence_coverage": round(sum(r.coverage for r in results) / len(results), 4),
        "abstention_rate": round(refused / len(results), 4),
        "mean_answer_chars": round(sum(len(r.answer) for r in results) / len(results), 1),
        "cached_rate": round(sum(1 for r in results if r.cached) / len(results), 4),
        # null, never 1.0 - a metric that passes when it could not run is a lie
        "judge_supported_rate": (
            round(sum(1 for r in judged if r.judge_supported) / len(judged), 4) if judged else None
        ),
        "judge_n": len(judged),
    }


def combine(retrieval: dict, generation: dict) -> dict:
    """Put the two stages side by side, which is the point of keeping them separate.

    `retrieval_hit_rate` with `grounded_rate` is the diagnostic pair: high recall and low
    grounding means the retriever is fine and the model is over-claiming; low recall means no
    amount of prompt work will help, because the answer was never in the context.
    """
    hit = retrieval.get("summary", {}).get("hit@10")
    grounded = generation.get("grounded_rate")
    if hit is None or grounded is None:
        diagnosis = "insufficient data"
    elif hit < 0.5 and grounded < 0.9:
        diagnosis = "retrieval-limited: fix the index before touching the prompt"
    elif hit >= 0.5 and grounded < 0.9:
        diagnosis = "generation-limited: the context is there, the model is over-claiming"
    elif hit >= 0.5 and grounded >= 0.9:
        diagnosis = "both stages healthy"
    else:
        diagnosis = "mixed: inspect the per-query rows"
    return {
        "retrieval_hit_rate": hit,
        "retrieval_recall@10": retrieval.get("summary", {}).get("recall@10"),
        "grounded_rate": grounded,
        "judge_supported_rate": generation.get("judge_supported_rate"),
        "abstention_rate": generation.get("abstention_rate"),
        "diagnosis": diagnosis,
    }
