"""Generation layer, LLM wrapper, and the regression gate.

The generation tests use the offline mock throughout, because that is exactly the mode CI
runs in - and the point being verified is that the metrics still work with no API key, no
network, and no spend.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.generate import (
    combine,
    generate_answer,
    grounding_report,
    parse_judge,
    sentences_of,
    summarize_generation,
)
from src.llm import OfflineLLM, get_client

ROOT = Path(__file__).resolve().parent.parent
CONTEXT = "Refunds are issued to the original payment method within 5 business days. " \
          "A refund request must include the order number."


# ---------------------------------------------------------------- sentences


def test_sentences_split_on_terminators():
    assert sentences_of("One. Two! Three?") == ["One.", "Two!", "Three?"]


def test_sentences_of_empty_string():
    assert sentences_of("") == []


# ---------------------------------------------------------------- grounding


def test_answer_fully_grounded_in_context():
    ok, unsupported, coverage = grounding_report(
        "Refunds are issued within 5 business days.", CONTEXT
    )
    assert ok is True
    assert unsupported == []
    assert coverage == 1.0


def test_answer_with_an_unsupported_claim_is_flagged():
    ok, unsupported, coverage = grounding_report(
        "Refunds take 5 business days. The bank charges a $20 fee.", CONTEXT
    )
    assert ok is False
    assert len(unsupported) == 1
    assert "bank charges" in unsupported[0]
    assert 0.0 < coverage < 1.0


def test_abstention_is_not_penalised():
    """A correct 'the context is insufficient' answer shares almost no vocabulary with the
    context, so scoring it as unfaithful would punish the model for good calibration."""
    answer = "The context does not contain enough information to answer."
    ok, unsupported, coverage = grounding_report(answer, CONTEXT)
    assert ok is True
    assert unsupported == []
    assert coverage == 1.0


def test_grounding_threshold_is_configurable():
    answer = "Refunds arrive quickly for most customers."
    assert grounding_report(answer, CONTEXT, threshold=0.9)[0] is False
    assert grounding_report(answer, CONTEXT, threshold=0.1)[0] is True


def test_grounding_of_empty_answer_is_vacuously_true():
    assert grounding_report("", CONTEXT) == (True, [], 1.0)


def test_grounding_is_deterministic():
    a = grounding_report("Refunds are issued within 5 business days.", CONTEXT)
    b = grounding_report("Refunds are issued within 5 business days.", CONTEXT)
    assert a == b


# ---------------------------------------------------------------- judge parsing


def test_judge_parse_supported():
    assert parse_judge("SUPPORTED: the claim appears verbatim")[0] is True


def test_judge_parse_unsupported():
    supported, reason = parse_judge("UNSUPPORTED: no mention of fees")
    assert supported is False
    assert reason == "no mention of fees"


def test_judge_parse_without_colon():
    assert parse_judge("unsupported")[0] is False


def test_judge_parse_empty_is_unsupported_not_an_error():
    """Absent is not supported. Defaulting to True would inflate the metric on a truncated
    response."""
    assert parse_judge("")[0] is False


# ---------------------------------------------------------------- generation driver


def test_generation_runs_offline_without_a_key():
    result = generate_answer(OfflineLLM(), "How long do refunds take?", CONTEXT, "q1")
    assert result.grounded is not None
    assert result.model == "offline-mock"
    assert "OFFLINE MOCK" in result.answer


def test_mock_answer_is_labelled_synthetic():
    """A report that looks like it came from a real model is the failure mode here."""
    result = generate_answer(OfflineLLM(), "q", CONTEXT)
    assert "no language model was called" in result.answer.lower()


def test_judge_stays_null_when_not_requested():
    result = generate_answer(OfflineLLM(), "q", CONTEXT, judge=False)
    assert result.judge_supported is None


def test_judge_is_skipped_for_the_mock_even_when_requested():
    """The mock cannot judge itself; reporting a fabricated verdict would be worse than null."""
    result = generate_answer(OfflineLLM(), "q", CONTEXT, judge=True)
    assert result.judge_supported is None


def test_context_is_truncated_to_the_budget():
    result = generate_answer(OfflineLLM(), "q", "x" * 10_000, max_context_chars=100)
    assert len(result.context) == 100


# ---------------------------------------------------------------- summaries


def _results(n=4, grounded=True):
    from src.generate import GenerationResult

    return [
        GenerationResult(query_id=f"q{i}", answer="a", context="c", grounded=grounded,
                         coverage=1.0 if grounded else 0.5)
        for i in range(n)
    ]


def test_summary_reports_null_judge_rate_when_unavailable():
    out = summarize_generation(_results())
    assert out["grounded_rate"] == 1.0
    assert out["judge_supported_rate"] is None
    assert out["judge_n"] == 0


def test_summary_computes_judge_rate_when_judged():
    rs = _results()
    for r in rs:
        r.judge_supported = True
    out = summarize_generation(rs)
    assert out["judge_supported_rate"] == 1.0
    assert out["judge_n"] == 4


def test_summary_counts_abstentions():
    from src.generate import GenerationResult

    rs = [GenerationResult(query_id="q", answer="The context does not contain enough information.",
                           context="c", grounded=True, coverage=1.0)]
    assert summarize_generation(rs)["abstention_rate"] == 1.0


def test_summary_rejects_empty_input():
    with pytest.raises(ValueError, match="no generation results"):
        summarize_generation([])


# ---------------------------------------------------------------- diagnosis


def test_combine_blames_retrieval_when_context_is_missing():
    out = combine({"summary": {"hit@10": 0.3, "recall@10": 0.3}}, {"grounded_rate": 0.8})
    assert "retrieval-limited" in out["diagnosis"]


def test_combine_blames_generation_when_context_is_there():
    out = combine({"summary": {"hit@10": 0.9, "recall@10": 0.8}}, {"grounded_rate": 0.5})
    assert "generation-limited" in out["diagnosis"]


def test_combine_reports_healthy_when_both_are_good():
    out = combine({"summary": {"hit@10": 0.9, "recall@10": 0.8}}, {"grounded_rate": 0.95})
    assert out["diagnosis"] == "both stages healthy"


def test_combine_refuses_to_diagnose_without_data():
    assert combine({}, {})["diagnosis"] == "insufficient data"


# ---------------------------------------------------------------- llm client


def test_get_client_falls_back_to_offline_when_no_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    client = get_client("groq")
    assert isinstance(client, OfflineLLM)
    assert "no key" in client.status() or "offline" in client.status()


def test_mock_provider_is_always_available():
    assert isinstance(get_client("mock"), OfflineLLM)


def test_unknown_provider_is_rejected():
    from src.llm import LLMClient

    with pytest.raises(ValueError, match="unknown provider"):
        LLMClient(provider="openai-please")


def test_cache_key_changes_with_prompt_and_model():
    """The cache key has to include the prompt, the model and the temperature - keying on the
    question alone would serve a cached answer generated under different instructions."""
    from src.llm import LLMClient

    a = LLMClient("groq")
    m1 = a._cache_key([{"role": "user", "content": "one"}], 0.0)
    m2 = a._cache_key([{"role": "user", "content": "two"}], 0.0)
    m3 = a._cache_key([{"role": "user", "content": "one"}], 0.7)
    assert m1 != m2
    assert m1 != m3


# ---------------------------------------------------------------- gate



def test_gate_passes_on_the_stored_results():
    payload = {
        "config": {"name": "dense_256"},
        "summary": {"recall@10": 0.4716, "ndcg@10": 0.3949, "hit@10": 0.65},
        "per_query": [],
    }
    floors = ROOT / "eval" / "thresholds.json"
    existing = json.loads(floors.read_text(encoding="utf-8"))
    from eval.gate import check_payload

    failures = check_payload(payload, existing["configs"], existing["relative_margin"])
    assert failures == []


def test_gate_fails_when_a_metric_falls_below_its_floor():
    from eval.gate import check_payload

    payload = {"config": {"name": "dense_256"},
               "summary": {"recall@10": 0.30, "ndcg@10": 0.3949, "hit@10": 0.65}}
    failures = check_payload(payload, {"dense_256": {"recall@10": 0.4487}}, 0.03)
    assert any(f["metric"] == "recall@10" and f["status"] == "FAIL" for f in failures)


def test_gate_tolerates_change_inside_the_margin():
    """floor 0.4487 * 1.03 = 0.4622, so 0.4650 passes: within the margin, not a regression."""
    from eval.gate import check_payload

    payload = {"config": {"name": "dense_256"}, "summary": {"recall@10": 0.4650}}
    assert check_payload(payload, {"dense_256": {"recall@10": 0.4487}}, 0.03) == []


def test_gate_flags_a_change_just_outside_the_margin():
    from eval.gate import check_payload

    payload = {"config": {"name": "dense_256"}, "summary": {"recall@10": 0.4600}}
    assert check_payload(payload, {"dense_256": {"recall@10": 0.4487}}, 0.03) != []


def test_gate_rejects_a_config_with_no_floor():
    """No recorded floor means nothing has been verified, which is not the same as passing."""
    from eval.gate import check_payload

    failures = check_payload({"config": {"name": "brand_new"},
                              "summary": {"recall@10": 0.9}}, {"dense_256": {}}, 0.03)
    assert failures and failures[0]["status"] == "NO FLOOR"


def test_gate_flags_a_missing_metric():
    from eval.gate import check_payload

    failures = check_payload({"config": {"name": "dense_256"}, "summary": {}},
                             {"dense_256": {"recall@10": 0.4}}, 0.03)
    assert failures and failures[0]["status"] == "MISSING"


def test_gate_detects_corpus_provenance_drift():
    from eval.gate import verify_provenance
    from src.corpus import GoldenSet, GoldenQuery

    golden = GoldenSet(name="golden_v1",
                       queries=[GoldenQuery("1", "q", ["1"], 1)],
                       corpus_sha256="deadbeef" * 8)
    problems = verify_provenance({"golden": {"corpus_sha256": "cafebabe"}}, golden)
    assert problems and problems[0]["status"] == "FAIL"


def test_gate_accepts_matching_provenance():
    from eval.gate import verify_provenance
    from src.corpus import GoldenSet, GoldenQuery

    sha = "ff593e4df9933955dc3af83be0c3fa28ac7465f627e08c2e53593e734d506517"
    golden = GoldenSet(name="golden_v1",
                       queries=[GoldenQuery("1", "q", ["1"], 1)], corpus_sha256=sha)
    assert verify_provenance({"golden": {"corpus_sha256": sha}}, golden) == []
