"""The LLM wrapper's failure handling, against a fake provider.

Retry, backoff and caching are the parts of this project most likely to fail *quietly* - a
run that dies at query 400 of 200 looks like a bug report, and a cache that never hits looks
like a working cache until the bill arrives. None of it can be tested against a real free-tier
API without spending the quota it exists to protect, so the provider is faked and the
transport behaviour is pinned directly.

The fake records what it was asked to do, which is how the assertions below check that a
429 was retried rather than merely not crashing.
"""
from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

from src.llm import LLMClient, OfflineLLM, RateLimited


class FakeRateLimitError(Exception):
    """Mimics the provider's 429 exception, including a response with headers."""

    status_code = 429

    def __init__(self, retry_after: str | None = None) -> None:
        super().__init__("rate limit exceeded")
        self.response = SimpleNamespace(headers={"retry-after": retry_after} if retry_after else {})


class FakeCompletions:
    def __init__(self, script, calls):
        self._script = script
        self._calls = calls

    def create(self, **kwargs):
        self._calls.append(kwargs)
        item = self._script.pop(0) if self._script else "ok"
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=f"answer:{len(self._calls)}"))],
            model=kwargs.get("model", "fake-model"),
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7),
            model_dump=lambda: {},
        )


class FakeOpenAI:
    def __init__(self, script, calls):
        self.chat = SimpleNamespace(completions=FakeCompletions(script, calls))


def _client(tmp_path, script, **kw):
    c = LLMClient(provider="groq", cache_dir=tmp_path, **kw)
    c._api_key = "test-key-not-a-real-secret"
    calls: list = []
    c._client = FakeOpenAI(script, calls)
    return c, calls


MSGS = [{"role": "user", "content": "why is the sky blue?"}]


# ---------------------------------------------------------------- retries


def test_a_429_is_retried_and_then_succeeds(tmp_path):
    client, calls = _client(tmp_path, [FakeRateLimitError(), FakeRateLimitError(), "ok"],
                            max_retries=5, rpm_limit=0)
    resp = client.complete(MSGS)
    assert resp.text.startswith("answer:")
    assert len(calls) == 3
    assert client.stats["retries"] == 2


def test_backoff_grows_between_retries(tmp_path):
    """Two retries must take longer than one. Constant backoff against a rate limit just means
    colliding with the other clients that were also told to wait."""
    one, _ = _client(tmp_path, [FakeRateLimitError(), "ok"], max_retries=3, rpm_limit=0)
    t1 = time.time()
    one.complete(MSGS, use_cache=False)
    single = time.time() - t1

    two, _ = _client(tmp_path, [FakeRateLimitError(), FakeRateLimitError(), "ok"],
                     max_retries=3, rpm_limit=0)
    t2 = time.time()
    two.complete(MSGS, use_cache=False)
    double = time.time() - t2
    assert double > single


def test_retry_after_header_is_honoured(tmp_path):
    """A provider saying 'come back in 2 seconds' is better information than our own guess."""
    c, calls = _client(tmp_path, [FakeRateLimitError(retry_after="1"), "ok"],
                       max_retries=3, rpm_limit=0)
    t0 = time.time()
    c.complete(MSGS, use_cache=False)
    assert time.time() - t0 >= 1.0


def test_exhausted_retries_raise_ratelimited_not_a_generic_error(tmp_path):
    """The message has to say the quota is spent, because that is the one case where the
    correct response is to stop rather than to debug."""
    client, _ = _client(tmp_path, [FakeRateLimitError()] * 10, max_retries=2, rpm_limit=0)
    with pytest.raises(RateLimited, match="free tier is spent"):
        client.complete(MSGS, use_cache=False)


def test_a_non_rate_limit_error_propagates_immediately(tmp_path):
    client, calls = _client(tmp_path, [FakeRateLimitError()], max_retries=5, rpm_limit=0)
    client._client = FakeOpenAI([ValueError("bad request")], calls)
    with pytest.raises(ValueError, match="bad request"):
        client.complete(MSGS, use_cache=False)
    assert len(calls) == 1  # not retried


# ---------------------------------------------------------------- cache


def test_a_second_identical_call_is_served_from_cache(tmp_path):
    client, calls = _client(tmp_path, ["ok", "ok"])
    first = client.complete(MSGS)
    second = client.complete(MSGS)
    assert len(calls) == 1
    assert second.cached is True
    assert first.cached is False
    assert client.stats["cache_hits"] == 1


def test_a_different_prompt_misses_the_cache(tmp_path):
    client, calls = _client(tmp_path, ["ok", "ok"])
    client.complete(MSGS)
    client.complete([{"role": "user", "content": "why is the sky red?"}])
    assert len(calls) == 2


def test_use_cache_false_bypasses_both_read_and_write(tmp_path):
    client, calls = _client(tmp_path, ["ok", "ok"])
    client.complete(MSGS, use_cache=False)
    client.complete(MSGS, use_cache=False)
    assert len(calls) == 2


def test_cache_records_tokens_and_model(tmp_path):
    client, _ = _client(tmp_path, ["ok"])
    client.complete(MSGS)
    files = list(tmp_path.rglob("*.json"))
    assert files
    blob = json.loads(files[0].read_text(encoding="utf-8"))
    assert blob["prompt_tokens"] == 11
    assert blob["completion_tokens"] == 7


def test_a_corrupt_cache_entry_falls_through_to_the_api(tmp_path):
    """A truncated write from a killed process must degrade to a live call, not an exception."""
    client, calls = _client(tmp_path, ["ok"])
    client.complete(MSGS)
    for f in tmp_path.rglob("*.json"):
        f.write_text("{ truncated", encoding="utf-8")
    assert client.complete(MSGS).cached is False
    assert len(calls) == 2


# ---------------------------------------------------------------- availability


def test_availability_reflects_the_key():
    assert LLMClient("groq").available is False
    assert LLMClient("gemini").available is False


def test_mock_provider_has_no_network_client():
    """It used to construct cleanly, report available=True, then raise OpenAIError on the
    first call. A class that looks usable and fails only when used is worse than one that
    refuses to be built."""
    with pytest.raises(ValueError, match="no network client"):
        LLMClient("mock")


def test_status_mentions_the_env_var_to_set():
    assert "GROQ_API_KEY" in LLMClient("groq").status()
    assert "GEMINI_API_KEY" in LLMClient("gemini").status()


def test_defaults_differ_by_provider():
    assert LLMClient("groq").model == "llama-3.3-70b-versatile"
    assert LLMClient("gemini").model == "gemini-2.0-flash"


def test_token_accounting_accumulates(tmp_path):
    client, _ = _client(tmp_path, ["ok", "ok"])
    client.complete(MSGS, use_cache=False)
    client.complete(MSGS, use_cache=False)
    assert client.stats["requests"] == 2
    assert client.stats["prompt_tokens"] == 22
    assert client.stats["completion_tokens"] == 14


def test_total_tokens_is_the_sum():
    from src.llm import LLMResponse

    assert LLMResponse(text="x", model="m", prompt_tokens=3, completion_tokens=4).total_tokens == 7


# ---------------------------------------------------------------- throttle


def test_throttle_spaces_calls_to_the_configured_rpm():
    client = LLMClient("groq", rpm_limit=1200)  # 20/s -> 50ms apart
    client._api_key = "k"
    start = time.time()
    client._throttle()
    client._throttle()
    client._throttle()
    assert time.time() - start >= 0.09


def test_rpm_limit_zero_disables_throttling():
    client = LLMClient("groq", rpm_limit=0)
    client._api_key = "k"
    start = time.time()
    for _ in range(5):
        client._throttle()
    assert time.time() - start < 0.5


# ---------------------------------------------------------------- offline mock


def test_offline_mock_is_deterministic():
    a = OfflineLLM().complete(MSGS)
    b = OfflineLLM().complete(MSGS)
    assert a.text == b.text


def test_offline_mock_differs_per_question():
    a = OfflineLLM().complete([{"role": "user", "content": "one"}])
    b = OfflineLLM().complete([{"role": "user", "content": "two"}])
    assert a.text != b.text


def test_offline_mock_reads_the_last_user_message():
    out = OfflineLLM().complete([
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ignored"},
        {"role": "user", "content": "the real question"},
    ])
    expected = OfflineLLM().complete([{"role": "user", "content": "the real question"}])
    assert out.text == expected.text
