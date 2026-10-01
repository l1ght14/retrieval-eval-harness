"""LLM access for the eval harness: one interface, two free providers, no surprises.

Three things this module exists to get right, in order of how much they would have bitten:

1. **Keys never touch disk.** Read from the environment at call time. There is no config
   file, no .env loading, no key in any artefact. A leak here would be the one genuinely
   serious mistake in the project.

2. **A 429 does not lose the run.** Groq's free tier is 30 requests per minute and somewhere
   between 250 and 14,400 requests a day depending on model. A 1,700-query eval that dies at
   query 400 has wasted an evening. So every call retries with exponential backoff and honours
   `retry-after`, and every response is cached on disk keyed by the full request.

3. **The harness runs with no key at all.** `OfflineLLM` returns deterministic, obviously-
   synthetic answers. That keeps the retrieval metrics - the actual subject of this project -
   runnable and testable in CI with no network and no spend. The generation layer is the part
   that needs a model; the retrieval layer is pure numpy, and the two are reported separately
   precisely so this split is visible.

The cache key is the point where the Warmstart project's lesson is pre-empted: it includes
model, temperature, the full prompt, and the harness version - not just the question. Two
customers asking the same question with different tool sets must not share a cache entry.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

# Bumped whenever prompt construction changes, so an old cache entry can never be served
# against new prompt logic.
HARNESS_VERSION = "1"

PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "env_key": "GROQ_API_KEY",
        "default_model": "llama-3.3-70b-versatile",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "env_key": "GEMINI_API_KEY",
        "default_model": "gemini-2.0-flash",
    },
    "mock": {"base_url": None, "env_key": None, "default_model": "offline-mock"},
}


@dataclass
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    cached: bool = False

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class RateLimited(Exception):
    """Raised when the provider's budget is exhausted rather than merely throttled."""


class LLMClient:
    """Thin, cached, retrying wrapper over an OpenAI-compatible chat endpoint.

    Deliberately not an agent framework. The point of this project is that you can see what
    was sent and what came back, so the abstraction stays one function wide.
    """

    def __init__(
        self,
        provider: str = "groq",
        model: str | None = None,
        cache_dir: Path | None = None,
        max_retries: int = 6,
        rpm_limit: int = 25,
        timeout: float = 60.0,
    ) -> None:
        if provider not in PROVIDERS:
            raise ValueError(f"unknown provider {provider!r}; choose from {sorted(PROVIDERS)}")
        if provider == "mock":
            # Previously this constructed successfully, reported `available == True`, and then
            # raised OpenAIError("Missing credentials") on the first call - a class that looks
            # usable and fails only when used. `OfflineLLM` is the mock; there is no network
            # client for it.
            raise ValueError(
                "provider 'mock' has no network client; use OfflineLLM() or get_client('mock')"
            )
        self.provider = provider
        spec = PROVIDERS[provider]
        self.model = model or spec["default_model"]
        self.max_retries = max_retries
        self.rpm_limit = rpm_limit
        self.timeout = timeout
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._api_key = os.environ.get(spec["env_key"]) if spec["env_key"] else None
        self._client = None
        self._last_call = 0.0
        self._min_interval = 60.0 / rpm_limit if rpm_limit else 0.0
        self.stats = {"requests": 0, "cache_hits": 0, "retries": 0, "prompt_tokens": 0,
                      "completion_tokens": 0}

    # ---------------------------------------------------------------- availability

    @property
    def available(self) -> bool:
        """True if this client can actually reach a model. False means use OfflineLLM."""
        return bool(self._api_key)

    def status(self) -> str:
        if not self._api_key:
            env = PROVIDERS[self.provider]["env_key"]
            return f"no key - set {env} to use {self.provider}, falling back to offline mock"
        return f"{self.provider}:{self.model}"

    # ---------------------------------------------------------------- cache

    def _cache_key(self, messages: list[dict], temperature: float, **kw) -> str:
        payload = {
            "harness": HARNESS_VERSION,
            "provider": self.provider,
            "model": self.model,
            "temperature": temperature,
            "messages": messages,
            "extra": {k: v for k, v in sorted(kw.items()) if k != "max_tokens"},
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path | None:
        if not self.cache_dir:
            return None
        return self.cache_dir / key[:2] / f"{key}.json"

    def _read_cache(self, key: str) -> LLMResponse | None:
        path = self._cache_path(key)
        if path and path.exists():
            try:
                blob = json.loads(path.read_text(encoding="utf-8"))
                return LLMResponse(
                    text=blob["text"],
                    model=blob["model"],
                    prompt_tokens=blob.get("prompt_tokens", 0),
                    completion_tokens=blob.get("completion_tokens", 0),
                    latency_s=0.0,
                    cached=True,
                )
            except (json.JSONDecodeError, KeyError):
                return None
        return None

    def _write_cache(self, key: str, resp: LLMResponse) -> None:
        path = self._cache_path(key)
        if not path:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "text": resp.text,
                    "model": resp.model,
                    "prompt_tokens": resp.prompt_tokens,
                    "completion_tokens": resp.completion_tokens,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    # ---------------------------------------------------------------- throttle

    def _throttle(self) -> None:
        """Keep ourselves under the free tier instead of collecting 429s.

        Paces requests to `rpm_limit` requests per minute, deliberately below the published
        cap so that a shared household or a concurrent run does not push us over.
        """
        if not self._min_interval:
            return
        wait = self._min_interval - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.time()

    # ---------------------------------------------------------------- call

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self._api_key,
                base_url=PROVIDERS[self.provider]["base_url"],
                timeout=self.timeout,
                max_retries=0,  # retries are ours, so backoff is observable
            )
        return self._client

    def complete(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 512,
        use_cache: bool = True,
        **kw,
    ) -> LLMResponse:
        key = self._cache_key(messages, temperature, **kw)
        if use_cache:
            hit = self._read_cache(key)
            if hit is not None:
                self.stats["cache_hits"] += 1
                return hit

        client = self._ensure_client()
        delay = 1.0
        for attempt in range(self.max_retries):
            self._throttle()
            started = time.time()
            try:
                completion = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kw,
                )
            except Exception as exc:  # noqa: BLE001 - provider errors vary by status
                status = getattr(exc, "status_code", None)
                if status == 429 or "rate_limit" in type(exc).__name__.lower():
                    retry_after = _retry_after(exc)
                    self.stats["retries"] += 1
                    if retry_after:
                        time.sleep(min(retry_after, 60.0))
                    else:
                        time.sleep(delay + _jitter(attempt))
                        delay = min(delay * 2, 60.0)
                    continue
                raise
            usage = completion.usage
            resp = LLMResponse(
                text=completion.choices[0].message.content or "",
                model=completion.model,
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                latency_s=time.time() - started,
            )
            self.stats["requests"] += 1
            self.stats["prompt_tokens"] += resp.prompt_tokens
            self.stats["completion_tokens"] += resp.completion_tokens
            if use_cache:
                self._write_cache(key, resp)
            return resp
        raise RateLimited(
            f"exhausted {self.max_retries} retries against {self.provider} - "
            f"the free tier is spent; rerun later, the cache will pick up where it stopped"
        )


def _retry_after(exc: Exception) -> float | None:
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    for key in ("retry-after", "x-ratelimit-reset-requests", "Retry-After"):
        if key in headers:
            try:
                return float(headers[key])
            except (TypeError, ValueError):
                return None
    return None


def _jitter(attempt: int) -> float:
    """Full jitter. Without it, every worker in a concurrent run retries in lockstep."""
    import random

    return random.uniform(0, min(2**attempt, 8))


class OfflineLLM:
    """Deterministic stand-in used when no API key is present.

    The answers are obviously synthetic and are marked as such, because the danger is a
    report that looks like it came from a real model. What *is* real when running offline is
    the retrieval path: the vectors, the ranking, recall@k, MRR and nDCG are computed exactly
    as they are with a model, and those are the numbers this project is actually about.
    """

    def __init__(self, model: str = "offline-mock") -> None:
        self.model = model
        self.available = True
        self.provider = "mock"
        self.stats = {"requests": 0, "cache_hits": 0, "retries": 0, "prompt_tokens": 0,
                      "completion_tokens": 0}

    def status(self) -> str:
        return "offline mock (no key present; retrieval metrics are unaffected)"

    def complete(self, messages: list[dict], **_) -> LLMResponse:
        question = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                question = m.get("content", "")
                break
        self.stats["requests"] += 1
        token = hashlib.sha256(question.encode("utf-8")).hexdigest()[:8]
        text = (
            f"[OFFLINE MOCK - no language model was called] "
            f"based on the retrieved context for query {token}, the relevant material would "
            f"be summarised here. Retrieval metrics in this run are genuine."
        )
        return LLMResponse(text=text, model=self.model, completion_tokens=len(text.split()))


def get_client(provider: str = "groq", cache_dir: Path | None = None, **kw) -> LLMClient | OfflineLLM:
    """Return a real client when a key is present, otherwise the offline mock.

    `provider="mock"` returns the mock directly. Handing back an `LLMClient` configured for
    "mock" would look available while having no base URL, and `complete()` would fail on the
    first call rather than up front - so the mock is returned as the mock.
    """
    if provider == "mock":
        return OfflineLLM()
    client = LLMClient(provider=provider, cache_dir=cache_dir, **kw)
    return client if client.available else OfflineLLM()
