"""Cerebras shared-inference client (OpenAI-compatible chat completions).

Design constraints this module owns:
- exponential backoff with jitter on 429/5xx and transport errors, honoring
  Retry-After;
- token usage captured from the ``usage`` field of every response and persisted
  to ``usage.jsonl`` with a per-call dollar cost from configured prices;
- fail-fast with an actionable message when the served context limit is below
  the configured reset threshold (the Cerebras free tier caps context far below
  what this harness needs — a paid-tier key is required);
- a startup capability probe: one tiny image-content message decides whether a
  model runs multimodal or text-only.

No other module talks HTTP to Cerebras.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

DEFAULT_BASE_URL = "https://api.cerebras.ai/v1"
RETRYABLE_STATUSES = {408, 429, 500, 502, 503, 504}

# 1x1 transparent PNG, used only to probe whether a model accepts image parts.
_PROBE_PNG_DATA_URI = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


class TransportError(RuntimeError):
    pass


class CerebrasApiError(TransportError):
    def __init__(self, status: int, body: str):
        self.status = status
        self.body = body
        super().__init__(f"Cerebras API error {status}: {body[:2000]}")


class ContextLimitError(CerebrasApiError):
    """Raised when the served context window rejects our request.

    Cerebras free-trial keys cap context (65k as of 2026-08) below the paid
    131k; this harness needs the paid tier to run its configured thresholds.
    """


@dataclass
class ModelConfig:
    id: str
    temperature: float = 0.7
    max_output_tokens: int = 8192
    max_tokens_param: str = "max_completion_tokens"
    reasoning_effort: str | None = None
    reasoning_effort_param: str = "reasoning_effort"
    stop: list[str] = field(default_factory=list)
    context_window: int = 131072  # expected; verified against the served value when reported
    price_input_per_mtok: float = 0.0
    price_output_per_mtok: float = 0.0
    supports_images: str = "never"  # "never" | "probe"
    extra_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    latency_s: float
    raw: dict


class UsageMeter:
    """Accumulates token usage and dollars; persists one JSON line per call."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cost_usd = 0.0
        self.calls = 0
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self, model_cfg: ModelConfig, purpose: str, result: ChatResult, attempts: int
    ) -> float:
        cost = (
            result.prompt_tokens / 1e6 * model_cfg.price_input_per_mtok
            + result.completion_tokens / 1e6 * model_cfg.price_output_per_mtok
        )
        self.prompt_tokens += result.prompt_tokens
        self.completion_tokens += result.completion_tokens
        self.cost_usd += cost
        self.calls += 1
        if self.path:
            usage = (result.raw or {}).get("usage") or {}
            entry = {
                "ts": time.time(),
                "model": model_cfg.id,
                "purpose": purpose,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
                "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get(
                    "reasoning_tokens", 0
                ),
                "cost_usd": round(cost, 6),
                "latency_s": round(result.latency_s, 3),
                "attempts": attempts,
                "finish_reason": result.finish_reason,
            }
            with self.path.open("a") as f:
                f.write(json.dumps(entry) + "\n")
        return cost

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


_CONTEXT_LIMIT_MARKERS = (
    "context length",
    "context window",
    "maximum context",
    "max context",
    "input length",
    "too many tokens",
    "token limit",
    "prompt is too long",
    "prompt too long",
    "exceeds maximum",
    "exceeded maximum",
    "max_model_len",
    "request too large",
    "reduce the length of the input",
)


def _is_context_limit_message(body: str) -> bool:
    lowered = body.lower()
    return any(marker in lowered for marker in _CONTEXT_LIMIT_MARKERS)


class CerebrasTransport:
    def __init__(
        self,
        model_cfg: ModelConfig,
        meter: UsageMeter,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_s: float = 600.0,
        max_retries: int = 8,
        backoff_base_s: float = 2.0,
        backoff_cap_s: float = 120.0,
        http_transport: httpx.BaseTransport | None = None,
    ):
        self.model_cfg = model_cfg
        self.meter = meter
        self.base_url = (base_url or os.environ.get("CEREBRAS_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        key = api_key or os.environ.get("CEREBRAS_API_KEY")
        if not key:
            raise TransportError(
                "CEREBRAS_API_KEY is not set; put it in .env or the environment "
                "(see .env.example). A paid-tier key is required."
            )
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self.backoff_cap_s = backoff_cap_s
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            timeout=timeout_s,
            transport=http_transport,
        )

    def close(self) -> None:
        self._client.close()

    # -- low level -----------------------------------------------------------

    def _request(self, method: str, path: str, payload: dict | None = None) -> tuple[dict, int]:
        """Request with retry; returns (parsed json, attempts used)."""
        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 2):
            try:
                resp = self._client.request(method, path, json=payload)
            except httpx.HTTPError as e:
                last_err = TransportError(f"transport failure calling {path}: {e!r}")
            else:
                if resp.status_code == 200:
                    return resp.json(), attempt
                body = resp.text
                if resp.status_code == 400 and _is_context_limit_message(body):
                    raise ContextLimitError(resp.status_code, body)
                if resp.status_code not in RETRYABLE_STATUSES:
                    raise CerebrasApiError(resp.status_code, body)
                last_err = CerebrasApiError(resp.status_code, body)
                retry_after = resp.headers.get("retry-after")
                if retry_after:
                    try:
                        time.sleep(min(float(retry_after), self.backoff_cap_s))
                        continue
                    except ValueError:
                        pass
            if attempt <= self.max_retries:
                delay = min(self.backoff_base_s * (2 ** (attempt - 1)), self.backoff_cap_s)
                time.sleep(random.uniform(0, delay))
        raise TransportError(
            f"giving up on {path} after {self.max_retries + 1} attempts: {last_err}"
        )

    # -- API surface ---------------------------------------------------------

    def list_models(self) -> list[dict]:
        data, _ = self._request("GET", "/models")
        return data.get("data", [])

    def public_catalog(self) -> list[dict]:
        """The unauthenticated public catalog, which — unlike /v1/models — carries
        ``limits.max_context_length``, ``limits.max_completion_tokens`` and
        per-token pricing. Returns [] if unreachable (e.g. custom base URLs)."""
        public_url = self.base_url.removesuffix("/v1") + "/public/v1/models"
        try:
            resp = self._client.get(public_url)
            if resp.status_code != 200:
                return []
            return resp.json().get("data", [])
        except httpx.HTTPError:
            return []

    def verify_served(self, min_context: int | None = None) -> dict:
        """Check the configured model is actually served; sanity-check context.

        Returns the served model entry (public-catalog entry when available,
        since it carries limits). Fails fast with a tier explanation when the
        served context limit is below ``min_context`` — running with a context
        window smaller than the reset threshold would corrupt runs mid-game.
        """
        models = self.list_models()
        by_id = {m.get("id"): m for m in models}
        entry = by_id.get(self.model_cfg.id)
        if entry is None:
            raise TransportError(
                f"model {self.model_cfg.id!r} is not served at {self.base_url}; "
                f"served models: {', '.join(sorted(by_id)) or '(none)'}"
            )
        public = {m.get("id"): m for m in self.public_catalog()}.get(self.model_cfg.id)
        if public:
            entry = {**entry, **public}
        if min_context:
            limits = entry.get("limits") or {}
            served_ctx = None
            for source in (limits, entry):
                for key in ("max_context_length", "context_length", "context_window"):
                    if isinstance(source.get(key), int):
                        served_ctx = source[key]
                        break
                if served_ctx is not None:
                    break
            if served_ctx is not None and served_ctx < min_context:
                raise TransportError(
                    f"served context for {self.model_cfg.id} is {served_ctx} tokens, below "
                    f"the configured threshold {min_context}. Cerebras free-trial keys cap "
                    "context below the paid tier — this harness requires a paid-tier key "
                    "(or lower context_reset_input_tokens accordingly)."
                )
        return entry

    def chat(
        self,
        messages: list[dict],
        purpose: str = "agent",
        max_output_tokens: int | None = None,
    ) -> ChatResult:
        cfg = self.model_cfg
        payload: dict[str, Any] = {
            "model": cfg.id,
            "messages": messages,
            "temperature": cfg.temperature,
            cfg.max_tokens_param: max_output_tokens or cfg.max_output_tokens,
        }
        if cfg.stop:
            payload["stop"] = cfg.stop
        if cfg.reasoning_effort:
            payload[cfg.reasoning_effort_param] = cfg.reasoning_effort
        payload.update(cfg.extra_params)
        t0 = time.monotonic()
        data, attempts = self._request("POST", "/chat/completions", payload)
        latency = time.monotonic() - t0
        try:
            choice = data["choices"][0]
            text = choice["message"].get("content") or ""
            finish = choice.get("finish_reason", "")
        except (KeyError, IndexError) as e:
            raise TransportError(f"malformed chat response: {json.dumps(data)[:2000]}") from e
        usage = data.get("usage") or {}
        result = ChatResult(
            text=text,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            finish_reason=finish,
            latency_s=latency,
            raw=data,
        )
        self.meter.record(cfg, purpose, result, attempts)
        return result

    def probe_image_support(self) -> bool:
        """True if the endpoint accepts an image content part for this model."""
        if self.model_cfg.supports_images != "probe":
            return False
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Reply with the single word: ok"},
                    {"type": "image_url", "image_url": {"url": _PROBE_PNG_DATA_URI}},
                ],
            }
        ]
        payload = {
            "model": self.model_cfg.id,
            "messages": messages,
            self.model_cfg.max_tokens_param: 8,
        }
        try:
            data, attempts = self._request("POST", "/chat/completions", payload)
        except CerebrasApiError:
            return False
        usage = data.get("usage") or {}
        self.meter.record(
            self.model_cfg,
            "image-probe",
            ChatResult(
                text="",
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
                finish_reason="probe",
                latency_s=0.0,
                raw=data,
            ),
            attempts,
        )
        return True
