import json

import httpx
import pytest

import arc3cb.transport as transport_mod
from arc3cb.transport import (
    CerebrasApiError,
    CerebrasTransport,
    ContextLimitError,
    ModelConfig,
    TransportError,
    UsageMeter,
)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    sleeps = []
    monkeypatch.setattr(transport_mod.time, "sleep", lambda s: sleeps.append(s))
    return sleeps


def make_transport(handler, tmp_path, **kwargs):
    cfg = kwargs.pop(
        "model_cfg",
        ModelConfig(
            id="test-model",
            price_input_per_mtok=1.0,
            price_output_per_mtok=2.0,
            reasoning_effort="low",
        ),
    )
    meter = UsageMeter(tmp_path / "usage.jsonl")
    t = CerebrasTransport(
        model_cfg=cfg,
        meter=meter,
        api_key="test-key-not-real",
        base_url="https://cerebras.test/v1",
        http_transport=httpx.MockTransport(handler),
        **kwargs,
    )
    return t, meter


def chat_response(text="hello", prompt_tokens=100, completion_tokens=10):
    return {
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def test_chat_success_records_usage_and_cost(tmp_path):
    seen = {}

    def handler(request):
        seen["payload"] = json.loads(request.content)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json=chat_response(prompt_tokens=1_000_000, completion_tokens=500_000))

    t, meter = make_transport(handler, tmp_path)
    result = t.chat([{"role": "user", "content": "hi"}], purpose="agent")
    assert result.text == "hello"
    assert seen["auth"] == "Bearer test-key-not-real"
    assert seen["payload"]["model"] == "test-model"
    assert seen["payload"]["max_completion_tokens"] == 8192
    assert seen["payload"]["reasoning_effort"] == "low"
    # $1/M input * 1M + $2/M output * 0.5M
    assert meter.cost_usd == pytest.approx(2.0)
    lines = (tmp_path / "usage.jsonl").read_text().strip().splitlines()
    entry = json.loads(lines[0])
    assert entry["prompt_tokens"] == 1_000_000
    assert entry["purpose"] == "agent"


def test_retries_on_429_then_succeeds(tmp_path, no_sleep):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429, text="rate limited", headers={"retry-after": "7"})
        return httpx.Response(200, json=chat_response())

    t, _ = make_transport(handler, tmp_path, backoff_base_s=0.01)
    result = t.chat([{"role": "user", "content": "hi"}])
    assert result.text == "hello"
    assert calls["n"] == 3
    assert 7.0 in no_sleep  # honored Retry-After


def test_retries_on_5xx_and_transport_errors(tmp_path):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, text="overloaded")
        if calls["n"] == 2:
            raise httpx.ConnectError("boom", request=request)
        return httpx.Response(200, json=chat_response())

    t, _ = make_transport(handler, tmp_path, backoff_base_s=0.01)
    assert t.chat([{"role": "user", "content": "hi"}]).text == "hello"
    assert calls["n"] == 3


def test_gives_up_after_max_retries(tmp_path):
    def handler(request):
        return httpx.Response(500, text="nope")

    t, _ = make_transport(handler, tmp_path, max_retries=2, backoff_base_s=0.01)
    with pytest.raises(TransportError, match="after 3 attempts"):
        t.chat([{"role": "user", "content": "hi"}])


def test_non_retryable_4xx_raises_immediately(tmp_path):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(401, text="bad key")

    t, _ = make_transport(handler, tmp_path)
    with pytest.raises(CerebrasApiError):
        t.chat([{"role": "user", "content": "hi"}])
    assert calls["n"] == 1


def test_context_limit_error_is_specific(tmp_path):
    def handler(request):
        return httpx.Response(
            400, text='{"message": "prompt exceeds maximum context length of 8192"}'
        )

    t, _ = make_transport(handler, tmp_path)
    with pytest.raises(ContextLimitError):
        t.chat([{"role": "user", "content": "hi"}])


def test_verify_served_unknown_model(tmp_path):
    def handler(request):
        return httpx.Response(200, json={"data": [{"id": "other-model"}]})

    t, _ = make_transport(handler, tmp_path)
    with pytest.raises(TransportError, match="not served"):
        t.verify_served()


def test_verify_served_context_too_small(tmp_path):
    def handler(request):
        return httpx.Response(
            200, json={"data": [{"id": "test-model", "context_length": 8192}]}
        )

    t, _ = make_transport(handler, tmp_path)
    with pytest.raises(TransportError, match="paid-tier"):
        t.verify_served(min_context=90_000)


def test_image_probe_false_on_rejection(tmp_path):
    def handler(request):
        payload = json.loads(request.content)
        if any(isinstance(m.get("content"), list) for m in payload["messages"]):
            return httpx.Response(400, text="image input not supported")
        return httpx.Response(200, json=chat_response())

    cfg = ModelConfig(id="test-model", supports_images="probe")
    t, _ = make_transport(handler, tmp_path, model_cfg=cfg)
    assert t.probe_image_support() is False

    cfg2 = ModelConfig(id="test-model", supports_images="never")
    t2, _ = make_transport(handler, tmp_path, model_cfg=cfg2)
    assert t2.probe_image_support() is False
