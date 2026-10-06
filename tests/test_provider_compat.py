"""Provider request compatibility at the HTTP boundary."""

import io
import json

from ltvevo.provider import ApiProvider, request_json


def test_disabled_thinking_can_omit_unsupported_reasoning_effort(monkeypatch):
    from ltvevo import provider as module

    requests = []

    def fake_urlopen(request, timeout):
        requests.append(json.loads(request.data))
        return io.BytesIO(
            b'{"choices":[{"finish_reason":"stop","message":{"content":"{\\"ok\\":true}"}}]}'
        )

    monkeypatch.setattr(module.urllib.request, "urlopen", fake_urlopen)
    client = ApiProvider(
        "https://api.example.com/v1", "example-model", "secret",
        thinking="disabled", send_reasoning_effort=False,
    )

    assert request_json(client, "high", [{"role": "user", "content": "Return JSON"}]) == {"ok": True}
    assert requests[0]["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in requests[0]
    assert "secret" not in repr(client)


def test_omitted_thinking_still_sends_requested_effort_by_default(monkeypatch):
    from ltvevo import provider as module

    requests = []

    def fake_urlopen(request, timeout):
        requests.append(json.loads(request.data))
        return io.BytesIO(
            b'{"choices":[{"finish_reason":"stop","message":{"content":"{\\"ok\\":true}"}}]}'
        )

    monkeypatch.setattr(module.urllib.request, "urlopen", fake_urlopen)
    client = ApiProvider(
        "https://api.example.com/v1", "example-model", "secret", thinking="omit",
    )

    assert request_json(client, "max", [{"role": "user", "content": "Return JSON"}]) == {"ok": True}
    assert "thinking" not in requests[0]
    assert requests[0]["reasoning_effort"] == "max"
