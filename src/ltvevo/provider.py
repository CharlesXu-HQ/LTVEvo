"""Minimal OpenAI-compatible JSON client for the experiment Agent."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from urllib.parse import urlsplit


class IncompleteResponseError(RuntimeError):
    def __init__(self, finish_reason: str | None):
        self.finish_reason = finish_reason
        super().__init__(f"Agent response incomplete: {finish_reason}")


@dataclass(frozen=True)
class ApiProvider:
    url: str
    model: str
    api_key: str = field(repr=False)
    thinking: str = "enabled"
    iteration_effort: str = "high"
    review_effort: str = "max"

    def __post_init__(self) -> None:
        parts = urlsplit(self.url)
        if (parts.scheme not in {"http", "https"} or not parts.netloc or parts.username
                or parts.password or parts.query or parts.fragment):
            raise ValueError("provider URL must be an HTTP(S) base URL or chat/completions endpoint")
        if not self.model or not self.api_key:
            raise ValueError("provider model and API key are required")
        if self.thinking not in {"enabled", "omit"}:
            raise ValueError("thinking must be enabled or omitted; disabled defeats high/max effort")
        if self.iteration_effort not in {"high", "max"} or self.review_effort != "max":
            raise ValueError("iteration effort must be high/max and anomaly review must be max")

    @property
    def endpoint(self) -> str:
        url = self.url.rstrip("/")
        return url if url.endswith("/chat/completions") else f"{url}/chat/completions"


def request_json(provider: ApiProvider, effort: str, messages: list[dict], *,
                 max_tokens: int = 16384, timeout: int = 180) -> dict:
    if effort not in {"high", "max"}:
        raise ValueError("Agent effort must be high or max")
    payload = {"model": provider.model, "reasoning_effort": effort,
               "response_format": {"type": "json_object"},
               "max_tokens": max_tokens, "messages": messages}
    if provider.thinking == "enabled":
        payload["thinking"] = {"type": "enabled"}
    request = urllib.request.Request(
        provider.endpoint, data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {provider.api_key}",
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read(1024).decode("utf-8", errors="replace")
        detail = detail.replace(provider.api_key, "[redacted]")
        raise RuntimeError(f"Agent provider returned HTTP {error.code}: {detail}") from None
    except urllib.error.URLError as error:
        raise RuntimeError(f"Agent provider connection failed: {error.reason}") from None
    choice = result["choices"][0]
    if choice.get("finish_reason") != "stop":
        raise IncompleteResponseError(choice.get("finish_reason"))
    answer = json.loads(choice["message"]["content"])
    if not isinstance(answer, dict):
        raise ValueError("Agent provider must return a JSON object")
    return answer
