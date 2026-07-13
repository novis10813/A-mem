from __future__ import annotations

import json
import time
import urllib.request
from typing import Any

from pydantic import BaseModel, ConfigDict

from .config import LLMConfig


class CompletionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    usage_source: str = "reported"
    latency_ms: float
    raw: dict[str, Any] | None = None


def complete(config: LLMConfig, prompt: str, *, temperature: float | None = None) -> CompletionResponse:
    started = time.perf_counter()
    if config.provider == "fake":
        text = str(config.params.get("response", ""))
        return CompletionResponse(
            text=text, prompt_tokens=len(prompt.split()), completion_tokens=len(text.split()),
            total_tokens=len(prompt.split()) + len(text.split()), usage_source="estimated",
            latency_ms=(time.perf_counter() - started) * 1000,
        )
    if config.provider == "ollama":
        import ollama
        options = dict(config.params)
        if temperature is not None:
            options["temperature"] = temperature
        response = ollama.chat(model=config.model, messages=[{"role": "user", "content": prompt}], options=options)
        usage = response if isinstance(response, dict) else response.model_dump()
        text = usage["message"]["content"]
        prompt_tokens, completion_tokens = usage.get("prompt_eval_count"), usage.get("eval_count")
    elif config.provider == "sglang":
        params = dict(config.params)
        base_url = params.pop("base_url", None)
        if base_url is None:
            host = params.pop("host", "http://localhost")
            port = int(params.pop("port", 30000))
            base_url = f"{host}:{port}"
        if temperature is not None:
            params["temperature"] = temperature
        payload = json.dumps({"text": prompt, "sampling_params": params}).encode("utf-8")
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=120) as result:
            body = json.loads(result.read().decode("utf-8"))
        text = str(body.get("text", ""))
        prompt_tokens = body.get("prompt_tokens")
        completion_tokens = body.get("completion_tokens")
    else:
        from openai import OpenAI
        params = dict(config.params)
        base_url = params.pop("base_url", None)
        api_key = params.pop("api_key", None)
        if config.provider == "vllm" and base_url is None:
            host = params.pop("host", "http://localhost")
            port = int(params.pop("port", 30000))
            base_url = f"{host}:{port}/v1"
        if temperature is not None:
            params["temperature"] = temperature
        client = (
            OpenAI(base_url=base_url, api_key=api_key or "EMPTY")
            if base_url else OpenAI(api_key=api_key)
        )
        response = client.chat.completions.create(model=config.model, messages=[{"role": "user", "content": prompt}], **params)
        text = response.choices[0].message.content or ""
        prompt_tokens = response.usage.prompt_tokens if response.usage else None
        completion_tokens = response.usage.completion_tokens if response.usage else None
    reported = prompt_tokens is not None and completion_tokens is not None
    if not reported:
        prompt_tokens, completion_tokens = len(prompt.split()), len(text.split())
    return CompletionResponse(
        text=text, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens, usage_source="reported" if reported else "estimated",
        latency_ms=(time.perf_counter() - started) * 1000,
    )
