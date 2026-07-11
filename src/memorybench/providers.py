from __future__ import annotations

import time
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


def complete(config: LLMConfig, prompt: str) -> CompletionResponse:
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
        response = ollama.chat(model=config.model, messages=[{"role": "user", "content": prompt}], options=config.params)
        usage = response if isinstance(response, dict) else response.model_dump()
        text = usage["message"]["content"]
        prompt_tokens, completion_tokens = usage.get("prompt_eval_count"), usage.get("eval_count")
    else:
        from openai import OpenAI
        params = dict(config.params)
        base_url = params.pop("base_url", None)
        client = OpenAI(base_url=base_url) if base_url else OpenAI()
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
