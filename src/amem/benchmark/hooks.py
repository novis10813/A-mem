"""Hook interfaces and standard observability hooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .schemas import UsageRecord


@dataclass(frozen=True)
class HookContext:
    phase: str
    sample_id: int | None = None
    qa_idx: int | None = None
    construction_run: int | None = None
    qa_run: int | None = None
    metadata: Mapping[str, Any] | None = None

    def as_metadata(self) -> dict[str, Any]:
        data = dict(self.metadata or {})
        for key in ("sample_id", "qa_idx", "construction_run", "qa_run"):
            value = getattr(self, key)
            if value is not None:
                data[key] = value
        return data


class Hook:
    def before_llm_call(self, context: HookContext, **kwargs: Any) -> None:
        pass

    def after_llm_call(self, context: HookContext, **kwargs: Any) -> None:
        pass


class NoOpHook(Hook):
    pass


class TokenUsageHook(Hook):
    def __init__(self, *, estimate_when_missing: bool = False, tokenizer: str = "words") -> None:
        self.estimate_when_missing = estimate_when_missing
        self.tokenizer = tokenizer
        self._records: list[UsageRecord] = []

    @property
    def records(self) -> tuple[UsageRecord, ...]:
        return tuple(self._records)

    def after_llm_call(
        self,
        context: HookContext,
        *,
        call_id: str,
        provider: str | None = None,
        model: str | None = None,
        usage: Mapping[str, Any] | None = None,
        prompt: str | None = None,
        completion: str | None = None,
        latency_seconds: float | None = None,
        cost_usd: float | None = None,
    ) -> None:
        if usage:
            record = UsageRecord(
                phase=context.phase,
                call_id=call_id,
                provider=provider,
                model=model,
                prompt_tokens=_int_or_none(usage.get("prompt_tokens")),
                completion_tokens=_int_or_none(usage.get("completion_tokens")),
                total_tokens=_int_or_none(usage.get("total_tokens")),
                cost_usd=cost_usd,
                latency_seconds=latency_seconds,
                source="reported",
                metadata=context.as_metadata(),
            )
        elif self.estimate_when_missing:
            record = UsageRecord(
                phase=context.phase,
                call_id=call_id,
                provider=provider,
                model=model,
                estimated_tokens=_estimate_tokens(prompt, completion),
                cost_usd=cost_usd,
                latency_seconds=latency_seconds,
                source="estimated",
                tokenizer=self.tokenizer,
                metadata=context.as_metadata(),
            )
        else:
            return
        self._records.append(record)


def summarize_usage(records: Sequence[UsageRecord]) -> dict[str, Any]:
    summary: dict[str, Any] = {"calls": len(records), "by_source": {}}
    for record in records:
        bucket = summary["by_source"].setdefault(
            record.source,
            {
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "estimated_tokens": 0,
                "latency_seconds": 0.0,
                "cost_usd": 0.0,
            },
        )
        bucket["calls"] += 1
        bucket["prompt_tokens"] += record.prompt_tokens or 0
        bucket["completion_tokens"] += record.completion_tokens or 0
        bucket["total_tokens"] += record.total_tokens or 0
        bucket["estimated_tokens"] += record.estimated_tokens or 0
        bucket["latency_seconds"] += record.latency_seconds or 0.0
        bucket["cost_usd"] += record.cost_usd or 0.0
    return summary


def _estimate_tokens(prompt: str | None, completion: str | None) -> int:
    return len(((prompt or "") + " " + (completion or "")).split())


def _int_or_none(value: Any) -> int | None:
    return None if value is None else int(value)
