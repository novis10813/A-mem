from amem.benchmark.hooks import HookContext, TokenUsageHook, summarize_usage
from amem.benchmark.schemas import UsageRecord


def test_token_usage_hook_records_reported_usage():
    hook = TokenUsageHook()
    hook.after_llm_call(
        HookContext(phase="qa", sample_id=1, qa_idx=2),
        call_id="answer",
        provider="openai",
        model="gpt-4o-mini",
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        latency_seconds=0.25,
    )

    assert hook.records == (
        UsageRecord(
            phase="qa",
            call_id="answer",
            provider="openai",
            model="gpt-4o-mini",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            latency_seconds=0.25,
            source="reported",
            metadata={"sample_id": 1, "qa_idx": 2},
        ),
    )


def test_token_usage_hook_estimates_when_usage_missing():
    hook = TokenUsageHook(estimate_when_missing=True, tokenizer="words")
    hook.after_llm_call(
        HookContext(phase="qa", sample_id=1, qa_idx=2),
        call_id="answer",
        provider="ollama",
        model="llama3.2:1b",
        prompt="one two three",
        completion="four five",
    )

    record = hook.records[0]
    assert record.estimated_tokens == 5
    assert record.source == "estimated"
    assert record.tokenizer == "words"


def test_summarize_usage_keeps_reported_and_estimated_separate():
    summary = summarize_usage(
        [
            UsageRecord(phase="qa", call_id="a", total_tokens=10, source="reported"),
            UsageRecord(phase="qa", call_id="b", estimated_tokens=7, source="estimated"),
        ]
    )

    assert summary["by_source"]["reported"]["total_tokens"] == 10
    assert summary["by_source"]["estimated"]["estimated_tokens"] == 7
    assert summary["calls"] == 2
