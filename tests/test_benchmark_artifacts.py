import json
from pathlib import Path

from amem.benchmark.artifacts import (
    read_jsonl,
    read_memory_store,
    write_jsonl,
    write_memory_store,
    write_qa_results_jsonl,
    write_usage_summary,
)
from amem.benchmark.schemas import MemoryRecord, MemoryStore, QAResult, UsageRecord


def test_memory_store_json_round_trip(tmp_path: Path):
    path = tmp_path / "memory_store_sample_0.json"
    store = MemoryStore(
        sample_id=0,
        records=(MemoryRecord(memory_id="m0", sample_id=0, text="hello"),),
    )

    write_memory_store(path, store)

    assert read_memory_store(path) == store
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["records"][0]["memory_id"] == "m0"


def test_jsonl_helpers_create_parent_dirs(tmp_path: Path):
    path = tmp_path / "nested" / "rows.jsonl"
    write_jsonl(path, [{"a": 1}, {"b": 2}])

    assert read_jsonl(path) == [{"a": 1}, {"b": 2}]


def test_qa_results_jsonl_and_usage_summary(tmp_path: Path):
    result = QAResult(
        experiment_id="exp",
        construction_run=0,
        qa_run=0,
        sample_id=0,
        qa_idx=0,
        question="q",
        reference="r",
        prediction="p",
        category=1,
        metrics={"f1": 0.5},
        retrieval={"items": []},
        context={"text": ""},
        prompt=None,
        usage=(UsageRecord(phase="qa", call_id="answer", total_tokens=9),),
    )

    write_qa_results_jsonl(tmp_path / "results.jsonl", [result])
    write_usage_summary(tmp_path / "usage_summary.json", result.usage)

    assert read_jsonl(tmp_path / "results.jsonl")[0]["usage"][0]["total_tokens"] == 9
    summary = json.loads((tmp_path / "usage_summary.json").read_text(encoding="utf-8"))
    assert summary["by_source"]["reported"]["total_tokens"] == 9
