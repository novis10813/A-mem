import json
from pathlib import Path

from amem.benchmark.context import build_memory_fields_context
from amem.benchmark.results import flatten_usage_rows, write_run_results
from amem.benchmark.schemas import MemoryRecord, MemoryStore, QAResult, UsageRecord
from amem.methods.amem.qa import robust_dict_to_qa_results


def test_build_memory_fields_context_includes_configured_fields_only():
    store = MemoryStore(
        sample_id=0,
        records=(
            MemoryRecord(
                memory_id="m0",
                sample_id=0,
                text="full",
                timestamp="t",
                content="content",
                keywords=("k1", "k2"),
                tags=("hidden",),
            ),
        ),
    )

    context = build_memory_fields_context(store, ["m0"], ["timestamp", "content", "keywords"])

    assert "content" in context["text"]
    assert "k1" in context["text"]
    assert "hidden" not in context["text"]


def test_write_run_results_writes_json_jsonl_and_usage_summary(tmp_path: Path):
    result = QAResult(
        experiment_id="exp",
        construction_run=0,
        qa_run=0,
        sample_id=0,
        qa_idx=1,
        question="q",
        reference="r",
        prediction="p",
        category=1,
        metrics={"f1": 1.0},
        retrieval={"items": []},
        context={"text": ""},
        prompt="prompt",
        usage=(UsageRecord(phase="qa", call_id="answer", total_tokens=12),),
    )

    write_run_results(tmp_path, [result])

    assert (tmp_path / "results.jsonl").exists()
    assert json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))[
        "individual_results"
    ][0]["qa_idx"] == 1
    assert json.loads((tmp_path / "usage_summary.json").read_text(encoding="utf-8"))[
        "by_source"
    ]["reported"]["total_tokens"] == 12
    assert flatten_usage_rows([result])[0]["total_tokens"] == 12


def test_robust_dict_to_qa_results_normalizes_existing_payload():
    payload = {
        "construction_run": 0,
        "qa_run": 1,
        "individual_results": [
            {
                "sample_id": 2,
                "qa_idx": 3,
                "question": "q",
                "reference": "r",
                "prediction": "p",
                "category": 4,
                "metrics": {"f1": 0.5},
                "retrieval_info": {"indices": [0]},
                "raw_context": "ctx",
                "user_prompt": "prompt",
            }
        ],
    }

    results = robust_dict_to_qa_results(payload, experiment_id="exp")

    assert results[0].experiment_id == "exp"
    assert results[0].retrieval["info"]["indices"] == [0]
    assert results[0].context["text"] == "ctx"
