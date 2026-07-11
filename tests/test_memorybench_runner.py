import json
from pathlib import Path

from memorybench.config import MemoryBenchConfig
from memorybench.runner import ExperimentRunner


def config(tmp_path: Path, *, on_error="stop") -> MemoryBenchConfig:
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps([{
        "qa": [
            {"question": "What city?", "answer": "Taipei", "category": 1, "evidence": ["D1:2"]},
            {"question": "Who arrived?", "answer": "Alice", "category": 1, "evidence": ["D1:1"]},
        ],
        "conversation": {
            "speaker_a": "Alice", "speaker_b": "Bob", "session_1_date_time": "1 Jan 2026",
            "session_1": [
                {"speaker": "Alice", "dia_id": "D1:1", "text": "Alice arrived."},
                {"speaker": "Bob", "dia_id": "D1:2", "text": "We are in Taipei."},
            ],
        }, "event_summary": {}, "observation": {}, "session_summary": {},
    }]), encoding="utf-8")
    return MemoryBenchConfig.model_validate({
        "experiment": {"id": "e2e"},
        "pipeline": {
            "stages": ["construction", "retrieve_qa"],
            "dataset": {"adapter": "locomo", "path": str(dataset)},
            "construction": {"adapter": "turn_rag", "runs": 1, "chunker": {"adapter": "turn"}},
            "retrieve_qa": {
                "runs": 1,
                "retrieval": {"adapter": "staged", "stages": [{"adapter": "bm25", "top_k": 2}]},
                "context": {"adapter": "records", "fields": ["timestamp", "speaker", "content"]},
                "qa": {"adapter": "extractive"},
                "metrics": [{"adapter": "exact_match"}],
            },
        },
        "runtime": {"artifact_root": str(tmp_path / "artifacts"), "resume": True, "on_error": on_error},
    })


def rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_runner_writes_canonical_artifacts_and_resumes_questions(tmp_path: Path):
    cfg = config(tmp_path)
    outcome = ExperimentRunner(cfg).run()
    root = tmp_path / "artifacts" / "e2e"
    assert outcome.exit_code == 0
    assert json.loads((root / "manifest.json").read_text())["fingerprint"] == cfg.fingerprint
    store_rows = rows(root / "construction" / "run_000" / "stores.jsonl")
    assert [r["records"][0]["record_id"] for r in store_rows] == ["locomo:0:turn:D1:1"]
    results = rows(root / "retrieve_qa" / "construction_000" / "run_000" / "results.jsonl")
    assert [r["question_id"] for r in results] == ["locomo:0:0", "locomo:0:1"]
    assert results[0]["retrieval"]["stages"][0]["adapter"] == "bm25"
    before = (root / "retrieve_qa" / "construction_000" / "run_000" / "results.jsonl").stat().st_mtime_ns
    assert ExperimentRunner(cfg).run().exit_code == 0
    assert (root / "retrieve_qa" / "construction_000" / "run_000" / "results.jsonl").stat().st_mtime_ns == before


def test_on_error_continue_writes_failed_row_and_partial_status(tmp_path: Path):
    cfg = config(tmp_path, on_error="continue")
    payload = cfg.model_dump(mode="json")
    payload["pipeline"]["retrieve_qa"]["qa"]["adapter"] = "failing"
    cfg = MemoryBenchConfig.model_validate(payload)
    outcome = ExperimentRunner(cfg).run()
    result_rows = rows(tmp_path / "artifacts/e2e/retrieve_qa/construction_000/run_000/results.jsonl")
    assert outcome.exit_code == 2
    assert all(row["status"] == "failed" and row["errors"] for row in result_rows)
