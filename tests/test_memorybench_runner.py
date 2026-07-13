import json
from pathlib import Path

import pytest

from memorybench.config import MemoryBenchConfig
from memorybench.runner import ExperimentRunner


def config(tmp_path: Path, *, on_error="stop", sample_count=1) -> MemoryBenchConfig:
    dataset = tmp_path / "dataset.json"
    sample = {
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
    }
    dataset.write_text(json.dumps([sample for _ in range(sample_count)]), encoding="utf-8")
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
    sample_dir = root / "construction" / "run_000" / "samples" / "locomo__0"
    assert json.loads((sample_dir / "store.json").read_text())["sample_id"] == "locomo:0"
    assert rows(sample_dir / "records.jsonl")[0]["record_id"] == "locomo:0:turn:D1:1"
    assert json.loads((sample_dir / "status.json").read_text())["status"] == "completed"
    results = rows(root / "retrieve_qa" / "construction_000" / "run_000" / "results.jsonl")
    assert [r["question_id"] for r in results] == ["locomo:0:0", "locomo:0:1"]
    assert results[0]["retrieval"]["stages"][0]["adapter"] == "bm25"
    question_path = root / "retrieve_qa/construction_000/run_000/questions/locomo__0__0.json"
    assert json.loads(question_path.read_text())["status"] == "completed"
    before = question_path.stat().st_mtime_ns
    assert ExperimentRunner(cfg).run().exit_code == 0
    assert question_path.stat().st_mtime_ns == before


def test_on_error_continue_writes_failed_row_and_partial_status(tmp_path: Path):
    cfg = config(tmp_path, on_error="continue")
    payload = cfg.model_dump(mode="json")
    payload["pipeline"]["retrieve_qa"]["qa"]["adapter"] = "failing"
    cfg = MemoryBenchConfig.model_validate(payload)
    outcome = ExperimentRunner(cfg).run()
    result_rows = rows(tmp_path / "artifacts/e2e/retrieve_qa/construction_000/run_000/results.jsonl")
    assert outcome.exit_code == 2
    assert all(row["status"] == "failed" and row["errors"] for row in result_rows)
    run_dir = tmp_path / "artifacts/e2e/retrieve_qa/construction_000/run_000"
    assert len(rows(run_dir / "errors.jsonl")) == 2
    assert (run_dir / "usage.jsonl").exists()


def test_on_error_stop_persists_failed_rows_and_fatal_manifest(tmp_path: Path):
    cfg = config(tmp_path, on_error="stop")
    payload = cfg.model_dump(mode="json")
    payload["pipeline"]["retrieve_qa"]["qa"]["adapter"] = "failing"
    cfg = MemoryBenchConfig.model_validate(payload)

    outcome = ExperimentRunner(cfg).run()

    root = tmp_path / "artifacts/e2e"
    run_dir = root / "retrieve_qa/construction_000/run_000"
    result_rows = rows(run_dir / "results.jsonl")
    assert outcome.exit_code == 1
    assert outcome.status == "failed"
    assert json.loads((root / "manifest.json").read_text())["status"] == "failed"
    assert all(row["status"] == "failed" and row["errors"] for row in result_rows)
    assert json.loads((run_dir / "status.json").read_text())["status"] == "partial"


def test_qa_selection_missing_construction_sample_is_partial_with_failed_rows(tmp_path: Path):
    cfg = config(tmp_path, on_error="continue", sample_count=2)
    payload = cfg.model_dump(mode="json")
    payload["pipeline"]["construction"]["selection"] = {"sample_limit": 1}
    payload["pipeline"]["retrieve_qa"]["selection"] = {"sample_limit": 2}
    cfg = MemoryBenchConfig.model_validate(payload)

    outcome = ExperimentRunner(cfg).run()

    run_dir = tmp_path / "artifacts/e2e/retrieve_qa/construction_000/run_000"
    result_rows = rows(run_dir / "results.jsonl")
    missing = [row for row in result_rows if row["sample_id"] == "locomo:1"]
    assert outcome.exit_code == 2
    assert len(result_rows) == 4
    assert len(missing) == 2
    assert all(row["status"] == "failed" for row in missing)
    assert all("No construction store" in row["errors"][0]["message"] for row in missing)
    assert (run_dir / "questions/locomo__1__0.json").exists()
    assert (run_dir / "questions/locomo__1__1.json").exists()


def test_sample_evaluation_failure_writes_failed_question_rows(tmp_path: Path, monkeypatch):
    cfg = config(tmp_path, on_error="continue")

    def fail_sample(*args, **kwargs):
        raise RuntimeError("sample evaluation failed")

    monkeypatch.setattr(ExperimentRunner, "_evaluate_question", fail_sample)
    outcome = ExperimentRunner(cfg).run()

    run_dir = tmp_path / "artifacts/e2e/retrieve_qa/construction_000/run_000"
    result_rows = rows(run_dir / "results.jsonl")
    assert outcome.exit_code == 2
    assert len(result_rows) == 2
    assert all(row["status"] == "failed" for row in result_rows)
    assert all(row["errors"][0]["message"] == "sample evaluation failed" for row in result_rows)
    assert json.loads((run_dir / "status.json").read_text())["failed"] == 2
    assert (run_dir / "questions/locomo__0__0.json").exists()
    assert (run_dir / "questions/locomo__0__1.json").exists()


def test_resume_requires_matching_config_fingerprint(tmp_path: Path):
    cfg = config(tmp_path)
    runner = ExperimentRunner(cfg)
    status_path = tmp_path / "status.json"
    status_path.write_text(json.dumps({
        "status": "completed",
        "fingerprint": "different-config",
        "execution_fingerprint": runner.execution_fingerprint,
    }), encoding="utf-8")

    assert not runner._completed_unit(status_path)


def test_retrieve_only_rejects_missing_experiment_memory_source(tmp_path: Path):
    cfg = config(tmp_path)
    payload = cfg.model_dump(mode="json")
    payload["experiment"]["id"] = "evaluate-only"
    payload["pipeline"]["stages"] = ["retrieve_qa"]
    payload["pipeline"]["construction"] = None
    payload["pipeline"]["retrieve_qa"]["memory_source"] = {
        "experiment_id": "missing-construction",
        "construction_runs": "all",
    }
    cfg = MemoryBenchConfig.model_validate(payload)

    with pytest.raises(FileNotFoundError, match="missing-construction"):
        ExperimentRunner(cfg).run()


def test_retrieve_only_loads_new_format_experiment_and_validates_dataset_hash(tmp_path: Path):
    source = config(tmp_path)
    source_payload = source.model_dump(mode="json")
    source_payload["experiment"]["id"] = "source"
    source_payload["pipeline"]["stages"] = ["construction"]
    source_payload["pipeline"]["retrieve_qa"] = None
    source = MemoryBenchConfig.model_validate(source_payload)
    assert ExperimentRunner(source).run().exit_code == 0

    evaluation_payload = config(tmp_path).model_dump(mode="json")
    evaluation_payload["experiment"]["id"] = "evaluation"
    evaluation_payload["pipeline"]["stages"] = ["retrieve_qa"]
    evaluation_payload["pipeline"]["construction"] = None
    evaluation_payload["pipeline"]["retrieve_qa"]["memory_source"] = {
        "experiment_id": "source",
        "construction_runs": [0],
    }
    evaluation = MemoryBenchConfig.model_validate(evaluation_payload)

    assert ExperimentRunner(evaluation).run().exit_code == 0
    result_path = tmp_path / "artifacts/evaluation/retrieve_qa/construction_000/run_000/results.jsonl"
    assert len(rows(result_path)) == 2

    evaluation.pipeline.dataset.path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        ExperimentRunner(evaluation).run()


def test_retrieve_only_rejects_incomplete_source_construction_run(tmp_path: Path):
    source = config(tmp_path)
    source_payload = source.model_dump(mode="json")
    source_payload["experiment"]["id"] = "source"
    source_payload["pipeline"]["stages"] = ["construction"]
    source_payload["pipeline"]["retrieve_qa"] = None
    source = MemoryBenchConfig.model_validate(source_payload)
    assert ExperimentRunner(source).run().exit_code == 0

    status_path = tmp_path / "artifacts/source/construction/run_000/status.json"
    status = json.loads(status_path.read_text())
    status["status"] = "partial"
    status_path.write_text(json.dumps(status), encoding="utf-8")

    evaluation_payload = config(tmp_path).model_dump(mode="json")
    evaluation_payload["experiment"]["id"] = "evaluation"
    evaluation_payload["pipeline"]["stages"] = ["retrieve_qa"]
    evaluation_payload["pipeline"]["construction"] = None
    evaluation_payload["pipeline"]["retrieve_qa"]["memory_source"] = {
        "experiment_id": "source",
        "construction_runs": [0],
    }
    evaluation = MemoryBenchConfig.model_validate(evaluation_payload)

    with pytest.raises(ValueError, match="construction run is not completed"):
        ExperimentRunner(evaluation).run()


def test_partial_resume_preserves_completed_questions_and_retries_failed(tmp_path: Path):
    cfg = config(tmp_path, on_error="continue")
    payload = cfg.model_dump(mode="json")
    payload["pipeline"]["retrieve_qa"]["qa"] = {
        "adapter": "failing",
        "params": {"question_ids": ["locomo:0:1"]},
    }
    cfg = MemoryBenchConfig.model_validate(payload)
    assert ExperimentRunner(cfg).run().exit_code == 2

    result_path = tmp_path / "artifacts/e2e/retrieve_qa/construction_000/run_000/results.jsonl"
    first_rows = rows(result_path)
    completed = next(row for row in first_rows if row["status"] == "completed")
    completed["prediction"] = "preserved completed result"
    question_path = (
        tmp_path / "artifacts/e2e/retrieve_qa/construction_000/run_000/questions"
        / f"{completed['question_id'].replace(':', '__')}.json"
    )
    question_path.write_text(json.dumps(completed), encoding="utf-8")

    assert ExperimentRunner(cfg).run().exit_code == 2
    second_rows = rows(result_path)
    assert next(
        row for row in second_rows if row["question_id"] == completed["question_id"]
    )["prediction"] == "preserved completed result"
