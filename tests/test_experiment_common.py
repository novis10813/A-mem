from pathlib import Path

import pytest

from scripts import experiment_common as ec


def test_validate_experiment_id_rejects_empty_and_traversal():
    for value in ("", " ", "../x", "x/y", r"x\y", ".."):
        with pytest.raises(ValueError):
            ec.validate_experiment_id(value)


def test_validate_experiment_id_accepts_safe_names():
    assert ec.validate_experiment_id("ollama_llama3.2:1b-nltk") == "ollama_llama3.2:1b-nltk"


def test_path_helpers_are_stable(tmp_path: Path):
    cache_dir = ec.construction_cache_dir(tmp_path / "artifacts" / "caches", "exp1", 0)
    results_dir = ec.qa_run_dir(
        tmp_path / "artifacts" / "results", "exp1", 0, "robust", 3
    )

    assert cache_dir == tmp_path / "artifacts" / "caches" / "exp1" / "construction_run_00"
    assert (
        results_dir
        == tmp_path
        / "artifacts"
        / "results"
        / "exp1"
        / "construction_run_00"
        / "robust"
        / "qa_run_03"
    )


def test_default_artifact_roots_are_grouped_under_artifacts():
    assert ec.DEFAULT_CACHE_ROOT == Path("artifacts/caches")
    assert ec.DEFAULT_RESULTS_ROOT == Path("artifacts/results")
    assert ec.DEFAULT_LOG_ROOT == Path("artifacts/logs")


def test_sha256_file_hashes_file_content(tmp_path: Path):
    path = tmp_path / "dataset.json"
    path.write_text("abc", encoding="utf-8")

    assert ec.sha256_file(path) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_build_manifest_payload_records_schema_and_provenance(tmp_path: Path):
    dataset = tmp_path / "dataset.json"
    dataset.write_text("[]", encoding="utf-8")

    payload = ec.build_manifest_payload(
        experiment_id="exp1",
        stage="qa_evaluation",
        dataset=dataset,
        created_at="2026-07-05T00:00:00",
        config_source="experiments/exp1.yaml",
        construction={"runs": 1},
        evaluation={"qa_runs": 2},
        runtime={"backend": {"name": "ollama"}},
        cache_experiment_id="cache1",
        source_cache_manifest={
            "experiment_id": "cache1",
            "artifact_schema_version": 2,
            "dataset": {"path": "data/locomo10.json"},
            "construction": {"runs": 1},
        },
    )

    assert payload["artifact_schema_version"] == 2
    assert payload["dataset"]["sha256"]
    assert payload["repo"].keys() == {"git_commit", "git_dirty"}
    assert payload["cache_experiment_id"] == "cache1"
    assert payload["source_cache_manifest"]["experiment_id"] == "cache1"


def test_construction_complete_requires_metadata_and_all_cache_files(tmp_path: Path):
    cache_dir = tmp_path / "construction_run_00"
    cache_dir.mkdir()
    sample_indices = [0]

    assert not ec.construction_complete(cache_dir, sample_indices)
    for path in ec.expected_cache_files(cache_dir, sample_indices):
        path.write_text("x", encoding="utf-8")

    assert ec.construction_complete(cache_dir, sample_indices)


def test_result_completion_checks(tmp_path: Path):
    qa_dir = tmp_path / "qa_run_00"
    qa_dir.mkdir()
    assert not ec.content_keywords_complete(qa_dir, ("none", "nltk"))
    (qa_dir / "none.json").write_text("{}", encoding="utf-8")
    assert not ec.content_keywords_complete(qa_dir, ("none", "nltk"))
    (qa_dir / "nltk.json").write_text("{}", encoding="utf-8")
    assert ec.content_keywords_complete(qa_dir, ("none", "nltk"))

    assert not ec.robust_complete(qa_dir)
    (qa_dir / "results.json").write_text("{}", encoding="utf-8")
    assert ec.robust_complete(qa_dir)
