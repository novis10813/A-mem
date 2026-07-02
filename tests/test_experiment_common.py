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
