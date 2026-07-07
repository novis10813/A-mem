"""Tests for experiment_data_loader.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from experiment_data_loader import (
    DATASET_PATH,
    RESULTS_ROOT,
    align_experiments,
    compute_across_run_variance,
    discover_experiments,
    get_qa_evidence,
    get_sample,
    list_construction_runs,
    list_qa_runs,
    load_experiment_results,
    question_key,
    resolve_evidence,
    results_to_question_map,
)


# ---------------------------------------------------------------------------
# question_key
# ---------------------------------------------------------------------------

class TestQuestionKey:
    def test_deterministic(self):
        k1 = question_key(0, "When did Caroline go to the LGBTQ support group?")
        k2 = question_key(0, "When did Caroline go to the LGBTQ support group?")
        assert k1 == k2

    def test_length(self):
        k = question_key(0, "Some question?")
        assert len(k) == 12

    def test_different_sample_different_key(self):
        q = "Same question?"
        assert question_key(0, q) != question_key(1, q)

    def test_different_question_different_key(self):
        assert question_key(0, "Question A?") != question_key(0, "Question B?")

    def test_hex_chars_only(self):
        k = question_key(3, "test?")
        assert all(c in "0123456789abcdef" for c in k)


# ---------------------------------------------------------------------------
# Experiment discovery (live artifacts)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not RESULTS_ROOT.exists(), reason="artifacts/results not present")
class TestDiscoverExperiments:
    def test_returns_list(self):
        exps = discover_experiments()
        assert isinstance(exps, list)

    def test_only_two_stage_format(self):
        """All discovered experiments must have at least one results.json."""
        exps = discover_experiments()
        for exp_id in exps:
            results = list((RESULTS_ROOT / exp_id).glob(
                "construction_run_*/robust/qa_run_*/results.json"
            ))
            assert len(results) > 0, f"{exp_id} has no robust qa_run results"

    def test_known_experiment_present(self):
        exps = discover_experiments()
        # At least one expected experiment
        expected = {"ollama_llama3.2-1b_nltk", "ollama_llama3.2-1b_none_rerank_k10"}
        assert expected & set(exps), f"None of {expected} found in {exps}"


# ---------------------------------------------------------------------------
# list_construction_runs / list_qa_runs
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not RESULTS_ROOT.exists(), reason="artifacts/results not present")
class TestListRuns:
    def test_construction_runs_sorted(self):
        exps = discover_experiments()
        if not exps:
            pytest.skip("No experiments found")
        runs = list_construction_runs(exps[0])
        assert runs == sorted(runs)

    def test_qa_runs_sorted(self):
        exps = discover_experiments()
        if not exps:
            pytest.skip("No experiments found")
        c_runs = list_construction_runs(exps[0])
        if not c_runs:
            pytest.skip("No construction runs")
        qa_runs = list_qa_runs(exps[0], c_runs[0])
        assert qa_runs == sorted(qa_runs)

    def test_unknown_experiment_returns_empty(self):
        assert list_construction_runs("does_not_exist_xyz") == []

    def test_unknown_construction_run_returns_empty(self):
        exps = discover_experiments()
        if not exps:
            pytest.skip("No experiments found")
        assert list_qa_runs(exps[0], 9999) == []


# ---------------------------------------------------------------------------
# load_experiment_results
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not RESULTS_ROOT.exists(), reason="artifacts/results not present")
class TestLoadExperimentResults:
    def _pick_first_run(self):
        exps = discover_experiments()
        if not exps:
            pytest.skip("No experiments found")
        exp_id = exps[0]
        c_runs = list_construction_runs(exp_id)
        if not c_runs:
            pytest.skip("No construction runs")
        qa_runs = list_qa_runs(exp_id, c_runs[0])
        if not qa_runs:
            pytest.skip("No QA runs")
        return exp_id, c_runs[0], qa_runs[0]

    def test_returns_dict_with_individual_results(self):
        exp_id, c_run, qa_run = self._pick_first_run()
        data = load_experiment_results(exp_id, c_run, qa_run)
        assert "individual_results" in data
        assert isinstance(data["individual_results"], list)
        assert len(data["individual_results"]) > 0

    def test_question_key_added(self):
        exp_id, c_run, qa_run = self._pick_first_run()
        data = load_experiment_results(exp_id, c_run, qa_run)
        for r in data["individual_results"][:5]:
            assert "question_key" in r
            assert len(r["question_key"]) == 12

    def test_question_key_matches_function(self):
        exp_id, c_run, qa_run = self._pick_first_run()
        data = load_experiment_results(exp_id, c_run, qa_run)
        r = data["individual_results"][0]
        expected = question_key(r["sample_id"], r["question"])
        assert r["question_key"] == expected

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_experiment_results("does_not_exist_xyz", 0, 0)


# ---------------------------------------------------------------------------
# align_experiments
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not RESULTS_ROOT.exists(), reason="artifacts/results not present")
class TestAlignExperiments:
    def _load_two_experiments(self):
        exps = discover_experiments()
        if len(exps) < 2:
            pytest.skip("Need at least 2 experiments")
        exp_a, exp_b = exps[0], exps[1]
        c_runs_a = list_construction_runs(exp_a)
        c_runs_b = list_construction_runs(exp_b)
        qa_runs_a = list_qa_runs(exp_a, c_runs_a[0]) if c_runs_a else []
        qa_runs_b = list_qa_runs(exp_b, c_runs_b[0]) if c_runs_b else []
        if not qa_runs_a or not qa_runs_b:
            pytest.skip("Missing QA runs")
        results_a = load_experiment_results(exp_a, c_runs_a[0], qa_runs_a[0])
        results_b = load_experiment_results(exp_b, c_runs_b[0], qa_runs_b[0])
        return results_a, results_b

    def test_aligned_rows_have_required_keys(self):
        results_a, results_b = self._load_two_experiments()
        rows = align_experiments(results_a, results_b)
        required = {"question_key", "sample_id", "question", "reference", "category", "exp_a", "exp_b"}
        for row in rows[:10]:
            assert required <= set(row.keys()), f"Missing keys in row: {row.keys()}"

    def test_question_key_stable(self):
        results_a, results_b = self._load_two_experiments()
        rows = align_experiments(results_a, results_b)
        for row in rows[:5]:
            expected = question_key(row["sample_id"], row["question"])
            assert row["question_key"] == expected

    def test_same_experiment_fully_aligned(self):
        exps = discover_experiments()
        if not exps:
            pytest.skip("No experiments found")
        exp_id = exps[0]
        c_runs = list_construction_runs(exp_id)
        qa_runs = list_qa_runs(exp_id, c_runs[0])
        if not qa_runs:
            pytest.skip("No QA runs")
        data = load_experiment_results(exp_id, c_runs[0], qa_runs[0])
        rows = align_experiments(data, data)
        for row in rows:
            assert row["exp_a"] is not None
            assert row["exp_b"] is not None


# ---------------------------------------------------------------------------
# compute_across_run_variance
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not RESULTS_ROOT.exists(), reason="artifacts/results not present")
class TestComputeAcrossRunVariance:
    def _pick_multi_run_experiment(self):
        for exp_id in discover_experiments():
            c_runs = list_construction_runs(exp_id)
            if not c_runs:
                continue
            qa_runs = list_qa_runs(exp_id, c_runs[0])
            if len(qa_runs) >= 2:
                return exp_id, c_runs[0]
        pytest.skip("No experiment with >= 2 QA runs found")

    def test_returns_dict(self):
        exp_id, c_run = self._pick_multi_run_experiment()
        variance = compute_across_run_variance(exp_id, c_run)
        assert isinstance(variance, dict)
        assert len(variance) > 0

    def test_entry_has_required_fields(self):
        exp_id, c_run = self._pick_multi_run_experiment()
        variance = compute_across_run_variance(exp_id, c_run)
        entry = next(iter(variance.values()))
        for field in ("question", "sample_id", "category", "values", "mean", "std", "runs_above_half", "total_runs"):
            assert field in entry, f"Missing field: {field}"

    def test_values_match_total_runs(self):
        exp_id, c_run = self._pick_multi_run_experiment()
        qa_runs = list_qa_runs(exp_id, c_run)
        variance = compute_across_run_variance(exp_id, c_run)
        for entry in list(variance.values())[:5]:
            assert entry["total_runs"] <= len(qa_runs)
            assert len(entry["values"]) == entry["total_runs"]


# ---------------------------------------------------------------------------
# Evidence resolver (live dataset)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not DATASET_PATH.exists(), reason="dataset not present")
class TestResolveEvidence:
    def test_get_sample_by_index(self):
        sample = get_sample(0)
        assert sample is not None
        assert "conversation" in sample
        assert "qa" in sample

    def test_get_qa_evidence(self):
        # Known question from sample 0
        evidence = get_qa_evidence(0, "When did Caroline go to the LGBTQ support group?")
        assert "D1:3" in evidence

    def test_resolve_known_evidence(self):
        turns = resolve_evidence(0, ["D1:3"])
        assert len(turns) == 1
        turn = turns[0]
        assert turn["ref"] == "D1:3"
        assert "support group" in turn["text"].lower()
        assert turn["speaker"] == "Caroline"
        assert "error" not in turn

    def test_resolve_multiple_evidence(self):
        turns = resolve_evidence(0, ["D1:3", "D2:8"])
        assert len(turns) == 2
        assert all("error" not in t for t in turns)

    def test_resolve_invalid_ref(self):
        turns = resolve_evidence(0, ["INVALID"])
        assert len(turns) == 1
        assert "error" in turns[0]

    def test_resolve_missing_sample(self):
        turns = resolve_evidence(9999, ["D1:3"])
        assert turns == []
