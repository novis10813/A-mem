import json
from pathlib import Path

from memorybench.cli import main
from memorybench.dashboard.data import ExperimentIndex


def test_list_components_and_validate(capsys, tmp_path: Path):
    assert main(["list-components"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed["construction"] == ["amem", "turn_rag"]
    config = tmp_path / "bad.yaml"
    config.write_text("experiment: {id: bad}\npipeline: {}\n", encoding="utf-8")
    assert main(["validate", "--config", str(config)]) == 1


def test_dashboard_index_reads_taxonomy_results_and_failures(tmp_path: Path):
    root = tmp_path / "experiments" / "demo"
    result_dir = root / "retrieve_qa/construction_000/run_000"
    result_dir.mkdir(parents=True)
    (root / "manifest.json").write_text(json.dumps({
        "experiment_id": "demo", "status": "partial", "fingerprint": "abc",
        "taxonomy": {"dimensions": [{"name": "question_type", "values": ["1", "2"]}]},
    }), encoding="utf-8")
    (result_dir / "results.jsonl").write_text(
        json.dumps({"question_id": "q1", "status": "failed", "labels": {"question_type": "2"},
                    "retrieval": {"stages": []}, "usage": [], "errors": [{"message": "boom"}]}) + "\n",
        encoding="utf-8",
    )
    index = ExperimentIndex(tmp_path / "experiments")
    overview = index.overview("demo")
    assert overview["taxonomy"][0]["name"] == "question_type"
    assert overview["failed_questions"] == 1
    assert index.qa_results("demo")[0]["question_id"] == "q1"
