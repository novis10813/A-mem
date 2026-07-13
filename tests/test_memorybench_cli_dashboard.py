import json
from pathlib import Path

from memorybench.cli import main
from memorybench.dashboard.data import ExperimentIndex
from memorybench.artifacts import write_memory_store
from memorybench.schemas import MemoryEdge, MemoryNode, MemoryRecord, MemoryStore


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
        "taxonomy": {"dimensions": [{"name": "question_type", "values": ["multi_hop", "temporal"]}]},
    }), encoding="utf-8")
    (result_dir / "results.jsonl").write_text(
        json.dumps({"question_id": "q1", "sample_id": "s1", "status": "failed", "labels": {"question_type": ["temporal"]},
                    "retrieval": {"stages": []}, "usage": [], "errors": [{"message": "boom"}]}) + "\n",
        encoding="utf-8",
    )
    index = ExperimentIndex(tmp_path / "experiments")
    overview = index.overview("demo")
    assert overview["taxonomy"][0]["name"] == "question_type"
    assert overview["failed_questions"] == 1
    assert index.qa_results("demo")[0]["question_id"] == "q1"


def test_dashboard_index_builds_comparison_trace_graph_and_usage_views(tmp_path: Path):
    experiments = tmp_path / "experiments"
    for experiment_id, prediction, score in (("a", "Taipei", 1.0), ("b", "Tokyo", 0.0)):
        root = experiments / experiment_id
        run = root / "retrieve_qa/construction_000/run_000"
        run.mkdir(parents=True)
        (root / "manifest.json").write_text(json.dumps({
            "experiment_id": experiment_id,
            "status": "completed",
            "fingerprint": experiment_id,
            "taxonomy": {"dimensions": [{"name": "question_type", "values": ["temporal"]}]},
        }), encoding="utf-8")
        (run / "results.jsonl").write_text(json.dumps({
            "question_id": "locomo:0:0",
            "sample_id": "locomo:0",
            "status": "completed",
            "question": "Where?",
            "reference": "Taipei",
            "prediction": prediction,
            "labels": {"question_type": ["temporal"]},
            "metrics": {"f1": score},
            "retrieval": {"stages": [{"adapter": "bm25", "output_ranking": ["m1"]}]},
            "usage": [{"source": "reported", "total_tokens": 10, "latency_ms": 5}],
            "errors": [],
        }) + "\n", encoding="utf-8")
        if experiment_id == "a":
            write_memory_store(
                root / "construction/run_000/samples/locomo__0",
                MemoryStore(
                    sample_id="locomo:0",
                    records=(MemoryRecord(record_id="m1", text="Taipei"),),
                    nodes=(MemoryNode(node_id="m1", type="note"),),
                    edges=(MemoryEdge(edge_id="e1", source_id="m1", target_id="m2", type="link"),),
                ),
            )

    index = ExperimentIndex(experiments)

    assert index.overview("a")["metrics"]["f1"] == 1.0
    assert index.overview("a")["taxonomy_breakdown"]["question_type"]["temporal"] == 1
    assert index.qa_compare("a", "b")[0]["prediction_a"] == "Taipei"
    assert index.retrieval_trace("a", "locomo:0:0")["stages"][0]["adapter"] == "bm25"
    assert index.memory_graph("a", 0, "locomo:0")["nodes"][0]["node_id"] == "m1"
    assert index.usage_summary("a")["reported"]["total_tokens"] == 10


def test_dashboard_preserves_multiple_qa_runs(tmp_path: Path):
    experiments = tmp_path / "experiments"
    for experiment_id in ("a", "b"):
        root = experiments / experiment_id
        (root / "manifest.json").parent.mkdir(parents=True)
        (root / "manifest.json").write_text(json.dumps({
            "experiment_id": experiment_id,
            "status": "completed",
            "fingerprint": experiment_id,
            "taxonomy": {"dimensions": []},
        }), encoding="utf-8")
        for qa_run in (0, 1):
            run = root / f"retrieve_qa/construction_000/run_{qa_run:03d}"
            run.mkdir(parents=True)
            (run / "results.jsonl").write_text(json.dumps({
                "question_id": "q1",
                "sample_id": "s1",
                "status": "completed",
                "question": "Where?",
                "reference": "Taipei",
                "prediction": f"{experiment_id}-{qa_run}",
                "provenance": {"construction_run": 0, "qa_run": qa_run},
                "retrieval": {"stages": [{"output_ranking": [f"m{qa_run}"]}]},
            }) + "\n", encoding="utf-8")

    index = ExperimentIndex(experiments)

    comparison = index.qa_compare("a", "b")
    assert len(comparison) == 2
    assert {(row["construction_run"], row["qa_run"]) for row in comparison} == {(0, 0), (0, 1)}
    assert index.retrieval_trace("a", "q1", qa_run=1)["items"] == []
    assert index.retrieval_trace("a", "q1", qa_run=1)["stages"][0]["output_ranking"] == ["m1"]
