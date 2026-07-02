import csv
import json
from pathlib import Path

from scripts import experiment_common as ec


def write_result(path: Path, construction_run: int, qa_run: int, condition: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "construction_run": construction_run,
        "qa_run": qa_run,
        "condition": condition,
        "aggregate_metrics": {
            "overall": {
                "f1": {
                    "mean": 0.5 + qa_run,
                    "std": 0.0,
                    "median": 0.5 + qa_run,
                    "min": 0.5 + qa_run,
                    "max": 0.5 + qa_run,
                    "count": 1,
                }
            }
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_summary_aggregation_preserves_run_condition_and_metric_columns(tmp_path: Path):
    mode_dir = tmp_path / "construction_run_00" / "content_keywords"
    write_result(mode_dir / "qa_run_00" / "none.json", 0, 0, "none")
    write_result(mode_dir / "qa_run_01" / "none.json", 0, 1, "none")

    ec.write_mode_summary(mode_dir, "content_keywords", ("none",))

    with (mode_dir / "per_run_metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert {row["construction_run"] for row in rows} == {"0"}
    assert {row["qa_run"] for row in rows} == {"0", "1"}
    assert {row["condition"] for row in rows} == {"none"}
    assert {row["metric"] for row in rows} == {"f1"}

    summary = json.loads((mode_dir / "summary_across_runs.json").read_text(encoding="utf-8"))
    assert summary["summary"][0]["runs"] == 2
