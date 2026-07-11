from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from memorybench.artifacts import read_jsonl


class ExperimentIndex:
    """Read-only projection over canonical MemoryBench artifacts."""

    def __init__(self, artifact_root: str | Path) -> None:
        self.artifact_root = Path(artifact_root)

    def experiments(self) -> list[str]:
        if not self.artifact_root.exists():
            return []
        return sorted(path.name for path in self.artifact_root.iterdir() if (path / "manifest.json").exists())

    def manifest(self, experiment_id: str) -> dict[str, Any]:
        return json.loads((self.artifact_root / experiment_id / "manifest.json").read_text(encoding="utf-8"))

    def qa_results(self, experiment_id: str) -> list[dict[str, Any]]:
        root = self.artifact_root / experiment_id / "retrieve_qa"
        rows = []
        for path in sorted(root.glob("construction_*/run_*/results.jsonl")) if root.exists() else ():
            rows.extend(read_jsonl(path))
        return rows

    def overview(self, experiment_id: str) -> dict[str, Any]:
        manifest = self.manifest(experiment_id)
        rows = self.qa_results(experiment_id)
        return {
            "experiment_id": experiment_id, "status": manifest.get("status"),
            "fingerprint": manifest.get("fingerprint"),
            "taxonomy": manifest.get("taxonomy", {}).get("dimensions", []),
            "questions": len(rows), "failed_questions": sum(row.get("status") == "failed" for row in rows),
            "errors": [error for row in rows for error in row.get("errors", [])],
        }


def create_dashboard(artifact_root: str | Path):
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("Install MemoryBench with the 'dashboard' extra") from exc
    index = ExperimentIndex(artifact_root)
    with gr.Blocks(title="MemoryBench Research Workbench") as app:
        gr.Markdown("# MemoryBench Research Workbench")
        with gr.Tabs():
            for title in ("Overview", "QA Compare", "Retrieval Trace", "Memory Explorer", "Usage & Latency"):
                with gr.Tab(title):
                    gr.JSON(label=title, value={"experiments": index.experiments()})
    return app
