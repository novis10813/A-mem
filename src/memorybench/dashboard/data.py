from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from memorybench.artifacts import artifact_key, read_jsonl, read_memory_store


class ExperimentIndex:
    """Read-only projections over canonical MemoryBench artifacts."""

    def __init__(self, artifact_root: str | Path) -> None:
        self.artifact_root = Path(artifact_root)

    def experiments(self) -> list[str]:
        if not self.artifact_root.exists():
            return []
        return sorted(
            path.name for path in self.artifact_root.iterdir()
            if path.is_dir() and (path / "manifest.json").exists()
        )

    def manifest(self, experiment_id: str) -> dict[str, Any]:
        path = self.artifact_root / experiment_id / "manifest.json"
        if not path.exists():
            raise FileNotFoundError(f"Experiment manifest not found: {experiment_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def qa_results(self, experiment_id: str) -> list[dict[str, Any]]:
        root = self.artifact_root / experiment_id / "retrieve_qa"
        rows = []
        for path in sorted(root.glob("construction_*/run_*/results.jsonl")) if root.exists() else ():
            rows.extend(read_jsonl(path))
        return rows

    def overview(self, experiment_id: str) -> dict[str, Any]:
        manifest = self.manifest(experiment_id)
        rows = self.qa_results(experiment_id)
        completed = [row for row in rows if row.get("status") == "completed"]
        metric_names = sorted({name for row in completed for name in row.get("metrics", {})})
        breakdown: dict[str, Counter[str]] = {}
        for row in rows:
            for dimension, values in row.get("labels", {}).items():
                if isinstance(values, str):
                    values = [values]
                breakdown.setdefault(dimension, Counter()).update(values)
        return {
            "experiment_id": experiment_id,
            "status": manifest.get("status"),
            "fingerprint": manifest.get("fingerprint"),
            "taxonomy": manifest.get("taxonomy", {}).get("dimensions", []),
            "taxonomy_breakdown": {
                name: dict(sorted(counts.items())) for name, counts in breakdown.items()
            },
            "questions": len(rows),
            "failed_questions": sum(row.get("status") == "failed" for row in rows),
            "metrics": {
                name: sum(float(row.get("metrics", {}).get(name, 0.0)) for row in completed)
                / len(completed)
                for name in metric_names
            } if completed else {},
            "errors": [error for row in rows for error in row.get("errors", [])],
            "usage": self.usage_summary(experiment_id),
        }

    def qa_compare(self, experiment_a: str, experiment_b: str) -> list[dict[str, Any]]:
        rows_a = {self._result_key(row): row for row in self.qa_results(experiment_a)}
        rows_b = {self._result_key(row): row for row in self.qa_results(experiment_b)}
        output = []
        keys = sorted(
            set(rows_a) | set(rows_b),
            key=lambda key: (
                -1 if key[0] is None else key[0],
                -1 if key[1] is None else key[1],
                key[2],
            ),
        )
        for construction_run, qa_run, question_id in keys:
            key = (construction_run, qa_run, question_id)
            left, right = rows_a.get(key), rows_b.get(key)
            base = left or right
            output.append({
                "construction_run": construction_run,
                "qa_run": qa_run,
                "question_id": question_id,
                "question": base.get("question", ""),
                "reference": base.get("reference", ""),
                "prediction_a": None if left is None else left.get("prediction"),
                "prediction_b": None if right is None else right.get("prediction"),
                "metrics_a": {} if left is None else left.get("metrics", {}),
                "metrics_b": {} if right is None else right.get("metrics", {}),
                "status_a": None if left is None else left.get("status"),
                "status_b": None if right is None else right.get("status"),
            })
        return output

    def retrieval_trace(
        self,
        experiment_id: str,
        question_id: str,
        construction_run: int | None = None,
        qa_run: int | None = None,
    ) -> dict[str, Any]:
        for row in self.qa_results(experiment_id):
            provenance = row.get("provenance", {})
            if (
                row.get("question_id") == question_id
                and (construction_run is None or provenance.get("construction_run") == construction_run)
                and (qa_run is None or provenance.get("qa_run") == qa_run)
            ):
                retrieval = row.get("retrieval", {})
                return {
                    "question_id": question_id,
                    "construction_run": provenance.get("construction_run"),
                    "qa_run": provenance.get("qa_run"),
                    "status": row.get("status"),
                    "prediction": row.get("prediction", ""),
                    "stages": retrieval.get("stages", []),
                    "neighbor_expansion": retrieval.get("neighbor_expansion", []),
                    "tool_traces": row.get("tool_traces", []),
                    "items": retrieval.get("items", []),
                    "context": row.get("context", {}),
                }
        raise KeyError(f"Question not found in experiment {experiment_id}: {question_id}")

    @staticmethod
    def _result_key(row: dict[str, Any]) -> tuple[int | None, int | None, str]:
        provenance = row.get("provenance", {})
        return (
            provenance.get("construction_run"),
            provenance.get("qa_run"),
            row["question_id"],
        )

    def memory_graph(
        self,
        experiment_id: str,
        construction_run: int,
        sample_id: str,
    ) -> dict[str, Any]:
        directory = (
            self.artifact_root / experiment_id / "construction"
            / f"run_{construction_run:03d}" / "samples" / artifact_key(sample_id)
        )
        store = read_memory_store(directory)
        return {
            "sample_id": sample_id,
            "records": [item.model_dump(mode="json") for item in store.records],
            "nodes": [item.model_dump(mode="json") for item in store.nodes],
            "edges": [item.model_dump(mode="json") for item in store.edges],
            "layers": [item.model_dump(mode="json") for item in store.layers],
        }

    def usage_summary(self, experiment_id: str) -> dict[str, dict[str, float]]:
        summary: dict[str, dict[str, float]] = {}
        root = self.artifact_root / experiment_id
        usage_files = sorted(root.glob("construction/run_*/usage.jsonl"))
        usage_files.extend(sorted(root.glob("retrieve_qa/construction_*/run_*/usage.jsonl")))
        if usage_files:
            records = [record for path in usage_files for record in read_jsonl(path)]
        else:
            records = [
                usage for row in self.qa_results(experiment_id) for usage in row.get("usage", [])
            ]
        for usage in records:
            source = usage.get("source", "unknown")
            bucket = summary.setdefault(source, {
                "calls": 0,
                "total_tokens": 0,
                "latency_ms": 0.0,
            })
            bucket["calls"] += 1
            bucket["total_tokens"] += usage.get("total_tokens") or 0
            bucket["latency_ms"] += usage.get("latency_ms") or 0.0
        return summary


def create_dashboard(artifact_root: str | Path):
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("Install MemoryBench with the 'dashboard' extra") from exc

    index = ExperimentIndex(artifact_root)
    experiments = index.experiments()
    default = experiments[0] if experiments else None

    def bar_figure(values: dict[str, float], title: str):
        import plotly.graph_objects as go

        return go.Figure(
            data=[go.Bar(x=list(values), y=list(values.values()))],
            layout={"title": title, "xaxis_title": "Group", "yaxis_title": "Value"},
        )

    def overview_view(experiment_id: str):
        payload = index.overview(experiment_id)
        taxonomy_values = {
            f"{dimension}:{label}": count
            for dimension, counts in payload["taxonomy_breakdown"].items()
            for label, count in counts.items()
        }
        return (
            payload,
            bar_figure(payload["metrics"], "Mean metrics"),
            bar_figure(taxonomy_values, "Dataset-native taxonomy"),
        )

    def memory_view(experiment_id: str, construction_run: float, sample_id: str):
        import networkx as nx
        import plotly.graph_objects as go

        payload = index.memory_graph(experiment_id, int(construction_run), sample_id)
        graph = nx.DiGraph()
        for node in payload["nodes"]:
            graph.add_node(node["node_id"], label=node.get("type", "node"))
        for edge in payload["edges"]:
            graph.add_edge(edge["source_id"], edge["target_id"], type=edge.get("type"))
        positions = nx.spring_layout(graph, seed=42) if graph.nodes else {}
        edge_x, edge_y = [], []
        for source, target in graph.edges:
            x0, y0 = positions[source]
            x1, y1 = positions[target]
            edge_x.extend((x0, x1, None))
            edge_y.extend((y0, y1, None))
        figure = go.Figure()
        figure.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", hoverinfo="none"))
        figure.add_trace(go.Scatter(
            x=[positions[node][0] for node in graph.nodes],
            y=[positions[node][1] for node in graph.nodes],
            text=[str(node) for node in graph.nodes],
            mode="markers+text",
            textposition="top center",
        ))
        figure.update_layout(title=f"Memory graph: {sample_id}", showlegend=False)
        return payload, figure

    def usage_view(experiment_id: str):
        payload = index.usage_summary(experiment_id)
        tokens = {source: values["total_tokens"] for source, values in payload.items()}
        return payload, bar_figure(tokens, "Tokens by accounting source")

    with gr.Blocks(title="MemoryBench Research Workbench") as app:
        gr.Markdown("# MemoryBench Research Workbench")
        with gr.Tab("Overview"):
            overview_experiment = gr.Dropdown(experiments, value=default, label="Experiment")
            initial_overview = overview_view(default) if default else ({}, bar_figure({}, "Mean metrics"), bar_figure({}, "Dataset-native taxonomy"))
            overview_output = gr.JSON(value=initial_overview[0])
            overview_metrics = gr.Plot(value=initial_overview[1])
            overview_taxonomy = gr.Plot(value=initial_overview[2])
            overview_experiment.change(
                overview_view,
                overview_experiment,
                [overview_output, overview_metrics, overview_taxonomy],
            )
        with gr.Tab("QA Compare"):
            experiment_a = gr.Dropdown(experiments, value=default, label="Experiment A")
            experiment_b = gr.Dropdown(experiments, value=default, label="Experiment B")
            compare_button = gr.Button("Compare")
            comparison_output = gr.JSON()
            compare_button.click(index.qa_compare, [experiment_a, experiment_b], comparison_output)
        with gr.Tab("Retrieval Trace"):
            trace_experiment = gr.Dropdown(experiments, value=default, label="Experiment")
            trace_question = gr.Textbox(label="Question ID")
            trace_construction_run = gr.Number(value=None, precision=0, label="Construction run (optional)")
            trace_qa_run = gr.Number(value=None, precision=0, label="QA run (optional)")
            trace_button = gr.Button("Load trace")
            trace_output = gr.JSON()
            trace_button.click(
                index.retrieval_trace,
                [trace_experiment, trace_question, trace_construction_run, trace_qa_run],
                trace_output,
            )
        with gr.Tab("Memory Explorer"):
            memory_experiment = gr.Dropdown(experiments, value=default, label="Experiment")
            memory_run = gr.Number(value=0, precision=0, label="Construction run")
            memory_sample = gr.Textbox(label="Sample ID")
            memory_button = gr.Button("Load memory")
            memory_output = gr.JSON(label="Normalized memory store")
            memory_plot = gr.Plot(label="Graph")
            memory_button.click(
                memory_view,
                [memory_experiment, memory_run, memory_sample],
                [memory_output, memory_plot],
            )
        with gr.Tab("Usage & Latency"):
            usage_experiment = gr.Dropdown(experiments, value=default, label="Experiment")
            initial_usage = usage_view(default) if default else ({}, bar_figure({}, "Tokens by accounting source"))
            usage_output = gr.JSON(value=initial_usage[0])
            usage_plot = gr.Plot(value=initial_usage[1])
            usage_experiment.change(usage_view, usage_experiment, [usage_output, usage_plot])
    return app
