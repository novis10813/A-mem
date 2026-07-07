#!/usr/bin/env python3
"""A-MEM Experiment Comparison Dashboard (Gradio).

Launch:
    uv run python scripts/experiment_dashboard.py
    # Opens at http://localhost:7860
"""

from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from experiment_data_loader import (
    CATEGORY_LABELS,
    DATASET_PATH,
    RESULTS_ROOT,
    align_experiments,
    compute_across_run_variance,
    discover_experiments,
    get_qa_evidence,
    list_construction_runs,
    list_qa_runs,
    load_experiment_results,
    resolve_evidence,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

METRIC_CHOICES = ["f1", "bleu1", "rouge1_f", "rougeL_f", "bert_f1", "sbert_similarity", "exact_match"]
THRESHOLD_DEFAULT = 0.5

def _metric_val(result: dict | None, metric: str) -> float:
    if result is None:
        return -1.0
    return float(result.get("metrics", {}).get(metric, 0.0) or 0.0)

def _badge(val: float, threshold: float = 0.5) -> str:
    if val < 0:
        return "—"
    mark = "✓" if val >= threshold else "✗"
    return f"{mark} {val:.3f}"

def _fmt_retrieval_info(ri: dict | None) -> str:
    if not ri:
        return "*(no retrieval info)*"
    schema = ri.get("schema_version", "legacy")
    if schema == 3:
        lines = [f"**Schema v3** | final_k={ri.get('final_k')}"]
        for stage in ri.get("stages", []):
            lines.append(
                f"- **{stage.get('name','?')}** ({stage.get('type','?')}): "
                f"in={stage.get('input_count','?')} → out={stage.get('output_count','?')}"
            )
        selected = ri.get("selected", [])
        if selected:
            lines.append(f"\n**Selected {len(selected)} memories:**")
            for c in selected[:5]:
                score = c.get("score")
                score_str = f" (score={score:.3f})" if score is not None else ""
                lines.append(f"- idx {c.get('memory_index','?')}{score_str}")
        return "\n".join(lines)
    # legacy schema
    lines = []
    if ri.get("rerank_mode"):
        lines.append(f"**Rerank:** {ri['rerank_mode']}")
    if ri.get("candidate_k"):
        lines.append(f"**Candidates:** {ri['candidate_k']}")
    final = ri.get("final_indices", [])
    if final:
        lines.append(f"**Selected indices:** {final[:10]}")
    scores = ri.get("rerank_scores", [])
    if scores:
        lines.append(f"**Scores:** {[round(s,3) for s in scores[:5]]}")
    return "\n".join(lines) if lines else "*(legacy format — limited info)*"


def _fmt_context(raw: str | None) -> str:
    if not raw:
        return "*(no context)*"
    # Insert newline between memory blocks for readability
    import re
    cleaned = re.sub(r"(memory tags:[^\n]+)\n?", r"\1\n\n---\n", raw)
    return cleaned.strip()


# ---------------------------------------------------------------------------
# Tab 1 data builders
# ---------------------------------------------------------------------------

def build_comparison_df(
    exp_a: str, c_run_a: int, qa_run_a: int,
    exp_b: str, c_run_b: int, qa_run_b: int,
    category_filter: str,
    sample_filter: str,
    status_filter: str,
    metric: str,
    threshold: float,
) -> tuple[pd.DataFrame, list[dict]]:
    """Build the question comparison dataframe and underlying row data."""
    data_a = load_experiment_results(exp_a, c_run_a, qa_run_a)
    data_b = load_experiment_results(exp_b, c_run_b, qa_run_b)
    rows = align_experiments(data_a, data_b)

    # Apply filters
    if category_filter and category_filter != "All":
        cat_num = int(category_filter.replace("Cat", "").strip())
        rows = [r for r in rows if r["category"] == cat_num]

    if sample_filter and sample_filter != "All":
        sid = int(sample_filter.split()[-1])
        rows = [r for r in rows if r["sample_id"] == sid]

    va = [_metric_val(r["exp_a"], metric) for r in rows]
    vb = [_metric_val(r["exp_b"], metric) for r in rows]

    if status_filter == "A > B":
        rows = [r for r, a, b in zip(rows, va, vb) if a > b + 0.01]
        va = [_metric_val(r["exp_a"], metric) for r in rows]
        vb = [_metric_val(r["exp_b"], metric) for r in rows]
    elif status_filter == "B > A":
        rows = [r for r, a, b in zip(rows, va, vb) if b > a + 0.01]
        va = [_metric_val(r["exp_a"], metric) for r in rows]
        vb = [_metric_val(r["exp_b"], metric) for r in rows]
    elif status_filter == "Both ✓":
        rows = [r for r, a, b in zip(rows, va, vb) if a >= threshold and b >= threshold]
        va = [_metric_val(r["exp_a"], metric) for r in rows]
        vb = [_metric_val(r["exp_b"], metric) for r in rows]
    elif status_filter == "Both ✗":
        rows = [r for r, a, b in zip(rows, va, vb) if a < threshold and b < threshold]
        va = [_metric_val(r["exp_a"], metric) for r in rows]
        vb = [_metric_val(r["exp_b"], metric) for r in rows]

    records = []
    for i, (r, a, b) in enumerate(zip(rows, va, vb)):
        records.append({
            "#": i + 1,
            "Sample": r["sample_id"],
            "Cat": r["category"],
            "Question": r["question"][:90] + ("…" if len(r["question"]) > 90 else ""),
            f"Exp A ({metric})": _badge(a, threshold),
            f"Exp B ({metric})": _badge(b, threshold),
        })

    df = pd.DataFrame(records) if records else pd.DataFrame(
        columns=["#", "Sample", "Cat", "Question", f"Exp A ({metric})", f"Exp B ({metric})"]
    )
    return df, rows


def build_summary_stats(rows: list[dict], metric: str, threshold: float) -> dict[str, int]:
    total = len(rows)
    both_ok = sum(1 for r in rows
                  if _metric_val(r["exp_a"], metric) >= threshold
                  and _metric_val(r["exp_b"], metric) >= threshold)
    a_better = sum(1 for r in rows
                   if _metric_val(r["exp_a"], metric) > _metric_val(r["exp_b"], metric) + 0.01)
    b_better = sum(1 for r in rows
                   if _metric_val(r["exp_b"], metric) > _metric_val(r["exp_a"], metric) + 0.01)
    both_wrong = sum(1 for r in rows
                     if _metric_val(r["exp_a"], metric) < threshold
                     and _metric_val(r["exp_b"], metric) < threshold)
    return {"total": total, "both_ok": both_ok, "a_better": a_better, "b_better": b_better, "both_wrong": both_wrong}


# ---------------------------------------------------------------------------
# Tab 2 data builders
# ---------------------------------------------------------------------------

def build_variance_df(exp_id: str, c_run: int, metric: str) -> tuple[pd.DataFrame, dict]:
    variance = compute_across_run_variance(exp_id, c_run, metric)
    if not variance:
        return pd.DataFrame(), {}

    records = []
    for entry in variance.values():
        records.append({
            "Sample": entry["sample_id"],
            "Cat": entry["category"],
            "Question": entry["question"][:80] + ("…" if len(entry["question"]) > 80 else ""),
            "Mean": entry["mean"],
            "Std": entry["std"],
            f"N>{THRESHOLD_DEFAULT}": f"{entry['runs_above_half']}/{entry['total_runs']}",
            "Min": entry["min"],
            "Max": entry["max"],
        })

    df = pd.DataFrame(records).sort_values("Std", ascending=False).reset_index(drop=True)
    df.index += 1
    df.insert(0, "#", df.index)
    return df, variance


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def make_app() -> gr.Blocks:
    experiments = discover_experiments(RESULTS_ROOT)
    if not experiments:
        experiments = ["(no experiments found)"]

    dataset_exists = DATASET_PATH.exists()

    with gr.Blocks(title="A-MEM Experiment Dashboard", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            "# 🧠 A-MEM Experiment Comparison Dashboard\n"
            "Compare per-question performance across different experiments and QA runs."
        )

        # ---- shared state ----
        _aligned_rows = gr.State([])

        with gr.Tabs():

            # ================================================================
            # TAB 1: Experiment Comparison
            # ================================================================
            with gr.Tab("📊 Experiment Comparison"):

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Experiment A")
                        exp_a_dd = gr.Dropdown(experiments, label="Experiment A", value=experiments[0])
                        c_run_a_dd = gr.Dropdown([], label="Construction Run", value=None)
                        qa_run_a_dd = gr.Dropdown([], label="QA Run", value=None)

                    with gr.Column(scale=1):
                        gr.Markdown("### Experiment B")
                        exp_b_dd = gr.Dropdown(experiments, label="Experiment B",
                                               value=experiments[-1] if len(experiments) > 1 else experiments[0])
                        c_run_b_dd = gr.Dropdown([], label="Construction Run", value=None)
                        qa_run_b_dd = gr.Dropdown([], label="QA Run", value=None)

                with gr.Row():
                    cat_filter = gr.Dropdown(
                        ["All"] + [f"Cat {i}" for i in range(1, 6)],
                        value="All", label="Category"
                    )
                    sample_filter = gr.Dropdown(
                        ["All"] + [f"Sample {i}" for i in range(10)],
                        value="All", label="Sample"
                    )
                    status_filter = gr.Dropdown(
                        ["All", "A > B", "B > A", "Both ✓", "Both ✗"],
                        value="All", label="Status"
                    )
                    metric_dd = gr.Dropdown(METRIC_CHOICES, value="f1", label="Metric")
                    threshold_sl = gr.Slider(0.0, 1.0, value=THRESHOLD_DEFAULT, step=0.05, label="Pass Threshold")

                compare_btn = gr.Button("🔄 Compare", variant="primary")

                # Summary row
                with gr.Row():
                    stat_total = gr.Number(label="Total Questions", interactive=False)
                    stat_both_ok = gr.Number(label="Both ✓", interactive=False)
                    stat_a_better = gr.Number(label="A > B", interactive=False)
                    stat_b_better = gr.Number(label="B > A", interactive=False)
                    stat_both_wrong = gr.Number(label="Both ✗", interactive=False)

                # Question table
                question_table = gr.Dataframe(
                    label="Questions (click a row to see details)",
                    interactive=False,
                    wrap=False,
                    show_search="filter",
                )

                # Detail panel
                gr.Markdown("---\n### 🔍 Question Detail")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Experiment A")
                        detail_pred_a = gr.Textbox(label="Prediction", lines=2, interactive=False)
                        detail_ref = gr.Textbox(label="Reference (ground truth)", lines=1, interactive=False)
                        detail_metrics_a = gr.Dataframe(label="Metrics", interactive=False, wrap=False)
                        detail_keywords_a = gr.Textbox(label="Query Keywords", interactive=False)
                        detail_context_a = gr.Textbox(label="Retrieved Context", lines=10, interactive=False)
                        detail_retrieval_a = gr.Markdown(label="Retrieval Info")

                    with gr.Column():
                        gr.Markdown("#### Experiment B")
                        detail_pred_b = gr.Textbox(label="Prediction", lines=2, interactive=False)
                        detail_ref_b = gr.Textbox(label="Reference", lines=1, interactive=False)
                        detail_metrics_b = gr.Dataframe(label="Metrics", interactive=False, wrap=False)
                        detail_keywords_b = gr.Textbox(label="Query Keywords", interactive=False)
                        detail_context_b = gr.Textbox(label="Retrieved Context", lines=10, interactive=False)
                        detail_retrieval_b = gr.Markdown(label="Retrieval Info")

                # Evidence accordion
                with gr.Accordion("📖 原始對話 Evidence（點擊展開）", open=False,
                                   visible=dataset_exists) as evidence_accordion:
                    evidence_md = gr.Markdown("*Select a question to see evidence.*")

                # ---- wiring: populate run dropdowns ----
                def update_c_runs_a(exp_id):
                    runs = list_construction_runs(exp_id)
                    choices = [str(r) for r in runs]
                    return gr.Dropdown(choices=choices, value=choices[0] if choices else None)

                def update_c_runs_b(exp_id):
                    return update_c_runs_a(exp_id)

                def update_qa_runs_a(exp_id, c_run):
                    if c_run is None:
                        return gr.Dropdown(choices=[], value=None)
                    runs = list_qa_runs(exp_id, int(c_run))
                    choices = [str(r) for r in runs]
                    return gr.Dropdown(choices=choices, value=choices[0] if choices else None)

                def update_qa_runs_b(exp_id, c_run):
                    return update_qa_runs_a(exp_id, c_run)

                exp_a_dd.change(update_c_runs_a, inputs=exp_a_dd, outputs=c_run_a_dd)
                exp_b_dd.change(update_c_runs_b, inputs=exp_b_dd, outputs=c_run_b_dd)
                c_run_a_dd.change(update_qa_runs_a, inputs=[exp_a_dd, c_run_a_dd], outputs=qa_run_a_dd)
                c_run_b_dd.change(update_qa_runs_b, inputs=[exp_b_dd, c_run_b_dd], outputs=qa_run_b_dd)

                # ---- compare button ----
                def do_compare(
                    exp_a, c_run_a, qa_run_a,
                    exp_b, c_run_b, qa_run_b,
                    cat_f, sample_f, status_f, metric, threshold,
                ):
                    if not all([exp_a, c_run_a, qa_run_a, exp_b, c_run_b, qa_run_b]):
                        return (pd.DataFrame(), [], 0, 0, 0, 0, 0)
                    df, rows = build_comparison_df(
                        exp_a, int(c_run_a), int(qa_run_a),
                        exp_b, int(c_run_b), int(qa_run_b),
                        cat_f, sample_f, status_f, metric, threshold,
                    )
                    stats = build_summary_stats(rows, metric, threshold)
                    return (
                        df, rows,
                        stats["total"], stats["both_ok"],
                        stats["a_better"], stats["b_better"], stats["both_wrong"],
                    )

                compare_btn.click(
                    do_compare,
                    inputs=[
                        exp_a_dd, c_run_a_dd, qa_run_a_dd,
                        exp_b_dd, c_run_b_dd, qa_run_b_dd,
                        cat_filter, sample_filter, status_filter, metric_dd, threshold_sl,
                    ],
                    outputs=[
                        question_table, _aligned_rows,
                        stat_total, stat_both_ok, stat_a_better, stat_b_better, stat_both_wrong,
                    ],
                )

                def show_detail_full(evt: gr.SelectData, rows: list[dict], metric: str):
                    """Return all detail outputs in correct order."""
                    empty_df = pd.DataFrame()
                    if not rows or evt is None:
                        return ("", "", empty_df, "", "", "", "", "", empty_df, "", "", "", "")

                    row_idx = evt.index[0]
                    if row_idx >= len(rows):
                        return ("", "", empty_df, "", "", "", "", "", empty_df, "", "", "", "")

                    row = rows[row_idx]
                    ref = str(row["reference"])
                    ra = row.get("exp_a")
                    rb = row.get("exp_b")

                    def fmt_metrics(r):
                        if r is None:
                            return pd.DataFrame()
                        m = r.get("metrics", {})
                        return pd.DataFrame([{"Metric": k, "Value": round(float(v), 4)}
                                             for k, v in m.items()])

                    # Evidence
                    ev_md = ""
                    if dataset_exists and row.get("question"):
                        ev_refs = get_qa_evidence(row["sample_id"], row["question"])
                        turns = resolve_evidence(row["sample_id"], ev_refs)
                        ev_lines = [f"**Evidence refs:** {ev_refs}\n"]
                        for t in turns:
                            if "error" in t:
                                ev_lines.append(f"- `{t['ref']}` — ⚠️ {t['error']}")
                            else:
                                ev_lines.append(
                                    f"- **`{t['ref']}`** | Session {t['session_num']} | {t['date_time']}\n"
                                    f"  **{t['speaker']}**: {t['text']}"
                                )
                        ev_md = "\n".join(ev_lines)

                    kw_a = ra.get("query_keywords", "") if ra else ""
                    kw_b = rb.get("query_keywords", "") if rb else ""

                    return (
                        ra["prediction"] if ra else "—",
                        ref,
                        fmt_metrics(ra),
                        kw_a,
                        _fmt_context(ra.get("raw_context")) if ra else "",
                        _fmt_retrieval_info(ra.get("retrieval_info") if ra else None),
                        rb["prediction"] if rb else "—",
                        ref,
                        fmt_metrics(rb),
                        kw_b,
                        _fmt_context(rb.get("raw_context")) if rb else "",
                        _fmt_retrieval_info(rb.get("retrieval_info") if rb else None),
                        ev_md,
                    )

                question_table.select(
                    show_detail_full,
                    inputs=[_aligned_rows, metric_dd],
                    outputs=[
                        detail_pred_a, detail_ref, detail_metrics_a,
                        detail_keywords_a, detail_context_a, detail_retrieval_a,
                        detail_pred_b, detail_ref_b, detail_metrics_b,
                        detail_keywords_b, detail_context_b, detail_retrieval_b,
                        evidence_md,
                    ],
                )

            # ================================================================
            # TAB 2: Across-Run Variance
            # ================================================================
            with gr.Tab("📈 Across-Run Variance"):
                gr.Markdown(
                    "Analyse how consistent an experiment is across multiple QA runs. "
                    "High std = unstable for that question."
                )
                with gr.Row():
                    var_exp_dd = gr.Dropdown(experiments, label="Experiment", value=experiments[0])
                    var_c_run_dd = gr.Dropdown([], label="Construction Run", value=None)
                    var_metric_dd = gr.Dropdown(METRIC_CHOICES, value="f1", label="Metric")

                var_btn = gr.Button("🔄 Compute Variance", variant="primary")

                var_table = gr.Dataframe(
                    label="Questions sorted by Std (highest = most unstable)",
                    interactive=False,
                    wrap=False,
                    show_search="filter",
                )

                _var_data = gr.State({})

                gr.Markdown("---\n### 📉 Per-Question Run Distribution")
                with gr.Row():
                    with gr.Column():
                        var_detail_q = gr.Textbox(label="Question", interactive=False)
                        var_detail_ref = gr.Textbox(label="Reference", interactive=False)
                    with gr.Column():
                        var_detail_stats = gr.Textbox(
                            label="Stats (mean / std / min / max / runs>0.5)",
                            interactive=False,
                        )
                var_detail_values = gr.Textbox(
                    label="Per-run values", interactive=False, lines=3
                )

                # ---- wiring ----
                def update_var_c_runs(exp_id):
                    runs = list_construction_runs(exp_id)
                    choices = [str(r) for r in runs]
                    return gr.Dropdown(choices=choices, value=choices[0] if choices else None)

                var_exp_dd.change(update_var_c_runs, inputs=var_exp_dd, outputs=var_c_run_dd)

                def do_variance(exp_id, c_run, metric):
                    if not exp_id or c_run is None:
                        return pd.DataFrame(), {}
                    df, var_data = build_variance_df(exp_id, int(c_run), metric)
                    return df, var_data

                var_btn.click(
                    do_variance,
                    inputs=[var_exp_dd, var_c_run_dd, var_metric_dd],
                    outputs=[var_table, _var_data],
                )

                def show_var_detail(evt: gr.SelectData, var_data: dict, var_df: pd.DataFrame):
                    if evt is None or not var_data:
                        return "", "", "", ""
                    row_idx = evt.index[0]
                    if var_df is None or row_idx >= len(var_df):
                        return "", "", "", ""
                    # Find the question_key from the df row
                    # The df is sorted, so we need to match by question text
                    q_text_truncated = var_df.iloc[row_idx]["Question"]
                    # Find matching entry in var_data
                    entry = None
                    for e in var_data.values():
                        q = e["question"]
                        truncated = q[:80] + ("…" if len(q) > 80 else "")
                        if truncated == q_text_truncated:
                            entry = e
                            break
                    if entry is None:
                        return "", "", "", ""
                    stats_str = (
                        f"Mean={entry['mean']:.4f}  Std={entry['std']:.4f}  "
                        f"Min={entry['min']:.4f}  Max={entry['max']:.4f}  "
                        f"Runs>{THRESHOLD_DEFAULT}={entry['runs_above_half']}/{entry['total_runs']}"
                    )
                    values_str = ", ".join(f"{v:.3f}" for v in entry["values"])
                    return entry["question"], str(entry["reference"]), stats_str, values_str

                var_table.select(
                    show_var_detail,
                    inputs=[_var_data, var_table],
                    outputs=[var_detail_q, var_detail_ref, var_detail_stats, var_detail_values],
                )

        # ---- initialize dropdowns on load ----
        def init_dropdowns():
            if not experiments or experiments[0] == "(no experiments found)":
                return [None, None, None, None, None]
            c_runs_a = list_construction_runs(experiments[0])
            c_runs_b = list_construction_runs(experiments[-1] if len(experiments) > 1 else experiments[0])
            qa_runs_a = list_qa_runs(experiments[0], c_runs_a[0]) if c_runs_a else []
            qa_runs_b = list_qa_runs(experiments[-1] if len(experiments) > 1 else experiments[0],
                                     c_runs_b[0]) if c_runs_b else []
            return (
                gr.Dropdown(choices=[str(r) for r in c_runs_a], value=str(c_runs_a[0]) if c_runs_a else None),
                gr.Dropdown(choices=[str(r) for r in qa_runs_a], value=str(qa_runs_a[0]) if qa_runs_a else None),
                gr.Dropdown(choices=[str(r) for r in c_runs_b], value=str(c_runs_b[0]) if c_runs_b else None),
                gr.Dropdown(choices=[str(r) for r in qa_runs_b], value=str(qa_runs_b[0]) if qa_runs_b else None),
                gr.Dropdown(choices=[str(r) for r in c_runs_a], value=str(c_runs_a[0]) if c_runs_a else None),
            )

        app.load(
            init_dropdowns,
            inputs=[],
            outputs=[c_run_a_dd, qa_run_a_dd, c_run_b_dd, qa_run_b_dd, var_c_run_dd],
        )

    return app


if __name__ == "__main__":
    app = make_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
