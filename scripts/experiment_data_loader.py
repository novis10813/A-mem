#!/usr/bin/env python3
"""Data loading and alignment utilities for the experiment comparison dashboard.

This module is UI-independent — all data logic lives here so it can be
tested independently of Gradio.

Key concepts
------------
- **experiment_id**: the directory name under ``artifacts/results/``, e.g.
  ``ollama_llama3.2-1b_nltk``
- **question_key**: a 12-char sha256 hex digest of ``"{sample_id}::{question}"``,
  used to stably align the same question across experiments
- **two-stage format**: ``results/<exp_id>/construction_run_NN/robust/qa_run_NN/results.json``

Only the *robust* QA mode is supported (content_keywords is treated as tech debt).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "artifacts" / "results"
DATASET_PATH = REPO_ROOT / "data" / "locomo10.json"

# ---------------------------------------------------------------------------
# Question key
# ---------------------------------------------------------------------------

def question_key(sample_id: int, question: str) -> str:
    """Return a stable 12-char hex key for a (sample_id, question) pair.

    Stable across experiments as long as the dataset does not change.
    """
    text = f"{sample_id}::{question}"
    return hashlib.sha256(text.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Experiment discovery
# ---------------------------------------------------------------------------

def is_two_stage_experiment(exp_dir: Path) -> bool:
    """Return True if the directory follows the two-stage robust result layout."""
    if not (exp_dir / "manifest.json").exists():
        return False
    # Must have at least one construction_run_*/robust/qa_run_*/results.json
    for results_file in exp_dir.glob("construction_run_*/robust/qa_run_*/results.json"):
        return True
    return False


def discover_experiments(results_root: Path = RESULTS_ROOT) -> list[str]:
    """Return sorted list of two-stage experiment IDs found under results_root."""
    if not results_root.exists():
        return []
    experiments = []
    for child in sorted(results_root.iterdir()):
        if child.is_dir() and is_two_stage_experiment(child):
            experiments.append(child.name)
    return experiments


def list_construction_runs(experiment_id: str, results_root: Path = RESULTS_ROOT) -> list[int]:
    """Return sorted list of construction run indices for an experiment."""
    exp_dir = results_root / experiment_id
    runs = []
    for path in sorted(exp_dir.glob("construction_run_*")):
        if not path.is_dir():
            continue
        m = re.search(r"construction_run_(\d+)$", path.name)
        if m:
            runs.append(int(m.group(1)))
    return sorted(runs)


def list_qa_runs(
    experiment_id: str,
    construction_run: int,
    results_root: Path = RESULTS_ROOT,
) -> list[int]:
    """Return sorted list of QA run indices for a given construction run."""
    mode_dir = (
        results_root / experiment_id / f"construction_run_{construction_run:02d}" / "robust"
    )
    if not mode_dir.exists():
        return []
    runs = []
    for path in sorted(mode_dir.glob("qa_run_*")):
        if not path.is_dir():
            continue
        m = re.search(r"qa_run_(\d+)$", path.name)
        if m and (path / "results.json").exists():
            runs.append(int(m.group(1)))
    return sorted(runs)


# ---------------------------------------------------------------------------
# Result loading
# ---------------------------------------------------------------------------

def load_result_file(path: Path) -> dict[str, Any]:
    """Load a results.json and add ``question_key`` to every individual result."""
    data = json.loads(path.read_text(encoding="utf-8"))
    for r in data.get("individual_results", []):
        r["question_key"] = question_key(r["sample_id"], r["question"])
    return data


def load_experiment_results(
    experiment_id: str,
    construction_run: int,
    qa_run: int,
    results_root: Path = RESULTS_ROOT,
) -> dict[str, Any]:
    """Load results for a specific (experiment, construction_run, qa_run) triple."""
    path = (
        results_root
        / experiment_id
        / f"construction_run_{construction_run:02d}"
        / "robust"
        / f"qa_run_{qa_run:02d}"
        / "results.json"
    )
    if not path.exists():
        raise FileNotFoundError(f"Results not found: {path}")
    return load_result_file(path)


def results_to_question_map(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index individual_results by question_key for O(1) lookup."""
    return {r["question_key"]: r for r in data.get("individual_results", [])}


# ---------------------------------------------------------------------------
# Cross-experiment alignment
# ---------------------------------------------------------------------------

def align_experiments(
    results_a: dict[str, Any],
    results_b: dict[str, Any],
) -> list[dict[str, Any]]:
    """Align two experiment result sets by question_key.

    Returns a list of dicts with keys:
        question_key, sample_id, question, reference, category,
        exp_a (result dict or None), exp_b (result dict or None)

    Questions that only appear in one experiment are included with None for
    the other side (shouldn't normally happen if both used the same dataset).
    """
    map_a = results_to_question_map(results_a)
    map_b = results_to_question_map(results_b)
    all_keys = sorted(set(map_a) | set(map_b))

    rows = []
    for qkey in all_keys:
        ra = map_a.get(qkey)
        rb = map_b.get(qkey)
        base = ra or rb  # at least one is non-None
        rows.append({
            "question_key": qkey,
            "sample_id": base["sample_id"],
            "qa_idx": base.get("qa_idx", 0),
            "question": base["question"],
            "reference": base["reference"],
            "category": base["category"],
            "exp_a": ra,
            "exp_b": rb,
        })
    return rows


# ---------------------------------------------------------------------------
# Across-run variance
# ---------------------------------------------------------------------------

def compute_across_run_variance(
    experiment_id: str,
    construction_run: int,
    metric: str = "f1",
    results_root: Path = RESULTS_ROOT,
) -> dict[str, dict[str, Any]]:
    """Compute per-question metric variance across all QA runs.

    Returns a dict keyed by question_key with:
        question, sample_id, category, reference,
        values (list of per-run metric values),
        mean, std, min, max,
        runs_above_half (count of runs where metric > 0.5),
        total_runs
    """
    qa_runs = list_qa_runs(experiment_id, construction_run, results_root)
    if not qa_runs:
        return {}

    # Accumulate per-question values across runs
    per_question: dict[str, dict[str, Any]] = {}

    for qa_run in qa_runs:
        try:
            data = load_experiment_results(experiment_id, construction_run, qa_run, results_root)
        except FileNotFoundError:
            continue
        for r in data.get("individual_results", []):
            qkey = r["question_key"]
            m_val = r.get("metrics", {}).get(metric, 0.0) or 0.0
            if qkey not in per_question:
                per_question[qkey] = {
                    "question_key": qkey,
                    "question": r["question"],
                    "sample_id": r["sample_id"],
                    "category": r["category"],
                    "reference": r["reference"],
                    "values": [],
                }
            per_question[qkey]["values"].append(float(m_val))

    # Compute summary statistics
    for qkey, entry in per_question.items():
        values = entry["values"]
        n = len(values)
        mean = sum(values) / n if n else 0.0
        variance = sum((v - mean) ** 2 for v in values) / n if n else 0.0
        std = variance ** 0.5
        entry["mean"] = round(mean, 4)
        entry["std"] = round(std, 4)
        entry["min"] = round(min(values), 4) if values else 0.0
        entry["max"] = round(max(values), 4) if values else 0.0
        entry["runs_above_half"] = sum(1 for v in values if v > 0.5)
        entry["total_runs"] = n

    return per_question


# ---------------------------------------------------------------------------
# Dataset / evidence resolver
# ---------------------------------------------------------------------------

_dataset_cache: list[dict[str, Any]] | None = None


def load_dataset(dataset_path: Path = DATASET_PATH) -> list[dict[str, Any]]:
    """Load the LoCoMo dataset (cached in memory after first call)."""
    global _dataset_cache
    if _dataset_cache is None:
        _dataset_cache = json.loads(dataset_path.read_text(encoding="utf-8"))
    return _dataset_cache


def get_sample(sample_id: int, dataset_path: Path = DATASET_PATH) -> dict[str, Any] | None:
    """Return the dataset sample with the given index."""
    dataset = load_dataset(dataset_path)
    for sample in dataset:
        if sample.get("sample_id") == sample_id:
            return sample
    # Fallback: treat sample_id as list index
    if 0 <= sample_id < len(dataset):
        return dataset[sample_id]
    return None


def get_qa_evidence(sample_id: int, question: str, dataset_path: Path = DATASET_PATH) -> list[str]:
    """Return the evidence refs (e.g. ['D1:3']) for a question in a sample."""
    sample = get_sample(sample_id, dataset_path)
    if sample is None:
        return []
    for qa in sample.get("qa", []):
        if qa.get("question") == question:
            return qa.get("evidence", [])
    return []


def resolve_evidence(
    sample_id: int,
    evidence_refs: list[str],
    dataset_path: Path = DATASET_PATH,
) -> list[dict[str, Any]]:
    """Resolve evidence refs like 'D1:3' to actual conversation turns.

    Each returned dict has:
        ref, session_num, date_time, speaker, text

    The format is ``D{session}:{turn_id}`` where session is 1-indexed and
    the turn is found by matching ``dia_id`` in the session list.
    """
    sample = get_sample(sample_id, dataset_path)
    if sample is None:
        return []

    conv = sample.get("conversation", {})
    results = []

    for ref in evidence_refs:
        # Parse e.g. "D1:3" or "D1:3; D2:4" (sometimes semicolons in evidence)
        # We treat each ref as a single dia_id
        m = re.match(r"D(\d+):(\d+)", ref.strip())
        if not m:
            results.append({"ref": ref, "error": "Cannot parse reference format"})
            continue

        session_num = int(m.group(1))
        session_key = f"session_{session_num}"
        date_time_key = f"session_{session_num}_date_time"

        session_turns = conv.get(session_key, [])
        date_time = conv.get(date_time_key, "")

        # Find the turn with matching dia_id
        matched_turn = None
        for turn in session_turns:
            if turn.get("dia_id") == ref.strip():
                matched_turn = turn
                break

        if matched_turn:
            results.append({
                "ref": ref,
                "session_num": session_num,
                "date_time": date_time,
                "speaker": matched_turn.get("speaker", ""),
                "text": matched_turn.get("text", ""),
            })
        else:
            results.append({
                "ref": ref,
                "session_num": session_num,
                "date_time": date_time,
                "error": f"Turn {ref} not found in session {session_num}",
            })

    return results


# ---------------------------------------------------------------------------
# Experiment metadata helpers
# ---------------------------------------------------------------------------

def load_experiment_manifest(
    experiment_id: str,
    results_root: Path = RESULTS_ROOT,
) -> dict[str, Any]:
    """Load the experiment manifest.json if it exists."""
    path = results_root / experiment_id / "manifest.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def get_experiment_label(experiment_id: str, results_root: Path = RESULTS_ROOT) -> str:
    """Return a human-readable label for an experiment."""
    manifest = load_experiment_manifest(experiment_id, results_root)
    # Try to extract useful config info from manifest
    cfg = manifest.get("config", {})
    model = cfg.get("model") or manifest.get("model", "")
    backend = cfg.get("backend") or manifest.get("backend", "")
    # Build a reasonable label
    parts = [experiment_id]
    if model:
        parts = [f"{model} ({backend})" if backend else model, experiment_id]
    return experiment_id  # keep it simple — show raw id, user knows their experiments


CATEGORY_LABELS = {
    1: "Cat 1 — Factual",
    2: "Cat 2 — Temporal",
    3: "Cat 3 — Inferential",
    4: "Cat 4 — Conversational",
    5: "Cat 5 — Existence",
}
