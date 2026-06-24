#!/usr/bin/env python3
"""Print a compact table from A-MEM k-sweep result JSON files."""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path


CAT_LABELS = {
    "overall": "Overall",
    "category_1": "Multi-Hop",
    "category_2": "Temporal",
    "category_3": "Open-Dom",
    "category_4": "Single-Hop",
    "category_5": "Adversarial",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print A-MEM k-sweep F1 summaries")
    parser.add_argument("--results-dir", type=Path, default=Path("results_k_sweep_ollama"))
    parser.add_argument("--model", default="", help="Optional filename prefix filter")
    return parser.parse_args()


def load_rows(results_dir: Path, model_filter: str) -> list[dict]:
    pattern = str(results_dir / "*.json")
    rows = []
    for path_text in sorted(glob.glob(pattern)):
        path = Path(path_text)
        if model_filter and model_filter not in path.name:
            continue
        name = path.stem
        if "_k" not in name:
            continue
        model_name, k_text = name.rsplit("_k", 1)
        k_value = int(k_text.split("-", 1)[0])
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        metrics = data.get("aggregate_metrics", {})
        row = {"model": model_name, "k": k_value}
        for key, label in CAT_LABELS.items():
            row[label] = metrics.get(key, {}).get("f1", {}).get("mean", float("nan")) * 100
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    rows = load_rows(args.results_dir, args.model)
    if not rows:
        print(f"No result JSON files found in {args.results_dir}")
        return

    by_model: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)

    for model, model_rows in sorted(by_model.items()):
        model_rows.sort(key=lambda row: row["k"])
        best = max(model_rows, key=lambda row: row["Overall"])
        print(f"\nModel: {model}")
        print(f"{'k':>4} | {'Overall':>8} | {'Multi-Hop':>9} | {'Temporal':>8} | {'Open-Dom':>8} | {'Single-Hop':>10} | {'Adversarial':>11}")
        print("-" * 86)
        for row in model_rows:
            marker = "  BEST" if row["k"] == best["k"] else ""
            print(
                f"{row['k']:>4} | {row['Overall']:>7.2f}% | {row['Multi-Hop']:>8.2f}% | "
                f"{row['Temporal']:>7.2f}% | {row['Open-Dom']:>7.2f}% | "
                f"{row['Single-Hop']:>9.2f}% | {row['Adversarial']:>10.2f}%{marker}"
            )


if __name__ == "__main__":
    main()
