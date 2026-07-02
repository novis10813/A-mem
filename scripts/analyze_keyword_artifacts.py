#!/usr/bin/env python3
"""Analyze keyword artifacts in cached A-MEM memories."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (SRC_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from amem.llm_text_parsers import _keyword_tokens, _normalize_keyword, _parse_list_items, sanitize_keywords


def iter_cache_files(cache_dir: Path) -> list[Path]:
    return sorted(cache_dir.glob("memory_cache_sample_*.pkl"))


def sample_idx_from_path(path: Path) -> int:
    match = re.search(r"memory_cache_sample_(\d+)\.pkl$", path.name)
    if not match:
        raise ValueError(f"Unexpected cache filename: {path.name}")
    return int(match.group(1))


def load_memories(path: Path) -> dict:
    with path.open("rb") as f:
        return pickle.load(f)


def note_keywords(note: Any) -> list[str]:
    keywords = getattr(note, "keywords", []) or []
    if isinstance(keywords, str):
        return _parse_list_items(keywords)
    return list(keywords)


def normalized_keywords(keywords: Iterable[Any]) -> list[str]:
    return [kw for kw in (_normalize_keyword(k) for k in keywords) if kw]


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze_cache(cache_dir: Path, output_dir: Path) -> dict:
    cache_files = iter_cache_files(cache_dir)
    if not cache_files:
        raise FileNotFoundError(f"No memory_cache_sample_*.pkl files found in {cache_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    global_token_df = Counter()
    global_token_tf = Counter()
    removed_keywords = Counter()
    per_sample_token_df: dict[int, Counter] = defaultdict(Counter)
    sample_note_counts: dict[int, int] = {}
    before_total = 0
    after_total = 0
    changed_notes = 0
    notes_with_removed_keywords = 0
    emptied_notes = 0
    total_notes = 0

    for cache_file in cache_files:
        sample_idx = sample_idx_from_path(cache_file)
        memories = load_memories(cache_file)
        sample_note_counts[sample_idx] = len(memories)
        total_notes += len(memories)

        for note_id, note in memories.items():
            content = getattr(note, "content", "") or ""
            before = normalized_keywords(note_keywords(note))
            after = sanitize_keywords(content, before)
            after_set = set(after)

            before_total += len(before)
            after_total += len(after)
            if before != after:
                changed_notes += 1
            if set(before) - after_set:
                notes_with_removed_keywords += 1
            if before and not after:
                emptied_notes += 1

            note_tokens = set()
            for keyword in before:
                tokens = _keyword_tokens(keyword)
                note_tokens.update(tokens)
                global_token_tf.update(tokens)
                if keyword not in after_set:
                    removed_keywords[keyword] += 1

            for token in note_tokens:
                per_sample_token_df[sample_idx][token] += 1
                global_token_df[token] += 1

    global_rows = []
    for token, document_frequency in global_token_df.most_common():
        sample_ids = [s for s, counts in per_sample_token_df.items() if token in counts]
        ratios = [
            per_sample_token_df[s][token] / sample_note_counts[s]
            for s in sample_ids
            if sample_note_counts[s]
        ]
        global_rows.append({
            "token": token,
            "document_frequency": document_frequency,
            "term_frequency": global_token_tf[token],
            "sample_coverage": len(sample_ids),
            "avg_sample_df_ratio": round(sum(ratios) / len(ratios), 6) if ratios else 0,
            "max_sample_df_ratio": round(max(ratios), 6) if ratios else 0,
        })

    per_sample_rows = []
    for sample_idx in sorted(per_sample_token_df):
        note_count = sample_note_counts[sample_idx]
        for token, document_frequency in per_sample_token_df[sample_idx].most_common():
            per_sample_rows.append({
                "sample_idx": sample_idx,
                "token": token,
                "document_frequency": document_frequency,
                "sample_df_ratio": round(document_frequency / note_count, 6) if note_count else 0,
            })

    removed_rows = [
        {"keyword": keyword, "count": count}
        for keyword, count in removed_keywords.most_common()
    ]

    summary = {
        "cache_dir": str(cache_dir),
        "sample_count": len(cache_files),
        "memory_count": total_notes,
        "notes_with_order_or_ranking_changes": changed_notes,
        "notes_with_removed_keywords": notes_with_removed_keywords,
        "emptied_notes": emptied_notes,
        "total_keywords_before": before_total,
        "total_keywords_after": after_total,
        "avg_keywords_before": before_total / total_notes if total_notes else 0,
        "avg_keywords_after": after_total / total_notes if total_notes else 0,
    }

    write_csv(
        output_dir / "global_top_tokens.csv",
        global_rows,
        [
            "token",
            "document_frequency",
            "term_frequency",
            "sample_coverage",
            "avg_sample_df_ratio",
            "max_sample_df_ratio",
        ],
    )
    write_csv(
        output_dir / "per_sample_top_tokens.csv",
        per_sample_rows,
        ["sample_idx", "token", "document_frequency", "sample_df_ratio"],
    )
    write_csv(output_dir / "removed_keywords.csv", removed_rows, ["keyword", "count"])
    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze keyword artifacts in A-MEM caches")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = analyze_cache(args.cache_dir, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
