#!/usr/bin/env python3
"""Visualize original vs rule-pruned A-MEM keywords."""

from __future__ import annotations

import argparse
import csv
import html
import json
import pickle
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (SRC_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from amem.llm_text_parsers import (
    KEYWORD_CONVERSATION_FILLERS,
    KEYWORD_FILTER_TERMS,
    KEYWORD_GENERIC_TERMS,
    KEYWORD_HARD_FILTER_TERMS,
    KEYWORD_STOPWORDS,
    KEYWORD_TIME_TERMS,
    _is_artifact_token,
    _keyword_tokens,
    _light_stem,
    _normalize_keyword,
    _parse_list_items,
    sanitize_keywords,
)


class _Stub:
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state: dict):
        self.__dict__.update(state)


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except Exception:
            return _Stub


def load_memory_cache(path: Path) -> dict:
    with path.open("rb") as f:
        return SafeUnpickler(f).load()


def iter_cache_files(cache_dir: Path) -> list[Path]:
    return sorted(cache_dir.glob("memory_cache_sample_*.pkl"))


def sample_idx_from_path(path: Path) -> int:
    match = re.search(r"memory_cache_sample_(\d+)\.pkl$", path.name)
    if not match:
        raise ValueError(f"Unexpected cache filename: {path.name}")
    return int(match.group(1))


def note_keywords(note: Any) -> list[str]:
    keywords = getattr(note, "keywords", []) or []
    if isinstance(keywords, str):
        return _parse_list_items(keywords)
    return list(keywords)


def normalized_keywords(keywords: Iterable[Any]) -> list[str]:
    result = []
    seen = set()
    for keyword in keywords:
        normalized = _normalize_keyword(keyword)
        if normalized and normalized not in seen:
            result.append(normalized)
            seen.add(normalized)
    return result


def legacy_light_stem(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def sanitize_with_stem(content: str, keywords: list[str], stem: Callable[[str], str]) -> list[str]:
    content_tokens = set(_keyword_tokens(content or ""))
    content_tokens |= {stem(token) for token in content_tokens}
    content_lower = (content or "").lower()
    seen = set()
    filtered = []

    for raw_keyword in keywords:
        keyword = _normalize_keyword(raw_keyword)
        if not keyword or keyword in seen:
            continue
        seen.add(keyword)
        tokens = _keyword_tokens(keyword)
        if any(_is_artifact_token(token) for token in tokens):
            continue
        if any(token in KEYWORD_HARD_FILTER_TERMS for token in tokens):
            continue
        meaningful_tokens = [token for token in tokens if token not in KEYWORD_FILTER_TERMS]
        if not meaningful_tokens:
            continue
        if content_tokens and not all(token in content_tokens or stem(token) in content_tokens for token in meaningful_tokens):
            continue
        filtered.append((keyword, meaningful_tokens))

    kept = []
    for keyword, tokens in sorted(filtered, key=lambda item: (-len(item[1]), item[0])):
        token_set = set(tokens)
        if any(token_set < set(existing_tokens) for _, existing_tokens in kept):
            continue
        kept.append((keyword, tokens))

    def score(item):
        keyword, tokens = item
        frequency = sum(content_lower.count(token) for token in tokens)
        positions = [
            content_lower.find(token)
            for token in tokens
            if content_lower.find(token) >= 0
        ]
        first_pos = min(positions) if positions else len(content_lower) + 1
        phrase_bonus = min(len(tokens), 3)
        return (-frequency, -phrase_bonus, first_pos, keyword)

    return [keyword for keyword, _ in sorted(kept, key=score)[:5]]


def removal_category(keyword: str, content: str, output_keywords: set[str], stem: Callable[[str], str]) -> str:
    tokens = _keyword_tokens(keyword)
    if any(_is_artifact_token(token) for token in tokens):
        return "speaker_artifact"
    if any(token in KEYWORD_CONVERSATION_FILLERS for token in tokens):
        return "conversation_filler"
    if any(token in KEYWORD_TIME_TERMS for token in tokens):
        return "time_term"
    if any(token in KEYWORD_GENERIC_TERMS for token in tokens):
        return "generic_term"

    content_tokens = set(_keyword_tokens(content or ""))
    content_tokens |= {stem(token) for token in content_tokens}
    meaningful_tokens = [token for token in tokens if token not in KEYWORD_FILTER_TERMS]
    if meaningful_tokens and not all(token in content_tokens or stem(token) in content_tokens for token in meaningful_tokens):
        return "ungrounded"

    token_set = set(meaningful_tokens)
    for kept in output_keywords:
        kept_tokens = set(token for token in _keyword_tokens(kept) if token not in KEYWORD_FILTER_TERMS)
        if token_set and token_set < kept_tokens:
            return "phrase_subsumed"

    if any(token in KEYWORD_STOPWORDS for token in tokens):
        return "stopword"
    return "ranking_or_cap"


def percentile(values: list[int], pct: float) -> float:
    if not values:
        return 0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((len(ordered) - 1) * pct)))
    return ordered[idx]


def collect_rows(cache_dir: Path) -> list[dict]:
    rows = []
    cache_files = iter_cache_files(cache_dir)
    if not cache_files:
        raise FileNotFoundError(f"No memory_cache_sample_*.pkl files found in {cache_dir}")

    for cache_file in cache_files:
        sample_idx = sample_idx_from_path(cache_file)
        for note_id, note in load_memory_cache(cache_file).items():
            content = getattr(note, "content", "") or ""
            original = normalized_keywords(note_keywords(note))
            rule = sanitize_with_stem(content, original, legacy_light_stem)
            stemmed = sanitize_keywords(content, original)
            rows.append({
                "sample_idx": sample_idx,
                "note_id": str(note_id),
                "content": content,
                "original": original,
                "rule_pruning": rule,
                "nltk_stem": stemmed,
            })
    return rows


def variant_stats(rows: list[dict], variant: str) -> dict:
    counts = [len(row[variant]) for row in rows]
    keywords = [keyword for row in rows for keyword in row[variant]]
    freq = Counter(keywords)
    return {
        "variant": variant,
        "note_count": len(rows),
        "total_keywords": len(keywords),
        "unique_keywords": len(freq),
        "avg_keywords_per_note": len(keywords) / len(rows) if rows else 0,
        "median_keywords_per_note": statistics.median(counts) if counts else 0,
        "p90_keywords_per_note": percentile(counts, 0.9),
        "empty_keyword_notes": sum(1 for count in counts if count == 0),
        "top_keywords": freq.most_common(30),
    }


def compare_variants(rows: list[dict], before: str, after: str, stem: Callable[[str], str]) -> dict:
    removed = Counter()
    removed_categories = Counter()
    recovered = Counter()
    notes_with_removed = 0
    notes_with_recovered = 0

    for row in rows:
        before_set = set(row[before])
        after_set = set(row[after])
        removed_items = before_set - after_set
        recovered_items = after_set - before_set
        if removed_items:
            notes_with_removed += 1
        if recovered_items:
            notes_with_recovered += 1
        for keyword in removed_items:
            removed[keyword] += 1
            removed_categories[removal_category(keyword, row["content"], after_set, stem)] += 1
        for keyword in recovered_items:
            recovered[keyword] += 1

    return {
        "before": before,
        "after": after,
        "removed": removed,
        "removed_categories": removed_categories,
        "recovered": recovered,
        "notes_with_removed": notes_with_removed,
        "notes_with_recovered": notes_with_recovered,
    }


def per_sample_stats(rows: list[dict], variants: list[str]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["sample_idx"]].append(row)

    output = []
    for sample_idx in sorted(grouped):
        sample_rows = grouped[sample_idx]
        item = {"sample_idx": sample_idx, "note_count": len(sample_rows)}
        for variant in variants:
            counts = [len(row[variant]) for row in sample_rows]
            keywords = [keyword for row in sample_rows for keyword in row[variant]]
            item[f"{variant}_total"] = len(keywords)
            item[f"{variant}_unique"] = len(set(keywords))
            item[f"{variant}_empty"] = sum(1 for count in counts if count == 0)
        output.append(item)
    return output


def compact_text(value: str, limit: int = 220) -> str:
    text = re.sub(r"\s+", " ", value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def empty_note_examples(rows: list[dict], variants: list[str], limit: int = 30) -> list[dict]:
    examples = []
    for variant in variants:
        count = 0
        for row in rows:
            if row[variant]:
                continue
            examples.append({
                "variant": variant,
                "sample_idx": row["sample_idx"],
                "note_id": row["note_id"],
                "original_keywords": "; ".join(row["original"]),
                "rule_pruning_keywords": "; ".join(row["rule_pruning"]),
                "nltk_stem_keywords": "; ".join(row["nltk_stem"]),
                "content_snippet": compact_text(row["content"]),
            })
            count += 1
            if count >= limit:
                break
    return examples


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def esc(value: Any) -> str:
    return html.escape(str(value))


def bar(width: float, label: str, color: str = "#2563eb") -> str:
    pct = max(0, min(100, width))
    return f'<div class="bar"><span style="width:{pct:.1f}%;background:{color}"></span><em>{esc(label)}</em></div>'


def metric_card(label: str, value: Any) -> str:
    return f'<div class="card"><div class="metric">{esc(value)}</div><div class="label">{esc(label)}</div></div>'


def table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th>{esc(header)}</th>" for header in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{esc(cell)}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def render_keyword_bars(counter: Counter, limit: int = 25, color: str = "#2563eb") -> str:
    if not counter:
        return "<p>No data</p>"
    max_count = max(counter.values())
    items = []
    for keyword, count in counter.most_common(limit):
        items.append(f"<div class='bar-row'><code>{esc(keyword)}</code>{bar(count / max_count * 100, count, color)}</div>")
    return "\n".join(items)


def render_html(
    cache_dir: Path,
    stats: list[dict],
    comparisons: list[dict],
    sample_rows: list[dict],
    empty_examples: list[dict],
) -> str:
    stat_by_variant = {item["variant"]: item for item in stats}
    max_total = max(item["total_keywords"] for item in stats)
    max_unique = max(item["unique_keywords"] for item in stats)
    max_empty = max(item["empty_keyword_notes"] for item in stats) or 1

    summary_rows = []
    for item in stats:
        summary_rows.append([
            item["variant"],
            item["total_keywords"],
            item["unique_keywords"],
            f'{item["avg_keywords_per_note"]:.2f}',
            item["median_keywords_per_note"],
            item["p90_keywords_per_note"],
            item["empty_keyword_notes"],
        ])

    comparison_summary_rows = []
    for comp in comparisons:
        comparison_summary_rows.append([
            f'{comp["before"]} -> {comp["after"]}',
            sum(comp["removed"].values()),
            len(comp["removed"]),
            comp["notes_with_removed"],
            sum(comp["recovered"].values()),
            len(comp["recovered"]),
            comp["notes_with_recovered"],
        ])

    sample_table_rows = []
    for row in sample_rows:
        sample_table_rows.append([
            row["sample_idx"],
            row["note_count"],
            row["original_unique"],
            row["rule_pruning_unique"],
            row["nltk_stem_unique"],
            row["original_empty"],
            row["rule_pruning_empty"],
            row["nltk_stem_empty"],
        ])

    empty_example_rows = []
    for row in empty_examples:
        empty_example_rows.append([
            row["variant"],
            row["sample_idx"],
            row["note_id"],
            row["original_keywords"],
            row["content_snippet"],
        ])

    original = stat_by_variant["original"]
    rule = stat_by_variant["rule_pruning"]
    stemmed = stat_by_variant["nltk_stem"]
    cards = "".join([
        metric_card("Notes", original["note_count"]),
        metric_card("Original unique", original["unique_keywords"]),
        metric_card("Rule unique", rule["unique_keywords"]),
        metric_card("Stemmed unique", stemmed["unique_keywords"]),
        metric_card("Rule empty notes", rule["empty_keyword_notes"]),
        metric_card("Stemmed empty notes", stemmed["empty_keyword_notes"]),
    ])

    metric_bars = []
    for item in stats:
        name = item["variant"]
        metric_bars.append(f"<h3>{esc(name)}</h3>")
        metric_bars.append(bar(item["total_keywords"] / max_total * 100, f'total {item["total_keywords"]}', "#059669"))
        metric_bars.append(bar(item["unique_keywords"] / max_unique * 100, f'unique {item["unique_keywords"]}', "#2563eb"))
        metric_bars.append(bar(item["empty_keyword_notes"] / max_empty * 100, f'empty {item["empty_keyword_notes"]}', "#dc2626"))

    comparison_sections = []
    for comp in comparisons:
        category_max = max(comp["removed_categories"].values()) if comp["removed_categories"] else 1
        category_rows = [
            f"<div class='bar-row'><code>{esc(cat)}</code>{bar(count / category_max * 100, count, '#7c3aed')}</div>"
            for cat, count in comp["removed_categories"].most_common()
        ]
        comparison_sections.append(f"""
        <section>
          <h2>{esc(comp["before"])} -> {esc(comp["after"])}</h2>
          <div class="grid two">
            <div>
              <h3>Top Removed Keywords</h3>
              {render_keyword_bars(comp["removed"], 25, "#dc2626")}
            </div>
            <div>
              <h3>Removed By Category</h3>
              {"".join(category_rows)}
            </div>
          </div>
          <h3>Top Recovered / Newly Kept Keywords</h3>
          {render_keyword_bars(comp["recovered"], 20, "#059669")}
        </section>
        """)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>A-MEM Keyword Pruning Comparison</title>
  <style>
    :root {{ --bg:#f8fafc; --ink:#0f172a; --muted:#64748b; --line:#cbd5e1; --card:#ffffff; }}
    body {{ margin:0; font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background:var(--bg); color:var(--ink); }}
    header {{ padding:32px 40px 24px; background:#111827; color:white; }}
    header h1 {{ margin:0 0 8px; font-size:28px; }}
    header p {{ margin:0; color:#cbd5e1; }}
    main {{ padding:28px 40px 48px; max-width:1400px; margin:0 auto; }}
    section {{ margin:0 0 32px; }}
    h2 {{ margin:0 0 16px; font-size:21px; }}
    h3 {{ margin:16px 0 10px; font-size:15px; color:#334155; }}
    .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin:0 0 24px; }}
    .card {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:16px; }}
    .metric {{ font-size:28px; font-weight:750; }}
    .label {{ color:var(--muted); font-size:13px; margin-top:4px; }}
    .grid {{ display:grid; gap:20px; }}
    .two {{ grid-template-columns:repeat(auto-fit,minmax(360px,1fr)); }}
    .panel {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:16px; }}
    .bar {{ height:26px; background:#e2e8f0; border-radius:5px; position:relative; overflow:hidden; }}
    .bar span {{ display:block; height:100%; min-width:2px; }}
    .bar em {{ position:absolute; inset:0; display:flex; align-items:center; padding-left:8px; font-style:normal; font-size:12px; color:#0f172a; font-weight:650; }}
    .bar-row {{ display:grid; grid-template-columns:minmax(120px, 240px) 1fr; gap:10px; align-items:center; margin:6px 0; }}
    code {{ background:#e2e8f0; border-radius:4px; padding:3px 5px; font-size:12px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
    table {{ width:100%; border-collapse:collapse; background:var(--card); border:1px solid var(--line); border-radius:8px; overflow:hidden; }}
    th,td {{ padding:8px 10px; border-bottom:1px solid #e2e8f0; text-align:left; font-size:13px; }}
    th {{ background:#f1f5f9; color:#334155; }}
  </style>
</head>
<body>
  <header>
    <h1>A-MEM Keyword Pruning Comparison</h1>
    <p>Cache: {esc(cache_dir)}</p>
  </header>
  <main>
    <section class="cards">{cards}</section>
    <section class="grid two">
      <div class="panel">
        <h2>Metric Bars</h2>
        {"".join(metric_bars)}
      </div>
      <div class="panel">
        <h2>Version Summary</h2>
        {table(["Variant","Total","Unique","Avg/note","Median","P90","Empty notes"], summary_rows)}
      </div>
    </section>
    <section>
      <h2>Filter Impact Summary</h2>
      {table(["Comparison","Removed total","Removed unique","Notes with removed","Recovered total","Recovered unique","Notes with recovered"], comparison_summary_rows)}
    </section>
    {"".join(comparison_sections)}
    <section>
      <h2>Per-Sample Breakdown</h2>
      {table(["Sample","Notes","Original unique","Rule unique","Stem unique","Original empty","Rule empty","Stem empty"], sample_table_rows)}
    </section>
    <section>
      <h2>Empty Note Examples</h2>
      {table(["Variant","Sample","Note ID","Original keywords","Content snippet"], empty_example_rows)}
    </section>
  </main>
</body>
</html>"""


def write_outputs(
    output_dir: Path,
    output_html: Path,
    rows: list[dict],
    stats: list[dict],
    comparisons: list[dict],
    sample_rows: list[dict],
    empty_examples: list[dict],
    cache_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_html.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "cache_dir": str(cache_dir),
        "variants": [
            {k: v for k, v in item.items() if k != "top_keywords"}
            for item in stats
        ],
        "comparisons": [
            {
                "before": comp["before"],
                "after": comp["after"],
                "notes_with_removed": comp["notes_with_removed"],
                "notes_with_recovered": comp["notes_with_recovered"],
                "total_removed_keywords": sum(comp["removed"].values()),
                "unique_removed_keywords": len(comp["removed"]),
                "total_recovered_keywords": sum(comp["recovered"].values()),
                "unique_recovered_keywords": len(comp["recovered"]),
                "removed_categories": dict(comp["removed_categories"]),
                "top_removed": comp["removed"].most_common(50),
                "top_recovered": comp["recovered"].most_common(50),
            }
            for comp in comparisons
        ],
        "per_sample": sample_rows,
        "empty_note_examples": empty_examples,
    }
    with (output_dir / "keyword_pruning_comparison_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    write_csv(
        output_dir / "keyword_pruning_variant_summary.csv",
        [
            {k: v for k, v in item.items() if k not in {"top_keywords"}}
            for item in stats
        ],
        [
            "variant",
            "note_count",
            "total_keywords",
            "unique_keywords",
            "avg_keywords_per_note",
            "median_keywords_per_note",
            "p90_keywords_per_note",
            "empty_keyword_notes",
        ],
    )
    for item in stats:
        write_csv(
            output_dir / f'{item["variant"]}_top_keywords.csv',
            [{"keyword": kw, "count": count} for kw, count in item["top_keywords"]],
            ["keyword", "count"],
        )
    write_csv(
        output_dir / "empty_note_examples.csv",
        empty_examples,
        [
            "variant",
            "sample_idx",
            "note_id",
            "original_keywords",
            "rule_pruning_keywords",
            "nltk_stem_keywords",
            "content_snippet",
        ],
    )
    for comp in comparisons:
        slug = f'{comp["before"]}_to_{comp["after"]}'
        write_csv(
            output_dir / f"{slug}_removed_keywords.csv",
            [{"keyword": kw, "count": count} for kw, count in comp["removed"].most_common()],
            ["keyword", "count"],
        )
        write_csv(
            output_dir / f"{slug}_removed_categories.csv",
            [{"category": cat, "count": count} for cat, count in comp["removed_categories"].most_common()],
            ["category", "count"],
        )
        write_csv(
            output_dir / f"{slug}_recovered_keywords.csv",
            [{"keyword": kw, "count": count} for kw, count in comp["recovered"].most_common()],
            ["keyword", "count"],
        )

    output_html.write_text(
        render_html(cache_dir, stats, comparisons, sample_rows, empty_examples),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize A-MEM keyword pruning variants")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--output-html",
        type=Path,
        default=Path("artifacts/output/keyword_pruning_comparison.html"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/output/keyword_pruning_comparison"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = collect_rows(args.cache_dir)
    variants = ["original", "rule_pruning", "nltk_stem"]
    stats = [variant_stats(rows, variant) for variant in variants]
    comparisons = [
        compare_variants(rows, "original", "rule_pruning", legacy_light_stem),
        compare_variants(rows, "rule_pruning", "nltk_stem", _light_stem),
        compare_variants(rows, "original", "nltk_stem", _light_stem),
    ]
    samples = per_sample_stats(rows, variants)
    empty_examples = empty_note_examples(rows, ["rule_pruning", "nltk_stem"])
    write_outputs(args.output_dir, args.output_html, rows, stats, comparisons, samples, empty_examples, args.cache_dir)
    print(f"Wrote {args.output_html}")
    print(f"Wrote detail CSV/JSON files under {args.output_dir}")


if __name__ == "__main__":
    main()
