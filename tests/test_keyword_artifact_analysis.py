import csv
import json
import pickle
from types import SimpleNamespace

from scripts.analyze_keyword_artifacts import analyze_cache


def write_cache(path, memories):
    with path.open("wb") as f:
        pickle.dump(memories, f)


def test_analyze_cache_writes_reports_without_mutating_inputs(tmp_path):
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "out"
    cache_dir.mkdir()

    sample0 = {
        "a": SimpleNamespace(
            content="Speaker Carolinesays : Caroline joined a charity race.",
            keywords=["carolinesays", "charity race", "race"],
        ),
        "b": SimpleNamespace(
            content="Wow, thanks! Photography helped the project.",
            keywords="wow, thanks, photography, project",
        ),
    }
    sample1 = {
        "c": SimpleNamespace(
            content="John enjoyed an RPG game with deep exploration.",
            keywords=["johnsays", "rpg game", "game"],
        )
    }
    cache0 = cache_dir / "memory_cache_sample_0.pkl"
    cache1 = cache_dir / "memory_cache_sample_1.pkl"
    write_cache(cache0, sample0)
    write_cache(cache1, sample1)
    before_bytes = cache0.read_bytes()

    summary = analyze_cache(cache_dir, output_dir)

    assert cache0.read_bytes() == before_bytes
    assert summary["sample_count"] == 2
    assert summary["memory_count"] == 3
    assert summary["notes_with_removed_keywords"] == 3
    assert summary["total_keywords_before"] == 10
    assert summary["total_keywords_after"] == 4
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "global_top_tokens.csv").exists()
    assert (output_dir / "per_sample_top_tokens.csv").exists()
    assert (output_dir / "removed_keywords.csv").exists()

    saved_summary = json.loads((output_dir / "summary.json").read_text())
    assert saved_summary["emptied_notes"] == 0

    with (output_dir / "removed_keywords.csv").open() as f:
        removed_rows = list(csv.DictReader(f))
    removed = {row["keyword"]: int(row["count"]) for row in removed_rows}
    assert removed["carolinesays"] == 1
    assert removed["johnsays"] == 1
    assert removed["wow"] == 1
    assert removed["thanks"] == 1

    with (output_dir / "per_sample_top_tokens.csv").open() as f:
        per_sample_rows = list(csv.DictReader(f))
    assert {row["sample_idx"] for row in per_sample_rows} == {"0", "1"}
