from pathlib import Path
from types import SimpleNamespace

import pytest

from amem import ablation


def make_memory(
    content="Speaker Alice says: I adopted a cat.",
    context="Pet adoption conversation",
    keywords=None,
    tags=None,
):
    return SimpleNamespace(
        content=content,
        context=context,
        keywords=keywords if keywords is not None else ["adoption", "cat"],
        tags=tags if tags is not None else ["pets", "family"],
    )


def test_build_embedding_text_uses_labeled_fields_in_requested_order():
    memory = make_memory()

    text = ablation.build_embedding_text(memory, ("content", "context", "keywords", "tags"))

    assert text == (
        "content: Speaker Alice says: I adopted a cat. "
        "context: Pet adoption conversation "
        "keywords: adoption, cat "
        "tags: pets, family"
    )


def test_build_embedding_text_skips_empty_metadata_fields():
    memory = make_memory(context="General", keywords=[], tags=[])

    text = ablation.build_embedding_text(memory, ("content", "context", "keywords", "tags"))

    assert text == "content: Speaker Alice says: I adopted a cat."


def test_expand_variants_core7_matches_planned_ablation_set():
    variants = ablation.expand_variants("core7")

    assert list(variants) == [
        "content",
        "content_context",
        "content_keywords",
        "content_tags",
        "content_keywords_tags",
        "content_context_keywords",
        "full",
    ]
    assert variants["full"] == ("content", "context", "keywords", "tags")


def test_expand_variants_accepts_comma_separated_known_variants():
    variants = ablation.expand_variants("content,full")

    assert variants == {
        "content": ("content",),
        "full": ("content", "context", "keywords", "tags"),
    }


def test_expand_variants_rejects_unknown_variant():
    with pytest.raises(ValueError, match="Unknown variant"):
        ablation.expand_variants("content,unknown")


def test_load_cached_memories_fails_fast_when_cache_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="memory_cache_sample_3.pkl"):
        ablation.load_cached_memories(tmp_path, sample_idx=3)


def test_extract_summary_rows_includes_overall_and_category_metrics():
    result = {
        "variant": "content",
        "total_questions": 2,
        "aggregate_metrics": {
            "overall": {
                "f1": {"mean": 0.25},
                "bleu1": {"mean": 0.5},
            },
            "category_1": {
                "f1": {"mean": 0.75},
                "bleu1": {"mean": 0.9},
            },
        },
    }

    rows = ablation.extract_summary_rows(result)

    assert rows == [
        {
            "variant": "content",
            "split": "overall",
            "total_questions": 2,
            "f1": 0.25,
            "bleu1": 0.5,
        },
        {
            "variant": "content",
            "split": "category_1",
            "total_questions": 2,
            "f1": 0.75,
            "bleu1": 0.9,
        },
    ]


def test_write_summary_files_creates_json_and_csv(tmp_path):
    results = [
        {
            "variant": "content",
            "total_questions": 1,
            "aggregate_metrics": {
                "overall": {
                    "f1": {"mean": 0.1},
                    "bleu1": {"mean": 0.2},
                }
            },
        }
    ]

    ablation.write_summary_files(results, tmp_path)

    summary_json = tmp_path / "summary.json"
    summary_csv = tmp_path / "summary.csv"
    assert summary_json.exists()
    assert summary_csv.exists()
    assert '"variant": "content"' in summary_json.read_text()
    assert summary_csv.read_text().splitlines() == [
        "variant,split,total_questions,f1,bleu1",
        "content,overall,1,0.1,0.2",
    ]


def test_build_answer_prompt_uses_supplied_category5_answer_options():
    prompt, temperature = ablation.build_answer_prompt(
        context="memory context",
        question="Was Alice's trip mentioned?",
        category=5,
        answer="Yes",
        temperature_c5=0.2,
        answer_options=("Yes", "Not mentioned in the conversation"),
    )

    assert temperature == 0.2
    assert "Select the correct answer: Yes or Not mentioned in the conversation" in prompt
