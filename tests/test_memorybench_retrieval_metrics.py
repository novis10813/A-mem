import pytest

from memorybench.components import DEFAULT_CROSS_ENCODER_MODEL, metric_scores, staged_retrieve
from memorybench.config import RetrievalConfig
from memorybench.schemas import MemoryRecord, MemoryStore, Question


def test_bm25_stage_uses_bm25_scores_and_deterministic_ranking():
    store = MemoryStore(sample_id="s", records=(
        MemoryRecord(record_id="long", text="museum " * 20 + "tickets"),
        MemoryRecord(record_id="focused", text="museum tickets cafe"),
        MemoryRecord(record_id="other", text="garden cooking"),
    ))
    question = Question(question_id="q", text="museum tickets", reference="")
    config = RetrievalConfig.model_validate({
        "adapter": "staged",
        "stages": [{"adapter": "bm25", "top_k": 2}],
    })

    result = staged_retrieve(question, store, config)

    assert [item["record_id"] for item in result["items"]] == ["focused", "long"]
    scores = result["stages"][0]["scores"]
    assert scores[0] > scores[1] > 0
    assert any(not float(score).is_integer() for score in scores)


def test_lightweight_metrics_match_expected_values():
    scores = metric_scores(
        ("exact_match", "f1", "bleu1"),
        prediction="Alice moved to Taipei",
        reference="Alice moved to Taipei yesterday",
    )

    assert scores["exact_match"] == 0.0
    assert scores["f1"] == pytest.approx(8 / 9)
    assert 0.0 < scores["bleu1"] < 1.0


def test_query_transform_stage_can_use_its_own_llm_and_records_usage():
    store = MemoryStore(sample_id="s", records=(
        MemoryRecord(record_id="m1", text="Taipei museum"),
    ))
    question = Question(question_id="q", text="Where did Alice go?", reference="")
    config = RetrievalConfig.model_validate({
        "adapter": "staged",
        "stages": [{
            "adapter": "query_transform",
            "top_k": 1,
            "llm": {
                "provider": "fake",
                "model": "keyword-model",
                "params": {"response": "Taipei museum"},
            },
        }, {"adapter": "bm25", "top_k": 1}],
    })

    result = staged_retrieve(question, store, config)

    assert result["stages"][0]["query"] == "Taipei museum"
    assert result["items"][0]["record_id"] == "m1"
    assert result["usage"][0]["model"] == "keyword-model"
    assert result["usage"][0]["source"] == "estimated"
    assert result["stages"][0]["stage_index"] == 0
    assert result["stages"][0]["latency_ms"] >= 0
    assert result["stages"][0]["usage"] == result["usage"]


def test_ranking_stage_can_select_original_question_after_query_transform():
    store = MemoryStore(sample_id="s", records=(
        MemoryRecord(record_id="original", text="Where Alice went"),
        MemoryRecord(record_id="transformed", text="Taipei museum"),
    ))
    question = Question(question_id="q", text="Where Alice went", reference="")
    config = RetrievalConfig.model_validate({
        "adapter": "staged",
        "stages": [
            {"adapter": "query_transform", "top_k": 2, "params": {"prefix": "Taipei museum "}},
            {"adapter": "bm25", "top_k": 1, "query": "original_question"},
        ],
    })

    result = staged_retrieve(question, store, config)

    assert result["items"][0]["record_id"] == "original"
    assert result["stages"][1]["query"] == "Where Alice went"
    assert DEFAULT_CROSS_ENCODER_MODEL == "cross-encoder/ms-marco-MiniLM-L6-v2"
