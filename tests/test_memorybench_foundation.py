import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from memorybench.config import MemoryBenchConfig, load_config
from memorybench.datasets.locomo import LoCoMoAdapter
from memorybench.registry import Registry


def minimal_config(**pipeline_overrides):
    pipeline = {
        "stages": ["construction", "retrieve_qa"],
        "dataset": {"adapter": "locomo", "path": "data/locomo10.json"},
        "construction": {"adapter": "turn_rag", "chunker": {"adapter": "turn"}},
        "retrieve_qa": {
            "retrieval": {"adapter": "staged", "stages": [{"adapter": "bm25", "top_k": 3}]},
            "context": {"adapter": "records", "fields": ["timestamp", "content"]},
            "qa": {"adapter": "extractive"},
        },
    }
    pipeline.update(pipeline_overrides)
    return {"experiment": {"id": "unit"}, "pipeline": pipeline, "runtime": {}}


def test_config_rejects_unknown_fields_and_has_stable_fingerprint():
    first = MemoryBenchConfig.model_validate(minimal_config())
    second = MemoryBenchConfig.model_validate(minimal_config())
    assert first.fingerprint == second.fingerprint
    payload = minimal_config()
    payload["runtime"]["surprise"] = True
    with pytest.raises(ValidationError):
        MemoryBenchConfig.model_validate(payload)


def test_retrieve_only_requires_memory_source():
    payload = minimal_config(stages=["retrieve_qa"])
    with pytest.raises(ValidationError, match="memory_source"):
        MemoryBenchConfig.model_validate(payload)


def test_registry_reports_unknown_adapter():
    registry = Registry("chunker")
    registry.register("turn", object)
    with pytest.raises(ValueError, match="Unknown chunker adapter 'window'.*turn"):
        registry.get("window")


def test_locomo_adapter_emits_stable_ids_taxonomy_and_turn_evidence(tmp_path: Path):
    raw = [{
        "qa": [{"question": "When?", "answer": "Today", "category": 2, "evidence": ["D1:1"]}],
        "conversation": {
            "speaker_a": "A", "speaker_b": "B", "session_1_date_time": "1 Jan 2026",
            "session_1": [{"speaker": "A", "dia_id": "D1:1", "text": "hello"}],
        },
        "event_summary": {}, "observation": {}, "session_summary": {},
    }]
    path = tmp_path / "locomo.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    bundle = LoCoMoAdapter().load(path)
    assert bundle.taxonomy.dimensions[0].values == ("1", "2", "3", "4", "5")
    assert bundle.samples[0].questions[0].question_id == "locomo:0:0"
    assert bundle.samples[0].turns[0].evidence_id == "D1:1"
    assert bundle.samples[0].questions[0].labels == {"question_type": "2"}


def test_yaml_load_round_trip(tmp_path: Path):
    import yaml
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(minimal_config()), encoding="utf-8")
    assert load_config(path).experiment.id == "unit"
