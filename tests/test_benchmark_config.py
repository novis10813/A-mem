from types import SimpleNamespace

import pytest

from amem.benchmark.config import BenchmarkConfig, translate_legacy_config
from amem.benchmark.registry import AdapterRegistry


def test_registry_returns_registered_adapter_and_rejects_unknown():
    registry = AdapterRegistry()
    adapter = object()
    registry.register("construction", "amem", adapter)

    assert registry.get("construction", "amem") is adapter
    with pytest.raises(KeyError, match="Unknown retrieval adapter: missing"):
        registry.get("retrieval", "missing")


def test_translate_legacy_config_preserves_existing_experiment_shape():
    legacy = SimpleNamespace(
        experiment_id="exp",
        dataset="data/locomo10.json",
        construction=SimpleNamespace(
            runs=1,
            keyword_pruning_mode="nltk",
            embedding_model="all-MiniLM-L6-v2",
        ),
        evaluation=SimpleNamespace(
            qa_mode="robust",
            qa_runs=2,
            retrieval_pipeline=SimpleNamespace(
                final_k=10,
                stages=(SimpleNamespace(type="embedding", name="embedding_candidates", top_k=10),),
            ),
        ),
        backend=SimpleNamespace(name="ollama", model="llama3.2:1b"),
        run=SimpleNamespace(resume=True),
    )

    config = translate_legacy_config(legacy)

    assert isinstance(config, BenchmarkConfig)
    assert config.construction.adapter == "amem"
    assert config.retrieval.adapter == "pipeline"
    assert config.qa.adapter == "robust_plain_text"
    assert config.run.hooks[0]["type"] == "token_usage"
