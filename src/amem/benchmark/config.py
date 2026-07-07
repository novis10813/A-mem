"""Component benchmark config objects and legacy translation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class ComponentConfig:
    adapter: str
    params: Mapping[str, Any] = field(default_factory=dict)
    hooks: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class RetrievalConfig(ComponentConfig):
    stages: tuple[Mapping[str, Any], ...] = ()
    tools: tuple[str, ...] = ()
    view: str | None = None


@dataclass(frozen=True)
class QAConfig(ComponentConfig):
    runs: int = 1
    backend: str = "ollama"
    model: str = "llama3.2:1b"


@dataclass(frozen=True)
class RunConfig:
    resume: bool = False
    hooks: tuple[Mapping[str, Any], ...] = field(
        default_factory=lambda: (
            {"type": "token_usage", "mode": "reported_or_estimated", "estimate_when_missing": True},
        )
    )


@dataclass(frozen=True)
class BenchmarkConfig:
    experiment_id: str
    dataset: str
    construction: ComponentConfig
    retrieval: RetrievalConfig
    context: ComponentConfig
    qa: QAConfig
    metrics: tuple[str, ...] = ("f1", "bleu1")
    run: RunConfig = field(default_factory=RunConfig)


def translate_legacy_config(config: Any) -> BenchmarkConfig:
    stages = tuple(
        {
            "type": stage.type,
            "name": getattr(stage, "name", stage.type),
            "top_k": stage.top_k,
            **({"query": stage.query} if hasattr(stage, "query") else {}),
            **({"model": stage.model} if getattr(stage, "model", None) else {}),
            **({"batch_size": stage.batch_size} if hasattr(stage, "batch_size") else {}),
        }
        for stage in config.evaluation.retrieval_pipeline.stages
    )
    return BenchmarkConfig(
        experiment_id=config.experiment_id,
        dataset=str(config.dataset),
        construction=ComponentConfig(
            adapter="amem",
            params={
                "runs": config.construction.runs,
                "keyword_pruning_mode": config.construction.keyword_pruning_mode,
                "embedding_model": config.construction.embedding_model,
            },
        ),
        retrieval=RetrievalConfig(adapter="pipeline", stages=stages),
        context=ComponentConfig(adapter="amem_full"),
        qa=QAConfig(
            adapter="robust_plain_text",
            runs=config.evaluation.qa_runs,
            backend=config.backend.name,
            model=config.backend.model,
        ),
        run=RunConfig(resume=bool(config.run.resume)),
    )
