"""YAML config support for two-stage A-MEM experiments."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

try:
    from .experiment_common import DEFAULT_CACHE_ROOT, DEFAULT_LOG_ROOT, DEFAULT_RESULTS_ROOT
except ImportError:  # pragma: no cover - script execution path
    from experiment_common import DEFAULT_CACHE_ROOT, DEFAULT_LOG_ROOT, DEFAULT_RESULTS_ROOT


@dataclass(frozen=True)
class PathsConfig:
    cache_root: Path = DEFAULT_CACHE_ROOT
    results_root: Path = DEFAULT_RESULTS_ROOT
    log_root: Path = DEFAULT_LOG_ROOT


@dataclass(frozen=True)
class BackendConfig:
    name: str = "ollama"
    model: str = "llama3.2:1b"
    sglang_host: str = "http://localhost"
    sglang_port: int = 30000


@dataclass(frozen=True)
class ConstructionConfig:
    runs: int = 1
    keyword_pruning_mode: str = "nltk"
    embedding_model: str = "all-MiniLM-L6-v2"
    max_workers: int = 1


@dataclass(frozen=True)
class RetrievalStageConfig:
    type: str
    name: str
    top_k: int
    query: str = "similarity_query"
    model: str | None = None
    batch_size: int = 32


@dataclass(frozen=True)
class RetrievalPipelineConfig:
    final_k: int = 10
    stages: tuple[RetrievalStageConfig, ...] = field(
        default_factory=lambda: (
            RetrievalStageConfig(
                type="embedding",
                name="embedding_candidates",
                top_k=10,
                query="similarity_query",
            ),
        )
    )


@dataclass(frozen=True)
class EvaluationConfig:
    qa_mode: str = "content_keywords"
    qa_runs: int = 1
    cache_experiment_id: str | None = None
    keyword_conditions: tuple[str, ...] = ("none", "nltk")
    retrieve_k: int = 10
    retrieval_pipeline: RetrievalPipelineConfig = field(default_factory=RetrievalPipelineConfig)
    temperature_c5: float = 0.5
    max_keywords: int = 5
    seed: int = 20260701


@dataclass(frozen=True)
class LimitsConfig:
    ratio: float = 1.0
    sample_limit: int | None = None
    turn_limit: int | None = None
    qa_limit: int | None = None


@dataclass(frozen=True)
class RunConfig:
    resume: bool = False
    log_level: str = "INFO"


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    dataset: Path = Path("data/locomo10.json")
    paths: PathsConfig = field(default_factory=PathsConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    construction: ConstructionConfig = field(default_factory=ConstructionConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    limits: LimitsConfig = field(default_factory=LimitsConfig)
    run: RunConfig = field(default_factory=RunConfig)
    config_source: Path | None = None


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    if isinstance(value, list):
        return tuple(str(part).strip() for part in value if str(part).strip())
    raise ValueError("keyword_conditions must be a list or comma-separated string")


def _load_retrieval_pipeline(raw: Mapping[str, Any]) -> RetrievalPipelineConfig:
    pipeline_raw = _mapping(raw.get("retrieval_pipeline"), "evaluation.retrieval_pipeline")
    final_k = int(pipeline_raw.get("final_k", raw.get("retrieve_k", 10)))
    stages_raw = pipeline_raw.get("stages")
    if stages_raw is None:
        stages = (
            RetrievalStageConfig(
                type="embedding",
                name="embedding_candidates",
                top_k=final_k,
                query="similarity_query",
            ),
        )
    else:
        if not isinstance(stages_raw, list):
            raise ValueError("evaluation.retrieval_pipeline.stages must be a list")
        stages = tuple(
            RetrievalStageConfig(
                type=str(_mapping(stage, f"evaluation.retrieval_pipeline.stages[{idx}]").get("type", "")),
                name=str(
                    _mapping(stage, f"evaluation.retrieval_pipeline.stages[{idx}]").get(
                        "name",
                        _mapping(stage, f"evaluation.retrieval_pipeline.stages[{idx}]").get("type", ""),
                    )
                ),
                top_k=int(_mapping(stage, f"evaluation.retrieval_pipeline.stages[{idx}]").get("top_k", final_k)),
                query=str(
                    _mapping(stage, f"evaluation.retrieval_pipeline.stages[{idx}]").get(
                        "query",
                        "original_question"
                        if _mapping(stage, f"evaluation.retrieval_pipeline.stages[{idx}]").get("type")
                        in {"bm25_rerank", "cross_encoder"}
                        else "similarity_query",
                    )
                ),
                model=(
                    None
                    if _mapping(stage, f"evaluation.retrieval_pipeline.stages[{idx}]").get("model") is None
                    else str(_mapping(stage, f"evaluation.retrieval_pipeline.stages[{idx}]").get("model"))
                ),
                batch_size=int(
                    _mapping(stage, f"evaluation.retrieval_pipeline.stages[{idx}]").get("batch_size", 32)
                ),
            )
            for idx, stage in enumerate(stages_raw)
        )
    return RetrievalPipelineConfig(final_k=final_k, stages=stages)


def load_experiment_config(path: Path | str) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("Experiment config must be a YAML mapping")
    if not raw.get("experiment_id"):
        raise ValueError("experiment_id is required")

    paths_raw = _mapping(raw.get("paths"), "paths")
    backend_raw = _mapping(raw.get("backend"), "backend")
    construction_raw = _mapping(raw.get("construction"), "construction")
    evaluation_raw = _mapping(raw.get("evaluation"), "evaluation")
    if "retrieval_mode" in evaluation_raw:
        raise ValueError("evaluation.retrieval_mode is no longer supported; use evaluation.retrieval_pipeline")
    if "rerank" in evaluation_raw or "rerank_mode" in evaluation_raw:
        raise ValueError("evaluation.rerank is no longer supported; use evaluation.retrieval_pipeline stages")
    limits_raw = _mapping(raw.get("limits"), "limits")
    run_raw = _mapping(raw.get("run"), "run")

    config = ExperimentConfig(
        experiment_id=str(raw["experiment_id"]),
        dataset=Path(raw.get("dataset", "data/locomo10.json")),
        paths=PathsConfig(
            cache_root=Path(paths_raw.get("cache_root", DEFAULT_CACHE_ROOT)),
            results_root=Path(paths_raw.get("results_root", DEFAULT_RESULTS_ROOT)),
            log_root=Path(paths_raw.get("log_root", DEFAULT_LOG_ROOT)),
        ),
        backend=BackendConfig(
            name=str(backend_raw.get("name", "ollama")),
            model=str(backend_raw.get("model", "llama3.2:1b")),
            sglang_host=str(backend_raw.get("sglang_host", "http://localhost")),
            sglang_port=int(backend_raw.get("sglang_port", 30000)),
        ),
        construction=ConstructionConfig(
            runs=int(construction_raw.get("runs", 1)),
            keyword_pruning_mode=str(construction_raw.get("keyword_pruning_mode", "nltk")),
            embedding_model=str(construction_raw.get("embedding_model", "all-MiniLM-L6-v2")),
            max_workers=int(construction_raw.get("max_workers", 1)),
        ),
        evaluation=EvaluationConfig(
            qa_mode=str(evaluation_raw.get("qa_mode", "content_keywords")),
            qa_runs=int(evaluation_raw.get("qa_runs", 1)),
            cache_experiment_id=(
                None
                if evaluation_raw.get("cache_experiment_id") is None
                else str(evaluation_raw.get("cache_experiment_id"))
            ),
            keyword_conditions=_tuple(evaluation_raw.get("keyword_conditions", ["none", "nltk"])),
            retrieve_k=int(evaluation_raw.get("retrieve_k", 10)),
            retrieval_pipeline=_load_retrieval_pipeline(evaluation_raw),
            temperature_c5=float(evaluation_raw.get("temperature_c5", 0.5)),
            max_keywords=int(evaluation_raw.get("max_keywords", 5)),
            seed=int(evaluation_raw.get("seed", 20260701)),
        ),
        limits=LimitsConfig(
            ratio=float(limits_raw.get("ratio", 1.0)),
            sample_limit=limits_raw.get("sample_limit"),
            turn_limit=limits_raw.get("turn_limit"),
            qa_limit=limits_raw.get("qa_limit"),
        ),
        run=RunConfig(
            resume=bool(run_raw.get("resume", False)),
            log_level=str(run_raw.get("log_level", "INFO")),
        ),
        config_source=config_path,
    )
    validate_experiment_config(config)
    return config


def validate_experiment_config(config: ExperimentConfig) -> None:
    if config.backend.name not in {"openai", "ollama", "sglang", "vllm"}:
        raise ValueError("backend.name must be one of openai, ollama, sglang, vllm")
    if config.construction.keyword_pruning_mode not in {"none", "simple", "nltk"}:
        raise ValueError("construction.keyword_pruning_mode must be none, simple, or nltk")
    if config.evaluation.qa_mode not in {"content_keywords", "robust", "both"}:
        raise ValueError("evaluation.qa_mode must be content_keywords, robust, or both")
    if set(config.evaluation.keyword_conditions) - {"none", "nltk"}:
        raise ValueError("evaluation.keyword_conditions supports only none and nltk")
    if config.construction.runs < 0:
        raise ValueError("construction.runs must be >= 0")
    if config.evaluation.qa_runs < 0:
        raise ValueError("evaluation.qa_runs must be >= 0")
    if config.evaluation.retrieve_k < 1:
        raise ValueError("evaluation.retrieve_k must be >= 1")
    pipeline = config.evaluation.retrieval_pipeline
    if pipeline.final_k < 1:
        raise ValueError("evaluation.retrieval_pipeline.final_k must be >= 1")
    if not pipeline.stages:
        raise ValueError("evaluation.retrieval_pipeline.stages must not be empty")
    if pipeline.stages[0].type not in {"embedding", "bm25"}:
        raise ValueError("first retrieval pipeline stage must be embedding or bm25")
    for stage in pipeline.stages:
        if stage.type not in {"embedding", "bm25", "embedding_rerank", "bm25_rerank", "cross_encoder", "limit"}:
            raise ValueError(f"unknown retrieval pipeline stage type: {stage.type}")
        if stage.query not in {"similarity_query", "original_question"}:
            raise ValueError(f"unknown retrieval pipeline query selector: {stage.query}")
        if stage.top_k < 1:
            raise ValueError("evaluation.retrieval_pipeline stage top_k must be >= 1")
        if stage.batch_size < 1:
            raise ValueError("evaluation.retrieval_pipeline cross_encoder batch_size must be >= 1")
    if config.construction.max_workers < 1:
        raise ValueError("construction.max_workers must be >= 1")
    if not (0.0 < config.limits.ratio <= 1.0):
        raise ValueError("limits.ratio must be between 0.0 and 1.0")


def build_args_from_config(config: ExperimentConfig) -> argparse.Namespace:
    return argparse.Namespace(
        experiment_id=config.experiment_id,
        dataset=config.dataset,
        cache_root=config.paths.cache_root,
        backend=config.backend.name,
        model=config.backend.model,
        construction_runs=config.construction.runs,
        keyword_pruning_mode=config.construction.keyword_pruning_mode,
        ratio=config.limits.ratio,
        sample_limit=config.limits.sample_limit,
        turn_limit=config.limits.turn_limit,
        max_workers=config.construction.max_workers,
        embedding_model=config.construction.embedding_model,
        sglang_host=config.backend.sglang_host,
        sglang_port=config.backend.sglang_port,
        resume=config.run.resume,
        log_level=config.run.log_level,
        config_source=str(config.config_source) if config.config_source else None,
        experiment_config=config,
    )


def evaluate_args_from_config(config: ExperimentConfig) -> argparse.Namespace:
    return argparse.Namespace(
        experiment_id=config.experiment_id,
        cache_experiment_id=config.evaluation.cache_experiment_id,
        dataset=config.dataset,
        cache_root=config.paths.cache_root,
        results_root=config.paths.results_root,
        backend=config.backend.name,
        model=config.backend.model,
        qa_mode=config.evaluation.qa_mode,
        keyword_conditions=config.evaluation.keyword_conditions,
        qa_runs=config.evaluation.qa_runs,
        construction_runs=config.construction.runs,
        retrieve_k=config.evaluation.retrieval_pipeline.final_k,
        retrieval_pipeline=config.evaluation.retrieval_pipeline,
        temperature_c5=config.evaluation.temperature_c5,
        max_keywords=config.evaluation.max_keywords,
        ratio=config.limits.ratio,
        sample_limit=config.limits.sample_limit,
        qa_limit=config.limits.qa_limit,
        max_workers=config.construction.max_workers,
        seed=config.evaluation.seed,
        embedding_model=config.construction.embedding_model,
        retrieval_mode="pipeline",
        rerank_mode="pipeline",
        rerank_model=None,
        rerank_top_n=None,
        rerank_batch_size=None,
        sglang_host=config.backend.sglang_host,
        sglang_port=config.backend.sglang_port,
        resume=config.run.resume,
        log_level=config.run.log_level,
        config_source=str(config.config_source) if config.config_source else None,
        experiment_config=config,
    )
