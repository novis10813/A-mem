from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .registry import component_catalog


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SelectionConfig(StrictModel):
    sample_ids: tuple[str, ...] | None = None
    question_ids: tuple[str, ...] | None = None
    sample_limit: int | None = Field(default=None, ge=1)
    turn_limit: int | None = Field(default=None, ge=1)
    question_limit: int | None = Field(default=None, ge=1)
    categories: tuple[str, ...] | None = None


class ComponentConfig(StrictModel):
    adapter: str
    params: dict[str, Any] = Field(default_factory=dict)


class LLMConfig(StrictModel):
    provider: Literal["fake", "openai", "ollama", "sglang", "vllm"]
    model: str
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_max_tokens(self):
        max_tokens = self.params.get("max_tokens")
        if max_tokens is not None and (
            isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1
        ):
            raise ValueError("llm.params.max_tokens must be a positive integer")
        return self


class DatasetConfig(ComponentConfig):
    path: Path


class ChunkerConfig(ComponentConfig):
    pass


class ConstructionConfig(ComponentConfig):
    runs: int = Field(default=1, ge=1)
    chunker: ChunkerConfig | None = None
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    llm: LLMConfig | None = None


class RetrievalStageConfig(ComponentConfig):
    top_k: int = Field(default=10, ge=1)
    llm: LLMConfig | None = None
    query: Literal["current", "original_question"] = "current"


class RetrievalConfig(ComponentConfig):
    stages: tuple[RetrievalStageConfig, ...] = ()


class ContextConfig(ComponentConfig):
    fields: tuple[str, ...] = ("timestamp", "content", "keywords")


class QAConfig(ComponentConfig):
    llm: LLMConfig | None = None


class MemorySourceConfig(StrictModel):
    experiment_id: str
    construction_runs: Literal["all"] | tuple[int, ...] = "all"


class RetrieveQAConfig(StrictModel):
    runs: int = Field(default=1, ge=1)
    retrieval: RetrievalConfig
    context: ContextConfig
    qa: QAConfig
    metrics: tuple[ComponentConfig, ...] = ()
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    memory_source: MemorySourceConfig | None = None


class PipelineConfig(StrictModel):
    stages: tuple[Literal["construction", "retrieve_qa"], ...]
    dataset: DatasetConfig
    construction: ConstructionConfig | None = None
    retrieve_qa: RetrieveQAConfig | None = None

    @model_validator(mode="after")
    def validate_phases(self):
        if not self.stages:
            raise ValueError("pipeline.stages requires at least one stage")
        if len(set(self.stages)) != len(self.stages):
            raise ValueError("duplicate pipeline stage")
        if "construction" in self.stages and self.construction is None:
            raise ValueError("construction stage requires pipeline.construction")
        if "retrieve_qa" in self.stages and self.retrieve_qa is None:
            raise ValueError("retrieve_qa stage requires pipeline.retrieve_qa")
        if self.stages == ("retrieve_qa",) and not self.retrieve_qa.memory_source:
            raise ValueError("retrieve-only pipeline requires memory_source")
        return self


class ExperimentConfig(StrictModel):
    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    description: str | None = None
    tags: tuple[str, ...] = ()


class RuntimeConfig(StrictModel):
    artifact_root: Path = Path("artifacts/experiments")
    max_workers: int = Field(default=1, ge=1)
    resume: bool = False
    on_error: Literal["stop", "continue"] = "stop"
    seed: int = 0


class MemoryBenchConfig(StrictModel):
    experiment: ExperimentConfig
    pipeline: PipelineConfig
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    @model_validator(mode="after")
    def validate_adapters(self):
        catalog = component_catalog()
        components = [("dataset", self.pipeline.dataset.adapter)]
        if self.pipeline.construction:
            construction = self.pipeline.construction
            components.append(("construction", construction.adapter))
            if construction.adapter == "turn_rag" and construction.chunker is None:
                raise ValueError("turn_rag requires a chunker")
            if construction.adapter == "amem" and construction.llm is None:
                raise ValueError("amem construction requires an llm")
            if construction.chunker:
                components.append(("chunker", construction.chunker.adapter))
        if self.pipeline.retrieve_qa:
            stage = self.pipeline.retrieve_qa
            if not stage.retrieval.stages:
                raise ValueError("staged retrieval requires at least one stage")
            if stage.qa.adapter == "robust" and stage.qa.llm is None:
                raise ValueError("robust QA requires an llm")
            components.extend((
                ("retrieval", stage.retrieval.adapter), ("context", stage.context.adapter),
                ("qa", stage.qa.adapter),
            ))
            components.extend(("retrieval_stage", item.adapter) for item in stage.retrieval.stages)
            components.extend(("metric", item.adapter) for item in stage.metrics)
        for family, adapter in components:
            catalog[family].get(adapter)
        return self

    @property
    def fingerprint(self) -> str:
        payload = self.model_dump(mode="json", exclude_none=True)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def load_config(path: str | Path) -> MemoryBenchConfig:
    with Path(path).open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return MemoryBenchConfig.model_validate(payload)
