from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import atomic_json, atomic_jsonl, read_jsonl
from .components import TurnChunker, TurnRAGConstruction, answer, build_context, metrics, staged_retrieve
from .config import MemoryBenchConfig, SelectionConfig
from .datasets import LoCoMoAdapter
from .schemas import DatasetBundle, DatasetSample, MemoryStore, QAResult, UsageRecord
from .providers import complete
from .methods.amem.serialization import notes_to_store


@dataclass(frozen=True)
class RunOutcome:
    exit_code: int
    status: str
    artifact_dir: Path


class ExperimentRunner:
    def __init__(self, config: MemoryBenchConfig) -> None:
        self.config = config
        self.root = config.runtime.artifact_root / config.experiment.id

    def run(self) -> RunOutcome:
        bundle = self._load_dataset()
        self._write_manifest("running", bundle)
        partial = False
        stores_by_run: dict[int, list[MemoryStore]] = {}
        if "construction" in self.config.pipeline.stages:
            stores_by_run = self._construct(bundle)
        if "retrieve_qa" in self.config.pipeline.stages:
            if not stores_by_run:
                stores_by_run = self._load_external_stores()
            partial = self._evaluate(bundle, stores_by_run)
        status = "partial" if partial else "completed"
        self._write_manifest(status, bundle)
        return RunOutcome(2 if partial else 0, status, self.root)

    def _load_dataset(self) -> DatasetBundle:
        dataset = self.config.pipeline.dataset
        if dataset.adapter != "locomo":
            raise ValueError(f"Unsupported dataset adapter '{dataset.adapter}'")
        return LoCoMoAdapter().load(dataset.path)

    def _construct(self, bundle: DatasetBundle) -> dict[int, list[MemoryStore]]:
        config = self.config.pipeline.construction
        samples = self._select_samples(bundle.samples, config.selection)
        result = {}
        for run_index in range(config.runs):
            path = self.root / "construction" / f"run_{run_index:03d}" / "stores.jsonl"
            if self._can_resume(path):
                stores = [MemoryStore.model_validate(row) for row in read_jsonl(path)]
            else:
                if config.adapter == "turn_rag":
                    if not config.chunker or config.chunker.adapter != "turn":
                        raise ValueError("turn_rag requires the turn chunker")
                    builder = TurnRAGConstruction(TurnChunker())
                elif config.adapter == "amem":
                    builder = None
                else:
                    raise ValueError(f"Unsupported construction adapter '{config.adapter}'")
                stores = []
                for sample in samples:
                    if config.selection.turn_limit:
                        sample = sample.model_copy(update={"turns": sample.turns[:config.selection.turn_limit]})
                    if builder:
                        stores.append(builder.build_sample(sample))
                    else:
                        stores.append(self._build_amem(sample, config))
                atomic_jsonl(path, (store.model_dump(mode="json") for store in sorted(stores, key=lambda x: x.sample_id)))
                atomic_json(path.with_name("status.json"), {"status": "completed", "fingerprint": self.config.fingerprint})
            result[run_index] = stores
        return result

    def _evaluate(self, bundle: DatasetBundle, stores_by_run: dict[int, list[MemoryStore]]) -> bool:
        config = self.config.pipeline.retrieve_qa
        partial = False
        samples = {sample.sample_id: sample for sample in self._select_samples(bundle.samples, config.selection)}
        for construction_run, stores in sorted(stores_by_run.items()):
            for qa_run in range(config.runs):
                path = self.root / "retrieve_qa" / f"construction_{construction_run:03d}" / f"run_{qa_run:03d}" / "results.jsonl"
                if self._can_resume(path):
                    continue
                rows = []
                for store in sorted(stores, key=lambda item: item.sample_id):
                    sample = samples.get(store.sample_id)
                    if not sample:
                        continue
                    for question in self._select_questions(sample, config.selection):
                        try:
                            retrieval = staged_retrieve(question, store, config.retrieval)
                            context = build_context(retrieval, config.context)
                            usage = ()
                            if config.qa.adapter == "robust":
                                if not config.qa.llm:
                                    raise ValueError("robust QA requires a stage-local llm config")
                                response = complete(config.qa.llm, f"Context:\n{context['text']}\n\nQuestion: {question.text}\nAnswer:")
                                prediction = response.text.strip()
                                usage = (UsageRecord(
                                    phase="retrieve_qa", component="qa", model=config.qa.llm.model,
                                    prompt_tokens=response.prompt_tokens, completion_tokens=response.completion_tokens,
                                    total_tokens=response.total_tokens, source=response.usage_source,
                                    latency_ms=response.latency_ms,
                                ),)
                            else:
                                prediction = answer(config.qa.adapter, question, context)
                            result = QAResult(
                                question_id=question.question_id, sample_id=sample.sample_id, status="completed",
                                question=question.text, reference=question.reference, prediction=prediction,
                                labels=question.labels, retrieval=retrieval, context=context,
                                metrics=metrics(config.metrics, prediction, question.reference), usage=usage,
                                provenance={"construction_run": construction_run, "qa_run": qa_run, "fingerprint": self.config.fingerprint},
                            )
                        except Exception as exc:
                            if self.config.runtime.on_error == "stop":
                                raise
                            partial = True
                            result = QAResult(
                                question_id=question.question_id, sample_id=sample.sample_id, status="failed",
                                question=question.text, reference=question.reference, labels=question.labels,
                                errors=({"type": type(exc).__name__, "message": str(exc)},),
                                provenance={"construction_run": construction_run, "qa_run": qa_run, "fingerprint": self.config.fingerprint},
                            )
                        rows.append(result.model_dump(mode="json"))
                atomic_jsonl(path, sorted(rows, key=lambda row: row["question_id"]))
                atomic_json(path.with_name("status.json"), {"status": "partial" if any(r["status"] == "failed" for r in rows) else "completed", "fingerprint": self.config.fingerprint})
                completed = [row for row in rows if row["status"] == "completed"]
                metric_names = sorted({name for row in completed for name in row.get("metrics", {})})
                atomic_json(path.with_name("summary.json"), {
                    "questions": len(rows), "completed": len(completed), "failed": len(rows) - len(completed),
                    "metrics": {
                        name: sum(row["metrics"].get(name, 0.0) for row in completed) / len(completed)
                        for name in metric_names
                    } if completed else {},
                    "usage": {"reported": {}, "estimated": {}},
                })
        return partial

    def _can_resume(self, data_path: Path) -> bool:
        if not self.config.runtime.resume or not data_path.exists():
            return False
        status_path = data_path.with_name("status.json")
        if not status_path.exists():
            return False
        import json
        status = json.loads(status_path.read_text(encoding="utf-8"))
        return status.get("status") == "completed" and status.get("fingerprint") == self.config.fingerprint

    def _write_manifest(self, status: str, bundle: DatasetBundle) -> None:
        atomic_json(self.root / "manifest.json", {
            "schema_version": "memorybench/v1", "experiment_id": self.config.experiment.id,
            "fingerprint": self.config.fingerprint, "status": status,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "config": self.config.model_dump(mode="json"), "taxonomy": bundle.taxonomy.model_dump(mode="json"),
        })

    def _load_external_stores(self) -> dict[int, list[MemoryStore]]:
        source = self.config.pipeline.retrieve_qa.memory_source
        return {0: [MemoryStore.model_validate(row) for row in read_jsonl(source)]}

    @staticmethod
    def _build_amem(sample: DatasetSample, config) -> MemoryStore:
        if not config.llm:
            raise ValueError("amem construction requires a stage-local llm config")
        if config.llm.provider == "fake":
            from .amem_native.memory_layer_robust import RobustMemoryNote
            notes = [RobustMemoryNote(
                content=turn.text, id=turn.turn_id, timestamp=turn.timestamp,
                keywords=[], context="General", category="Uncategorized", tags=[],
            ) for turn in sample.turns]
            return notes_to_store(sample.sample_id, notes)
        from .amem_native.memory_layer_robust import RobustAgenticMemorySystem
        backend = "openai" if config.llm.provider in {"openai", "sglang", "vllm"} else config.llm.provider
        system = RobustAgenticMemorySystem(
            llm_backend=backend, llm_model=config.llm.model,
            api_base=config.llm.params.get("base_url"), retrieval_mode=config.params.get("retrieval_mode", "embedding"),
        )
        for turn in sample.turns:
            system.add_note(turn.text, time=turn.timestamp)
        return notes_to_store(sample.sample_id, system.memories.values())

    @staticmethod
    def _select_samples(samples: tuple[DatasetSample, ...], selection: SelectionConfig) -> list[DatasetSample]:
        result = [s for s in samples if not selection.sample_ids or s.sample_id in selection.sample_ids]
        return result[:selection.sample_limit]

    @staticmethod
    def _select_questions(sample: DatasetSample, selection: SelectionConfig):
        result = [q for q in sample.questions if not selection.question_ids or q.question_id in selection.question_ids]
        return result[:selection.question_limit]
