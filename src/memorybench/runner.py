from __future__ import annotations

import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .artifacts import (
    artifact_key,
    atomic_json,
    atomic_jsonl,
    read_memory_store,
    write_memory_store,
)
from .components import answer, metrics
from .config import MemoryBenchConfig, SelectionConfig
from .methods.amem.qa import build_amem_qa_prompt
from .methods.amem.retrieval import retrieve_amem
from .providers import complete
from .registry import component_catalog
from .schemas import DatasetBundle, DatasetSample, MemoryStore, QAResult, Question, UsageRecord


@dataclass(frozen=True)
class RunOutcome:
    exit_code: int
    status: str
    artifact_dir: Path


class ExperimentRunner:
    def __init__(self, config: MemoryBenchConfig) -> None:
        self.config = config
        self.root = config.runtime.artifact_root / config.experiment.id
        self.execution_fingerprint = self._execution_fingerprint()

    def run(self) -> RunOutcome:
        bundle = self._load_dataset()
        stores_by_run: dict[int, list[MemoryStore]] = {}
        if self.config.pipeline.stages == ("retrieve_qa",):
            stores_by_run = self._load_external_stores(bundle)

        self._write_manifest("running", bundle)
        partial = False
        fatal = False
        if "construction" in self.config.pipeline.stages:
            stores_by_run, construction_partial, construction_fatal = self._construct(bundle)
            partial |= construction_partial
            fatal |= construction_fatal
        if "retrieve_qa" in self.config.pipeline.stages:
            qa_partial, qa_fatal = self._evaluate(bundle, stores_by_run)
            partial |= qa_partial
            fatal |= qa_fatal
        status = "failed" if fatal else "partial" if partial else "completed"
        self._write_manifest(status, bundle)
        return RunOutcome(1 if fatal else 2 if partial else 0, status, self.root)

    def _load_dataset(self) -> DatasetBundle:
        dataset = self.config.pipeline.dataset
        if not dataset.path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset.path}")
        adapter = component_catalog()["dataset"].get(dataset.adapter)()
        return adapter.load(dataset.path)

    def _construct(
        self,
        bundle: DatasetBundle,
    ) -> tuple[dict[int, list[MemoryStore]], bool, bool]:
        config = self.config.pipeline.construction
        samples = self._select_samples(bundle.samples, config.selection)
        stores_by_run: dict[int, list[MemoryStore]] = {}
        any_partial = False
        any_fatal = False
        for run_index in range(config.runs):
            run_dir = self.root / "construction" / f"run_{run_index:03d}"
            stores: list[MemoryStore] = []
            failures: list[dict[str, Any]] = []

            def build(sample: DatasetSample) -> MemoryStore:
                sample_dir = run_dir / "samples" / artifact_key(sample.sample_id)
                if self._completed_unit(sample_dir / "status.json") and (sample_dir / "store.json").exists():
                    return read_memory_store(sample_dir)
                selected = sample
                if config.selection.turn_limit:
                    selected = sample.model_copy(update={"turns": sample.turns[: config.selection.turn_limit]})
                store = self._build_sample(selected, config)
                write_memory_store(sample_dir, store)
                atomic_json(sample_dir / "status.json", {
                    "status": "completed",
                    "fingerprint": self.config.fingerprint,
                    "execution_fingerprint": self.execution_fingerprint,
                    "sample_id": sample.sample_id,
                })
                error_path = sample_dir / "error.json"
                if error_path.exists():
                    error_path.unlink()
                return store

            for sample, value, error in self._run_sample_tasks(samples, build):
                if error is None:
                    stores.append(value)
                    continue
                failure = self._error_payload(error, sample_id=sample.sample_id, stage="construction")
                failures.append(failure)
                any_fatal |= self.config.runtime.on_error == "stop"
                sample_dir = run_dir / "samples" / artifact_key(sample.sample_id)
                atomic_json(sample_dir / "error.json", failure)
                atomic_json(sample_dir / "status.json", {
                    "status": "failed",
                    "fingerprint": self.config.fingerprint,
                    "execution_fingerprint": self.execution_fingerprint,
                    "sample_id": sample.sample_id,
                })

            stores.sort(key=lambda item: item.sample_id)
            partial = bool(failures)
            any_partial |= partial
            atomic_jsonl(run_dir / "errors.jsonl", failures)
            atomic_jsonl(
                run_dir / "usage.jsonl",
                (
                    usage
                    for store in stores
                    for usage in store.metadata.get("usage", ())
                ),
            )
            atomic_json(run_dir / "status.json", {
                "status": "partial" if partial else "completed",
                "fingerprint": self.config.fingerprint,
                "execution_fingerprint": self.execution_fingerprint,
                "selected_samples": len(samples),
                "completed_samples": len(stores),
                "failed_samples": len(failures),
            })
            stores_by_run[run_index] = stores
        return stores_by_run, any_partial, any_fatal

    def _build_sample(self, sample: DatasetSample, config: Any) -> MemoryStore:
        if config.adapter == "turn_rag":
            if not config.chunker:
                raise ValueError("turn_rag requires the turn chunker")
            chunker = component_catalog()["chunker"].get(config.chunker.adapter)()
            builder = component_catalog()["construction"].get(config.adapter)(chunker)
            return builder.build_sample(sample)
        if config.adapter == "amem":
            builder = component_catalog()["construction"].get(config.adapter)(config)
            return builder.build_sample(sample)
        return component_catalog()["construction"].get(config.adapter)(config).build_sample(sample)

    def _evaluate(
        self,
        bundle: DatasetBundle,
        stores_by_run: dict[int, list[MemoryStore]],
    ) -> tuple[bool, bool]:
        config = self.config.pipeline.retrieve_qa
        selected_samples = {
            sample.sample_id: sample for sample in self._select_samples(bundle.samples, config.selection)
        }
        any_partial = False
        any_fatal = False
        for construction_run, stores in sorted(stores_by_run.items()):
            for qa_run in range(config.runs):
                run_dir = (
                    self.root / "retrieve_qa" / f"construction_{construction_run:03d}"
                    / f"run_{qa_run:03d}"
                )

                def evaluate_store(store: MemoryStore) -> list[QAResult]:
                    sample = selected_samples.get(store.sample_id)
                    if sample is None:
                        return []
                    results = []
                    for question in self._select_questions(sample, config.selection):
                        question_path = run_dir / "questions" / f"{artifact_key(question.question_id)}.json"
                        prior = self._read_completed_question(question_path)
                        if prior is not None:
                            results.append(prior)
                            continue
                        result = self._evaluate_question(
                            question, sample, store, construction_run, qa_run,
                        )
                        atomic_json(question_path, result.model_dump(mode="json"))
                        results.append(result)
                    return results

                all_results: list[QAResult] = []
                task_failures: list[dict[str, Any]] = []
                available_samples = {store.sample_id for store in stores}
                for sample in selected_samples.values():
                    if sample.sample_id in available_samples:
                        continue
                    error = FileNotFoundError(
                        f"No construction store available for sample '{sample.sample_id}'"
                    )
                    for question in self._select_questions(sample, config.selection):
                        result = self._failed_question_result(
                            question,
                            sample,
                            construction_run,
                            qa_run,
                            error,
                        )
                        atomic_json(
                            run_dir / "questions" / f"{artifact_key(question.question_id)}.json",
                            result.model_dump(mode="json"),
                        )
                        all_results.append(result)
                for store, value, error in self._run_sample_tasks(stores, evaluate_store):
                    if error is None:
                        all_results.extend(value)
                        continue
                    sample = selected_samples.get(store.sample_id)
                    questions = () if sample is None else self._select_questions(sample, config.selection)
                    if sample is not None and questions:
                        for question in questions:
                            result = self._failed_question_result(
                                question,
                                sample,
                                construction_run,
                                qa_run,
                                error,
                            )
                            atomic_json(
                                run_dir / "questions" / f"{artifact_key(question.question_id)}.json",
                                result.model_dump(mode="json"),
                            )
                            all_results.append(result)
                    else:
                        task_failures.append(
                            self._error_payload(error, sample_id=store.sample_id, stage="retrieve_qa")
                        )
                    any_fatal |= self.config.runtime.on_error == "stop"

                all_results.sort(key=lambda result: (result.sample_id, result.question_id))
                rows = [result.model_dump(mode="json") for result in all_results]
                failed_results = [result for result in all_results if result.status == "failed"]
                errors = task_failures + [error for result in failed_results for error in result.errors]
                usage = [record.model_dump(mode="json") for result in all_results for record in result.usage]
                atomic_jsonl(run_dir / "results.jsonl", rows)
                atomic_jsonl(run_dir / "errors.jsonl", errors)
                atomic_jsonl(run_dir / "usage.jsonl", usage)
                partial = bool(errors)
                any_partial |= partial
                any_fatal |= self.config.runtime.on_error == "stop" and partial
                atomic_json(run_dir / "status.json", {
                    "status": "partial" if partial else "completed",
                    "fingerprint": self.config.fingerprint,
                    "execution_fingerprint": self.execution_fingerprint,
                    "questions": len(rows),
                    "completed": sum(result.status == "completed" for result in all_results),
                    "failed": len(failed_results) + len(task_failures),
                })
                self._write_summary(run_dir, all_results)
        return any_partial, any_fatal

    def _evaluate_question(
        self,
        question: Question,
        sample: DatasetSample,
        store: MemoryStore,
        construction_run: int,
        qa_run: int,
    ) -> QAResult:
        config = self.config.pipeline.retrieve_qa
        provenance = {
            "construction_run": construction_run,
            "qa_run": qa_run,
            "fingerprint": self.config.fingerprint,
            "execution_fingerprint": self.execution_fingerprint,
        }
        try:
            if config.retrieval.adapter != "staged":
                raise ValueError(f"Unsupported retrieval adapter '{config.retrieval.adapter}'")
            if config.context.adapter == "amem":
                retrieval = retrieve_amem(question, store, config.retrieval)
                context = component_catalog()["context"].get("amem")(retrieval, config.context)
            else:
                retrieval = component_catalog()["retrieval"].get(config.retrieval.adapter)(question, store, config.retrieval)
                context = component_catalog()["context"].get(config.context.adapter)(retrieval, config.context)
            usage = tuple(UsageRecord.model_validate(item) for item in retrieval.get("usage", ()))
            if config.qa.adapter == "robust":
                if not config.qa.llm:
                    raise ValueError("robust QA requires a stage-local llm config")
                prompt, temperature = build_amem_qa_prompt(
                    question,
                    context,
                    {**config.qa.params, "seed": self.config.runtime.seed},
                )
                response = complete(config.qa.llm, prompt, temperature=temperature)
                prediction = response.text.strip()
                usage += (UsageRecord(
                    phase="retrieve_qa",
                    component="qa",
                    provider=config.qa.llm.provider,
                    model=config.qa.llm.model,
                    prompt_tokens=response.prompt_tokens,
                    completion_tokens=response.completion_tokens,
                    total_tokens=response.total_tokens,
                    source=response.usage_source,
                    latency_ms=response.latency_ms,
                ),)
            else:
                prediction = answer(config.qa.adapter, question, context, config.qa.params)
            return QAResult(
                question_id=question.question_id,
                sample_id=sample.sample_id,
                status="completed",
                question=question.text,
                reference=question.reference,
                prediction=prediction,
                labels=question.labels,
                retrieval=retrieval,
                context=context,
                metrics=metrics(config.metrics, prediction, question.reference),
                usage=usage,
                provenance=provenance,
            )
        except Exception as exc:
            return self._failed_question_result(
                question,
                sample,
                construction_run,
                qa_run,
                exc,
            )

    def _read_completed_question(self, path: Path) -> QAResult | None:
        if not self.config.runtime.resume or not path.exists():
            return None
        result = QAResult.model_validate_json(path.read_text(encoding="utf-8"))
        if (
            result.status == "completed"
            and result.provenance.get("fingerprint") == self.config.fingerprint
            and result.provenance.get("execution_fingerprint") == self.execution_fingerprint
        ):
            return result
        return None

    def _completed_unit(self, status_path: Path) -> bool:
        if not self.config.runtime.resume or not status_path.exists():
            return False
        status = json.loads(status_path.read_text(encoding="utf-8"))
        return (
            status.get("status") == "completed"
            and status.get("fingerprint") == self.config.fingerprint
            and status.get("execution_fingerprint") == self.execution_fingerprint
        )

    def _load_external_stores(self, bundle: DatasetBundle) -> dict[int, list[MemoryStore]]:
        source = self.config.pipeline.retrieve_qa.memory_source
        source_root = self.config.runtime.artifact_root / source.experiment_id
        manifest_path = source_root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Memory source experiment not found: {source.experiment_id}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_hash = self._dataset_sha256()
        if manifest.get("dataset", {}).get("sha256") != expected_hash:
            raise ValueError(
                f"Memory source dataset fingerprint mismatch: {source.experiment_id}"
            )
        if manifest.get("status") != "completed":
            raise ValueError(
                f"Memory source experiment is not completed: {source.experiment_id}"
            )
        available_runs = sorted((source_root / "construction").glob("run_*"))
        if source.construction_runs != "all":
            requested = set(source.construction_runs)
            available_runs = [
                path for path in available_runs
                if int(path.name.removeprefix("run_")) in requested
            ]
        result = {}
        for run_dir in available_runs:
            run_status_path = run_dir / "status.json"
            if not run_status_path.exists():
                raise ValueError(f"Memory source construction run is not completed: {run_dir.name}")
            run_status = json.loads(run_status_path.read_text(encoding="utf-8"))
            if (
                run_status.get("status") != "completed"
                or run_status.get("fingerprint") != manifest.get("fingerprint")
            ):
                raise ValueError(f"Memory source construction run is not completed: {run_dir.name}")
            stores = []
            for path in sorted((run_dir / "samples").glob("*/store.json")):
                sample_status_path = path.parent / "status.json"
                if not sample_status_path.exists():
                    raise ValueError(
                        f"Memory source sample store is not completed: {path.parent.name}"
                    )
                sample_status = json.loads(sample_status_path.read_text(encoding="utf-8"))
                if (
                    sample_status.get("status") != "completed"
                    or sample_status.get("fingerprint") != manifest.get("fingerprint")
                ):
                    raise ValueError(
                        f"Memory source sample store is not completed: {path.parent.name}"
                    )
                stores.append(read_memory_store(path.parent))
            if stores:
                result[int(run_dir.name.removeprefix("run_"))] = stores
        if not result:
            raise FileNotFoundError(
                f"Memory source experiment has no selected construction stores: {source.experiment_id}"
            )
        return result

    def _write_manifest(self, status: str, bundle: DatasetBundle) -> None:
        atomic_json(self.root / "manifest.json", {
            "schema_version": "memorybench/v1",
            "experiment_id": self.config.experiment.id,
            "fingerprint": self.config.fingerprint,
            "execution_fingerprint": self.execution_fingerprint,
            "status": status,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": self._git_commit(),
            "dataset": {
                "adapter": self.config.pipeline.dataset.adapter,
                "path": str(self.config.pipeline.dataset.path),
                "sha256": self._dataset_sha256(),
            },
            "config": self.config.model_dump(mode="json"),
            "taxonomy": bundle.taxonomy.model_dump(mode="json"),
            "components": {"schema": "memorybench/v1"},
        })

    def _dataset_sha256(self) -> str:
        return hashlib.sha256(self.config.pipeline.dataset.path.read_bytes()).hexdigest()

    def _execution_fingerprint(self) -> str:
        revision = self._git_commit() or "no-git"
        try:
            dirty = subprocess.run(
                ["git", "diff", "--binary", "HEAD"],
                check=True,
                capture_output=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError):
            dirty = b""
        payload = (
            f"{self.config.fingerprint}:{revision}:".encode("utf-8")
            + dirty
            + self._source_snapshot()
        )
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _source_snapshot() -> bytes:
        try:
            paths = subprocess.run(
                ["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", "src/memorybench"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines()
        except (OSError, subprocess.CalledProcessError):
            return b""
        digest = hashlib.sha256()
        for source_path in sorted(paths):
            path = Path(source_path)
            if not path.is_file():
                continue
            try:
                contents = path.read_bytes()
            except OSError:
                continue
            digest.update(source_path.encode("utf-8"))
            digest.update(b"\0")
            digest.update(contents)
        return digest.digest()

    @staticmethod
    def _git_commit() -> str | None:
        try:
            return subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    def _run_sample_tasks(
        self,
        items: list[Any],
        operation: Callable[[Any], Any],
    ) -> list[tuple[Any, Any | None, Exception | None]]:
        if self.config.runtime.max_workers == 1:
            output = []
            for item in items:
                try:
                    output.append((item, operation(item), None))
                except Exception as exc:  # converted to structured unit error by caller
                    output.append((item, None, exc))
            return output
        output = []
        workers = min(self.config.runtime.max_workers, max(1, len(items)))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(operation, item): item for item in items}
            for future in as_completed(futures):
                item = futures[future]
                try:
                    output.append((item, future.result(), None))
                except Exception as exc:
                    output.append((item, None, exc))
        return output

    @staticmethod
    def _error_payload(
        error: Exception,
        *,
        sample_id: str,
        stage: str,
        question_id: str | None = None,
    ) -> dict[str, Any]:
        return {
            "stage": stage,
            "sample_id": sample_id,
            "question_id": question_id,
            "type": type(error).__name__,
            "message": str(error),
            "retryable": True,
        }

    def _failed_question_result(
        self,
        question: Question,
        sample: DatasetSample,
        construction_run: int,
        qa_run: int,
        error: Exception,
    ) -> QAResult:
        return QAResult(
            question_id=question.question_id,
            sample_id=sample.sample_id,
            status="failed",
            question=question.text,
            reference=question.reference,
            labels=question.labels,
            errors=(self._error_payload(
                error,
                sample_id=sample.sample_id,
                question_id=question.question_id,
                stage="retrieve_qa",
            ),),
            provenance={
                "construction_run": construction_run,
                "qa_run": qa_run,
                "fingerprint": self.config.fingerprint,
                "execution_fingerprint": self.execution_fingerprint,
            },
        )

    @staticmethod
    def _write_summary(run_dir: Path, results: list[QAResult]) -> None:
        completed = [result for result in results if result.status == "completed"]
        metric_names = sorted({name for result in completed for name in result.metrics})
        usage_by_source: dict[str, dict[str, float]] = {}
        for result in results:
            for record in result.usage:
                bucket = usage_by_source.setdefault(record.source, {"calls": 0, "tokens": 0, "latency_ms": 0.0})
                bucket["calls"] += 1
                bucket["tokens"] += record.total_tokens or 0
                bucket["latency_ms"] += record.latency_ms or 0.0
        atomic_json(run_dir / "summary.json", {
            "questions": len(results),
            "completed": len(completed),
            "failed": len(results) - len(completed),
            "metrics": {
                name: sum(result.metrics.get(name, 0.0) for result in completed) / len(completed)
                for name in metric_names
            } if completed else {},
            "usage": usage_by_source,
        })

    @staticmethod
    def _select_samples(
        samples: tuple[DatasetSample, ...],
        selection: SelectionConfig,
    ) -> list[DatasetSample]:
        result = [sample for sample in samples if not selection.sample_ids or sample.sample_id in selection.sample_ids]
        return result[: selection.sample_limit]

    @staticmethod
    def _select_questions(sample: DatasetSample, selection: SelectionConfig) -> list[Question]:
        result = [
            question for question in sample.questions
            if not selection.question_ids or question.question_id in selection.question_ids
        ]
        if selection.categories:
            selected = set(selection.categories)
            result = [
                question for question in result
                if selected.intersection(label for values in question.labels.values() for label in values)
            ]
        return result[: selection.question_limit]
