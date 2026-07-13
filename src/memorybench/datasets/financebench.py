from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from memorybench.schemas import (
    DatasetBundle,
    DatasetSample,
    DatasetTaxonomy,
    Question,
    TaxonomyDimension,
    Turn,
)


PREPARED_SCHEMA_VERSION = "memorybench/financebench/v1"
PREPARATION_MANIFEST_SCHEMA_VERSION = "memorybench/financebench-preparation/v1"


@dataclass(frozen=True)
class FinanceBenchScope:
    question_count: int
    document_count: int

    def __post_init__(self) -> None:
        if self.question_count < 1 or self.document_count < 1:
            raise ValueError("FinanceBench scope counts must be at least 1")

    def validate(self, document_names: Sequence[str], question_ids: Sequence[str]) -> None:
        duplicate_question_ids = sorted({
            question_id for question_id in question_ids if question_ids.count(question_id) > 1
        })
        if duplicate_question_ids:
            raise ValueError(f"duplicate FinanceBench financebench_id {duplicate_question_ids[0]}")
        if len(set(document_names)) != len(document_names):
            raise ValueError("duplicate FinanceBench document name in prepared dataset")
        if len(question_ids) != self.question_count:
            raise ValueError(
                f"FinanceBench scope expected {self.question_count} unique question IDs, got {len(question_ids)}"
            )
        if len(document_names) != self.document_count:
            raise ValueError(
                f"FinanceBench scope expected {self.document_count} document names, got {len(document_names)}"
            )


PUBLIC_FINANCEBENCH_SCOPE = FinanceBenchScope(question_count=150, document_count=84)


class FinanceBenchAdapter:
    def __init__(self, scope: FinanceBenchScope = PUBLIC_FINANCEBENCH_SCOPE) -> None:
        self.scope = scope

    def load(self, path: str | Path) -> DatasetBundle:
        prepared_path = Path(path)
        encoded = prepared_path.read_bytes()
        self._validate_manifest(prepared_path, encoded)
        try:
            payload = json.loads(encoded)
        except json.JSONDecodeError as exc:
            raise ValueError(f"FinanceBench prepared file is not valid JSON: {prepared_path}") from exc
        if not isinstance(payload, dict) or payload.get("schema_version") != PREPARED_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported FinanceBench prepared schema in {prepared_path}; "
                f"expected {PREPARED_SCHEMA_VERSION}"
            )
        documents = self._list(payload, "documents", "prepared dataset")
        samples = []
        question_types: set[str] = set()
        reasoning_types: set[str] = set()
        document_names = []
        question_ids = []
        for document in sorted(documents, key=lambda item: str(item.get("doc_name", ""))):
            doc_name = self._string(document, "doc_name", "prepared document")
            document_names.append(doc_name)
            metadata = self._mapping(document, "metadata", f"prepared document {doc_name}")
            turns = tuple(self._turn(raw, doc_name) for raw in self._list(document, "turns", doc_name))
            if not turns:
                raise ValueError(f"FinanceBench document {doc_name} has no prepared turns")
            questions = []
            for raw in self._list(document, "questions", doc_name):
                question, question_type, reasoning = self._question(raw, doc_name)
                questions.append(question)
                question_ids.append(question.question_id)
                question_types.add(question_type)
                if reasoning is not None:
                    reasoning_types.add(reasoning)
            samples.append(DatasetSample(
                sample_id=f"financebench:{doc_name}",
                turns=turns,
                questions=tuple(questions),
                metadata=dict(metadata),
            ))
        self.scope.validate(document_names, question_ids)
        dimensions = [TaxonomyDimension(
            name="question_type",
            values=tuple(sorted(question_types)),
            source="FinanceBench question_type",
        )]
        if reasoning_types:
            dimensions.append(TaxonomyDimension(
                name="question_reasoning",
                values=tuple(sorted(reasoning_types)),
                source="FinanceBench question_reasoning",
            ))
        return DatasetBundle(
            dataset_id="financebench",
            taxonomy=DatasetTaxonomy(dimensions=tuple(dimensions)),
            samples=tuple(samples),
        )

    @staticmethod
    def _validate_manifest(prepared_path: Path, encoded: bytes) -> None:
        manifest_path = prepared_path.with_name("manifest.json")
        if not manifest_path.exists():
            raise ValueError(f"FinanceBench manifest not found beside {prepared_path}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"FinanceBench manifest is not valid JSON: {manifest_path}") from exc
        if not isinstance(manifest, Mapping):
            raise ValueError(f"FinanceBench manifest must be an object: {manifest_path}")
        if manifest.get("schema_version") != PREPARATION_MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"Unsupported FinanceBench manifest schema: {manifest_path}")
        if manifest.get("status") != "completed":
            raise ValueError("FinanceBench manifest status is not completed; rerun prepare-financebench")
        actual = hashlib.sha256(encoded).hexdigest()
        if manifest.get("prepared_sha256") != actual:
            raise ValueError("FinanceBench prepared_sha256 mismatch; rerun prepare-financebench")

    @classmethod
    def _turn(cls, raw: Mapping[str, Any], doc_name: str) -> Turn:
        cls._integer_or_none(raw, "part_index", f"turn in {doc_name}")
        cls._integer(raw, "page_index", f"turn in {doc_name}")
        return Turn(
            turn_id=cls._string(raw, "turn_id", f"turn in {doc_name}"),
            evidence_id=cls._string(raw, "evidence_id", f"turn in {doc_name}"),
            speaker="document",
            text=cls._string(raw, "text", f"turn in {doc_name}"),
            session_id=doc_name,
        )

    @classmethod
    def _question(cls, raw: Mapping[str, Any], doc_name: str) -> tuple[Question, str, str | None]:
        question_type = cls._string(raw, "question_type", f"question in {doc_name}")
        reasoning = raw.get("question_reasoning")
        if reasoning is not None and (not isinstance(reasoning, str) or not reasoning):
            raise ValueError(f"question_reasoning must be a non-empty string or null in {doc_name}")
        evidence_ids = cls._string_list(raw, "evidence_ids", f"question in {doc_name}")
        labels = {"question_type": (question_type,)}
        if reasoning is not None:
            labels["question_reasoning"] = (reasoning,)
        return Question(
            question_id=cls._string(raw, "question_id", f"question in {doc_name}"),
            text=cls._string(raw, "text", f"question in {doc_name}"),
            reference=cls._string(raw, "reference", f"question in {doc_name}"),
            evidence_ids=tuple(evidence_ids),
            labels=labels,
        ), question_type, reasoning

    @staticmethod
    def _mapping(raw: Mapping[str, Any], key: str, context: str) -> Mapping[str, Any]:
        value = raw.get(key)
        if not isinstance(value, Mapping):
            raise ValueError(f"{key} must be an object in {context}")
        return value

    @staticmethod
    def _list(raw: Mapping[str, Any], key: str, context: str) -> list[Mapping[str, Any]]:
        value = raw.get(key)
        if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
            raise ValueError(f"{key} must be a list of objects in {context}")
        return value

    @staticmethod
    def _string_list(raw: Mapping[str, Any], key: str, context: str) -> list[str]:
        value = raw.get(key)
        if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
            raise ValueError(f"{key} must be a list of non-empty strings in {context}")
        return value

    @staticmethod
    def _string(raw: Mapping[str, Any], key: str, context: str) -> str:
        value = raw.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{key} must be a non-empty string in {context}")
        return value

    @staticmethod
    def _integer(raw: Mapping[str, Any], key: str, context: str) -> int:
        value = raw.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{key} must be a non-negative integer in {context}")
        return value

    @staticmethod
    def _integer_or_none(raw: Mapping[str, Any], key: str, context: str) -> int | None:
        value = raw.get(key)
        if value is None:
            return None
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{key} must be a positive integer or null in {context}")
        return value
