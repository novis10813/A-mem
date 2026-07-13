# FinanceBench Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the public FinanceBench PDF corpus locally preparable and runnable through MemoryBench's existing native A-Mem pipeline with stable page-level evidence provenance.

**Architecture:** A dedicated preparation command downloads the resolved upstream public inputs, extracts complete PDF pages, and writes a deterministic `prepared.json` plus integrity manifest below `artifacts/`. A new local-only `FinanceBenchAdapter` turns that prepared file into per-document `DatasetSample` objects; existing construction, retrieval, QA, artifact, and runner paths remain unchanged.

**Tech Stack:** Python 3.13, `uv`, Pydantic v2, `pypdf>=6.0` as a lazy optional dependency, `urllib.request`, `concurrent.futures`, pytest, and the existing llama.cpp-compatible `vllm` provider path.

## Global Constraints

- Use `uv` and Python 3.13; run Python code with `uv run python`.
- Library and CLI code belongs in `src/memorybench/`; tests belong in `tests/`; generated data belongs below `artifacts/` and must not be committed.
- The public FinanceBench scope is exactly the 150 public questions and their 84 referenced PDFs; do not add unrelated catalog documents or cross-document retrieval.
- `memorybench run` must perform no download or PDF extraction; it reads only the local prepared dataset file.
- `pypdf` must be lazy behind a `financebench` optional extra. Importing adapters, listing components, validating configs, and non-FinanceBench runs must not import it.
- Do not add or rename a llama.cpp provider in this branch. The committed configs use the existing `vllm` compatibility route at `http://127.0.0.1:8080/v1` with model `llama3.2`.
- Use complete PDF pages as the default chunk. Only pages above 1,200 whitespace-delimited words split at blank-line boundaries; parts share their page evidence ID.
- Preserve the raw zero-based `evidence_page_num`, use the evidence object's nested `doc_name`, and omit `question_reasoning` labels when their source value is null.
- Keep A-Mem construction within a document sequential. Parallelize only independent documents, with preparation workers controlled separately from `runtime.max_workers`.
- Do not claim official FinanceBench numerical-reasoning or human-evaluation scores; exact match, F1, and BLEU-1 are diagnostic metrics in this branch.

---

## File Structure

| Path | Responsibility |
| --- | --- |
| `src/memorybench/datasets/financebench.py` | Prepared-file schema constants, manifest validation, and `FinanceBenchAdapter` conversion into MemoryBench schemas. |
| `src/memorybench/datasets/financebench_prepare.py` | Upstream resolution/download, source validation, layout-text extraction, page normalization, manifest creation, resume logic, and preparation result reporting. |
| `src/memorybench/datasets/__init__.py` | Publicly export `FinanceBenchAdapter`. |
| `src/memorybench/registry.py` | Register the `financebench` dataset adapter. |
| `src/memorybench/cli.py` | Parse and dispatch `prepare-financebench --output --workers`. |
| `pyproject.toml` and `uv.lock` | Define and lock the lazy `financebench` dependency extra. |
| `configs/financebench_llamacpp_smoke.yaml` | One-document, one-question local llama.cpp characterization run. |
| `configs/financebench_llamacpp.yaml` | Full 84-document, 150-question local llama.cpp run. |
| `tests/test_memorybench_financebench.py` | Unit, adapter, preparation, CLI, config, and fake-A-Mem integration coverage without network or `pypdf`. |
| `README.md` | Explain installation, preparation, llama.cpp compatibility config, run commands, and the non-official scoring boundary. |

## Prepared File Contract

`prepared.json` is the only dataset path passed to the runner:

```json
{
  "schema_version": "memorybench/financebench/v1",
  "source": {
    "repository": "patronus-ai/financebench",
    "revision": "0123456789abcdef0123456789abcdef01234567"
  },
  "documents": [
    {
      "doc_name": "3M_2018_10K",
      "metadata": {
        "company": "3M",
        "gics_sector": "Industrials",
        "doc_type": "10k",
        "doc_period": 2018,
        "source_url": "https://investors.3m.com/financials/sec-filings/content/0001558370-19-000470/0001558370-19-000470.pdf"
      },
      "turns": [
        {
          "turn_id": "financebench:3M_2018_10K:page:59",
          "evidence_id": "financebench:3M_2018_10K:page:59",
          "page_index": 59,
          "part_index": null,
          "text": "Document: 3M_2018_10K\\nPDF page index: 59\\n\\nPurchases of property, plant and equipment were (1,577)."
        }
      ],
      "questions": [
        {
          "question_id": "financebench_id_03029",
          "text": "What is the FY2018 capital expenditure amount?",
          "reference": "$1577.00",
          "evidence_ids": ["financebench:3M_2018_10K:page:59"],
          "question_type": "metrics-generated",
          "question_reasoning": "Information extraction"
        }
      ]
    }
  ]
}
```

The sibling `manifest.json` has schema version `memorybench/financebench-preparation/v1`, a `status` of `completed`, and a `prepared_sha256` that exactly matches `prepared.json`. The adapter rejects a missing, failed, malformed, or mismatched manifest, preventing a stale prepared file from being run after a failed re-preparation.

### Task 1: Add the Local Prepared-Dataset Adapter

**Files:**

- Create: `src/memorybench/datasets/financebench.py`
- Modify: `src/memorybench/datasets/__init__.py:1-3`
- Modify: `src/memorybench/registry.py:38-53`
- Create: `tests/test_memorybench_financebench.py`

**Interfaces:**

- Consumes: a `Path` to the `prepared.json` contract above and its sibling `manifest.json`.
- Produces: `FinanceBenchAdapter.load(path: str | Path) -> DatasetBundle` registered as `component_catalog()["dataset"].get("financebench")`.
- Defines for later tasks: `PREPARED_SCHEMA_VERSION = "memorybench/financebench/v1"` and `PREPARATION_MANIFEST_SCHEMA_VERSION = "memorybench/financebench-preparation/v1"`.

- [ ] **Step 1: Write failing adapter and registry tests**

Create `tests/test_memorybench_financebench.py` with these helpers and tests. The fixture deliberately uses a null `question_reasoning` value in one question and an explicit reasoning label in another.

```python
import hashlib
import json
from pathlib import Path

import pytest

from memorybench.registry import component_catalog


def write_prepared(tmp_path: Path, *, manifest_status: str = "completed") -> Path:
    prepared = {
        "schema_version": "memorybench/financebench/v1",
        "source": {"repository": "patronus-ai/financebench", "revision": "deadbeef"},
        "documents": [{
            "doc_name": "Acme_2023_10K",
            "metadata": {
                "company": "Acme", "gics_sector": "Industrials", "doc_type": "10k",
                "doc_period": 2023, "source_url": "https://example.test/acme.pdf",
            },
            "turns": [{
                "turn_id": "financebench:Acme_2023_10K:page:0",
                "evidence_id": "financebench:Acme_2023_10K:page:0",
                "page_index": 0,
                "part_index": None,
                "text": "Document: Acme_2023_10K\\nPDF page index: 0\\n\\nRevenue was $10.",
            }],
            "questions": [
                {
                    "question_id": "financebench_id_00001", "text": "What was revenue?",
                    "reference": "$10", "evidence_ids": ["financebench:Acme_2023_10K:page:0"],
                    "question_type": "metrics-generated", "question_reasoning": None,
                },
                {
                    "question_id": "financebench_id_00002", "text": "Which pages support it?",
                    "reference": "$10", "evidence_ids": ["financebench:Acme_2023_10K:page:0"],
                    "question_type": "domain-relevant", "question_reasoning": "Information extraction",
                },
            ],
        }],
    }
    path = tmp_path / "prepared.json"
    encoded = json.dumps(prepared, sort_keys=True).encode("utf-8")
    path.write_bytes(encoded)
    (tmp_path / "manifest.json").write_text(json.dumps({
        "schema_version": "memorybench/financebench-preparation/v1",
        "status": manifest_status,
        "prepared_sha256": hashlib.sha256(encoded).hexdigest(),
    }), encoding="utf-8")
    return path


def test_financebench_adapter_emits_document_samples_and_native_taxonomy(tmp_path: Path):
    from memorybench.datasets.financebench import FinanceBenchAdapter

    bundle = FinanceBenchAdapter().load(write_prepared(tmp_path))

    assert bundle.dataset_id == "financebench"
    assert bundle.samples[0].sample_id == "financebench:Acme_2023_10K"
    assert bundle.samples[0].turns[0].evidence_id == "financebench:Acme_2023_10K:page:0"
    assert bundle.samples[0].questions[0].labels == {"question_type": ("metrics-generated",)}
    assert bundle.samples[0].questions[1].labels == {
        "question_type": ("domain-relevant",),
        "question_reasoning": ("Information extraction",),
    }
    assert [dimension.name for dimension in bundle.taxonomy.dimensions] == [
        "question_type", "question_reasoning",
    ]
    assert component_catalog()["dataset"].get("financebench") is FinanceBenchAdapter


def test_financebench_adapter_rejects_unprepared_or_tampered_input(tmp_path: Path):
    from memorybench.datasets.financebench import FinanceBenchAdapter

    path = write_prepared(tmp_path, manifest_status="failed")
    with pytest.raises(ValueError, match="manifest status is not completed"):
        FinanceBenchAdapter().load(path)

    path = write_prepared(tmp_path)
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="prepared_sha256 mismatch"):
        FinanceBenchAdapter().load(path)
```

- [ ] **Step 2: Run the tests to verify they fail before implementation**

Run:

```bash
uv run python -m pytest tests/test_memorybench_financebench.py -v
```

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'memorybench.datasets.financebench'`.

- [ ] **Step 3: Implement the adapter and registration**

Create `src/memorybench/datasets/financebench.py`. Keep all PDF-specific imports out of this module.

```python
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

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


class FinanceBenchAdapter:
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
        for document in sorted(documents, key=lambda item: str(item.get("doc_name", ""))):
            doc_name = self._string(document, "doc_name", "prepared document")
            metadata = self._mapping(document, "metadata", f"prepared document {doc_name}")
            turns = tuple(self._turn(raw, doc_name) for raw in self._list(document, "turns", doc_name))
            if not turns:
                raise ValueError(f"FinanceBench document {doc_name} has no prepared turns")
            questions = []
            for raw in self._list(document, "questions", doc_name):
                question, question_type, reasoning = self._question(raw, doc_name)
                questions.append(question)
                question_types.add(question_type)
                if reasoning is not None:
                    reasoning_types.add(reasoning)
            samples.append(DatasetSample(
                sample_id=f"financebench:{doc_name}",
                turns=turns,
                questions=tuple(questions),
                metadata=dict(metadata),
            ))
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
```

Update `src/memorybench/datasets/__init__.py` to:

```python
from .financebench import FinanceBenchAdapter
from .locomo import LoCoMoAdapter

__all__ = ["FinanceBenchAdapter", "LoCoMoAdapter"]
```

Update `src/memorybench/registry.py` imports and dataset registration:

```python
    from .datasets.financebench import FinanceBenchAdapter
    from .datasets.locomo import LoCoMoAdapter
```

```python
    catalog["dataset"].register("financebench", FinanceBenchAdapter)
    catalog["dataset"].register("locomo", LoCoMoAdapter)
```

- [ ] **Step 4: Run the focused tests and existing foundation coverage**

Run:

```bash
uv run python -m pytest tests/test_memorybench_financebench.py tests/test_memorybench_foundation.py -v
```

Expected: PASS. The `financebench` adapter appears in the registry, a null source reasoning value produces no reasoning label, and a failed or tampered manifest is rejected.

- [ ] **Step 5: Commit the adapter milestone**

```bash
git add src/memorybench/datasets/financebench.py src/memorybench/datasets/__init__.py src/memorybench/registry.py tests/test_memorybench_financebench.py
git commit -m "feat: add FinanceBench prepared dataset adapter"
```

### Task 2: Add Pure Source Validation and Page-First Normalization

**Files:**

- Modify: `pyproject.toml:18-23`
- Modify: `uv.lock`
- Create: `src/memorybench/datasets/financebench_prepare.py`
- Modify: `tests/test_memorybench_financebench.py`

**Interfaces:**

- Consumes: public FinanceBench question rows, document metadata rows, and `dict[str, list[str]]` of extracted page texts.
- Produces: `build_prepared_document(doc_name, metadata, questions, pages, max_page_words=1200) -> tuple[dict[str, Any], dict[str, Any]]`.
- Defines for Task 3: `MAX_PAGE_WORDS = 1200`, `extract_layout_pages(pdf_path: Path) -> list[str]`, `source_questions_by_document(rows) -> dict[str, list[dict[str, Any]]]`, and `source_metadata_by_document(rows) -> dict[str, dict[str, Any]]`.

- [ ] **Step 1: Add failing pure-normalization tests**

Append these tests to `tests/test_memorybench_financebench.py`:

```python
def raw_question(question_id: str, page_index: int, *, reasoning: str | None = "Information extraction"):
    return {
        "financebench_id": question_id,
        "doc_name": "Acme_2023_10K",
        "question": "What was revenue?",
        "answer": "$10",
        "question_type": "metrics-generated",
        "question_reasoning": reasoning,
        "evidence": [{
            "doc_name": "Acme_2023_10K",
            "evidence_page_num": page_index,
            "evidence_text": "Revenue was $10.",
            "evidence_text_full_page": "Revenue was $10.",
        }],
    }


def raw_metadata():
    return {
        "doc_name": "Acme_2023_10K",
        "company": "Acme",
        "gics_sector": "Industrials",
        "doc_type": "10k",
        "doc_period": 2023,
        "doc_link": "https://example.test/acme.pdf",
    }


def test_build_prepared_document_uses_page_fallback_and_page_evidence_ids():
    from memorybench.datasets.financebench_prepare import build_prepared_document

    document, report = build_prepared_document(
        "Acme_2023_10K", raw_metadata(), [raw_question("financebench_id_00001", 1)],
        ["Cover page", ""], max_page_words=1200,
    )

    assert [turn["evidence_id"] for turn in document["turns"]] == [
        "financebench:Acme_2023_10K:page:0",
        "financebench:Acme_2023_10K:page:1",
    ]
    assert document["turns"][1]["text"].endswith("Revenue was $10.")
    assert report["evidence_fallback_pages"] == [1]
    assert document["questions"][0]["evidence_ids"] == ["financebench:Acme_2023_10K:page:1"]


def test_page_split_preserves_one_page_evidence_id_and_blank_line_boundaries():
    from memorybench.datasets.financebench_prepare import build_prepared_document

    document, _ = build_prepared_document(
        "Acme_2023_10K", raw_metadata(), [raw_question("financebench_id_00002", 0)],
        ["one two three\\n\\nfour five six\\n\\nseven eight nine"], max_page_words=4,
    )

    assert [turn["turn_id"] for turn in document["turns"]] == [
        "financebench:Acme_2023_10K:page:0:part:1",
        "financebench:Acme_2023_10K:page:0:part:2",
        "financebench:Acme_2023_10K:page:0:part:3",
    ]
    assert {turn["evidence_id"] for turn in document["turns"]} == {
        "financebench:Acme_2023_10K:page:0"
    }


def test_source_validation_rejects_cross_document_or_unrecoverable_evidence():
    from memorybench.datasets.financebench_prepare import build_prepared_document

    cross_document = raw_question("financebench_id_00003", 0)
    cross_document["evidence"][0]["doc_name"] = "Other_2023_10K"
    with pytest.raises(ValueError, match="evidence doc_name"):
        build_prepared_document("Acme_2023_10K", raw_metadata(), [cross_document], ["page"], max_page_words=1200)

    empty_evidence = raw_question("financebench_id_00004", 0)
    empty_evidence["evidence"][0]["evidence_text_full_page"] = ""
    with pytest.raises(ValueError, match="required evidence page 0 has no text"):
        build_prepared_document("Acme_2023_10K", raw_metadata(), [empty_evidence], [""], max_page_words=1200)
```

- [ ] **Step 2: Run the normalization tests to verify they fail**

Run:

```bash
uv run python -m pytest tests/test_memorybench_financebench.py -k 'prepared_document or page_split or source_validation' -v
```

Expected: FAIL during collection because `memorybench.datasets.financebench_prepare` does not exist.

- [ ] **Step 3: Define the optional dependency and implement the pure normalization API**

Add this line under `[project.optional-dependencies]` in `pyproject.toml`:

```toml
financebench = ["pypdf>=6.0"]
```

Run `uv lock` immediately after the metadata change so `uv.lock` contains the resolved `pypdf` package.

Create `src/memorybench/datasets/financebench_prepare.py` with these imports, constants, validation helpers, and pure normalization functions. The code below defines the complete data contract used by Tasks 3 and 4.

```python
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


MAX_PAGE_WORDS = 1200


def source_questions_by_document(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in rows:
        doc_name = required_string(raw, "doc_name", "FinanceBench question")
        required_string(raw, "financebench_id", f"FinanceBench question in {doc_name}")
        required_string(raw, "question", f"FinanceBench question in {doc_name}")
        required_string(raw, "answer", f"FinanceBench question in {doc_name}")
        required_string(raw, "question_type", f"FinanceBench question in {doc_name}")
        reasoning = raw.get("question_reasoning")
        if reasoning is not None and (not isinstance(reasoning, str) or not reasoning):
            raise ValueError(f"question_reasoning must be a non-empty string or null in {doc_name}")
        evidence = required_list(raw, "evidence", f"FinanceBench question in {doc_name}")
        if not evidence:
            raise ValueError(f"FinanceBench question {raw['financebench_id']} has no evidence")
        grouped[doc_name].append(dict(raw))
    if not grouped:
        raise ValueError("FinanceBench annotations contain no questions")
    return {doc_name: sorted(items, key=lambda item: item["financebench_id"]) for doc_name, items in sorted(grouped.items())}


def source_metadata_by_document(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for raw in rows:
        doc_name = required_string(raw, "doc_name", "FinanceBench document metadata")
        if doc_name in result:
            raise ValueError(f"duplicate FinanceBench metadata doc_name {doc_name}")
        for key in ("company", "gics_sector", "doc_type", "doc_link"):
            required_string(raw, key, f"FinanceBench metadata {doc_name}")
        required_integer(raw, "doc_period", f"FinanceBench metadata {doc_name}")
        result[doc_name] = dict(raw)
    return result


def build_prepared_document(
    doc_name: str,
    metadata: Mapping[str, Any],
    questions: Sequence[Mapping[str, Any]],
    pages: Sequence[str],
    *,
    max_page_words: int = MAX_PAGE_WORDS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if max_page_words < 1:
        raise ValueError("max_page_words must be at least 1")
    fallbacks = evidence_fallbacks(doc_name, questions)
    turns = []
    empty_pages = []
    fallback_pages = []
    for page_index, extracted in enumerate(pages):
        if not isinstance(extracted, str):
            raise ValueError(f"extracted page {page_index} for {doc_name} is not text")
        text = extracted.strip()
        if not text and page_index in fallbacks:
            text = fallbacks[page_index].strip()
            if text:
                fallback_pages.append(page_index)
        if not text:
            empty_pages.append(page_index)
            continue
        page_id = f"financebench:{doc_name}:page:{page_index}"
        parts = split_layout_text(text, max_page_words)
        for part_number, part in enumerate(parts, start=1):
            turn_id = page_id if len(parts) == 1 else f"{page_id}:part:{part_number}"
            turns.append({
                "turn_id": turn_id,
                "evidence_id": page_id,
                "page_index": page_index,
                "part_index": None if len(parts) == 1 else part_number,
                "text": f"Document: {doc_name}\\nPDF page index: {page_index}\\n\\n{part}",
            })
    present_evidence = {turn["evidence_id"] for turn in turns}
    for page_index in sorted(fallbacks):
        page_id = f"financebench:{doc_name}:page:{page_index}"
        if page_id not in present_evidence:
            raise ValueError(f"required evidence page {page_index} has no text in {doc_name}")
    normalized_questions = []
    for raw in sorted(questions, key=lambda item: required_string(item, "financebench_id", doc_name)):
        reasoning = raw.get("question_reasoning")
        if reasoning is not None and (not isinstance(reasoning, str) or not reasoning):
            raise ValueError(f"question_reasoning must be a non-empty string or null in {doc_name}")
        evidence_ids = [f"financebench:{doc_name}:page:{index}" for index in evidence_page_indexes(doc_name, raw)]
        normalized_questions.append({
            "question_id": required_string(raw, "financebench_id", doc_name),
            "text": required_string(raw, "question", doc_name),
            "reference": required_string(raw, "answer", doc_name),
            "evidence_ids": evidence_ids,
            "question_type": required_string(raw, "question_type", doc_name),
            "question_reasoning": reasoning,
        })
    return {
        "doc_name": doc_name,
        "metadata": {
            "company": required_string(metadata, "company", doc_name),
            "gics_sector": required_string(metadata, "gics_sector", doc_name),
            "doc_type": required_string(metadata, "doc_type", doc_name),
            "doc_period": required_integer(metadata, "doc_period", doc_name),
            "source_url": required_string(metadata, "doc_link", doc_name),
        },
        "turns": turns,
        "questions": normalized_questions,
    }, {
        "doc_name": doc_name,
        "page_count": len(pages),
        "turn_count": len(turns),
        "empty_pages": empty_pages,
        "evidence_fallback_pages": fallback_pages,
    }


def evidence_fallbacks(doc_name: str, questions: Sequence[Mapping[str, Any]]) -> dict[int, str]:
    fallbacks: dict[int, str] = {}
    for raw in questions:
        for evidence in required_list(raw, "evidence", f"question in {doc_name}"):
            evidence_doc = required_string(evidence, "doc_name", f"evidence in {doc_name}")
            if evidence_doc != doc_name:
                raise ValueError(f"evidence doc_name {evidence_doc} does not match question doc_name {doc_name}")
            page_index = required_integer(evidence, "evidence_page_num", f"evidence in {doc_name}")
            fallback = evidence.get("evidence_text_full_page")
            if not isinstance(fallback, str):
                raise ValueError(f"evidence_text_full_page must be a string in evidence in {doc_name}")
            existing = fallbacks.setdefault(page_index, fallback)
            if existing != fallback:
                raise ValueError(f"conflicting full-page evidence text for {doc_name} page {page_index}")
    return fallbacks


def evidence_page_indexes(doc_name: str, question: Mapping[str, Any]) -> list[int]:
    indexes = []
    for evidence in required_list(question, "evidence", f"question in {doc_name}"):
        evidence_doc = required_string(evidence, "doc_name", f"evidence in {doc_name}")
        if evidence_doc != doc_name:
            raise ValueError(f"evidence doc_name {evidence_doc} does not match question doc_name {doc_name}")
        indexes.append(required_integer(evidence, "evidence_page_num", f"evidence in {doc_name}"))
    return sorted(set(indexes))


def split_layout_text(text: str, max_page_words: int) -> list[str]:
    stripped = text.strip()
    if len(stripped.split()) <= max_page_words:
        return [stripped]
    blocks = [block.strip() for block in re.split(r"\\n[\\t ]*\\n+", stripped) if block.strip()]
    parts: list[str] = []
    current: list[str] = []
    current_words = 0
    for block in blocks:
        block_words = len(block.split())
        if current and current_words + block_words > max_page_words:
            parts.append("\\n\\n".join(current))
            current, current_words = [], 0
        current.append(block)
        current_words += block_words
    if current:
        parts.append("\\n\\n".join(current))
    return parts or [stripped]


def extract_layout_pages(pdf_path: Path) -> list[str]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "FinanceBench preparation requires pypdf. Run: uv sync --extra financebench"
        ) from exc
    reader = PdfReader(pdf_path, strict=False)
    return [page.extract_text(extraction_mode="layout") or "" for page in reader.pages]


def pypdf_version() -> str:
    try:
        import pypdf
    except ImportError as exc:
        raise RuntimeError(
            "FinanceBench preparation requires pypdf. Run: uv sync --extra financebench"
        ) from exc
    return str(pypdf.__version__)


def required_string(raw: Mapping[str, Any], key: str, context: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string in {context}")
    return value


def required_integer(raw: Mapping[str, Any], key: str, context: str) -> int:
    value = raw.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{key} must be a non-negative integer in {context}")
    return value


def required_list(raw: Mapping[str, Any], key: str, context: str) -> list[Mapping[str, Any]]:
    value = raw.get(key)
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
        raise ValueError(f"{key} must be a list of objects in {context}")
    return value
```

- [ ] **Step 4: Run the normalization tests with no FinanceBench extra installed**

Run:

```bash
uv run python -m pytest tests/test_memorybench_financebench.py -k 'prepared_document or page_split or source_validation' -v
```

Expected: PASS. The tests call only pure functions and must not import `pypdf`.

- [ ] **Step 5: Commit the normalization milestone**

```bash
git add pyproject.toml uv.lock src/memorybench/datasets/financebench_prepare.py tests/test_memorybench_financebench.py
git commit -m "feat: normalize FinanceBench PDF pages"
```

### Task 3: Implement Reproducible Preparation, Resume, and CLI Dispatch

**Files:**

- Modify: `src/memorybench/datasets/financebench_prepare.py`
- Modify: `src/memorybench/cli.py:15-46`
- Modify: `tests/test_memorybench_financebench.py`

**Interfaces:**

- Consumes: `prepare_financebench(output: Path, workers: int = 4, *, fetch: Callable[[str], bytes] = fetch_url, extractor: Callable[[Path], list[str]] | None = None)`.
- Produces: `PreparationResult(output: Path, manifest_path: Path, revision: str, document_count: int)` and the `prepared.json`/`manifest.json` contract.
- CLI: `python -m memorybench prepare-financebench [--output PATH] [--workers POSITIVE_INT]` prints JSON with `output`, `manifest`, `revision`, and `documents`.

- [ ] **Step 1: Write failing preparation and CLI tests with injected I/O**

Append this code to `tests/test_memorybench_financebench.py`:

```python
def test_prepare_downloads_only_referenced_pdfs_and_reuses_verified_pdf(tmp_path: Path):
    from memorybench.datasets.financebench_prepare import prepare_financebench

    questions = [raw_question("financebench_id_00005", 0)]
    metadata = [raw_metadata(), {
        "doc_name": "Unused_2023_10K", "company": "Unused", "gics_sector": "Utilities",
        "doc_type": "10k", "doc_period": 2023, "doc_link": "https://example.test/unused.pdf",
    }]
    calls = []

    def fetch(url: str) -> bytes:
        calls.append(url)
        if url.endswith("/commits/main"):
            return b'{"sha":"revision123"}'
        if url.endswith("data/financebench_open_source.jsonl"):
            return (json.dumps(questions[0]) + "\\n").encode("utf-8")
        if url.endswith("data/financebench_document_information.jsonl"):
            return "".join(json.dumps(row) + "\\n" for row in metadata).encode("utf-8")
        if url.endswith("pdfs/Acme_2023_10K.pdf"):
            return b"%PDF-1.7 fake financebench fixture"
        raise AssertionError(url)

    result = prepare_financebench(
        tmp_path / "financebench", workers=1, fetch=fetch, extractor=lambda path: ["Revenue was $10."],
    )

    assert result.document_count == 1
    assert (result.output / "prepared.json").exists()
    manifest = json.loads((result.output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["required_documents"] == ["Acme_2023_10K"]
    assert manifest["documents"]["Acme_2023_10K"]["pdf_url"].endswith("pdfs/Acme_2023_10K.pdf")
    assert any(url.endswith("pdfs/Acme_2023_10K.pdf") for url in calls)
    assert not any("Unused_2023_10K.pdf" in url for url in calls)

    prepare_financebench(
        tmp_path / "financebench", workers=1, fetch=fetch, extractor=lambda path: ["Revenue was $10."],
    )
    assert sum(url.endswith("pdfs/Acme_2023_10K.pdf") for url in calls) == 1


def test_prepare_writes_identical_prepared_json_for_one_and_two_workers(tmp_path: Path):
    from memorybench.datasets.financebench_prepare import prepare_financebench

    first_question = raw_question("financebench_id_00007", 0)
    second_question = raw_question("financebench_id_00008", 0)
    second_question["doc_name"] = "Beta_2023_10K"
    second_question["evidence"][0]["doc_name"] = "Beta_2023_10K"
    second_metadata = {
        "doc_name": "Beta_2023_10K", "company": "Beta", "gics_sector": "Utilities",
        "doc_type": "10k", "doc_period": 2023, "doc_link": "https://example.test/beta.pdf",
    }

    def fetch(url: str) -> bytes:
        if url.endswith("/commits/main"):
            return b'{"sha":"revision123"}'
        if url.endswith("financebench_open_source.jsonl"):
            return (json.dumps(first_question) + "\\n" + json.dumps(second_question) + "\\n").encode("utf-8")
        if url.endswith("financebench_document_information.jsonl"):
            return (json.dumps(raw_metadata()) + "\\n" + json.dumps(second_metadata) + "\\n").encode("utf-8")
        if url.endswith("Acme_2023_10K.pdf") or url.endswith("Beta_2023_10K.pdf"):
            return b"%PDF-1.7 fake financebench fixture"
        raise AssertionError(url)

    first = prepare_financebench(
        tmp_path / "one", workers=1, fetch=fetch, extractor=lambda path: [f"Revenue for {path.stem}."],
    )
    second = prepare_financebench(
        tmp_path / "two", workers=2, fetch=fetch, extractor=lambda path: [f"Revenue for {path.stem}."],
    )

    assert (first.output / "prepared.json").read_bytes() == (second.output / "prepared.json").read_bytes()


def test_prepare_checks_for_pypdf_before_any_network_fetch(monkeypatch, tmp_path: Path):
    import memorybench.datasets.financebench_prepare as preparation

    monkeypatch.setattr(
        preparation,
        "pypdf_version",
        lambda: (_ for _ in ()).throw(RuntimeError("FinanceBench preparation requires pypdf. Run: uv sync --extra financebench")),
    )
    fetched = False

    def fetch(url: str) -> bytes:
        nonlocal fetched
        fetched = True
        return b""

    with pytest.raises(RuntimeError, match="requires pypdf"):
        preparation.prepare_financebench(tmp_path / "financebench", fetch=fetch)

    assert fetched is False


def test_prepare_writes_failed_manifest_when_required_evidence_cannot_be_recovered(tmp_path: Path):
    from memorybench.datasets.financebench_prepare import prepare_financebench

    question = raw_question("financebench_id_00006", 0)
    question["evidence"][0]["evidence_text_full_page"] = ""

    def fetch(url: str) -> bytes:
        if url.endswith("/commits/main"):
            return b'{"sha":"revision123"}'
        if url.endswith("financebench_open_source.jsonl"):
            return (json.dumps(question) + "\\n").encode("utf-8")
        if url.endswith("financebench_document_information.jsonl"):
            return (json.dumps(raw_metadata()) + "\\n").encode("utf-8")
        if url.endswith("Acme_2023_10K.pdf"):
            return b"%PDF-1.7 fake financebench fixture"
        raise AssertionError(url)

    with pytest.raises(RuntimeError, match="FinanceBench preparation failed"):
        prepare_financebench(tmp_path / "financebench", workers=1, fetch=fetch, extractor=lambda path: [""])

    manifest = json.loads((tmp_path / "financebench" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["documents"]["Acme_2023_10K"]["status"] == "failed"
    assert not (tmp_path / "financebench" / "prepared.json").exists()


def test_prepare_financebench_cli_prints_result(monkeypatch, capsys, tmp_path: Path):
    from memorybench.cli import main
    from memorybench.datasets.financebench_prepare import PreparationResult

    def fake_prepare(output: Path, workers: int):
        assert output == tmp_path / "prepared"
        assert workers == 3
        return PreparationResult(output, output / "manifest.json", "revision123", 1)

    monkeypatch.setattr("memorybench.datasets.financebench_prepare.prepare_financebench", fake_prepare)

    assert main(["prepare-financebench", "--output", str(tmp_path / "prepared"), "--workers", "3"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "documents": 1,
        "manifest": str(tmp_path / "prepared" / "manifest.json"),
        "output": str(tmp_path / "prepared"),
        "revision": "revision123",
    }
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run:

```bash
uv run python -m pytest tests/test_memorybench_financebench.py -k 'prepare_' -v
```

Expected: FAIL because `prepare_financebench` and `PreparationResult` are not defined and the CLI does not recognize `prepare-financebench`.

- [ ] **Step 3: Implement the network, persistence, resume, and CLI layers**

Append these imports and public interface to `src/memorybench/datasets/financebench_prepare.py`:

```python
import hashlib
import json
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import quote
from urllib.request import Request, urlopen

from memorybench.artifacts import atomic_json
from memorybench.datasets.financebench import (
    PREPARED_SCHEMA_VERSION,
    PREPARATION_MANIFEST_SCHEMA_VERSION,
)


UPSTREAM_REPOSITORY = "patronus-ai/financebench"
UPSTREAM_COMMIT_URL = "https://api.github.com/repos/patronus-ai/financebench/commits/main"
UPSTREAM_RAW_ROOT = "https://raw.githubusercontent.com/patronus-ai/financebench"


@dataclass(frozen=True)
class PreparationResult:
    output: Path
    manifest_path: Path
    revision: str
    document_count: int


def fetch_url(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "memorybench-financebench-preparer"})
    with urlopen(request, timeout=120) as response:
        return response.read()


def prepare_financebench(
    output: Path,
    workers: int = 4,
    *,
    fetch: Callable[[str], bytes] = fetch_url,
    extractor: Callable[[Path], list[str]] | None = None,
) -> PreparationResult:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    output = Path(output)
    source_dir = output / "source"
    pdf_dir = source_dir / "pdfs"
    manifest_path = output / "manifest.json"
    prepared_path = output / "prepared.json"
    reports: dict[str, dict[str, Any]] = {}
    revision = ""
    question_bytes = b""
    metadata_bytes = b""
    extraction_version = "unavailable"
    try:
        extraction_version = pypdf_version() if extractor is None else "injected"
        revision = resolve_revision(fetch)
        question_bytes = fetch(raw_url(revision, "data/financebench_open_source.jsonl"))
        metadata_bytes = fetch(raw_url(revision, "data/financebench_document_information.jsonl"))
        atomic_bytes(source_dir / "financebench_open_source.jsonl", question_bytes)
        atomic_bytes(source_dir / "financebench_document_information.jsonl", metadata_bytes)
        questions = read_jsonl_bytes(question_bytes, "financebench_open_source.jsonl")
        metadata = read_jsonl_bytes(metadata_bytes, "financebench_document_information.jsonl")
        questions_by_doc = source_questions_by_document(questions)
        metadata_by_doc = source_metadata_by_document(metadata)
        missing_metadata = sorted(set(questions_by_doc) - set(metadata_by_doc))
        if missing_metadata:
            raise ValueError(f"FinanceBench metadata missing documents: {', '.join(missing_metadata)}")
        previous = read_manifest(manifest_path)
        reports = {
            doc_name: {
                "status": "pending",
                "pdf_url": raw_url(revision, f"pdfs/{doc_name}.pdf"),
                "document_source_url": required_string(metadata_by_doc[doc_name], "doc_link", doc_name),
            }
            for doc_name in sorted(questions_by_doc)
        }
        pdfs, failures = download_pdfs(
            revision, sorted(questions_by_doc), pdf_dir, previous, fetch, workers,
        )
        for doc_name, path in pdfs.items():
            reports[doc_name].update({"status": "downloaded", "pdf_sha256": sha256_file(path)})
        if failures:
            for doc_name, failure in failures.items():
                reports[doc_name].update(failure)
            raise ValueError(f"PDF download failed for {', '.join(sorted(failures))}")
        pages_by_doc, failures = extract_documents(pdfs, extractor, workers)
        if failures:
            for doc_name, failure in failures.items():
                reports[doc_name].update(failure)
            raise ValueError(f"PDF extraction failed for {', '.join(sorted(failures))}")
        documents = []
        for doc_name in sorted(questions_by_doc):
            try:
                document, report = build_prepared_document(
                    doc_name, metadata_by_doc[doc_name], questions_by_doc[doc_name], pages_by_doc[doc_name],
                )
            except Exception as exc:
                reports[doc_name].update({"status": "failed", "stage": "normalize", "error": f"{type(exc).__name__}: {exc}"})
                raise
            report = {**reports[doc_name], **report}
            report.update({
                "status": "completed",
                "pdf_sha256": sha256_file(pdfs[doc_name]),
            })
            reports[doc_name] = report
            documents.append(document)
        prepared = {
            "schema_version": PREPARED_SCHEMA_VERSION,
            "source": {"repository": UPSTREAM_REPOSITORY, "revision": revision},
            "documents": documents,
        }
        atomic_json(prepared_path, prepared)
        manifest = manifest_payload(
            "completed", revision, question_bytes, metadata_bytes, reports,
            prepared_sha256=sha256_file(prepared_path), extraction_version=extraction_version,
        )
        atomic_json(manifest_path, manifest)
        return PreparationResult(output, manifest_path, revision, len(documents))
    except Exception as exc:
        atomic_json(manifest_path, manifest_payload(
            "failed", revision, question_bytes, metadata_bytes, reports,
            error=f"{type(exc).__name__}: {exc}", extraction_version=extraction_version,
        ))
        raise RuntimeError(f"FinanceBench preparation failed: {exc}") from exc
```

Add these helpers below that interface. They establish the exact URLs, atomic byte write, per-PDF checksum resume behavior, deterministic JSONL parsing, and manifest layout.

```python
def resolve_revision(fetch: Callable[[str], bytes]) -> str:
    payload = json.loads(fetch(UPSTREAM_COMMIT_URL).decode("utf-8"))
    revision = payload.get("sha")
    if not isinstance(revision, str) or not revision:
        raise ValueError("FinanceBench upstream commit response has no sha")
    return revision


def raw_url(revision: str, relative_path: str) -> str:
    return f"{UPSTREAM_RAW_ROOT}/{revision}/{quote(relative_path, safe='/')}"


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl_bytes(payload: bytes, source_name: str) -> list[dict[str, Any]]:
    result = []
    for line_number, line in enumerate(payload.decode("utf-8").splitlines(), start=1):
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL in {source_name} line {line_number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"JSONL object required in {source_name} line {line_number}")
        result.append(value)
    return result


def read_manifest(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, Mapping) else {}


def download_pdfs(
    revision: str,
    doc_names: Sequence[str],
    pdf_dir: Path,
    previous: Mapping[str, Any],
    fetch: Callable[[str], bytes],
    workers: int,
) -> tuple[dict[str, Path], dict[str, dict[str, str]]]:
    prior_reports = previous.get("documents", {}) if isinstance(previous.get("documents"), Mapping) else {}

    def download(doc_name: str) -> tuple[str, Path]:
        path = pdf_dir / f"{doc_name}.pdf"
        prior = prior_reports.get(doc_name, {})
        expected = prior.get("pdf_sha256") if isinstance(prior, Mapping) else None
        if path.is_file() and isinstance(expected, str) and sha256_file(path) == expected:
            return doc_name, path
        payload = fetch(raw_url(revision, f"pdfs/{doc_name}.pdf"))
        if not payload.startswith(b"%PDF-"):
            raise ValueError(f"FinanceBench source is not a PDF for {doc_name}")
        atomic_bytes(path, payload)
        return doc_name, path

    result: dict[str, Path] = {}
    failures: dict[str, dict[str, str]] = {}
    if workers == 1:
        for doc_name in doc_names:
            try:
                name, path = download(doc_name)
                result[name] = path
            except Exception as exc:
                failures[doc_name] = {"status": "failed", "stage": "download", "error": f"{type(exc).__name__}: {exc}"}
        return result, failures
    with ThreadPoolExecutor(max_workers=min(workers, len(doc_names))) as executor:
        futures = {executor.submit(download, doc_name): doc_name for doc_name in doc_names}
        for future in as_completed(futures):
            requested = futures[future]
            try:
                doc_name, path = future.result()
                result[doc_name] = path
            except Exception as exc:
                failures[requested] = {"status": "failed", "stage": "download", "error": f"{type(exc).__name__}: {exc}"}
    return result, failures


def extract_documents(
    pdfs: Mapping[str, Path],
    extractor: Callable[[Path], list[str]] | None,
    workers: int,
) -> tuple[dict[str, list[str]], dict[str, dict[str, str]]]:
    active_extractor = extractor or extract_layout_pages
    items = sorted(pdfs.items())
    result: dict[str, list[str]] = {}
    failures: dict[str, dict[str, str]] = {}
    if workers == 1:
        for doc_name, path in items:
            try:
                result[doc_name] = active_extractor(path)
            except Exception as exc:
                failures[doc_name] = {"status": "failed", "stage": "extract", "error": f"{type(exc).__name__}: {exc}"}
        return result, failures
    executor_type = ProcessPoolExecutor if extractor is None else ThreadPoolExecutor
    with executor_type(max_workers=min(workers, len(items))) as executor:
        futures = {executor.submit(active_extractor, path): doc_name for doc_name, path in items}
        for future in as_completed(futures):
            doc_name = futures[future]
            try:
                result[doc_name] = future.result()
            except Exception as exc:
                failures[doc_name] = {"status": "failed", "stage": "extract", "error": f"{type(exc).__name__}: {exc}"}
    return result, failures


def manifest_payload(
    status: str,
    revision: str,
    question_bytes: bytes,
    metadata_bytes: bytes,
    reports: Mapping[str, Mapping[str, Any]],
    *,
    prepared_sha256: str | None = None,
    error: str | None = None,
    extraction_version: str,
) -> dict[str, Any]:
    payload = {
        "schema_version": PREPARATION_MANIFEST_SCHEMA_VERSION,
        "status": status,
        "upstream": {"repository": UPSTREAM_REPOSITORY, "revision": revision},
        "source_sha256": {
            "financebench_open_source.jsonl": hashlib.sha256(question_bytes).hexdigest() if question_bytes else None,
            "financebench_document_information.jsonl": hashlib.sha256(metadata_bytes).hexdigest() if metadata_bytes else None,
        },
        "parameters": {
            "max_page_words": MAX_PAGE_WORDS,
            "extraction_mode": "layout",
            "pypdf_version": extraction_version,
        },
        "required_documents": sorted(reports),
        "documents": {name: reports[name] for name in sorted(reports)},
    }
    if prepared_sha256 is not None:
        payload["prepared_sha256"] = prepared_sha256
    if error is not None:
        payload["error"] = error
    return payload
```

Keep extraction concurrency bounded by `workers`; use `ProcessPoolExecutor` for the default top-level `extract_layout_pages` worker and the injected-extractor thread pool only in tests. Add a test that compares the one-worker and two-worker prepared JSON payloads for the same injected page inputs before relying on multi-process extraction. Never send chunks from a single PDF into concurrent A-Mem work.

Update `src/memorybench/cli.py` with a positive integer parser and a command branch:

```python
def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed
```

```python
    prepare_financebench = commands.add_parser("prepare-financebench")
    prepare_financebench.add_argument("--output", type=Path, default=Path("artifacts/datasets/financebench"))
    prepare_financebench.add_argument("--workers", type=positive_int, default=4)
```

Place this branch before the dashboard fallback in `main`:

```python
        if args.command == "prepare-financebench":
            from .datasets.financebench_prepare import prepare_financebench

            result = prepare_financebench(args.output, args.workers)
            print(json.dumps({
                "output": str(result.output),
                "manifest": str(result.manifest_path),
                "revision": result.revision,
                "documents": result.document_count,
            }, sort_keys=True))
            return 0
```

- [ ] **Step 4: Run preparation and CLI tests, then the complete test suite**

Run:

```bash
uv run python -m pytest tests/test_memorybench_financebench.py -k 'prepare_' -v
uv run python -m pytest -v
```

Expected: PASS. The fake fetcher observes only the one referenced PDF URL, the second preparation reuses its checksum-verified PDF, failure leaves a failed manifest, and the CLI emits deterministic JSON.

- [ ] **Step 5: Commit the preparation-command milestone**

```bash
git add src/memorybench/datasets/financebench_prepare.py src/memorybench/cli.py tests/test_memorybench_financebench.py
git commit -m "feat: add FinanceBench preparation command"
```

### Task 4: Add Runnable llama.cpp Configurations, Documentation, and a Fake A-Mem Integration Test

**Files:**

- Create: `configs/financebench_llamacpp_smoke.yaml`
- Create: `configs/financebench_llamacpp.yaml`
- Modify: `README.md:14-39`
- Modify: `tests/test_memorybench_financebench.py`

**Interfaces:**

- Consumes: `artifacts/datasets/financebench/prepared.json`, the registered `financebench` adapter, and llama.cpp's OpenAI-compatible server on port 8080.
- Produces: a small local characterization config, a full corpus config, reproducible artifact directories, and a CI-safe fake-provider runner test.

- [ ] **Step 1: Write failing configuration and fake-A-Mem runner tests**

Append these tests to `tests/test_memorybench_financebench.py`:

```python
def test_financebench_llamacpp_configs_validate_without_importing_pypdf():
    from memorybench.config import load_config

    smoke = load_config("configs/financebench_llamacpp_smoke.yaml")
    full = load_config("configs/financebench_llamacpp.yaml")

    assert smoke.pipeline.dataset.adapter == "financebench"
    assert smoke.pipeline.construction.llm.provider == "vllm"
    assert smoke.pipeline.construction.llm.params == {"host": "http://127.0.0.1", "port": 8080}
    assert smoke.pipeline.construction.selection.sample_limit == 1
    assert smoke.pipeline.retrieve_qa.selection.question_limit == 1
    assert full.pipeline.construction.selection.sample_limit is None
    assert full.runtime.max_workers == 1


def test_financebench_adapter_runs_through_amem_with_fake_provider(tmp_path: Path):
    from memorybench.config import MemoryBenchConfig
    from memorybench.runner import ExperimentRunner

    prepared_path = write_prepared(tmp_path)
    config = MemoryBenchConfig.model_validate({
        "experiment": {"id": "financebench-fake"},
        "pipeline": {
            "stages": ["construction", "retrieve_qa"],
            "dataset": {"adapter": "financebench", "path": str(prepared_path)},
            "construction": {
                "adapter": "amem",
                "llm": {
                    "provider": "fake", "model": "fake-amem",
                    "params": {"responses": [
                        "KEYWORDS: revenue\\nCONTEXT: Revenue was $10\\nTAGS: revenue",
                    ]},
                },
                "params": {"retrieval_mode": "bm25", "keyword_pruning_mode": "simple"},
            },
            "retrieve_qa": {
                "retrieval": {"adapter": "staged", "stages": [{"adapter": "bm25", "top_k": 1}]},
                "context": {"adapter": "amem"},
                "qa": {
                    "adapter": "robust",
                    "llm": {"provider": "fake", "model": "fake-qa", "params": {"response": "$10"}},
                },
                "metrics": [{"adapter": "exact_match"}],
                "selection": {"sample_limit": 1, "question_limit": 1},
            },
        },
        "runtime": {"artifact_root": str(tmp_path / "artifacts"), "on_error": "stop"},
    })

    outcome = ExperimentRunner(config).run()

    assert outcome.exit_code == 0
    results = (outcome.artifact_dir / "retrieve_qa/construction_000/run_000/results.jsonl").read_text(encoding="utf-8")
    assert '"prediction": "$10"' in results
    assert '"financebench:Acme_2023_10K:page:0"' in results
```

- [ ] **Step 2: Run the configuration tests to verify they fail before files exist**

Run:

```bash
uv run python -m pytest tests/test_memorybench_financebench.py -k 'llamacpp_configs or fake_provider' -v
```

Expected: FAIL with `FileNotFoundError` for `configs/financebench_llamacpp_smoke.yaml`.

- [ ] **Step 3: Add the exact YAML configurations and README instructions**

Create `configs/financebench_llamacpp_smoke.yaml`:

```yaml
experiment:
  id: financebench_llamacpp_smoke
  description: One-document FinanceBench A-Mem characterization through local llama.cpp
pipeline:
  stages: [construction, retrieve_qa]
  dataset:
    adapter: financebench
    path: artifacts/datasets/financebench/prepared.json
  construction:
    adapter: amem
    runs: 1
    llm:
      provider: vllm
      model: llama3.2
      params:
        host: http://127.0.0.1
        port: 8080
    params:
      retrieval_mode: bm25
      keyword_pruning_mode: simple
    selection:
      sample_limit: 1
  retrieve_qa:
    runs: 1
    retrieval:
      adapter: staged
      stages:
        - adapter: bm25
          top_k: 5
    context:
      adapter: amem
    qa:
      adapter: robust
      llm:
        provider: vllm
        model: llama3.2
        params:
          host: http://127.0.0.1
          port: 8080
    metrics:
      - adapter: exact_match
      - adapter: f1
      - adapter: bleu1
    selection:
      sample_limit: 1
      question_limit: 1
runtime:
  artifact_root: artifacts/experiments
  max_workers: 1
  resume: true
  on_error: continue
```

Create `configs/financebench_llamacpp.yaml`:

```yaml
experiment:
  id: financebench_llamacpp
  description: Full public FinanceBench A-Mem evaluation through local llama.cpp
pipeline:
  stages: [construction, retrieve_qa]
  dataset:
    adapter: financebench
    path: artifacts/datasets/financebench/prepared.json
  construction:
    adapter: amem
    runs: 1
    llm:
      provider: vllm
      model: llama3.2
      params:
        host: http://127.0.0.1
        port: 8080
    params:
      retrieval_mode: bm25
      keyword_pruning_mode: simple
  retrieve_qa:
    runs: 1
    retrieval:
      adapter: staged
      stages:
        - adapter: bm25
          top_k: 5
    context:
      adapter: amem
    qa:
      adapter: robust
      llm:
        provider: vllm
        model: llama3.2
        params:
          host: http://127.0.0.1
          port: 8080
    metrics:
      - adapter: exact_match
      - adapter: f1
      - adapter: bleu1
runtime:
  artifact_root: artifacts/experiments
  max_workers: 1
  resume: true
  on_error: continue
```

Append this section to `README.md` after the existing quick-start material:

````markdown
## FinanceBench (local PDF corpus)

Prepare the public FinanceBench question-linked PDF corpus once. Preparation downloads 84 source PDFs for the 150 public questions, extracts complete pages, and writes only ignored artifacts. Benchmark runs are offline after this step.

```bash
uv sync --extra dev --extra providers --extra financebench
uv run python -m memorybench prepare-financebench --output artifacts/datasets/financebench --workers 4
uv run python -m memorybench validate --config configs/financebench_llamacpp_smoke.yaml
uv run python -m memorybench run --config configs/financebench_llamacpp_smoke.yaml
```

The FinanceBench configs use the existing `vllm` provider label solely to call a llama.cpp OpenAI-compatible server at `http://127.0.0.1:8080/v1` with model `llama3.2`; this branch does not add a llama.cpp provider. Start full runs with `runtime.max_workers: 1`, then measure two and four workers before changing the configuration. The included exact match, F1, and BLEU-1 values are diagnostic metrics, not official FinanceBench scores.
````

- [ ] **Step 4: Run focused integration tests, validation, and all automated verification**

Run:

```bash
uv sync --extra dev --extra providers --extra financebench
uv run python -m pytest tests/test_memorybench_financebench.py -v
uv run python -m memorybench validate --config configs/financebench_llamacpp_smoke.yaml
uv run python -m memorybench validate --config configs/financebench_llamacpp.yaml
uv run python -m pytest -v
uv run python -m compileall -q src/memorybench tests
```

Expected: all pytest tests pass, both configs print a JSON object with `"valid": true`, their configured `experiment_id`, and a SHA-256 `fingerprint`, and `compileall` prints no errors. These commands do not contact llama.cpp or download FinanceBench.

- [ ] **Step 5: Commit the runnable-config and documentation milestone**

```bash
git add configs/financebench_llamacpp_smoke.yaml configs/financebench_llamacpp.yaml README.md tests/test_memorybench_financebench.py
git commit -m "docs: add FinanceBench llama.cpp run configs"
```

### Task 5: Perform the Manual Local Acceptance Sequence

**Files:**

- Generated only: `artifacts/datasets/financebench/`
- Generated only: `artifacts/experiments/financebench_llamacpp_smoke/`
- Generated only: `artifacts/experiments/financebench_llamacpp/`

**Interfaces:**

- Consumes: the installed `financebench` and `providers` extras plus the local llama.cpp server at `http://127.0.0.1:8080/v1` exposing model `llama3.2`.
- Produces: ignored preparation and experiment artifacts; no repository source files change.

- [ ] **Step 1: Confirm the intended local model is available**

Run:

```bash
curl --fail --silent --show-error http://127.0.0.1:8080/v1/models
```

Expected: JSON includes an item whose `id` is `llama3.2`.

- [ ] **Step 2: Prepare the local FinanceBench corpus**

Run:

```bash
uv run python -m memorybench prepare-financebench --output artifacts/datasets/financebench --workers 4
```

Expected: JSON reports `documents: 84`; `artifacts/datasets/financebench/manifest.json` has `status: "completed"`; `prepared.json` exists; all required PDF failures are absent from the manifest.

- [ ] **Step 3: Run and inspect the one-document smoke experiment**

Run:

```bash
uv run python -m memorybench run --config configs/financebench_llamacpp_smoke.yaml
```

Expected: exit code `0` or `2`; exit code `2` is acceptable only when `errors.jsonl` identifies a provider or parsing failure. Inspect the completed question row and verify every `retrieval.items[*].evidence_refs[*]` uses a `financebench:<doc>:page:<zero-based-page>` ID.

- [ ] **Step 4: Calibrate document-level parallelism without changing note order**

Create ignored calibration configs by copying the committed full config:

```bash
cp configs/financebench_llamacpp.yaml artifacts/financebench_llamacpp_w1.yaml
cp configs/financebench_llamacpp.yaml artifacts/financebench_llamacpp_w2.yaml
cp configs/financebench_llamacpp.yaml artifacts/financebench_llamacpp_w4.yaml
```

Apply these exact substitutions to the copied files:

```diff
*** Begin Patch
*** Update File: artifacts/financebench_llamacpp_w1.yaml
@@
-  id: financebench_llamacpp
+  id: financebench_llamacpp_w1
*** Update File: artifacts/financebench_llamacpp_w2.yaml
@@
-  id: financebench_llamacpp
+  id: financebench_llamacpp_w2
@@
-  max_workers: 1
+  max_workers: 2
*** Update File: artifacts/financebench_llamacpp_w4.yaml
@@
-  id: financebench_llamacpp
+  id: financebench_llamacpp_w4
@@
-  max_workers: 1
+  max_workers: 4
*** End Patch
```

Run:

```bash
uv run python -m memorybench run --config artifacts/financebench_llamacpp_w1.yaml
uv run python -m memorybench run --config artifacts/financebench_llamacpp_w2.yaml
uv run python -m memorybench run --config artifacts/financebench_llamacpp_w4.yaml
```

Expected: choose the highest worker count that improves elapsed wall time without provider failures, excessive queueing, or GPU-memory errors. Keep per-document page turns sequential in every run.

- [ ] **Step 5: Launch the full baseline evaluation and preserve its artifacts**

Run:

```bash
uv run python -m memorybench run --config configs/financebench_llamacpp.yaml
```

Expected: canonical results, usage, retrieval traces, and construction stores are written beneath `artifacts/experiments/financebench_llamacpp/`. Do not stage any content under `artifacts/`.

## Plan Self-Review

### Spec coverage

- Public scope, offline preparation, resolved upstream provenance, checksums, resume, and generated-artifact placement are implemented in Task 3 and manually verified in Task 5.
- Page-first layout extraction, the 1,200-word blank-line split policy, zero-based IDs, and evidence-page fallback are implemented and unit-tested in Task 2.
- Per-document samples, native taxonomy, null reasoning labels, local-only loading, and manifest integrity checks are implemented and tested in Task 1.
- Existing llama.cpp compatibility configuration, BM25/simple keyword mode, independent-document parallelism, and the diagnostic metric boundary are captured in Task 4 and calibrated in Task 5.
- Lazy dependencies, non-network CI, and the project-required full test and compile checks are explicitly verified in Task 4.

### Placeholder scan

The plan contains no deferred implementation markers, unspecified paths, or unspecified test behavior. Every named function, configuration file, command, and output contract is defined in a preceding task.

### Type consistency

`prepared.json` uses `question_id`, `reference`, `evidence_ids`, `question_type`, and `question_reasoning`; Task 1 consumes those exact names and Tasks 2 and 3 produce them. Both the adapter and preparation manifest use the schema-version constants defined in Task 1. `PreparationResult` fields match the CLI JSON keys and the CLI test assertion.
