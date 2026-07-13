from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import quote
from urllib.request import Request, urlopen

from memorybench.artifacts import atomic_json
from memorybench.datasets.financebench import (
    PREPARED_SCHEMA_VERSION,
    PREPARATION_MANIFEST_SCHEMA_VERSION,
)


MAX_PAGE_WORDS = 1200
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
    question_bytes: bytes | None = None
    metadata_bytes: bytes | None = None
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
        referenced_metadata = [
            row for row in metadata
            if isinstance(row.get("doc_name"), str) and row["doc_name"] in questions_by_doc
        ]
        metadata_by_doc = source_metadata_by_document(referenced_metadata)
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
                reports[doc_name].update({
                    "status": "failed",
                    "stage": "normalize",
                    "error": f"{type(exc).__name__}: {exc}",
                })
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
        prepared_path.unlink(missing_ok=True)
        atomic_json(manifest_path, manifest_payload(
            "failed", revision, question_bytes, metadata_bytes, reports,
            error=f"{type(exc).__name__}: {exc}", extraction_version=extraction_version,
        ))
        raise RuntimeError(f"FinanceBench preparation failed: {exc}") from exc


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
    prior_upstream = previous.get("upstream")
    same_revision = (
        isinstance(prior_upstream, Mapping)
        and prior_upstream.get("repository") == UPSTREAM_REPOSITORY
        and prior_upstream.get("revision") == revision
    )

    def download(doc_name: str) -> tuple[str, Path]:
        path = pdf_dir / f"{doc_name}.pdf"
        prior = prior_reports.get(doc_name, {})
        expected = prior.get("pdf_sha256") if isinstance(prior, Mapping) else None
        current_url = raw_url(revision, f"pdfs/{doc_name}.pdf")
        can_resume = (
            same_revision
            and isinstance(prior, Mapping)
            and prior.get("pdf_url") == current_url
            and prior.get("status") in {"downloaded", "completed"}
            and isinstance(expected, str)
        )
        if path.is_file() and can_resume and sha256_file(path) == expected:
            return doc_name, path
        payload = fetch(current_url)
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
                failures[doc_name] = {
                    "status": "failed", "stage": "download", "error": f"{type(exc).__name__}: {exc}",
                }
        return result, failures
    with ThreadPoolExecutor(max_workers=min(workers, len(doc_names))) as executor:
        futures = {executor.submit(download, doc_name): doc_name for doc_name in doc_names}
        for future in as_completed(futures):
            requested = futures[future]
            try:
                doc_name, path = future.result()
                result[doc_name] = path
            except Exception as exc:
                failures[requested] = {
                    "status": "failed", "stage": "download", "error": f"{type(exc).__name__}: {exc}",
                }
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
                failures[doc_name] = {
                    "status": "failed", "stage": "extract", "error": f"{type(exc).__name__}: {exc}",
                }
        return result, failures
    executor_type = ProcessPoolExecutor if extractor is None else ThreadPoolExecutor
    with executor_type(max_workers=min(workers, len(items))) as executor:
        futures = {executor.submit(active_extractor, path): doc_name for doc_name, path in items}
        for future in as_completed(futures):
            doc_name = futures[future]
            try:
                result[doc_name] = future.result()
            except Exception as exc:
                failures[doc_name] = {
                    "status": "failed", "stage": "extract", "error": f"{type(exc).__name__}: {exc}",
                }
    return result, failures


def manifest_payload(
    status: str,
    revision: str,
    question_bytes: bytes | None,
    metadata_bytes: bytes | None,
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
            "financebench_open_source.jsonl": hashlib.sha256(question_bytes).hexdigest() if question_bytes is not None else None,
            "financebench_document_information.jsonl": hashlib.sha256(metadata_bytes).hexdigest() if metadata_bytes is not None else None,
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
    return {
        doc_name: sorted(items, key=lambda item: item["financebench_id"])
        for doc_name, items in sorted(grouped.items())
    }


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
                "text": f"Document: {doc_name}\nPDF page index: {page_index}\n\n{part}",
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
    blocks = [block.strip() for block in re.split(r"\n[\t ]*\n+", stripped) if block.strip()]
    parts: list[str] = []
    current: list[str] = []
    current_words = 0
    for block in blocks:
        block_words = len(block.split())
        if current and current_words + block_words > max_page_words:
            parts.append("\n\n".join(current))
            current, current_words = [], 0
        current.append(block)
        current_words += block_words
    if current:
        parts.append("\n\n".join(current))
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
