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
