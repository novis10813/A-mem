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
                "text": "Document: Acme_2023_10K\nPDF page index: 0\n\nRevenue was $10.",
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
        ["one two three\n\nfour five six\n\nseven eight nine"], max_page_words=4,
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
