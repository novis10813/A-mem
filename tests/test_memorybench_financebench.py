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
