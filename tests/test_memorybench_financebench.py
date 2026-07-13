import hashlib
import json
import tomllib
from pathlib import Path

import pytest

from memorybench.registry import component_catalog


def test_financebench_extra_declares_pdf_crypto_dependencies():
    with (Path(__file__).parents[1] / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)

    dependencies = set(project["project"]["optional-dependencies"]["financebench"])
    assert {"pypdf>=6.0", "cryptography>=3.1"} <= dependencies


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
            return (json.dumps(questions[0]) + "\n").encode("utf-8")
        if url.endswith("data/financebench_document_information.jsonl"):
            return "".join(json.dumps(row) + "\n" for row in metadata).encode("utf-8")
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


def test_prepare_ignores_duplicate_unreferenced_metadata_rows(tmp_path: Path):
    from memorybench.datasets.financebench_prepare import prepare_financebench

    metadata = [raw_metadata(), {
        "doc_name": "FOOTLOCKER_2023_annualreport", "company": "Foot Locker",
        "gics_sector": "Consumer Discretionary", "doc_type": "10k", "doc_period": 2023,
        "doc_link": "https://example.test/footlocker.pdf",
    }, {
        "doc_name": "FOOTLOCKER_2023_annualreport", "company": "Foot Locker",
        "gics_sector": "Consumer Discretionary", "doc_type": "10k", "doc_period": 2022,
        "doc_link": "https://example.test/footlocker.pdf",
    }]

    def fetch(url: str) -> bytes:
        if url.endswith("/commits/main"):
            return b'{"sha":"revision123"}'
        if url.endswith("financebench_open_source.jsonl"):
            return (json.dumps(raw_question("financebench_id_00010", 0)) + "\n").encode("utf-8")
        if url.endswith("financebench_document_information.jsonl"):
            return "".join(json.dumps(row) + "\n" for row in metadata).encode("utf-8")
        if url.endswith("pdfs/Acme_2023_10K.pdf"):
            return b"%PDF-1.7 fake financebench fixture"
        raise AssertionError(url)

    result = prepare_financebench(
        tmp_path / "financebench", workers=1, fetch=fetch, extractor=lambda path: ["Revenue was $10."],
    )

    assert result.document_count == 1
    manifest = json.loads((tmp_path / "financebench" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["required_documents"] == ["Acme_2023_10K"]


def test_prepare_refetches_pdf_when_upstream_revision_changes(tmp_path: Path):
    from memorybench.datasets.financebench_prepare import prepare_financebench

    question = raw_question("financebench_id_00009", 0)
    revisions = iter(("revision123", "revision456"))
    calls = []

    def fetch(url: str) -> bytes:
        calls.append(url)
        if url.endswith("/commits/main"):
            return json.dumps({"sha": next(revisions)}).encode("utf-8")
        if url.endswith("financebench_open_source.jsonl"):
            return (json.dumps(question) + "\n").encode("utf-8")
        if url.endswith("financebench_document_information.jsonl"):
            return (json.dumps(raw_metadata()) + "\n").encode("utf-8")
        if url.endswith("pdfs/Acme_2023_10K.pdf"):
            return b"%PDF-1.7 revision-specific financebench fixture"
        raise AssertionError(url)

    prepare_financebench(
        tmp_path / "financebench", workers=1, fetch=fetch, extractor=lambda path: ["Revenue was $10."],
    )
    prepare_financebench(
        tmp_path / "financebench", workers=1, fetch=fetch, extractor=lambda path: ["Revenue was $10."],
    )

    pdf_calls = [url for url in calls if url.endswith("pdfs/Acme_2023_10K.pdf")]
    assert pdf_calls == [
        "https://raw.githubusercontent.com/patronus-ai/financebench/revision123/pdfs/Acme_2023_10K.pdf",
        "https://raw.githubusercontent.com/patronus-ai/financebench/revision456/pdfs/Acme_2023_10K.pdf",
    ]
    manifest = json.loads((tmp_path / "financebench" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["upstream"]["revision"] == "revision456"


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
            return (json.dumps(first_question) + "\n" + json.dumps(second_question) + "\n").encode("utf-8")
        if url.endswith("financebench_document_information.jsonl"):
            return (json.dumps(raw_metadata()) + "\n" + json.dumps(second_metadata) + "\n").encode("utf-8")
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
            return (json.dumps(question) + "\n").encode("utf-8")
        if url.endswith("financebench_document_information.jsonl"):
            return (json.dumps(raw_metadata()) + "\n").encode("utf-8")
        if url.endswith("Acme_2023_10K.pdf"):
            return b"%PDF-1.7 fake financebench fixture"
        raise AssertionError(url)

    prepare_financebench(
        tmp_path / "financebench", workers=1, fetch=fetch, extractor=lambda path: ["Revenue was $10."],
    )
    assert (tmp_path / "financebench" / "prepared.json").exists()

    with pytest.raises(RuntimeError, match="FinanceBench preparation failed"):
        prepare_financebench(tmp_path / "financebench", workers=1, fetch=fetch, extractor=lambda path: [""])

    manifest = json.loads((tmp_path / "financebench" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["documents"]["Acme_2023_10K"]["status"] == "failed"
    assert not (tmp_path / "financebench" / "prepared.json").exists()


def test_prepare_records_sha256_for_empty_fetched_source_on_failure(tmp_path: Path):
    from memorybench.datasets.financebench_prepare import prepare_financebench

    def fetch(url: str) -> bytes:
        if url.endswith("/commits/main"):
            return b'{"sha":"revision123"}'
        if url.endswith("financebench_open_source.jsonl"):
            return b""
        if url.endswith("financebench_document_information.jsonl"):
            return (json.dumps(raw_metadata()) + "\n").encode("utf-8")
        raise AssertionError(url)

    with pytest.raises(RuntimeError, match="FinanceBench preparation failed"):
        prepare_financebench(tmp_path / "financebench", workers=1, fetch=fetch, extractor=lambda path: [""])

    manifest = json.loads((tmp_path / "financebench" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_sha256"]["financebench_open_source.jsonl"] == hashlib.sha256(b"").hexdigest()


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


def test_financebench_llamacpp_configs_validate_without_importing_pypdf():
    import sys

    from memorybench.config import load_config

    smoke = load_config("configs/financebench_llamacpp_smoke.yaml")
    full = load_config("configs/financebench_llamacpp.yaml")
    assert "pypdf" not in sys.modules

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
