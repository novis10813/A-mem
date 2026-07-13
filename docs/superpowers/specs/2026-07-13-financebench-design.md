# FinanceBench Integration Design

**Status:** Approved design; implementation has not started.

## Goal

Run the public FinanceBench sample through MemoryBench's native A-Mem pipeline using the complete PDF pages of the documents referenced by the public questions. The benchmark must be reproducible, offline after preparation, and trace every question and retrieved note back to its source PDF page.

## Scope and non-goals

The first implementation covers the 150 public FinanceBench questions and only the 84 PDF documents referenced by those questions. A document is one `DatasetSample`; its complete extracted page set is the retrieval corpus for its questions.

The first implementation does not do any of the following:

- Retrieve across unrelated FinanceBench documents or the full 361-document metadata catalog.
- Add a new `llama_cpp` provider or change the A-Mem provider architecture.
- Reconstruct tables into cells or rows, remove headers or footers heuristically, OCR image-only pages, or use cross-page chunks.
- Claim official FinanceBench numerical-reasoning or human-evaluation scores.

## Data preparation lifecycle

Add a dedicated command:

```bash
uv run python -m memorybench prepare-financebench \
  --output artifacts/datasets/financebench \
  --workers 4
```

The command resolves and records the upstream FinanceBench revision, downloads the public question annotations and document metadata, derives the 84 required document names, downloads `pdfs/<doc_name>.pdf` from that same resolved repository revision, extracts their text, produces a normalized dataset, and writes an integrity manifest. The metadata `doc_link` is retained as source provenance but is not used as the download target, so the files stay aligned with the published page annotations. It is the only operation that needs network access or PDF extraction.

All generated files live below the configurable output directory, whose default is `artifacts/datasets/financebench/`. They are untracked. The expected layout is:

```text
artifacts/datasets/financebench/
  source/
    financebench_open_source.jsonl
    financebench_document_information.jsonl
    pdfs/<doc_name>.pdf
  prepared.json
  manifest.json
```

`manifest.json` records the resolved upstream revision, source URLs, SHA-256 digests, required-document list, per-document preparation status, extraction version and options, page counts, chunking parameters, and errors. Download and extraction outputs are written atomically. A later invocation resumes only files whose recorded digests still match; it returns a nonzero exit code and does not mark the dataset prepared if any required PDF or evidence page cannot be prepared.

`prepared.json` is a single deterministic JSON file. Its document, page, turn, and question arrays are sorted by stable identifiers. This allows the existing runner to use the dataset path as a single SHA-256-addressable input. `memorybench run` loads this file locally and never downloads or extracts source data.

## Upstream compatibility rules

The preparation code validates the actual public JSONL data rather than trusting only its prose documentation.

- Each evidence object uses its nested `doc_name` as the evidence-document identifier.
- `evidence_page_num` is zero-based and maps directly to the zero-based PDF page index.
- Each public question's evidence is in its top-level `doc_name`; preparation rejects a future data revision that violates this assumption rather than silently assigning a question to the wrong sample.
- `question_type` is required. `question_reasoning` is included as a label only when it is non-null.

## PDF extraction and page-first chunking

Add `pypdf` as a lazy `financebench` optional dependency. The preparation command imports it only when invoked and calls `PdfReader(..., strict=False)` with `page.extract_text(extraction_mode="layout")` for every PDF page.

If the `financebench` extra is not installed, the command exits before network activity with an install instruction containing `uv sync --extra financebench`.

The initial chunking policy is page-first:

- A normally sized extracted PDF page becomes exactly one turn and one A-Mem note.
- The text preserves meaningful line breaks and layout spacing. It receives a deterministic prefix containing the FinanceBench document name and zero-based source page index.
- Pages whose extracted content exceeds 1,200 whitespace-delimited words split at blank-line boundaries into ordered parts. Each part has its own `turn_id` but shares the page-level `evidence_id`.
- No table-cell parsing, row reconstruction, page-window overlap, or cross-page merge occurs in this version.
- If an annotated evidence page extracts to empty text, preparation uses that annotation's `evidence_text_full_page` as an explicit fallback and records the fallback in `manifest.json`. If any other page is empty, it is recorded in the manifest. If a required evidence page is still empty after fallback, preparation fails.
- An empty non-evidence page is omitted from the prepared turns so A-Mem never receives an empty note; its original zero-based page index remains represented in the manifest.

The shared page-level ID maintains page-granularity evidence even when an exceptional page produces multiple turns:

```text
sample_id:   financebench:<doc_name>
turn_id:     financebench:<doc_name>:page:<zero_based_page>
turn_id:     financebench:<doc_name>:page:<zero_based_page>:part:<one_based_part>
evidence_id: financebench:<doc_name>:page:<zero_based_page>
```

## Normalized dataset and adapter

`prepared.json` contains one normalized document entry per required document. Each entry has its source metadata, ordered prepared turns, and ordered normalized questions. A prepared turn contains the stable turn ID, shared page evidence ID, zero-based page index, optional part index, and extracted text. A normalized question contains the original `financebench_id`, question text, reference answer, page evidence IDs, and source labels.

Add `FinanceBenchAdapter` and register it as dataset adapter `financebench`. It reads only `prepared.json` and emits:

- One `DatasetSample` for each document, with `sample_id` `financebench:<doc_name>`.
- `Turn` objects with `speaker="document"`, `session_id=<doc_name>`, stable IDs, and page-level evidence IDs.
- `Question` objects with the original FinanceBench ID, question text, human answer, evidence page IDs, `question_type`, and a non-null `question_reasoning` label when available.
- A `DatasetTaxonomy` whose `question_type` and non-null `question_reasoning` values come directly from the prepared public annotations.
- Document metadata including company, GICS sector, document type, document period, and source URL.

The adapter validates the normalized schema version and fails with a useful error when the file is malformed, incomplete, or uses an unsupported schema version. It does not import `pypdf`.

## Experiment configuration and llama.cpp compatibility

Add two user-facing YAML configurations:

- `configs/financebench_llamacpp_smoke.yaml` runs one complete document and one question with `runtime.max_workers: 1`.
- `configs/financebench_llamacpp.yaml` selects all documents and questions. It begins at one worker; users calibrate two and four workers against the server's measured throughput and memory pressure.

Both configurations use the existing `vllm` provider route solely as an OpenAI-compatible llama.cpp compatibility path:

```yaml
llm:
  provider: vllm
  model: llama3.2
  params:
    host: http://127.0.0.1
    port: 8080
```

This branch intentionally does not add or rename a provider. Documentation must state that the configuration targets llama.cpp despite the temporary `vllm` provider label.

A-Mem construction uses `retrieval_mode: bm25` and `keyword_pruning_mode: simple`. This avoids selecting SentenceTransformer or NLTK for the first FinanceBench run. Retrieval uses staged BM25, context uses the existing A-Mem context adapter, and QA uses the existing robust QA prompt. Existing `exact_match`, `f1`, and `bleu1` metrics remain diagnostic only.

Users install the required optional dependencies with:

```bash
uv sync --extra dev --extra providers --extra financebench
```

## Parallelism and safety

Preparation parallelizes independent documents. PDF download and extraction work use a bounded worker count supplied by `prepare-financebench --workers`; normalized output remains deterministic because the final assembly sorts all IDs.

The existing runner parallelizes independent samples through `runtime.max_workers`, so it can construct or evaluate separate document samples concurrently. A-Mem notes inside one document remain strictly sequential because each added note can affect links and memory evolution for following notes. The recommended experiment calibration is one worker first, then two and four only if llama.cpp maintains useful throughput without exhausting its GPU context or queueing all requests.

## Verification strategy

Automated tests cover:

- Preparation selection of only referenced documents, stable manifest output, resume behavior, and a failure when an evidence page cannot be prepared.
- Layout-text extraction and the page-overflow split policy with small local PDF fixtures or injected extraction results; tests do not download external data.
- FinanceBench adapter IDs, page evidence mapping, null `question_reasoning` handling, taxonomy output, and rejection of malformed prepared input.
- CLI parsing and nonzero error handling for `prepare-financebench`.
- A configuration validation and smoke execution path using the fake provider, so CI never contacts the llama.cpp server.

Manual acceptance checks use the local llama.cpp server at `http://127.0.0.1:8080/v1`: run the smoke config after preparation, inspect the artifact manifest and retrieval page IDs, then calibrate `runtime.max_workers` before launching the full configuration.
