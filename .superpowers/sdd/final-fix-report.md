# FinanceBench Final-Review Fix Report

## Scope

Addressed the final review findings from base `ae45fd5` in one focused change:

- Enforce the binding public scope of 150 unique FinanceBench IDs across 84 document names before any PDF download, and reject duplicate IDs.
- Enforce the same invariant when loading a prepared FinanceBench dataset.
- Raise `ValueError` for a syntactically valid, non-object manifest.
- Retain question-derived `required_documents` in a failed manifest when metadata validation fails before per-document reports are initialized.

## Design choice: explicit injected-fixture seam

`FinanceBenchScope(question_count, document_count)` centralizes duplicate and cardinality validation. `PUBLIC_FINANCEBENCH_SCOPE` is fixed at 150/84 and is the default for both `prepare_financebench` and `FinanceBenchAdapter`, so the CLI and registered production adapter stay strict. Tests deliberately pass a small `FinanceBenchScope` only through their injected I/O/adapter construction path. The fake runner test temporarily registers an adapter instance with that explicit fixture scope. This keeps the production default strict without changing the prepared JSON schema.

## TDD evidence

### RED

After adding the focused regressions and before production changes:

```bash
uv run python -m pytest tests/test_memorybench_financebench.py -k 'scope or non_object_manifest or failed_manifest_preserves' -v
```

Result: `8 failed, 16 deselected`. The default adapter accepted the undersized prepared fixture; `FinanceBenchScope` and the preparation `scope` seam did not exist; a list manifest did not produce the required `ValueError`; and early metadata failure did not preserve the requested manifest provenance.

### GREEN

After implementation, the same focused command reported:

```text
8 passed, 16 deselected in 0.07s
```

It covers duplicate IDs, missing/extra question and document counts before PDF requests, adapter default/public-scope rejection, prepared-loader duplicate-ID rejection, non-object manifest handling, and failed-manifest required-document provenance.

## Files changed

- `src/memorybench/datasets/financebench.py`
  - Added `FinanceBenchScope` and strict `PUBLIC_FINANCEBENCH_SCOPE` defaults.
  - Validates prepared document names and question IDs, including duplicates.
  - Rejects non-object manifests with `ValueError`.
- `src/memorybench/datasets/financebench_prepare.py`
  - Validates the question-derived scope before metadata/PDF work.
  - Carries question-derived `required_documents` into both completed and failed manifests.
- `tests/test_memorybench_financebench.py`
  - Adds review regressions and uses the explicit small fixture scope for existing injected fixtures.

## Verification

```bash
uv run python -m pytest tests/test_memorybench_financebench.py -v
# 24 passed in 0.26s

uv run python -m pytest -v
# 60 passed in 0.46s

uv run python -m compileall -q src/memorybench tests
# exit 0; no output

uv lock --check
# Resolved 131 packages in 0.37ms

git diff --check
# exit 0; no output
```

## Self-review

- Scope validation runs immediately after question parsing and before metadata validation, download, extraction, or normalization.
- The default production path remains 150/84; only test calls explicitly opt into small scopes.
- The adapter does not import PDF/crypto dependencies and the prepared payload schema is unchanged.
- Existing revision-bound resume behavior, local-only loading, diagnostic metrics/configuration, and lazy `pypdf`/`cryptography` behavior were not modified.
- No README, configuration, lockfile, or artifact files changed.

## Concerns

None. The public scope is cardinality-and-uniqueness bound as requested; fixture-size overrides are intentionally explicit and test-only.
