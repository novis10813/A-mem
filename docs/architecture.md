# MemoryBench architecture

The canonical flow is `Dataset → Construction → Retrieve/QA → Artifacts → Dashboard`.

`memorybench.config` validates a fully stage-local YAML configuration with Pydantic v2 and
computes a stable SHA-256 fingerprint. Dataset adapters expose native taxonomy rather than a
global hard-coded label set. LoCoMo IDs use `locomo:<sample>:<question>` and turn evidence IDs
remain traceable to the source dialogue.

Construction emits graph-capable `MemoryStore` objects. A store may contain records, nodes,
edges, layers, and references to method-private state. Turn-RAG uses a replaceable `Chunker`;
the native A-Mem core remains under `memorybench.amem_native`, while normalized conversion is
owned by `memorybench.methods.amem`.

One-shot retrieval is an ordered stage pipeline. Each trace records its query, input/output
ranking, scores, effective config, and elapsed time. The separate `InteractiveRetrievalAdapter`
protocol reserves agent/tool retrieval without pretending it is a linear ranker.

Canonical artifacts are JSON/JSONL below `artifacts/experiments/<experiment_id>`. Writes are
atomic. A completed unit can resume only when its status marker and config fingerprint match.
Failed questions are data rows, not log-only side effects. Exit codes are 0 completed, 1 fatal,
and 2 partial.

The dashboard reads only canonical artifacts. Any future `analysis_cache/` index is disposable
and must never become the source of truth.
