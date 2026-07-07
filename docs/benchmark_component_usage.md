# Component Benchmark 使用與整合指南

本文說明 component benchmark migration 之後，這個 repo 應該怎麼跑既有 A-Mem 實驗、怎麼讀新的 normalized artifacts，以及未來整合其他 memory / RAG / graph 方法時應該遵守的原則。

目前狀態是漸進式 migration：

- 舊的 two-stage A-Mem build/evaluate CLI 仍是主要入口。
- build 階段會在既有 pickle cache 旁邊額外輸出 normalized `MemoryStore`。
- robust evaluate 階段會在既有 robust `results.json` 旁邊額外輸出 normalized QA result。
- dashboard data loader 會優先讀 normalized result，沒有 normalized result 時回退到舊 robust result。
- `src/amem/benchmark/` 已提供 schema、artifact IO、context/result helpers、registry/config skeleton、token usage hook。

## 正確使用方式

### 1. 建立 memories

照舊使用 `scripts/build_memories.py` 或 config-driven wrapper：

```bash
uv run python scripts/build_memories.py \
  --experiment-id ollama_llama3.2-1b_nltk \
  --dataset data/locomo10.json \
  --backend ollama \
  --model llama3.2:1b \
  --construction-runs 1 \
  --keyword-pruning-mode nltk \
  --max-workers 10 \
  --resume
```

舊 cache 仍會寫在：

```text
artifacts/caches/<experiment_id>/construction_run_00/
  memory_cache_sample_0.pkl
  retriever_cache_sample_0.pkl
  retriever_cache_embeddings_sample_0.npy
```

新增 normalized construction artifacts 會寫在：

```text
artifacts/caches/<experiment_id>/construction_run_00/normalized/
  memory_store_sample_0.json
  memory_records_sample_0.jsonl
  memory_nodes_sample_0.jsonl
  memory_edges_sample_0.jsonl
```

`memory_store_sample_<n>.json` 是跨方法比較的主要 artifact。A-Mem 目前會輸出：

- `records`: 每個 A-Mem note 的 flat text view，適合 embedding/BM25/RAG。
- `nodes`: 每個 note 對應一個 `amem_note` node。
- `edges`: A-Mem note links 對應 `amem_link` edge。
- `private_refs`: 指回舊 pickle/retriever cache，方便需要 method-private state 時回讀。

### 2. 評估 QA

照舊使用 `scripts/evaluate_memories.py`：

```bash
uv run python scripts/evaluate_memories.py \
  --experiment-id ollama_llama3.2-1b_nltk \
  --dataset data/locomo10.json \
  --backend ollama \
  --model llama3.2:1b \
  --qa-mode robust \
  --qa-runs 1 \
  --retrieve-k 10 \
  --resume
```

舊 robust result 仍會寫在：

```text
artifacts/results/<experiment_id>/construction_run_00/robust/qa_run_00/results.json
```

新增 normalized QA result 會寫在：

```text
artifacts/results/<experiment_id>/construction_run_00/robust/qa_run_00/normalized/
  results.json
  results.jsonl
  usage_summary.json
```

`results.json` 和 `results.jsonl` 使用 `QAResult` schema。每筆結果至少包含：

- `question`, `reference`, `prediction`, `category`, `metrics`
- `retrieval`: normalized retrieval trace or legacy robust `retrieval_info`
- `context`: answer prompt context text
- `prompt`: QA prompt
- `usage`: token/cost/latency usage records
- `metadata`: source format and method-specific metadata

目前 robust legacy normalization 會保留 retrieval/context/prompt/metrics，但 `usage` 只有在實際 LLM wrapper 或 adapter 呼叫 `TokenUsageHook` 時才會有資料。若某次結果的 `usage` 是空陣列，代表該 run 尚未接上 token accounting，而不是 token 消耗為 0。

### 3. Dashboard

`scripts/experiment_data_loader.py` 會優先讀：

```text
artifacts/results/<experiment_id>/construction_run_00/robust/qa_run_00/normalized/results.json
```

如果 normalized result 不存在，才回退到舊路徑：

```text
artifacts/results/<experiment_id>/construction_run_00/robust/qa_run_00/results.json
```

因此既有 Gradio dashboard 可以漸進式支援新格式，不需要一次搬掉舊 results。

### 4. 用 Python 讀 normalized artifacts

讀 memory store：

```python
from pathlib import Path

from amem.benchmark.artifacts import read_memory_store

store = read_memory_store(
    Path("artifacts/caches/ollama_llama3.2-1b_nltk")
    / "construction_run_00"
    / "normalized"
    / "memory_store_sample_0.json"
)

print(store.records[0].text)
print(store.nodes[0].node_type)
```

讀 normalized QA rows：

```python
from pathlib import Path

from amem.benchmark.artifacts import read_jsonl

rows = read_jsonl(
    Path("artifacts/results/ollama_llama3.2-1b_nltk")
    / "construction_run_00"
    / "robust"
    / "qa_run_00"
    / "normalized"
    / "results.jsonl"
)

print(rows[0]["question"])
print(rows[0]["metrics"])
```

## 新 schema 的角色

### `MemoryStore`

`MemoryStore` 是 construction 階段的跨方法輸出。它不是只給 graph 方法用，也不是只給 RAG 用。

- 一般 chunked RAG 可以只填 `records`。
- A-Mem 可以填 `records + nodes + edges`。
- MRAgent / Zep / Graphiti 類方法應該填 `nodes + edges + layers`，並視需要提供 `records` 作為 flat retrieval baseline。

重要原則：graph 方法的 canonical artifact 應該是 graph store，不要被迫壓成 flat records。flatten graph into records 可以做，但要在 config/result 裡明確標成 baseline view。

### `MemoryRecord`

`MemoryRecord` 是 flat text view，適合：

- embedding retrieval
- BM25 retrieval
- CrossEncoder reranking
- 非 graph RAG
- dashboard 快速顯示 retrieved evidence

Graph 方法可以輸出 records，但 records 應該是 derived view，不應取代 nodes/edges/layers。

### `QAResult`

`QAResult` 是 evaluation 階段的跨方法輸出。無論是 one-shot RAG、A-Mem robust QA，或 tool-calling graph QA，都應該最後落到這個 schema，讓 dashboard 和分析工具可以用同一套欄位比較。

## Token accounting

Token/cost/latency accounting 是 observability hook，不是 method adapter。

使用原則：

- 不應把 token accounting 寫進 construction/retrieval/QA 方法語意。
- 若 provider 回傳 usage，寫 `source: "reported"`。
- 若本地模型或 API 沒有 usage，才用 tokenizer/估算器，並寫 `source: "estimated"`。
- aggregate table 不要把 reported 和 estimated 混在一起報，至少要保留 `source` 分組。

基本 API：

```python
from amem.benchmark.hooks import HookContext, TokenUsageHook

hook = TokenUsageHook(estimate_when_missing=True, tokenizer="words")
hook.after_llm_call(
    HookContext(phase="qa", sample_id=0, qa_idx=3),
    call_id="answer",
    provider="ollama",
    model="llama3.2:1b",
    prompt="question and context",
    completion="answer",
)

print(hook.records[0].source)  # estimated
```

實作新的 LLM wrapper 或 QA adapter 時，應該在每次 LLM call 後呼叫 hook。tool-calling QA 要每個 LLM planning/answer step 都記 usage，不能只記最後答案。

## 其他實驗的 integration 原則

### 1. 先決定三個正式比較軸

每個實驗都應該明確標出：

- `construction`: 如何從 conversation 建 memory。
- `retrieval`: 如何從 memory 取得 evidence 或 memory access tools。
- `qa`: 如何根據 question 和 memory access 產生 answer。

低階差異可以用 hooks 或 stage params，但如果它會影響論文結論，就必須寫進 config、manifest 或 result metadata。例如：

- reranker 有無啟用
- query 用 original question 還是 generated keywords
- context 是否包含 links / graph path / temporal filter
- token usage 是 reported 還是 estimated

### 2. Construction integration

新增 construction 方法時，至少要能輸出 `MemoryStore`：

```python
from amem.benchmark.schemas import MemoryRecord, MemoryStore

store = MemoryStore(
    sample_id=sample_id,
    records=(
        MemoryRecord(
            memory_id="chunk-0",
            sample_id=sample_id,
            text="memory text used for retrieval",
            content="original content",
            keywords=("keyword",),
            metadata={"source_method": "my_rag"},
        ),
    ),
    metadata={"construction_adapter": "my_rag"},
)
```

如果方法有 private state，例如 graph database、pickle object、vector index、FAISS index，放在 method-private artifact，並用 `private_refs` 指回去：

```python
store = MemoryStore(
    sample_id=sample_id,
    records=records,
    private_refs={"faiss_index": "private/sample_0.faiss"},
)
```

不要把大型 binary state 塞進 `memory_store_sample_<n>.json`。

### 3. Graph / temporal graph integration

Graph 方法應該使用：

- `MemoryNode`: entity、event、episode、topic、community、note 等節點。
- `MemoryEdge`: relation、mentions、source、temporal relation、community membership 等邊。
- `MemoryLayer`: episode / semantic entity / personal event / topic / community 等層。

Temporal graph 請把時間語意放進標準欄位或 metadata：

- node-level `timestamp`
- edge-level `valid_at`, `invalid_at`
- `properties` / `metadata` 補充 method-specific temporal semantics

範例：

```python
from amem.benchmark.schemas import MemoryEdge, MemoryLayer, MemoryNode, MemoryStore

store = MemoryStore(
    sample_id=sample_id,
    nodes=(
        MemoryNode(node_id="episode-1", node_type="episode", text="Alice moved to Taipei."),
        MemoryNode(node_id="entity-alice", node_type="entity", label="Alice"),
    ),
    edges=(
        MemoryEdge(
            edge_id="edge-1",
            source_id="entity-alice",
            target_id="episode-1",
            edge_type="mentioned_in",
            valid_at="2026-01-01T10:00:00",
        ),
    ),
    layers=(MemoryLayer(name="episode", node_ids=("episode-1",), edge_ids=("edge-1",)),),
    metadata={"construction_adapter": "temporal_graph"},
)
```

### 4. Retrieval integration

One-shot retrieval 應該輸出 `RetrievedItem`：

```python
from amem.benchmark.schemas import RetrievedItem

item = RetrievedItem(
    item_id="chunk-0",
    rank=1,
    text="retrieved evidence",
    item_type="record",
    score=0.82,
    source_stage="embedding",
    metadata={"query": "original_question"},
)
```

Tool-calling / graph retrieval 應該把每次 tool call 記進 `retrieval.tool_calls`：

```python
from amem.benchmark.schemas import RetrievalToolCall

tool_call = RetrievalToolCall(
    tool_name="query_event_keywords",
    arguments={"keywords": ["taipei"]},
    output_items=(item,),
    output_text="retrieved graph evidence",
)
```

原則：

- retrieval algorithm 是正式 adapter 或 explicit stage，不要藏在 hook 裡。
- graph traversal path、node IDs、edge IDs、temporal filters 應該放進 `RetrievedItem.metadata` 或 `RetrievalToolCall.metadata`。
- 如果把 graph flatten 成 records 做 embedding/BM25，config/result 要明確標 `view: records_from_graph` 或類似 metadata。

### 5. QA integration

新增 QA 方法時，最後要輸出 `QAResult`。最小欄位：

```python
from amem.benchmark.schemas import QAResult

result = QAResult(
    experiment_id=experiment_id,
    construction_run=construction_run,
    qa_run=qa_run,
    sample_id=sample_id,
    qa_idx=qa_idx,
    question=question,
    reference=reference,
    prediction=prediction,
    category=category,
    metrics=metrics,
    retrieval={"items": [item_dict], "tool_calls": []},
    context={"text": context_text},
    prompt=prompt,
    usage=usage_records,
    metadata={"qa_adapter": "my_qa"},
)
```

One-shot QA 可以先 build context 再 answer。Tool-calling QA 可以在 QA loop 中呼叫 retrieval tools，但最後仍要把 tool calls、retrieved support、prompt/context 和 final answer 寫回 `QAResult`。

### 6. Result 與 dashboard integration

新的 evaluation output 優先寫 normalized result：

```text
artifacts/results/<experiment_id>/construction_run_XX/<mode>/qa_run_XX/normalized/
  results.json
  results.jsonl
  usage_summary.json
```

Dashboard loader 目前 robust mode 會優先讀 normalized result。若新增非 robust mode，需要同步擴充 discovery/loading 邏輯，不要只寫 method-specific viewer。

Dashboard 應該盡量依賴這些共通欄位：

- `question_key`
- `question`, `reference`, `prediction`, `metrics`
- `retrieval.items`
- `retrieval.tool_calls`
- `context.text`
- `usage`
- `metadata`

## 新方法整合 checklist

新增一個新實驗方法時，請依序確認：

1. Construction 是否輸出 `MemoryStore`。
2. Method-private artifacts 是否只放在 private path，並由 `private_refs` 指向。
3. Graph 方法是否保留 nodes/edges/layers，而不是只輸出 flat records。
4. Retrieval 是否輸出 `RetrievedItem` 或 `RetrievalToolCall` trace。
5. QA 是否輸出 `QAResult`。
6. Token accounting 是否透過 `TokenUsageHook` 或等價 hook 記錄，並標明 reported/estimated。
7. Result 是否寫到 normalized result path。
8. Dashboard loader 是否能在不理解 method internals 的情況下讀出比較所需欄位。
9. Config/manifest/metadata 是否能看出這次比較的 construction、retrieval、QA、context、token accounting 條件。

## 驗證建議

文件或 schema 相關變更：

```bash
uv run python -m pytest tests/test_reproduction_package.py -v
```

Benchmark foundation 變更：

```bash
uv run python -m pytest \
  tests/test_benchmark_schemas.py \
  tests/test_benchmark_hooks.py \
  tests/test_benchmark_artifacts.py \
  tests/test_benchmark_config.py \
  tests/test_amem_serialization.py \
  tests/test_benchmark_results.py \
  tests/test_experiment_data_loader.py \
  -v
```

Build/evaluate integration 變更：

```bash
uv run python -m py_compile scripts/build_memories.py scripts/evaluate_memories.py scripts/experiment_data_loader.py
uv run python -m pytest tests/test_experiment_entrypoints.py tests/test_experiment_data_loader.py -v
```

完整 smoke run 仍建議用小 limits，避免直接啟動長實驗：

```bash
uv run python scripts/build_memories.py \
  --experiment-id smoke_component_usage \
  --dataset data/locomo10.json \
  --backend ollama \
  --model llama3.2:1b \
  --construction-runs 1 \
  --keyword-pruning-mode nltk \
  --sample-limit 1 \
  --turn-limit 2 \
  --max-workers 1 \
  --resume

uv run python scripts/evaluate_memories.py \
  --experiment-id smoke_component_usage \
  --dataset data/locomo10.json \
  --backend ollama \
  --model llama3.2:1b \
  --qa-mode robust \
  --qa-runs 1 \
  --sample-limit 1 \
  --qa-limit 1 \
  --max-workers 1 \
  --resume
```
