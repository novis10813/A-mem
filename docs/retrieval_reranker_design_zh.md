# A-MEM Retrieval 與 Reranker 設計記錄

本文記錄本 fork 目前已實作的 retrieval 設計，重點是分清楚 memory construction、第一階段 retrieval、第二階段 reranking 與 QA context 之間的邊界。

## 1. 目前的 robust retrieval baseline

Robust baseline 指 `test_advanced_robust.py` 與 two-stage workflow 中的 `--qa-mode robust`。

資料流：

```text
question
  -> LLM 產生 query keywords
  -> retrieval_pipeline 第一階段產生 candidates
  -> optional rerank stages 重新排序 candidates
  -> 取 top retrieval_pipeline.final_k memory notes
  -> 取回 memory notes
  -> render raw_context
  -> LLM answer prompt
```

特性：

- Memory construction 階段會建立 `RobustMemoryNote`，包含 `content`、`timestamp`、`context`、`keywords`、`tags`、`links`。
- Robust retriever indexing text 目前主要使用 `context + keywords`。
- QA context 會包含 `timestamp/content/context/keywords/tags`，並會附帶 linked neighbors。
- `evaluation.retrieval_pipeline.final_k` 是最後進入回答 prompt 的 memory 數量。

這是最接近原始 A-MEM robust evaluator 的 baseline。若 pipeline 只有一個 `embedding` stage，行為就是原本的 embedding robust baseline。

## 2. Keyword pruning modes

Keyword pruning 由 `src/amem/llm_text_parsers.py` 控制。

| Mode | 語義 | 用途 |
|---|---|---|
| `none` | 只 normalize、deduplicate、cap 到 `max_keywords`；不做 grounding 或 stopword filtering | 原始 LLM keyword baseline/control |
| `simple` | 規則過濾與 content grounding，但不使用 stemming | 分離 rule filtering 的效果 |
| `nltk` | 規則過濾、content grounding，加 PorterStemmer | 目前較嚴格的 keyword pruning 設計 |

要注意：construction 階段的 `--keyword-pruning-mode` 會影響寫入 memory cache 的 keywords。content-keyword 評估中的 `--keyword-conditions none,nltk` 則是在同一份 cache 上重新 transform stored keywords，用來做 fixed-cache 對照。

## 3. Content+keywords fixed-cache retrieval

`scripts/run_content_keyword_pruning_experiment.py` 和 two-stage `--qa-mode content_keywords` 是一條獨立的 retrieval 實驗路徑。

資料流：

```text
existing memory cache
  -> transform stored keywords by condition
  -> embedding document = content + keywords
  -> query keywords embedding search
  -> context = timestamp + content + keywords
  -> LLM answer prompt
```

特性：

- 不使用 robust memory system 的 `find_related_memories_raw()`。
- 不包含 `context/tags/links` 到 retrieval document 或 answer context。
- 適合隔離比較 keyword transformation 對 retrieval/QA 的影響。
- fixed-cache 版本不重建 memories；rebuild 版本會按 pruning condition 重建 memories。

因此 `content_keywords` 結果不能直接等同於 robust retrieval 結果；兩者的 context 欄位與 retrieval document 都不同。

## 4. Robust retrieval pipeline 設計

Robust retrieval 已改成 config-first 的多 stage pipeline，模組入口是 `src/amem/retrieval_pipeline.py`。v1 只套用在 `qa_mode: robust`；`content_keywords` 保持原本獨立路徑。

所有 stage 共用同一個介面：

```text
RetrievalRequest + list[MemoryCandidate] -> list[MemoryCandidate]
```

目前支援的 stage：

| Stage type | 角色 | 預設 query |
|---|---|---|
| `embedding` | 第一階段 candidate generator，使用 cached embedding retriever | `similarity_query` |
| `bm25` | 第一階段 candidate generator，從 memory cache 即時建 BM25 index | `similarity_query` |
| `embedding_rerank` | 對輸入 candidates 使用 cached embeddings 做 cosine similarity 重排 | `similarity_query` |
| `bm25_rerank` | 對輸入 candidates 建臨時 BM25 index 後重排 | `original_question` |
| `cross_encoder` | 用 CrossEncoder 對 `(question, candidate memory text)` 打分重排 | `original_question` |
| `limit` | 中途裁切 candidate pool | 不使用 query |

第一個 stage 必須是 `embedding` 或 `bm25`。`final_k` 是最後放進 answer prompt 的 memory 數量；各 stage 的 `top_k` 是該 stage 輸出的候選數量。

## 5. 實驗執行方式

新的 robust pipeline 實驗建議使用 YAML config 和 `scripts/run_experiment.py`。不要用舊的 `scripts/run_experiment.sh --retrieval-mode/--rerank-mode` 跑 robust pipeline。

先確認 Ollama：

```bash
ollama serve
ollama pull llama3.2:1b
curl -sf http://localhost:11434/api/tags
```

### Embedding -> BM25 rerank

範例 config：`configs/robust_bm25_rerank.yaml`

```yaml
experiment_id: ollama_llama3.2-1b_robust_embed_bm25_k10
dataset: data/locomo10.json

backend:
  name: ollama
  model: llama3.2:1b

construction:
  runs: 1
  keyword_pruning_mode: none
  embedding_model: all-MiniLM-L6-v2
  max_workers: 10

evaluation:
  qa_mode: robust
  qa_runs: 1
  retrieval_pipeline:
    final_k: 10
    stages:
      - type: embedding
        name: embedding_candidates
        top_k: 50
        query: similarity_query
      - type: bm25_rerank
        name: bm25_rerank
        top_k: 10
        query: original_question

limits:
  ratio: 1.0

run:
  resume: true
```

執行：

```bash
uv run python scripts/run_experiment.py --config configs/robust_bm25_rerank.yaml --resume
```

輸出：

```text
artifacts/caches/ollama_llama3.2-1b_robust_embed_bm25_k10/
artifacts/results/ollama_llama3.2-1b_robust_embed_bm25_k10/
artifacts/logs/ollama_llama3.2-1b_robust_embed_bm25_k10/
```

### Embedding -> CrossEncoder rerank

資料流：

```text
question
  -> LLM 產生 query keywords
  -> embedding stage 用 query_keywords 取 top 50 candidates
  -> CrossEncoder 對 (original question, candidate memory text) 打分
  -> 依 score 重排，取 top final_k
  -> render raw_context
  -> LLM answer prompt
```

config 片段：

```yaml
evaluation:
  qa_mode: robust
  qa_runs: 1
  retrieval_pipeline:
    final_k: 10
    stages:
      - type: embedding
        name: embedding_candidates
        top_k: 50
      - type: cross_encoder
        name: cross_encoder_rerank
        top_k: 10
        model: cross-encoder/ms-marco-MiniLM-L6-v2
        batch_size: 32
```

關鍵語義：

- `final_k` 是最後放進 answer prompt 的 memory 數量。
- `embedding.top_k` 是第一階段 embedding similarity 的候選池大小。
- CrossEncoder 不生成文字，只對 `(question, memory_text)` pair 輸出 relevance score。
- tie-break 保留第一階段 similarity candidate 順序，讓分數相同時結果穩定。
- Reranker 在 QA evaluation 階段即時運作，不改 memory cache schema。

### 固定同一份 cache 比較不同 pipeline

如果只想換 retrieval pipeline，不想重建 memories，使用 `evaluation.cache_experiment_id` 指向既有 cache，並只跑 evaluation：

```yaml
experiment_id: compare_embed_bm25
dataset: data/locomo10.json

backend:
  name: ollama
  model: llama3.2:1b

construction:
  runs: 1
  keyword_pruning_mode: none
  embedding_model: all-MiniLM-L6-v2
  max_workers: 10

evaluation:
  cache_experiment_id: ollama_llama3.2-1b_robust_embed_bm25_k10
  qa_mode: robust
  qa_runs: 1
  retrieval_pipeline:
    final_k: 10
    stages:
      - type: embedding
        name: embedding_candidates
        top_k: 50
      - type: bm25_rerank
        name: bm25_rerank
        top_k: 10
```

```bash
uv run python scripts/evaluate_memories.py --config configs/compare_embed_bm25.yaml --resume
```

`scripts/run_experiment.py` 會 build 再 evaluate；`scripts/evaluate_memories.py --config` 只做 QA evaluation。

## 6. Result metadata

啟用 robust evaluation 時，每題結果會保留 retrieval metadata。

重要欄位：

| Field | 說明 |
|---|---|
| `retrieval_info.schema_version` | retrieval metadata schema；pipeline 版為 `3` |
| `query_keywords` | LLM 從 question 產生、用於第一階段 similarity retrieval 的 keywords |
| `retrieval_info.similarity_query` | 實際送進第一階段 generator 的 similarity query |
| `retrieval_info.original_question` | rerank stage 通常使用的原始 question |
| `retrieval_info.final_k` | 最終進入 answer prompt 的 memory 數量 |
| `retrieval_info.stages` | 每個 stage 的 `name/type/query/top_k/input_count/output_count` |
| `retrieval_info.candidates` | 最終 stage 的完整 candidate records |
| `retrieval_info.selected` | 最終選中並進 prompt 的 candidate records；分析應優先使用此欄位 |
| `retrieval_info.selected[].scores` | 各 stage 寫入的 score，例如 `bm25_rerank` 或 `cross_encoder_rerank` |
| `retrieval_info.selected[].ranks` | 各 stage 寫入的 rank |
| `retrieval_info.selected[].stage_trace` | candidate 通過 stage 的排序軌跡 |

這些欄位用來檢查 reranker 是否真的改變了 retrieval order，也能分析「第一階段有沒有召回正確記憶」與「第二階段是否把正確記憶往前排」。

## 7. BM25 first-stage retrieval 設計

BM25 可以作為 robust QA evaluation 的第一階段 generator，也可以作為第二階段 reranker。不屬於 memory construction，也不改 cache schema。

BM25 first-stage config：

```yaml
evaluation:
  qa_mode: robust
  retrieval_pipeline:
    final_k: 10
    stages:
      - type: bm25
        name: bm25_candidates
        top_k: 10
        query: similarity_query
```

- `type: bm25` 從同一份 memory cache 即時建 BM25 index，適合固定 cache 比較 retrieval effect。
- `cache_experiment_id` 指向讀取 cache 的 experiment；`experiment_id` 指向本次 evaluation result。
- 若要隔離 first-stage retrieval 效果，pipeline 只放一個 generator stage。
- 若要測試第二階段排序效果，使用 `embedding -> bm25_rerank`、`embedding -> cross_encoder`，或 `bm25 -> embedding_rerank -> cross_encoder`。

## 8. 模組邊界

- `src/amem/retrieval_pipeline.py`：pipeline abstraction、candidate generators、embedding/BM25 reranker、CrossEncoder stage、canonical `MemoryCandidate`。
- `src/amem/reranking.py`：CrossEncoder reranker implementation 與 factory。
- `src/amem/memory_layer_robust.py`：robust memory system 呼叫 retrieval pipeline、render context、寫 retrieval metadata。
- `test_advanced_robust.py`：legacy robust evaluator；保留 reranker flags 以維持單檔入口可用。
- `scripts/evaluate_memories.py`：two-stage QA evaluation；從 YAML 建立 retrieval pipeline。
- `scripts/build_memories.py`：不應因 reranker 改動；reranker 不屬於 memory construction。
- `scripts/run_experiment.py`：config-first wrapper；先 build，再 evaluate。
- `scripts/run_experiment.sh`：legacy wrapper；不要用於新的 robust pipeline 實驗。
