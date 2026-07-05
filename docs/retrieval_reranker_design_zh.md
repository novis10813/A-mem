# A-MEM Retrieval 與 Reranker 設計記錄

本文記錄本 fork 目前已實作的 retrieval 設計，重點是分清楚 memory construction、第一階段 retrieval、第二階段 reranking 與 QA context 之間的邊界。

## 1. 目前的 robust retrieval baseline

Robust baseline 指 `test_advanced_robust.py` 與 two-stage workflow 中的 `--qa-mode robust`。

資料流：

```text
question
  -> LLM 產生 query keywords
  -> SimpleEmbeddingRetriever.search(query_keywords, retrieve_k)
  -> 取回 memory notes
  -> render raw_context
  -> LLM answer prompt
```

特性：

- Memory construction 階段會建立 `RobustMemoryNote`，包含 `content`、`timestamp`、`context`、`keywords`、`tags`、`links`。
- Robust retriever indexing text 目前主要使用 `context + keywords`。
- QA context 會包含 `timestamp/content/context/keywords/tags`，並會附帶 linked neighbors。
- `retrieve_k` 是最後進入回答 prompt 的 memory 數量。

這是最接近原始 A-MEM robust evaluator 的 baseline。若 reranker 關閉，`--rerank-mode off`，行為應維持這個 baseline。

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

## 4. CrossEncoder reranker 設計

Reranker 目前只接在 robust retrieval path，模組入口是 `src/amem/reranking.py`。

資料流：

```text
question
  -> LLM 產生 query keywords
  -> embedding retriever 用 query_keywords 取 top rerank_top_n candidates
  -> CrossEncoder 對 (original question, candidate memory text) 打分
  -> 依 score 重排，取 top retrieve_k
  -> render raw_context
  -> LLM answer prompt
```

CLI 範例：

```bash
uv run python scripts/evaluate_memories.py \
  --experiment-id ollama_llama3.2-1b_none_rerank_k10 \
  --dataset data/locomo10.json \
  --backend ollama \
  --model llama3.2:1b \
  --qa-mode robust \
  --qa-runs 1 \
  --retrieve-k 10 \
  --rerank-mode cross_encoder \
  --rerank-top-n 50 \
  --rerank-batch-size 32 \
  --resume
```

預設：

- `--rerank-mode off`
- `--rerank-model cross-encoder/ms-marco-MiniLM-L6-v2`
- `--rerank-top-n 50`
- `--rerank-batch-size 32`

關鍵語義：

- `retrieve_k` 是最後放進 answer prompt 的 memory 數量。
- `rerank_top_n` 是第一階段 embedding similarity 的候選池大小。
- CrossEncoder 不生成文字，只對 `(question, memory_text)` pair 輸出 relevance score。
- tie-break 保留第一階段 similarity candidate 順序，讓分數相同時結果穩定。
- Reranker 在 QA evaluation 階段即時運作，不改 memory cache schema。

## 5. Result metadata

啟用 robust evaluation 時，每題結果會保留 retrieval metadata。

重要欄位：

| Field | 說明 |
|---|---|
| `query_keywords` | LLM 從 question 產生、用於第一階段 similarity retrieval 的 keywords |
| `retrieval_info.similarity_query` | 實際送進 embedding retriever 的 query |
| `retrieval_info.rerank_query` | CrossEncoder 使用的原始 question |
| `retrieval_info.candidate_k` | 第一階段候選池大小 |
| `retrieval_info.candidate_indices` | 第一階段 similarity 排序後的 candidate memory indices |
| `retrieval_info.final_indices` | reranker 或 baseline 最終選中的 memory indices |
| `retrieval_info.rerank_scores` | CrossEncoder score；reranker 關閉時為空 |
| `retrieval_info.rerank_mode` | `off` 或 `cross_encoder` |

這些欄位用來檢查 reranker 是否真的改變了 retrieval order，也能分析「第一階段有沒有召回正確記憶」與「第二階段是否把正確記憶往前排」。

## 6. BM25 first-stage retrieval 設計

BM25 是 robust QA evaluation 的可選第一階段 retriever，不屬於 memory construction，也不改 cache schema。

資料流：

```text
existing memory cache
  -> 從 memory notes 即時建立 BM25 index
  -> LLM 產生 query keywords
  -> BM25.search(query_keywords, retrieve_k)
  -> render robust raw_context
  -> LLM answer prompt
```

CLI 範例：

```bash
uv run python scripts/evaluate_memories.py \
  --experiment-id ollama_llama3.2-1b_none_bm25_k10 \
  --cache-experiment-id ollama_llama3.2-1b_none_base_cache \
  --dataset data/locomo10.json \
  --backend ollama \
  --model llama3.2:1b \
  --qa-mode robust \
  --qa-runs 30 \
  --retrieve-k 10 \
  --retrieval-mode bm25 \
  --rerank-mode off \
  --resume
```

語義：

- `retrieval_mode=embedding` 是既有 robust baseline。
- `retrieval_mode=bm25` 從同一份 memory cache 建 BM25 index，適合固定 cache 比較 retrieval effect。
- `cache_experiment_id` 指向讀取 cache 的 experiment；`experiment_id` 指向本次 evaluation result。
- 主比較建議使用 `--rerank-mode off`，避免第一階段 retrieval 和第二階段 reranking 效果混在一起。
- 若同時啟用 CrossEncoder，candidate pool 會先由 `retrieval_mode` 產生，再交給 CrossEncoder 重排。
- `retrieval_info.retrieval_mode` 會記錄 `embedding` 或 `bm25`。

## 7. 模組邊界

- `src/amem/reranking.py`：reranker abstraction、CrossEncoder implementation、factory。
- `src/amem/memory_layer_robust.py`：robust memory system 的 candidate selection、BM25 retriever、reranker integration、retrieval metadata。
- `test_advanced_robust.py`：legacy robust evaluator；保留 reranker flags 以維持單檔入口可用。
- `scripts/evaluate_memories.py`：two-stage QA evaluation；只在 evaluation 階段切換 retriever 或載入 reranker。
- `scripts/build_memories.py`：不應因 reranker 改動；reranker 不屬於 memory construction。
- `scripts/run_experiment.sh`：wrapper；retrieval/reranker flags 只傳給 evaluate command。
