# A-MEM Baseline 與設計差異對照

本文整理目前 repo 中常用 baseline 與已實作設計的差異。比較實驗時請先確認改動發生在 construction、retrieval、reranking 還是 answer context。

## 1. Baseline matrix

| 名稱 | 入口 | Memory construction | Retrieval document | QA context | Rerank | 適合回答的問題 |
|---|---|---|---|---|---|---|
| Legacy robust baseline | `test_advanced_robust.py` | 依 `--keyword_pruning_mode` 建 memories | robust retriever 內部 text，主要是 `context + keywords` | `timestamp/content/context/keywords/tags` 加 linked neighbors | 無 | robust A-MEM 在某模型與某 `retrieve_k` 下的整體 QA 表現 |
| Two-stage robust baseline | `scripts/build_memories.py` + `scripts/evaluate_memories.py --qa-mode robust` | build 階段一次建 cache | 讀 cache retriever；缺 retriever cache 時從 memory 重建 | 同 robust baseline | 預設無 | 固定 cache 下分離 construction variance 與 QA variance |
| Keyword pruning construction baseline | `scripts/build_memories.py --keyword-pruning-mode none/simple/nltk` | 建 cache 時決定 stored keywords | 影響 robust retriever 與後續 cache contents | 依 stored memory note | 無或可搭配 rerank | construction-time keyword pruning 對整體系統的影響 |
| Content+keywords fixed-cache | `scripts/evaluate_memories.py --qa-mode content_keywords` | 不重建，讀既有 cache | `content + transformed keywords` | 只含 `timestamp/content/keywords` | 無 | 同一份 memory notes 上，keyword transform 對 retrieval/QA 的影響 |
| Content+keyword pruning legacy | `scripts/run_content_keyword_pruning_experiment.py` | 不重建，讀 legacy cache | `content + transformed keywords` | 只含 `timestamp/content/keywords` | 無 | 舊版 fixed-cache pruning 分析 |
| Content+keyword rebuild legacy | `scripts/run_content_keyword_rebuild_experiment.py` | 每個 pruning condition 重建 memories | `content + keywords` | 只含 `timestamp/content/keywords` | 無 | construction + retrieval + QA 的 system-level pruning 效果 |
| Robust CrossEncoder rerank | `scripts/evaluate_memories.py --qa-mode robust --rerank-mode cross_encoder` | 不因 reranker 重建 | 第一階段同 robust similarity candidate pool | 最終 top `retrieve_k` robust context | CrossEncoder | 第二階段 reranking 是否改善 robust retrieval 排序 |

## 2. 不要混淆的設定

### `retrieve_k` vs `rerank_top_n`

- `retrieve_k`：最後進入 answer prompt 的 memory 數量。
- `rerank_top_n`：reranker 啟用時，第一階段 embedding similarity 取出的候選池大小。

CrossEncoder reranker 的典型設定是：

```text
embedding top 50 candidates -> CrossEncoder rerank -> final top 10 context
```

對應 CLI：

```bash
--rerank-top-n 50 --retrieve-k 10
```

### Construction pruning vs QA keyword conditions

- `build_memories.py --keyword-pruning-mode` 會影響 memory cache 裡 stored keywords。
- `evaluate_memories.py --keyword-conditions` 只用於 `content_keywords` mode，在同一份 cache 上轉換 keywords。

如果要比較 construction-time pruning，應該建立不同 experiment id 的 cache。  
如果要比較 fixed-cache keyword transformation，應該用同一份 cache 跑 `content_keywords` conditions。

### Robust mode vs content_keywords mode

Robust mode 使用 robust memory system，context 較完整，並可包含 linked neighbors。  
Content_keywords mode 是刻意受限的 ablation，只使用 timestamp/content/keywords。

因此：

- robust vs robust rerank 可以看 reranking 是否改善同一 robust retrieval pipeline。
- content_keywords none vs nltk 可以看 keyword transform 是否改善受限 context 的 retrieval。
- robust vs content_keywords 不是單一因素比較，因為 retrieval document、context 欄位和 neighborhood behavior 都不同。

## 3. 目前 reranker 實驗設定範例

目前啟動過的 none-cache + robust reranker 實驗：

```text
tmux session: amem_none_rerank
experiment_id: ollama_llama3.2-1b_none_rerank_k10
```

命令：

```bash
PYTHONUNBUFFERED=1 bash scripts/run_experiment.sh \
  --experiment-id ollama_llama3.2-1b_none_rerank_k10 \
  --backend ollama \
  --model llama3.2:1b \
  --construction-runs 1 \
  --qa-runs 1 \
  --qa-mode robust \
  --keyword-pruning-mode none \
  --retrieve-k 10 \
  --rerank-mode cross_encoder \
  --rerank-top-n 50 \
  --rerank-batch-size 32 \
  --max-workers 10 \
  --resume
```

語義：

- build 階段：用 `keyword_pruning_mode=none` 建一份新的 robust memory cache。
- evaluate 階段：只跑 `robust` QA。
- 第一階段：用 LLM 產生的 query keywords 做 embedding similarity，取 top 50。
- 第二階段：用 `cross-encoder/ms-marco-MiniLM-L6-v2` 對原始 question 和 candidate memory text 打分。
- 最後：取 reranked top 10 放進 answer prompt。

輸出位置：

```text
artifacts/caches/ollama_llama3.2-1b_none_rerank_k10/
artifacts/results/ollama_llama3.2-1b_none_rerank_k10/
artifacts/logs/ollama_llama3.2-1b_none_rerank_k10/
```

## 4. 建議比較組合

### Reranker 效果

固定同一份 `keyword_pruning_mode=none` cache，比較：

```bash
--qa-mode robust --rerank-mode off --retrieve-k 10
--qa-mode robust --rerank-mode cross_encoder --rerank-top-n 50 --retrieve-k 10
```

這個比較主要回答：第二階段 reranker 是否改善 final memory ordering。

### Keyword pruning 效果

固定同一份 cache，比較：

```bash
--qa-mode content_keywords --keyword-conditions none,nltk
```

這個比較主要回答：在受限的 `content + keywords` retrieval document 下，keyword pruning 是否改善 retrieval/QA。

### Construction variance

使用：

```bash
--construction-runs 30 --qa-runs 1
```

這個比較主要回答：不同 memory construction run 對結果的變異有多大。

### QA variance

使用：

```bash
--construction-runs 1 --qa-runs 30
```

這個比較主要回答：同一份 memory cache 下，QA answering 的隨機性有多大。
