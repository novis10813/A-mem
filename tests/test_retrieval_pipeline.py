from types import SimpleNamespace

from amem.retrieval_pipeline import (
    BM25Reranker,
    CrossEncoderRerankerStage,
    EmbeddingCandidateGenerator,
    EmbeddingRerankerStage,
    RetrievalPipeline,
    RetrievalRequest,
)


class FakeRetriever:
    def __init__(self, indices):
        self.indices = indices
        self.calls = []

    def search(self, query, k):
        self.calls.append((query, k))
        return self.indices[:k]


class FakeReranker:
    mode = "cross_encoder"

    def __init__(self, ordered):
        self.ordered = ordered
        self.calls = []

    def rerank(self, query, candidates, top_k):
        self.calls.append((query, candidates, top_k))
        scores = {index: float(len(self.ordered) - rank) for rank, index in enumerate(self.ordered)}
        return [
            SimpleNamespace(index=index, score=scores[index])
            for index in self.ordered
            if index in {candidate.memory_index for candidate in candidates}
        ][:top_k]


class FakeEmbeddingModel:
    def encode(self, texts):
        vectors = {
            "museum": [1.0, 0.0],
        }
        return [vectors.get(text, [0.0, 1.0]) for text in texts]


class FakeEmbeddingRetriever(FakeRetriever):
    def __init__(self, indices, embeddings):
        super().__init__(indices)
        self.model = FakeEmbeddingModel()
        self.embeddings = embeddings


def memory_text(memory):
    return memory.text


def make_memories():
    return [
        SimpleNamespace(id="m0", text="garden cooking"),
        SimpleNamespace(id="m1", text="museum tickets"),
        SimpleNamespace(id="m2", text="museum cafe"),
    ]


def test_embedding_generator_creates_ranked_candidates():
    memories = make_memories()
    retriever = FakeRetriever([2, 0, 1])
    stage = EmbeddingCandidateGenerator(
        top_k=2,
        retriever=retriever,
        memories=memories,
        memory_text=memory_text,
    )

    candidates = stage.run(RetrievalRequest("museum", None, 2), [])

    assert retriever.calls == [("museum", 2)]
    assert [candidate.memory_index for candidate in candidates] == [2, 0]
    assert candidates[0].ranks == {"embedding_candidates": 1}
    assert candidates[0].source_stage == "embedding_candidates"
    assert candidates[0].stage_trace[0]["stage"] == "embedding_candidates"


def test_bm25_reranker_uses_original_question():
    memories = make_memories()
    candidates = EmbeddingCandidateGenerator(
        top_k=3,
        retriever=FakeRetriever([0, 1, 2]),
        memories=memories,
        memory_text=memory_text,
    ).run(RetrievalRequest("irrelevant", "museum tickets", 2), [])

    reranked = BM25Reranker(top_k=2).run(
        RetrievalRequest("irrelevant", "museum tickets", 2),
        candidates,
    )

    assert [candidate.memory_index for candidate in reranked] == [1, 2]
    assert "bm25_rerank" in reranked[0].scores
    assert reranked[0].ranks["bm25_rerank"] == 1


def test_embedding_reranker_scores_existing_candidates():
    memories = make_memories()
    candidates = [
        EmbeddingCandidateGenerator(
            top_k=3,
            retriever=FakeRetriever([0, 1, 2]),
            memories=memories,
            memory_text=memory_text,
        ).run(RetrievalRequest("irrelevant", None, 3), [])[index]
        for index in [0, 1, 2]
    ]
    retriever = FakeEmbeddingRetriever([0, 1, 2], [[0.0, 1.0], [1.0, 0.0], [0.8, 0.2]])
    reranked = EmbeddingRerankerStage(top_k=2, retriever=retriever).run(
        RetrievalRequest("museum", None, 2),
        candidates,
    )

    assert [candidate.memory_index for candidate in reranked] == [1, 2]
    assert "embedding_rerank" in reranked[0].scores
    assert reranked[0].ranks["embedding_rerank"] == 1


def test_pipeline_embedding_bm25_final_k():
    memories = make_memories()
    pipeline = RetrievalPipeline(
        final_k=1,
        stages=[
            EmbeddingCandidateGenerator(
                top_k=3,
                retriever=FakeRetriever([0, 1, 2]),
                memories=memories,
                memory_text=memory_text,
            ),
            BM25Reranker(top_k=2),
        ],
    )

    selected = pipeline.run(RetrievalRequest("museum", "museum cafe", 1))

    assert [candidate.memory_index for candidate in selected] == [2]
    assert [candidate.memory_index for candidate in pipeline.last_candidates] == [2, 1]
    assert [stage["name"] for stage in pipeline.last_stage_info] == [
        "embedding_candidates",
        "bm25_rerank",
    ]


def test_pipeline_preserves_scores_across_bm25_and_cross_encoder():
    memories = make_memories()
    reranker = FakeReranker([1, 2, 0])
    pipeline = RetrievalPipeline(
        final_k=2,
        stages=[
            EmbeddingCandidateGenerator(
                top_k=3,
                retriever=FakeRetriever([0, 1, 2]),
                memories=memories,
                memory_text=memory_text,
            ),
            BM25Reranker(top_k=3),
            CrossEncoderRerankerStage(reranker=reranker, top_k=2),
        ],
    )

    selected = pipeline.run(RetrievalRequest("museum", "museum tickets", 2))

    assert [candidate.memory_index for candidate in selected] == [1, 2]
    assert "bm25_rerank" in selected[0].scores
    assert selected[0].scores["cross_encoder_rerank"] == 3.0
    assert selected[0].ranks["embedding_candidates"] == 2
    assert selected[0].ranks["cross_encoder_rerank"] == 1


def test_empty_candidates_do_not_crash():
    pipeline = RetrievalPipeline(
        final_k=2,
        stages=[
            EmbeddingCandidateGenerator(
                top_k=3,
                retriever=FakeRetriever([]),
                memories=[],
                memory_text=memory_text,
            ),
            BM25Reranker(top_k=2),
        ],
    )

    assert pipeline.run(RetrievalRequest("museum", "museum", 2)) == []
    assert pipeline.last_stage_info[-1]["output_count"] == 0
