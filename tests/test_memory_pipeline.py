from types import SimpleNamespace

from amem.memory_layer_robust import BM25MemoryRetriever, RobustAgenticMemorySystem
from amem.memory_pipeline import (
    MemoryProcessingPipeline,
    PipelineHook,
    PipelineTimingHook,
    merge_timing_summaries,
)


class RecordingHook(PipelineHook):
    def __init__(self):
        self.events = []

    def before_stage(self, stage_name, context):
        self.events.append(("before", stage_name, context.note is not None))

    def after_stage(self, stage_name, context):
        self.events.append(("after", stage_name, context.note is not None))


class FakeRetriever:
    def __init__(self):
        self.documents = []

    def add_documents(self, documents):
        self.documents.extend(documents)


class SearchingRetriever:
    def __init__(self, indices):
        self.indices = indices
        self.calls = []

    def search(self, query, k):
        self.calls.append((query, k))
        return self.indices[:k]


class ScoreReranker:
    mode = "test"

    def __init__(self, ordered_indices):
        self.ordered_indices = ordered_indices
        self.calls = []

    def rerank(self, query, candidates, top_k):
        self.calls.append((query, candidates, top_k))
        scores = {
            index: float(len(self.ordered_indices) - rank)
            for rank, index in enumerate(self.ordered_indices)
        }
        return [
            SimpleNamespace(index=index, score=scores[index])
            for index in self.ordered_indices
            if index in {candidate[0] for candidate in candidates}
        ][:top_k]


class FakeSystem:
    def __init__(self):
        self.calls = []

    def construct_memory_note(self, content, time=None, **kwargs):
        self.calls.append(("construct", content, time, kwargs))
        return SimpleNamespace(
            id="note-1",
            content=content,
            context="ctx",
            keywords=["alpha"],
            tags=[],
            links=[],
        )

    def generate_memory_links(self, context):
        self.calls.append(("link", context.note.content))
        context.evolution_label = True

    def evolve_related_memories(self, context):
        self.calls.append(("evolve", context.note.content))


def test_default_pipeline_runs_stages_and_hooks_in_order():
    hook = RecordingHook()
    pipeline = MemoryProcessingPipeline(hooks=[hook])
    system = FakeSystem()

    evolved, note = pipeline.process(system, "hello", time="2026-01-01", extra=True)

    assert evolved is True
    assert note.id == "note-1"
    assert system.calls == [
        ("construct", "hello", "2026-01-01", {"extra": True}),
        ("link", "hello"),
        ("evolve", "hello"),
    ]
    assert hook.events == [
        ("before", "memory_construction", False),
        ("after", "memory_construction", True),
        ("before", "link_generation", True),
        ("after", "link_generation", True),
        ("before", "memory_evolution", True),
        ("after", "memory_evolution", True),
    ]


def test_pipeline_accepts_replacement_stage():
    class ReplacementLinkStage:
        name = "custom_link"

        def run(self, context):
            context.system.calls.append(("custom_link", context.note.content))
            context.evolution_label = False

    pipeline = MemoryProcessingPipeline(link_generation_stage=ReplacementLinkStage())
    system = FakeSystem()

    evolved, _ = pipeline.process(system, "hello")

    assert evolved is False
    assert system.calls == [
        ("construct", "hello", None, {}),
        ("custom_link", "hello"),
        ("evolve", "hello"),
    ]


def test_add_note_uses_pipeline_result_then_stores_and_indexes_note():
    note = SimpleNamespace(
        id="note-42",
        content="content",
        context="stored context",
        keywords=["stored", "keyword"],
        tags=[],
        links=[],
    )

    class FakePipeline:
        def process(self, system, content, time=None, **kwargs):
            assert content == "content"
            assert time == "2026-01-01"
            assert kwargs == {"importance_score": 3.0}
            return True, note

    system = RobustAgenticMemorySystem.__new__(RobustAgenticMemorySystem)
    system.memories = {}
    system.retriever = FakeRetriever()
    system.pipeline = FakePipeline()
    system.evo_cnt = 0
    system.evo_threshold = 100

    note_id = RobustAgenticMemorySystem.add_note(
        system,
        "content",
        time="2026-01-01",
        importance_score=3.0,
    )

    assert note_id == "note-42"
    assert system.memories == {"note-42": note}
    assert system.retriever.documents == ["stored context keywords: stored, keyword"]
    assert system.evo_cnt == 1


def test_add_note_stores_constructed_note_when_link_stage_fails():
    note = SimpleNamespace(
        id="note-link-failed",
        content="content",
        context="constructed context",
        keywords=["constructed"],
        tags=[],
        links=[],
    )

    system = RobustAgenticMemorySystem.__new__(RobustAgenticMemorySystem)
    system.memories = {}
    system.retriever = FakeRetriever()
    system.pipeline = MemoryProcessingPipeline()
    system.evo_cnt = 0
    system.evo_threshold = 100
    system.construct_memory_note = lambda content, time=None, **kwargs: note

    def fail_link(context):
        raise RuntimeError("link failed")

    system.generate_memory_links = fail_link
    system.evolve_related_memories = lambda context: None

    note_id = RobustAgenticMemorySystem.add_note(system, "content")

    assert note_id == "note-link-failed"
    assert system.memories == {"note-link-failed": note}
    assert system.retriever.documents == ["constructed context keywords: constructed"]
    assert system.evo_cnt == 0


def test_find_related_memories_uses_rerank_candidate_pool_then_final_k():
    memories = {
        f"note-{idx}": SimpleNamespace(
            id=f"note-{idx}",
            timestamp=f"2026-01-0{idx}",
            content=f"content {idx}",
            context=f"context {idx}",
            keywords=[f"keyword-{idx}"],
            tags=[f"tag-{idx}"],
            links=[],
        )
        for idx in range(4)
    }
    retriever = SearchingRetriever([0, 1, 2, 3])
    reranker = ScoreReranker([2, 0, 1, 3])
    system = RobustAgenticMemorySystem.__new__(RobustAgenticMemorySystem)
    system.memories = memories
    system.retriever = retriever
    system.reranker = reranker
    system.rerank_top_n = 4
    system.last_retrieval_info = {}

    context = RobustAgenticMemorySystem.find_related_memories_raw(
        system,
        "similarity keywords",
        k=2,
        rerank_query="original question",
    )

    assert retriever.calls == [("similarity keywords", 4)]
    assert reranker.calls[0][0] == "original question"
    assert reranker.calls[0][2] == 2
    assert "memory content: content 2" in context
    assert "memory content: content 0" in context
    assert "memory content: content 1" not in context
    assert system.last_retrieval_info == {
        "similarity_query": "similarity keywords",
        "rerank_query": "original question",
        "candidate_k": 4,
        "candidate_indices": [0, 1, 2, 3],
        "final_indices": [2, 0],
        "rerank_scores": [4.0, 3.0],
        "rerank_mode": "test",
        "retrieval_mode": "embedding",
    }


def test_find_related_memories_keeps_existing_k_when_reranker_disabled():
    memories = {
        f"note-{idx}": SimpleNamespace(
            id=f"note-{idx}",
            timestamp=f"2026-01-0{idx}",
            content=f"content {idx}",
            context=f"context {idx}",
            keywords=[f"keyword-{idx}"],
            tags=[],
            links=[],
        )
        for idx in range(3)
    }
    retriever = SearchingRetriever([2, 1, 0])
    system = RobustAgenticMemorySystem.__new__(RobustAgenticMemorySystem)
    system.memories = memories
    system.retriever = retriever
    system.reranker = None
    system.rerank_top_n = None
    system.last_retrieval_info = {}

    context = RobustAgenticMemorySystem.find_related_memories_raw(system, "query", k=2)

    assert retriever.calls == [("query", 2)]
    assert "memory content: content 2" in context
    assert "memory content: content 1" in context
    assert "memory content: content 0" not in context
    assert system.last_retrieval_info["final_indices"] == [2, 1]
    assert system.last_retrieval_info["rerank_mode"] == "off"
    assert system.last_retrieval_info["retrieval_mode"] == "embedding"


def test_bm25_memory_retriever_ranks_lexical_matches_and_preserves_ties():
    retriever = BM25MemoryRetriever(
        [
            "garden cooking",
            "museum tickets",
            "museum cafe",
        ]
    )

    assert retriever.search("museum", 3) == [1, 2, 0]


def test_find_related_memories_records_bm25_retrieval_mode():
    memories = {
        "note-0": SimpleNamespace(
            id="note-0",
            timestamp="2026-01-01",
            content="Avery bought museum tickets",
            context="museum plans",
            keywords=["tickets"],
            tags=[],
            links=[],
        ),
        "note-1": SimpleNamespace(
            id="note-1",
            timestamp="2026-01-02",
            content="Avery cooked dinner",
            context="home dinner",
            keywords=["cooking"],
            tags=[],
            links=[],
        ),
    }
    system = RobustAgenticMemorySystem.__new__(RobustAgenticMemorySystem)
    system.memories = memories
    system.retriever = BM25MemoryRetriever(["museum tickets", "home cooking"])
    system.reranker = None
    system.rerank_top_n = None
    system.retrieval_mode = "bm25"
    system.last_retrieval_info = {}

    context = RobustAgenticMemorySystem.find_related_memories_raw(system, "museum", k=1)

    assert "memory content: Avery bought museum tickets" in context
    assert "memory content: Avery cooked dinner" not in context
    assert system.last_retrieval_info["candidate_indices"] == [0]
    assert system.last_retrieval_info["final_indices"] == [0]
    assert system.last_retrieval_info["retrieval_mode"] == "bm25"


def test_pipeline_timing_hook_records_stage_summary():
    timing_hook = PipelineTimingHook()
    pipeline = MemoryProcessingPipeline(hooks=[timing_hook])
    system = FakeSystem()

    pipeline.process(system, "hello")

    summary = timing_hook.summary()
    assert set(summary) == {"memory_construction", "link_generation", "memory_evolution"}
    for stats in summary.values():
        assert stats["count"] == 1
        assert stats["total_seconds"] >= 0.0
        assert stats["min_seconds"] >= 0.0
        assert stats["max_seconds"] >= stats["min_seconds"]
        assert stats["avg_seconds"] == stats["total_seconds"] / stats["count"]


def test_merge_timing_summaries_aggregates_stage_counts_and_durations():
    merged = merge_timing_summaries([
        {
            "memory_construction": {
                "count": 2,
                "total_seconds": 5.0,
                "min_seconds": 2.0,
                "max_seconds": 3.0,
                "avg_seconds": 2.5,
            }
        },
        {},
        {
            "memory_construction": {
                "count": 1,
                "total_seconds": 4.0,
                "min_seconds": 4.0,
                "max_seconds": 4.0,
                "avg_seconds": 4.0,
            },
            "link_generation": {
                "count": 1,
                "total_seconds": 1.5,
                "min_seconds": 1.5,
                "max_seconds": 1.5,
                "avg_seconds": 1.5,
            },
        },
    ])

    assert merged == {
        "link_generation": {
            "count": 1,
            "total_seconds": 1.5,
            "min_seconds": 1.5,
            "max_seconds": 1.5,
            "avg_seconds": 1.5,
        },
        "memory_construction": {
            "count": 3,
            "total_seconds": 9.0,
            "min_seconds": 2.0,
            "max_seconds": 4.0,
            "avg_seconds": 3.0,
        },
    }
