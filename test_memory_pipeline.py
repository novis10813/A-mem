from types import SimpleNamespace

from memory_layer_robust import RobustAgenticMemorySystem
from memory_pipeline import MemoryProcessingPipeline, PipelineHook


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
