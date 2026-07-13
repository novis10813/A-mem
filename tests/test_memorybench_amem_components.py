from memorybench.config import ConstructionConfig, ContextConfig, RetrievalConfig
from memorybench.methods.amem.construction import AMemConstruction
from memorybench.methods.amem.context import build_amem_context
from memorybench.methods.amem.qa import build_amem_qa_prompt
from memorybench.methods.amem.retrieval import retrieve_amem
from memorybench.schemas import DatasetSample, MemoryRecord, MemoryStore, Question, Turn


def sample() -> DatasetSample:
    return DatasetSample(
        sample_id="locomo:0",
        turns=(
            Turn(turn_id="m0", evidence_id="D1:1", speaker="Alice", text="Alice moved to Taipei", session_id="1"),
            Turn(turn_id="m1", evidence_id="D1:2", speaker="Bob", text="Bob visited Taipei", session_id="1"),
        ),
        questions=(),
    )


def test_amem_construction_runs_analysis_and_link_evolution_with_scripted_llm():
    config = ConstructionConfig.model_validate({
        "adapter": "amem",
        "llm": {
            "provider": "fake",
            "model": "scripted",
            "params": {"responses": [
                "KEYWORDS: Alice, Taipei\nCONTEXT: Alice moved to Taipei\nTAGS: relocation",
                "KEYWORDS: Bob, Taipei\nCONTEXT: Bob visited Taipei\nTAGS: travel",
                "DECISION: STRENGTHEN\nREASON: same city",
                "CONNECTIONS: 0\nTAGS: travel, taipei",
            ]},
        },
        "params": {"retrieval_mode": "bm25"},
    })

    store = AMemConstruction(config).build_sample(sample())

    assert [record.record_id for record in store.records] == ["m0", "m1"]
    assert store.records[1].metadata["links"] == ["m0"]
    assert [(edge.source_id, edge.target_id) for edge in store.edges] == [("m1", "m0")]
    assert len(store.metadata["usage"]) == 4
    assert all(record["phase"] == "construction" for record in store.metadata["usage"])


def test_amem_retrieval_expands_selected_note_neighbors_and_context():
    store = MemoryStore(sample_id="s", records=(
        MemoryRecord(record_id="m0", text="Alice moved to Taipei", content="Alice moved", metadata={"context": "relocation", "links": []}),
        MemoryRecord(record_id="m1", text="Bob visited museum", content="Bob visited", metadata={"context": "museum trip", "links": ["m0"]}),
    ))
    question = Question(question_id="q", text="museum", reference="")
    retrieval = retrieve_amem(
        question,
        store,
        RetrievalConfig.model_validate({
            "adapter": "staged",
            "stages": [{"adapter": "bm25", "top_k": 1}],
        }),
    )

    assert [item["record_id"] for item in retrieval["items"]] == ["m1", "m0"]
    assert retrieval["neighbor_expansion"] == [{"source_id": "m1", "neighbor_id": "m0"}]
    context = build_amem_context(retrieval, ContextConfig(adapter="amem"))
    assert "memory context: museum trip" in context["text"]
    assert "memory content: Alice moved" in context["text"]


def test_amem_qa_prompt_preserves_temporal_category_behavior():
    question = Question(
        question_id="q",
        text="When did Alice move?",
        reference="May",
        labels={"question_type": ("temporal",)},
    )

    prompt, temperature = build_amem_qa_prompt(question, {"text": "dated context"}, {})

    assert "Use DATE OF CONVERSATION" in prompt
    assert temperature == 0.7


def test_amem_adversarial_prompt_contains_deterministic_answer_choices():
    question = Question(
        question_id="q5",
        text="What was never discussed?",
        reference="Not mentioned in the conversation",
        labels={"question_type": ("adversarial",)},
    )

    first, temperature = build_amem_qa_prompt(question, {"text": "context"}, {"seed": 17})
    second, _ = build_amem_qa_prompt(question, {"text": "context"}, {"seed": 17})

    assert first == second
    assert "Select the correct answer" in first
    assert "Not mentioned in the conversation" in first
    assert temperature == 0.5
