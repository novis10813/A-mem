from types import SimpleNamespace

from amem.methods.amem.serialization import memories_to_store, memory_note_to_record


def test_memory_note_to_record_preserves_core_fields():
    note = SimpleNamespace(
        id="note-1",
        content="Speaker Alice says: I moved to Taipei.",
        context="Alice talked about moving.",
        keywords=["Alice", "Taipei"],
        tags=["move"],
        links=["note-2"],
        timestamp="2026-01-01T10:00:00",
    )

    record = memory_note_to_record(note, sample_id=3)

    assert record.memory_id == "note-1"
    assert record.sample_id == 3
    assert record.content == note.content
    assert record.summary == note.context
    assert record.keywords == ("Alice", "Taipei")
    assert record.links == ("note-2",)
    assert "memory content:" in record.text


def test_memories_to_store_adds_note_nodes_and_link_edges():
    memories = {
        "note-1": SimpleNamespace(
            id="note-1",
            content="A",
            context="ctx",
            keywords=[],
            tags=[],
            links=["note-2"],
            timestamp=None,
        ),
        "note-2": SimpleNamespace(
            id="note-2",
            content="B",
            context="ctx",
            keywords=[],
            tags=[],
            links=[],
            timestamp=None,
        ),
    }

    store = memories_to_store(memories, sample_id=4, private_refs={"pickle": "private/cache.pkl"})

    assert [record.memory_id for record in store.records] == ["note-1", "note-2"]
    assert {node.node_id for node in store.nodes} == {"note-1", "note-2"}
    assert store.edges[0].source_id == "note-1"
    assert store.edges[0].target_id == "note-2"
    assert store.private_refs["pickle"] == "private/cache.pkl"
