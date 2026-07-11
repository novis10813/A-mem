from dataclasses import dataclass

from memorybench.methods.amem.serialization import notes_to_store


@dataclass
class Note:
    id: str
    content: str
    timestamp: str
    context: str
    keywords: list[str]
    tags: list[str]
    links: list[str]


def test_notes_to_store_preserves_amem_fields_graph_and_links():
    store = notes_to_store("locomo:0", [
        Note("n1", "Alice arrived", "today", "arrival", ["alice"], ["event"], ["n2"]),
        Note("n2", "Bob waved", "today", "greeting", ["bob"], ["event"], []),
    ])
    assert store.records[0].metadata["context"] == "arrival"
    assert store.records[0].keywords == ("alice",)
    assert store.edges[0].source_id == "n1"
    assert store.edges[0].target_id == "n2"
    assert store.layers[0].name == "amem_notes"
