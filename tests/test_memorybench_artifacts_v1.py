from memorybench.artifacts import read_memory_store, write_memory_store
from memorybench.schemas import MemoryEdge, MemoryNode, MemoryRecord, MemoryStore


def test_sharded_memory_store_round_trip(tmp_path):
    store = MemoryStore(
        sample_id="dataset:7",
        records=(MemoryRecord(record_id="r1", text="hello"),),
        nodes=(MemoryNode(node_id="n1", type="entity", text="Alice"),),
        edges=(MemoryEdge(edge_id="e1", source_id="n1", target_id="n2", type="knows"),),
        private_refs={"index": "private/index.bin"},
        metadata={"method": "graph"},
    )

    write_memory_store(tmp_path, store)

    header = (tmp_path / "store.json").read_text()
    assert '"records": "records.jsonl"' in header
    assert '"private_refs"' in header
    assert (tmp_path / "private").is_dir()
    assert read_memory_store(tmp_path) == store
