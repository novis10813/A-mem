from amem.benchmark.schemas import (
    MemoryEdge,
    MemoryLayer,
    MemoryNode,
    MemoryRecord,
    MemoryStore,
    QAResult,
    RetrievedItem,
    RetrievalToolCall,
    UsageRecord,
    from_jsonable,
    to_jsonable,
)


def test_memory_store_round_trip_preserves_graph_and_records():
    store = MemoryStore(
        sample_id=7,
        records=(
            MemoryRecord(
                memory_id="m1",
                sample_id=7,
                text="Alice visited Taipei.",
                timestamp="2026-01-01T10:00:00",
                content="Alice visited Taipei.",
                keywords=("alice", "taipei"),
                links=("m2",),
            ),
        ),
        nodes=(MemoryNode(node_id="n1", node_type="entity", label="Alice"),),
        edges=(
            MemoryEdge(
                edge_id="e1",
                source_id="n1",
                target_id="n2",
                edge_type="visited",
                valid_at="2026-01-01T10:00:00",
            ),
        ),
        layers=(MemoryLayer(name="semantic_entity", node_ids=("n1", "n2"), edge_ids=("e1",)),),
        private_refs={"pickle": "private/memory_cache_sample_7.pkl"},
    )

    payload = to_jsonable(store)
    restored = from_jsonable(MemoryStore, payload)

    assert restored == store
    assert payload["records"][0]["keywords"] == ["alice", "taipei"]
    assert payload["layers"][0]["name"] == "semantic_entity"


def test_qa_result_round_trip_preserves_usage_and_tool_calls():
    result = QAResult(
        experiment_id="exp",
        construction_run=0,
        qa_run=1,
        sample_id=2,
        qa_idx=3,
        question="Where did Alice go?",
        reference="Taipei",
        prediction="Alice went to Taipei.",
        category=4,
        metrics={"f1": 0.8},
        retrieval={
            "items": [
                to_jsonable(RetrievedItem(item_id="m1", rank=1, text="Alice visited Taipei."))
            ],
            "tool_calls": [
                to_jsonable(
                    RetrievalToolCall(
                        tool_name="query_event_keywords",
                        arguments={"keywords": ["taipei"]},
                        output_text="m1",
                    )
                )
            ],
        },
        context={"text": "Alice visited Taipei."},
        prompt="Answer the question.",
        usage=(UsageRecord(phase="qa", call_id="answer", total_tokens=42),),
    )

    restored = from_jsonable(QAResult, to_jsonable(result))

    assert restored == result
    assert restored.usage[0].source == "reported"
