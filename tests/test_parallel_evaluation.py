from test_advanced_robust import merge_sample_outputs


def test_merge_sample_outputs_sorts_by_sample_and_combines_counts():
    merged = merge_sample_outputs([
        {
            "sample_idx": 2,
            "results": [{"sample_id": 2, "question": "q2"}],
            "metrics": [{"f1": 0.2}],
            "categories": [3],
            "category_counts": {3: 1},
            "error_num": 1,
            "stage_timing": {},
        },
        {
            "sample_idx": 0,
            "results": [{"sample_id": 0, "question": "q0"}],
            "metrics": [{"f1": 0.0}],
            "categories": [1],
            "category_counts": {1: 1},
            "error_num": 0,
            "stage_timing": {
                "memory_construction": {
                    "count": 2,
                    "total_seconds": 6.0,
                    "min_seconds": 2.0,
                    "max_seconds": 4.0,
                    "avg_seconds": 3.0,
                }
            },
        },
    ])

    assert merged["results"] == [
        {"sample_id": 0, "question": "q0"},
        {"sample_id": 2, "question": "q2"},
    ]
    assert merged["metrics"] == [{"f1": 0.0}, {"f1": 0.2}]
    assert merged["categories"] == [1, 3]
    assert merged["category_counts"] == {1: 1, 3: 1}
    assert merged["total_questions"] == 2
    assert merged["error_num"] == 1
    assert merged["stage_timing"] == {
        "memory_construction": {
            "count": 2,
            "total_seconds": 6.0,
            "min_seconds": 2.0,
            "max_seconds": 4.0,
            "avg_seconds": 3.0,
        }
    }
    assert merged["sample_timing"] == [
        {
            "sample_idx": 0,
            "stage_timing": {
                "memory_construction": {
                    "count": 2,
                    "total_seconds": 6.0,
                    "min_seconds": 2.0,
                    "max_seconds": 4.0,
                    "avg_seconds": 3.0,
                }
            },
        },
        {"sample_idx": 2, "stage_timing": {}},
    ]
