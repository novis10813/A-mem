from amem.reranking import CrossEncoderReranker


class FakeCrossEncoder:
    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def predict(self, pairs, batch_size, show_progress_bar):
        self.calls.append(
            {
                "pairs": pairs,
                "batch_size": batch_size,
                "show_progress_bar": show_progress_bar,
            }
        )
        return self.scores


def make_reranker(scores):
    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
    reranker.model_name = "fake-model"
    reranker.batch_size = 7
    reranker.model = FakeCrossEncoder(scores)
    return reranker


def test_cross_encoder_reranker_orders_by_score_descending():
    reranker = make_reranker([0.2, 0.9, 0.5])

    results = reranker.rerank(
        "where did Pat go?",
        [(4, "doc 4"), (2, "doc 2"), (9, "doc 9")],
        top_k=2,
    )

    assert [candidate.index for candidate in results] == [2, 9]
    assert [candidate.score for candidate in results] == [0.9, 0.5]
    assert reranker.model.calls == [
        {
            "pairs": [
                ("where did Pat go?", "doc 4"),
                ("where did Pat go?", "doc 2"),
                ("where did Pat go?", "doc 9"),
            ],
            "batch_size": 7,
            "show_progress_bar": False,
        }
    ]


def test_cross_encoder_reranker_preserves_candidate_order_for_ties():
    reranker = make_reranker([0.5, 0.8, 0.8])

    results = reranker.rerank(
        "query",
        [(10, "first"), (11, "second"), (12, "third")],
        top_k=3,
    )

    assert [candidate.index for candidate in results] == [11, 12, 10]


def test_cross_encoder_reranker_returns_empty_for_empty_candidates():
    reranker = make_reranker([])

    assert reranker.rerank("query", [], top_k=3) == []
