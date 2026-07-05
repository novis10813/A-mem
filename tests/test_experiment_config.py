from pathlib import Path

import pytest

from scripts.experiment_config import (
    build_args_from_config,
    evaluate_args_from_config,
    load_experiment_config,
)


def write_config(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_load_experiment_config_fills_defaults_and_builds_namespaces(tmp_path: Path):
    path = write_config(
        tmp_path / "exp.yaml",
        """
experiment_id: exp1
dataset: data/locomo10.json
backend:
  name: ollama
  model: llama3.2:1b
construction:
  runs: 2
  keyword_pruning_mode: nltk
evaluation:
  qa_mode: robust
  qa_runs: 3
  keyword_conditions: [none, nltk]
  retrieval_pipeline:
    final_k: 10
    stages:
      - type: embedding
        name: embedding_candidates
        top_k: 50
      - type: cross_encoder
        name: cross_encoder_rerank
        top_k: 10
        batch_size: 16
run:
  resume: true
""",
    )

    config = load_experiment_config(path)
    build_args = build_args_from_config(config)
    eval_args = evaluate_args_from_config(config)

    assert config.experiment_id == "exp1"
    assert build_args.construction_runs == 2
    assert build_args.keyword_pruning_mode == "nltk"
    assert eval_args.qa_mode == "robust"
    assert eval_args.keyword_conditions == ("none", "nltk")
    assert eval_args.retrieve_k == 10
    assert eval_args.retrieval_pipeline.final_k == 10
    assert eval_args.retrieval_pipeline.stages[0].type == "embedding"
    assert eval_args.retrieval_pipeline.stages[1].type == "cross_encoder"
    assert eval_args.retrieval_pipeline.stages[1].batch_size == 16
    assert eval_args.resume is True


def test_config_rejects_invalid_retrieval_pipeline_sizes(tmp_path: Path):
    path = write_config(
        tmp_path / "bad.yaml",
        """
experiment_id: exp1
evaluation:
  retrieval_pipeline:
    final_k: 0
    stages:
      - type: embedding
        top_k: 5
""",
    )

    with pytest.raises(ValueError, match="final_k"):
        load_experiment_config(path)


def test_config_rejects_unsupported_modes(tmp_path: Path):
    path = write_config(
        tmp_path / "bad.yaml",
        """
experiment_id: exp1
construction:
  keyword_pruning_mode: aggressive
""",
    )

    with pytest.raises(ValueError, match="keyword_pruning_mode"):
        load_experiment_config(path)


def test_config_rejects_non_generator_first_stage(tmp_path: Path):
    path = write_config(
        tmp_path / "bad.yaml",
        """
experiment_id: exp1
evaluation:
  retrieval_pipeline:
    final_k: 10
    stages:
      - type: bm25_rerank
        top_k: 10
""",
    )

    with pytest.raises(ValueError, match="first retrieval pipeline stage"):
        load_experiment_config(path)


def test_config_rejects_unknown_stage_type_and_query(tmp_path: Path):
    bad_type = write_config(
        tmp_path / "bad_type.yaml",
        """
experiment_id: exp1
evaluation:
  retrieval_pipeline:
    final_k: 10
    stages:
      - type: vector_magic
        top_k: 10
""",
    )

    with pytest.raises(ValueError, match="first retrieval pipeline stage|unknown"):
        load_experiment_config(bad_type)

    bad_query = write_config(
        tmp_path / "bad_query.yaml",
        """
experiment_id: exp1
evaluation:
  retrieval_pipeline:
    final_k: 10
    stages:
      - type: embedding
        top_k: 10
        query: generated_answer
""",
    )

    with pytest.raises(ValueError, match="query selector"):
        load_experiment_config(bad_query)


def test_config_rejects_legacy_retrieval_fields(tmp_path: Path):
    path = write_config(
        tmp_path / "bad.yaml",
        """
experiment_id: exp1
evaluation:
  retrieval_mode: bm25
  rerank:
    mode: cross_encoder
""",
    )

    with pytest.raises(ValueError, match="retrieval_mode"):
        load_experiment_config(path)
