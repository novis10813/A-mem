from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_gitignore_excludes_generated_reproduction_artifacts():
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")

    required_patterns = [
        ".venv/",
        "__pycache__/",
        ".pytest_cache/",
        "artifacts/",
        "cached_memories*/",
        "logs/",
        "results_*.json",
        "results_*/",
        "output/",
        "*.pdf",
    ]
    for pattern in required_patterns:
        assert pattern in gitignore


def test_reproduction_readme_documents_uv_and_no_bundled_results_policy():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "This public fork is a reproduction-oriented extension" in readme
    assert "uv sync" in readme
    assert "uv run python -m pytest tests/test_ablation.py -v" in readme
    assert "Generated caches, logs, result files, and visualization outputs are intentionally not committed" in readme
    assert "WujiangXu/A-mem" in readme


def test_ollama_helper_scripts_live_inside_repo_scripts_directory():
    scripts_dir = ROOT / "scripts"

    k_sweep = scripts_dir / "k_sweep_ollama.sh"
    print_k = scripts_dir / "print_k_results.py"
    assert k_sweep.exists()
    assert print_k.exists()

    k_sweep_text = k_sweep.read_text(encoding="utf-8")
    assert "uv run test_advanced_robust.py" in k_sweep_text
    assert "Run this script from the A-MEM repo root" in k_sweep_text
    assert 'OUTDIR="artifacts/results/k_sweep_ollama"' in k_sweep_text
    assert 'LOGDIR="artifacts/logs/k_sweep_ollama"' in k_sweep_text
    assert "PYEOF" not in k_sweep_text


def test_two_stage_wrapper_defaults_to_artifacts_directory():
    wrapper = (ROOT / "scripts" / "run_experiment.sh").read_text(encoding="utf-8")

    assert 'cache_root="artifacts/caches"' in wrapper
    assert 'results_root="artifacts/results"' in wrapper
    assert 'log_root="artifacts/logs"' in wrapper


def test_two_stage_wrapper_forwards_reranker_flags_to_evaluation_only():
    wrapper = (ROOT / "scripts" / "run_experiment.sh").read_text(encoding="utf-8")

    assert 'rerank_mode="off"' in wrapper
    assert 'retrieval_mode="embedding"' in wrapper
    assert "--retrieval-mode" in wrapper
    assert "--rerank-mode" in wrapper
    assert "--rerank-model" in wrapper
    assert "--rerank-top-n" in wrapper
    assert "--rerank-batch-size" in wrapper
    build_section = wrapper.split("build_cmd=(", 1)[1].split("eval_cmd=(", 1)[0]
    eval_section = wrapper.split("eval_cmd=(", 1)[1].split(")", 1)[0]
    assert "--retrieval-mode" not in build_section
    assert "--retrieval-mode" in eval_section
    assert "--rerank-mode" not in build_section
    assert "--rerank-mode" in eval_section


def test_retrieval_design_docs_are_linked_from_agents_guide():
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    design_doc = ROOT / "docs" / "retrieval_reranker_design_zh.md"
    baseline_doc = ROOT / "docs" / "baseline_comparison_zh.md"

    assert design_doc.exists()
    assert baseline_doc.exists()
    assert "docs/retrieval_reranker_design_zh.md" in agents
    assert "docs/baseline_comparison_zh.md" in agents


def test_baseline_doc_records_core_retrieval_comparison_terms():
    baseline_doc = (ROOT / "docs" / "baseline_comparison_zh.md").read_text(
        encoding="utf-8"
    )

    required_terms = [
        "final_k",
        "top_k",
        "retrieval_pipeline",
        "content_keywords",
        "keyword_pruning_mode",
        "cross_encoder",
        "bm25",
        "cache_experiment_id",
        "ollama_llama3.2-1b_none_rerank_k10",
    ]
    for term in required_terms:
        assert term in baseline_doc
