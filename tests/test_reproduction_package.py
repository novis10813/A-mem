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
