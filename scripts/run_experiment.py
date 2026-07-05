#!/usr/bin/env python3
"""Run a two-stage A-MEM experiment from a YAML config."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
for path in (REPO_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiment_common import repo_path, validate_experiment_id  # noqa: E402
from experiment_config import load_experiment_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an A-MEM experiment from YAML config")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-level", default=None)
    return parser.parse_args()


def check_backend(config) -> None:
    if config.backend.name != "ollama":
        return
    subprocess.run(
        ["curl", "-sf", "http://localhost:11434/api/tags"],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        check=True,
    )


def run_logged(command: list[str], log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            handle.write(line)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def main() -> None:
    args = parse_args()
    config_path = repo_path(args.config)
    config = load_experiment_config(config_path)
    experiment_id = validate_experiment_id(config.experiment_id)
    dataset = repo_path(config.dataset)
    if not dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset}")
    check_backend(config)

    log_dir = repo_path(config.paths.log_root) / experiment_id
    common_flags = ["--config", str(config_path)]
    if args.resume:
        common_flags.append("--resume")
    if args.log_level:
        common_flags.extend(["--log-level", args.log_level])

    build_cmd = ["uv", "run", "python", "scripts/build_memories.py", *common_flags]
    eval_cmd = ["uv", "run", "python", "scripts/evaluate_memories.py", *common_flags]
    run_logged(build_cmd, log_dir / "build.log")
    run_logged(eval_cmd, log_dir / f"evaluate_{config.evaluation.qa_mode}.log")


if __name__ == "__main__":
    main()
