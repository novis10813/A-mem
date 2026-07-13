from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from .config import load_config
from .runner import ExperimentRunner
from .registry import public_component_names


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="memorybench")
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("validate", "run"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True, type=Path)
    dashboard = commands.add_parser("dashboard")
    dashboard.add_argument("--artifact-root", type=Path, default=Path("artifacts/experiments"))
    dashboard.add_argument("--server-name", default="127.0.0.1")
    dashboard.add_argument("--server-port", type=int, default=7860)
    prepare_financebench = commands.add_parser("prepare-financebench")
    prepare_financebench.add_argument("--output", type=Path, default=Path("artifacts/datasets/financebench"))
    prepare_financebench.add_argument("--workers", type=positive_int, default=4)
    commands.add_parser("list-components")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "list-components":
            print(json.dumps(public_component_names(), indent=2, sort_keys=True))
            return 0
        if args.command == "validate":
            config = load_config(args.config)
            print(json.dumps({"valid": True, "experiment_id": config.experiment.id, "fingerprint": config.fingerprint}))
            return 0
        if args.command == "run":
            return ExperimentRunner(load_config(args.config)).run().exit_code
        if args.command == "prepare-financebench":
            from .datasets.financebench_prepare import prepare_financebench

            result = prepare_financebench(args.output, args.workers)
            print(json.dumps({
                "output": str(result.output),
                "manifest": str(result.manifest_path),
                "revision": result.revision,
                "documents": result.document_count,
            }, sort_keys=True))
            return 0
        from .dashboard.data import create_dashboard
        create_dashboard(args.artifact_root).launch(server_name=args.server_name, server_port=args.server_port)
        return 0
    except (ValidationError, ValueError, OSError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
