from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from .config import load_config
from .runner import ExperimentRunner
from .registry import public_component_names


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
        from .dashboard.data import create_dashboard
        create_dashboard(args.artifact_root).launch(server_name=args.server_name, server_port=args.server_port)
        return 0
    except (ValidationError, ValueError, OSError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
