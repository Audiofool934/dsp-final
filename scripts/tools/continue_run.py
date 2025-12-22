from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from subprocess import run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continue experiments in an existing run directory."
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Existing run id under outputs/results (defaults to latest).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="outputs/results",
        help="Root results directory.",
    )
    args, extra = parser.parse_known_args()
    args.extra = extra
    return args


def main() -> None:
    args = parse_args()
    cmd = [
        sys.executable,
        "scripts/tools/run_all_experiments.py",
        "--resume",
    ]
    if args.run_id:
        cmd.extend(["--run-id", args.run_id])
    cmd.extend(args.extra)

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", ".")
    print(" ".join(cmd))
    run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
