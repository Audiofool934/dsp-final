from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from subprocess import PIPE, STDOUT, Popen

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ResNet experiments for an existing run")
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="outputs/results")
    parser.add_argument("--config", type=str, default="configs/experiments.yaml")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--skip-if-exists", action="store_true")
    parser.add_argument("--grid-epochs", type=int, default=None)
    parser.add_argument("--grid-batch-size", type=int, default=None)
    parser.add_argument("--grid-num-workers", type=int, default=None)
    return parser.parse_args()


def load_config(path: str) -> dict:
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {}
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def run_cmd(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", ".")
    env.setdefault("PYTHONUNBUFFERED", "1")
    with log_path.open("w", encoding="utf-8") as f:
        process = Popen(cmd, stdout=PIPE, stderr=STDOUT, text=True, env=env)
        for line in process.stdout:
            sys.stdout.write(line)
            f.write(line)
        return process.wait()


def should_skip(outputs: list[Path], skip_if_exists: bool) -> bool:
    if not skip_if_exists:
        return False
    return all(path.exists() for path in outputs)


def main() -> None:
    args = parse_args()
    base_dir = Path(args.results_dir) / args.run_id
    if not base_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {base_dir}")

    cfg = load_config(args.config)
    retrieval_cfg = cfg.get("retrieval", {})
    frame_lengths = retrieval_cfg.get("frame_lengths", [512, 1024, 2048])
    hop_lengths = retrieval_cfg.get("hop_lengths", [256, 512, 1024])

    classification_cfg = cfg.get("classification", {})
    clf_frame_length = classification_cfg.get("frame_length", 1024)
    clf_hop_length = classification_cfg.get("hop_length", 512)
    clf_n_mels = classification_cfg.get("n_mels", 40)
    clf_epochs = classification_cfg.get("epochs", 30)
    clf_batch_size = classification_cfg.get("batch_size", 32)
    clf_lr = classification_cfg.get("lr", 1e-3)

    grid_cfg = cfg.get("classification_grid", {})
    grid_epochs = (
        args.grid_epochs if args.grid_epochs is not None else grid_cfg.get("epochs", 20)
    )
    grid_batch_size = (
        args.grid_batch_size if args.grid_batch_size is not None else grid_cfg.get("batch_size", 32)
    )
    grid_num_workers = (
        args.grid_num_workers if args.grid_num_workers is not None else grid_cfg.get("num_workers", 2)
    )

    logs_dir = base_dir / "logs"
    features_dir = base_dir / "features"
    history_dir = base_dir / "history"
    models_dir = base_dir / "models"
    predictions_dir = base_dir / "predictions"
    retrieval_dir = base_dir / "retrieval"
    plots_dir = base_dir / "plots"

    steps = [
        {
            "name": "train_cnn_resnet",
            "cmd": [
                "python",
                "scripts/models/train_cnn.py",
                "--frame-length",
                str(clf_frame_length),
                "--hop-length",
                str(clf_hop_length),
                "--n-mels",
                str(clf_n_mels),
                "--epochs",
                str(clf_epochs),
                "--batch-size",
                str(clf_batch_size),
                "--lr",
                str(clf_lr),
                "--output",
                str(models_dir / "cnn.pt"),
                "--history",
                str(history_dir / "train_cnn.csv"),
                "--feature-cache",
                str(features_dir),
            ],
            "outputs": [models_dir / "cnn.pt", history_dir / "train_cnn.csv"],
        },
        {
            "name": "classification_grid_resnet",
            "cmd": [
                "python",
                "scripts/tasks/run_classification_grid.py",
                "--frame-lengths",
                *[str(v) for v in frame_lengths],
                "--hop-lengths",
                *[str(v) for v in hop_lengths],
                "--epochs",
                str(grid_epochs),
                "--batch-size",
                str(grid_batch_size),
                "--num-workers",
                str(grid_num_workers),
                "--output",
                str(history_dir / "classification_grid.csv"),
                "--history-dir",
                str(history_dir / "classification_grid"),
                "--feature-cache",
                str(features_dir),
            ],
            "outputs": [history_dir / "classification_grid.csv"],
        },
        {
            "name": "cnn_infer",
            "cmd": [
                "python",
                "scripts/models/infer_cnn.py",
                "--checkpoint",
                str(models_dir / "cnn.pt"),
                "--output",
                str(predictions_dir / "cnn_fold5.csv"),
                "--feature-cache",
                str(features_dir),
            ],
            "outputs": [predictions_dir / "cnn_fold5.csv"],
        },
        {
            "name": "retrieval_ml_cnn",
            "cmd": [
                "python",
                "scripts/tasks/run_retrieval_ml.py",
                "--model-type",
                "cnn",
                "--model",
                str(models_dir / "cnn.pt"),
                "--feature-cache",
                str(features_dir),
            ],
            "outputs": [logs_dir / "retrieval_ml_cnn.log"],
        },
        {
            "name": "plot_history",
            "cmd": [
                "python",
                "scripts/tools/plot_history.py",
                "--history",
                str(history_dir / "train_cnn.csv"),
                "--output",
                str(plots_dir / "cnn_history.png"),
            ],
            "outputs": [plots_dir / "cnn_history.png"],
        },
    ]

    summary_path = base_dir / "summary.json"
    summary = {"run_id": args.run_id, "results_dir": str(base_dir), "steps": []}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
    completed = {step.get("name"): step for step in summary.get("steps", []) if step.get("return_code") == 0}

    updated_steps = []
    for step in steps:
        name = step["name"]
        log_path = logs_dir / f"{name}.log"

        if name in completed and args.skip_if_exists:
            updated_steps.append(completed[name])
            continue

        if should_skip(step.get("outputs", []), args.skip_if_exists):
            updated_steps.append(
                {
                    "name": name,
                    "command": step["cmd"],
                    "log": str(log_path),
                    "return_code": 0,
                    "skipped": True,
                }
            )
            continue

        print(f"=== Running: {name} ===")
        code = run_cmd(step["cmd"], log_path)

        entry = {
            "name": name,
            "command": step["cmd"],
            "log": str(log_path),
            "return_code": code,
        }
        updated_steps.append(entry)
        if code != 0 and not args.continue_on_error:
            print(f"Step failed: {name} (exit {code})")
            break

    summary["run_id"] = args.run_id
    summary["results_dir"] = str(base_dir)
    summary["steps"] = updated_steps
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Run summary updated: {summary_path}")


if __name__ == "__main__":
    main()
